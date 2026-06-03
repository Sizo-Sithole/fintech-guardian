# feature_engineer.py
import os
import json
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Create business-intelligent features with ethical safeguards."""

    # Load shedding follows SA's published "stage" schedule; max observed ~12h/day.
    # We treat it as cyclical over a 24h-equivalent period so extreme/low values
    # bend back toward each other (infrastructure stress is periodic, not linear).
    LOAD_SHEDDING_PERIOD = 24.0

    CORR_THRESHOLD = 0.85          # drop one of any pair above this
    MIN_REGION_SAMPLES = 50        # flag thin regional benchmarks

    # Feature pairs that are designed to coexist (cyclical encoding halves and
    # the two-part hurdle/intensity split). Exempt from multicollinearity prune
    # so the methodology stays intact even when the data makes them collinear.
    PROTECTED_PAIRS = [
        {'load_shedding_sin', 'load_shedding_cos'},
        {'has_support_contact', 'support_tickets_log'},
    ]

    def __init__(self, df):
        self.df = df.copy()
        self.feature_metadata = {}      # {"feature": "business rationale"}
        self.new_features = []
        self.dropped_features = []
        self._max_corr = 0.0

    # ------------------------------------------------------------------ #
    def create_financial_strain_ratio(self):
        """Calculate debt-to-income ratio with safe zero handling."""
        income = self.df['income'].to_numpy(dtype=float)
        debt = self.df['debt'].to_numpy(dtype=float)

        # np.where guards division by zero / missing income -> NaN, then flag
        ratio = np.where(
            (income > 0) & ~np.isnan(income) & ~np.isnan(debt),
            debt / np.where(income == 0, np.nan, income),
            np.nan)
        self.df['debt_to_income'] = ratio

        # Zero-income / missing cases get an explicit indicator rather than a
        # fabricated ratio, so the model can learn the missingness pattern.
        self.df['strain_undefined'] = self.df['debt_to_income'].isna().astype(int)

        self._register('debt_to_income',
                       "Debt-to-income ratio. Identifies customers at risk of "
                       "default due to overcommitment; core financial strain signal.")
        self._register('strain_undefined',
                       "Flags rows where income is zero/missing so strain ratio "
                       "cannot be computed - preserves data-limitation context.")
        return self.df

    # ------------------------------------------------------------------ #
    def encode_load_shedding_impact(self):
        """Convert load shedding hours to cyclical features."""
        hrs = self.df['load_shedding_hours'].to_numpy(dtype=float)
        angle = 2 * np.pi * hrs / self.LOAD_SHEDDING_PERIOD
        self.df['load_shedding_sin'] = np.sin(angle)
        self.df['load_shedding_cos'] = np.cos(angle)

        self._register('load_shedding_sin',
                       "Cyclical (sine) encoding of load-shedding hours. Preserves "
                       "circular relationship so adjacent extremes stay close; "
                       "proxies transaction-failure risk from power instability.")
        self._register('load_shedding_cos',
                       "Cyclical (cosine) encoding of load-shedding hours; paired "
                       "with sine to fully represent periodic infrastructure stress.")
        return self.df

    # ------------------------------------------------------------------ #
    def regional_benchmarks(self, train_stats=None):
        """Add region-level aggregates WITHOUT leakage.

        Training mode (train_stats is None): compute region means ONLY from the
        training split (rows where churned is known), never using the target.
        Inference mode: merge precomputed train_stats so test rows never see
        statistics derived from themselves.
        """
        if train_stats is None:
            # Training split = rows with a known label; benchmark inputs exclude target.
            train_mask = self.df['churned'].notna()
            train_df = self.df[train_mask]
            stats = (train_df.groupby('region')
                     .agg(region_income_mean=('income', 'mean'),
                          region_dti_mean=('debt_to_income', 'mean'),
                          region_count=('region', 'size'))
                     .reset_index())
            self._train_stats = stats
        else:
            stats = train_stats
            self._train_stats = stats

        merged = self.df.merge(stats, on='region', how='left')

        # Relative position vs region peers (mean-centred, leakage-safe)
        merged['income_vs_region'] = merged['income'] - merged['region_income_mean']
        merged['dti_vs_region'] = merged['debt_to_income'] - merged['region_dti_mean']
        self.df = merged

        # Flag thin benchmarks
        thin = stats[stats['region_count'] < self.MIN_REGION_SAMPLES]['region'].tolist()
        if thin:
            self.feature_metadata['_region_warning'] = (
                f"Regions with <{self.MIN_REGION_SAMPLES} training samples "
                f"(unstable benchmark): {thin}")

        for f, r in {
            'region_income_mean': "Mean income of the region (training-only) - economic baseline.",
            'region_dti_mean': "Mean debt-to-income of region (training-only) peer benchmark.",
            'income_vs_region': "Applicant income relative to regional peers; contextualises affordability.",
            'dti_vs_region': "Applicant strain relative to regional peers; leakage-safe relative risk.",
        }.items():
            self._register(f, r)
        # region_count is bookkeeping, not a model feature
        self.feature_metadata['region_count'] = "Training sample size per region (diagnostic, not a predictor)."
        return self.df

    # ------------------------------------------------------------------ #
    def handle_zero_inflated_tickets(self):
        """Two-part transformation for zero-inflated support_tickets."""
        tickets = self.df['support_tickets'].fillna(0).to_numpy(dtype=float)
        # Part 1: did the customer contact support at all? (the "hurdle")
        self.df['has_support_contact'] = (tickets > 0).astype(int)
        # Part 2: intensity given contact, log1p-compressed to tame the long tail
        self.df['support_tickets_log'] = np.log1p(tickets)

        self._register('has_support_contact',
                       "Binary hurdle for zero-inflated support_tickets (~70% zeros); "
                       "separates 'ever contacted support' from contact volume.")
        self._register('support_tickets_log',
                       "log1p of support_tickets - compresses heavy tail of the "
                       "count component in the two-part model.")
        return self.df

    # ------------------------------------------------------------------ #
    def check_multicollinearity(self):
        """Reject engineered features with |corr| > threshold."""
        candidates = [f for f in self.new_features if f in self.df.columns]
        numeric = self.df[candidates].select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return self.df

        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self._max_corr = float(np.nanmax(upper.to_numpy())) if upper.notna().any().any() else 0.0

        to_drop = set()
        for col in upper.columns:
            for row in upper.index:
                val = upper.loc[row, col]
                if pd.notna(val) and val > self.CORR_THRESHOLD:
                    if {row, col} in self.PROTECTED_PAIRS:
                        continue  # intentional design pair; keep both
                    # Drop the second-created feature of the pair
                    drop = col if self.new_features.index(col) > self.new_features.index(row) else row
                    to_drop.add(drop)

        for col in to_drop:
            if col in self.df.columns:
                self.df.drop(columns=col, inplace=True)
                self.dropped_features.append(col)
                self.feature_metadata[col] = (self.feature_metadata.get(col, '') +
                                              f" [DROPPED: |corr|>{self.CORR_THRESHOLD}]")
        self.new_features = [f for f in self.new_features if f not in to_drop]
        return self.df

    # ------------------------------------------------------------------ #
    def _register(self, name, rationale):
        self.feature_metadata[name] = rationale
        if name not in self.new_features:
            self.new_features.append(name)

    def __str__(self):
        n = len([f for f in self.new_features if f in self.df.columns])
        return (f"Created {n} new features | Max pairwise |corr|: {self._max_corr:.2f} | "
                f"Dropped {len(self.dropped_features)} for multicollinearity")

    # ------------------------------------------------------------------ #
    def run_full_engineering(self):
        """Execute all feature engineering steps."""
        try:
            self.create_financial_strain_ratio()
            self.encode_load_shedding_impact()
            self.regional_benchmarks()
            self.handle_zero_inflated_tickets()
            self.check_multicollinearity()
            return self.df
        except Exception as e:
            raise ValueError(f"Feature engineering failed: {str(e)}") from e


def main():
    in_path = os.path.join('data', 'processed', 'cleaned_customers.csv')
    out_path = os.path.join('data', 'processed', 'engineered_features.csv')
    meta_path = os.path.join('data', 'processed', 'feature_metadata.json')

    if not os.path.exists(in_path):
        raise ValueError(f"Cleaned dataset (Milestone 1 artifact) not found at {in_path}")

    df = pd.read_csv(in_path)
    fe = FeatureEngineer(df)
    fe.run_full_engineering()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fe.df.to_csv(out_path, index=False, encoding='utf-8-sig')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(fe.feature_metadata, f, indent=2, ensure_ascii=False)

    print(fe)
    if '_region_warning' in fe.feature_metadata:
        print("  -", fe.feature_metadata['_region_warning'])


if __name__ == '__main__':
    main()
