# data_cleaner.py
import os
import pandas as pd
import numpy as np
from datetime import datetime


class DataCleaningError(Exception):
    """Raised when data validation fails during cleaning."""
    pass


class DataCleaner:
    """Ethical data cleaning for South African fintech customer data."""

    # Region variants -> standardized canonical name
    REGION_MAP = {
        'JHB': 'Johannesburg',
        'Joburg': 'Johannesburg',
        'Johburg': 'Johannesburg',
        'Joburgh': 'Johannesburg',
        'Gauteng': 'Johannesburg',
        'CPT': 'Cape Town',
        'Capetown': 'Cape Town',
        'Cape Town': 'Cape Town',
        'Western Cape': 'Cape Town',
        'DBN': 'Durban',
        'Durban': 'Durban',
        'eThekwini': 'Durban',
        'Pta': 'Pretoria',
        'Pretoria': 'Pretoria',
        'Tshwane': 'Pretoria',
        'PE': 'Port Elizabeth',
        'Gqeberha': 'Port Elizabeth',
        'Nelson Mandela Bay': 'Port Elizabeth',
        'Port Elizabeth': 'Port Elizabeth',
    }

    # Loan purpose variants -> canonical purpose
    PURPOSE_MAP = {
        'debt consolidation': 'debt_consolidation',
        'consolidation': 'debt_consolidation',
        'debt_consolidate': 'debt_consolidation',
        'home_improve': 'home_improvement',
        'home improvement': 'home_improvement',
        'renovation': 'home_improvement',
        'car_loan': 'car',
        'vehicle': 'car',
        'auto': 'car',
    }

    def __init__(self, df):
        self.df = df.copy()
        self.ethical_notes = []   # Document bias observations here
        self.cleaning_log = []
        self._rows_in = len(self.df)
        self._region_fixes = 0

    # ------------------------------------------------------------------ #
    def _parse_income(self, value):
        """Convert malformed income strings ('R15,000') to float."""
        if pd.isna(value):
            return np.nan
        s = str(value).strip().replace('R', '').replace(',', '').replace(' ', '')
        if s == '' or s.lower() == 'nan':
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    def validate_income(self):
        """Cap income outliers at 99th percentile; handle negatives."""
        # Record township missingness BEFORE imputing/flagging (raw signal)
        raw_missing = self.df['income'].isna() | \
            (self.df['income'].astype(str).str.strip().isin(['', 'nan', 'NaN']))

        # Parse strings -> numeric
        self.df['income'] = self.df['income'].apply(self._parse_income)

        # Negatives are invalid (likely sign errors) -> treat as missing
        neg_count = (self.df['income'] < 0).sum()
        self.df.loc[self.df['income'] < 0, 'income'] = np.nan

        # Cap outliers at 99th percentile (preserve missingness)
        valid = self.df['income'].dropna()
        if len(valid):
            cap = valid.quantile(0.99)
            capped = (self.df['income'] > cap).sum()
            self.df.loc[self.df['income'] > cap, 'income'] = cap
            self.cleaning_log.append(
                f"Capped {int(capped)} incomes at 99th pct (R{cap:,.0f}); "
                f"nulled {int(neg_count)} negatives")

        # --- Bias note: township vs non-township income missingness ---
        if 'township_flag' in self.df.columns:
            t_miss = raw_missing[self.df['township_flag'] == 1].mean()
            n_miss = raw_missing[self.df['township_flag'] == 0].mean()
            if pd.notna(t_miss) and pd.notna(n_miss) and t_miss > n_miss:
                gap = (t_miss - n_miss) / n_miss * 100 if n_miss else float('inf')
                self.ethical_notes.append(
                    f"WARNING: Township applicants show {gap:.0f}% higher income "
                    f"missingness ({t_miss:.0%} vs {n_miss:.0%}) - may indicate form "
                    f"accessibility issues. Missingness preserved, not imputed away.")
        return self.df

    # ------------------------------------------------------------------ #
    def standardize_regions(self):
        """Map region variants to standardized names (JHB -> Johannesburg)."""
        original = self.df['region'].copy()

        def _map(val):
            if pd.isna(val):
                return val
            key = str(val).strip()
            if key in self.REGION_MAP:
                return self.REGION_MAP[key]
            # Fuzzy fallback: case-insensitive match against known variants
            low = key.lower().replace(' ', '')
            for variant, canon in self.REGION_MAP.items():
                if variant.lower().replace(' ', '') == low:
                    return canon
            return key  # leave unrecognised values intact for auditing

        self.df['region'] = self.df['region'].apply(_map)
        self._region_fixes = int((original.astype(str) != self.df['region'].astype(str)).sum())

        # Standardize loan_purpose too
        if 'loan_purpose' in self.df.columns:
            self.df['loan_purpose'] = self.df['loan_purpose'].apply(
                lambda v: self.PURPOSE_MAP.get(str(v).strip(), v) if pd.notna(v) else v)

        self.cleaning_log.append(f"Standardized {self._region_fixes} region variants")
        return self.df

    # ------------------------------------------------------------------ #
    def create_missing_indicators(self):
        """Create binary flags for missing values in critical columns."""
        for col in ['income', 'township_flag']:
            if col in self.df.columns:
                self.df[f'{col}_missing'] = self.df[col].isna().astype(int)
        self.cleaning_log.append("Created missing indicators: income_missing, township_flag_missing")
        return self.df

    # ------------------------------------------------------------------ #
    def to_csv(self, output_path):
        """Export cleaned data with UTF-8 encoding and datetime preservation."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(
            output_path,
            index=False,
            encoding='utf-8-sig',
            date_format='%Y-%m-%d %H:%M:%S')
        self.cleaning_log.append(f"Wrote cleaned data to {output_path}")

    # ------------------------------------------------------------------ #
    def __str__(self):
        return (f"Cleaned {self._rows_in:,} rows | "
                f"Fixed {self._region_fixes:,} region variants | "
                f"{len(self.ethical_notes)} ethical note(s)")

    # ------------------------------------------------------------------ #
    def run_full_cleaning(self):
        """Execute all cleaning steps in ethical sequence."""
        try:
            self.validate_income()
            self.standardize_regions()
            self.create_missing_indicators()
            self.cleaning_log.append("Full cleaning completed")
            return self.df
        except Exception as e:
            raise DataCleaningError(f"Cleaning failed: {str(e)}") from e


def main():
    raw_path = os.path.join('data', 'raw', 'customer_loans_q1_2024.csv')
    out_path = os.path.join('data', 'processed', 'cleaned_customers.csv')

    if not os.path.exists(raw_path):
        raise DataCleaningError(f"Raw dataset not found at {raw_path}")

    df = pd.read_csv(raw_path)
    cleaner = DataCleaner(df)
    cleaner.run_full_cleaning()
    cleaner.to_csv(out_path)
    print(cleaner)
    for note in cleaner.ethical_notes:
        print("  -", note)


if __name__ == '__main__':
    main()
