"""
data_generator.py — Generates realistic synthetic telecom churn data.
Keeps the rest of the pipeline independent of any specific data source.
"""

import numpy as np
import pandas as pd
from config import cfg


class ChurnDataGenerator:
    """
    Produces a labelled DataFrame with realistic feature correlations.

    Usage
    -----
    >>> gen = ChurnDataGenerator()
    >>> df  = gen.generate()
    """

    def __init__(self, config=None):
        self.cfg = config or cfg
        self.rng = np.random.default_rng(self.cfg.data.random_state)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        n = self.cfg.data.n_samples
        df = self._base_features(n)
        df = self._add_derived(df)
        df = self._assign_churn(df)
        return df.reset_index(drop=True)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _base_features(self, n: int) -> pd.DataFrame:
        rng = self.rng
        return pd.DataFrame({
            # Demographics
            "gender": rng.choice(["Male", "Female"], n),
            "age": rng.integers(18, 80, n),
            # Contract / service
            "tenure_months": rng.integers(1, 73, n),
            "contract_type": rng.choice(
                ["Month-to-month", "One year", "Two year"],
                n, p=[0.55, 0.25, 0.20],
            ),
            "internet_service": rng.choice(
                ["Fiber optic", "DSL", "No"],
                n, p=[0.44, 0.34, 0.22],
            ),
            "payment_method": rng.choice(
                ["Electronic check", "Mailed check",
                 "Bank transfer", "Credit card"],
                n, p=[0.34, 0.23, 0.22, 0.21],
            ),
            # Billing
            "monthly_charges": rng.uniform(18, 119, n).round(2),
            "num_products": rng.integers(1, 8, n),
            # Usage
            "support_calls": rng.integers(0, 11, n),
            "avg_session_minutes": rng.exponential(35, n).clip(1, 300).round(1),
            "days_since_last_login": rng.integers(0, 91, n),
        })

    def _add_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        df["total_charges"] = (
            df["monthly_charges"] * df["tenure_months"]
            + self.rng.normal(0, 20, len(df))
        ).clip(lower=0).round(2)
        return df

    def _assign_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Churn probability driven by domain logic so features carry signal.
        """
        score = np.zeros(len(df))

        # Month-to-month customers churn more
        score += np.where(df["contract_type"] == "Month-to-month", 0.30, 0.0)
        score += np.where(df["contract_type"] == "One year", 0.05, 0.0)

        # Electronic check correlates with churn in real data
        score += np.where(df["payment_method"] == "Electronic check", 0.12, 0.0)

        # Short tenure → higher risk
        score += np.where(df["tenure_months"] < 12, 0.20, 0.0)
        score += np.where(df["tenure_months"] > 48, -0.15, 0.0)

        # High support calls signal dissatisfaction
        score += (df["support_calls"] / 10.0) * 0.25

        # Low engagement
        score += np.where(df["days_since_last_login"] > 30, 0.10, 0.0)
        score += np.where(df["avg_session_minutes"] < 10, 0.08, 0.0)

        # High charges relative to products
        score += np.where(df["monthly_charges"] > 85, 0.10, 0.0)

        # Fiber optic has higher churn in telecom
        score += np.where(df["internet_service"] == "Fiber optic", 0.07, 0.0)

        # Add noise
        score += self.rng.normal(0, 0.08, len(df))

        # Sigmoid → probability
        prob = 1 / (1 + np.exp(-score * 2.5))

        # Adjust to target churn rate
        threshold = np.quantile(prob, 1 - self.cfg.data.churn_rate)
        df["churn"] = (prob >= threshold).astype(int)
        return df
