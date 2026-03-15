"""
feature_engineer.py — Domain-driven feature creation.
Operates on raw DataFrame columns; returns an enriched copy.
"""

import numpy as np
import pandas as pd
from config import cfg


class FeatureEngineer:
    """
    Adds business-logic features before the sklearn preprocessing step.

    Usage
    -----
    >>> fe  = FeatureEngineer()
    >>> df2 = fe.transform(df)
    """

    def __init__(self, config=None):
        self.cfg = config or cfg

    # ── Public API ────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._ratio_features(df)
        df = self._engagement_features(df)
        df = self._value_segment(df)
        df = self._contract_risk(df)
        return df

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ratio features expose hidden patterns."""
        df["charge_per_tenure"] = (
            df["monthly_charges"] / df["tenure_months"].clip(lower=1)
        ).round(4)
        df["support_call_rate"] = (
            df["support_calls"] / df["tenure_months"].clip(lower=1)
        ).round(4)
        df["avg_charge_per_product"] = (
            df["monthly_charges"] / df["num_products"].clip(lower=1)
        ).round(2)
        return df

    def _engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite engagement score (0–1, higher = more engaged)."""
        recency = 1 - df["days_since_last_login"].clip(0, 90) / 90
        frequency = df["avg_session_minutes"].clip(0, 120) / 120
        df["engagement_score"] = ((recency + frequency) / 2).round(4)
        df["is_disengaged"] = (df["engagement_score"] < 0.25).astype(int)
        return df

    def _value_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag high-value customers (top-25 % monthly charges)."""
        threshold = df["monthly_charges"].quantile(0.75)
        df["is_high_value"] = (df["monthly_charges"] >= threshold).astype(int)
        return df

    def _contract_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode known high-churn contract types as a binary risk flag."""
        df["is_monthly_contract"] = (
            df["contract_type"] == "Month-to-month"
        ).astype(int)
        df["is_electronic_check"] = (
            df["payment_method"] == "Electronic check"
        ).astype(int)
        return df
