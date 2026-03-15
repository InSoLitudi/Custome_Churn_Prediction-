"""
preprocessor.py — sklearn ColumnTransformer wrapping numeric scaling
and one-hot encoding.  Fit only on training data to prevent leakage.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from config import cfg


class ChurnPreprocessor:
    """
    Wraps sklearn's ColumnTransformer for convenient fit/transform.

    Usage
    -----
    >>> prep = ChurnPreprocessor()
    >>> X_train_t = prep.fit_transform(X_train)
    >>> X_test_t  = prep.transform(X_test)
    """

    def __init__(self, config=None):
        self.cfg = config or cfg
        self.pipeline_ = None
        self.feature_names_out_ = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.pipeline_ = self._build()
        result = self.pipeline_.fit_transform(X)
        self.feature_names_out_ = self._get_feature_names(X)
        return result

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("Call fit_transform first.")
        return self.pipeline_.transform(X)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build(self) -> ColumnTransformer:
        num_cols = self._num_cols()
        cat_cols = self._cat_cols()

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols),
            ],
            remainder="drop",
        )

    def _num_cols(self):
        base = self.cfg.features.numeric_features.copy()
        # Include engineered numeric features present at transform time
        extra = [
            "charge_per_tenure", "support_call_rate", "avg_charge_per_product",
            "engagement_score", "is_disengaged", "is_high_value",
            "is_monthly_contract", "is_electronic_check",
        ]
        return base + extra

    def _cat_cols(self):
        return self.cfg.features.categorical_features.copy()

    def _get_feature_names(self, X: pd.DataFrame):
        try:
            num_names = self._num_cols()
            cat_names = (
                self.pipeline_
                    .named_transformers_["cat"]
                    .named_steps["ohe"]
                    .get_feature_names_out(self._cat_cols())
                    .tolist()
            )
            return num_names + cat_names
        except Exception:
            return None
