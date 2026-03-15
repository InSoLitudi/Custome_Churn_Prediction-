"""
predictor.py — Inference interface for single records or DataFrames.
Keeps preprocessing + trained model together in one serialisable bundle.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict

from preprocessor.feature_engineer import FeatureEngineer
from preprocessor import ChurnPreprocessor
from trainer import ChurnTrainer
from config import cfg


class ChurnPredictor:
    """
    End-to-end inference: raw DataFrame → churn probability + risk label.

    Usage
    -----
    >>> predictor = ChurnPredictor(engineer, preprocessor, trainer)
    >>> results   = predictor.predict_df(new_customers_df)
    """

    RISK_BANDS = [
        (0.70, "🔴 Critical"),
        (0.50, "🟠 High"),
        (0.30, "🟡 Medium"),
        (0.00, "🟢 Low"),
    ]

    def __init__(
        self,
        engineer: FeatureEngineer,
        preprocessor: ChurnPreprocessor,
        trainer: ChurnTrainer,
        config=None,
    ):
        self.engineer = engineer
        self.preprocessor = preprocessor
        self.trainer = trainer
        self.cfg = config or cfg

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: raw features → enriched prediction DataFrame."""
        X = self._transform(df)
        probas = self.trainer.predict_proba(X)
        preds  = (probas >= self.cfg.models.threshold).astype(int)

        out = df.copy()
        out["churn_probability"] = probas.round(4)
        out["churn_prediction"]  = preds
        out["risk_band"]         = [self._risk_label(p) for p in probas]
        return out

    def predict_single(self, record: Dict) -> Dict:
        """Convenience wrapper for a single customer dict."""
        df = pd.DataFrame([record])
        result = self.predict_df(df).iloc[0]
        return {
            "churn_probability": result["churn_probability"],
            "churn_prediction":  int(result["churn_prediction"]),
            "risk_band":         result["risk_band"],
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        df_fe = self.engineer.transform(df)
        return self.preprocessor.transform(df_fe)

    def _risk_label(self, prob: float) -> str:
        for threshold, label in self.RISK_BANDS:
            if prob >= threshold:
                return label
        return "🟢 Low"
