"""
evaluator.py — Computes, stores, and pretty-prints all evaluation metrics
plus feature importances.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score,
    roc_curve, confusion_matrix,
    classification_report,
)

from config import cfg
from trainer import ChurnTrainer


class ChurnEvaluator:
    """
    Evaluates a ChurnTrainer on a held-out test set.

    Usage
    -----
    >>> ev = ChurnEvaluator(trainer, feature_names)
    >>> results = ev.evaluate(X_test, y_test)
    >>> ev.print_report()
    """

    def __init__(self, trainer: ChurnTrainer,
                 feature_names: Optional[List[str]] = None,
                 config=None):
        self.trainer = trainer
        self.feature_names = feature_names
        self.cfg = config or cfg
        self.results_: Dict[str, Any] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        proba = self.trainer.predict_proba(X)
        preds = self.trainer.predict(X)

        metrics = {
            "roc_auc":   round(roc_auc_score(y, proba), 4),
            "f1":        round(f1_score(y, preds, zero_division=0), 4),
            "precision": round(precision_score(y, preds, zero_division=0), 4),
            "recall":    round(recall_score(y, preds, zero_division=0), 4),
            "accuracy":  round(accuracy_score(y, preds), 4),
        }

        fpr, tpr, thresholds = roc_curve(y, proba)
        cm = confusion_matrix(y, preds)

        self.results_ = {
            "model_name":   self.trainer.best_model.name,
            "metrics":      metrics,
            "all_models":   {
                name: {"cv_auc": round(m.cv_auc, 4)}
                for name, m in self.trainer.trained_models_.items()
            },
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
            "confusion_matrix": cm.tolist(),
            "feature_importances": self._feature_importances(),
        }
        return self.results_

    def print_report(self) -> None:
        if not self.results_:
            raise RuntimeError("Call evaluate first.")
        m = self.results_["metrics"]
        sep = "─" * 48
        print(f"\n{sep}")
        print(f"  Model : {self.results_['model_name']}")
        print(sep)
        for k, v in m.items():
            print(f"  {k:<15} {v:.4f}")
        print(sep)
        print(f"\n  Confusion matrix:")
        cm = np.array(self.results_["confusion_matrix"])
        print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

        fi = self.results_["feature_importances"]
        if fi:
            print(f"\n  Top-10 Feature Importances:")
            df = pd.DataFrame(fi).nlargest(10, "importance")
            for _, row in df.iterrows():
                bar = "█" * int(row["importance"] * 40)
                print(f"    {row['feature']:<35} {bar}  {row['importance']:.4f}")
        print()

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.results_, f, indent=2)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _feature_importances(self):
        est = self.trainer.best_model.estimator
        importances = None

        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
        elif hasattr(est, "coef_"):
            importances = np.abs(est.coef_[0])

        if importances is None or self.feature_names is None:
            return []

        return [
            {"feature": name, "importance": round(float(imp), 6)}
            for name, imp in zip(self.feature_names, importances)
        ]