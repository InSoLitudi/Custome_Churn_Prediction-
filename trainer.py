"""
trainer.py — Trains multiple classifiers and picks the best by CV AUC.
Each model is wrapped in a lightweight object so evaluation stays clean.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import cfg


@dataclass
class TrainedModel:
    name: str
    estimator: Any
    cv_auc: float
    params: Dict[str, Any] = field(default_factory=dict)


class ChurnTrainer:
    """
    Fits all configured classifiers and exposes the best one.

    Usage
    -----
    >>> trainer = ChurnTrainer()
    >>> trainer.fit(X_train, y_train)
    >>> best = trainer.best_model
    """

    _REGISTRY = {
        "logistic_regression": LogisticRegression,
        "random_forest":       RandomForestClassifier,
        "gradient_boosting":   GradientBoostingClassifier,
    }

    def __init__(self, config=None):
        self.cfg = config or cfg
        self.trained_models_: Dict[str, TrainedModel] = {}
        self.best_model: Optional[TrainedModel] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ChurnTrainer":
        cv = StratifiedKFold(
            n_splits=self.cfg.models.cv_folds, shuffle=True,
            random_state=self.cfg.data.random_state,
        )

        for name, params in self.cfg.models.models.items():
            cls = self._REGISTRY.get(name)
            if cls is None:
                print(f"[trainer] Unknown model '{name}', skipping.")
                continue

            estimator = cls(**params)
            auc_scores = cross_val_score(
                estimator, X, y,
                cv=cv, scoring="roc_auc", n_jobs=-1,
            )
            cv_auc = float(auc_scores.mean())

            # Final fit on full training set
            estimator.fit(X, y)

            tm = TrainedModel(
                name=name, estimator=estimator,
                cv_auc=cv_auc, params=params,
            )
            self.trained_models_[name] = tm
            print(f"  [{name}] CV AUC = {cv_auc:.4f} "
                  f"(±{auc_scores.std():.4f})")

        self.best_model = max(
            self.trained_models_.values(), key=lambda m: m.cv_auc
        )
        print(f"\n  ✓ Best model: {self.best_model.name} "
              f"(AUC {self.best_model.cv_auc:.4f})")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return churn probability using the best model."""
        return self.best_model.estimator.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        thr = threshold or self.cfg.models.threshold
        return (self.predict_proba(X) >= thr).astype(int)
