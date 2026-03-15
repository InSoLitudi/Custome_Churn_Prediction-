"""
config.py — Central configuration for the churn prediction pipeline.
All hyperparameters, paths, and flags live here so nothing is hard-coded.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    n_samples: int = 10_000
    test_size: float = 0.20
    val_size: float = 0.10          # fraction of TRAIN used for validation
    random_state: int = 42
    target_col: str = "churn"
    churn_rate: float = 0.27        # realistic ~27 % churn


# ── Features ──────────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    numeric_features: List[str] = field(default_factory=lambda: [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_products",
        "support_calls",
        "avg_session_minutes",
        "days_since_last_login",
    ])
    categorical_features: List[str] = field(default_factory=lambda: [
        "contract_type",
        "payment_method",
        "internet_service",
        "gender",
    ])
    engineered_features: List[str] = field(default_factory=lambda: [
        "charge_per_tenure",
        "support_call_rate",
        "engagement_score",
        "is_high_value",
    ])


# ── Models ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "logistic_regression": {
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
        },
        "gradient_boosting": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "random_state": 42,
        },
    })
    threshold: float = 0.40         # classification decision threshold
    cv_folds: int = 5


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    use_smote: bool = False         # flip to True to enable oversampling
    scale_numeric: bool = True
    tune_hyperparams: bool = False  # enable grid-search when True


# ── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    primary_metric: str = "roc_auc"
    report_metrics: List[str] = field(default_factory=lambda: [
        "roc_auc", "f1", "precision", "recall", "accuracy",
    ])


# ── Master config ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# Singleton used by the rest of the pipeline
cfg = Config()