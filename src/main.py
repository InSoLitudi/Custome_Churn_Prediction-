"""
main.py — Orchestrates the full pipeline end-to-end and serialises results.
"""

import json
import sys
from sklearn.model_selection import train_test_split

from config import cfg
from data_generator import ChurnDataGenerator
from preprocessor.feature_engineer import FeatureEngineer
from preprocessor.preprocessor import ChurnPreprocessor
from trainer import ChurnTrainer
from evaluator import ChurnEvaluator
from predictor import ChurnPredictor


def run_pipeline():
    print("=" * 56)
    print("  Customer Churn Prediction Pipeline")
    print("=" * 56)

    # ── 1. Generate data ──────────────────────────────────────────────────────
    print("\n[1/5] Generating synthetic data …")
    gen = ChurnDataGenerator()
    df  = gen.generate()
    print(f"      {len(df):,} records  |  "
          f"churn rate = {df['churn'].mean():.1%}")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2/5] Engineering features …")
    engineer = FeatureEngineer()
    df_fe    = engineer.transform(df)
    print(f"      {df_fe.shape[1]} columns after engineering")

    # ── 3. Split & preprocess ─────────────────────────────────────────────────
    print("\n[3/5] Splitting & preprocessing …")
    target  = cfg.data.target_col
    feature_cols = [c for c in df_fe.columns if c != target]
    X = df_fe[feature_cols]
    y = df_fe[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=y,
    )
    print(f"      train={len(X_train):,}  test={len(X_test):,}")

    preprocessor = ChurnPreprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)
    print(f"      {X_train_t.shape[1]} features after encoding")

    # ── 4. Train models ───────────────────────────────────────────────────────
    print("\n[4/5] Training models (5-fold CV) …")
    trainer = ChurnTrainer()
    trainer.fit(X_train_t, y_train.values)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on held-out test set …")
    evaluator = ChurnEvaluator(
        trainer,
        feature_names=preprocessor.feature_names_out_,
    )
    results = evaluator.evaluate(X_test_t, y_test.values)
    evaluator.print_report()

    # ── Save results ──────────────────────────────────────────────────────────
    evaluator.to_json("results.json")
    print("  Results saved to results.json")

    # ── Demo: predictor on 5 new customers ───────────────────────────────────
    print("\n── Sample Predictions ──────────────────────────────")
    predictor = ChurnPredictor(engineer, preprocessor, trainer)
    raw_cols = [c for c in df.columns if c != target]
    sample = df[raw_cols].sample(5, random_state=7).reset_index(drop=True)
    preds  = predictor.predict_df(sample)
    for i, row in preds.iterrows():
        print(f"  Customer {i+1}:  P(churn)={row['churn_probability']:.2%}"
              f"  {row['risk_band']}")

    return results


if __name__ == "__main__":
    run_pipeline()
