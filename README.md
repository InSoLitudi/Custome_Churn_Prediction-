# ChurnGuard — Customer Churn Prediction Dashboard

A production-style machine learning web application that predicts customer churn risk using a modular scikit-learn pipeline, served through a clean Flask dashboard. Built as a full end-to-end project covering data generation, feature engineering, model training, evaluation, and real-time inference.

---

## Overview

Customer churn — when a subscriber cancels or stops using a service — is one of the most costly problems in subscription-based businesses. Acquiring a new customer typically costs five times more than retaining an existing one. ChurnGuard tackles this by scoring every customer with a churn probability and surfacing the top risk factors driving that score, giving retention teams clear and actionable signals.

This project demonstrates how to structure an ML pipeline for maintainability and reuse: every stage lives in its own module with a single responsibility, and the Flask app simply wires them together into a live web interface.

---

## Features

**Machine Learning Pipeline**
- Synthetic telecom dataset with realistic churn correlations (10,000 records, ~27% churn rate)
- Domain-driven feature engineering: support call rate, engagement score, contract risk flags, charge-per-tenure ratio, and more
- Three classifiers trained and compared via 5-fold stratified cross-validation: Logistic Regression, Random Forest, and Gradient Boosting
- Automatic best-model selection based on ROC AUC
- Full evaluation suite: AUC, F1, precision, recall, accuracy, ROC curve, confusion matrix, and feature importances

**Flask Web Dashboard**
- **Overview page** — KPI strip, interactive ROC curve, model comparison chart, confusion matrix tiles, and feature importance bars
- **Customers page** — paginated table of 200 pre-scored customers with probability bars, risk band filters (Critical / High / Medium / Low), and actual churn labels
- **Predict page** — 10-field customer profile form that posts to a REST endpoint, animates a live gauge, and displays the top contributing risk factors; keeps a session history of all predictions

**Model Performance (Gradient Boosting)**

| Metric    | Score  |
|-----------|--------|
| ROC AUC   | 0.9646 |
| F1 Score  | 0.8250 |
| Precision | 0.8205 |
| Recall    | 0.8296 |
| Accuracy  | 0.9050 |

---

## Project Structure

```
churn_flask/
│
├── app.py                  # Flask application: routes, API endpoints, pipeline bootstrap
├── config.py               # Central configuration — hyperparameters, feature lists, thresholds
├── data_generator.py       # Synthetic dataset generation with realistic churn correlations
├── feature_engineer.py     # Domain-driven feature creation (ratios, engagement score, risk flags)
├── preprocessor.py         # sklearn ColumnTransformer — numeric scaling + one-hot encoding
├── trainer.py              # Multi-model training with cross-validation and best-model selection
├── evaluator.py            # Metrics, ROC curve, confusion matrix, feature importances
├── predictor.py            # End-to-end inference: raw customer record → probability + risk band
│
├── templates/
│   ├── base.html           # Shared layout with sidebar navigation
│   ├── index.html          # Overview / model performance page
│   ├── customers.html      # Paginated customer risk table
│   └── predict.html        # Live prediction form with animated gauge
│
└── static/
    ├── css/main.css        # Full stylesheet with CSS variables, responsive grid layout
    └── js/main.js          # Shared utilities and page transitions
```

Each module is independently importable and testable. The Flask app (`app.py`) only orchestrates — it holds no ML logic itself.

---

## Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone or download the project
cd churn_flask

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install flask scikit-learn pandas numpy

# 4. Run the app
python app.py
```

Open your browser at `http://127.0.0.1:5000`.

The pipeline trains automatically on first startup (takes ~15–30 seconds). All subsequent requests are served from the in-memory trained model.

---

## API Endpoints

| Method | Endpoint            | Description                                      |
|--------|---------------------|--------------------------------------------------|
| GET    | `/`                 | Overview dashboard                               |
| GET    | `/customers`        | Paginated customer table (`?page=1&risk=all`)    |
| GET    | `/predict`          | Prediction form UI                               |
| POST   | `/api/predict`      | JSON: single-customer churn prediction           |
| GET    | `/api/model-stats`  | JSON: metrics, ROC curve, feature importances    |
| GET    | `/api/customers`    | JSON: risk distribution summary                  |

**Example: POST `/api/predict`**

```json
{
  "tenure_months": 6,
  "monthly_charges": 89,
  "support_calls": 7,
  "days_since_last_login": 45,
  "avg_session_minutes": 12,
  "num_products": 2,
  "contract_type": "Month-to-month",
  "payment_method": "Electronic check",
  "internet_service": "Fiber optic",
  "gender": "Female"
}
```

**Response:**

```json
{
  "churn_probability": 0.847,
  "churn_prediction": 1,
  "prob_pct": 84.7,
  "risk_band": "🔴 Critical",
  "risk_label": "Critical",
  "risk_color": "danger",
  "factors": [
    { "name": "Support call rate",  "score": 19.1 },
    { "name": "Monthly contract",   "score": 17.8 },
    { "name": "Short tenure",       "score": 12.5 },
    { "name": "High charges",       "score": 6.3  },
    { "name": "Days since login",   "score": 5.0  }
  ]
}
```

---

## Key Design Decisions

**Single-responsibility modules.** Each `.py` file owns exactly one concern. You can swap out the trainer, replace the data source, or change the preprocessor without touching anything else.

**Config-driven.** All thresholds, feature lists, model hyperparameters, and split ratios live in `config.py`. Nothing is hard-coded across the codebase.

**Train-once, serve-many.** The pipeline boots at Flask startup and stays in memory. Inference is sub-millisecond after the initial training cost.

**No data leakage.** The `ChurnPreprocessor` is fitted only on training data and applied to the test set and live predictions identically, matching production behaviour.

**Risk banding.** Predictions are bucketed into four actionable bands (Critical ≥70%, High 50–70%, Medium 30–50%, Low <30%) so business teams can prioritise without reading raw probabilities.

---

## Customising for Real Data

To use your own customer dataset instead of the synthetic generator:

1. Replace `ChurnDataGenerator.generate()` in `data_generator.py` with a loader that reads your CSV or database and returns a `pandas.DataFrame` with the same column names defined in `config.py → FeatureConfig`.
2. Adjust `FeatureConfig.numeric_features` and `categorical_features` in `config.py` to match your schema.
3. Re-run `python app.py` — the rest of the pipeline adapts automatically.

---

## Tech Stack

| Layer       | Technology                              |
|-------------|-----------------------------------------|
| ML          | scikit-learn, pandas, NumPy             |
| Web         | Flask 3.x, Jinja2                       |
| Charts      | Chart.js 4 (CDN)                        |
| Styling     | Vanilla CSS with custom properties      |
| Fonts       | DM Sans + DM Mono (Google Fonts)        |

No heavy frontend framework is used — the dashboard is intentionally lightweight and easy to extend.

---

## License

MIT — free to use, modify, and distribute.