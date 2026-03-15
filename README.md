<div align="center">

# 🛡️ ChurnGuard
### Customer Churn Prediction Dashboard

*A production-style ML web app that predicts customer churn risk in real time*

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-16a77e?style=flat-square)

</div>

---

## 🌟 Overview

Customer churn — when a subscriber cancels or stops using a service — is one of the most costly problems in subscription-based businesses. **Acquiring a new customer typically costs five times more than retaining an existing one.** ChurnGuard tackles this by scoring every customer with a churn probability and surfacing the top risk factors driving that score, giving retention teams clear and actionable signals.

This project demonstrates how to structure an ML pipeline for maintainability and reuse: every stage lives in its own module with a single responsibility, and the Flask app simply wires them together into a live web interface.

> 💡 **Built as a full end-to-end project** — from synthetic data generation all the way to a live REST API and interactive dashboard.

---

## ✨ Features

### 🤖 Machine Learning Pipeline
- 📊 Synthetic telecom dataset with realistic churn correlations — 10,000 records, ~27% churn rate
- 🔧 Domain-driven feature engineering: support call rate, engagement score, contract risk flags, charge-per-tenure ratio, and more
- 🏆 Three classifiers trained and compared via 5-fold stratified cross-validation:
  - Logistic Regression
  - Random Forest
  - **Gradient Boosting** ✅ *(best model)*
- 🎯 Automatic best-model selection based on ROC AUC
- 📈 Full evaluation suite: AUC, F1, precision, recall, accuracy, ROC curve, confusion matrix, and feature importances

### 🖥️ Flask Web Dashboard

| Page | Description |
|------|-------------|
| 📊 **Overview** | KPI strip, interactive ROC curve, model comparison chart, confusion matrix tiles, and feature importance bars |
| 👥 **Customers** | Paginated table of 200 pre-scored customers with probability bars and risk band filters |
| 🔮 **Predict** | Live 10-field form with animated gauge, top risk factors, and session prediction history |

### 📉 Model Performance — Gradient Boosting

| Metric | Score | Rating |
|--------|-------|--------|
| 🎯 ROC AUC | **0.9646** | ⭐⭐⭐⭐⭐ |
| ⚖️ F1 Score | **0.8250** | ⭐⭐⭐⭐ |
| 🔍 Precision | **0.8205** | ⭐⭐⭐⭐ |
| 📡 Recall | **0.8296** | ⭐⭐⭐⭐ |
| ✅ Accuracy | **0.9050** | ⭐⭐⭐⭐⭐ |

---

## 🗂️ Project Structure

```
churn_flask/
│
├── 🐍 app.py                  # Flask application: routes, API endpoints, pipeline bootstrap
├── ⚙️  config.py               # Central configuration — hyperparameters, feature lists, thresholds
├── 🏭 data_generator.py       # Synthetic dataset generation with realistic churn correlations
├── 🔬 feature_engineer.py     # Domain-driven feature creation (ratios, engagement score, risk flags)
├── 🔄 preprocessor.py         # sklearn ColumnTransformer — numeric scaling + one-hot encoding
├── 🎓 trainer.py              # Multi-model training with cross-validation and best-model selection
├── 📊 evaluator.py            # Metrics, ROC curve, confusion matrix, feature importances
├── 🔮 predictor.py            # End-to-end inference: raw customer record → probability + risk band
│
├── 📄 templates/
│   ├── base.html              # Shared layout with sidebar navigation
│   ├── index.html             # Overview / model performance page
│   ├── customers.html         # Paginated customer risk table
│   └── predict.html           # Live prediction form with animated gauge
│
└── 🎨 static/
    ├── css/main.css           # Full stylesheet with CSS variables, responsive grid layout
    └── js/main.js             # Shared utilities and page transitions
```

> 💡 Each module is independently importable and testable. `app.py` only orchestrates — it holds no ML logic itself.

---

## 🚀 Installation

> **Requirements:** Python 3.9+

```bash
# 📥 1. Clone or download the project
cd churn_flask

# 🐍 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # 🪟 Windows: venv\Scripts\activate

# 📦 3. Install dependencies
pip install flask scikit-learn pandas numpy

# ▶️  4. Run the app
python app.py
```

🌐 Open your browser at **`http://127.0.0.1:5000`**

> ⏱️ The pipeline trains automatically on first startup (~15–30 seconds). All subsequent requests are served from the in-memory trained model.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | 📊 Overview dashboard |
| `GET` | `/customers` | 👥 Paginated customer table (`?page=1&risk=all`) |
| `GET` | `/predict` | 🔮 Prediction form UI |
| `POST` | `/api/predict` | 🤖 JSON: single-customer churn prediction |
| `GET` | `/api/model-stats` | 📈 JSON: metrics, ROC curve, feature importances |
| `GET` | `/api/customers` | 📋 JSON: risk distribution summary |

### 📤 Example Request — `POST /api/predict`

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

### 📥 Example Response

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

## 🎨 Risk Bands

Predictions are bucketed into four actionable bands so business teams can prioritise at a glance:

| Band | Threshold | Action |
|------|-----------|--------|
| 🔴 **Critical** | ≥ 70% | Immediate retention outreach |
| 🟠 **High** | 50–70% | Proactive intervention recommended |
| 🟡 **Medium** | 30–50% | Monitor and nurture |
| 🟢 **Low** | < 30% | Customer is stable |

---

## 🧠 Key Design Decisions

**🧩 Single-responsibility modules** — Each `.py` file owns exactly one concern. You can swap out the trainer, replace the data source, or change the preprocessor without touching anything else.

**⚙️ Config-driven** — All thresholds, feature lists, model hyperparameters, and split ratios live in `config.py`. Nothing is hard-coded across the codebase.

**⚡ Train-once, serve-many** — The pipeline boots at Flask startup and stays in memory. Inference is sub-millisecond after the initial training cost.

**🔒 No data leakage** — The `ChurnPreprocessor` is fitted only on training data and applied to the test set and live predictions identically, matching production behaviour.

---

## 🔧 Customising for Real Data

To plug in your own customer dataset instead of the synthetic generator:

1. 📂 Replace `ChurnDataGenerator.generate()` in `data_generator.py` with a loader that reads your CSV or database and returns a `pandas.DataFrame` with matching column names.
2. ✏️ Adjust `FeatureConfig.numeric_features` and `categorical_features` in `config.py` to match your schema.
3. ▶️ Re-run `python app.py` — the rest of the pipeline adapts automatically.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| 🤖 ML | scikit-learn, pandas, NumPy | Model training, feature engineering, data processing |
| 🌐 Web | Flask 3.x, Jinja2 | Routing, templating, REST API |
| 📊 Charts | Chart.js 4 (CDN) | ROC curve, model comparison bar chart |
| 🎨 Styling | Vanilla CSS with custom properties | Responsive layout, dark-mode ready variables |
| 🔤 Fonts | DM Sans + DM Mono (Google Fonts) | Clean UI and monospace metric display |

> 🪶 No heavy frontend framework — the dashboard is intentionally lightweight and easy to extend.

---

## 📄 License

MIT — free to use, modify, and distribute. ❤️

---

<div align="center">
  <sub>Built with 🐍 Python · 🌶️ Flask · 🤖 scikit-learn</sub>
</div>
