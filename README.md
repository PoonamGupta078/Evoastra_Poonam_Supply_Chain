# 📦 Supply Chain Analytics — Production ML System

> **Dual-Model ML Pipeline | XGBoost Regression + Classification | Time-Series Forecasting | SHAP Explainability | Data Drift Monitoring | CI/CD**

A production-grade, end-to-end machine learning system for supply chain optimization. This project goes far beyond a typical Kaggle notebook — it implements a fully operational ML pipeline with leakage-free preprocessing, model serving (FastAPI + Streamlit), automated drift detection, and GitHub Actions CI/CD.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                      │
│  data/supplychain_cleaned.csv  (41,650 orders × 43 features)        │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────────┐
        ▼               ▼                   ▼
┌──────────────┐ ┌──────────────┐  ┌──────────────────┐
│ src/train.py │ │ src/train_   │  │ src/forecast.py  │
│  XGBoost     │ │ classifier.py│  │  Holt-Winters    │
│  Regression  │ │  XGBoost     │  │  Exponential     │
│  (Sales)     │ │  Classifier  │  │  Smoothing       │
│              │ │  (Delivery)  │  │  (Revenue)       │
└──────┬───────┘ └──────┬───────┘  └────────┬─────────┘
       │                │                   │
       ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ARTIFACT LAYER (artifacts/)                       │
│  model.pkl │ classifier_model.pkl │ forecast_model.pkl │ *.json     │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│   app.py     │ │ src/app.py   │ │ src/drift_       │
│  Streamlit   │ │  FastAPI     │ │ detector.py      │
│  Dashboard   │ │  REST API    │ │  PSI Monitoring  │
│  (7 Tabs)    │ │  (Swagger)   │ │                  │
└──────────────┘ └──────────────┘ └──────────────────┘
```

---

## ✨ Key Features

### 🤖 Machine Learning Models
| Model | Type | Target | Key Metric | Purpose |
|-------|------|--------|------------|---------|
| **Sales Predictor** | XGBoost Regressor | `sales_log` | R² = 0.69 | Predict order-level revenue |
| **Delivery Risk** | XGBoost Classifier | `delay_flag` | AUC-ROC ≈ 0.72 | Predict late delivery probability |
| **Revenue Forecast** | Holt-Winters ETS | Weekly Revenue | MAPE ≈ 34% | Forecast next 30 days of revenue |

### 🛡️ Data Leakage Prevention
- **Sales model**: `product_price` and `order_item_quantity` are removed (because `sales = price × qty` is a formula, not ML)
- **Delivery model**: Post-delivery observations removed (actual shipping days, delivery status)
- **Automated leakage detector**: Warns if any feature has |correlation| > 0.95 with the target

### 📊 7-Tab Streamlit Dashboard
1. **📊 EDA** — Interactive Plotly charts, distribution analysis, correlation heatmaps
2. **💰 Predict Sales** — Enter order features → get revenue prediction with SHAP explanation
3. **🚚 Delivery Risk** — Enter order features → get late delivery probability with SHAP
4. **📈 Revenue Forecast** — 30-day Holt-Winters forecast with uncertainty bands
5. **🔮 Forecast Explainability** — Level/Trend/Seasonal decomposition of the ETS model
6. **💼 Business Insights** — KPIs: total revenue, AOV, late delivery rate, market breakdown
7. **⚙️ Model Monitoring** — All model metrics + data drift status in one view

### 🔍 Model Explainability (SHAP)
- **TreeExplainer** for both XGBoost models
- Feature importance waterfall charts for every single prediction
- Handles sparse matrix conversion automatically

### 📡 Data Drift Detection (PSI)
- Population Stability Index (PSI) computed per feature
- Thresholds: Stable (< 0.1), Moderate (0.1–0.25), Major (> 0.25)
- Automatic retraining recommendation when major drift is detected

### 🚀 Production Serving
- **FastAPI** REST API with Swagger docs (`/docs`)
- **Lifespan-managed** model loading (loads once at startup via `app.state`)
- **Docker + Docker Compose** for containerized deployment
- **GitHub Actions** CI/CD pipeline (lint → test → validate structure)

---

## 📁 Project Structure

```
supply-chain-analytics_evoastra/
├── app.py                      # Streamlit dashboard (7 tabs)
├── requirements.txt            # Pinned dependencies
├── retrain.sh                  # One-command full retraining script
├── README.md
├── LICENSE
│
├── src/
│   ├── config.py               # Centralized config (paths, features, hyperparameters)
│   ├── preprocess.py           # Shared preprocessing pipeline (leakage-free)
│   ├── train.py                # XGBoost Regressor training (sales prediction)
│   ├── train_classifier.py     # XGBoost Classifier training (delivery risk)
│   ├── evaluate.py             # Model evaluation with multiple metrics
│   ├── predict.py              # Inference utilities
│   ├── forecast.py             # Holt-Winters ETS revenue forecasting
│   ├── drift_detector.py       # PSI-based data drift monitoring
│   └── app.py                  # FastAPI REST API service
│
├── tests/
│   ├── test_preprocess.py      # 4 tests: leakage removal, target creation, date extraction, idempotency
│   ├── test_predict.py         # Schema alignment test
│   ├── test_forecast.py        # Forecast output column validation
│   ├── test_drift.py           # 5 tests: PSI computation, drift detection, retrain recommendation
│   └── test_validate_data.py   # 4 tests: empty data, negative prices/quantities, clean data
│
├── artifacts/                  # Serialized models, metrics, and reference data
│   ├── model.pkl               # Sales XGBoost model
│   ├── classifier_model.pkl    # Delivery XGBoost model
│   ├── forecast_model.pkl      # Holt-Winters ETS model
│   ├── forecast_output.csv     # 30-day forecast values
│   ├── columns.pkl / classifier_columns.pkl
│   ├── metrics.json / classifier_metrics.json / forecast_metrics.json
│   ├── drift_report.json       # PSI drift report
│   ├── reference_data.pkl      # Reference distribution for drift detection
│   └── frequency_maps.pkl      # High-cardinality column encodings
│
├── data/
│   └── supplychain_cleaned.csv # 41,650 orders × 43 features
│
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions: lint → test → validate
│
├── notebooks/                  # Jupyter exploration notebooks
├── docs/                       # Additional documentation
└── dashboard/                  # Dashboard assets
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone & Install
```bash
git clone https://github.com/PoonamGupta078/Evoastra_Poonam_Supply_Chain.git
cd Evoastra_Poonam_Supply_Chain
pip install -r requirements.txt
```

### 2. Train Models (optional — pre-trained artifacts included)
```bash
python src/train.py                # Sales regression
python src/train_classifier.py     # Delivery classifier
python src/forecast.py             # Revenue forecast
python src/drift_detector.py       # Data drift report
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

### 4. Run the FastAPI Service (optional)
```bash
uvicorn src.app:app --reload
```
Open **http://127.0.0.1:8000/docs** for interactive API docs.

### 5. Run Tests
```bash
pytest tests/ -v
```

### 6. Run with Docker (optional)
```bash
cd deployment
docker-compose up --build
```

---

## 🧠 Technical Deep-Dive

### Preprocessing Pipeline (`src/preprocess.py`)
- **Column normalization**: All column names lowercased, spaces → underscores
- **Date feature engineering**: Extracts `order_month`, `order_day`, `order_week`, `is_weekend` from order dates
- **Target engineering**: Creates `sales_log = log1p(sales)` for regression, `delay_flag` for classification
- **Leakage removal**: Drops 15+ columns that are mathematically derived from the target
- **High-cardinality encoding**: Frequency encoding for columns with 100+ unique values (product names, cities, zip codes)
- **Mixed-type pipeline**: `ColumnTransformer` with `StandardScaler` for numeric + `OneHotEncoder` for categorical
- **Leakage assertion**: Automated correlation check post-preprocessing — raises warning if any feature has |r| > 0.95 with target

### XGBoost Sales Regression (`src/train.py`)
- **Hyperparameter tuning**: `RandomizedSearchCV` with 15 iterations × 3-fold CV
- **Search space**: learning_rate ∈ {0.01, 0.05, 0.1}, max_depth ∈ {3, 5, 7}, n_estimators ∈ {100, 200, 300}
- **Evaluation**: R², RMSE, MAE on both log-scale and real dollar scale
- **Artifacts**: Saves model, column order, and metrics as `.pkl` / `.json`

### XGBoost Delivery Classifier (`src/train_classifier.py`)
- **Class imbalance**: `scale_pos_weight` tuned in {1, 2, 3}
- **Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, full classification report
- **Feature set**: Only order-time features (no post-delivery observations)

### Revenue Forecasting (`src/forecast.py`)
- **Model**: Holt's Double Exponential Smoothing (statsmodels `ExponentialSmoothing`)
- **Why not Prophet?** Prophet requires CmdStan (C++ compiler) — statsmodels is pure Python, runs everywhere
- **Data prep**: Weekly aggregation, sparse tail-week trimming (< 50% of median volume)
- **Holdout**: Last 2 weeks held out for MAPE/RMSE evaluation
- **Output**: 30-day daily forecast with ±15% confidence bands, saved as CSV

### Data Drift Detection (`src/drift_detector.py`)
- **Method**: Population Stability Index (PSI) — compares current data distribution to training reference
- **Thresholds**: PSI < 0.1 (stable), 0.1–0.25 (moderate), > 0.25 (major drift)
- **Recommendation engine**: Automatically flags when retraining is needed
- **Coverage**: All numeric features in the training set

### Centralized Configuration (`src/config.py`)
- **Single source of truth**: All paths, feature lists, leaky columns, hyperparameter grids
- **188 lines** of documented configuration — no magic strings anywhere in the codebase
- **Leakage documentation**: Every leaky column has a comment explaining *why* it leaks

---

## 🧪 Test Suite

| Test File | Tests | What It Verifies |
|-----------|-------|-----------------|
| `test_preprocess.py` | 4 | Leakage removal, target creation, date feature extraction, idempotency |
| `test_predict.py` | 1 | DataFrame schema alignment for inference |
| `test_forecast.py` | 1 | Forecast output contains expected columns (ds, yhat, yhat_lower, yhat_upper) |
| `test_drift.py` | 5 | PSI computation, shifted distributions, retrain recommendation logic |
| `test_validate_data.py` | 4 | Empty data handling, negative values, clean data validation |
| **Total** | **15** | — |

---

## 🔄 CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

```
Push to main/dev → Code Linting (flake8) → Run Tests (pytest) → Validate Structure
```

- **Code Linting**: `flake8` with max-line-length=120 and sensible ignores
- **Run Tests**: Full pytest suite — pipeline fails if any test fails
- **Validate Structure**: Checks that required files exist (requirements.txt, README, Dockerfile, etc.)

---

## 📊 Dataset

- **Source**: DataCo Global Supply Chain Dataset
- **Size**: 41,650 orders × 43 features
- **Date range**: January 2015 – January 2018
- **Key columns**: order details, product info, customer demographics, shipping data, delivery status

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **ML Framework** | XGBoost 2.1.3, scikit-learn 1.5.2 |
| **Time-Series** | statsmodels (Holt-Winters ETS) |
| **Explainability** | SHAP 0.46.0 |
| **Dashboard** | Streamlit 1.40.1, Plotly 5.24.1 |
| **API** | FastAPI 0.115.8, Uvicorn, Pydantic |
| **Testing** | pytest 9.1.0 |
| **CI/CD** | GitHub Actions |
| **Deployment** | Docker, Docker Compose |
| **Data** | Pandas 2.2.3, NumPy 1.26.4 |

---

## 📜 License

See [LICENSE](./LICENSE) for details.

---

## 👤 Author

**Poonam Gupta** — [GitHub](https://github.com/PoonamGupta078)

Built as part of the Evoastra internship program.
