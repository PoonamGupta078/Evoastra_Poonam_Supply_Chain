# 📦 Supply Chain Sales Prediction — MLOps Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://supply-chain-analytics-capstone-h4rlfx53hsdwa4uyybnjwz.streamlit.app/)

An end-to-end ML system for predicting supply chain sales values, with automated training, real-time inference, SHAP explainability, and Streamlit deployment.

🔗 **Live Demo:** https://supply-chain-analytics-capstone-h4rlfx53hsdwa4uyybnjwz.streamlit.app

---

## 🎯 Problem Statement

Build an intelligent supply chain analytics system that forecasts demand, identifies bottleneck risks, optimizes inventory allocation, and supports real-time decision-making — trained on global supply chain datasets exceeding 100K records.

---

## 🔁 End-to-End Flow

```
Raw Data → Preprocessing → Feature Alignment → Encoding → XGBoost Model → Prediction → Streamlit UI
```

### 1. Raw Input
Source: `DataCoSupplyChain.csv` (180K+ rows)

Example input fields: `Type`, `Days for shipping (real)`, `Product Price`, `Customer Segment`

### 2. Preprocessing (`preprocess.py`)
- Clean column names and handle missing values
- Extract date features (month, week, day of week)
- Engineer features: `delay_flag`, `is_weekend`
- Apply log transforms: `sales_log`, `profit_log`

### 3. Feature Alignment & Encoding
```python
df = df.reindex(columns=columns, fill_value=0)
X_transformed = preprocessor.transform(df_single)
```

### 4. Model — XGBoost Regressor
```python
pred_log = model.predict(X_transformed)
prediction = np.expm1(pred_log)  # Reverse log transform
```
Best result: **R² = 0.967 | RMSE = 20.87**

### 5. Streamlit Frontend
- 📊 EDA tab: metrics, bar charts, pie charts from real data
- 🔮 Predict tab: user inputs → live XGBoost prediction

---

## 🏗️ CI/CD Pipeline
```
Code Push / Schedule
    → GitHub Actions
        → train.py (retrain model)
        → predict.py (test inference)
        → Update artifacts/
        → Auto Deploy (Railway / Render)
            → FastAPI Response
               →Streamlit Cloud
```

---

##  Project Structure

```
evoastra-supply-chain-capstone/
├── data/
│   ├── raw/                        # Raw CSV datasets
│   ├── processed/                  # Cleaned and feature-engineered data
│   └── data_dictionary.md
├── notebooks/
│   ├── 01_eda_analytics.ipynb
│   ├── 02_statistical_modeling.ipynb
│   └── 03_ml_pipeline.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── inference_api.py
├── models/
│   └── best_model.pkl
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── api_config.yaml
├── requirements.txt
└── README.md
```

---

##  Phase Progression

### Phase 1 — Data Analytics
- Data cleaning, missing value handling, outlier detection
- Exploratory data analysis (univariate, bivariate, multivariate)
- KPI definition: Order Fulfillment Rate, Lead Time Variability, Inventory Turnover
- Interactive dashboard (Power BI / Streamlit)
- Root cause analysis using Pareto charts and correlation heatmaps

**Tools:** Python, Pandas, Matplotlib, Seaborn, Plotly, Power BI

---

### Phase 2 — Data Science
- Feature engineering: lag features, rolling averages, interaction terms
- Statistical testing: t-tests, ANOVA, chi-squared, Granger causality
- Regression models: Linear, Ridge, Lasso
- Time-series forecasting: ARIMA, SARIMA, Prophet
- Model evaluation: RMSE, MAE, R², cross-validation
- Business risk scenario matrices (best / worst / likely)

**Tools:** Scikit-learn, Statsmodels, SciPy, Prophet

---

### Phase 3 — AI / Machine Learning
- Advanced models: Random Forest, XGBoost, Neural Networks
- Hyperparameter tuning: GridSearchCV, Optuna
- Model explainability: SHAP values, force plots, summary plots
- REST API deployment with FastAPI
- Model monitoring: prediction drift, data drift, retraining triggers

**Tools:** XGBoost, TensorFlow, SHAP, FastAPI, Docker, MLflow, Evidently AI

---

---

## 📈 Model Results

| Phase | Model | R² | RMSE |
|-------|-------|----|------|
| Phase 2 | Lasso Regression | 0.9306 | 35.08 |
| Phase 3 | Random Forest | 0.964 | 22.14 |
| Phase 3 | **XGBoost (best)** | **0.967** | **20.87** |

---

## 🔧 Tech Stack

| Area | Tools |
|------|-------|
| Analytics | Python, Pandas, Matplotlib, Seaborn, Plotly |
| ML / AI | XGBoost, Scikit-learn, SHAP, Optuna |
| Frontend | Streamlit |
| Backend | FastAPI, Docker |
| CI/CD | GitHub Actions |

---

## 🚀 Run Locally

```bash
git clone https://github.com/Aharshi3614/supply-chain-analytics-capstone.git
cd supply-chain-analytics-capstone
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 System Capabilities

- ✅ Live Streamlit dashboard with EDA + Prediction
- ✅ XGBoost model with R² = 0.967
- ✅ Deployed on Streamlit Cloud (free, public URL)
- ✅ FastAPI backend with Docker support
- ✅ CI/CD via GitHub Actions

---

## Architecture Diagram

![architecture_diagram](https://github.com/user-attachments/assets/b81d9861-afac-45c7-bad2-485040a888fa)

---

© 2026 Evoastra Ventures (OPC) Pvt Ltd. All rights reserved.
