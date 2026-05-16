from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import shap
from src.preprocess import preprocess
app = FastAPI()

# Load artifacts
model = joblib.load("artifacts/model.pkl")
columns = joblib.load("artifacts/columns.pkl")

# 🔥 get components for  SHAP (no change to your pipeline)
preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]

explainer = shap.TreeExplainer(xgb_model)


@app.get("/")
def home():
    return {"message": "Single Prediction API"}


@app.get("/predict")
def predict():

    # -------------------------
    # LOAD DATA (same as predict.py)
    # -------------------------
    df = pd.read_csv("data/supplychain_cleaned.csv", encoding="latin1")

    # -------------------------
    # PREPROCESS
    # -------------------------
    df = preprocess(df)

    # -------------------------
    # ALIGN COLUMNS
    # -------------------------
    df = df.reindex(columns=columns, fill_value=0)

    # -------------------------
    # TAKE ONE SAMPLE
    # -------------------------
    df_single = df.iloc[[0]]

    # -------------------------
    # PREDICT
    # -------------------------
    pred_log = model.predict(df_single)[0]
    prediction = float(np.expm1(pred_log))

    # -------------------------
    # TRANSFORM FOR SHAP
    # -------------------------
    X_transformed = preprocessor.transform(df_single)
    
    # -------------------------
    # CLEAN SHAP (READABLE)
    # -------------------------
    def clean_feature_name(name):
        name = name.replace("num__", "").replace("cat__", "")
        name = name.replace("_", " ").title()
        return name
    
    shap_values = explainer(X_transformed)
    shap_values = shap_values.values[0]
    
    feature_names = preprocessor.get_feature_names_out()
    
    shap_dict = {}
    for i in range(len(feature_names)):
        val = float(shap_values[i])
        if not np.isfinite(val):
            val = 0.0
        
        clean_name = clean_feature_name(feature_names[i])
        shap_dict[clean_name] = val
    
    top_features = dict(
        sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    )

    # -------------------------
    # RETURN
    # -------------------------
    return {
        "raw_input": df_single.to_dict(),
        "prediction": prediction,
        "top_features": top_features
    }
