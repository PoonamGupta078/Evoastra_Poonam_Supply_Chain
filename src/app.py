"""
FastAPI inference service for Supply Chain Sales Prediction.

Endpoints:
  GET  /         — welcome message
  GET  /health   — container health check (for Docker)
  POST /predict  — single prediction with SHAP explanation
  POST /predict/batch — batch predictions

Fixes vs. original:
  1. /predict changed from GET to POST with Pydantic request model
  2. Added /health endpoint (docker-compose healthcheck was failing)
  3. Accepts custom order data instead of always predicting row 0
  4. Uses shared preprocess() function
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess, predict_dataframe
from config import MODEL_PATH, COLUMNS_PATH

# ── App Setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="Supply Chain Sales Prediction API",
    description="XGBoost-based sales prediction with SHAP explainability",
    version="2.0.0",
)

# ── Load Artifacts (once at startup) ────────────────────────────────
model = joblib.load(MODEL_PATH)
columns = joblib.load(COLUMNS_PATH)

# Extract components for SHAP
preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]
explainer = shap.TreeExplainer(xgb_model)


# ── Request / Response Models ────────────────────────────────────────
class OrderInput(BaseModel):
    """Schema for a single order prediction request."""

    type: str = Field("debit", description="Payment type")
    days_for_shipping_real: int = Field(3, ge=0, le=60)
    days_for_shipment_scheduled: int = Field(4, ge=0, le=60)
    product_price: float = Field(50.0, ge=0)
    customer_segment: str = Field("consumer")
    order_item_quantity: int = Field(5, ge=1, le=100)
    department_name: str = Field("fitness")
    shipping_mode: str = Field("standard class")


class BatchInput(BaseModel):
    """Schema for batch prediction requests."""

    orders: list[OrderInput]


class PredictionResponse(BaseModel):
    """Schema for a single prediction response."""

    predicted_sales: float
    top_features: dict
    input_summary: dict


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    model_loaded: bool
    n_features: int


# ── Helper ───────────────────────────────────────────────────────────
def _clean_feature_name(name: str) -> str:
    """Strip sklearn auto-generated prefixes for readable feature names."""
    name = name.replace("num__", "").replace("cat__", "")
    name = name.replace("_", " ").title()
    return name


def _order_to_dataframe(order: OrderInput) -> pd.DataFrame:
    """Convert a Pydantic OrderInput into a DataFrame row."""
    real = order.days_for_shipping_real
    scheduled = order.days_for_shipment_scheduled

    return pd.DataFrame(
        [
            {
                "type": order.type,
                "days_for_shipping_(real)": float(real),
                "days_for_shipment_(scheduled)": float(scheduled),
                "product_price": order.product_price,
                "customer_segment": order.customer_segment,
                "order_item_quantity": float(order.order_item_quantity),
                "department_name": order.department_name,
                "shipping_mode": order.shipping_mode,
                "delivery_status": (
                    "late delivery" if real > scheduled else "shipping on time"
                ),
                "late_delivery_risk": 1 if real > scheduled else 0,
                "sales": order.product_price * order.order_item_quantity,
            }
        ]
    )


def _get_shap_explanation(df_single: pd.DataFrame) -> dict:
    """Compute SHAP values for a single preprocessed row."""
    df_processed = preprocess(df_single)
    df_aligned = df_processed.reindex(columns=columns, fill_value=0)

    X_transformed = preprocessor.transform(df_aligned)
    shap_values = explainer(X_transformed)
    vals = shap_values.values[0]

    feature_names = preprocessor.get_feature_names_out()
    shap_dict = {}
    for i, name in enumerate(feature_names):
        val = float(vals[i])
        if not np.isfinite(val):
            val = 0.0
        shap_dict[_clean_feature_name(name)] = round(val, 4)

    # Return top 10 by absolute value
    return dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10])


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/")
def home():
    """Welcome endpoint."""
    return {
        "message": "Supply Chain Sales Prediction API",
        "version": "2.0.0",
        "endpoints": ["/health", "/predict", "/predict/batch", "/docs"],
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """
    Health check endpoint for Docker container healthchecks.
    FIX: This endpoint didn't exist — docker-compose was checking
    /health but getting 404, marking container as unhealthy.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        n_features=len(columns),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderInput):
    """
    Predict sales for a single order.

    FIX: Changed from GET (always predicting row 0) to POST
    with a proper Pydantic request model that accepts custom
    order data via JSON body.
    """
    df_input = _order_to_dataframe(order)

    # Use shared prediction function — same preprocessing as training
    prediction = predict_dataframe(df_input, model, columns)
    pred_value = float(prediction[0])

    # SHAP explanation
    top_features = _get_shap_explanation(df_input)

    return PredictionResponse(
        predicted_sales=round(pred_value, 2),
        top_features=top_features,
        input_summary=order.model_dump(),
    )


@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    """
    Predict sales for multiple orders at once.
    New endpoint — matches what api_config.yaml specifies.
    """
    results = []
    for order in batch.orders:
        df_input = _order_to_dataframe(order)
        prediction = predict_dataframe(df_input, model, columns)
        results.append(
            {
                "predicted_sales": round(float(prediction[0]), 2),
                "input_summary": order.model_dump(),
            }
        )

    return {"predictions": results, "count": len(results)}
