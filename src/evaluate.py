"""
Standalone evaluation script.

Loads the trained model and test data, computes metrics, and prints
a summary report.  This is what a real ML team runs after every
training job to validate the model before deploying it.
"""

import json
import os
import sys

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess
from config import (
    DATA_PATH,
    MODEL_PATH,
    COLUMNS_PATH,
    METRICS_PATH,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
)


def evaluate():
    """Load saved model, re-split data identically, and evaluate."""
    print("=" * 60)
    print("  Supply Chain Model — Evaluation Report")
    print("=" * 60)

    # Load model + columns
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    print(f"\n📂 Model loaded from: {MODEL_PATH}")

    # Load and preprocess data
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df = preprocess(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Reproduce the same split used during training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Align columns
    X_test = X_test.reindex(columns=columns, fill_value=0)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics — log scale
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    # Metrics — real dollar scale
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred)
    rmse_dollars = float(np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
    mae_dollars = float(mean_absolute_error(y_test_real, y_pred_real))

    print(f"\n{'─' * 40}")
    print(f"  📈 TEST SET METRICS (log-scale)")
    print(f"{'─' * 40}")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"\n  📈 TEST SET METRICS (real dollars)")
    print(f"{'─' * 40}")
    print(f"  RMSE : ${rmse_dollars:.2f}")
    print(f"  MAE  : ${mae_dollars:.2f}")

    # Load saved metrics for comparison
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            saved = json.load(f)
        print(f"\n  📋 Saved metrics (from training):")
        print(f"     R²   : {saved.get('r2', 'N/A')}")
        print(f"     RMSE : {saved.get('rmse_log', 'N/A')}")

    print(f"\n  Test samples: {len(X_test)}")
    print(f"  Features: {len(columns)}")
    print(f"{'=' * 60}")

    return {"r2": r2, "rmse": rmse, "mae": mae}


if __name__ == "__main__":
    evaluate()
