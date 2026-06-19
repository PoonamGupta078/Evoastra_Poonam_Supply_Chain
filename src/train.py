"""
Model training script for Supply Chain Sales Prediction (Regression).

Leakage-free model: product_price and order_item_quantity are explicitly
removed because sales = product_price × order_item_quantity (r = 1.0).
profit_log is also removed as profit is derived from sales.

Produces:
  - artifacts/model.pkl      — fitted sklearn Pipeline (preprocessor + XGBoost)
  - artifacts/columns.pkl    — exact feature column list/order
  - artifacts/metrics.json   — R², RMSE, MAE, training metadata
"""

import json
import time
import os
import sys

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess  # noqa: E402
from config import (  # noqa: E402
    DATA_PATH,
    MODEL_PATH,
    COLUMNS_PATH,
    METRICS_PATH,
    TARGET_COL,
    PARAM_DIST,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    N_ITER,
    SALES_EXTRA_LEAKY,
)


def train():
    """Run the full training pipeline and save artifacts."""
    start_time = time.time()

    print("=" * 60)
    print("  Supply Chain Sales Prediction — Training (Leakage-Free)")
    print("=" * 60)
    print("\n  [INFO] Removed leaky features:")
    for col in SALES_EXTRA_LEAKY:
        print(f"     - {col}")

    print(f"\n[*] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    print(f"   Raw shape: {df.shape}")

    # Preprocess with SALES-specific leaky column removal
    df = preprocess(df, is_training=True, extra_drop_cols=SALES_EXTRA_LEAKY)
    print(f"   After preprocessing: {df.shape}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found after preprocessing.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"\n[*] Target: {TARGET_COL}")
    print(f"   Features: {X.shape[1]} columns")

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    print(f"   Categorical: {len(cat_cols)} | Numeric: {len(num_cols)}")

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ]
    )

    xgb = XGBRegressor(random_state=RANDOM_STATE, verbosity=0, device="cpu")
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", xgb)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n[*] Split: {len(X_train)} train / {len(X_test)} test")

    print(f"\n[*] RandomizedSearchCV: {N_ITER} iterations x {CV_FOLDS}-fold CV")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"   Best params: {search.best_params_}")

    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred)
    rmse_dollars = float(np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
    mae_dollars = float(mean_absolute_error(y_test_real, y_pred_real))

    elapsed = time.time() - start_time

    print(f"\n{'-' * 40}")
    print("  [METRICS] TEST SET METRICS (log-scale)")
    print(f"{'-' * 40}")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print("  [METRICS] TEST SET METRICS (real dollars)")
    print(f"{'-' * 40}")
    print(f"  RMSE : ${rmse_dollars:.2f}")
    print(f"  MAE  : ${mae_dollars:.2f}")
    print(f"  Training time: {elapsed:.1f}s")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), COLUMNS_PATH)

    metrics = {
        "r2": round(r2, 4),
        "rmse_log": round(rmse, 4),
        "mae_log": round(mae, 4),
        "rmse_dollars": round(rmse_dollars, 2),
        "mae_dollars": round(mae_dollars, 2),
        "n_features": X.shape[1],
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "best_params": search.best_params_,
        "training_time_seconds": round(elapsed, 1),
        "leakage_columns_removed": SALES_EXTRA_LEAKY,
        "random_state": RANDOM_STATE,
        "note": "Leakage-free: product_price and order_item_quantity removed",
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\n[OK] Artifacts saved:")
    print(f"   Model    -> {MODEL_PATH}")
    print(f"   Columns  -> {COLUMNS_PATH}")
    print(f"   Metrics  -> {METRICS_PATH}")
    print(f"\n{'=' * 60}")

    return best_model, metrics


if __name__ == "__main__":
    train()