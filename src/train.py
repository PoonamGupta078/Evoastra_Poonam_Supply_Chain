"""
Model training script for Supply Chain Sales Prediction.

Produces:
  - artifacts/model.pkl      — fitted sklearn Pipeline (preprocessor + XGBoost)
  - artifacts/columns.pkl    — exact feature column list/order
  - artifacts/metrics.json   — R², RMSE, MAE, training metadata

Fixes vs. original:
  1. Added random_state to train_test_split for reproducibility
  2. Computes and saves R²/RMSE/MAE on held-out test set
  3. Prints clear training summary
  4. Uses centralized config for all parameters
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

# Add src to path so config is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess
from config import (
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
)


def train():
    """Run the full training pipeline and save artifacts."""
    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. LOAD & PREPROCESS
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Supply Chain Sales Prediction — Training")
    print("=" * 60)

    print(f"\n📂 Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    print(f"   Raw shape: {df.shape}")

    df = preprocess(df, is_training=True)
    print(f"   After preprocessing: {df.shape}")

    # ------------------------------------------------------------------
    # 2. SPLIT FEATURES / TARGET
    # ------------------------------------------------------------------
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found after preprocessing.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"\n🎯 Target: {TARGET_COL}")
    print(f"   Features: {X.shape[1]} columns")

    # ------------------------------------------------------------------
    # 3. IDENTIFY COLUMN TYPES
    # ------------------------------------------------------------------
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    print(f"   Categorical: {len(cat_cols)} | Numeric: {len(num_cols)}")

    # ------------------------------------------------------------------
    # 4. BUILD PIPELINE
    # ------------------------------------------------------------------
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

    xgb = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", xgb)])

    # ------------------------------------------------------------------
    # 5. TRAIN / TEST SPLIT — with fixed random_state for reproducibility
    #    FIX: Original had no random_state, giving different results each run.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"\n📊 Split: {len(X_train)} train / {len(X_test)} test")

    # ------------------------------------------------------------------
    # 6. HYPERPARAMETER SEARCH
    # ------------------------------------------------------------------
    print(f"\n🔍 RandomizedSearchCV: {N_ITER} iterations × {CV_FOLDS}-fold CV")

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

    # ------------------------------------------------------------------
    # 7. EVALUATE ON HELD-OUT TEST SET
    #    FIX: Original never computed or saved any metrics.
    # ------------------------------------------------------------------
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    # Also compute metrics in real dollar space (inverse log transform)
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred)
    rmse_dollars = float(np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
    mae_dollars = float(mean_absolute_error(y_test_real, y_pred_real))

    elapsed = time.time() - start_time

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
    print(f"\n  ⏱️  Training time: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 8. SAVE ARTIFACTS
    # ------------------------------------------------------------------
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
        "leakage_columns_removed": True,
        "random_state": RANDOM_STATE,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n✅ Artifacts saved:")
    print(f"   Model    → {MODEL_PATH}")
    print(f"   Columns  → {COLUMNS_PATH}")
    print(f"   Metrics  → {METRICS_PATH}")
    print(f"\n{'=' * 60}")

    return best_model, metrics


if __name__ == "__main__":
    train()