"""
Standalone evaluation script.

Evaluates BOTH trained models and prints a full side-by-side report:
  1. Sales Regression  (XGBoost Regressor) — R², RMSE, MAE
  2. Delivery Classifier (XGBoost Classifier) — Accuracy, AUC-ROC, F1, Precision, Recall

This is what a real ML team runs after every training job to validate
both models before deploying them.
"""

import json
import os
import sys

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess  # noqa: E402
from config import (  # noqa: E402
    DATA_PATH,
    MODEL_PATH,
    COLUMNS_PATH,
    METRICS_PATH,
    TARGET_COL,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_COLUMNS_PATH,
    CLASSIFIER_METRICS_PATH,
    CLASSIFIER_TARGET_COL,
    CLASSIFIER_EXTRA_DROP,
    TEST_SIZE,
    RANDOM_STATE,
    SALES_EXTRA_LEAKY,
)


# ─────────────────────────────────────────────────────────────────────
# REGRESSION EVALUATION
# ─────────────────────────────────────────────────────────────────────
def evaluate_regression():
    """Load saved regression model, re-split data identically, and evaluate."""
    print("=" * 60)
    print("  [1/2] SALES REGRESSION EVALUATION")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"  [SKIP] Model not found at {MODEL_PATH}. Run src/train.py first.")
        return None

    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    print(f"\n[INFO] Model loaded from: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df = preprocess(df, extra_drop_cols=SALES_EXTRA_LEAKY)

    if TARGET_COL not in df.columns:
        print(f"  [ERROR] Target column '{TARGET_COL}' missing after preprocessing.")
        return None

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_test = X_test.reindex(columns=columns, fill_value=0)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred)
    rmse_dollars = float(np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
    mae_dollars = float(mean_absolute_error(y_test_real, y_pred_real))

    print(f"\n{'-' * 40}")
    print("  [METRICS] TEST SET METRICS (log-scale)")
    print(f"{'-' * 40}")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print("\n  [METRICS] TEST SET METRICS (real dollars)")
    print(f"{'-' * 40}")
    print(f"  RMSE : ${rmse_dollars:.2f}")
    print(f"  MAE  : ${mae_dollars:.2f}")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            saved = json.load(f)
        print("\n  [INFO] Saved metrics (from training):")
        print(f"     R²   : {saved.get('r2', 'N/A')}")
        print(f"     RMSE : {saved.get('rmse_log', 'N/A')}")

    print(f"\n  Test samples : {len(X_test)}")
    print(f"  Features     : {len(columns)}")
    print(f"{'=' * 60}\n")

    return {"r2": r2, "rmse": rmse, "mae": mae,
            "rmse_dollars": rmse_dollars, "mae_dollars": mae_dollars}


# ─────────────────────────────────────────────────────────────────────
# CLASSIFIER EVALUATION
# ─────────────────────────────────────────────────────────────────────
def evaluate_classifier():
    """Load saved classifier model, re-split data identically, and evaluate."""
    print("=" * 60)
    print("  [2/2] LATE DELIVERY CLASSIFIER EVALUATION")
    print("=" * 60)

    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        print(f"  [SKIP] Model not found at {CLASSIFIER_MODEL_PATH}. Run src/train_classifier.py first.")
        return None

    model = joblib.load(CLASSIFIER_MODEL_PATH)
    columns = joblib.load(CLASSIFIER_COLUMNS_PATH)
    print(f"\n[INFO] Model loaded from: {CLASSIFIER_MODEL_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df = preprocess(df, extra_drop_cols=CLASSIFIER_EXTRA_DROP)

    if CLASSIFIER_TARGET_COL not in df.columns:
        print(f"  [ERROR] Target column '{CLASSIFIER_TARGET_COL}' missing after preprocessing.")
        return None

    X = df.drop(columns=[CLASSIFIER_TARGET_COL])
    y = df[CLASSIFIER_TARGET_COL].astype(int)

    # Stratified split — same strategy as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_test = X_test.reindex(columns=columns, fill_value=0)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(f"\n{'-' * 40}")
    print("  [METRICS] TEST SET METRICS")
    print(f"{'-' * 40}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}  (of predicted Late, how many are actually Late)")
    print(f"  Recall    : {recall:.4f}  (of actual Late, how many did we catch)")

    if os.path.exists(CLASSIFIER_METRICS_PATH):
        with open(CLASSIFIER_METRICS_PATH, "r") as f:
            saved = json.load(f)
        print("\n  [INFO] Saved metrics (from training):")
        print(f"     AUC-ROC  : {saved.get('auc_roc', 'N/A')}")
        print(f"     F1 Score : {saved.get('f1_score', 'N/A')}")
        print(f"     Recall   : {saved.get('recall', 'N/A')}")

    print("\n  Class distribution in test set:")
    print(f"     On Time (0) : {(y_test == 0).sum():,}")
    print(f"     Late    (1) : {(y_test == 1).sum():,}")

    print(f"\n{'-' * 40}")
    print("  [REPORT] Per-Class Classification Report")
    print(f"{'-' * 40}")
    print(classification_report(y_test, y_pred, target_names=["On Time", "Late"]))

    print(f"  Test samples : {len(X_test)}")
    print(f"  Features     : {len(columns)}")
    print(f"{'=' * 60}\n")

    return {
        "accuracy": accuracy, "auc": auc, "f1": f1,
        "precision": precision, "recall": recall,
    }


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def evaluate():
    """Run full evaluation report for both models."""
    reg_metrics = evaluate_regression()
    clf_metrics = evaluate_classifier()
    return {"regression": reg_metrics, "classifier": clf_metrics}


if __name__ == "__main__":
    evaluate()
