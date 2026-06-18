"""
Late Delivery Risk Classifier — Binary classification.

Target: delay_flag (1 = Late delivery, 0 = On time / Advance)

LEAKAGE-FREE DESIGN:
  Features used are ALL available at ORDER TIME (before delivery):
    ✅ days_for_shipment_(scheduled) — planned shipping days (known at order)
    ✅ shipping_mode                  — chosen at order time
    ✅ type (payment)                 — known at order time
    ✅ customer_segment               — known at order time
    ✅ department_name, category_name — product info at order time
    ✅ market, order_region           — geography, known at order time
    ✅ product_price, order_item_qty  — known at order time
    ✅ temporal features              — derived from order date

  Post-delivery information REMOVED (would be leakage):
    ❌ days_for_shipping_(real)  — actual days, only known AFTER delivery
    ❌ delivery_days             — engineered copy of the above
    ❌ delivery_status           — describes the delivery outcome (source of target)
    ❌ late_delivery_risk        — original column = (actual > scheduled), post-delivery
    ❌ sales_log                 — redundant; price and qty are kept as direct features

Produces:
  - artifacts/classifier_model.pkl   — fitted sklearn Pipeline
  - artifacts/classifier_columns.pkl — exact feature column order
  - artifacts/classifier_metrics.json — accuracy, AUC, F1, precision, recall
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess
from config import (
    DATA_PATH,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_COLUMNS_PATH,
    CLASSIFIER_METRICS_PATH,
    CLASSIFIER_TARGET_COL,
    CLASSIFIER_EXTRA_DROP,
    PARAM_DIST_CLF,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
)


def train_classifier():
    """Run the full classifier training pipeline and save artifacts."""
    start_time = time.time()

    print("=" * 60)
    print("  Supply Chain — Late Delivery Risk Classifier")
    print("=" * 60)
    print("\n  [INFO] Post-delivery columns removed (leakage prevention):")
    for col in CLASSIFIER_EXTRA_DROP:
        print(f"     - {col}")

    print(f"\n[*] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    print(f"   Raw shape: {df.shape}")

    # Preprocess with CLASSIFIER-specific leaky column removal
    df = preprocess(df, is_training=False, extra_drop_cols=CLASSIFIER_EXTRA_DROP)
    print(f"   After preprocessing: {df.shape}")

    if CLASSIFIER_TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{CLASSIFIER_TARGET_COL}' not found after preprocessing. "
            f"Ensure 'delivery_status' is in the raw CSV."
        )

    # Class distribution
    class_counts = df[CLASSIFIER_TARGET_COL].value_counts()
    class_pct = df[CLASSIFIER_TARGET_COL].value_counts(normalize=True) * 100
    print(f"\n[*] Target: {CLASSIFIER_TARGET_COL}")
    print(f"   0 (On Time) : {class_counts.get(0, 0):,}  ({class_pct.get(0, 0):.1f}%)")
    print(f"   1 (Late)    : {class_counts.get(1, 0):,}  ({class_pct.get(1, 0):.1f}%)")

    X = df.drop(columns=[CLASSIFIER_TARGET_COL])
    y = df[CLASSIFIER_TARGET_COL].astype(int)

    print(f"\n   Features: {X.shape[1]} columns")
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    print(f"   Categorical: {len(cat_cols)} | Numeric: {len(num_cols)}")

    # Pipeline
    preprocessor_step = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ]
    )

    xgb_clf = XGBClassifier(
        random_state=RANDOM_STATE,
        verbosity=0,
        eval_metric="logloss",
    )

    pipeline = Pipeline([("preprocessor", preprocessor_step), ("model", xgb_clf)])

    # Stratified split — preserves class ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[*] Split: {len(X_train)} train / {len(X_test)} test")

    print(f"\n[*] RandomizedSearchCV: 10 iterations x {CV_FOLDS}-fold CV (scoring: ROC-AUC)")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DIST_CLF,
        n_iter=10,
        cv=CV_FOLDS,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"   Best params: {search.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    elapsed = time.time() - start_time

    print(f"\n{'-' * 40}")
    print(f"  [METRICS] TEST SET METRICS")
    print(f"{'-' * 40}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}  (of predicted Late, how many are actually Late)")
    print(f"  Recall    : {recall:.4f}  (of actual Late, how many did we catch)")
    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"\n{classification_report(y_test, y_pred, target_names=['On Time', 'Late'])}")

    # Save artifacts
    os.makedirs(os.path.dirname(CLASSIFIER_MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, CLASSIFIER_MODEL_PATH)
    joblib.dump(X.columns.tolist(), CLASSIFIER_COLUMNS_PATH)

    metrics = {
        "model_type": "XGBoost Classifier (Late Delivery Risk)",
        "accuracy": round(accuracy, 4),
        "auc_roc": round(auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "n_features": X.shape[1],
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "class_distribution": {
            "on_time": int(class_counts.get(0, 0)),
            "late": int(class_counts.get(1, 0)),
        },
        "best_params": search.best_params_,
        "training_time_seconds": round(elapsed, 1),
        "leakage_columns_removed": CLASSIFIER_EXTRA_DROP,
        "note": "All features are available at ORDER TIME — no post-delivery leakage",
    }

    with open(CLASSIFIER_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n[OK] Artifacts saved:")
    print(f"   Model   -> {CLASSIFIER_MODEL_PATH}")
    print(f"   Columns -> {CLASSIFIER_COLUMNS_PATH}")
    print(f"   Metrics -> {CLASSIFIER_METRICS_PATH}")
    print(f"\n{'=' * 60}")

    return best_model, metrics


if __name__ == "__main__":
    train_classifier()
