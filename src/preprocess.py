"""
Shared preprocessing function used by train.py, train_classifier.py,
and all serving endpoints (Streamlit + FastAPI).

Design principle: ONE code path for training AND inference — eliminates
train/serve skew.

Key leakage-prevention parameters:
  extra_drop_cols — allows each model to exclude its own leaky features:
    • SALES_EXTRA_LEAKY      → removes product_price, order_item_quantity, profit_log
    • CLASSIFIER_EXTRA_DROP  → removes post-delivery observations

Fixes applied vs. original:
  1. Date parsing handles both raw CSV and cleaned CSV column names
  2. Leaky financial columns removed (order_item_total, etc.)
  3. High-cardinality categoricals frequency-encoded (not one-hot)
  4. Profit no longer clipped to >= 0 (negative profit is real signal)
  5. Raw date columns dropped after feature extraction
  6. extra_drop_cols parameter for per-model leakage control
"""

import pandas as pd
import numpy as np
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LEAKY_COLUMNS,
    DROP_COLUMNS,
    HIGH_CARDINALITY_COLUMNS,
    LEAKAGE_CORRELATION_THRESHOLD,
)


def preprocess(df, is_training=False, extra_drop_cols=None):
    """
    Clean and feature-engineer a supply chain DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or semi-cleaned DataFrame.
    is_training : bool
        If True, runs leakage correlation check.
    extra_drop_cols : list[str] or None
        Additional columns to drop AFTER base processing.
        Use SALES_EXTRA_LEAKY for the regression model.
        Use CLASSIFIER_EXTRA_DROP for the delivery classifier.

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame ready for training or inference.
    """
    df = df.copy()

    # -----------------------------------------------------------------
    # 1. STANDARDIZE COLUMN NAMES
    # -----------------------------------------------------------------
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # -----------------------------------------------------------------
    # 2. PARSE DATE COLUMNS
    #    Handles both raw CSV ('order_date_(dateorders)') and
    #    cleaned CSV ('order_date') column names.
    # -----------------------------------------------------------------
    order_date_col = None
    for candidate in ["order_date_(dateorders)", "order_date"]:
        if candidate in df.columns:
            df["_order_date_parsed"] = pd.to_datetime(df[candidate], errors="coerce")
            order_date_col = "_order_date_parsed"
            break

    for candidate in ["shipping_date_(dateorders)", "shipping_date"]:
        if candidate in df.columns:
            df["_shipping_date_parsed"] = pd.to_datetime(df[candidate], errors="coerce")
            break

    if order_date_col is not None:
        null_pct = df[order_date_col].isna().mean()
        if null_pct > 0.05:
            warnings.warn(
                f"Date parsing: {null_pct:.1%} of order_date values are null. "
                f"Check the date format."
            )

    # -----------------------------------------------------------------
    # 3. EXTRACT TEMPORAL FEATURES (from order_date)
    # -----------------------------------------------------------------
    if order_date_col is not None:
        df["order_month"] = df[order_date_col].dt.month.fillna(0).astype(int)
        df["order_day"] = df[order_date_col].dt.day.fillna(0).astype(int)
        df["order_week"] = (
            df[order_date_col]
            .dt.isocalendar()
            .week.astype(float)
            .fillna(0)
            .astype(int)
        )
        df["is_weekend"] = df[order_date_col].dt.weekday.fillna(0).astype(int) >= 5
    else:
        for col, default in [
            ("order_month", 0),
            ("order_day", 0),
            ("order_week", 0),
            ("is_weekend", False),
        ]:
            if col not in df.columns:
                df[col] = default

    for tmp_col in ["_order_date_parsed", "_shipping_date_parsed"]:
        if tmp_col in df.columns:
            df.drop(columns=[tmp_col], inplace=True)

    # -----------------------------------------------------------------
    # 4. HANDLE SALES (needed for log transform)
    # -----------------------------------------------------------------
    if "sales" not in df.columns:
        df["sales"] = 0

    # -----------------------------------------------------------------
    # 5. HANDLE PROFIT
    # -----------------------------------------------------------------
    if "order_profit_per_order" in df.columns:
        df["profit"] = df["order_profit_per_order"]
    elif "profit" not in df.columns:
        df["profit"] = 0

    # -----------------------------------------------------------------
    # 6. FEATURE ENGINEERING
    # -----------------------------------------------------------------
    # Delivery days (engineered from actual shipping days)
    if "days_for_shipping_(real)" in df.columns:
        df["delivery_days"] = df["days_for_shipping_(real)"]
    elif "delivery_days" not in df.columns:
        df["delivery_days"] = 0

    # Delay flag — binary target for the classifier
    # (1 = Late delivery, 0 = On time / Advance shipping)
    if "delivery_status" in df.columns:
        df["delay_flag"] = (
            df["delivery_status"].str.lower() == "late delivery"
        ).astype(int)
    elif "delay_flag" not in df.columns:
        df["delay_flag"] = 0

    # -----------------------------------------------------------------
    # 7. SAFE LOG TRANSFORM
    # -----------------------------------------------------------------
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0).clip(lower=0)
    df["sales_log"] = np.log1p(df["sales"])

    df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0)
    df["profit_log"] = np.log1p(np.abs(df["profit"]))
    df["profit_negative"] = (df["profit"] < 0).astype(int)

    # -----------------------------------------------------------------
    # 8. FREQUENCY-ENCODE HIGH-CARDINALITY COLUMNS
    # -----------------------------------------------------------------
    for col in HIGH_CARDINALITY_COLUMNS:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(0).astype(float)
            df.drop(columns=[col], inplace=True)

    # -----------------------------------------------------------------
    # 9. DROP BASE LEAKY COLUMNS (financial derivations of sales)
    # -----------------------------------------------------------------
    all_drop = list(set(DROP_COLUMNS + LEAKY_COLUMNS))
    df.drop(columns=[c for c in all_drop if c in df.columns], inplace=True)

    # -----------------------------------------------------------------
    # 10. DROP MODEL-SPECIFIC LEAKY COLUMNS
    #     extra_drop_cols lets each model specify its own leaky features.
    #     • Sales model:      SALES_EXTRA_LEAKY (removes price × qty formula)
    #     • Classifier model: CLASSIFIER_EXTRA_DROP (removes post-delivery info)
    # -----------------------------------------------------------------
    if extra_drop_cols:
        df.drop(
            columns=[c for c in extra_drop_cols if c in df.columns],
            inplace=True,
        )

    # -----------------------------------------------------------------
    # 11. FORCE TYPE CONSISTENCY
    # -----------------------------------------------------------------
    num_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str)

    # -----------------------------------------------------------------
    # 12. LEAKAGE DETECTION (training only)
    # -----------------------------------------------------------------
    if is_training and "sales_log" in df.columns:
        _assert_no_leakage(df, "sales_log")

    return df


def _assert_no_leakage(df, target_col):
    """Warn if any numeric feature correlates > threshold with target."""
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if target_col not in numeric_df.columns:
        return

    correlations = numeric_df.corr()[target_col].drop(target_col, errors="ignore")
    high_corr = correlations[correlations.abs() > LEAKAGE_CORRELATION_THRESHOLD]

    if len(high_corr) > 0:
        warnings.warn(
            f"\n⚠️  POTENTIAL LEAKAGE DETECTED!\n"
            f"Features with |correlation| > {LEAKAGE_CORRELATION_THRESHOLD} "
            f"with '{target_col}':\n"
            f"{high_corr.to_string()}\n",
            UserWarning,
            stacklevel=2,
        )


def _get_pipeline_cat_cols(model):
    """Return categorical input columns from a fitted sklearn pipeline preprocessor."""
    if not hasattr(model, "named_steps"):
        return []

    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "transformers_"):
        return []

    cat_cols = []
    for name, transformer, cols_list in preprocessor.transformers_:
        if name == "cat":
            cat_cols = list(cols_list)
            break
    return cat_cols


# =========================================================================
# INFERENCE HELPERS
# =========================================================================

def predict_dataframe(df, model, columns):
    """
    Sales regression inference.

    Preprocesses input, fills missing features intelligently, and returns
    predicted sales in real dollars (inverse log transform applied).

    Parameters
    ----------
    df : pd.DataFrame
        Raw user input (single row or batch).
    model : sklearn Pipeline
        Fitted sales regression pipeline.
    columns : list[str]
        Exact training feature column order.

    Returns
    -------
    np.ndarray  — predicted sales in dollars
    """
    from config import SALES_EXTRA_LEAKY

    df = preprocess(df, extra_drop_cols=SALES_EXTRA_LEAKY)

    cat_cols = _get_pipeline_cat_cols(model)
    for col in columns:
        if col not in df.columns:
            df[col] = "Unknown" if col in cat_cols else 0.0

    df = df[columns]

    preds_log = model.predict(df)
    return np.expm1(preds_log)


def predict_classifier(df, model, columns):
    """
    Late delivery risk classifier inference.

    Preprocesses input with classifier-specific leakage removal,
    and returns (binary_prediction, probability_of_late_delivery).

    Parameters
    ----------
    df : pd.DataFrame
        Raw user input at ORDER time (no post-delivery info needed).
    model : sklearn Pipeline
        Fitted delivery classifier pipeline.
    columns : list[str]
        Exact training feature column order for the classifier.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (binary_predictions, late_delivery_probabilities)
    """
    from config import CLASSIFIER_EXTRA_DROP

    df = preprocess(df, extra_drop_cols=CLASSIFIER_EXTRA_DROP)

    cat_cols = _get_pipeline_cat_cols(model)
    for col in columns:
        if col not in df.columns:
            df[col] = "Unknown" if col in cat_cols else 0.0

    # Remove the target column if it accidentally ended up in df
    # (delay_flag may be 0 as a placeholder from preprocessing)
    if "delay_flag" in df.columns and "delay_flag" not in columns:
        df = df.drop(columns=["delay_flag"])

    df = df[columns]

    preds = model.predict(df)
    proba = model.predict_proba(df)[:, 1]   # probability of class 1 (Late)
    return preds, proba