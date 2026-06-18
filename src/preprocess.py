"""
Shared preprocessing function used by train.py, predict.py, and all
serving endpoints (Streamlit + FastAPI).

Design principle: ONE code path for training AND inference — eliminates
train/serve skew.

Fixes applied vs. original:
  1. Date parsing now handles actual column names in cleaned CSV
  2. Leaky financial columns removed (order_item_total, etc.)
  3. High-cardinality categoricals frequency-encoded (not one-hot)
  4. Profit no longer clipped to >= 0 (negative profit is real signal)
  5. Raw date columns dropped after feature extraction
"""

import pandas as pd
import numpy as np
import warnings

# Import config — single source of truth for column lists
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LEAKY_COLUMNS,
    DROP_COLUMNS,
    HIGH_CARDINALITY_COLUMNS,
    LEAKAGE_CORRELATION_THRESHOLD,
)


def preprocess(df, is_training=False):
    """
    Clean and feature-engineer a supply chain DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or semi-cleaned DataFrame (either the original
        DataCoSupplyChain.csv or the already-cleaned CSV).
    is_training : bool
        If True, runs a leakage correlation check and raises
        a warning if any feature correlates > threshold with
        the target.

    Returns
    -------
    pd.DataFrame
        Cleaned, feature-engineered DataFrame ready for model
        training or inference.
    """
    df = df.copy()

    # -----------------------------------------------------------------
    # 1. STANDARDIZE COLUMN NAMES
    # -----------------------------------------------------------------
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # -----------------------------------------------------------------
    # 2. PARSE DATE COLUMNS
    #    FIX: The original code checked for 'shipping_date_(dateorders)'
    #    which only exists in the raw CSV.  The cleaned CSV has
    #    'order_date' and 'shipping_date' directly.  We now handle
    #    both cases.
    # -----------------------------------------------------------------
    order_date_col = None
    for candidate in ["order_date_(dateorders)", "order_date"]:
        if candidate in df.columns:
            df["_order_date_parsed"] = pd.to_datetime(
                df[candidate], errors="coerce"
            )
            order_date_col = "_order_date_parsed"
            break

    shipping_date_col = None
    for candidate in ["shipping_date_(dateorders)", "shipping_date"]:
        if candidate in df.columns:
            df["_shipping_date_parsed"] = pd.to_datetime(
                df[candidate], errors="coerce"
            )
            shipping_date_col = "_shipping_date_parsed"
            break

    # Validate: warn if > 5% of dates failed to parse
    if order_date_col is not None:
        null_pct = df[order_date_col].isna().mean()
        if null_pct > 0.05:
            warnings.warn(
                f"Date parsing: {null_pct:.1%} of order_date values are null "
                f"after parsing. Check the date format."
            )

    # -----------------------------------------------------------------
    # 3. EXTRACT TEMPORAL FEATURES (from order_date, not shipping_date)
    #    FIX: Original derived these from shipping_date which was NaT,
    #    resulting in all zeros.
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
        # Fallback — if pre-computed features already exist, keep them
        for col, default in [
            ("order_month", 0),
            ("order_day", 0),
            ("order_week", 0),
            ("is_weekend", False),
        ]:
            if col not in df.columns:
                df[col] = default

    # Clean up temporary parsed columns
    for tmp_col in ["_order_date_parsed", "_shipping_date_parsed"]:
        if tmp_col in df.columns:
            df.drop(columns=[tmp_col], inplace=True)

    # -----------------------------------------------------------------
    # 4. HANDLE SALES SAFELY (needed for log transform)
    # -----------------------------------------------------------------
    if "sales" not in df.columns:
        df["sales"] = 0

    # -----------------------------------------------------------------
    # 5. HANDLE PROFIT
    #    FIX: Removed .clip(lower=0) — negative profit is real business
    #    signal (loss-making orders).  Clipping destroyed that info.
    # -----------------------------------------------------------------
    if "order_profit_per_order" in df.columns:
        df["profit"] = df["order_profit_per_order"]
    elif "profit" not in df.columns:
        df["profit"] = 0

    # -----------------------------------------------------------------
    # 6. FEATURE ENGINEERING
    # -----------------------------------------------------------------
    # Delivery days
    if "days_for_shipping_(real)" in df.columns:
        df["delivery_days"] = df["days_for_shipping_(real)"]
    elif "delivery_days" not in df.columns:
        df["delivery_days"] = 0

    # Delay flag (from delivery_status, not from dates)
    if "delivery_status" in df.columns:
        df["delay_flag"] = (
            df["delivery_status"].str.lower() == "late delivery"
        ).astype(int)
    elif "delay_flag" not in df.columns:
        df["delay_flag"] = 0

    # -----------------------------------------------------------------
    # 7. SAFE LOG TRANSFORM
    #    Using log1p to handle zeros.  For profit, we use the absolute
    #    value with a sign indicator to preserve negative profits.
    # -----------------------------------------------------------------
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0).clip(lower=0)
    df["sales_log"] = np.log1p(df["sales"])

    df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0)
    df["profit_log"] = np.log1p(np.abs(df["profit"]))
    df["profit_negative"] = (df["profit"] < 0).astype(int)

    # -----------------------------------------------------------------
    # 8. FREQUENCY-ENCODE HIGH-CARDINALITY COLUMNS
    #    FIX: product_name, customer_city, etc. have thousands of
    #    unique values.  One-hot encoding them creates 28K+ columns.
    #    Frequency encoding replaces each value with its count in the
    #    dataset — captures popularity/commonness as a single number.
    # -----------------------------------------------------------------
    for col in HIGH_CARDINALITY_COLUMNS:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(0).astype(float)
            df.drop(columns=[col], inplace=True)

    # -----------------------------------------------------------------
    # 9. DROP LEAKY COLUMNS
    #    FIX: Original only dropped 'sales' and 'profit'.  Left
    #    order_item_total (r=0.99 with target), order_item_discount,
    #    order_profit_per_order, etc. — all derived from sales.
    # -----------------------------------------------------------------
    all_drop = list(set(DROP_COLUMNS + LEAKY_COLUMNS))
    df.drop(columns=[c for c in all_drop if c in df.columns], inplace=True)

    # -----------------------------------------------------------------
    # 10. FORCE TYPE CONSISTENCY
    # -----------------------------------------------------------------
    num_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str)

    # -----------------------------------------------------------------
    # 11. LEAKAGE DETECTION (training only)
    #     Automatically check if any remaining numeric feature
    #     correlates > threshold with the target.
    # -----------------------------------------------------------------
    if is_training and "sales_log" in df.columns:
        _assert_no_leakage(df, "sales_log")

    return df


def _assert_no_leakage(df, target_col):
    """
    Check if any numeric feature has dangerously high correlation
    with the target.  Raises a warning (not an error) so training
    can proceed but the issue is visible.
    """
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if target_col not in numeric_df.columns:
        return

    correlations = numeric_df.corr()[target_col].drop(target_col, errors="ignore")
    high_corr = correlations[correlations.abs() > LEAKAGE_CORRELATION_THRESHOLD]

    if len(high_corr) > 0:
        warnings.warn(
            f"\n⚠️  POTENTIAL LEAKAGE DETECTED!\n"
            f"The following features have |correlation| > "
            f"{LEAKAGE_CORRELATION_THRESHOLD} with '{target_col}':\n"
            f"{high_corr.to_string()}\n"
            f"Consider removing these features.\n",
            UserWarning,
            stacklevel=2,
        )


def predict_dataframe(df, model, columns):
    """
    Shared prediction function — used by both Streamlit and FastAPI
    to guarantee identical preprocessing during inference.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame (can be a single row or batch).
    model : sklearn.pipeline.Pipeline
        The loaded model pipeline.
    columns : list[str]
        The exact column list/order from training.

    Returns
    -------
    np.ndarray
        Predictions in real dollar values (not log-scale).
    """
    df = preprocess(df)
    
    # Intelligently fill missing columns based on their expected type
    preprocessor = model.named_steps["preprocessor"]
    
    cat_cols = []
    for name, transformer, cols_list in preprocessor.transformers_:
        if name == "cat":
            cat_cols = cols_list
            
    # Add missing columns with appropriate defaults
    for col in columns:
        if col not in df.columns:
            if col in cat_cols:
                df[col] = "Unknown"
            else:
                df[col] = 0.0
                
    # Force alignment and correct order
    df = df[columns]
    
    preds_log = model.predict(df)
    preds = np.expm1(preds_log)
    return preds