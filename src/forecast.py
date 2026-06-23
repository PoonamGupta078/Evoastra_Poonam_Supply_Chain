"""
Revenue forecasting module using statsmodels Exponential Smoothing (Holt-Winters).

Why statsmodels instead of Prophet?
  - Prophet requires a C++ compiler (CmdStan / mingw32-make) to build its Stan backend.
  - statsmodels is pure Python — no compilation step, works on any OS out of the box.
  - Exponential Smoothing with trend + seasonality (Holt-Winters) is a well-established,
    interpretable time-series method that matches Prophet's capability for weekly supply
    chain data.

This module forecasts weekly revenue for the next 30 days based on historical order data.
"""

import os
import sys
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    DATA_PATH,
    FORECAST_MODEL_PATH,
    FORECAST_OUTPUT_PATH,
    FORECAST_METRICS_PATH,
)


def _build_weekly_series(data_path: str) -> pd.Series:
    """Load CSV and aggregate revenue to a weekly DatetimeIndex series."""
    df = pd.read_csv(data_path, encoding="latin1")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["revenue"] = df["product_price"] * df["order_item_quantity"]
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])
    weekly = (
        df.groupby(pd.Grouper(key="order_date", freq="W"))["revenue"]
        .sum()
    )
    return weekly


def train_forecast() -> None:
    """Train Holt-Winters model on weekly revenue and save artifacts."""
    print("=" * 60)
    print("  Supply Chain Revenue Forecasting (Holt-Winters ETS)")
    print("=" * 60)

    # 1. Load and aggregate
    print(f"[*] Loading data from: {DATA_PATH}")
    weekly = _build_weekly_series(DATA_PATH)

    # Remove sparse tail weeks (last week often partial data)
    weekly = weekly[weekly > 0]
    # Trim trailing weeks that have < 50% of the median volume (partial/sparse data)
    median_vol = weekly.median()
    while len(weekly) > 10 and weekly.iloc[-1] < 0.5 * median_vol:
        weekly = weekly.iloc[:-1]

    # 2. Train / holdout split (last 2 weeks = test)
    n_test = 2
    train_series = weekly.iloc[:-n_test]
    test_series  = weekly.iloc[-n_test:]
    print(f"[*] Split: {len(train_series)} weeks train / {len(test_series)} weeks test")

    # 3. Fit Holt's Double Exponential Smoothing (trend only)
    #    With ~160 weeks and only 3 full yearly cycles, estimating a 52-period
    #    seasonal component is statistically unreliable. Holt's linear trend
    #    method is more robust and typically more accurate on this data size.
    print("[*] Fitting Holt's Double Exponential Smoothing (trend) model...")
    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)

    # 4. Evaluate on holdout
    y_pred_test = model.forecast(n_test)
    y_true = test_series.values
    y_pred = y_pred_test.values

    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    print(f"\n{'-' * 40}")
    print("  [METRICS] HOLDOUT EVALUATION (2 Weeks)")
    print(f"{'-' * 40}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  RMSE : ${rmse:,.2f}")

    # 5. Retrain on full dataset, then forecast 30 days ahead (≈ 4-5 weeks)
    print("\n[*] Retraining on full dataset and forecasting next 30 days...")
    full_model = ExponentialSmoothing(
        weekly,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)

    # Forecast 5 weeks forward (covers ≥30 calendar days)
    forecast_weeks = full_model.forecast(5)

    # Build a daily output DataFrame (expand weekly forecast to daily)
    last_date = weekly.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq="D")
    # Interpolate weekly → daily by assigning each day the weekly average
    weekly_forecasts = forecast_weeks.values
    daily_yhat = np.repeat(weekly_forecasts, 7)[:30]  # repeat each week's value across 7 days

    # Simple confidence interval: ±15% band (reasonable for supply-chain revenue)
    ci_pct = 0.15
    output_df = pd.DataFrame({
        "ds":          future_dates.strftime("%Y-%m-%d"),
        "yhat":        daily_yhat.round(2),
        "yhat_lower":  (daily_yhat * (1 - ci_pct)).round(2),
        "yhat_upper":  (daily_yhat * (1 + ci_pct)).round(2),
    })

    # 6. Save artifacts
    os.makedirs(os.path.dirname(FORECAST_MODEL_PATH), exist_ok=True)
    joblib.dump(full_model, FORECAST_MODEL_PATH)
    output_df.to_csv(FORECAST_OUTPUT_PATH, index=False)

    metrics = {
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "n_train_weeks": int(len(train_series)),
        "n_test_weeks": int(n_test),
        "forecast_horizon_days": 30,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(FORECAST_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[OK] Artifacts saved:")
    print(f"   Model   -> {FORECAST_MODEL_PATH}")
    print(f"   Output  -> {FORECAST_OUTPUT_PATH}")
    print(f"   Metrics -> {FORECAST_METRICS_PATH}")
    print(f"{'=' * 60}\n")


def get_forecast() -> pd.DataFrame:
    """Return the saved 30-day forecast CSV as a DataFrame."""
    if not os.path.exists(FORECAST_OUTPUT_PATH):
        raise FileNotFoundError(
            f"Forecast output not found at {FORECAST_OUTPUT_PATH}. "
            "Run `python src/forecast.py` first."
        )
    return pd.read_csv(FORECAST_OUTPUT_PATH)


if __name__ == "__main__":
    train_forecast()
