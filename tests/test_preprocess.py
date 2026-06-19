import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from preprocess import preprocess  # noqa: E402
from config import LEAKY_COLUMNS  # noqa: E402

def test_preprocess_drops_leaky_columns():
    """Ensure all mathematically derived target-leaky columns are removed."""
    # Create dummy df with leaky columns
    data = {col: [1, 2, 3] for col in LEAKY_COLUMNS}
    data["sales"] = [10, 20, 30]
    data["profit"] = [5, 10, 15]
    data["order_date"] = ["2020-01-01", "2020-01-02", "2020-01-03"]
    df = pd.DataFrame(data)

    df_cleaned = preprocess(df)

    # Check leaky columns are gone
    for col in LEAKY_COLUMNS:
        assert col not in df_cleaned.columns, f"Leaky column {col} survived preprocessing!"

def test_preprocess_creates_target():
    """Ensure the log-transformed target variables are created."""
    df = pd.DataFrame({"sales": [0, 10, 100], "profit": [-5, 0, 5]})
    df_cleaned = preprocess(df)

    assert "sales_log" in df_cleaned.columns
    assert "profit_log" in df_cleaned.columns
    assert "profit_negative" in df_cleaned.columns

    # Verify log transform: log1p(0) = 0, log1p(10) ~ 2.39
    assert df_cleaned["sales_log"].iloc[0] == 0.0
    assert np.isclose(df_cleaned["sales_log"].iloc[1], np.log1p(10))

    # Verify absolute log on profit
    assert df_cleaned["profit_negative"].iloc[0] == 1
    assert np.isclose(df_cleaned["profit_log"].iloc[0], np.log1p(5))

def test_preprocess_date_extraction():
    """Ensure temporal features are extracted and raw date is dropped."""
    df = pd.DataFrame({
        "order_date": ["2026-01-15 10:00:00", "2026-06-20 14:30:00"],
        "sales": [100, 200]
    })
    
    df_cleaned = preprocess(df)
    
    assert "order_date" not in df_cleaned.columns
    assert "order_month" in df_cleaned.columns
    
    # Month should be 1 and 6
    assert df_cleaned["order_month"].iloc[0] == 1
    assert df_cleaned["order_month"].iloc[1] == 6

def test_preprocess_is_idempotent():
    """Running preprocess twice should not break or change schema/values."""
    df = pd.DataFrame({
        "sales": [10.0, 20.0],
        "profit": [5.0, 10.0],
        "order_date": ["2026-01-01", "2026-01-02"]
    })
    
    df_first = preprocess(df, is_training=True)
    df_second = preprocess(df_first, is_training=False)
    
    pd.testing.assert_frame_equal(df_first, df_second)
