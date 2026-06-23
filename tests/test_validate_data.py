import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from train import validate_data

def test_validate_empty_dataframe():
    """Empty dataframe should raise ValueError."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_data(df)

def test_validate_negative_price():
    """Negative prices should raise ValueError."""
    df = pd.DataFrame({
        "product_price": [10.0, -5.0, 20.0],
        "order_item_quantity": [1, 2, 1]
    })
    with pytest.raises(ValueError, match="negative product_price"):
        validate_data(df)

def test_validate_negative_quantity():
    """Negative quantity should raise ValueError."""
    df = pd.DataFrame({
        "product_price": [10.0, 15.0, 20.0],
        "order_item_quantity": [1, -2, 1]
    })
    with pytest.raises(ValueError, match="negative order_item_quantity"):
        validate_data(df)

def test_validate_clean_data_passes():
    """Clean data should not raise any exceptions."""
    df = pd.DataFrame({
        "product_price": [10.0, 15.0, 20.0],
        "order_item_quantity": [1, 2, 1],
        "order_date_(dateorders)": ["1/1/2026", "2/1/2026", "3/1/2026"]
    })
    # Should not raise exception
    validate_data(df)
