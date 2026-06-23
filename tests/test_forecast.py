import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from forecast import get_forecast  # noqa: E402


def test_forecast_output_columns(monkeypatch, tmp_path):
    """Test that get_forecast() returns a DataFrame with the expected columns."""
    import pandas as pd

    mock_df = pd.DataFrame({
        "ds": ["2026-01-01"],
        "yhat": [100.0],
        "yhat_lower": [90.0],
        "yhat_upper": [110.0],
    })

    # Write mock CSV to a temp file and patch FORECAST_OUTPUT_PATH
    mock_csv = tmp_path / "forecast_output.csv"
    mock_df.to_csv(mock_csv, index=False)

    monkeypatch.setattr(
        "forecast.FORECAST_OUTPUT_PATH",
        str(mock_csv),
    )

    df = get_forecast()

    assert "ds" in df.columns
    assert "yhat" in df.columns
    assert "yhat_lower" in df.columns
    assert "yhat_upper" in df.columns
    assert len(df) == 1
