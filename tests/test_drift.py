import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from drift_detector import compute_psi, check_drift, recommend_retrain  # noqa: E402

def test_psi_identical_distributions():
    """Identical distributions should have PSI close to 0."""
    np.random.seed(42)
    reference = pd.Series(np.random.normal(0, 1, 1000))
    current = reference.copy()
    
    psi = compute_psi(reference, current)
    assert psi < 0.01

def test_psi_shifted_distribution():
    """A majorly shifted distribution should have high PSI (>0.25)."""
    np.random.seed(42)
    reference = pd.Series(np.random.normal(0, 1, 1000))
    current = pd.Series(np.random.normal(10, 1, 1000))
    
    psi = compute_psi(reference, current)
    assert psi > 0.25

def test_recommend_retrain_true():
    """Should return True if any column has major drift."""
    report = {
        "col_a": {"psi": 0.05, "status": "stable"},
        "col_b": {"psi": 0.35, "status": "major"}
    }
    assert recommend_retrain(report) is True

def test_recommend_retrain_false():
    """Should return False if all columns are stable or moderate."""
    report = {
        "col_a": {"psi": 0.05, "status": "stable"},
        "col_b": {"psi": 0.15, "status": "moderate"}
    }
    assert recommend_retrain(report) is False

def test_check_drift_skips_missing_columns():
    """Should handle mismatching columns gracefully."""
    reference_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    current_df = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]})
    
    # We pass ["a", "b", "c"] to check_drift
    report = check_drift(reference_df, current_df, ["a", "b", "c"])
    
    # Only "a" should be evaluated because it's the only one present in both
    assert "a" in report
    assert "b" not in report
    assert "c" not in report
