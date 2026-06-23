"""
Data drift detection using Population Stability Index (PSI).

Monitors distributional shifts in numeric features between a reference
dataset (training) and a current dataset (production inference).
"""

import os
import sys
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    DATA_PATH,
    REFERENCE_DATA_PATH,
    DRIFT_REPORT_PATH,
)


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) for a single numeric feature.
    
    Args:
        reference: pd.Series of baseline data.
        current: pd.Series of new data.
        bins: Number of quantile bins to use.
        
    Returns:
        PSI value (float).
    """
    # Create bin boundaries based on reference quantiles
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.nanquantile(reference, quantiles)
    
    # Ensure bin edges are unique and sorted
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0  # Degenerate feature (all same value)
        
    # Expand slightly to capture all boundaries
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # Calculate percentages in each bin
    ref_counts = pd.cut(reference, bins=bin_edges).value_counts(sort=False).values
    cur_counts = pd.cut(current, bins=bin_edges).value_counts(sort=False).values
    
    ref_pct = ref_counts / np.sum(ref_counts)
    cur_pct = cur_counts / np.sum(cur_counts)
    
    # Handle empty bins to avoid log(0)
    ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
    cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
    
    # PSI Formula
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def check_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, numeric_cols: list) -> dict:
    """
    Computes PSI for all numeric columns.
    
    Returns dict: {col: {"psi": float, "status": "stable"/"moderate"/"major"}}
    """
    report = {}
    
    for col in numeric_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
            
        ref_series = reference_df[col].dropna()
        cur_series = current_df[col].dropna()
        
        if len(ref_series) == 0 or len(cur_series) == 0:
            continue
            
        psi_value = compute_psi(ref_series, cur_series)
        
        if psi_value < 0.1:
            status = "stable"
        elif psi_value < 0.25:
            status = "moderate"
        else:
            status = "major"
            
        report[col] = {
            "psi": round(psi_value, 4),
            "status": status
        }
        
    return report


def recommend_retrain(drift_report: dict) -> bool:
    """Returns True if ANY column has status == 'major'."""
    return any(details["status"] == "major" for details in drift_report.values())


def save_drift_report(drift_report: dict, recommend: bool) -> None:
    """Saves drift results to JSON."""
    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    
    output = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "recommend_retrain": recommend,
        "features": drift_report
    }
    
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    print("=" * 60)
    print("  Data Drift Monitoring (PSI)")
    print("=" * 60)
    
    if not os.path.exists(REFERENCE_DATA_PATH):
        raise FileNotFoundError(
            f"Reference data not found at {REFERENCE_DATA_PATH}. Train a model first."
        )
        
    reference_data = joblib.load(REFERENCE_DATA_PATH)
    print(f"[*] Loaded reference data: {reference_data.shape}")
    
    current_data = pd.read_csv(DATA_PATH, encoding="latin1")
    current_data.columns = current_data.columns.str.lower().str.replace(" ", "_")
    print(f"[*] Loaded current data: {current_data.shape}")
    
    numeric_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()
    
    print("[*] Computing PSI...")
    drift_report = check_drift(reference_data, current_data, numeric_cols)
    retrain_needed = recommend_retrain(drift_report)
    
    save_drift_report(drift_report, retrain_needed)
    
    print(f"\n{'-' * 40}")
    print("  [DRIFT REPORT]")
    print(f"{'-' * 40}")
    
    stable_count = 0
    moderate_count = 0
    major_count = 0
    
    for col, details in drift_report.items():
        st = details["status"]
        if st == "major":
            print(f"  🔴 MAJOR: {col} (PSI: {details['psi']})")
            major_count += 1
        elif st == "moderate":
            print(f"  🟡 MODERATE: {col} (PSI: {details['psi']})")
            moderate_count += 1
        else:
            stable_count += 1
            
    print(f"\n  Stable: {stable_count} | Moderate: {moderate_count} | Major: {major_count}")
    
    print(f"\n[*] Report saved to {DRIFT_REPORT_PATH}")
    print(f"{'=' * 60}\n")
    
    if retrain_needed:
        print(">>> WARNING: MAJOR DRIFT DETECTED. RETRAINING RECOMMENDED. <<<")
        sys.exit(1)
    else:
        print(">>> NO MAJOR DRIFT. DATA IS STABLE. <<<")
        sys.exit(0)
