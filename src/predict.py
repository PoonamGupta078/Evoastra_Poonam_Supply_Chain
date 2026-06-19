import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess  # noqa: E402
from config import MODEL_PATH, COLUMNS_PATH, SALES_EXTRA_LEAKY  # noqa: E402


def predict_from_csv(path):
    """
    Load data from a CSV, preprocess it, align features, and predict sales.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found at: {path}")

    # Load artifacts
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        raise FileNotFoundError(
            "Model artifacts not found. Please run 'python src/train.py' first."
        )

    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)

    # 1. LOAD DATA
    df = pd.read_csv(path, encoding="latin1")

    # 2. PREPROCESS (with sales model specific leakage drop)
    df = preprocess(df, is_training=False, extra_drop_cols=SALES_EXTRA_LEAKY)

    # 3. ALIGN COLUMNS
    df = df.reindex(columns=columns, fill_value=0)

    # 4. PREDICT (LOG SCALE)
    preds_log = model.predict(df)

    # 5. CONVERT BACK TO REAL SALES
    preds = np.expm1(preds_log)

    return preds


if __name__ == "__main__":
    # Check if a file was provided, otherwise default to cleaned data path
    from config import DATA_PATH
    test_path = DATA_PATH if os.path.exists(DATA_PATH) else "DataCoSupplyChain.csv"

    print(f"[*] Running predictions on: {test_path}")
    try:
        preds = predict_from_csv(test_path)
        print("\nSample Predictions (Real Sales):")
        print(preds[:5])
    except Exception as e:
        print(f"Error during prediction: {e}")