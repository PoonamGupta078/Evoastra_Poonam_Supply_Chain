import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from preprocess import predict_dataframe  # noqa: E402

class DummyModel:
    """Mock model to test the predict_dataframe function without loading large files."""
    def predict(self, X):
        # Return log-scaled dummy predictions (e.g., log1p(100) and log1p(200))
        return np.array([np.log1p(100), np.log1p(200)])

def test_predict_dataframe_schema_alignment():
    """Ensure predict_dataframe aligns the schema and inverses the log transform."""
    columns = ["feature1", "feature2", "feature3"]
    model = DummyModel()
    
    # Input has missing feature3, extra feature4
    df_input = pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [10, 20],
        "feature4": [100, 200],
        "sales": [0, 0]
    })
    
    # This should internally call preprocess, then reindex to `columns`, then predict
    preds = predict_dataframe(df_input, model, columns)
    
    # Length should match
    assert len(preds) == 2
    
    # Output should be in real dollars (inverse of log1p, so 100 and 200)
    assert np.isclose(preds[0], 100)
    assert np.isclose(preds[1], 200)
