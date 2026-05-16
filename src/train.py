import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from preprocess import preprocess

# Load data
df = pd.read_csv("data/supplychain_cleaned.csv", encoding="latin1")
# Preprocess
df = preprocess(df)

target = "sales_log"

X = df.drop(columns=[target])
y = df[target]

# Column split
cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns

# Preprocessor
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols),
    ("num", "passthrough", num_cols)
])

# Model
xgb = XGBRegressor(random_state=42)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgb)
])

# Param space (REAL one)
param_dist = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5, 7],
    "model__subsample": [0.7, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.9, 1.0]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=15,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

search.fit(X_train, y_train)

best_model = search.best_estimator_

# Save EVERYTHING
joblib.dump(best_model, "artifacts/model.pkl")
joblib.dump(X.columns.tolist(), "artifacts/columns.pkl")

print("Training complete")