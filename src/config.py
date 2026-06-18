"""
Centralized configuration for the Supply Chain ML pipeline.

All paths, feature lists, column definitions, and hyperparameter ranges
are defined here so that every script in the project shares a single
source of truth.  No more magic strings scattered across files.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "supplychain_cleaned.csv")

# Sales Regression Model
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
COLUMNS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "columns.pkl")
METRICS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "metrics.json")

# Late Delivery Classifier
CLASSIFIER_MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "classifier_model.pkl")
CLASSIFIER_COLUMNS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "classifier_columns.pkl")
CLASSIFIER_METRICS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "classifier_metrics.json")

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
TARGET_COL = "sales_log"          # Sales regression target
RAW_TARGET_COL = "sales"
CLASSIFIER_TARGET_COL = "delay_flag"  # Late delivery classifier target

# ---------------------------------------------------------------------------
# Leaky columns — mathematically derived from the sales target
# These MUST be removed before training to prevent data leakage.
#
# Rationale:
#   sales = product_price × order_item_quantity          (exact, r=1.00)
#   order_item_total ≈ sales − discount                  (r ≈ 0.99)
#   order_item_discount = sales × discount_rate           (derived from sales)
#   order_profit_per_order = revenue − cost               (post-sale)
#   order_item_profit_ratio = profit / total              (post-sale)
#   order_item_discount_rate = discount / sales           (derived from sales)
#   sales_per_customer = aggregated sales                 (derived from sales)
#   benefit_per_order = alias for profit                  (derived from sales)
# ---------------------------------------------------------------------------
LEAKY_COLUMNS = [
    "order_item_total",
    "order_item_discount",
    "order_item_discount_rate",
    "order_item_profit_ratio",
    "order_profit_per_order",
    "sales_per_customer",
    "benefit_per_order",
]

# ---------------------------------------------------------------------------
# SALES REGRESSION — Additional columns to drop to eliminate leakage
#
# sales = product_price × order_item_quantity     (r ≈ 1.00, exact formula)
# profit_log is derived from profit which is derived from sales
# ---------------------------------------------------------------------------
SALES_EXTRA_LEAKY = [
    "product_price",       # Direct multiplier: sales = price × qty
    "order_item_quantity", # Direct multiplier: sales = price × qty
    "profit_log",          # log(|profit|) where profit = sales − cost ≈ f(sales)
    "profit_negative",     # sign(profit), derived from profit → derived from sales
]

# ---------------------------------------------------------------------------
# DELIVERY CLASSIFIER — Columns to drop (post-delivery observations)
# These are NOT available at order time — using them would be leakage.
#
# Rationale:
#   days_for_shipping_(real) — actual days, only known AFTER delivery
#   delivery_days            — engineered copy of the above
#   delivery_status          — describes the delivery outcome (IS the target)
#   late_delivery_risk       — original column = (actual_days > scheduled_days)
#                              This is definitionally the same as delay_flag
#   sales_log                — log(price × qty), redundant with price/qty features
# ---------------------------------------------------------------------------
CLASSIFIER_EXTRA_DROP = [
    "days_for_shipping_(real)",   # post-delivery: actual shipping duration
    "delivery_days",              # engineered from days_for_shipping_(real)
    "delivery_status",            # source of the target variable
    "late_delivery_risk",         # derived from actual vs scheduled days
    "sales_log",                  # derived from sales (price × qty kept directly)
]

# ---------------------------------------------------------------------------
# Columns to always drop (IDs, PII, raw targets, raw dates)
# ---------------------------------------------------------------------------
DROP_COLUMNS = [
    # Raw target and derivatives
    "sales",
    "profit",
    # IDs — no predictive value
    "order_id",
    "order_item_id",
    "customer_id",
    "order_customer_id",
    "product_card_id",
    "category_id",
    "department_id",
    # PII — should never be features
    "customer_email",
    "customer_password",
    "customer_fname",
    "customer_lname",
    "customer_street",
    "latitude",
    "longitude",
    # Free-text / image — not useful for tabular ML
    "product_description",
    "product_image",
    # Raw date strings — replaced by engineered features
    "order_date",
    "shipping_date",
    "shipping_date_(dateorders)",
    "order_date_(dateorders)",
]

# ---------------------------------------------------------------------------
# High-cardinality columns to frequency-encode (instead of one-hot)
# These would explode into thousands of columns with OneHotEncoder.
# ---------------------------------------------------------------------------
HIGH_CARDINALITY_COLUMNS = [
    "product_name",
    "customer_city",
    "customer_state",
    "customer_country",
    "customer_zipcode",
    "order_city",
    "order_state",
    "order_country",
]

# ---------------------------------------------------------------------------
# Hyperparameter search space for XGBoost Regressor
# ---------------------------------------------------------------------------
PARAM_DIST = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5, 7],
    "model__subsample": [0.7, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.9, 1.0],
}

# Hyperparameter search space for XGBoost Classifier
PARAM_DIST_CLF = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5, 7],
    "model__subsample": [0.7, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.9, 1.0],
    "model__scale_pos_weight": [1, 2, 3],   # handle class imbalance
}

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3
N_ITER = 15

# ---------------------------------------------------------------------------
# Leakage detection threshold
# ---------------------------------------------------------------------------
LEAKAGE_CORRELATION_THRESHOLD = 0.95
