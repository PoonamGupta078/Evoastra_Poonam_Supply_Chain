"""
Streamlit dashboard for Supply Chain Sales Prediction.

Two tabs:
  📊 EDA — interactive charts and summary metrics from the dataset
  🔮 Predict — user inputs → live XGBoost prediction

Fixes vs. original:
  1. EDA tab uses correct lowercase column names (was showing "N/A")
  2. Predict button is inside tab2 (was rendering outside both tabs)
  3. Prediction uses shared preprocess() function (eliminates train/serve skew)
  4. User inputs actually drive the prediction (not frozen at row 0 values)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from preprocess import preprocess, predict_dataframe
from config import DATA_PATH, MODEL_PATH, COLUMNS_PATH

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Analytics",
    layout="wide",
    page_icon="📦",
)
st.title("📦 Supply Chain Sales Prediction")
st.markdown("**End-to-End ML Pipeline | XGBoost + SHAP Explainability**")


# ── Load Model (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns


model, columns = load_model()


# ── Load Data (cached) ──────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="latin1")


# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 EDA", "🔮 Predict"])

# =====================================================================
# TAB 1 — Exploratory Data Analysis
# FIX: Original used Title Case column names ('Sales', 'Type',
# 'Customer Segment', 'Benefit per order') which don't exist in the
# cleaned CSV.  All columns are lowercase.
# =====================================================================
with tab1:
    st.header("Exploratory Data Analysis")
    try:
        df_eda = load_data()

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df_eda):,}")

        if "sales" in df_eda.columns:
            col2.metric("Avg Sales", f"${df_eda['sales'].mean():.2f}")
            col3.metric("Total Sales", f"${df_eda['sales'].sum():,.0f}")
        else:
            col2.metric("Avg Sales", "N/A")
            col3.metric("Total Sales", "N/A")

        if "order_profit_per_order" in df_eda.columns:
            col4.metric(
                "Avg Profit",
                f"${df_eda['order_profit_per_order'].mean():.2f}",
            )
        else:
            col4.metric("Avg Profit", "N/A")

        st.divider()

        # Charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if "type" in df_eda.columns and "sales" in df_eda.columns:
                fig1 = px.bar(
                    df_eda.groupby("type")["sales"].sum().reset_index(),
                    x="type",
                    y="sales",
                    title="Total Sales by Payment Type",
                    color="type",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            if "customer_segment" in df_eda.columns and "sales" in df_eda.columns:
                fig2 = px.pie(
                    df_eda.groupby("customer_segment")["sales"].sum().reset_index(),
                    names="customer_segment",
                    values="sales",
                    title="Sales by Customer Segment",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Additional charts
        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            if "delivery_status" in df_eda.columns:
                fig3 = px.pie(
                    df_eda["delivery_status"].value_counts().reset_index(),
                    names="delivery_status",
                    values="count",
                    title="Delivery Status Distribution",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                st.plotly_chart(fig3, use_container_width=True)

        with chart_col4:
            if "shipping_mode" in df_eda.columns and "sales" in df_eda.columns:
                fig4 = px.bar(
                    df_eda.groupby("shipping_mode")["sales"].mean().reset_index(),
                    x="shipping_mode",
                    y="sales",
                    title="Avg Sales by Shipping Mode",
                    color="shipping_mode",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                )
                fig4.update_layout(showlegend=False)
                st.plotly_chart(fig4, use_container_width=True)

    except FileNotFoundError:
        st.warning("Data file not found. Please ensure the CSV exists.")

# =====================================================================
# TAB 2 — Prediction
# FIX: Entire predict block is now inside `with tab2:`.
# FIX: Uses shared predict_dataframe() — same preprocessing as training.
# FIX: Builds a proper input DataFrame instead of borrowing row 0.
# =====================================================================
with tab2:
    st.header("🔮 Predict Sales")
    st.markdown("Enter order details below to get a predicted sales value.")

    col1, col2 = st.columns(2)

    with col1:
        order_type = st.selectbox(
            "Payment Type",
            ["debit", "transfer", "cash", "payment"],
        )
        days_shipping_real = st.number_input(
            "Actual Shipping Days", min_value=0, max_value=60, value=3
        )
        days_shipping_scheduled = st.number_input(
            "Scheduled Shipping Days", min_value=0, max_value=60, value=4
        )
        product_price = st.number_input(
            "Product Price ($)", min_value=0.0, max_value=5000.0, value=50.0
        )

    with col2:
        customer_segment = st.selectbox(
            "Customer Segment", ["consumer", "corporate", "home office"]
        )
        order_quantity = st.number_input(
            "Order Quantity", min_value=1, max_value=100, value=5
        )
        department = st.selectbox(
            "Department",
            ["fitness", "outdoors", "apparel", "footwear", "golf", "fan shop"],
        )
        shipping_mode = st.selectbox(
            "Shipping Mode",
            ["standard class", "second class", "first class", "same day"],
        )

    # FIX: Button is inside tab2 (was at column 0 = outside both tabs)
    if st.button("🚀 Predict Sales", use_container_width=True):
        try:
            # Build a proper input DataFrame from user inputs
            input_data = pd.DataFrame(
                [
                    {
                        "type": order_type,
                        "days_for_shipping_(real)": float(days_shipping_real),
                        "days_for_shipment_(scheduled)": float(
                            days_shipping_scheduled
                        ),
                        "product_price": float(product_price),
                        "customer_segment": customer_segment,
                        "order_item_quantity": float(order_quantity),
                        "department_name": department,
                        "shipping_mode": shipping_mode,
                        "delivery_status": (
                            "late delivery"
                            if days_shipping_real > days_shipping_scheduled
                            else "shipping on time"
                        ),
                        "late_delivery_risk": (
                            1
                            if days_shipping_real > days_shipping_scheduled
                            else 0
                        ),
                        # Sales is the target — set to 0 for inference
                        "sales": float(product_price) * float(order_quantity),
                    }
                ]
            )

            # Use the shared prediction function (same preprocessing as training)
            prediction = predict_dataframe(input_data, model, columns)
            pred_value = float(prediction[0])

            st.success(f"💰 Predicted Sales: **${pred_value:,.2f}**")

            # Show input summary
            with st.expander("📋 Input Details"):
                st.dataframe(input_data.T.rename(columns={0: "Value"}))

        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback

            st.code(traceback.format_exc())