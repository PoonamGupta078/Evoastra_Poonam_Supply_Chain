import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Supply Chain Analytics", layout="wide", page_icon="📦")
st.title("📦 Supply Chain Sales Prediction")
st.markdown("**Evoastra Ventures | XGBoost Model | R² = 0.967**")

@st.cache_resource
def load_model():
    model = joblib.load("artifacts/model.pkl")
    columns = joblib.load("artifacts/columns.pkl")
    return model, columns

model, columns = load_model()

tab1, tab2 = st.tabs(["📊 EDA", "🔮 Predict"])

with tab1:
    st.header("Exploratory Data Analysis")
    try:
        df = pd.read_csv("data/supplychain_cleaned.csv")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Avg Sales", f"${df['Sales'].mean():.2f}" if 'Sales' in df.columns else "N/A")
        col3.metric("Avg Profit", f"${df['Benefit per order'].mean():.2f}" if 'Benefit per order' in df.columns else "N/A")
        if 'Type' in df.columns and 'Sales' in df.columns:
            fig = px.bar(df, x='Type', y='Sales', title='Sales by Order Type', color='Type')
            st.plotly_chart(fig, use_container_width=True)
        if 'Customer Segment' in df.columns and 'Sales' in df.columns:
            fig2 = px.pie(df, names='Customer Segment', values='Sales', title='Sales by Customer Segment')
            st.plotly_chart(fig2, use_container_width=True)
    except FileNotFoundError:
        st.warning("Data file not found.")

with tab2:
    st.header("🔮 Predict Sales")
    col1, col2 = st.columns(2)
    with col1:
        order_type = st.selectbox("Order Type", ["DEBIT", "TRANSFER", "CASH", "PAYMENT"])
        days_shipping_real = st.number_input("Actual Shipping Days", min_value=0, max_value=60, value=3)
        days_shipping_scheduled = st.number_input("Scheduled Shipping Days", min_value=0, max_value=60, value=4)
        product_price = st.number_input("Product Price ($)", min_value=0.0, value=50.0)
    with col2:
        customer_segment = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])
        order_quantity = st.number_input("Order Quantity", min_value=1, max_value=100, value=5)
        department = st.selectbox("Department", ["Fitness", "Outdoors", "Apparel", "Electronics", "Golf", "Fan Shop"])

if st.button("🚀 Predict", use_container_width=True):
        try:
            sample = pd.DataFrame([{col: 0 for col in columns}])
            sample["type"] = order_type.lower()
            sample["days_for_shipping_(real)"] = float(days_shipping_real)
            sample["days_for_shipment_(scheduled)"] = float(days_shipping_scheduled)
            sample["product_price"] = float(product_price)
            sample["customer_segment"] = customer_segment.lower()
            sample["order_item_quantity"] = float(order_quantity)
            sample["department_name"] = department.lower()
            pred_log = model.predict(sample)[0]
            prediction = float(np.expm1(pred_log))
            st.success(f"💰 Predicted Sales: **${prediction:.2f}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")