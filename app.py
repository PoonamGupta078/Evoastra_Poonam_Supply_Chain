import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

st.set_page_config(page_title="Supply Chain Analytics", layout="wide", page_icon="📦")
st.title("📦 Supply Chain Sales Prediction")
st.markdown("**Evoastra Ventures | XGBoost Model | R² = 0.967**")

# Change this when backend is deployed
API_URL = "http://localhost:8000/predict"

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
        st.warning("Data file not found. Please check the data/processed/ folder.")

with tab2:
    st.header("🔮 Predict Sales")
    st.markdown("Fill in the details below:")

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
        payload = {
            "Type": order_type,
            "Days for shipping (real)": days_shipping_real,
            "Days for shipment (scheduled)": days_shipping_scheduled,
            "Product Price": product_price,
            "Customer Segment": customer_segment,
            "Order Item Quantity": order_quantity,
            "Department Name": department
        }
        try:
            response = requests.get(API_URL)
            result = response.json()
            st.success(f"💰 Predicted Sales: **${result['prediction']:.2f}**")
        except Exception as e:
            st.error(f"Backend not connected yet: {e}")
            st.info("Start the FastAPI backend with: uvicorn src.inference_api:app --reload")