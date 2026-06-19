"""
Supply Chain Analytics — Dual-Model Streamlit Dashboard.

Three tabs:
  📊 EDA          — Interactive charts and summary metrics
  💰 Predict Sales — Leakage-free XGBoost regression (no price × qty)
  🚚 Delivery Risk — XGBoost binary classifier (order-time features only)

Leakage-free design:
  • Sales model:      product_price and order_item_quantity removed
                      (sales = price × qty is a perfect formula, not ML)
  • Delivery model:   post-delivery observations removed
                      (actual shipping days, delivery_status, delay_flag)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from preprocess import preprocess, predict_dataframe, predict_classifier
from config import (
    DATA_PATH,
    MODEL_PATH, COLUMNS_PATH, METRICS_PATH,
    CLASSIFIER_MODEL_PATH, CLASSIFIER_COLUMNS_PATH, CLASSIFIER_METRICS_PATH,
)

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Analytics",
    layout="wide",
    page_icon="📦",
    initial_sidebar_state="collapsed",
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8e 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    .risk-high   { color: #ff4b4b; font-size: 1.6rem; font-weight: bold; }
    .risk-medium { color: #ffa500; font-size: 1.6rem; font-weight: bold; }
    .risk-low    { color: #00c851; font-size: 1.6rem; font-weight: bold; }
    div[data-testid="stTabs"] button { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("📦 Supply Chain Analytics")
st.markdown("**Dual-Model ML System | XGBoost Regression + Classification | Leakage-Free**")
st.divider()


# ── Model Loading (cached) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Sales model...")
def load_sales_model():
    model   = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return model, columns, metrics


@st.cache_resource(show_spinner="Loading Delivery Risk model...")
def load_clf_model():
    model   = joblib.load(CLASSIFIER_MODEL_PATH)
    columns = joblib.load(CLASSIFIER_COLUMNS_PATH)
    with open(CLASSIFIER_METRICS_PATH) as f:
        metrics = json.load(f)
    return model, columns, metrics


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return pd.read_csv(DATA_PATH, encoding="latin1")


# Load everything
try:
    sales_model, sales_columns, sales_metrics = load_sales_model()
except Exception as e:
    st.error(f"Failed to load sales model: {e}")
    st.stop()

try:
    clf_model, clf_columns, clf_metrics = load_clf_model()
except Exception as e:
    st.error(f"Failed to load delivery classifier: {e}")
    st.stop()

# ── Summary metrics in header ─────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Sales Model R²",     f"{sales_metrics.get('r2', 'N/A')}")
m2.metric("Sales RMSE ($)",     f"${sales_metrics.get('rmse_dollars', 'N/A')}")
m3.metric("Delivery AUC-ROC",   f"{clf_metrics.get('auc_roc', 'N/A')}")
m4.metric("Delivery F1 Score",  f"{clf_metrics.get('f1_score', 'N/A')}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 EDA", "💰 Predict Sales", "🚚 Delivery Risk"])


# =====================================================================
# TAB 1 — EDA
# =====================================================================
with tab1:
    st.header("Exploratory Data Analysis")
    try:
        df_eda = load_data()
        df_eda.columns = df_eda.columns.str.lower().str.replace(" ", "_")

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Total Records", f"{len(df_eda):,}")
        r1c2.metric("Avg Sales ($)", f"${df_eda['sales'].mean():.2f}" if "sales" in df_eda.columns else "N/A")
        r1c3.metric("Total Sales ($)", f"${df_eda['sales'].sum():,.0f}" if "sales" in df_eda.columns else "N/A")
        late_pct = (df_eda["delivery_status"].str.lower() == "late delivery").mean() * 100 if "delivery_status" in df_eda.columns else 0
        r1c4.metric("Late Delivery Rate", f"{late_pct:.1f}%")

        st.divider()
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if "type" in df_eda.columns and "sales" in df_eda.columns:
                fig1 = px.bar(
                    df_eda.groupby("type")["sales"].sum().reset_index(),
                    x="type", y="sales", title="Total Sales by Payment Type",
                    color="type", color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig1.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            if "customer_segment" in df_eda.columns and "sales" in df_eda.columns:
                fig2 = px.pie(
                    df_eda.groupby("customer_segment")["sales"].sum().reset_index(),
                    names="customer_segment", values="sales",
                    title="Sales by Customer Segment",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                st.plotly_chart(fig2, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            if "delivery_status" in df_eda.columns:
                fig3 = px.pie(
                    df_eda["delivery_status"].value_counts().reset_index(),
                    names="delivery_status", values="count",
                    title="Delivery Status Distribution",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                st.plotly_chart(fig3, use_container_width=True)

        with chart_col4:
            if "shipping_mode" in df_eda.columns and "sales" in df_eda.columns:
                fig4 = px.bar(
                    df_eda.groupby("shipping_mode")["sales"].mean().reset_index(),
                    x="shipping_mode", y="sales", title="Avg Sales by Shipping Mode",
                    color="shipping_mode",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                )
                fig4.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig4, use_container_width=True)

        # Bonus: sales distribution
        if "sales" in df_eda.columns:
            st.subheader("Sales Distribution")
            fig5 = px.histogram(
                df_eda[df_eda["sales"] < df_eda["sales"].quantile(0.99)],
                x="sales", nbins=80,
                title="Sales Distribution (excluding top 1% outliers)",
                color_discrete_sequence=["#2d5a8e"],
            )
            fig5.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig5, use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ Data file not found. Please ensure `data/supplychain_cleaned.csv` exists.")
    except Exception as e:
        st.error(f"EDA error: {e}")


# =====================================================================
# TAB 2 — SALES PREDICTION (Regression)
# =====================================================================
with tab2:
    st.header("💰 Predict Sales")
    st.markdown(
        "Predicts **sales revenue** based on order profile. "
        "**Product price and quantity are intentionally excluded** — they would give "
        "a trivially perfect R² (sales = price × qty). The model learns real business patterns."
    )

    with st.expander("ℹ️ Model Details", expanded=False):
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("R² Score",    str(sales_metrics.get("r2", "N/A")))
        mc2.metric("RMSE ($)",    f"${sales_metrics.get('rmse_dollars', 'N/A')}")
        mc3.metric("Features",    str(sales_metrics.get("n_features", "N/A")))
        st.caption("Lower R² than a leaky model, but genuinely predictive of business patterns.")

    st.divider()

    sc1, sc2 = st.columns(2)

    with sc1:
        s_payment_type = st.selectbox(
            "Payment Type", ["debit", "transfer", "cash", "payment"],
            key="s_payment_type",
        )
        s_customer_seg = st.selectbox(
            "Customer Segment", ["consumer", "corporate", "home office"],
            key="s_customer_seg",
        )
        s_shipping_mode = st.selectbox(
            "Shipping Mode", ["standard class", "second class", "first class", "same day"],
            key="s_shipping_mode",
        )
        s_market = st.selectbox(
            "Market", ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"],
            key="s_market",
        )

    with sc2:
        s_department = st.selectbox(
            "Department",
            ["fitness", "outdoors", "apparel", "footwear", "golf", "fan shop",
             "technology", "pet shop", "health and beauty"],
            key="s_department",
        )
        s_category = st.selectbox(
            "Category",
            ["cleats", "men's footwear", "women's apparel", "indoor/outdoor games",
             "cardio equipment", "sporting goods", "golf bags & carts", "fishing",
             "camping & hiking", "electronics", "water sports"],
            key="s_category",
        )
        s_order_region = st.selectbox(
            "Order Region",
            ["West of USA", "East of USA", "Central America", "South America",
             "Western Europe", "Southern Europe", "Eastern Asia", "South Asia",
             "Southeast Asia", "Oceania", "Eastern Africa", "West Africa", "Caribbean"],
            key="s_order_region",
        )
        s_scheduled_days = st.number_input(
            "Scheduled Shipping Days", min_value=1, max_value=30, value=4,
            key="s_scheduled_days",
        )
        s_product_price = st.number_input(
            "Product Price ($)", min_value=0.0, value=50.0,
            key="s_product_price",
        )
        s_order_qty = st.number_input(
            "Order Quantity", min_value=1, max_value=100, value=5,
            key="s_order_qty",
        )

    if st.button("💰 Predict Sales Revenue", use_container_width=True, key="btn_sales"):
        try:
            input_data = pd.DataFrame([{
                "type": s_payment_type.lower(),
                "days_for_shipment_(scheduled)": float(s_scheduled_days),
                "customer_segment": s_customer_seg.lower(),
                "department_name": s_department.lower(),
                "category_name": s_category.lower(),
                "shipping_mode": s_shipping_mode.lower(),
                "market": s_market.lower(),
                "order_region": s_order_region.lower(),
                "product_price": float(s_product_price),
                "order_item_quantity": float(s_order_qty),
                "delivery_status": "shipping on time",
                "late_delivery_risk": 0,
                "sales": float(s_product_price) * float(s_order_qty),
            }])

            prediction = predict_dataframe(input_data, sales_model, sales_columns)
            pred_value = float(prediction[0])

            st.success(f"💰 Predicted Sales: **${pred_value:,.2f}**")
            st.caption(
                "This prediction is based on customer profile, department, region and shipping mode — "
                "NOT on price × quantity. It reflects genuine business patterns."
            )

            with st.expander("📋 Input Summary"):
                st.dataframe(input_data.T.rename(columns={0: "Value"}))

            # SHAP explanation
            with st.expander("🔍 SHAP Feature Importance (why this prediction?)"):
                try:
                    import shap
                    import matplotlib.pyplot as plt

                    preprocessor_step = sales_model.named_steps["preprocessor"]
                    xgb_step = sales_model.named_steps["model"]

                    # Preprocess the input for SHAP
                    from preprocess import preprocess as _preprocess
                    from config import SALES_EXTRA_LEAKY
                    df_shap = _preprocess(input_data.copy(), extra_drop_cols=SALES_EXTRA_LEAKY)
                    for col in sales_columns:
                        if col not in df_shap.columns:
                            df_shap[col] = 0.0
                    df_shap = df_shap[sales_columns]
                    X_transformed = preprocessor_step.transform(df_shap)
                    if hasattr(X_transformed, "toarray"):
                        X_transformed = X_transformed.toarray()
                    X_transformed = X_transformed.astype(float)

                    explainer = shap.TreeExplainer(xgb_step)
                    shap_explanation = explainer(X_transformed)
                    shap_vals = shap_explanation.values

                    fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
                    feat_names = preprocessor_step.get_feature_names_out()
                    shap.summary_plot(
                        shap_vals, X_transformed,
                        feature_names=feat_names,
                        max_display=15, show=False, plot_type="dot",
                    )
                    st.pyplot(plt.gcf(), bbox_inches="tight")
                    plt.close("all")
                except Exception as shap_err:
                    st.info(f"SHAP unavailable: {shap_err}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())


# =====================================================================
# TAB 3 — DELIVERY RISK (Classification)
# =====================================================================
with tab3:
    st.header("🚚 Late Delivery Risk Prediction")
    st.markdown(
        "Predicts the **probability of a late delivery** using only information available "
        "**at order time**. Post-delivery observations (actual shipping days, delivery status) "
        "are excluded to prevent leakage."
    )

    with st.expander("ℹ️ Model Details", expanded=False):
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("AUC-ROC",    str(clf_metrics.get("auc_roc", "N/A")))
        dc2.metric("F1 Score",   str(clf_metrics.get("f1_score", "N/A")))
        dc3.metric("Precision",  str(clf_metrics.get("precision", "N/A")))
        dc4.metric("Recall",     str(clf_metrics.get("recall", "N/A")))
        st.caption(
            "**Recall** is most important in supply chain: we want to catch as many "
            "late deliveries as possible so ops teams can intervene."
        )

    st.divider()

    dc1, dc2 = st.columns(2)

    with dc1:
        d_payment_type = st.selectbox(
            "Payment Type", ["debit", "transfer", "cash", "payment"],
            key="d_payment_type",
        )
        d_customer_seg = st.selectbox(
            "Customer Segment", ["consumer", "corporate", "home office"],
            key="d_customer_seg",
        )
        d_shipping_mode = st.selectbox(
            "Shipping Mode", ["standard class", "second class", "first class", "same day"],
            key="d_shipping_mode",
        )
        d_market = st.selectbox(
            "Market", ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"],
            key="d_market",
        )
        d_product_price = st.number_input(
            "Product Price ($)", min_value=0.0, value=50.0,
            key="d_product_price",
        )

    with dc2:
        d_department = st.selectbox(
            "Department",
            ["fitness", "outdoors", "apparel", "footwear", "golf", "fan shop",
             "technology", "pet shop", "health and beauty"],
            key="d_department",
        )
        d_category = st.selectbox(
            "Category",
            ["cleats", "men's footwear", "women's apparel", "indoor/outdoor games",
             "cardio equipment", "sporting goods", "golf bags & carts", "fishing",
             "camping & hiking", "electronics", "water sports"],
            key="d_category",
        )
        d_order_region = st.selectbox(
            "Order Region",
            ["West of USA", "East of USA", "Central America", "South America",
             "Western Europe", "Southern Europe", "Eastern Asia", "South Asia",
             "Southeast Asia", "Oceania", "Eastern Africa", "West Africa", "Caribbean"],
            key="d_order_region",
        )
        d_scheduled_days = st.number_input(
            "Scheduled Shipping Days", min_value=1, max_value=30, value=4,
            key="d_scheduled_days",
        )
        d_order_qty = st.number_input(
            "Order Quantity", min_value=1, max_value=100, value=5,
            key="d_order_qty",
        )

    if st.button("🚚 Predict Delivery Risk", use_container_width=True, key="btn_delivery"):
        try:
            input_data = pd.DataFrame([{
                "type": d_payment_type.lower(),
                "days_for_shipment_(scheduled)": float(d_scheduled_days),
                "customer_segment": d_customer_seg.lower(),
                "department_name": d_department.lower(),
                "category_name": d_category.lower(),
                "shipping_mode": d_shipping_mode.lower(),
                "market": d_market.lower(),
                "order_region": d_order_region.lower(),
                "product_price": float(d_product_price),
                "order_item_quantity": float(d_order_qty),
                # No delivery_status, no actual shipping days — order-time only
                "sales": float(d_product_price) * float(d_order_qty),
            }])

            binary_pred, probability = predict_classifier(input_data, clf_model, clf_columns)
            risk_pct = float(probability[0]) * 100
            is_late = bool(binary_pred[0])

            # Risk gauge
            gauge_color = (
                "#ff4b4b" if risk_pct >= 60 else
                "#ffa500" if risk_pct >= 35 else
                "#00c851"
            )
            risk_label = (
                "🔴 HIGH RISK" if risk_pct >= 60 else
                "🟡 MEDIUM RISK" if risk_pct >= 35 else
                "🟢 LOW RISK"
            )

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                number={"suffix": "%", "font": {"size": 40, "color": gauge_color}},
                title={"text": "Late Delivery Probability", "font": {"size": 18}},
                delta={"reference": 50, "increasing": {"color": "#ff4b4b"},
                       "decreasing": {"color": "#00c851"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": gauge_color, "thickness": 0.3},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "steps": [
                        {"range": [0, 35],  "color": "#e8f5e9"},
                        {"range": [35, 60], "color": "#fff3e0"},
                        {"range": [60, 100], "color": "#ffebee"},
                    ],
                    "threshold": {
                        "line": {"color": "#333", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            gauge_fig.update_layout(
                height=300,
                margin=dict(t=60, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
            )

            g_col, v_col = st.columns([2, 1])
            with g_col:
                st.plotly_chart(gauge_fig, use_container_width=True)
            with v_col:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:1.5rem; font-weight:bold;'>{risk_label}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"**Prediction:** {'🕐 Late Delivery' if is_late else '✅ On Time'}")
                st.markdown(f"**Confidence:** {risk_pct:.1f}% chance of delay")
                if is_late:
                    st.warning("⚡ Recommend proactive intervention: contact carrier, adjust routing.")
                else:
                    st.success("✅ Delivery expected on time based on order profile.")

            with st.expander("📋 Input Summary"):
                st.dataframe(input_data.T.rename(columns={0: "Value"}))

            # SHAP explanation
            with st.expander("🔍 SHAP Feature Importance (why this risk score?)"):
                try:
                    import shap
                    import matplotlib.pyplot as plt

                    preprocessor_step = clf_model.named_steps["preprocessor"]
                    xgb_step = clf_model.named_steps["model"]

                    from preprocess import preprocess as _preprocess
                    from config import CLASSIFIER_EXTRA_DROP
                    df_shap = _preprocess(input_data.copy(), extra_drop_cols=CLASSIFIER_EXTRA_DROP)

                    # Remove delay_flag if it's a placeholder (not in clf_columns)
                    if "delay_flag" in df_shap.columns and "delay_flag" not in clf_columns:
                        df_shap = df_shap.drop(columns=["delay_flag"])

                    for col in clf_columns:
                        if col not in df_shap.columns:
                            df_shap[col] = 0.0
                    df_shap = df_shap[clf_columns]

                    X_transformed = preprocessor_step.transform(df_shap)
                    if hasattr(X_transformed, "toarray"):
                        X_transformed = X_transformed.toarray()
                    X_transformed = X_transformed.astype(float)
                    explainer = shap.TreeExplainer(xgb_step)
                    shap_explanation = explainer(X_transformed)
                    shap_vals = shap_explanation.values

                    fig_shap, _ = plt.subplots(figsize=(8, 4))
                    feat_names = preprocessor_step.get_feature_names_out()
                    shap.summary_plot(
                        shap_vals, X_transformed,
                        feature_names=feat_names,
                        max_display=15, show=False, plot_type="dot",
                    )
                    st.pyplot(plt.gcf(), bbox_inches="tight")
                    plt.close("all")
                except Exception as shap_err:
                    st.info(f"SHAP unavailable: {shap_err}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())