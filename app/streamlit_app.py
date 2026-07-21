"""
Streamlit app for the Customer Segmentation project.

Run with:
    streamlit run app/streamlit_app.py

Features:
1. Dashboard view of all discovered segments (sizes, profiles, business actions).
2. "Score a customer" tool — enter Recency/Frequency/Monetary for a single
   customer and instantly see which segment they fall into and what action
   to take. This is the "deployable tool" layer that turns the notebook
   analysis into something a non-technical stakeholder could actually use.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "outputs")
sys.path.append(os.path.join(BASE, "notebook"))

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")


@st.cache_data
def load_outputs():
    rfm = pd.read_csv(os.path.join(OUT_DIR, "rfm_clusters.csv"))
    summary = pd.read_csv(os.path.join(OUT_DIR, "cluster_summary.csv"))
    with open(os.path.join(OUT_DIR, "business_impact.json")) as f:
        impact = json.load(f)
    with open(os.path.join(OUT_DIR, "algorithm_comparison.json")) as f:
        comparison = json.load(f)
    return rfm, summary, impact, comparison


st.title("🛍️ Customer Segmentation Dashboard")
st.caption("RFM Analysis + K-Means Clustering — from raw transactions to targeted marketing actions.")

try:
    rfm, summary, impact, comparison = load_outputs()
except FileNotFoundError:
    st.error(
        "No outputs found. Run the analysis first:\n\n"
        "```\ncd notebook && python analysis_core.py\n```"
    )
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Segment Overview", "🔍 Score a Customer", "⚙️ Methodology"])

# ---------------------------------------------------------------
# TAB 1: SEGMENT OVERVIEW
# ---------------------------------------------------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{impact['total_customers']:,}")
    col2.metric("Total Historical Revenue", f"£{impact['total_historical_revenue']:,.0f}")
    col3.metric("At-Risk Customers", f"{impact['at_risk_customer_count']:,}",
                f"{impact['at_risk_pct_of_base']}% of base")
    col4.metric("Est. Recoverable Revenue",
                f"£{impact['estimated_recoverable_revenue_low']:,.0f} – "
                f"£{impact['estimated_recoverable_revenue_high']:,.0f}")

    st.markdown("---")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Segment Sizes")
        fig = px.bar(summary.sort_values("Count", ascending=True),
                     x="Count", y="Segment", orientation="h",
                     color="Segment", text="Count")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Segment Profiles (avg RFM)")
        st.dataframe(
            summary[["Segment", "Recency", "Frequency", "Monetary", "Count"]]
            .sort_values("Count", ascending=False)
            .style.format({"Recency": "{:.1f}", "Frequency": "{:.1f}", "Monetary": "£{:.2f}"}),
            use_container_width=True, hide_index=True
        )

    st.subheader("Recommended Business Actions")
    st.dataframe(
        summary[["Segment", "Count", "Recommended_Action"]].sort_values("Count", ascending=False),
        use_container_width=True, hide_index=True
    )

    st.subheader("Customer Distribution: Recency vs Monetary Value")
    fig2 = px.scatter(rfm, x="Recency", y="Monetary", color="Segment",
                       log_y=True, opacity=0.5,
                       labels={"Recency": "Days since last purchase", "Monetary": "Lifetime spend (log scale)"})
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------
# TAB 2: SCORE A CUSTOMER
# ---------------------------------------------------------------
with tab2:
    st.subheader("Find out which segment a customer belongs to")
    st.caption("Enter a customer's RFM values to see their segment and the recommended action — "
               "this simulates how the model would be used inside a CRM tool.")

    c1, c2, c3 = st.columns(3)
    recency = c1.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=30)
    frequency = c2.number_input("Frequency (number of orders)", min_value=1, max_value=200, value=5)
    monetary = c3.number_input("Monetary (total spend, £)", min_value=0.0, max_value=100000.0, value=150.0, step=10.0)

    if st.button("Score this customer", type="primary"):
        # Nearest-centroid assignment in the same log/standardized space as training,
        # approximated here using the existing cluster means (good enough for a demo tool).
        input_rfm = pd.DataFrame({"Recency": [recency], "Frequency": [frequency], "Monetary": [monetary]})

        cluster_means = summary.set_index("Segment")[["Recency", "Frequency", "Monetary"]]
        # simple normalized distance in log space
        log_input = np.log1p(input_rfm.values[0])
        log_means = np.log1p(cluster_means.values)
        dists = np.linalg.norm(log_means - log_input, axis=1)
        best_idx = int(np.argmin(dists))
        matched_segment = cluster_means.index[best_idx]
        action = summary.set_index("Segment").loc[matched_segment, "Recommended_Action"]

        st.success(f"**Predicted Segment:** {matched_segment}")
        st.info(f"**Recommended Action:** {action}")

    st.markdown("---")
    st.caption(
        "Note: this uses a simplified nearest-centroid lookup for demo purposes. "
        "In production, you would persist the trained scaler + K-Means model (e.g. with joblib) "
        "and call `model.predict()` directly on new customer data."
    )

# ---------------------------------------------------------------
# TAB 3: METHODOLOGY
# ---------------------------------------------------------------
with tab3:
    st.subheader("How this was built")
    st.markdown("""
    1. **Data cleaning** — removed cancelled orders, guest checkouts (missing Customer ID), and invalid rows.
    2. **RFM feature engineering** — Recency, Frequency, Monetary computed per customer.
    3. **Transform & scale** — log-transform (to fix right-skew) + standardization.
    4. **Model selection** — K chosen via Elbow Method + Silhouette Score.
    5. **Clustering** — K-Means (primary model), compared against DBSCAN.
    6. **Business layer** — clusters ranked and labeled into named segments, each mapped to a marketing action.
    7. **Impact estimate** — recoverable revenue from the At-Risk segment, based on an assumed 15–20% win-back conversion rate.
    """)

    st.subheader("Algorithm Comparison")
    st.json(comparison)

    st.subheader("Business Impact Assumptions")
    st.json(impact)

    st.caption("Full analysis code: `notebook/analysis_core.py` · Full walkthrough: `notebook/customer_segmentation_analysis.ipynb`")
