# app.py
# Location: dashboard/app.py
#
# Main Streamlit dashboard for the Supply Chain Disruption
# Intelligence Platform.
#
# Run with: streamlit run dashboard/app.py
#
# Structure:
#   Sidebar     — navigation and filters
#   Page 1      — Executive Overview (KPIs + charts)
#   Page 2      — Supplier Risk Rankings
#   Page 3      — Alert Feed
#   Page 4      — Model Performance

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/ingestion'))
sys.path.append(os.path.dirname(__file__))

from queries import (
    get_kpi_metrics,
    get_supplier_risk_summary,
    get_delivery_status_breakdown,
    get_transport_risk_breakdown,
    get_monthly_trend,
    get_high_risk_alerts,
    get_country_risk_map_data,
    get_category_risk,
)
from components.kpi_cards import render_kpi_cards
from components.risk_table import render_risk_table
from components.alert_feed import render_alert_feed


# ── Page Config ───────────────────────────────────────────────────────────────
# Must be the very first Streamlit call

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .risk-critical { color: #e74c3c; font-weight: bold; }
    .risk-high     { color: #e67e22; font-weight: bold; }
    .risk-medium   { color: #f1c40f; font-weight: bold; }
    .risk-low      { color: #2ecc71; font-weight: bold; }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🚢")
    st.markdown("## 🚢 Supply Chain Intelligence")
    st.markdown("*Real-time disruption risk platform*")
    st.divider()

    # Page navigation
    page = st.radio(
        "Navigate",
        [
            "📊 Executive Overview",
            "🏭 Supplier Risk Rankings",
            "🚨 Alert Feed",
            "🤖 Model Performance",
        ]
    )

    st.divider()

    # Refresh button
    # st.cache_data.clear() wipes all cached query results
    # so the dashboard fetches fresh data from PostgreSQL
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Data: DataCo Supply Chain")
    st.caption("Model: XGBoost v6 | AUC: 0.7562")
    st.caption("Pipeline: Kafka → PostgreSQL → MLflow")


# ── Data Loading ──────────────────────────────────────────────────────────────
# @st.cache_data caches the query result for 5 minutes.
# Without this, every user interaction reruns ALL queries —
# very slow on 180K rows. Cache means fast reloads.

@st.cache_data(ttl=300)  # 300 seconds = 5 minutes
def load_all_data():
    """Load all dashboard data once and cache it."""
    return {
        'kpis':       get_kpi_metrics(),
        'suppliers':  get_supplier_risk_summary(),
        'status':     get_delivery_status_breakdown(),
        'transport':  get_transport_risk_breakdown(),
        'monthly':    get_monthly_trend(),
        'alerts':     get_high_risk_alerts(limit=100),
        'map_data':   get_country_risk_map_data(),
        'categories': get_category_risk(),
    }


# Load data — shows spinner while fetching
with st.spinner("Loading supply chain data..."):
    data = load_all_data()


# ── PAGE 1: Executive Overview ────────────────────────────────────────────────

if page == "📊 Executive Overview":

    st.markdown('<div class="main-header">📊 Executive Overview</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time supply chain disruption intelligence — 180,519 shipments analyzed</div>',
                unsafe_allow_html=True)

    # KPI Cards
    render_kpi_cards(data['kpis'])
    st.divider()

    # Row 1 — Delivery Status + Transport Risk
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Delivery Status Breakdown")
        status_df = data['status']
        fig = px.pie(
            status_df,
            values='count',
            names='status',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,  # donut chart
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True, height=350, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🚚 Transport Mode Risk")
        transport_df = data['transport']
        fig = px.bar(
            transport_df,
            x='transport_mode',
            y='late_rate_pct',
            color='late_rate_pct',
            color_continuous_scale='RdYlGn_r',
            text='late_rate_pct',
            labels={
                'transport_mode': 'Transport Mode',
                'late_rate_pct': 'Late Rate %'
            },
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            height=350,
            margin=dict(t=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Row 2 — Monthly Trend
    st.subheader("📈 Monthly Late Rate Trend")
    monthly_df = data['monthly']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_df['order_month'],
        y=monthly_df['late_rate_pct'],
        mode='lines+markers',
        name='Late Rate %',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Bar(
        x=monthly_df['order_month'],
        y=monthly_df['total_shipments'],
        name='Total Shipments',
        yaxis='y2',
        opacity=0.3,
        marker_color='#3498db',
    ))
    fig.update_layout(
        xaxis=dict(title='Month', tickmode='linear'),
        yaxis=dict(title='Late Rate %', side='left'),
        yaxis2=dict(
            title='Total Shipments',
            side='right',
            overlaying='y'
        ),
        legend=dict(x=0.01, y=0.99),
        height=380,
        margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Row 3 — World Map + Category Risk
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("🌍 Supplier Risk World Map")
        map_df = data['map_data']

        fig = px.choropleth(
            map_df,
            locations='origin_country',
            locationmode='country names',
            color='risk_score',
            color_continuous_scale='RdYlGn_r',
            range_color=[30, 80],
            hover_data={
                'origin_country': True,
                'risk_score': ':.1f',
                'late_rate_pct': ':.1f',
                'shipment_count': True,
            },
            labels={
                'risk_score': 'Risk Score',
                'late_rate_pct': 'Late Rate %',
                'origin_country': 'Country',
            },
        )
        fig.update_layout(
            height=400,
            margin=dict(t=10, b=0, l=0, r=0),
            coloraxis_colorbar=dict(title="Risk Score"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📦 Top Risk Categories")
        cat_df = data['categories']

        fig = px.bar(
            cat_df.head(10),
            x='late_rate_pct',
            y='product_category',
            orientation='h',
            color='late_rate_pct',
            color_continuous_scale='RdYlGn_r',
            labels={
                'late_rate_pct': 'Late Rate %',
                'product_category': 'Category'
            },
        )
        fig.update_layout(
            height=400,
            margin=dict(t=10),
            coloraxis_showscale=False,
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig, use_container_width=True)


# ── PAGE 2: Supplier Risk Rankings ────────────────────────────────────────────

elif page == "🏭 Supplier Risk Rankings":

    st.markdown('<div class="main-header">🏭 Supplier Risk Rankings</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ranked by composite risk score across all shipments</div>',
                unsafe_allow_html=True)

    render_risk_table(data['suppliers'])

    st.divider()

    # Top 15 suppliers bar chart
    st.subheader("📊 Top 15 Highest Risk Suppliers")
    top15 = data['suppliers'].head(15)

    fig = px.bar(
        top15,
        x='avg_risk_score',
        y='supplier_name',
        orientation='h',
        color='avg_risk_score',
        color_continuous_scale='RdYlGn_r',
        text='avg_risk_score',
        hover_data=['country', 'total_shipments', 'late_rate_pct'],
        labels={
            'avg_risk_score': 'Risk Score',
            'supplier_name': 'Supplier'
        },
    )
    fig.update_traces(
        texttemplate='%{text:.3f}',
        textposition='outside'
    )
    fig.update_layout(
        height=500,
        margin=dict(t=20),
        coloraxis_showscale=False,
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig, use_container_width=True)


# ── PAGE 3: Alert Feed ────────────────────────────────────────────────────────

elif page == "🚨 Alert Feed":

    st.markdown('<div class="main-header">🚨 Disruption Alert Feed</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Shipments flagged for high disruption risk</div>',
                unsafe_allow_html=True)

    render_alert_feed(data['alerts'])


# ── PAGE 4: Model Performance ─────────────────────────────────────────────────

elif page == "🤖 Model Performance":

    st.markdown('<div class="main-header">🤖 Model Performance</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">XGBoost disruption risk model — training metrics and feature analysis</div>',
                unsafe_allow_html=True)

    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AUC-ROC",   "0.7562", help="Area Under ROC Curve")
    with col2:
        st.metric("Recall",    "0.7194", help="% of late shipments correctly caught")
    with col3:
        st.metric("Precision", "0.6969", help="% of flagged shipments actually late")
    with col4:
        st.metric("F1 Score",  "0.7079", help="Harmonic mean of precision and recall")

    st.divider()

    # Feature importance chart
    st.subheader("📊 Feature Importance")

    feature_importance = {
        'mode_First Class':       0.4827,
        'mode_Standard Class':    0.2435,
        'transport_risk_score':   0.1650,
        'mode_Second Class':      0.0360,
        'mode_Same Day':          0.0130,
        'supplier_composite_risk':0.0056,
        'country_risk_score':     0.0050,
        'supplier_late_rate':     0.0049,
        'supplier_risk_index':    0.0047,
        'reliability_score':      0.0047,
        'order_dayofweek':        0.0040,
        'order_month':            0.0038,
        'category_risk_score':    0.0035,
        'quantity_log':           0.0030,
        'is_bulk_order':          0.0028,
    }

    fi_df = pd.DataFrame(
        list(feature_importance.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True)

    fig = px.bar(
        fi_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues',
        text='Importance',
    )
    fig.update_traces(
        texttemplate='%{text:.4f}',
        textposition='outside'
    )
    fig.update_layout(
        height=500,
        margin=dict(t=20),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # MLflow experiment summary
    st.subheader("🧪 MLflow Experiment Summary")
    st.markdown("All training runs tracked at **http://localhost:5000**")

    runs_data = {
        'Run': [
            'xgboost_leakage',
            'xgboost_no_leakage',
            'xgboost_tuned',
            'xgboost_high_recall',
            'xgboost_v5_balanced',
            'xgboost_final_v5 ✅',
        ],
        'AUC': [0.9810, 0.7497, 0.7557, 0.7551, 0.7562, 0.7562],
        'Recall': [0.9999, 0.5742, 0.6090, 0.8791, 0.7194, 0.7194],
        'Precision': [0.9557, 0.8271, 0.7981, 0.6036, 0.6969, 0.6969],
        'F1': [0.9773, 0.6778, 0.6908, 0.7157, 0.7079, 0.7079],
        'Note': [
            '❌ Data leakage',
            '❌ Recall too low',
            '❌ Recall still low',
            '❌ Precision too low',
            '✅ Best balance',
            '✅ Production model',
        ]
    }
    runs_df = pd.DataFrame(runs_data)
    st.dataframe(runs_df, use_container_width=True, hide_index=True)

    st.divider()

    # What the model can and can't do
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ What the Model Does Well")
        st.markdown("""
        - Catches **72% of late shipments** before they happen
        - Strong signal from transport mode (First Class = 100% late in data)
        - Consistent across 5-fold cross validation (±0.003 std)
        - Fast inference — scores 1000 shipments in milliseconds via API
        - Fully explainable — top risk factors returned per prediction
        """)
    with col2:
        st.subheader("⚠️ Known Limitations")
        st.markdown("""
        - Transport mode dominates (88% importance) — other features weak
        - Dataset limited to 2015-2017 DataCo records
        - No external signals (weather, port congestion, news)
        - AUC ceiling ~0.76 with current features
        - Adding real-time external data would push AUC toward 0.85+
        """)