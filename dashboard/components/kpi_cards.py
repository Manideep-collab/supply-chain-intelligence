# kpi_cards.py
# Top summary metrics displayed as cards at the top of the dashboard.

import streamlit as st


def render_kpi_cards(metrics: dict):
    """
    Renders 4 KPI cards in a row.
    metrics dict comes from queries.get_kpi_metrics()
    """

    col1, col2, col3, col4 = st.columns(4)

    total = int(metrics.get('total_shipments', 0))
    late  = int(metrics.get('total_late', 0))
    rate  = float(metrics.get('late_rate_pct', 0))
    rel   = float(metrics.get('avg_reliability', 0))
    high_risk = int(metrics.get('high_risk_country_count', 0))
    fc_count  = int(metrics.get('first_class_count', 0))

    with col1:
        st.metric(
            label="📦 Total Shipments",
            value=f"{total:,}",
            help="Total shipments in the system"
        )

    with col2:
        st.metric(
            label="⚠️ Late Deliveries",
            value=f"{late:,}",
            delta=f"{rate}% late rate",
            delta_color="inverse",   # red when positive (more late = bad)
            help="Total late deliveries and overall late rate"
        )

    with col3:
        st.metric(
            label="🌍 High Risk Origins",
            value=f"{high_risk:,}",
            help="Supplier countries with risk score > 70%"
        )

    with col4:
        st.metric(
            label="⭐ Avg Reliability",
            value=f"{rel:.3f}",
            help="Average supplier reliability score (0=worst, 1=best)"
        )