# alert_feed.py
# High risk shipment alert feed.

import streamlit as st
import pandas as pd


def render_alert_feed(df: pd.DataFrame):
    """
    Renders the alert feed showing highest-risk shipments.
    df comes from queries.get_high_risk_alerts()
    """
    st.subheader("🚨 High Risk Alert Feed")

    if df.empty:
        st.success("✅ No high risk shipments detected")
        return

    # Summary line
    critical = len(df[df['risk_score'] >= 75])
    high     = len(df[df['risk_score'] >= 55])

    col1, col2 = st.columns(2)
    with col1:
        st.error(f"🔴 {critical} CRITICAL risk shipments")
    with col2:
        st.warning(f"🟠 {high} HIGH risk shipments")

    st.divider()

    # Render each alert as an expander card
    for _, row in df.head(20).iterrows():
        score = float(row['risk_score'])

        # Color code by severity
        if score >= 75:
            icon   = "🔴"
            status = "CRITICAL"
        elif score >= 55:
            icon   = "🟠"
            status = "HIGH"
        else:
            icon   = "🟡"
            status = "MEDIUM"

        with st.expander(
            f"{icon} [{status}] Shipment {row['shipment_id']} — "
            f"{row['origin_country']} → {row['destination_country']} "
            f"| Risk: {score:.1f}/100"
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Supplier**")
                st.write(row.get('supplier_name', row['supplier_id']))
                st.write("**Origin**")
                st.write(row['origin_country'])
                st.write("**Destination**")
                st.write(row['destination_country'])

            with col2:
                st.write("**Transport Mode**")
                st.write(row['transport_mode'])
                st.write("**Product**")
                st.write(row.get('product_category', 'N/A'))
                st.write("**Scheduled Date**")
                st.write(str(row.get('scheduled_date', 'N/A')))

            with col3:
                st.write("**Risk Score**")
                st.metric("", f"{score:.1f}/100")
                st.write("**Country Risk**")
                st.metric("", f"{float(row['country_risk']):.1f}/100")
                st.write("**Actual Outcome**")
                st.write("🔴 Late" if row['is_late'] == 1 else "🟢 On Time")