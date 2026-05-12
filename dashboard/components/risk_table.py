# risk_table.py
# Supplier risk ranking table with color coding.

import streamlit as st
import pandas as pd


def risk_color(score: float) -> str:
    """Return color based on risk score."""
    if score >= 0.75:
        return "🔴"
    elif score >= 0.55:
        return "🟠"
    elif score >= 0.35:
        return "🟡"
    else:
        return "🟢"


def render_risk_table(df: pd.DataFrame):
    """
    Renders the supplier risk ranking table.
    df comes from queries.get_supplier_risk_summary()
    """
    st.subheader("🏭 Supplier Risk Rankings")

    # Search filter
    search = st.text_input(
        "🔍 Search supplier or country",
        placeholder="e.g. India, Indonesia..."
    )
    if search:
        mask = (
            df['supplier_name'].str.contains(search, case=False, na=False) |
            df['country'].str.contains(search, case=False, na=False)
        )
        df = df[mask]

    # Risk filter
    st.caption("🔺 Move slider RIGHT to show only higher-risk suppliers")
    risk_filter = st.select_slider(
        "Show suppliers with risk score above:",
        options=[0.0, 0.25, 0.35, 0.50, 0.65, 0.75],
        value=0.0,
        format_func=lambda x: {
            0.0:  "0.0 — All suppliers",
            0.25: "0.25 — Medium+",
            0.35: "0.35 — Above average",
            0.50: "0.50 — High risk only",
            0.65: "0.65 — Very high risk",
            0.75: "0.75 — Critical only",
        }[x]
    )
    df = df[df['avg_risk_score'] >= risk_filter]

    # Show count feedback
    total_suppliers = len(df)
    st.caption(f"Showing **{total_suppliers}** suppliers with risk score ≥ {risk_filter}")

    # Add emoji indicator column
    df = df.copy()
    df['Risk'] = df['avg_risk_score'].apply(risk_color)

    # Display columns
    display_cols = {
        'Risk':               'Risk',
        'supplier_name':      'Supplier',
        'country':            'Country',
        'total_shipments':    'Shipments',
        'late_rate_pct':      'Late Rate %',
        'avg_reliability':    'Reliability',
        'avg_risk_score':     'Risk Score',
        'dominant_transport': 'Main Transport',
    }

    display_df = df[list(display_cols.keys())].rename(columns=display_cols)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Late Rate %": st.column_config.ProgressColumn(
                "Late Rate %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0,
                max_value=1,
                format="%.3f",
            ),
        }
    )

    st.caption(f"Showing {len(display_df):,} suppliers")