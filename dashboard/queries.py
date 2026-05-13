# All database queries for the Streamlit dashboard.
# Centralized here so schema changes only need one fix.

import pandas as pd
import sys
import os
from sqlalchemy import create_engine

def get_engine():
    """
    Reads DB credentials from Streamlit secrets (cloud)
    or environment variables (local).
    """
    try:
        # Streamlit Cloud — reads from secrets
        import streamlit as st
        host     = st.secrets["POSTGRES_HOST"]
        port     = st.secrets["POSTGRES_PORT"]
        db       = st.secrets["POSTGRES_DB"]
        user     = st.secrets["POSTGRES_USER"]
        password = st.secrets["POSTGRES_PASSWORD"]
    except Exception:
        # Local — reads from .env file
        from dotenv import load_dotenv
        load_dotenv()
        host     = os.getenv("POSTGRES_HOST")
        port     = os.getenv("POSTGRES_PORT")
        db       = os.getenv("POSTGRES_DB")
        user     = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")

    conn_string = (
        f"postgresql://{user}:{password}@{host}:{port}/{db}"
    )
    return create_engine(conn_string, pool_pre_ping=True)

engine = get_engine()

def get_kpi_metrics():
    """
    Top-level KPI numbers shown in the summary cards.
    Returns a single-row dict of key metrics.
    """
    query = """
        SELECT
            COUNT(*)                                    AS total_shipments,
            SUM(CASE WHEN is_late = 1 THEN 1 ELSE 0 END)  AS total_late,
            ROUND(AVG(is_late::float * 100)::numeric, 1)   AS late_rate_pct,
            ROUND(AVG(reliability_score::numeric), 3)       AS avg_reliability,
            SUM(CASE WHEN transport_risk_score = 4
                     THEN 1 ELSE 0 END)                    AS first_class_count,
            SUM(CASE WHEN country_risk_score > 0.7
                     THEN 1 ELSE 0 END)                    AS high_risk_country_count
        FROM feature_store
    """
    df = pd.read_sql(query, engine)
    return df.iloc[0].to_dict()


def get_supplier_risk_summary():
    """
    Per-supplier risk summary for the risk table and map.
    Joins feature_store with suppliers for names.
    """
    query = """
        SELECT
            f.supplier_id,
            s.supplier_name,
            s.country,
            COUNT(*)                                        AS total_shipments,
            ROUND(AVG(f.is_late::float * 100)::numeric, 1) AS late_rate_pct,
            ROUND(AVG(f.reliability_score::numeric), 3)     AS avg_reliability,
            ROUND(AVG(f.supplier_composite_risk::numeric), 3) AS avg_risk_score,
            ROUND(AVG(f.country_risk_score::numeric), 3)    AS country_risk,
            MODE() WITHIN GROUP (ORDER BY f.transport_mode) AS dominant_transport
        FROM feature_store f
        LEFT JOIN suppliers s ON f.supplier_id = s.supplier_id
        GROUP BY f.supplier_id, s.supplier_name, s.country
        HAVING COUNT(*) >= 10
        ORDER BY avg_risk_score DESC
    """
    return pd.read_sql(query, engine)


def get_delivery_status_breakdown():
    """Delivery status counts for pie/bar charts."""
    query = """
        SELECT
            status,
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
        FROM shipments
        GROUP BY status
        ORDER BY count DESC
    """
    return pd.read_sql(query, engine)


def get_transport_risk_breakdown():
    """Late rate by transport mode."""
    query = """
        SELECT
            transport_mode,
            COUNT(*)                                        AS total,
            SUM(CASE WHEN is_late = 1 THEN 1 ELSE 0 END)  AS late_count,
            ROUND(AVG(f.is_late::float * 100)::numeric, 1) AS late_rate_pct,
            ROUND(AVG(transport_risk_score::numeric), 2)    AS avg_risk_score
        FROM feature_store f
        GROUP BY transport_mode
        ORDER BY late_rate_pct DESC
    """
    return pd.read_sql(query, engine)


def get_monthly_trend():
    """
    Monthly shipment volume and late rate over time.
    Used for the trend line chart.
    """
    query = """
        SELECT
            order_month,
            order_quarter,
            COUNT(*)                                        AS total_shipments,
            SUM(CASE WHEN is_late = 1 THEN 1 ELSE 0 END)  AS late_shipments,
            ROUND(AVG(f.is_late::float * 100)::numeric, 1) AS late_rate_pct,
            ROUND(AVG(reliability_score::numeric), 3)       AS avg_reliability
        FROM feature_store f
        GROUP BY order_month, order_quarter
        ORDER BY order_quarter, order_month
    """
    return pd.read_sql(query, engine)


def get_high_risk_alerts(limit=50):
    """
    Shipments with highest risk scores for the alert feed.
    """
    query = f"""
        SELECT
            f.shipment_id,
            f.supplier_id,
            sup.supplier_name,
            f.origin_country,
            f.destination_country,
            f.transport_mode,
            s.product_category,
            s.scheduled_date,
            ROUND(f.supplier_composite_risk::numeric * 100, 1) AS risk_score,
            ROUND(f.country_risk_score::numeric * 100, 1)      AS country_risk,
            f.transport_risk_score,
            f.is_late
        FROM feature_store f
        LEFT JOIN shipments s
            ON f.shipment_id = s.shipment_id
        LEFT JOIN suppliers sup
            ON f.supplier_id = sup.supplier_id
        WHERE f.supplier_composite_risk > 0.65
           OR f.transport_risk_score = 4
        ORDER BY f.supplier_composite_risk DESC
        LIMIT {limit}
    """
    return pd.read_sql(query, engine)


def get_country_risk_map_data():
    """
    Country-level aggregated risk for the world map.
    Returns one row per country with avg risk score.
    """
    query = """
        SELECT
            origin_country,
            COUNT(*)                                            AS shipment_count,
            ROUND(AVG(country_risk_score::numeric * 100), 1)   AS risk_score,
            ROUND(AVG(is_late::float * 100)::numeric, 1)       AS late_rate_pct,
            ROUND(AVG(reliability_score::numeric), 3)           AS avg_reliability
        FROM feature_store
        GROUP BY origin_country
        HAVING COUNT(*) >= 10
        ORDER BY risk_score DESC
    """
    return pd.read_sql(query, engine)


def get_category_risk():
    """Product category risk breakdown."""
    query = """
        SELECT
            s.product_category,
            COUNT(*)                                        AS total,
            ROUND(AVG(f.is_late::float * 100)::numeric, 1) AS late_rate_pct,
            ROUND(AVG(f.category_risk_score::numeric), 3)  AS risk_score
        FROM feature_store f
        LEFT JOIN shipments s ON f.shipment_id = s.shipment_id
        GROUP BY s.product_category
        ORDER BY late_rate_pct DESC
        LIMIT 15
    """
    return pd.read_sql(query, engine)

def get_demand_forecast(category: str):
    """Fetch forecast for one product category."""
    query = """
        SELECT
            forecast_date,
            predicted_qty,
            lower_bound,
            upper_bound,
            product_category
        FROM demand_forecasts
        WHERE product_category = %(category)s
        ORDER BY forecast_date
    """
    return pd.read_sql(query, engine, params={'category': category})


def get_forecast_categories():
    """Get list of categories that have forecasts."""
    query = """
        SELECT DISTINCT product_category
        FROM demand_forecasts
        ORDER BY product_category
    """
    df = pd.read_sql(query, engine)
    return df['product_category'].tolist()


def get_anomaly_alerts():
    """Fetch anomalous suppliers."""
    query = """
        SELECT
            a.supplier_id,
            s.supplier_name,
            s.country,
            a.anomaly_score,
            a.risk_score,
            a.is_anomaly,
            a.detection_date
        FROM anomaly_scores a
        LEFT JOIN suppliers s ON a.supplier_id = s.supplier_id
        WHERE a.is_anomaly = 1
        ORDER BY a.anomaly_score DESC
    """
    return pd.read_sql(query, engine)
