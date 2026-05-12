# Uses Isolation Forest to detect anomalous supplier behavior.
#
# What counts as anomalous:
#   - Supplier risk score significantly above their own historical average
#   - Unusual combination of features (high transport risk + low reliability
#     + high country risk simultaneously)
#   - Patterns that look nothing like the majority of suppliers
#
# Isolation Forest works by:
#   1. Randomly selecting a feature and a split value
#   2. Splitting the data
#   3. Repeating until each point is isolated
#   4. Anomalies get isolated in very few splits (short path length)
#   5. Normal points need many splits (long path length)
#
# Output: anomaly_score between -1 and 1
#   Negative = more anomalous
#   The model labels points as -1 (anomaly) or 1 (normal)

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from datetime import date
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../ingestion'))
from db import get_engine


# ── Configuration ─────────────────────────────────────────────────────────────

# Features used for anomaly detection
# These are supplier-level aggregated features —
# we detect anomalies at the supplier level, not per shipment
ANOMALY_FEATURES = [
    'avg_composite_risk',
    'avg_reliability',
    'avg_country_risk',
    'avg_transport_risk',
    'late_rate',
    'total_shipments',
    'avg_category_risk',
]

# contamination: expected proportion of anomalies in the data
# 0.05 = we expect ~5% of suppliers to be genuinely anomalous
# This affects how aggressively the model flags anomalies
CONTAMINATION = 0.05


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_supplier_features(engine) -> pd.DataFrame:
    """
    Aggregate feature_store to supplier level.
    Anomaly detection works at supplier level —
    we want to flag suppliers whose overall profile is unusual,
    not individual shipments.
    """
    query = """
        SELECT
            f.supplier_id,
            s.supplier_name,
            s.country,
            COUNT(*)                                            AS total_shipments,
            ROUND(AVG(f.supplier_composite_risk)::numeric, 4)  AS avg_composite_risk,
            ROUND(AVG(f.reliability_score)::numeric, 4)        AS avg_reliability,
            ROUND(AVG(f.country_risk_score)::numeric, 4)       AS avg_country_risk,
            ROUND(AVG(f.transport_risk_score)::numeric, 4)     AS avg_transport_risk,
            ROUND(AVG(f.is_late::float)::numeric, 4)           AS late_rate,
            ROUND(AVG(f.category_risk_score)::numeric, 4)      AS avg_category_risk
        FROM feature_store f
        LEFT JOIN suppliers s ON f.supplier_id = s.supplier_id
        GROUP BY f.supplier_id, s.supplier_name, s.country
        HAVING COUNT(*) >= 10
        ORDER BY avg_composite_risk DESC
    """
    df = pd.read_sql(query, engine)
    print(f"✅ Loaded {len(df)} supplier profiles for anomaly detection")
    return df


# ── Anomaly Detection ─────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Isolation Forest on supplier-level features.

    Steps:
    1. Extract numerical features
    2. Scale them (StandardScaler makes all features comparable)
    3. Fit Isolation Forest
    4. Get anomaly labels (-1 = anomaly, 1 = normal)
    5. Get anomaly scores (more negative = more anomalous)
    """

    # Extract feature matrix
    X = df[ANOMALY_FEATURES].fillna(0)

    # Scale features — important for Isolation Forest
    # Without scaling, 'total_shipments' (range: 10-24840) would
    # dominate over 'late_rate' (range: 0-1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=200,       # number of trees — more = more stable
        contamination=CONTAMINATION,
        random_state=42,
        max_samples='auto',     # use all samples for each tree
    )

    # fit_predict returns -1 (anomaly) or 1 (normal) for each row
    labels = iso_forest.fit_predict(X_scaled)

    # decision_function returns raw anomaly scores
    # More negative = more isolated = more anomalous
    # We normalize to 0-1 range for easier interpretation
    raw_scores = iso_forest.decision_function(X_scaled)

    # Normalize: flip sign so higher = more anomalous, scale to 0-1
    normalized_scores = 1 - (
        (raw_scores - raw_scores.min()) /
        (raw_scores.max() - raw_scores.min())
    )

    # Add results back to dataframe
    df = df.copy()
    df['is_anomaly']    = (labels == -1).astype(int)
    df['anomaly_score'] = normalized_scores.round(4)

    anomaly_count = df['is_anomaly'].sum()
    print(f"✅ Anomaly detection complete")
    print(f"   Total suppliers analyzed : {len(df)}")
    print(f"   Anomalies detected       : {anomaly_count} "
          f"({anomaly_count/len(df)*100:.1f}%)")

    # Print the anomalous suppliers
    anomalies = df[df['is_anomaly'] == 1].sort_values(
        'anomaly_score', ascending=False
    )
    print(f"\n🚨 Anomalous Suppliers:")
    for _, row in anomalies.iterrows():
        print(f"   {row['supplier_name']:<35} "
              f"score={row['anomaly_score']:.3f} | "
              f"late_rate={row['late_rate']:.1%} | "
              f"risk={row['avg_composite_risk']:.3f}")

    return df


# ── Save Results ──────────────────────────────────────────────────────────────

def save_anomalies(df: pd.DataFrame, engine):
    """
    Save anomaly detection results to PostgreSQL.
    Only saves anomalous suppliers — normal ones don't need alerting.
    """
    today = date.today()

    # Clear today's results to avoid duplicates on re-run
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM anomaly_scores WHERE detection_date = :today"),
            {'today': today}
        )

    # Prepare records for all suppliers (anomalous + normal)
    # We save all so dashboard can show the full picture
    records = []
    for _, row in df.iterrows():
        records.append({
            'supplier_id':    row['supplier_id'],
            'anomaly_score':  float(row['anomaly_score']),
            'is_anomaly':     int(row['is_anomaly']),
            'risk_score':     float(row['avg_composite_risk']),
            'detection_date': today,
            'features_used':  json.dumps(ANOMALY_FEATURES),
        })

    records_df = pd.DataFrame(records)
    records_df.to_sql(
        'anomaly_scores',
        engine,
        if_exists='append',
        index=False,
        method='multi',
    )

    anomaly_count = df['is_anomaly'].sum()
    print(f"\n✅ Saved {len(records_df)} records to anomaly_scores table")
    print(f"   ({anomaly_count} flagged as anomalous)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔍 Starting Anomaly Detection Pipeline\n")
    engine = get_engine()

    # Load supplier profiles
    df = load_supplier_features(engine)

    # Run anomaly detection
    df_results = detect_anomalies(df)

    # Save to PostgreSQL
    save_anomalies(df_results, engine)

    print("\n🎯 Anomaly detection complete")
    print("   Results saved to anomaly_scores table")
    print("   Refresh dashboard to see updated alerts")


if __name__ == "__main__":
    main()