# Purpose: Transform raw shipment + supplier data from PostgreSQL
# into a clean, ML-ready feature matrix for XGBoost training.
#
# Built after full EDA on 180,519 rows. Every decision is data-backed.
#
# Key EDA findings that shaped this file:
#   1. Delay range is narrow (-2 to +4) — use buckets not ratios
#   2. First Class shipping = 100% late rate — ordinal risk encoding
#   3. Market risk near-identical across all 5 markets — dropped
#   4. Small countries/categories need Bayesian smoothing
#   5. Classes nearly balanced (54.8% late) — no resampling needed
#   6. Raw numerical correlations weak — XGBoost handles non-linearity

import pandas as pd
import numpy as np
from sqlalchemy import text
import sys
import os

# Add ingestion folder to path so we can import db.py
sys.path.append(os.path.join(os.path.dirname(__file__), '../ingestion'))
from db import get_engine


# STEP 1: LOAD RAW DATA FROM POSTGRESQL

def load_raw_data(engine):
    """
    Pull joined shipment + supplier data directly from PostgreSQL.
    Always work from the database, never from raw CSV in this stage.
    """
    query = """
        SELECT 
            s.shipment_id,
            s.supplier_id,
            s.origin_country,
            s.destination_country,
            s.product_category,
            s.quantity,
            s.scheduled_date,
            s.actual_date,
            s.delay_days,
            s.status,
            s.transport_mode,
            sup.reliability_score
        FROM shipments s
        LEFT JOIN suppliers sup 
            ON s.supplier_id = sup.supplier_id
    """
    df = pd.read_sql(query, engine)
    print(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns from PostgreSQL")
    return df


# STEP 2: BUILD ALL FEATURES

def build_features(df):
    """
    Core feature engineering function.
    Transforms raw columns into meaningful ML signals.
    Each section is explained with the EDA finding behind it.
    """

    # Global late rate — used as prior in Bayesian smoothing
    # From EDA: 54.8% of all shipments are late
    global_late_rate = 0.548

    # Smoothing factor — minimum shipments needed for full confidence
    # Below this, we blend toward the global average
    m = 50


    # ── TARGET VARIABLE ───────────────────────────────────────
    # Binary classification target
    # 1 = Late delivery, 0 = everything else (on time, early, canceled)
    df['is_late'] = (df['status'] == 'Late delivery').astype(int)


    # ── DELAY FEATURES ────────────────────────────────────────
    # EDA finding: delay range is only -2 to +4 days.
    # Ratios are meaningless in such a narrow range.
    # Buckets capture the pattern cleanly instead.

    # Early flag: arrived before scheduled date
    df['is_early'] = (df['delay_days'] < 0).astype(int)

    # On time flag: arrived exactly on scheduled date
    df['is_on_time'] = (df['delay_days'] == 0).astype(int)

    # Delay bucket — ordinal severity score
    # 0=Early, 1=On Time, 2=Slight(1d), 3=Moderate(2-3d), 4=Severe(4d+)
    df['delay_bucket'] = pd.cut(
        df['delay_days'],
        bins=[-3, -1, 0, 1, 3, 999],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)


    # ── TRANSPORT MODE FEATURES ───────────────────────────────
    # EDA finding — late rates by mode:
    #   First Class    → 100.00% late  (highest risk)
    #   Second Class   →  79.73% late
    #   Same Day       →  47.83% late
    #   Standard Class →  39.77% late  (lowest risk)
    #
    # Ordinal encoding captures this real-world risk ranking.
    # One-hot encoding is also kept so model has both representations.

    transport_risk_map = {
        'First Class':    4,  # 100% late — counterintuitively worst
        'Second Class':   3,  # 79.73% late
        'Same Day':       2,  # 47.83% late
        'Standard Class': 1,  # 39.77% late — safest option
    }
    df['transport_risk_score'] = (
        df['transport_mode']
        .map(transport_risk_map)
        .fillna(2)  # unknown modes get middle risk
    )

    # One-hot encoding for transport mode
    transport_dummies = pd.get_dummies(
        df['transport_mode'],
        prefix='mode'
    )
    df = pd.concat([df, transport_dummies], axis=1)


    # ── SUPPLIER FEATURES ─────────────────────────────────────
    # Two layers of supplier risk:
    #   Layer 1: reliability_score (from suppliers table)
    #   Layer 2: observed late rate from actual shipment history
    # Combining both gives a more complete supplier risk picture.

    # Per-supplier shipment counts and late rate
    supplier_stats = df.groupby('supplier_id').agg(
        supplier_total=('is_late', 'count'),
        supplier_late_rate=('is_late', 'mean')
    )
    df = df.join(supplier_stats, on='supplier_id')

    # Bayesian smoothed supplier risk
    # Suppliers with few shipments are pulled toward global average
    # Formula: (n * observed_rate + m * prior) / (n + m)
    df['supplier_risk_index'] = (
        (df['supplier_total'] * df['supplier_late_rate'] +
         m * global_late_rate) /
        (df['supplier_total'] + m)
    ).round(4)

    # Composite risk: blend smoothed index with reliability score
    # reliability_score = 1 means perfect, 0 means worst
    # We invert it (1 - score) so higher = more risky, consistent with other features
    df['supplier_composite_risk'] = (
        df['supplier_risk_index'] * 0.6 +
        (1 - df['reliability_score']) * 0.4
    ).round(4)


    # ── GEOGRAPHIC FEATURES ───────────────────────────────────
    # EDA finding: small countries (Laos=6, Armenia=5 shipments)
    # showed 100% late rate — statistically meaningless.
    # Bayesian smoothing fixes this.
    #
    # EDA finding: destination market risk was nearly identical
    # across all 5 markets (56.8% to 57.7%) — dropped as useless.

    country_stats = df.groupby('origin_country').agg(
        country_total=('is_late', 'count'),
        country_late_rate=('is_late', 'mean')
    )
    df = df.join(country_stats, on='origin_country')

    df['country_risk_score'] = (
        (df['country_total'] * df['country_late_rate'] +
         m * global_late_rate) /
        (df['country_total'] + m)
    ).round(4)

    # market_risk_score intentionally excluded —
    # all 5 markets within 1% of each other (EDA Cell 10)


    # ── PRODUCT CATEGORY FEATURES ─────────────────────────────
    # EDA finding: real spread exists (Golf Bags 68.85% vs avg 54.8%)
    # but some categories have very few shipments — smooth them.

    category_stats = df.groupby('product_category').agg(
        category_total=('is_late', 'count'),
        category_late_rate=('is_late', 'mean')
    )
    df = df.join(category_stats, on='product_category')

    df['category_risk_score'] = (
        (df['category_total'] * df['category_late_rate'] +
         m * global_late_rate) /
        (df['category_total'] + m)
    ).round(4)

    # Label encode product category for the model
    df['category_encoded'] = pd.factorize(df['product_category'])[0]


    # ── QUANTITY FEATURES ─────────────────────────────────────
    # Log transform reduces the effect of extreme order sizes
    df['quantity_log'] = np.log1p(df['quantity'])

    # Bulk orders (top 25%) may strain supplier fulfillment capacity
    df['is_bulk_order'] = (
        df['quantity'] > df['quantity'].quantile(0.75)
    ).astype(int)


    # ── DATE / TIME FEATURES ──────────────────────────────────
    # Seasonality matters in supply chains —
    # Q4 holiday pressure, month-end rushes, day-of-week patterns

    if df['scheduled_date'].notna().any():
        df['scheduled_date'] = pd.to_datetime(df['scheduled_date'])

        df['order_month'] = df['scheduled_date'].dt.month
        df['order_dayofweek'] = df['scheduled_date'].dt.dayofweek
        df['order_quarter'] = df['scheduled_date'].dt.quarter

        # Month-end orders tend to be rushed
        df['is_month_end'] = (
            df['scheduled_date'].dt.day >= 25
        ).astype(int)

        # Q4 = Oct/Nov/Dec = holiday season = highest demand pressure
        df['is_q4'] = (df['order_quarter'] == 4).astype(int)

    print(f"✅ Feature engineering complete — {len(df.columns)} total columns built")
    return df



# STEP 3: SELECT FINAL ML-READY FEATURES

def get_ml_ready_features(df):
    """
    Select the final 22 features going into XGBoost.
    Drops raw strings, IDs, intermediate columns, and
    any features proven useless by EDA.

    Returns:
        X            — feature matrix (DataFrame)
        y            — target vector (Series)
        feature_cols — list of column names (for MLflow logging)
    """

    feature_cols = [
        # Supplier risk
        'reliability_score',
        'supplier_late_rate',
        'supplier_risk_index',
        'supplier_composite_risk',

        # Geographic risk
        'country_risk_score',
        # market_risk_score EXCLUDED — all markets within 1% (EDA)

        # Transport mode
        'transport_risk_score',       # ordinal: 1-4
        'mode_First Class',           # one-hot
        'mode_Same Day',
        'mode_Second Class',
        'mode_Standard Class',

        # Delay history
        'delay_bucket',               # 0=early to 4=severe
        'is_early',
        'is_on_time',

        # Product category
        'category_encoded',
        'category_risk_score',

        # Order size
        'quantity_log',
        'is_bulk_order',

        # Time / seasonality
        'order_month',
        'order_dayofweek',
        'order_quarter',
        'is_month_end',
        'is_q4',
    ]

    # Only use columns that actually exist in the dataframe
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols   = [c for c in feature_cols if c not in df.columns]

    if missing_cols:
        print(f"⚠️  Columns not found (skipped): {missing_cols}")

    # Fill any remaining nulls with 0 — safe default for all features here
    X = df[available_cols].fillna(0)
    y = df['is_late']

    print(f"\n✅ Final feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"✅ Target distribution  : "
          f"{y.sum():,} late ({y.mean()*100:.1f}%) | "
          f"{(~y.astype(bool)).sum():,} on time ({(1-y.mean())*100:.1f}%)")
    print(f"\n📋 Features selected:")
    for i, col in enumerate(available_cols, 1):
        print(f"   {i:2d}. {col}")

    return X, y, available_cols


# STEP 4: SAVE FEATURE STORE TO POSTGRESQL

def save_features_to_db(df, engine):
    """
    Persist the engineered feature table back into PostgreSQL
    as 'feature_store'. This is the input table for Phase 4 modeling.

    Uses if_exists='replace' so re-running this script always
    gives you a fresh, consistent feature store.
    """

    # Columns to persist — raw strings and intermediates excluded
    cols_to_store = [
        'shipment_id',
        'supplier_id',
        'origin_country',
        'destination_country',
        'transport_mode',
        'transport_risk_score',
        'reliability_score',
        'supplier_late_rate',
        'supplier_risk_index',
        'supplier_composite_risk',
        'country_risk_score',
        'category_risk_score',
        'category_encoded',
        'delay_bucket',
        'is_early',
        'is_on_time',
        'is_late',
        'quantity_log',
        'is_bulk_order',
    ]

    # Add date columns only if they were created
    date_cols = [
        'order_month', 'order_dayofweek', 'order_quarter',
        'is_month_end', 'is_q4'
    ]
    cols_to_store += [c for c in date_cols if c in df.columns]

    # Add one-hot mode columns
    mode_cols = [c for c in df.columns if c.startswith('mode_')]
    cols_to_store += mode_cols

    feature_store = df[cols_to_store].copy()

    feature_store.to_sql(
        name='feature_store',
        con=engine,
        if_exists='replace',   # always rebuild fresh
        index=False,
        method='multi',
        chunksize=5000         # batch inserts — faster for 180K rows
    )

    print(f"\n✅ Saved {len(feature_store):,} rows → 'feature_store' table in PostgreSQL")


# MAIN — runs the full pipeline in order

if __name__ == "__main__":
    print("🚀 Starting feature engineering pipeline...\n")

    engine = get_engine()

    # Step 1: Load
    df_raw = load_raw_data(engine)

    # Step 2: Build features
    df_features = build_features(df_raw)

    # Step 3: Get ML-ready matrix (prints feature list)
    X, y, feature_cols = get_ml_ready_features(df_features)

    # Step 4: Save to DB
    save_features_to_db(df_features, engine)

    print("\n🎯 Feature engineering pipeline complete.")
    print("   feature_store table is ready for Phase 4 — XGBoost training.")