# bulk_load.py
# Bulk loads the FULL DataCo dataset directly into PostgreSQL.
# This bypasses Kafka intentionally — Kafka is for real-time NEW events.
# Historical data always gets bulk loaded first. Industry standard pattern.

import pandas as pd
from sqlalchemy import text
from db import get_engine
from datetime import datetime

def parse_date(date_str):
    if not date_str or str(date_str) == 'nan':
        return None
    try:
        return datetime.strptime(str(date_str).strip(), "%m/%d/%Y %H:%M").date()
    except:
        return None

def bulk_load_shipments():
    engine = get_engine()

    print("📂 Reading full DataCo dataset...")
    df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding='latin1')
    print(f"📊 Total rows: {len(df):,}")

    # Clear existing test data
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE supplier_risk_scores, shipments, demand_signals CASCADE"))
    print("🗑️  Cleared existing test data")

    # Build records
    records = []
    demand_records = []

    for _, row in df.iterrows():
        supplier_id = f"SUP_{str(row['Order Country']).strip().replace(' ', '_').upper()[:20]}"

        records.append({
            "shipment_id": str(row["Order Item Id"]),
            "supplier_id": supplier_id,
            "origin_country": str(row["Order Country"]),
            "destination_country": str(row["Market"]),
            "product_category": str(row["Category Name"]),
            "quantity": int(row["Order Item Quantity"]) if pd.notna(row["Order Item Quantity"]) else 0,
            "scheduled_date": parse_date(row["order date (DateOrders)"]),
            "actual_date": parse_date(row["shipping date (DateOrders)"]),
            "delay_days": int(row["Days for shipping (real)"]) - int(row["Days for shipment (scheduled)"]),
            "status": str(row["Delivery Status"]),
            "transport_mode": str(row["Shipping Mode"]),
        })

        demand_records.append({
            "product_category": str(row["Category Name"]),
            "region": str(row["Order Region"]),
            "demand_quantity": int(row["Order Item Quantity"]) if pd.notna(row["Order Item Quantity"]) else 0,
            "signal_date": parse_date(row["order date (DateOrders)"]),
        })

    print(f"⚙️  Built {len(records):,} shipment records")
    print("💾 Writing to PostgreSQL in batches...")

    # Write in batches of 5000 — faster than row by row
    # and doesn't overwhelm memory like doing all at once
    batch_size = 5000
    shipment_df = pd.DataFrame(records)
    demand_df = pd.DataFrame(demand_records)

    # Use pandas to_sql for fast bulk insert
    shipment_df.to_sql(
        'shipments',
        engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=batch_size
    )
    print(f"✅ Shipments loaded: {len(shipment_df):,} rows")

    demand_df.to_sql(
        'demand_signals',
        engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=batch_size
    )
    print(f"✅ Demand signals loaded: {len(demand_df):,} rows")

    # Verify
    with engine.connect() as conn:
        s_count = conn.execute(text("SELECT COUNT(*) FROM shipments")).scalar()
        d_count = conn.execute(text("SELECT COUNT(*) FROM demand_signals")).scalar()
        print(f"\n📊 Final counts:")
        print(f"   shipments      → {s_count:,} rows")
        print(f"   demand_signals → {d_count:,} rows")

if __name__ == "__main__":
    bulk_load_shipments()