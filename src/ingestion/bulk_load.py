# bulk_load.py
# Cloud-safe version — inserts row by row using psycopg2 directly.
# Slower than batch insert but works reliably on Render PostgreSQL
# which has statement size limits that reject large multi-row inserts.

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

def get_connection():
    """Direct psycopg2 connection — bypasses SQLAlchemy batching."""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        port=os.getenv('POSTGRES_PORT'),
        dbname=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        connect_timeout=30,
        # Important for Render — keeps connection alive
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )


def parse_date(date_str):
    if not date_str or str(date_str) == 'nan':
        return None
    try:
        return datetime.strptime(
            str(date_str).strip(), "%m/%d/%Y %H:%M"
        ).date()
    except:
        return None


def safe_int(val, default=0):
    try:
        return int(val)
    except:
        return default


def safe_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default


def bulk_load():
    print("📂 Reading full DataCo dataset...")
    df = pd.read_csv(
        "data/DataCoSupplyChainDataset.csv",
        encoding='latin1'
    )
    print(f"📊 Total rows: {len(df):,}")

    conn = get_connection()
    cur  = conn.cursor()

    # Clear existing data
    print("🗑️  Clearing existing data...")
    cur.execute("""
        TRUNCATE TABLE supplier_risk_scores,
                       shipments,
                       demand_signals
        CASCADE
    """)
    conn.commit()
    print("✅ Tables cleared")

    # ── Insert shipments ──────────────────────────────────────
    print("\n💾 Inserting shipments...")

    shipment_sql = """
        INSERT INTO shipments (
            shipment_id, supplier_id, origin_country,
            destination_country, product_category, quantity,
            scheduled_date, actual_date, delay_days,
            status, transport_mode
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (shipment_id) DO NOTHING
    """

    demand_sql = """
        INSERT INTO demand_signals (
            product_category, region,
            demand_quantity, signal_date
        ) VALUES (%s, %s, %s, %s)
    """

    batch_size   = 1000  # commit every 1000 rows
    success      = 0
    errors       = 0
    ship_batch   = []
    demand_batch = []

    for idx, row in df.iterrows():
        try:
            country = str(row['Order Country']).strip()
            supplier_id = f"SUP_{country.replace(' ', '_').upper()[:20]}"

            ship_batch.append((
                str(row['Order Item Id']),
                supplier_id,
                country,
                str(row['Market']),
                str(row['Category Name']),
                safe_int(row['Order Item Quantity']),
                parse_date(row['order date (DateOrders)']),
                parse_date(row['shipping date (DateOrders)']),
                safe_int(row['Days for shipping (real)']) -
                safe_int(row['Days for shipment (scheduled)']),
                str(row['Delivery Status']),
                str(row['Shipping Mode']),
            ))

            demand_batch.append((
                str(row['Category Name']),
                str(row['Order Region']),
                safe_int(row['Order Item Quantity']),
                parse_date(row['order date (DateOrders)']),
            ))

            # Commit in batches
            if len(ship_batch) >= batch_size:
                cur.executemany(shipment_sql, ship_batch)
                cur.executemany(demand_sql, demand_batch)
                conn.commit()
                success += len(ship_batch)
                ship_batch   = []
                demand_batch = []
                print(f"   ✅ Inserted {success:,} rows...")

        except Exception as e:
            errors += 1
            conn.rollback()
            ship_batch   = []
            demand_batch = []
            if errors <= 3:
                print(f"   ⚠️  Error at row {idx}: {e}")
            continue

    # Insert remaining rows
    if ship_batch:
        try:
            cur.executemany(shipment_sql, ship_batch)
            cur.executemany(demand_sql, demand_batch)
            conn.commit()
            success += len(ship_batch)
        except Exception as e:
            print(f"   ⚠️  Final batch error: {e}")
            conn.rollback()

    cur.close()
    conn.close()

    print(f"\n✅ Bulk load complete")
    print(f"   Inserted : {success:,} rows")
    print(f"   Errors   : {errors}")


if __name__ == "__main__":
    bulk_load()