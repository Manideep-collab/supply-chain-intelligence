# fast_load.py
# Uses PostgreSQL COPY command via psycopg2 — much faster than INSERT
# COPY streams the entire CSV in one network operation
# vs INSERT which does one round trip per batch

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        port=os.getenv('POSTGRES_PORT'),
        dbname=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        connect_timeout=30,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )


def copy_table(conn, csv_path, table_name, columns):
    """
    Uses PostgreSQL COPY FROM STDIN — streams entire CSV
    in one operation. Orders of magnitude faster than INSERT.
    """
    print(f"📥 Loading {table_name} from {csv_path}...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        next(f)  # skip header row
        cur = conn.cursor()
        cur.copy_from(
            f,
            table_name,
            sep=',',
            null='',
            columns=columns
        )
        conn.commit()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cur.fetchone()[0]
        print(f"✅ {table_name}: {count:,} rows loaded")
        cur.close()


def main():
    print("🚀 Fast loading data to Render PostgreSQL...\n")

    conn = get_connection()
    cur = conn.cursor()

    # Clear existing data
    print("🗑️  Clearing existing data...")
    cur.execute("""
        TRUNCATE TABLE supplier_risk_scores,
                    shipments,
                    demand_signals,
                    suppliers
        CASCADE
    """)
    conn.commit()
    cur.close()
    print("✅ Cleared\n")

    # Load suppliers first (shipments references it)
    copy_table(
        conn,
        'data/export_suppliers.csv',
        'suppliers',
        ['supplier_id', 'supplier_name', 'country',
         'region', 'category', 'reliability_score', 'created_at']
    )

    # Load shipments
    copy_table(
        conn,
        'data/export_shipments.csv',
        'shipments',
        ['shipment_id', 'supplier_id', 'origin_country',
         'destination_country', 'product_category', 'quantity',
         'scheduled_date', 'actual_date', 'delay_days',
         'status', 'transport_mode', 'created_at']
    )

    # Load demand signals
    copy_table(
        conn,
        'data/export_demand_signals.csv',
        'demand_signals',
        ['signal_id','product_category', 'region',
         'demand_quantity', 'signal_date', 'created_at']
    )

    conn.close()
    print("\n🎯 Fast load complete!")


if __name__ == "__main__":
    main()