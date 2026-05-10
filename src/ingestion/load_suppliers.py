# load_suppliers.py
# Generates supplier nodes directly from DataCo dataset.
# Each unique origin country = one supplier node.
# This ensures supplier_id always exists before shipments reference it.

import pandas as pd
from sqlalchemy import text
from db import get_engine

def load_suppliers():
    engine = get_engine()

    print("📂 Reading DataCo dataset to extract supplier nodes...")
    df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding='latin1')

    # Each unique order country = one supplier node
    countries = df['Order Country'].dropna().unique()
    print(f"🌍 Found {len(countries)} unique supplier countries")

    # Also pull delivery stats per country to build reliability score
    # reliability = % of orders that were NOT late
    country_stats = df.groupby('Order Country').agg(
        total_orders=('Order Id', 'count'),
        late_orders=('Late_delivery_risk', 'sum')
    ).reset_index()
    country_stats['reliability_score'] = round(
        1 - (country_stats['late_orders'] / country_stats['total_orders']), 4
    )

    with engine.begin() as conn:
        for _, row in country_stats.iterrows():
            country = str(row['Order Country']).strip()
            supplier_id = f"SUP_{country.replace(' ', '_').upper()[:20]}"

            sql = text("""
                INSERT INTO suppliers (
                    supplier_id, supplier_name, country,
                    region, category, reliability_score
                ) VALUES (
                    :supplier_id, :supplier_name, :country,
                    :region, :category, :reliability_score
                )
                ON CONFLICT (supplier_id) DO NOTHING
            """)

            conn.execute(sql, {
                "supplier_id": supplier_id,
                "supplier_name": f"{country} Supply Node",
                "country": country,
                "region": "Global",
                "category": "Mixed",
                "reliability_score": float(row['reliability_score']),
            })

    print("✅ Supplier nodes loaded successfully")


if __name__ == "__main__":
    load_suppliers()