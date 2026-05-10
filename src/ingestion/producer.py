# producer.py
# The producer's job: read shipment data and PUBLISH it to Kafka.
# Think of this as the "data source" side — simulating a live ERP system
# that fires events whenever a shipment is created or updated.

import os
import json
import time
import pandas as pd
from kafka import KafkaProducer
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── Kafka Producer Setup ──────────────────────────────────────────────────────
# KafkaProducer is the client that connects to our Kafka broker (the server).
# 
# value_serializer: Kafka only understands bytes, not Python dicts.
# So we tell it: "before sending any message, convert it to JSON 
# string, then encode as UTF-8 bytes."
# lambda m: means "for any message m, do this transformation"

def create_producer():
    """Create and return a Kafka producer instance."""
    producer = KafkaProducer(
        bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        value_serializer=lambda m: json.dumps(m).encode('utf-8'),
        # If Kafka is temporarily busy, retry up to 3 times before failing
        retries=3,
        # Wait up to 1 second to batch messages for efficiency
        linger_ms=1000,
    )
    print("✅ Kafka Producer connected")
    return producer


# ── Data Cleaning ─────────────────────────────────────────────────────────────
# Raw CSV data is messy. We clean each row BEFORE sending to Kafka.
# Rule: never send dirty data downstream — clean at the source.

def clean_shipment_row(row):
    """
    Takes one raw CSV row, extracts relevant fields,
    cleans them, and returns a structured dict (our 'event').
    
    We're selecting only the columns we actually need —
    not all 53 columns from the CSV.
    """
    
    def safe_float(val):
        """Convert to float safely, return 0.0 if it fails."""
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def safe_int(val):
        """Convert to int safely, return 0 if it fails."""
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    return {
        # Order identifiers
        "order_id": str(row.get("Order Id", "")),
        "order_item_id": str(row.get("Order Item Id", "")),

        # Customer & geography
        "customer_id": str(row.get("Customer Id", "")),
        "customer_segment": str(row.get("Customer Segment", "Unknown")),
        "market": str(row.get("Market", "Unknown")),
        "order_region": str(row.get("Order Region", "Unknown")),
        "order_country": str(row.get("Order Country", "Unknown")),
        "order_city": str(row.get("Order City", "Unknown")),

        # Product info
        "product_name": str(row.get("Product Name", "Unknown")),
        "category_name": str(row.get("Category Name", "Unknown")),
        "department_name": str(row.get("Department Name", "Unknown")),

        # Shipping info — this is the core of our disruption analysis
        "shipping_mode": str(row.get("Shipping Mode", "Unknown")),
        "delivery_status": str(row.get("Delivery Status", "Unknown")),
        "days_scheduled": safe_int(row.get("Days for shipment (scheduled)", 0)),
        "days_actual": safe_int(row.get("Days for shipping (real)", 0)),
        # Delay = actual - scheduled. Positive = late, Negative = early
        "delay_days": safe_int(row.get("Days for shipping (real)", 0)) - 
                      safe_int(row.get("Days for shipment (scheduled)", 0)),

        # Target variable for our ML model
        "late_delivery_risk": safe_int(row.get("Late_delivery_risk", 0)),

        # Financial data
        "sales": safe_float(row.get("Sales", 0)),
        "order_quantity": safe_int(row.get("Order Item Quantity", 0)),
        "order_profit": safe_float(row.get("Order Profit Per Order", 0)),
        "benefit_per_order": safe_float(row.get("Benefit per order", 0)),

        # Timestamps
        "order_date": str(row.get("order date (DateOrders)", "")),
        "shipping_date": str(row.get("shipping date (DateOrders)", "")),

        # Metadata — when this event was produced (right now)
        "event_timestamp": datetime.utcnow().isoformat(),
        "order_status": str(row.get("Order Status", "Unknown")),
    }


# ── Main Stream Function ──────────────────────────────────────────────────────

def stream_shipments(csv_path, topic, delay_seconds=1, max_rows=None):
    """
    Reads the CSV and streams each row into Kafka one by one.
    
    csv_path: path to DataCo CSV
    topic: Kafka topic name (from .env)
    delay_seconds: pause between messages (simulates real-time arrival)
    max_rows: stop after N rows (useful for testing)
    """
    
    print(f"📂 Loading dataset from: {csv_path}")
    
    # latin1 encoding because DataCo CSV uses non-UTF8 characters
    df = pd.read_csv(csv_path, encoding='latin1', nrows=max_rows)
    print(f"📊 Loaded {len(df):,} rows | Streaming to topic: '{topic}'")
    
    producer = create_producer()
    
    success_count = 0
    error_count = 0

    for index, row in df.iterrows():
        try:
            # Clean the raw row into a structured event dict
            event = clean_shipment_row(row)
            
            # Send to Kafka
            # key: we use order_id as the message key.
            # Kafka uses keys for partitioning — same key always goes 
            # to the same partition, preserving order for that order_id.
            producer.send(
                topic=topic,
                key=event["order_id"].encode('utf-8'),
                value=event
            )
            
            success_count += 1
            
            # Print progress every 100 messages
            if success_count % 100 == 0:
                print(f"📤 Sent {success_count:,} events | "
                      f"Latest: Order {event['order_id']} | "
                      f"Status: {event['delivery_status']}")
            
            # This delay simulates real-time — in production this wouldn't
            # exist because events come naturally over time
            time.sleep(delay_seconds)

        except Exception as e:
            error_count += 1
            print(f"⚠️  Error on row {index}: {e}")
            continue

    # flush() forces any buffered messages to actually send before we exit
    producer.flush()
    print(f"\n✅ Stream complete | Sent: {success_count:,} | Errors: {error_count}")


if __name__ == "__main__":
    stream_shipments(
        csv_path="data/DataCoSupplyChainDataset.csv",
        topic=os.getenv("KAFKA_TOPIC_SHIPMENTS", "shipment-events"),
        delay_seconds=0.5,   # Send 2 events per second
        max_rows=500          # Start with 500 rows for testing
    )