# consumer.py
# The consumer's job: SUBSCRIBE to Kafka topic, read each message,
# and write it permanently into PostgreSQL.
#
# Key concept: the consumer runs FOREVER in a loop.
# It's always listening. When a new message arrives, it processes it.
# This is exactly how real data pipelines work.

import os
import json
from kafka import KafkaConsumer
from sqlalchemy import text
from dotenv import load_dotenv
from datetime import datetime
from db import get_engine

load_dotenv()


# ── Kafka Consumer Setup ──────────────────────────────────────────────────────

def create_consumer(topic):
    """
    Create and return a Kafka consumer.
    
    group_id: identifies this consumer as part of a group.
    If you run multiple consumers with the same group_id,
    Kafka automatically splits the work between them.
    That's how companies scale — just add more consumers.
    
    auto_offset_reset='earliest': if this consumer has never run before,
    start reading from the very first message in the topic.
    If it has run before, continue from where it left off.
    Kafka remembers your position (called 'offset') automatically.
    
    value_deserializer: reverse of the producer's serializer.
    Convert the bytes back into a Python dict.
    """
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        group_id='supply-chain-consumer-group',
        auto_offset_reset='earliest',
        enable_auto_commit=True,   # automatically mark messages as "read"
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        # Wait up to 1 second to collect messages before processing
        consumer_timeout_ms=1000,
    )
    print(f"✅ Kafka Consumer connected | Listening on topic: '{topic}'")
    return consumer


# ── Database Write Functions ──────────────────────────────────────────────────

def insert_shipment(conn, event):
    """
    Insert one shipment event into the shipments table.
    
    We use INSERT ... ON CONFLICT DO NOTHING because:
    - shipment_id is our primary key
    - if we restart the consumer, it re-reads old messages
    - we don't want duplicate rows — just skip if already exists
    """
    
    # Parse dates safely — CSV dates are strings, DB needs proper dates
    def parse_date(date_str):
        if not date_str or date_str == 'nan':
            return None
        try:
            # DataCo format: "1/31/2018 22:56"
            return datetime.strptime(date_str.strip(), "%m/%d/%Y %H:%M").date()
        except:
            return None

    sql = text("""
        INSERT INTO shipments (
            shipment_id, supplier_id, origin_country, destination_country,
            product_category, quantity, scheduled_date, actual_date,
            delay_days, status, transport_mode
        ) VALUES (
            :shipment_id, :supplier_id, :origin_country, :destination_country,
            :product_category, :quantity, :scheduled_date, :actual_date,
            :delay_days, :status, :transport_mode
        )
        ON CONFLICT (shipment_id) DO NOTHING
    """)

    conn.execute(sql, {
        "shipment_id": event["order_item_id"],
        # We use order_country as a proxy for supplier_id
        # In Phase 3 we'll build proper supplier profiles
        "supplier_id": f"SUP_{event['order_country'].replace(' ', '_').upper()[:20]}",
        "origin_country": event["order_country"],
        "destination_country": event["market"],
        "product_category": event["category_name"],
        "quantity": event["order_quantity"],
        "scheduled_date": parse_date(event["order_date"]),
        "actual_date": parse_date(event["shipping_date"]),
        "delay_days": event["delay_days"],
        "status": event["delivery_status"],
        "transport_mode": event["shipping_mode"],
    })


def insert_demand_signal(conn, event):
    """
    Every order is also a demand signal — someone wanted this product.
    We record product + region + quantity as demand data.
    Prophet will later use this to forecast future demand.
    """
    sql = text("""
        INSERT INTO demand_signals (
            product_category, region, demand_quantity, signal_date
        ) VALUES (
            :product_category, :region, :demand_quantity, :signal_date
        )
    """)

    def parse_date(date_str):
        if not date_str or date_str == 'nan':
            return None
        try:
            return datetime.strptime(date_str.strip(), "%m/%d/%Y %H:%M").date()
        except:
            return None

    conn.execute(sql, {
        "product_category": event["category_name"],
        "region": event["order_region"],
        "demand_quantity": event["order_quantity"],
        "signal_date": parse_date(event["order_date"]),
    })


# ── Main Consumer Loop ────────────────────────────────────────────────────────

def run_consumer(topic):
    """
    The main loop. Runs forever, processing messages as they arrive.
    Each message: insert shipment + insert demand signal into PostgreSQL.
    """
    consumer = create_consumer(topic)
    engine = get_engine()

    total_processed = 0
    total_errors = 0

    print("🎧 Listening for messages... (Press Ctrl+C to stop)\n")

    try:
        # This loop runs forever — it's a "poll loop"
        # KafkaConsumer is iterable — each iteration gives you one message
        while True:
            # Poll for messages — waits up to 1 second if none available
            messages = consumer.poll(timeout_ms=1000)
            
            if not messages:
                # No messages right now, keep waiting
                continue

            # messages is a dict: {TopicPartition: [list of messages]}
            # We flatten it to just get the message objects
            for topic_partition, message_list in messages.items():
                for message in message_list:
                    try:
                        event = message.value  # already deserialized to dict
                        
                        # Use a transaction — if either insert fails,
                        # both are rolled back. Data stays consistent.
                        with engine.begin() as conn:
                            insert_shipment(conn, event)
                            insert_demand_signal(conn, event)
                        
                        total_processed += 1
                        
                        if total_processed % 50 == 0:
                            print(f"✅ Processed {total_processed:,} messages | "
                                  f"Latest: {event['category_name']} | "
                                  f"Region: {event['order_region']} | "
                                  f"Delay: {event['delay_days']}d")

                    except Exception as e:
                        total_errors += 1
                        print(f"⚠️  Error processing message: {e}")
                        continue

    except KeyboardInterrupt:
        # User pressed Ctrl+C — shut down cleanly
        print(f"\n🛑 Consumer stopped")
        print(f"📊 Total processed: {total_processed:,} | Errors: {total_errors}")
    finally:
        consumer.close()
        print("👋 Consumer connection closed cleanly")


if __name__ == "__main__":
    run_consumer(
        topic=os.getenv("KAFKA_TOPIC_SHIPMENTS", "shipment-events")
    )