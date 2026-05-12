from src.ingestion.db import get_engine
from sqlalchemy import text

engine = get_engine()

with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM shipments;"))
    count = result.fetchone()[0]

print("Shipment rows:", count)