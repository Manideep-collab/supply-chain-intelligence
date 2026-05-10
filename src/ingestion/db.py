# db.py
# This file creates and manages connection to PostgreSQL.
# SQLAlchemy is an ORM (Object Relational Mapper) - it lets us 
# talk to PostgreSQL using Python instead of raw SQL strings.

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load all variables from our .env file into memory
load_dotenv()

def get_engine():
    """
    Builds the PostgreSQL connection string and returns an engine.
    Engine = the actual connection pool to our database.
    Connection string format: postgresql://user:password@host:port/dbname
    """
    conn_string = (
        f"postgresql://{os.getenv('POSTGRES_USER')}"
        f":{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}"
        f":{os.getenv('POSTGRES_PORT')}"
        f"/{os.getenv('POSTGRES_DB')}"
    )
    
    # pool_pre_ping=True means SQLAlchemy checks if connection 
    # is alive before using it — prevents stale connection errors
    engine = create_engine(conn_string, pool_pre_ping=True)
    return engine


def test_connection():
    """Quick test to verify database is reachable."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            print(f"✅ PostgreSQL connected: {result.fetchone()[0]}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")


if __name__ == "__main__":
    test_connection()