-- Suppliers table
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id VARCHAR(50) PRIMARY KEY,
    supplier_name VARCHAR(255) NOT NULL,
    country VARCHAR(100),
    region VARCHAR(100),
    category VARCHAR(100),
    reliability_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Shipments table
CREATE TABLE IF NOT EXISTS shipments (
    shipment_id VARCHAR(50) PRIMARY KEY,
    supplier_id VARCHAR(50) REFERENCES suppliers(supplier_id),
    origin_country VARCHAR(100),
    destination_country VARCHAR(100),
    product_category VARCHAR(100),
    quantity INTEGER,
    scheduled_date DATE,
    actual_date DATE,
    delay_days INTEGER,
    status VARCHAR(50),
    transport_mode VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Demand signals table
CREATE TABLE IF NOT EXISTS demand_signals (
    signal_id SERIAL PRIMARY KEY,
    product_category VARCHAR(100),
    region VARCHAR(100),
    demand_quantity INTEGER,
    signal_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Risk scores table (filled by our ML model later)
CREATE TABLE IF NOT EXISTS supplier_risk_scores (
    score_id SERIAL PRIMARY KEY,
    supplier_id VARCHAR(50) REFERENCES suppliers(supplier_id),
    risk_score FLOAT,
    risk_label VARCHAR(20),
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);