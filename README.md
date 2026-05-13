# Supply Chain Disruption Intelligence Platform

> End-to-end real-time supply chain analytics platform — from raw logistics data to executive dashboards with ML-powered disruption risk scoring, demand forecasting, and anomaly detection.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-purple)](https://mlflow.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)](https://postgresql.org)

---

## 🌐 Live Demo

| Service | URL |
|---|---|
| 📊 Streamlit Dashboard | [supply-chain-intelligence.streamlit.app](https://supply-chain-intelligence-gexdg8kesw4b2hjxeq2p2x.streamlit.app/) |
| ⚡ FastAPI Docs | [supply-chain-api-d1b3.onrender.com/docs](https://supply-chain-api-d1b3.onrender.com/docs) |
| 🔍 Health Check | [supply-chain-api-d1b3.onrender.com/health](https://supply-chain-api-d1b3.onrender.com/health) |

> **Note:** Render free tier spins down after 15 minutes of inactivity. First request may take ~30 seconds to wake up.

---

## 📌 Project Overview

Post-COVID supply chain disruptions cost companies billions. This platform addresses that by building a system that continuously monitors supplier and logistics data, scores each supplier node for disruption risk, forecasts future demand, and surfaces everything in a live dashboard — so decision-makers get warnings **before** things break, not after.

### What This System Does

- **Streams** 180,519 historical shipment events through Apache Kafka into PostgreSQL
- **Engineers** 19 pre-shipment features including Bayesian-smoothed supplier risk scores
- **Scores** every shipment for disruption risk using a trained XGBoost classifier (AUC: 0.7562)
- **Forecasts** 5-week demand for top 5 product categories using XGBoost regression with lag features
- **Detects** anomalous supplier behavior using Isolation Forest (8 anomalies across 144 suppliers)
- **Serves** real-time predictions via a REST API with Pydantic validation
- **Visualizes** everything across a 7-page interactive Streamlit dashboard

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCE                              │
│         DataCo Supply Chain CSV — 180,519 shipments            │
│                      2015 – 2018                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    producer.py
              (Kafka Producer — streams rows
               as live shipment events)
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      APACHE KAFKA                               │
│                 Topic: shipment-events                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    consumer.py
              (Kafka Consumer — parses events,
               writes to PostgreSQL tables)
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POSTGRESQL DATABASE                           │
│                                                                 │
│  suppliers          →    164 rows                               │
│  shipments          →  180,519 rows                             │
│  demand_signals     →  180,519 rows                             │
│  feature_store      →  180,519 rows                             │
│  anomaly_scores     →    144 rows                               │
│  demand_forecasts   →     25 rows                               │
└──────┬──────────────────────┬───────────────────────┬──────────┘
       │                      │                       │
       ▼                      ▼                       ▼
feature_engineering    anomaly_detection       demand_forecast
     .py                    .py                    .py
  19 features          Isolation Forest       XGBoost Regressor
  Bayesian smoothing   contamination=0.05     lag + rolling features
       │                      │                       │
       └──────────────────────┼───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    XGBOOST CLASSIFIER                            │
│                                                                 │
│  Training rows : 144,415    Test rows : 36,104                 │
│  CV Folds      : 5-fold Stratified                              │
│                                                                 │
│  AUC-ROC   : 0.7562    CV AUC  : 0.7548 ± 0.0028              │
│  Recall    : 0.7194    CV F1   : 0.6859 ± 0.0032              │
│  Precision : 0.6969    CV Acc  : 0.7008 ± 0.0026              │
│  F1 Score  : 0.7079                                             │
│                                                                 │
│  6 experiments tracked in MLflow                                │
│  Registered as: supply-chain-risk-model/v6                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
┌─────────────────────┐    ┌────────────────────────────┐
│   FastAPI (REST)    │    │   Streamlit Dashboard      │
│                     │    │                            │
│ POST /predict/risk  │    │  📊 Executive Overview     │
│ POST /predict/batch │    │  🏭 Supplier Rankings      │
│ GET  /health        │    │  🚨 Alert Feed             │
│ GET  /model/info    │    │  🔮 Demand Forecast        │
│ GET  /docs          │    │  🔍 Anomaly Detection      │
│                     │    │  🎯 Risk Predictor         │
│ Deployed: Render    │    │  🤖 Model Performance      │
└─────────────────────┘    │                            │
                           │  Deployed: Streamlit Cloud │
                           └────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Streaming** | Apache Kafka + Zookeeper | Real-time shipment event ingestion |
| **Storage** | PostgreSQL 15 | Persistent data warehouse |
| **Processing** | Pandas, SQLAlchemy, psycopg2 | Data transformation and loading |
| **Feature Engineering** | Pandas, NumPy | 19 ML features with Bayesian smoothing |
| **Risk Model** | XGBoost Classifier | Disruption risk scoring (0–100) |
| **Forecast Model** | XGBoost Regressor | 5-week demand forecasting with CI |
| **Anomaly Detection** | Isolation Forest (scikit-learn) | Unusual supplier behavior detection |
| **Experiment Tracking** | MLflow 2.10 | 6 runs tracked, model registry |
| **API** | FastAPI + Uvicorn | REST API for real-time predictions |
| **Dashboard** | Streamlit + Plotly | 7-page interactive dashboard |
| **Infrastructure** | Docker Compose | Local orchestration (4 containers) |
| **Deployment** | Render + Streamlit Cloud | Public cloud hosting |

---

## 📊 Dataset

**DataCo Smart Supply Chain Dataset**

- 180,519 shipment records
- Date range: January 2015 – January 2018
- 164 unique supplier origin countries
- 5 destination markets: Europe, Pacific Asia, USCA, LATAM, Africa
- 50 product categories
- 4 shipping modes: Standard Class, Same Day, Second Class, First Class

**Key EDA Findings:**

- 54.8% late delivery rate — nearly balanced classes (no resampling needed)
- Delay range: -2 to +4 days — narrow, modeled with ordinal buckets not ratios
- First Class shipping: 100% late delivery rate across 27,814 shipments
- Standard Class: 39.77% late — safest transport option
- All 5 destination markets within 1% late rate of each other — excluded from features as noise
- Raw numerical correlations near-zero — confirms XGBoost over linear models

---

## 🤖 ML Pipeline Details

### Feature Engineering (19 features)

| Feature | Description | Why Included |
|---|---|---|
| `transport_risk_score` | Ordinal: Standard=1, Same Day=2, Second=3, First=4 | 60-point spread between modes in EDA |
| `reliability_score` | Supplier reliability from historical data | Direct quality signal |
| `supplier_late_rate` | Per-supplier historical late delivery rate | Strongest supplier predictor |
| `supplier_risk_index` | Bayesian smoothed supplier risk (m=50) | Handles low-volume suppliers |
| `supplier_composite_risk` | Blend: risk_index×0.6 + (1-reliability)×0.4 | Combined supplier signal |
| `country_risk_score` | Bayesian smoothed geographic risk (m=50) | Handles countries with few shipments |
| `category_risk_score` | Bayesian smoothed product category risk | Real spread across categories |
| `category_encoded` | Label encoded product category | Category identity for model |
| `quantity_log` | Log-transformed order quantity | Smooths extreme values |
| `is_bulk_order` | 1 if quantity in top 25th percentile | Large orders may strain suppliers |
| `order_month` | Month of order (1–12) | Seasonal demand patterns |
| `order_dayofweek` | Day of week (0=Monday, 6=Sunday) | Weekend order behavior |
| `order_quarter` | Quarter (1–4) | Quarterly business cycles |
| `is_month_end` | 1 if order day >= 25 | Month-end rush orders |
| `is_q4` | 1 if October–December | Holiday season pressure |
| `mode_First Class` | One-hot transport mode | Model needs binary + ordinal both |
| `mode_Same Day` | One-hot transport mode | Model needs binary + ordinal both |
| `mode_Second Class` | One-hot transport mode | Model needs binary + ordinal both |
| `mode_Standard Class` | One-hot transport mode | Model needs binary + ordinal both |

**Note on Bayesian Smoothing:** Countries like Laos (6 shipments, 100% late rate) are statistically unreliable. Formula: `(n × observed + 50 × global_mean) / (n + 50)` pulls small-sample scores toward the global average of 54.8%.

**Note on Market Risk Exclusion:** All 5 destination markets had late rates between 56.8% and 57.7% — less than 1% spread. Including it would add pure noise.

**Note on Data Leakage:** The first MLflow run (xgboost_baseline) used `is_on_time`, `is_early`, and `delay_bucket` as features — all derived from actual delivery outcomes. These were removed in subsequent runs since a real prediction system only has pre-shipment information available.

### XGBoost Classifier — Experiment History

| Run | AUC | Recall | Precision | F1 | Note |
|---|---|---|---|---|---|
| xgboost_baseline | 0.9810 | 0.9999 | 0.9557 | 0.9773 | ❌ Data leakage |
| xgboost_no_leakage | 0.7497 | 0.5742 | 0.8271 | 0.6778 | ❌ Recall too low |
| xgboost_tuned | 0.7557 | 0.6090 | 0.7981 | 0.6908 | ❌ Recall still low |
| xgboost_high_recall | 0.7551 | 0.8791 | 0.6036 | 0.7157 | ❌ Precision too low |
| xgboost_balanced | 0.7562 | 0.7194 | 0.6969 | 0.7079 | ✅ Production model |
| xgboost_final_v5 | 0.7562 | 0.7194 | 0.6969 | 0.7079 | ✅ Registered v6 |

**Final Model Configuration:**
```json
{
  "n_estimators": 500,
  "max_depth": 8,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "min_child_weight": 1,
  "gamma": 0.0,
  "scale_pos_weight": 1.5,
  "test_size": 0.2,
  "random_state": 42,
  "cv_folds": 5
}
```

**Why scale_pos_weight=1.5?** Missing a late shipment (False Negative) costs a company far more than investigating a false alarm (False Positive). Upweighting the late class increases recall from 0.61 to 0.72.

**Top Feature Importances (Final Model):**
```
mode_First Class          0.4827  (48.3%)
mode_Standard Class       0.2435  (24.4%)
transport_risk_score      0.1650  (16.5%)
mode_Second Class         0.0360   (3.6%)
mode_Same Day             0.0130   (1.3%)
supplier_composite_risk   0.0056   (0.6%)
country_risk_score        0.0050   (0.5%)
supplier_late_rate        0.0049   (0.5%)
supplier_risk_index       0.0047   (0.5%)
reliability_score         0.0047   (0.5%)
```

Transport mode dominates at 88% importance — directly consistent with the EDA finding that First Class has a 100% late delivery rate across 27,814 shipments.

### Anomaly Detection

- **Algorithm:** Isolation Forest
- **Suppliers analyzed:** 144 (with ≥10 shipments)
- **Features used:** avg_composite_risk, avg_reliability, avg_country_risk, avg_transport_risk, late_rate, total_shipments, avg_category_risk
- **Contamination:** 0.05 (expects ~5% anomalies)
- **Result:** 8 anomalies detected (5.6%)

Top anomalies detected:
```
Burkina Faso Supply Node       score=1.000 | late_rate=9.1%  | risk=0.316
Estonia Supply Node            score=0.873 | late_rate=96.5% | risk=0.807
Luxemburgo Supply Node         score=0.782 | late_rate=100%  | risk=0.774
República del Congo Supply Node score=0.717 | late_rate=92.3% | risk=0.745
```

### Demand Forecasting

- **Algorithm:** XGBoost Regressor with recursive multi-step forecasting
- **Categories forecast:** Cleats, Women's Apparel, Indoor/Outdoor Games, Cardio Equipment, Shop By Sport
- **History:** 145 weeks per category (Jan 2015 – Jan 2018)
- **Aggregation:** Weekly totals (daily quantities too small/noisy at 1-3 units)
- **Horizon:** 5 weeks ahead
- **Confidence intervals:** 95% (1.96 × residual std from validation set)
- **Lag features:** lag-1, lag-2, lag-4, lag-52 (same week last year)
- **Rolling features:** 4-week mean, 4-week std, 12-week mean, 4-week trend
- **Calendar features:** month, quarter, week_of_year, is_q4, is_month_end

Sample forecast results:
```
Cleats              → 536 units/week  [318 – 754]
Women's Apparel     → 448 units/week  [257 – 639]
Indoor/Outdoor Games→ 454 units/week  [266 – 641]
Cardio Equipment    → 251 units/week  [138 – 363]
Shop By Sport       → 244 units/week  [137 – 350]
```

---

## 📁 Project Structure

```
supply-chain-intelligence/
├── data/
│   ├── DataCoSupplyChainDataset.csv    # 180,519 rows, primary dataset
│   ├── supply_chain_data.csv           # supplier quality data
│   └── model_outputs/
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       ├── classification_report.txt
│       ├── feature_names.json
│       └── run_id.txt
├── models/
│   ├── xgboost_model.json              # exported model for deployment
│   └── feature_names.json
├── notebooks/
│   └── 01_eda.ipynb                    # full EDA on 180,519 rows
├── sql/
│   └── init.sql                        # PostgreSQL schema (6 tables)
├── src/
│   ├── ingestion/
│   │   ├── db.py                       # PostgreSQL connection helper
│   │   ├── producer.py                 # Kafka producer
│   │   ├── consumer.py                 # Kafka consumer
│   │   ├── bulk_load.py                # psycopg2 batch ingestion
│   │   ├── fast_load.py                # COPY-based cloud ingestion
│   │   └── load_suppliers.py           # supplier table population
│   ├── features/
│   │   └── feature_engineering.py      # 19-feature engineering pipeline
│   ├── models/
│   │   ├── train.py                    # XGBoost training + MLflow logging
│   │   ├── anomaly_detection.py        # Isolation Forest pipeline
│   │   ├── demand_forecast.py          # XGBoost demand forecasting
│   │   └── export_model.py             # export model from MLflow registry
│   └── api/
│       ├── main.py                     # FastAPI application + endpoints
│       ├── model.py                    # model loading and prediction logic
│       └── schemas.py                  # Pydantic request/response schemas
├── dashboard/
│   ├── app.py                          # 7-page Streamlit application
│   ├── queries.py                      # all PostgreSQL queries centralized
│   └── components/
│       ├── kpi_cards.py                # KPI summary cards
│       ├── risk_table.py               # supplier risk table with filters
│       └── alert_feed.py               # high-risk shipment alerts
├── .streamlit/
│   └── secrets.toml                    # local secrets (gitignored)
├── docker-compose.yml                  # Kafka + Zookeeper + PostgreSQL + Kafka UI
├── requirements.txt                    # full development dependencies
├── requirements_deploy.txt             # production dependencies (Render)
├── Procfile                            # Render start command
├── render.yaml                         # Render deployment configuration
├── runtime.txt                         # Python 3.11.9
└── .env.example                        # environment variable template
```

---

## 🚀 Local Setup

### Prerequisites

- Python 3.11
- Docker Desktop
- Git

### Step 1 — Clone and Setup

```bash
git clone https://github.com/Manideep-collab/supply-chain-intelligence.git
cd supply-chain-intelligence

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Step 2 — Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:
```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=supply_chain
POSTGRES_USER=scadmin
POSTGRES_PASSWORD=scpassword123

KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_SHIPMENTS=shipment-events

MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_MODEL_NAME=supply-chain-risk-model
MLFLOW_MODEL_VERSION=6

FASTAPI_URL=http://localhost:8000
```

### Step 3 — Start Infrastructure

```bash
docker-compose up -d
docker-compose ps
```

Four containers start: PostgreSQL (5432), Kafka (9092), Zookeeper (2181), Kafka UI (8080).

### Step 4 — Initialize Database and Load Data

```bash
# Create schema
python -c "
from src.ingestion.db import get_engine
from pathlib import Path
from sqlalchemy import text
engine = get_engine()
with engine.begin() as conn:
    conn.execute(text(Path('sql/init.sql').read_text()))
print('Schema created')
"

cd src/ingestion
python load_suppliers.py     # loads 164 supplier nodes
python bulk_load.py          # loads 180,519 shipments
```

### Step 5 — Run ML Pipeline

```bash
cd src/features
python feature_engineering.py   # builds 19-feature store

mlflow server --host 0.0.0.0 --port 5000   # start MLflow (keep running)

cd src/models
python train.py                 # trains XGBoost, logs 1 MLflow run
python anomaly_detection.py     # detects anomalous suppliers
python demand_forecast.py       # generates 5-week forecasts
```

### Step 6 — Start All Services

Open three terminals:

```bash
# Terminal 1
mlflow server --host 0.0.0.0 --port 5000

# Terminal 2
cd src/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3
streamlit run dashboard/app.py
```

### Step 7 — Access Everything

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Interactive Docs | http://localhost:8000/docs |
| MLflow Experiment Tracker | http://localhost:5000 |
| Kafka UI | http://localhost:8080 |

---

## ⚡ API Reference

### `POST /predict/risk`

Score a single shipment for disruption risk.

**High-risk request example:**
```json
{
  "transport_risk_score": 4,
  "mode_First Class": true,
  "mode_Same Day": false,
  "mode_Second Class": false,
  "mode_Standard Class": false,
  "reliability_score": 0.2,
  "supplier_late_rate": 0.8,
  "supplier_risk_index": 0.75,
  "supplier_composite_risk": 0.78,
  "country_risk_score": 0.7,
  "category_risk_score": 0.65,
  "category_encoded": 5,
  "quantity_log": 1.2,
  "is_bulk_order": 1,
  "order_month": 12,
  "order_dayofweek": 4,
  "order_quarter": 4,
  "is_month_end": 1,
  "is_q4": 1
}
```

**Response:**
```json
{
  "risk_score": 87.3,
  "risk_label": "CRITICAL",
  "late_probability": 0.873,
  "predicted_late": true,
  "top_risk_factors": [
    {"feature": "mode_First Class", "value": 1.0, "importance": 0.4827},
    {"feature": "transport_risk_score", "value": 4.0, "importance": 0.1650}
  ],
  "model_version": "supply-chain-risk-model/v6"
}
```

### Risk Labels

| Label | Probability Range | Recommended Action |
|---|---|---|
| 🟢 LOW | 0% – 35% | Standard monitoring |
| 🟡 MEDIUM | 35% – 55% | Monitor closely |
| 🟠 HIGH | 55% – 75% | Arrange contingency plan |
| 🔴 CRITICAL | 75% – 100% | Immediate action required |

### `POST /predict/risk/batch`
Score up to 1,000 shipments in a single request. Returns aggregate stats (total, high-risk count, average score).

### `GET /health`
Returns model load status, version, and feature count.

### `GET /model/info`
Returns model version, full feature list, risk thresholds, and performance metrics.

---

## 📈 Dashboard Pages

| Page | What It Shows |
|---|---|
| 📊 Executive Overview | 4 KPI cards, delivery status donut chart, transport risk bar chart, monthly late rate trend line, world risk heatmap, top category risk |
| 🏭 Supplier Risk Rankings | Searchable + filterable supplier table with progress bar columns, top 15 risk bar chart |
| 🚨 Alert Feed | Expandable cards for shipments with supplier_composite_risk > 0.65 or transport_risk_score = 4 |
| 🔮 Demand Forecast | 5-week forecast chart with 95% confidence bands, category selector dropdown, raw forecast table |
| 🔍 Anomaly Detection | Isolation Forest results bar chart, anomaly score progress bars, how-it-works explanation |
| 🎯 Risk Predictor | Live form calling FastAPI in real time — returns risk score, label, probability, top factors, recommendation |
| 🤖 Model Performance | Metrics, feature importance chart, all 6 MLflow run comparison table, model strengths and limitations |

---

## 🐳 Docker Compose Services

```yaml
zookeeper:   port 2181  — Kafka cluster management
kafka:       port 9092  — message broker
postgres:    port 5432  — primary database
kafka-ui:    port 8080  — visual Kafka monitoring
```

All data persists via Docker volumes — stops and restarts don't lose data.

---

## 🔑 Key Design Decisions

**Why Kafka instead of writing directly to PostgreSQL?**
Kafka decouples producers from consumers. The same shipment event can simultaneously feed the database, the ML model, and an alerting system independently. It also provides fault tolerance — if PostgreSQL goes down, Kafka holds messages and the consumer catches up automatically on restart. In production, the producer script would be replaced by Kafka connectors to ERP systems like SAP.

**Why XGBoost over linear models or deep learning?**
EDA showed near-zero linear correlations between raw numerical features and delay outcomes (`reliability_score` vs `delay_days`: -0.046). XGBoost captures non-linear interactions that linear models miss entirely. It also trains in seconds on 180K rows and provides feature importance for stakeholder explainability — critical in risk systems.

**Why Bayesian smoothing on risk scores?**
Countries with few shipments (Laos: 6 shipments, 100% late rate; Armenia: 5 shipments, 100% late rate) dominated rankings without smoothing. The formula `(n × observed + 50 × global_mean) / (n + 50)` pulls small-sample scores toward the global average of 54.8%, making rankings statistically meaningful.

**Why exclude destination market as a feature?**
EDA showed all 5 destination markets had late rates between 56.8% and 57.7% — less than 1% spread across the entire dataset. Including it would add noise without any predictive signal.

**Why scale_pos_weight=1.5?**
In supply chain operations, missing a disruption (False Negative) costs far more than investigating a false alarm (False Positive). Upweighting the late class by 1.5x pushes recall from 0.61 to 0.72 — catching 11% more real disruptions at the cost of acceptable precision reduction.

**Why XGBoost for demand forecasting instead of Prophet?**
Prophet requires C++ build tools (CmdStan) which are unavailable on Windows + Python 3.11. More importantly, using XGBoost for both classification and regression demonstrates understanding that the same algorithm family solves multiple problem types — a stronger portfolio story than mixing frameworks.

---

## 🗂️ Environment Variables

| Variable | Description | Example |
|---|---|---|
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | Database name | `supply_chain` |
| `POSTGRES_USER` | Database user | `scadmin` |
| `POSTGRES_PASSWORD` | Database password | `scpassword123` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker | `localhost:9092` |
| `KAFKA_TOPIC_SHIPMENTS` | Kafka topic | `shipment-events` |
| `MLFLOW_TRACKING_URI` | MLflow server | `http://localhost:5000` |
| `MLFLOW_MODEL_NAME` | Registered model name | `supply-chain-risk-model` |
| `MLFLOW_MODEL_VERSION` | Model version to serve | `6` |
| `FASTAPI_URL` | FastAPI base URL | `http://localhost:8000` |

---

## 👤 Author

**Manideep Palnati**
BBA — Artificial Intelligence & Data Science
Woxsen University, Hyderabad

[![GitHub](https://img.shields.io/badge/GitHub-Manideep--collab-black)](https://github.com/Manideep-collab)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
