# main.py
# Location: src/api/main.py
#
# The FastAPI application.
# Defines all API endpoints and ties schemas + model together.
#
# Endpoints:
#   GET  /health              → check if API and model are alive
#   GET  /docs                → auto-generated interactive documentation
#   POST /predict/risk        → score a single shipment
#   POST /predict/risk/batch  → score multiple shipments at once
#   GET  /model/info          → model metadata and feature list

import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '../ingestion'))

from schemas import (
    ShipmentFeatures,
    RiskPrediction,
    HealthResponse,
    BatchShipmentRequest,
    BatchRiskResponse,
)
from model import risk_model, FEATURE_NAMES


# ── Startup & Shutdown ────────────────────────────────────────────────────────
# lifespan handles what happens when FastAPI starts and stops.
# We load the model on startup so it's ready for the first request.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — runs before API accepts any requests
    print("🚀 Starting Supply Chain Risk API...")
    print("📦 Loading model from MLflow...")
    risk_model.load()
    print("✅ API ready to serve predictions\n")
    yield
    # Shutdown — runs when API is stopped
    print("👋 Supply Chain Risk API shutting down")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Supply Chain Disruption Risk API",
    description="""
## Supply Chain Disruption Intelligence Platform

Real-time disruption risk scoring for supply chain shipments.

### What This API Does
- Scores individual shipments with a 0-100 disruption risk score
- Classifies risk as LOW / MEDIUM / HIGH / CRITICAL
- Explains which factors drove the risk score
- Supports batch scoring for multiple shipments at once

### Model Details
- Algorithm: XGBoost Classifier
- Training data: 180,519 shipments
- Features: 19 pre-shipment signals
- AUC-ROC: 0.7562 | Recall: 0.72 | Precision: 0.70

### Risk Labels
| Label | Probability | Meaning |
|-------|-------------|---------|
| LOW | 0-35% | Shipment likely on time |
| MEDIUM | 35-55% | Monitor closely |
| HIGH | 55-75% | Arrange contingency |
| CRITICAL | 75%+ | Immediate action needed |
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allows the Streamlit dashboard (Phase 6)
# to call this API from a different port without being blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, restrict to your dashboard URL
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check if the API is alive and the model is loaded.
    Used by monitoring systems and Docker health checks.
    """
    return HealthResponse(
        status="healthy" if risk_model.is_loaded else "degraded",
        model_loaded=risk_model.is_loaded,
        model_version=risk_model.model_version or "not loaded",
        features_count=len(FEATURE_NAMES),
    )


@app.get("/model/info", tags=["System"])
async def model_info():
    """
    Returns metadata about the loaded model.
    Useful for debugging and documentation.
    """
    if not risk_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check MLflow server."
        )

    return {
        "model_version":  risk_model.model_version,
        "features":       FEATURE_NAMES,
        "features_count": len(FEATURE_NAMES),
        "risk_labels":    ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "risk_thresholds": {
            "LOW":      "0% - 35%",
            "MEDIUM":   "35% - 55%",
            "HIGH":     "55% - 75%",
            "CRITICAL": "75% - 100%",
        },
        "model_metrics": {
            "auc_roc":   0.7562,
            "recall":    0.7194,
            "precision": 0.6969,
            "f1_score":  0.7079,
        }
    }


@app.post("/predict/risk", response_model=RiskPrediction, tags=["Predictions"])
async def predict_risk(shipment: ShipmentFeatures):
    """
    Score a single shipment for disruption risk.

    Send the 19 pre-shipment features, receive:
    - risk_score: 0-100
    - risk_label: LOW/MEDIUM/HIGH/CRITICAL
    - late_probability: raw model probability
    - top_risk_factors: which features drove this score
    """
    if not risk_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check MLflow server is running."
        )

    try:
        # Convert Pydantic model to dict
        # by_alias=True preserves "mode_First Class" format
        features_dict = shipment.model_dump(by_alias=True)

        # Run prediction
        result = risk_model.predict(features_dict)
        return RiskPrediction(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/risk/batch",
    response_model=BatchRiskResponse,
    tags=["Predictions"]
)
async def predict_risk_batch(request: BatchShipmentRequest):
    """
    Score multiple shipments in a single request.
    More efficient than calling /predict/risk repeatedly.
    Maximum 1000 shipments per batch.
    """
    if not risk_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check MLflow server is running."
        )

    if len(request.shipments) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 1000 shipments."
        )

    if len(request.shipments) == 0:
        raise HTTPException(
            status_code=400,
            detail="Batch must contain at least 1 shipment."
        )

    try:
        # Convert all shipments to dicts
        features_list = [
            s.model_dump(by_alias=True)
            for s in request.shipments
        ]

        # Batch predict
        results = risk_model.predict_batch(features_list)
        predictions = [RiskPrediction(**r) for r in results]

        # Aggregate stats
        high_risk_count = sum(
            1 for p in predictions
            if p.risk_label in ["HIGH", "CRITICAL"]
        )
        avg_risk = float(np.mean([p.risk_score for p in predictions]))

        return BatchRiskResponse(
            predictions=predictions,
            total_shipments=len(predictions),
            high_risk_count=high_risk_count,
            average_risk_score=round(avg_risk, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint — confirms API is running."""
    return {
        "message":     "Supply Chain Disruption Risk API",
        "version":     "1.0.0",
        "docs":        "/docs",
        "health":      "/health",
        "predict":     "/predict/risk",
    }