# Pydantic schemas define the exact shape of data coming into
# and going out of our API.

# Why this matters:
# - FastAPI automatically validates every incoming request against these
# - If a field is missing or wrong type, API returns a clear error
#   before the model even sees the data
# - Acts as live documentation — anyone calling the API knows
#   exactly what to send
#
# Every field maps directly to our 19 model features from feature_names.json

from pydantic import BaseModel, Field
from typing import Optional


# ── INPUT SCHEMA ──────────────────────────────────────────────────────────────
# These are the 19 features our XGBoost model was trained on.
# The API caller sends these values, we run prediction, return risk score.
#
# Field(...) means required — no default, must be provided
# Field(default) means optional — uses default if not provided

class ShipmentFeatures(BaseModel):
    """
    Input features for disruption risk prediction.
    All 19 fields match exactly what XGBoost was trained on.
    """

    # Transport mode features
    # transport_risk_score: ordinal encoding from EDA
    # 1=Standard Class (safest), 4=First Class (riskiest — 100% late rate)
    transport_risk_score: int = Field(
        ...,
        ge=1, le=4,
        description="Transport mode risk: 1=Standard, 2=Same Day, 3=Second, 4=First Class"
    )

    # One-hot encoded transport modes
    # Exactly one of these should be True, rest False
    mode_First_Class: bool = Field(..., alias="mode_First Class")
    mode_Same_Day: bool = Field(..., alias="mode_Same Day")
    mode_Second_Class: bool = Field(..., alias="mode_Second Class")
    mode_Standard_Class: bool = Field(..., alias="mode_Standard Class")

    # Supplier risk features
    reliability_score: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Supplier reliability: 0=worst, 1=best"
    )
    supplier_late_rate: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Historical late delivery rate for this supplier"
    )
    supplier_risk_index: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Bayesian smoothed supplier risk score"
    )
    supplier_composite_risk: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Combined supplier risk index"
    )

    # Geographic risk
    country_risk_score: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Bayesian smoothed country-level risk score"
    )

    # Product features
    category_risk_score: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Bayesian smoothed product category risk"
    )
    category_encoded: int = Field(
        ...,
        ge=0,
        description="Label encoded product category"
    )

    # Order size features
    quantity_log: float = Field(
        ...,
        ge=0.0,
        description="Log-transformed order quantity"
    )
    is_bulk_order: int = Field(
        ...,
        ge=0, le=1,
        description="1 if order quantity is in top 25 percentile"
    )

    # Time/seasonality features
    order_month: int = Field(
        ...,
        ge=1, le=12,
        description="Month of order (1-12)"
    )
    order_dayofweek: int = Field(
        ...,
        ge=0, le=6,
        description="Day of week (0=Monday, 6=Sunday)"
    )
    order_quarter: int = Field(
        ...,
        ge=1, le=4,
        description="Quarter of year (1-4)"
    )
    is_month_end: int = Field(
        ...,
        ge=0, le=1,
        description="1 if order placed on day >= 25 of month"
    )
    is_q4: int = Field(
        ...,
        ge=0, le=1,
        description="1 if order placed in Q4 (Oct-Dec)"
    )

    class Config:
        # Allow both "mode_First Class" and "mode_First_Class" as field names
        populate_by_name = True


# ── OUTPUT SCHEMA ─────────────────────────────────────────────────────────────

class RiskPrediction(BaseModel):
    """
    Response returned by POST /predict/risk
    Contains the risk score, label, and confidence.
    """

    # Risk score 0-100 (converted from model's 0.0-1.0 probability)
    risk_score: float = Field(
        description="Disruption risk score: 0 (safe) to 100 (certain disruption)"
    )

    # Human readable label
    risk_label: str = Field(
        description="LOW / MEDIUM / HIGH / CRITICAL"
    )

    # Raw probability from model
    late_probability: float = Field(
        description="Raw model probability of late delivery (0.0 to 1.0)"
    )

    # Binary prediction
    predicted_late: bool = Field(
        description="True if model predicts late delivery"
    )

    # Which features drove this prediction most
    top_risk_factors: list = Field(
        description="Top features contributing to this risk score"
    )

    # Model metadata
    model_version: str = Field(
        description="Model version used for this prediction"
    )


class HealthResponse(BaseModel):
    """Response for GET /health endpoint."""
    status: str
    model_loaded: bool
    model_version: str
    features_count: int


class BatchShipmentRequest(BaseModel):
    """
    For scoring multiple shipments at once.
    More efficient than calling /predict/risk one by one.
    """
    shipments: list[ShipmentFeatures]


class BatchRiskResponse(BaseModel):
    """Response for batch prediction endpoint."""
    predictions: list[RiskPrediction]
    total_shipments: int
    high_risk_count: int
    average_risk_score: float