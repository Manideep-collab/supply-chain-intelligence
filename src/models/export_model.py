# Exports the trained XGBoost model from MLflow registry
# to a standalone file for production deployment.
#
# Why we do this:
# MLflow is a development/experiment tracking tool.
# In production we don't want FastAPI depending on a running
# MLflow server. We export the model once and serve it directly.

import os
import json
import mlflow
import mlflow.xgboost
from dotenv import load_dotenv

load_dotenv()

def export_model():
    """
    Load model from MLflow registry and save as standalone files.
    Creates two files:
      models/xgboost_model.json  — the trained XGBoost model
      models/feature_names.json  — feature list in correct order
    """

    print("📦 Connecting to MLflow...")
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    model_name    = os.getenv("MLFLOW_MODEL_NAME", "supply-chain-risk-model")
    model_version = os.getenv("MLFLOW_MODEL_VERSION", "6")
    model_uri     = f"models:/{model_name}/{model_version}"

    print(f"📥 Loading: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Save XGBoost model as JSON — portable, no MLflow needed
    model_path = "models/xgboost_model.json"
    model.save_model(model_path)
    print(f"✅ Model saved: {model_path}")

    # Save feature names — API needs these to build input correctly
    feature_names = [
        "transport_risk_score",
        "reliability_score",
        "supplier_late_rate",
        "supplier_risk_index",
        "supplier_composite_risk",
        "country_risk_score",
        "category_risk_score",
        "category_encoded",
        "quantity_log",
        "is_bulk_order",
        "order_month",
        "order_dayofweek",
        "order_quarter",
        "is_month_end",
        "is_q4",
        "mode_First Class",
        "mode_Same Day",
        "mode_Second Class",
        "mode_Standard Class",
    ]

    feature_path = "models/feature_names.json"
    with open(feature_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"✅ Features saved: {feature_path}")

    print("\n🎯 Model export complete")
    print("   models/ folder is ready for deployment")


if __name__ == "__main__":
    export_model()