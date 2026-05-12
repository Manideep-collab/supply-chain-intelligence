# Handles all model-related operations:
#   - Loading the trained XGBoost model from MLflow registry
#   - Running predictions
#   - Converting raw probabilities to risk scores and labels
#
# Completely separated from API logic — clean separation of concerns.
# If we ever swap XGBoost for a different model, only this file changes.

import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Feature Configuration ─────────────────────────────────────────────────────
# These must match EXACTLY what the model was trained on in Phase 4.
# Order matters — XGBoost expects features in the same order every time.

FEATURE_NAMES = [
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

# Risk thresholds — converting probability to business-readable labels
# These are business decisions, not ML decisions.
# You can adjust these based on how sensitive you want the system to be.
RISK_THRESHOLDS = {
    "LOW":      (0.0,  0.35),   # 0-35% probability → low risk
    "MEDIUM":   (0.35, 0.55),   # 35-55% → medium risk
    "HIGH":     (0.55, 0.75),   # 55-75% → high risk
    "CRITICAL": (0.75, 1.01),   # 75%+   → critical risk
}


# ── Model Loader ──────────────────────────────────────────────────────────────

class RiskModel:
    """
    Singleton class — loads the model once when the API starts,
    then reuses it for every prediction request.

    Why singleton: loading a model from MLflow takes 2-3 seconds.
    If we loaded it on every request, the API would be very slow.
    Load once at startup, serve thousands of requests instantly.
    """

    def __init__(self):
        self.model = None
        self.model_version = None
        self.is_loaded = False

    def load(self):
        """
        Load trained XGBoost model from exported file.
        No MLflow dependency in production.
        """
        try:
            import xgboost as xgb
            import json

            # Look for model file relative to project root
            # Works both locally and on Render
            base_dir   = os.path.dirname(
                os.path.dirname(os.path.dirname(__file__))
            )
            model_path   = os.path.join(base_dir, "models", "xgboost_model.json")
            feature_path = os.path.join(base_dir, "models", "feature_names.json")

            print(f"📦 Loading model from: {model_path}")

            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)

            # Load feature names
            with open(feature_path, "r") as f:
                global FEATURE_NAMES
                FEATURE_NAMES = json.load(f)

            self.model_version = "supply-chain-risk-model/v6"
            self.is_loaded     = True

            print(f"✅ Model loaded successfully")
            print(f"   Features: {len(FEATURE_NAMES)}")

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.is_loaded = False
            raise

    def _features_to_dataframe(self, features: dict) -> pd.DataFrame:
        """
        Convert incoming request dict into a DataFrame
        with columns in exactly the right order for XGBoost.

        XGBoost is strict about column order — it must match training.
        """
        # Handle the alias mapping for mode columns
        # API receives "mode_First Class" but pydantic stores as "mode_First_Class"
        # We map back to the original training column names
        alias_map = {
            "mode_First_Class":    "mode_First Class",
            "mode_Same_Day":       "mode_Same Day",
            "mode_Second_Class":   "mode_Second Class",
            "mode_Standard_Class": "mode_Standard Class",
        }

        normalized = {}
        for key, value in features.items():
            mapped_key = alias_map.get(key, key)
            normalized[mapped_key] = value

        # Build DataFrame with exact column order from training
        row = {}
        for feature in FEATURE_NAMES:
            row[feature] = normalized.get(feature, 0)

        df = pd.DataFrame([row])

        # Convert bool columns to int — XGBoost expects numbers
        bool_cols = [
            "mode_First Class", "mode_Same Day",
            "mode_Second Class", "mode_Standard Class"
        ]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)

        return df[FEATURE_NAMES]  # enforce exact order


    def _probability_to_risk(self, probability: float) -> tuple:
        """
        Convert raw model probability (0.0-1.0) to:
        - risk_score: 0-100 integer (easier for dashboards)
        - risk_label: LOW / MEDIUM / HIGH / CRITICAL

        This is a business translation layer — the model outputs
        a probability, but business users want a score and label.
        """
        risk_score = round(probability * 100, 2)

        # Determine label from thresholds
        risk_label = "LOW"
        for label, (low, high) in RISK_THRESHOLDS.items():
            if low <= probability < high:
                risk_label = label
                break

        return risk_score, risk_label


    def _get_top_risk_factors(self, features_df: pd.DataFrame) -> list:
        """
        Identify which features most contributed to this specific prediction.

        We use the model's feature importances × feature values
        to find which features pushed the risk score highest.
        This gives an explainability layer to every prediction.
        """
        importances = self.model.feature_importances_

        # Multiply importance by normalized feature value
        # Higher = this feature contributed more to this prediction
        contributions = []
        for i, feature in enumerate(FEATURE_NAMES):
            value      = float(features_df.iloc[0][feature])
            importance = float(importances[i])
            contribution = importance * abs(value)
            contributions.append({
                "feature":      feature,
                "value":        round(value, 4),
                "importance":   round(importance, 4),
                "contribution": round(contribution, 4),
            })

        # Sort by contribution, return top 5
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        return contributions[:5]


    def predict(self, features: dict) -> dict:
        """
        Main prediction function.
        Takes a dict of feature values, returns full risk assessment.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert to DataFrame
        features_df = self._features_to_dataframe(features)

        # Get probability of late delivery
        # predict_proba returns [[prob_on_time, prob_late]]
        # We take [:, 1] for probability of being late
        probabilities  = self.model.predict_proba(features_df)
        late_prob      = float(probabilities[0][1])
        predicted_late = bool(self.model.predict(features_df)[0])

        # Convert to risk score and label
        risk_score, risk_label = self._probability_to_risk(late_prob)

        # Get top contributing features
        top_factors = self._get_top_risk_factors(features_df)

        return {
            "risk_score":       risk_score,
            "risk_label":       risk_label,
            "late_probability": round(late_prob, 4),
            "predicted_late":   predicted_late,
            "top_risk_factors": top_factors,
            "model_version":    self.model_version,
        }


    def predict_batch(self, features_list: list) -> list:
        """
        Score multiple shipments in one call.
        More efficient than calling predict() in a loop
        because XGBoost can vectorize batch predictions.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build batch DataFrame
        rows = []
        for features in features_list:
            df = self._features_to_dataframe(features)
            rows.append(df.iloc[0])

        batch_df = pd.DataFrame(rows)[FEATURE_NAMES]

        # Batch predictions — one call for all rows
        probabilities = self.model.predict_proba(batch_df)
        predictions   = self.model.predict(batch_df)

        results = []
        for i in range(len(features_list)):
            late_prob      = float(probabilities[i][1])
            predicted_late = bool(predictions[i])
            risk_score, risk_label = self._probability_to_risk(late_prob)
            top_factors    = self._get_top_risk_factors(
                batch_df.iloc[[i]]
            )

            results.append({
                "risk_score":       risk_score,
                "risk_label":       risk_label,
                "late_probability": round(late_prob, 4),
                "predicted_late":   predicted_late,
                "top_risk_factors": top_factors,
                "model_version":    self.model_version,
            })

        return results


# Single instance — created once, reused everywhere
# This is the object that main.py imports and uses
risk_model = RiskModel()