# Purpose: Train an XGBoost disruption risk model on our
# feature_store data, track everything with MLflow, and
# save the best model for serving in Phase 5 (FastAPI).

# What this script does step by step:
#   1. Load feature_store from PostgreSQL
#   2. Split into train/test sets
#   3. Train XGBoost with cross-validation
#   4. Log all params + metrics to MLflow
#   5. Save trained model as MLflow artifact
#   6. Print feature importance ranking

import os
import sys
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '../ingestion'))
from db import get_engine

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# All hyperparameters in one place — easy to experiment with
# ─────────────────────────────────────────────────────────────

CONFIG = {
    # XGBoost hyperparameters
    # n_estimators: how many trees to build
    # More trees = more learning but slower and can overfit
    "n_estimators": 500,

    # max_depth: how deep each tree can grow
    # Deeper = learns more complex patterns but can overfit
    "max_depth": 8,

    # learning_rate: how much each new tree corrects previous errors
    # Lower = more careful learning, needs more trees
    "learning_rate": 0.05,

    # subsample: fraction of training data used per tree
    # 0.8 = each tree only sees 80% of data — reduces overfitting
    "subsample": 0.8,

    # colsample_bytree: fraction of features used per tree
    # 0.8 = each tree only sees 80% of features — reduces overfitting
    "colsample_bytree": 0.8,

    # min_child_weight: minimum data points needed in a leaf node
    # Higher = more conservative model
    "min_child_weight": 1,

    # gamma: minimum loss reduction to make a split
    # Higher = more conservative — only splits if it significantly helps
    "gamma": 0.0,

    # scale_pos_weight: handles class imbalance
    # Our classes are nearly balanced (54.8/45.2) so we use 1.0
    "scale_pos_weight": 1.5,

    # Experiment settings
    "test_size": 0.2,        # 80% train, 20% test
    "random_state": 42,      # for reproducibility
    "cv_folds": 5,           # 5-fold cross validation
    "experiment_name": "supply-chain-disruption-risk",
}

# STEP 1: LOAD DATA FROM FEATURE STORE

def load_features(engine):
    """
    Load the feature_store table we built in Phase 3.
    This is our complete, engineered, ML-ready dataset.
    """
    query = """
        SELECT * FROM feature_store
    """
    df = pd.read_sql(query, engine)
    print(f"✅ Loaded feature_store: {len(df):,} rows × {df.shape[1]} columns")
    return df


def prepare_xy(df):
    """
    Separate features (X) from target (y).
    Drop columns that are IDs or raw strings — not useful to the model.
    """
    # Columns the model should NOT see
    drop_cols = [
        'shipment_id',        # just an ID
        'supplier_id',        # string ID — risk scores capture this info
        'origin_country',     # string — country_risk_score captures this
        'destination_country',# string — excluded (EDA: market risk useless)
        'transport_mode',     # string — transport_risk_score captures this
        'is_late',            # this is our TARGET — not a feature
        'is_on_time',
        'is_early',
        'delay_bucket',
    ]

    # Only drop columns that actually exist
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df['is_late']

    # Fill any nulls — shouldn't exist but safe to handle
    X = X.fillna(0)

    print(f"✅ Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"✅ Target balance: {y.sum():,} late ({y.mean()*100:.1f}%) | "
          f"{(~y.astype(bool)).sum():,} on time ({(1-y.mean())*100:.1f}%)")

    return X, y


# STEP 2: TRAIN MODEL

def train_model(X_train, y_train, config):
    """
    Train XGBoost classifier with the config parameters.

    eval_set allows XGBoost to monitor validation loss during training
    and stop early if it stops improving — prevents overfitting.
    """

    model = xgb.XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        min_child_weight=config["min_child_weight"],
        gamma=config["gamma"],
        scale_pos_weight=config["scale_pos_weight"],
        random_state=config["random_state"],
        eval_metric="logloss",  # log loss for binary classification
        early_stopping_rounds=20,  # stop if no improvement for 20 rounds
        verbosity=0,
    )

    # Split train into train + validation for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=config["random_state"],
        stratify=y_train
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    print(f"✅ Model trained | Best iteration: {model.best_iteration}")
    return model


# STEP 3: EVALUATE MODEL

def evaluate_model(model, X_test, y_test):
    """
    Calculate all evaluation metrics on the held-out test set.

    Why these metrics:
    - Accuracy: overall % correct — can be misleading with imbalance
    - Precision: of shipments flagged as late, how many actually were?
    - Recall: of actually late shipments, how many did we catch?
    - F1: harmonic mean of precision and recall — balanced metric
    - AUC-ROC: model's ability to distinguish late vs on-time
               0.5 = random guessing, 1.0 = perfect
    """

    # Class predictions (0 or 1)
    y_pred = model.predict(X_test)

    # Probability predictions (0.0 to 1.0)
    # We use [:, 1] to get probability of class 1 (late)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    return metrics, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred, save_path):
    """
    Confusion matrix tells us exactly where the model makes mistakes.

    True Positive  (TP): predicted late,    actually late    ✅
    True Negative  (TN): predicted on time, actually on time ✅
    False Positive (FP): predicted late,    actually on time ❌ (false alarm)
    False Negative (FN): predicted on time, actually late    ❌ (missed risk)

    For supply chain risk, False Negatives are more costly —
    missing a disruption is worse than a false alarm.
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['On Time', 'Late'],
        yticklabels=['On Time', 'Late']
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Confusion matrix saved: {save_path}")


def plot_feature_importance(model, feature_names, save_path):
    """
    Feature importance shows which features the model relied on most.
    This is one of the most valuable outputs for stakeholders —
    it explains WHY the model makes its predictions.
    """
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp['feature'], feat_imp['importance'],
             color='steelblue')
    plt.title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Feature importance saved: {save_path}")

    # Print top 10
    print("\n📊 Top 10 Most Important Features:")
    top10 = feat_imp.sort_values('importance', ascending=False).head(10)
    for _, row in top10.iterrows():
        bar = "█" * int(row['importance'] * 100)
        print(f"   {row['feature']:<30} {bar} {row['importance']:.4f}")


# STEP 4: CROSS VALIDATION

def cross_validate_model(X, y, config):
    """
    Cross-validation trains the model on 5 different splits
    and averages the results. This gives you a much more reliable
    estimate of real-world performance than a single train/test split.

    StratifiedKFold ensures each fold has the same class balance
    as the full dataset — important for classification.
    """
    print(f"\n🔄 Running {config['cv_folds']}-fold cross validation...")

    skf = StratifiedKFold(
        n_splits=config['cv_folds'],
        shuffle=True,
        random_state=config['random_state']
    )

    cv_scores = {
        'accuracy': [], 'precision': [],
        'recall': [], 'f1': [], 'auc_roc': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_fold_train = X.iloc[train_idx]
        X_fold_val   = X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_val   = y.iloc[val_idx]

        # Train fold model
        fold_model = xgb.XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            min_child_weight=config["min_child_weight"],
            gamma=config["gamma"],
            random_state=config["random_state"],
            eval_metric="logloss",
            verbosity=0,
        )
        fold_model.fit(X_fold_train, y_fold_train, verbose=False)

        # Evaluate fold
        y_pred = fold_model.predict(X_fold_val)
        y_prob = fold_model.predict_proba(X_fold_val)[:, 1]

        cv_scores['accuracy'].append(accuracy_score(y_fold_val, y_pred))
        cv_scores['precision'].append(precision_score(y_fold_val, y_pred))
        cv_scores['recall'].append(recall_score(y_fold_val, y_pred))
        cv_scores['f1'].append(f1_score(y_fold_val, y_pred))
        cv_scores['auc_roc'].append(roc_auc_score(y_fold_val, y_prob))

        print(f"   Fold {fold}: "
              f"Acc={cv_scores['accuracy'][-1]:.4f} | "
              f"F1={cv_scores['f1'][-1]:.4f} | "
              f"AUC={cv_scores['auc_roc'][-1]:.4f}")

    # Average across all folds
    cv_means = {f"cv_{k}_mean": round(np.mean(v), 4)
                for k, v in cv_scores.items()}
    cv_stds  = {f"cv_{k}_std":  round(np.std(v), 4)
                for k, v in cv_scores.items()}

    print(f"\n✅ CV Results (mean ± std):")
    for metric in ['accuracy', 'f1', 'auc_roc']:
        mean = cv_means[f'cv_{metric}_mean']
        std  = cv_stds[f'cv_{metric}_std']
        print(f"   {metric:<12}: {mean:.4f} ± {std:.4f}")

    return {**cv_means, **cv_stds}


# MAIN — Full Training Pipeline with MLflow

def main():
    print("🚀 Starting Phase 4 — XGBoost Training Pipeline\n")

    # Create output directory for plots
    os.makedirs("data/model_outputs", exist_ok=True)

    engine = get_engine()

    # ── Load and prepare data ─────────────────────────────────
    df = load_features(engine)
    X, y = prepare_xy(df)
    feature_names = list(X.columns)

    # ── Train/test split ──────────────────────────────────────
    # stratify=y ensures both splits have same class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y
    )
    print(f"\n📊 Split: {len(X_train):,} train | {len(X_test):,} test")

    # ── MLflow experiment setup ───────────────────────────────
    # Set tracking URI — where MLflow server is running
    mlflow.set_tracking_uri("http://localhost:5000")

    # Set experiment name — groups all related runs together
    mlflow.set_experiment(CONFIG["experiment_name"])

    # ── Cross validation (outside MLflow run) ─────────────────
    cv_metrics = cross_validate_model(X_train, y_train, CONFIG)

    # ── MLflow run — everything inside gets tracked ───────────
    # Think of this as: "start recording this experiment"
    with mlflow.start_run(run_name="xgboost_final_v5"):

        print("\n📝 MLflow run started — tracking everything...\n")

        # Log all hyperparameters
        # These appear in the MLflow UI so you can compare runs
        mlflow.log_params({
            "n_estimators":      CONFIG["n_estimators"],
            "max_depth":         CONFIG["max_depth"],
            "learning_rate":     CONFIG["learning_rate"],
            "subsample":         CONFIG["subsample"],
            "colsample_bytree":  CONFIG["colsample_bytree"],
            "min_child_weight":  CONFIG["min_child_weight"],
            "gamma":             CONFIG["gamma"],
            "test_size":         CONFIG["test_size"],
            "train_rows":        len(X_train),
            "test_rows":         len(X_test),
            "n_features":        len(feature_names),
            "features":          str(feature_names),
        })

        # ── Train final model ─────────────────────────────────
        print("🔧 Training final model on full training set...")
        model = train_model(X_train, y_train, CONFIG)

        # ── Evaluate on test set ──────────────────────────────
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

        print("\n📊 Test Set Results:")
        print(f"   Accuracy  : {metrics['accuracy']}")
        print(f"   Precision : {metrics['precision']}")
        print(f"   Recall    : {metrics['recall']}")
        print(f"   F1 Score  : {metrics['f1_score']}")
        print(f"   AUC-ROC   : {metrics['auc_roc']}")

        # ── Log all metrics to MLflow ─────────────────────────
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(cv_metrics)

        # ── Generate and log plots ────────────────────────────
        cm_path = "data/model_outputs/confusion_matrix.png"
        fi_path = "data/model_outputs/feature_importance.png"

        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_feature_importance(model, feature_names, fi_path)

        # Log plots as MLflow artifacts
        # Artifacts are files attached to a run — viewable in UI
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)

        # ── Log classification report ─────────────────────────
        report = classification_report(y_test, y_pred,
                                       target_names=['On Time', 'Late'])
        print(f"\n📋 Classification Report:\n{report}")

        # Save report as text artifact
        report_path = "data/model_outputs/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(f"XGBoost Disruption Risk Model\n")
            f.write(f"{'='*40}\n\n")
            f.write(f"Config:\n{json.dumps(CONFIG, indent=2)}\n\n")
            f.write(f"Classification Report:\n{report}\n\n")
            f.write(f"CV Metrics:\n{json.dumps(cv_metrics, indent=2)}")
        mlflow.log_artifact(report_path)

        # ── Save model to MLflow ──────────────────────────────
        # This saves the trained model so we can load it later
        # in FastAPI (Phase 5) without retraining
        mlflow.xgboost.log_model(
            model,
            artifact_path="xgboost_model",
            registered_model_name="supply-chain-risk-model"
        )
        print("\n✅ Model saved to MLflow model registry")

        # ── Save feature names for serving ───────────────────
        feature_path = "data/model_outputs/feature_names.json"
        with open(feature_path, "w") as f:
            json.dump(feature_names, f)
        mlflow.log_artifact(feature_path)

        # Get the run ID — needed to load model in Phase 5
        run_id = mlflow.active_run().info.run_id
        print(f"\n🔑 MLflow Run ID: {run_id}")
        print(f"   Save this — needed for Phase 5 model serving")

        # Save run ID to file for Phase 5
        with open("data/model_outputs/run_id.txt", "w") as f:
            f.write(run_id)

    print("\n🎯 Training complete!")
    print(f"   View results at: http://localhost:5000")
    print(f"   Experiment: {CONFIG['experiment_name']}")


if __name__ == "__main__":
    main()