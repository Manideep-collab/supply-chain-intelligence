# Demand forecasting using XGBoost Regression.
# Approach:
#   1. Aggregate demand signals to weekly totals per category
#   2. Build time-series features (lag features, rolling stats, date features)
#   3. Train XGBoostRegressor on historical weekly demand
#   4. Predict next 30 days (4-5 weeks ahead)
#   5. Generate confidence intervals using prediction residuals
#
# Why XGBoost over Prophet:
#   - Prophet requires C++ compilation (fails on Windows + Python 3.11)
#   - XGBoost handles time-series well via lag features
#   - Consistent with rest of project stack
#   - Easier to explain and extend

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '../ingestion'))
from db import get_engine


# ── Configuration ─────────────────────────────────────────────────────────────

TOP_CATEGORIES = [
    'Cleats',
    "Women's Apparel",
    'Indoor/Outdoor Games',
    'Cardio Equipment',
    'Shop By Sport',
]

FORECAST_WEEKS = 5    # how many weeks ahead to forecast
FREQ           = 'W'  # weekly aggregation


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_weekly_demand(engine, category: str) -> pd.DataFrame:
    """
    Load demand signals for one category and aggregate to weekly totals.
    Weekly smooths the noisy daily signal (quantities are mostly 1-3).
    """
    query = """
        SELECT
            signal_date,
            SUM(demand_quantity) AS total_demand
        FROM demand_signals
        WHERE product_category = %(category)s
          AND signal_date IS NOT NULL
        GROUP BY signal_date
        ORDER BY signal_date
    """
    df = pd.read_sql(query, engine, params={'category': category})
    df['signal_date'] = pd.to_datetime(df['signal_date'])

    # Resample to weekly — fills gaps, smooths noise
    df = df.set_index('signal_date')
    df = df.resample(FREQ).sum()
    df = df.reset_index()
    df = df[df['total_demand'] > 0]  # drop empty weeks

    return df


# ── Feature Engineering for Time Series ──────────────────────────────────────

def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features that capture time-series patterns.

    Lag features: demand from N weeks ago
    - lag_1: last week's demand (strongest predictor)
    - lag_2: 2 weeks ago
    - lag_4: 4 weeks ago (monthly pattern)
    - lag_52: same week last year (yearly seasonality)

    Rolling features: recent trend
    - rolling_mean_4: average of last 4 weeks
    - rolling_std_4: volatility of last 4 weeks
    - rolling_mean_12: average of last 12 weeks (quarterly trend)

    Date features: calendar effects
    - month: captures seasonal patterns
    - quarter: Q4 = holiday season = high demand
    - week_of_year: fine-grained seasonality
    """
    df = df.copy()
    df = df.sort_values('signal_date').reset_index(drop=True)

    # Lag features
    df['lag_1']  = df['total_demand'].shift(1)
    df['lag_2']  = df['total_demand'].shift(2)
    df['lag_4']  = df['total_demand'].shift(4)
    df['lag_52'] = df['total_demand'].shift(52)  # same week last year

    # Rolling statistics
    df['rolling_mean_4']  = df['total_demand'].shift(1).rolling(4).mean()
    df['rolling_std_4']   = df['total_demand'].shift(1).rolling(4).std()
    df['rolling_mean_12'] = df['total_demand'].shift(1).rolling(12).mean()

    # Trend: difference from 4 weeks ago
    df['trend_4w'] = df['lag_1'] - df['lag_4']

    # Date features
    df['month']        = df['signal_date'].dt.month
    df['quarter']      = df['signal_date'].dt.quarter
    df['week_of_year'] = df['signal_date'].dt.isocalendar().week.astype(int)
    df['is_q4']        = (df['quarter'] == 4).astype(int)
    df['is_month_end'] = (df['signal_date'].dt.day >= 25).astype(int)

    # Drop rows where lag features are NaN (first N rows)
    df = df.dropna()

    return df


FEATURE_COLS = [
    'lag_1', 'lag_2', 'lag_4', 'lag_52',
    'rolling_mean_4', 'rolling_std_4', 'rolling_mean_12',
    'trend_4w',
    'month', 'quarter', 'week_of_year', 'is_q4', 'is_month_end',
]


# ── Model Training ────────────────────────────────────────────────────────────

def train_forecast_model(df: pd.DataFrame):
    """
    Train XGBoostRegressor on historical weekly demand.

    Uses last 20% of data as validation set — time-based split.
    We split by time not randomly because:
      - Random split would let the model "see the future" during training
      - Time-based split simulates real forecasting conditions
    """
    df_feat = build_time_features(df)

    X = df_feat[FEATURE_COLS]
    y = df_feat['total_demand']

    # Time-based train/val split — last 20% as validation
    split_idx = int(len(df_feat) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Calculate residuals on validation set for confidence intervals
    val_preds = model.predict(X_val)
    residuals = y_val.values - val_preds
    residual_std = np.std(residuals)

    # Validation metrics
    mae  = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    return model, df_feat, residual_std, mae, rmse


# ── Forecasting ───────────────────────────────────────────────────────────────

def generate_forecast(
    model,
    df_feat: pd.DataFrame,
    residual_std: float,
    n_weeks: int = FORECAST_WEEKS
) -> pd.DataFrame:
    """
    Generate multi-step forecast by iterating week by week.

    For each future week:
    1. Build features using the last known values
    2. Predict demand
    3. Add prediction to history so next week can use it as lag_1

    This is called "recursive forecasting" — each prediction
    feeds into the next one as a lag feature.
    """
    # Start with the most recent data
    history = df_feat.copy()
    last_date = history['signal_date'].max()

    forecasts = []

    for i in range(1, n_weeks + 1):
        # Next forecast date
        next_date = last_date + pd.Timedelta(weeks=i)

        # Build features for next week using recent history
        recent = history.tail(52)  # last year of data for lag_52

        lag_1  = float(history['total_demand'].iloc[-1])
        lag_2  = float(history['total_demand'].iloc[-2]) if len(history) >= 2 else lag_1
        lag_4  = float(history['total_demand'].iloc[-4]) if len(history) >= 4 else lag_1
        lag_52 = float(history['total_demand'].iloc[-52]) if len(history) >= 52 else lag_1

        roll4  = float(history['total_demand'].tail(4).mean())
        roll4s = float(history['total_demand'].tail(4).std()) if len(history) >= 4 else 0
        roll12 = float(history['total_demand'].tail(12).mean())
        trend  = lag_1 - lag_4

        row = {
            'lag_1':           lag_1,
            'lag_2':           lag_2,
            'lag_4':           lag_4,
            'lag_52':          lag_52,
            'rolling_mean_4':  roll4,
            'rolling_std_4':   roll4s,
            'rolling_mean_12': roll12,
            'trend_4w':        trend,
            'month':           next_date.month,
            'quarter':         next_date.quarter,
            'week_of_year':    next_date.isocalendar()[1],
            'is_q4':           int(next_date.quarter == 4),
            'is_month_end':    int(next_date.day >= 25),
        }

        X_next = pd.DataFrame([row])[FEATURE_COLS]
        predicted = float(model.predict(X_next)[0])
        predicted = max(0, predicted)  # clip negatives

        # 95% confidence interval using residual std
        # 1.96 = z-score for 95% confidence
        lower = max(0, predicted - 1.96 * residual_std)
        upper = predicted + 1.96 * residual_std

        forecasts.append({
            'forecast_date': next_date.date(),
            'predicted_qty': round(predicted, 0),
            'lower_bound':   round(lower, 0),
            'upper_bound':   round(upper, 0),
        })

        # Add prediction to history for next iteration
        new_row = pd.DataFrame([{
            'signal_date':   next_date,
            'total_demand':  predicted,
            **row
        }])
        history = pd.concat([history, new_row], ignore_index=True)

    return pd.DataFrame(forecasts)


# ── Save Results ──────────────────────────────────────────────────────────────

def save_forecasts(all_forecasts: pd.DataFrame, engine):
    """Save forecast results to PostgreSQL."""
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE demand_forecasts"))

    all_forecasts.to_sql(
        'demand_forecasts',
        engine,
        if_exists='append',
        index=False,
        method='multi',
    )
    print(f"\n✅ Saved {len(all_forecasts)} forecast records to PostgreSQL")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔮 Starting XGBoost Demand Forecasting Pipeline\n")
    engine = get_engine()

    all_forecasts = []

    for category in TOP_CATEGORIES:
        try:
            print(f"📈 Forecasting: {category}")

            # Load data
            df = load_weekly_demand(engine, category)
            print(f"   History: {len(df)} weeks")

            if len(df) < 60:
                print(f"   ⚠️  Skipping — insufficient history")
                continue

            # Train model
            model, df_feat, residual_std, mae, rmse = train_forecast_model(df)
            print(f"   MAE={mae:.1f} | RMSE={rmse:.1f} | "
                  f"Residual STD={residual_std:.1f}")

            # Generate forecast
            forecast_df = generate_forecast(model, df_feat, residual_std)
            forecast_df['product_category'] = category
            all_forecasts.append(forecast_df)

            # Print preview
            first = forecast_df.iloc[0]
            print(f"   ✅ Next week forecast: "
                  f"{first['predicted_qty']:.0f} units "
                  f"[{first['lower_bound']:.0f} - {first['upper_bound']:.0f}]")

        except Exception as e:
            print(f"   ❌ Failed for {category}: {e}")
            continue

    if all_forecasts:
        combined = pd.concat(all_forecasts, ignore_index=True)
        save_forecasts(combined, engine)
        print("\n🎯 Forecasting pipeline complete")
        print(f"   {len(combined)} forecast records saved")
    else:
        print("❌ No forecasts generated")


if __name__ == "__main__":
    main()