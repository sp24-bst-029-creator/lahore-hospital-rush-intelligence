"""Train and evaluate time-only and time+weather Random Forest models."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = OUTPUT_DIR / "hourly_arrivals_with_weather.csv"
TIME_FEATURES = ["hour", "day_of_week", "month", "day_of_year"]
WEATHER_FEATURES = ["temperature_c", "humidity_percent", "precipitation_mm", "wind_speed_kmh"]

def score(y, pred):
    pred = np.maximum(pred, 0)
    return mean_absolute_error(y, pred), float(np.sqrt(mean_squared_error(y, pred)))

def main():
    data = pd.read_csv(DATA_FILE, parse_dates=["arrival_hour"])
    needed = ["arrival_hour", "arrival_count"] + TIME_FEATURES + WEATHER_FEATURES
    missing = [c for c in needed if c not in data.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    data = data.dropna(subset=needed).sort_values("arrival_hour").reset_index(drop=True)
    split = max(1, min(int(len(data)*0.8), len(data)-1))
    train, test = data.iloc[:split], data.iloc[split:]
    baseline = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    weather_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    baseline.fit(train[TIME_FEATURES], train.arrival_count)
    weather_model.fit(train[TIME_FEATURES + WEATHER_FEATURES], train.arrival_count)
    p1 = np.maximum(baseline.predict(test[TIME_FEATURES]), 0)
    p2 = np.maximum(weather_model.predict(test[TIME_FEATURES + WEATHER_FEATURES]), 0)
    m1, r1 = score(test.arrival_count, p1); m2, r2 = score(test.arrival_count, p2)
    # Refit final deployable estimators on all historical observations after
    # evaluating on the untouched chronological holdout.
    final_baseline = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    final_weather = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    final_baseline.fit(data[TIME_FEATURES], data.arrival_count)
    final_weather.fit(data[TIME_FEATURES + WEATHER_FEATURES], data.arrival_count)
    joblib.dump({"model": final_weather, "features": TIME_FEATURES+WEATHER_FEATURES,
                 "target": "arrival_count", "baseline_model": final_baseline,
                 "baseline_features": TIME_FEATURES, "weather_features": WEATHER_FEATURES,
                 "model_type": "RandomForestRegressor"}, OUTPUT_DIR/"hospital_rush_model.joblib")
    pd.DataFrame([
        {"model":"Random Forest - time only", "MAE":m1, "RMSE":r1},
        {"model":"Random Forest - time + historical weather", "MAE":m2, "RMSE":r2},
    ]).to_csv(OUTPUT_DIR/"random_forest_comparison.csv", index=False)
    pd.DataFrame([{"rows":len(data), "train_rows":len(train), "test_rows":len(test),
                   "time_only_MAE":m1, "time_only_RMSE":r1,
                   "time_weather_MAE":m2, "time_weather_RMSE":r2}]).to_csv(OUTPUT_DIR/"model_evaluation.csv", index=False)
    print("Random Forest training complete. Model and evaluation saved in outputs/.")
    print("Note: do not use historical weather features for future forecasts without future weather forecasts.")

if __name__ == "__main__": main()
