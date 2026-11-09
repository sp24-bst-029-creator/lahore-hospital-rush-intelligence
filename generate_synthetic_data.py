"""Generate fictional hospital arrival records for an academic demonstration."""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / "edstays.csv.gz"
RANDOM_SEED = 42
YEAR = 2025
rng = np.random.default_rng(RANDOM_SEED)

hospitals = {
    "Lahore General Demonstration Hospital": 2.5,
    "Central Lahore Demonstration Hospital": 2.2,
    "East Lahore Demonstration Hospital": 1.9,
    "North Lahore Demonstration Hospital": 1.7,
    "South Lahore Demonstration Hospital": 1.5,
}

def hourly_multiplier(hour):
    if 0 <= hour <= 5: return 0.45
    if 6 <= hour <= 9: return 0.85
    if 10 <= hour <= 13: return 1.10
    if 14 <= hour <= 17: return 1.20
    if 18 <= hour <= 22: return 1.65
    return 0.85

def weekday_multiplier(day):
    return 1.12 if day >= 5 else 1.0

def seasonal_multiplier(month):
    values = {1:1.05, 2:1.00, 3:1.02, 4:0.98, 5:1.08, 6:1.12,
              7:1.10, 8:1.04, 9:0.98, 10:1.00, 11:1.03, 12:1.08}
    return values[month]

def main():
    hours = pd.date_range(f"{YEAR}-01-01", f"{YEAR+1}-01-01", freq="h", inclusive="left")
    records = []
    stay_id = subject_id = 1
    for hospital, base_rate in hospitals.items():
        for hour in hours:
            expected = base_rate * hourly_multiplier(hour.hour) * weekday_multiplier(hour.dayofweek) * seasonal_multiplier(hour.month)
            count = rng.poisson(expected)
            for _ in range(count):
                arrival = hour + pd.Timedelta(minutes=int(rng.integers(0,60)), seconds=int(rng.integers(0,60)))
                stay_minutes = int(np.clip(rng.gamma(shape=2.5, scale=90), 15, 900))
                records.append({
                    "subject_id": 100000 + subject_id, "stay_id": stay_id, "hospital": hospital,
                    "department": "Emergency", "age": int(np.clip(rng.normal(38,19),1,95)),
                    "gender": rng.choice(["Female","Male"], p=[0.46,0.54]),
                    "arrival_transport": rng.choice(["Ambulance","Walk-in","Private vehicle"], p=[0.25,0.50,0.25]),
                    "disposition": rng.choice(["Discharged","Admitted","Transferred"], p=[0.70,0.25,0.05]),
                    "intime": arrival, "outtime": arrival + pd.Timedelta(minutes=stay_minutes),
                    "data_source": "synthetic_historical"
                })
                stay_id += 1; subject_id += 1
    df = pd.DataFrame(records).sort_values("intime").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False, compression="gzip")
    print(f"Generated {len(df):,} synthetic patient visits: {OUTPUT_FILE}")
    print("Reminder: all hospitals and patient records are fictional demonstration data.")

if __name__ == "__main__":
    main()
