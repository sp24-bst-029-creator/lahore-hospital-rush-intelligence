"""Aggregate synthetic visit records into hourly arrivals and join historical weather."""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    visits_path = DATA_DIR / "edstays.csv.gz"
    weather_path = DATA_DIR / "lahore_historical_weather_2025.csv"
    if not visits_path.exists():
        raise FileNotFoundError("Run: python generate_synthetic_data.py")
    visits = pd.read_csv(visits_path, compression="gzip", parse_dates=["intime"])
    visits["arrival_hour"] = visits["intime"].dt.floor("h")
    hospitals = sorted(visits["hospital"].dropna().unique())
    all_hours = pd.date_range(visits["arrival_hour"].min(), visits["arrival_hour"].max(), freq="h")
    pieces = []
    for hospital in hospitals:
        counts = (visits.loc[visits.hospital == hospital]
                  .groupby("arrival_hour").size().rename("arrival_count"))
        part = counts.reindex(all_hours, fill_value=0).rename_axis("arrival_hour").reset_index()
        part["hospital"] = hospital
        pieces.append(part)
    by_hospital = pd.concat(pieces, ignore_index=True)
    by_hospital["hour"] = by_hospital.arrival_hour.dt.hour
    by_hospital["day_of_week"] = by_hospital.arrival_hour.dt.dayofweek
    by_hospital["month"] = by_hospital.arrival_hour.dt.month
    by_hospital["day_of_year"] = by_hospital.arrival_hour.dt.dayofyear
    by_hospital.to_csv(OUTPUT_DIR / "hourly_arrivals_by_hospital.csv", index=False)

    total = (by_hospital.groupby("arrival_hour", as_index=False).arrival_count.sum())
    total["hour"] = total.arrival_hour.dt.hour
    total["day_of_week"] = total.arrival_hour.dt.dayofweek
    total["month"] = total.arrival_hour.dt.month
    total["day_of_year"] = total.arrival_hour.dt.dayofyear
    total.to_csv(OUTPUT_DIR / "hourly_arrivals.csv", index=False)

    if weather_path.exists():
        weather = pd.read_csv(weather_path)
        weather["weather_time"] = pd.to_datetime(weather["weather_time"], errors="coerce")
        merged = total.merge(weather, left_on="arrival_hour", right_on="weather_time", how="left")
        merged = merged.drop(columns=["weather_time"], errors="ignore")
        merged.to_csv(OUTPUT_DIR / "hourly_arrivals_with_weather.csv", index=False)
        print("Joined weather for retrospective analysis. Weather is not used as a causal factor.")
    print(f"Processed {len(total):,} hourly records for {len(hospitals)} fictional hospitals.")

if __name__ == "__main__":
    main()
