"""
03_build_features.py
Merge hourly BIXI trips and hourly weather into one table and add calendar
features. Output: data/processed/features_hourly.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# Quebec public holidays during the period covered (2026)
HOLIDAYS_2026 = pd.to_datetime([
    "2026-04-03",  # Good Friday
    "2026-05-18",  # National Patriots' Day
    "2026-06-24",  # Fete nationale du Quebec
    "2026-07-01",  # Canada Day
])

WEATHER_COLS = ["TEMP", "DEW_POINT_TEMP", "PRECIP_AMOUNT", "RELATIVE_HUMIDITY",
                "STATION_PRESSURE", "WIND_SPEED", "WIND_DIRECTION"]


def main() -> None:
    trips = pd.read_csv(PROCESSED / "bixi_hourly.csv", parse_dates=["hour_local"])
    weather = pd.read_csv(PROCESSED / "weather_hourly.csv", parse_dates=["hour_local"])
    weather = weather[["hour_local"] + WEATHER_COLS].drop_duplicates("hour_local")

    # Keep only the period covered by BIXI data (the file ends on August 31)
    weather = weather[weather["hour_local"] <= trips["hour_local"].max()]

    # Start from the weather table; an hour with no trip row means 0 trips
    df = weather.merge(trips, on="hour_local", how="left")
    df["trips"] = df["trips"].fillna(0).astype(int)
    df = df.dropna()   # drop hours where a weather measurement is missing

    # Calendar features
    df["hour"] = df["hour_local"].dt.hour
    df["day_of_week"] = df["hour_local"].dt.dayofweek      # 0 = Monday
    df["month"] = df["hour_local"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = df["hour_local"].dt.normalize().isin(HOLIDAYS_2026).astype(int)

    df = df.sort_values("hour_local").reset_index(drop=True)
    out = PROCESSED / "features_hourly.csv"
    df.to_csv(out, index=False)

    print(f"Rows: {len(df):,}  |  from {df['hour_local'].min()} to {df['hour_local'].max()}")
    print(f"Hours with zero trips: {(df['trips'] == 0).sum()}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"File written: {out}")
    print(df.head())


if __name__ == "__main__":
    main()