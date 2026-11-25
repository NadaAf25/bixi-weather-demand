"""
03_build_features.py
Merge hourly BIXI trips and hourly weather into one table and add calendar
features. Output: data/processed/features_hourly.csv

Usage:
    py src/03_build_features.py               # 2026, the modelling year
    py src/03_build_features.py --year 2025   # 2025, used only by 16_year_transfer.py
"""
import argparse

import pandas as pd

from config import (BIXI_2025_CSV, BIXI_CSV, FEATURES_2025_CSV, FEATURES_CSV,
                    HOLIDAYS_2025, HOLIDAYS_2026, WEATHER, WEATHER_2025_CSV, WEATHER_CSV)

INPUTS = {   # year: (trips CSV, weather CSV, output CSV, public holidays)
    2026: (BIXI_CSV, WEATHER_CSV, FEATURES_CSV, HOLIDAYS_2026),
    2025: (BIXI_2025_CSV, WEATHER_2025_CSV, FEATURES_2025_CSV, HOLIDAYS_2025),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, choices=sorted(INPUTS), default=2026)
    args = parser.parse_args()
    bixi_csv, weather_csv, out_csv, holiday_list = INPUTS[args.year]
    holidays = pd.to_datetime(holiday_list)

    trips = pd.read_csv(bixi_csv, parse_dates=["hour_local"])
    weather = pd.read_csv(weather_csv, parse_dates=["hour_local"])
    weather = weather[["hour_local"] + WEATHER].drop_duplicates("hour_local")

    # Keep only the period covered by BIXI data (up to its last hour with a trip)
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
    df["is_holiday"] = df["hour_local"].dt.normalize().isin(holidays).astype(int)

    df = df.sort_values("hour_local").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Rows: {len(df):,}  |  from {df['hour_local'].min()} to {df['hour_local'].max()}")
    print(f"Hours with zero trips: {(df['trips'] == 0).sum()}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"File written: {out_csv}")
    print(df.head())


if __name__ == "__main__":
    main()