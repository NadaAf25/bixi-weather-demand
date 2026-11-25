"""
03_build_features.py
Merge hourly BIXI trips and hourly weather into one table and add calendar
features. Output: data/processed/features_hourly.csv
"""
import pandas as pd

from config import BIXI_CSV, FEATURES_CSV, HOLIDAYS_2026, WEATHER, WEATHER_CSV


def main() -> None:
    holidays = pd.to_datetime(HOLIDAYS_2026)

    trips = pd.read_csv(BIXI_CSV, parse_dates=["hour_local"])
    weather = pd.read_csv(WEATHER_CSV, parse_dates=["hour_local"])
    weather = weather[["hour_local"] + WEATHER].drop_duplicates("hour_local")

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
    df["is_holiday"] = df["hour_local"].dt.normalize().isin(holidays).astype(int)

    df = df.sort_values("hour_local").reset_index(drop=True)
    df.to_csv(FEATURES_CSV, index=False)

    print(f"Rows: {len(df):,}  |  from {df['hour_local'].min()} to {df['hour_local'].max()}")
    print(f"Hours with zero trips: {(df['trips'] == 0).sum()}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"File written: {FEATURES_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()
