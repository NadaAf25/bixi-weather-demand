"""
12_station_comparison.py
Does the choice of weather station change the results?

Two stations are compared over April-August 2026:
    airport   Montreal/Trudeau (702S006), the station used by the pipeline
    mctavish  McTavish (7024745), downtown

Part 1  How different are the two sets of observations (temperature, rain)?
Part 2  Same model, same calendar features, same rolling-origin protocol: only the
        weather columns change. Compare MAE, RMSE and bias per fold.

The model uses the simulator features (calendar + temperature + rain), because
those are the variables both stations provide.

Prerequisite:
    py src/01_extract_weather.py --station 7024745 --out data/processed/weather_mctavish.csv

Output:
    data/processed/station_comparison.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import (FEATURES_CSV, GB_PARAMS, ROLLING_FOLDS, SIMULATOR_FEATURES,
                    STATION_COMPARISON_CSV, TARGET, WEATHER_CSV, WEATHER_MCTAVISH_CSV)

TIME_CANDIDATES = ["hour_local", "ts_local", "datetime", "ts_utc"]
WEATHER_COLS = ["TEMP", "PRECIP_AMOUNT"]


def find_time_column(features: pd.DataFrame) -> str:
    for col in TIME_CANDIDATES:
        if col in features.columns:
            return col
    raise SystemExit(f"No time column found in features_hourly.csv. Columns: {list(features.columns)}")


def compare_observations(airport: pd.DataFrame, mctavish: pd.DataFrame) -> None:
    both = airport.set_index("ts_utc")[WEATHER_COLS].join(
        mctavish.set_index("ts_utc")[WEATHER_COLS], lsuffix="_a", rsuffix="_m", how="inner")
    t = both.dropna(subset=["TEMP_a", "TEMP_m"])
    d = t["TEMP_m"] - t["TEMP_a"]
    print(f"Hours in common: {len(both)} | with both temperatures: {len(t)}")
    print(f"Temperature McTavish - airport: mean {d.mean():.2f} C, mean abs {d.abs().mean():.2f} C, "
          f"max abs {d.abs().max():.1f} C, corr {t['TEMP_a'].corr(t['TEMP_m']):.4f}")
    p = both.dropna(subset=["PRECIP_AMOUNT_a", "PRECIP_AMOUNT_m"])
    ra, rm = p["PRECIP_AMOUNT_a"] > 0, p["PRECIP_AMOUNT_m"] > 0
    print(f"Rainy hours: airport {int(ra.sum())} | McTavish {int(rm.sum())} | "
          f"same verdict {(ra == rm).mean() * 100:.1f}%")
    print(f"Total rain: airport {p['PRECIP_AMOUNT_a'].sum():.0f} mm | "
          f"McTavish {p['PRECIP_AMOUNT_m'].sum():.0f} mm\n")


def with_station(features: pd.DataFrame, key: str, weather: pd.DataFrame) -> pd.DataFrame:
    """Replace the weather columns of the features by those of one station."""
    w = (weather[[key] + WEATHER_COLS].drop_duplicates(key)
         .assign(**{key: lambda x: pd.to_datetime(x[key])}))
    base = features.drop(columns=WEATHER_COLS, errors="ignore").assign(
        **{key: pd.to_datetime(features[key])})
    return base.merge(w, on=key, how="inner")


def rolling_scores(df: pd.DataFrame, key: str, station: str) -> list[dict]:
    month = pd.to_datetime(df[key]).dt.month
    rows = []
    for test_month, name in ROLLING_FOLDS:
        train, test = df[month < test_month], df[month == test_month]
        model = HistGradientBoostingRegressor(**GB_PARAMS)
        model.fit(train[SIMULATOR_FEATURES], train[TARGET])
        err = model.predict(test[SIMULATOR_FEATURES]) - test[TARGET].to_numpy()
        rainy = (test["PRECIP_AMOUNT"] > 0).to_numpy()
        rows.append({
            "station": station, "fold": name, "n_test": len(test),
            "mae": np.abs(err).mean(), "rmse": np.sqrt((err ** 2).mean()),
            "bias": err.mean(),
            "mae_rainy": np.abs(err[rainy]).mean() if rainy.any() else np.nan,
        })
    return rows


def main() -> None:
    if not WEATHER_MCTAVISH_CSV.exists():
        raise SystemExit("Missing data/processed/weather_mctavish.csv. Run:\n"
                         "  py src/01_extract_weather.py --station 7024745 "
                         "--out data/processed/weather_mctavish.csv")
    airport = pd.read_csv(WEATHER_CSV)
    mctavish = pd.read_csv(WEATHER_MCTAVISH_CSV)

    print("=== Part 1: observations ===")
    compare_observations(airport, mctavish)

    print("=== Part 2: model performance (same protocol, only the weather changes) ===")
    features = pd.read_csv(FEATURES_CSV)
    key = find_time_column(features)
    rows = []
    for name, weather in [("airport", airport), ("mctavish", mctavish)]:
        data = with_station(features, key, weather)
        print(f"{name}: {len(data)} rows after joining on '{key}'")
        rows += rolling_scores(data, key, name)

    out = pd.DataFrame(rows)
    STATION_COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.round(1).to_csv(STATION_COMPARISON_CSV, index=False)
    print()
    print(out.round(1).to_string(index=False))
    print("\nMean over the folds:")
    print(out.groupby("station")[["mae", "rmse", "bias", "mae_rainy"]].mean().round(1))
    print(f"\nWritten: {STATION_COMPARISON_CSV}")


if __name__ == "__main__":
    main()