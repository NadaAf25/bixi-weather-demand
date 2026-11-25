"""
11_scenario_intervals.py
Add a 90% prediction interval (low and high bound) to every what-if scenario
of the simulator.

Same method as 10_prediction_intervals.py (conformalised quantile regression):
    quantile models trained on April-June, margin calibrated on July.
That setup was checked on August: 90.7% of real hours fell inside the interval.

The simulator's point forecast (07_build_scenarios.py) comes from a model
trained on all the data, so the interval is stretched when needed to always
contain that forecast. The script reports how many scenarios were adjusted.

Input : data/processed/features_hourly.csv, scenario_predictions.csv
Output: data/processed/scenario_intervals.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from conformal import conformal_margin
from config import (ALPHA, CALIBRATION_MONTH, FEATURES_CSV, GB_PARAMS,
                    INTERVAL_TRAIN_LAST_MONTH, NIGHT_HOURS, RUSH_HOURS,
                    SCENARIO_INTERVALS_CSV, SCENARIOS_CSV, SIMULATOR_FEATURES, TARGET)

FEATURES = SIMULATOR_FEATURES


def main() -> None:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["hour_local"])
    train, cal = df[df["month"] <= INTERVAL_TRAIN_LAST_MONTH], df[df["month"] == CALIBRATION_MONTH]

    lo_m = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA / 2, **GB_PARAMS)
    hi_m = HistGradientBoostingRegressor(loss="quantile", quantile=1 - ALPHA / 2, **GB_PARAMS)
    lo_m.fit(train[FEATURES], train[TARGET])
    hi_m.fit(train[FEATURES], train[TARGET])
    cal_lo, cal_hi = lo_m.predict(cal[FEATURES]), hi_m.predict(cal[FEATURES])
    q = conformal_margin(np.maximum(cal_lo - cal[TARGET], cal[TARGET] - cal_hi), ALPHA)
    print(f"Calibration margin (July): {q:.0f} trips/hour")

    grid = pd.read_csv(SCENARIOS_CSV)
    X = (grid.rename(columns={"temp_c": "TEMP", "precip_mm": "PRECIP_AMOUNT"})
             .assign(is_holiday=0)[FEATURES])
    raw_lo, raw_hi = lo_m.predict(X), hi_m.predict(X)
    lower = np.clip(np.minimum(raw_lo, raw_hi) - q, 0, None)
    upper = np.maximum(raw_lo, raw_hi) + q

    # The interval must contain the simulator's own point forecast
    point = grid["predicted_trips"].to_numpy()
    adjusted = int(((lower > point) | (upper < point)).sum())
    grid["lower_trips"] = np.minimum(lower, point).round().astype(int)
    grid["upper_trips"] = np.maximum(upper, point).round().astype(int)

    assert (grid["lower_trips"] <= grid["predicted_trips"]).all()
    assert (grid["predicted_trips"] <= grid["upper_trips"]).all()

    grid.to_csv(SCENARIO_INTERVALS_CSV, index=False)
    width = grid["upper_trips"] - grid["lower_trips"]
    print(f"{len(grid):,} scenarios written: {SCENARIO_INTERVALS_CSV}")
    print(f"Scenarios stretched to contain the point forecast: {adjusted:,} "
          f"({adjusted / len(grid):.1%})")
    print(f"Average interval width: {width.mean():.0f} trips/hour")
    for label, hours in [("Night (0-5 am)", NIGHT_HOURS),
                         ("Rush hours (7-9, 16-18)", RUSH_HOURS)]:
        print(f"  {label:<24} {width[grid['hour'].isin(hours)].mean():.0f}")
    wed = grid[(grid.temp_c == 22) & (grid.day_of_week == 2) & (grid.hour == 8)]
    print("\nWednesday 8 am, 22 C:")
    print(wed[["rain_label", "lower_trips", "predicted_trips", "upper_trips"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
