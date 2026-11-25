"""
10_prediction_intervals.py
How uncertain is a prediction? Build 90% prediction intervals with conformal
prediction and CHECK how often they really contain the actual number of trips.

Chronological split (no shuffling, the future never leaks into the past):
    train        April - June
    calibration  July       (used only to size the intervals)
    test         August     (never seen: coverage is measured here)

Two methods, same model inputs as the simulator (calendar + temperature + rain):
    1. Constant-width conformal: point forecast +/- one fixed margin.
    2. Conformalised quantile regression (CQR): a low and a high quantile model,
       widened by a margin learned on the calibration month. The width adapts to
       the situation (wide at rush hour, narrow at night).

A 90% interval should contain about 90% of the actual values. We report the real
coverage overall, for rainy hours and for rush hours, plus the average width.

Outputs:
    data/processed/interval_results.csv
    figures/10_prediction_intervals.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from conformal import conformal_margin
from config import (ALPHA, CALIBRATION_MONTH, DAYTIME_FIRST_HOUR, DAYTIME_LAST_HOUR,
                    FEATURES_CSV, FIGURES, FINAL_TEST_MONTH, GB_PARAMS,
                    INTERVAL_RESULTS_CSV, INTERVAL_TRAIN_LAST_MONTH, NIGHT_HOURS,
                    RUSH_HOURS, SIMULATOR_FEATURES, TARGET)

FIGURES.mkdir(parents=True, exist_ok=True)

FEATURES = SIMULATOR_FEATURES


def main() -> None:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["hour_local"])
    train = df[df["month"] <= INTERVAL_TRAIN_LAST_MONTH]
    cal = df[df["month"] == CALIBRATION_MONTH]
    test = df[df["month"] == FINAL_TEST_MONTH]
    print(f"Train {len(train):,} h (Apr-Jun) | calibration {len(cal):,} h (Jul) | "
          f"test {len(test):,} h (Aug)\n")

    # 1. constant-width conformal around the point forecast
    point = HistGradientBoostingRegressor(**GB_PARAMS).fit(train[FEATURES], train[TARGET])
    margin = conformal_margin(np.abs(cal[TARGET] - point.predict(cal[FEATURES])), ALPHA)
    mid = point.predict(test[FEATURES])
    const_lo, const_hi = np.clip(mid - margin, 0, None), mid + margin

    # 2. conformalised quantile regression
    lo_m = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA / 2, **GB_PARAMS)
    hi_m = HistGradientBoostingRegressor(loss="quantile", quantile=1 - ALPHA / 2, **GB_PARAMS)
    lo_m.fit(train[FEATURES], train[TARGET])
    hi_m.fit(train[FEATURES], train[TARGET])
    cal_lo, cal_hi = lo_m.predict(cal[FEATURES]), hi_m.predict(cal[FEATURES])
    q = conformal_margin(np.maximum(cal_lo - cal[TARGET], cal[TARGET] - cal_hi), ALPHA)
    raw_lo, raw_hi = lo_m.predict(test[FEATURES]), hi_m.predict(test[FEATURES])
    cqr_lo = np.clip(np.minimum(raw_lo, raw_hi) - q, 0, None)
    cqr_hi = np.maximum(raw_lo, raw_hi) + q

    y = test[TARGET].to_numpy()
    rain = (test["PRECIP_AMOUNT"] > 0).to_numpy()
    day = test["hour"].between(DAYTIME_FIRST_HOUR, DAYTIME_LAST_HOUR).to_numpy()
    rush = test["hour"].isin(RUSH_HOURS).to_numpy() & (test["is_weekend"] == 0).to_numpy()
    night = test["hour"].isin(NIGHT_HOURS).to_numpy()
    groups = {"All hours": np.ones(len(y), bool), "Rainy hours": rain,
              "Dry daytime hours": day & ~rain, "Weekday rush hours": rush,
              "Night (0-5 am)": night}

    rows = []
    for method, lo, hi in [("Constant width", const_lo, const_hi),
                           ("CQR (adaptive width)", cqr_lo, cqr_hi)]:
        for name, mask in groups.items():
            inside = (y[mask] >= lo[mask]) & (y[mask] <= hi[mask])
            rows.append({"method": method, "group": name, "n_hours": int(mask.sum()),
                         "coverage_pct": round(100 * inside.mean(), 1),
                         "mean_width": round(float((hi - lo)[mask].mean()))})
    res = pd.DataFrame(rows)
    res.to_csv(INTERVAL_RESULTS_CSV, index=False)

    print("Target coverage: 90%. Real coverage on August (never seen):\n")
    print(res.pivot(index="group", columns="method",
                    values="coverage_pct").loc[list(groups)].to_string(), "\n")
    print("Average interval width (trips per hour):\n")
    print(res.pivot(index="group", columns="method",
                    values="mean_width").loc[list(groups)].to_string())
    print(f"\nWritten: {INTERVAL_RESULTS_CSV}")

    # chart: two weeks of August, CQR band
    win = test["hour_local"].between("2026-08-10", "2026-08-16 23:00").to_numpy()
    t = test.loc[win, "hour_local"]
    plt.figure(figsize=(12, 5))
    plt.fill_between(t, cqr_lo[win], cqr_hi[win], color="#2a9d8f", alpha=0.3,
                     label="90% prediction interval (CQR)")
    plt.plot(t, y[win], color="black", lw=1.3, label="Actual")
    plt.title("BIXI trips per hour: actual values and 90% prediction interval (Aug 10-16, 2026)")
    plt.ylabel("Trips per hour")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "10_prediction_intervals.png", dpi=150)
    plt.close()
    print(f"Chart saved in: {FIGURES}")


if __name__ == "__main__":
    main()
