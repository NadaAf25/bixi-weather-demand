"""
14_lag_experiment.py
Do lagged-demand features improve the forecast?

Same rolling-origin folds as 09_validate_models.py (train on the months before,
test on June, July, August), same model, same hours: only the features change.

    Simulator model              calendar + temperature + rain (no knowledge of past trips)
    Day ahead                    + lag_24, lag_168, mean of the 24 h ending 24 h earlier
    One hour ahead               + lag_1, lag_24, lag_168, 3 h and 24 h means
    Calendar + lags, no weather  does the weather still add something once past demand is known?

Lags need real past trips: they are for forecasting only. The what-if simulator asks
"what if it rained?" for hours that have not happened, so it keeps no lag.

The first week of April has no lag_168, so those hours are dropped for ALL models
(otherwise the models would not be tested on the same hours).

Output:
    data/processed/lag_results.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import (CALENDAR, FEATURES_CSV, GB_PARAMS, LAG_RESULTS_CSV, ROLLING_FOLDS,
                    SIMULATOR_FEATURES, TARGET)
from lags import DAY_AHEAD, ONE_HOUR_AHEAD, add_lags

SIMULATOR_LABEL = "Simulator model (calendar + temp + rain)"
FEATURE_SETS = {
    SIMULATOR_LABEL: SIMULATOR_FEATURES,
    "Day ahead: + lag_24, lag_168, 24 h mean": SIMULATOR_FEATURES + DAY_AHEAD,
    "One hour ahead: + lag_1, lag_24, lag_168, 3 h/24 h means":
        SIMULATOR_FEATURES + ONE_HOUR_AHEAD,
    "Day ahead, calendar + lags, no weather": CALENDAR + DAY_AHEAD,
}


def main() -> None:
    df = add_lags(pd.read_csv(FEATURES_CSV, parse_dates=["hour_local"]))
    needed = sorted({c for features in FEATURE_SETS.values() for c in features})
    complete = df.dropna(subset=needed)
    print(f"{len(df) - len(complete)} of {len(df)} hours dropped for every model "
          f"(no lag_168 in the first week, or a gap in the data).\n")

    rows = []
    for month, name in ROLLING_FOLDS:
        train = complete[complete["month"] < month]
        test = complete[complete["month"] == month]
        for label, features in FEATURE_SETS.items():
            model = HistGradientBoostingRegressor(**GB_PARAMS)
            model.fit(train[features], train[TARGET])
            pred = np.clip(model.predict(test[features]), 0, None)
            err = pred - test[TARGET].to_numpy()
            rows.append({"test_month": name, "model": label, "n_hours": len(test),
                         "train_hours": len(train), "MAE": np.abs(err).mean(),
                         "RMSE": np.sqrt((err ** 2).mean()), "bias": err.mean()})

    res = pd.DataFrame(rows).round(1)
    LAG_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(LAG_RESULTS_CSV, index=False)

    table = res.pivot(index="model", columns="test_month", values="MAE")[
        [n for _, n in ROLLING_FOLDS]]
    table["mean"] = table.mean(axis=1).round(1)
    table["vs_simulator_%"] = ((table["mean"] / table.loc[SIMULATOR_LABEL, "mean"] - 1)
                               * 100).round(1)
    print("MAE (trips per hour) by test month, lower is better:\n")
    print(table.sort_values("mean").to_string())
    print("\nRMSE and bias per fold:\n")
    print(res[["test_month", "model", "RMSE", "bias"]].to_string(index=False))
    print(f"\nWritten: {LAG_RESULTS_CSV}")


if __name__ == "__main__":
    main()