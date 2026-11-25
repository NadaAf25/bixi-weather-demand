"""
15_rain_experiment.py
The error analysis (13_error_analysis.py) shows that the simulator model predicts
too many trips when it rains. Can a constraint or recent-rain features fix it?

Same rolling-origin folds as 09_validate_models.py (test June, July, August).
Only the model changes:
    1. Simulator model (calendar + temperature + rain), as it is today
    2. Same features, rain forced to never increase demand (monotonic constraint)
    3. Model 2 + recent rain (rain_3h, rained_prev_hour)

Rain features come from weather only (see rain_features.py). The what-if simulator
cannot give a rain history, so model 3 is a forecasting experiment.
As elsewhere, negative predictions are clipped to zero.

Output:
    data/processed/rain_results.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import (DAYTIME_FIRST_HOUR, DAYTIME_LAST_HOUR, FEATURES_CSV, GB_PARAMS,
                    RAIN_RESULTS_CSV, ROLLING_FOLDS, SIMULATOR_FEATURES, TARGET)
from rain_features import RAIN_FEATURES, add_rain_features, monotone_constraints

MODELS = {   # label: (features, monotonic constraint on rain)
    "Simulator model (current)": (SIMULATOR_FEATURES, False),
    "+ monotone rain": (SIMULATOR_FEATURES, True),
    "+ monotone rain + recent rain": (SIMULATOR_FEATURES + RAIN_FEATURES, True),
}


def predict_folds(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for month, name in ROLLING_FOLDS:
        train, test = df[df["month"] < month], df[df["month"] == month]
        for label, (features, monotone) in MODELS.items():
            params = dict(GB_PARAMS)
            if monotone:
                params["monotonic_cst"] = monotone_constraints(features)
            model = HistGradientBoostingRegressor(**params)
            model.fit(train[features], train[TARGET])
            part = test[["hour", "PRECIP_AMOUNT", TARGET]].copy()
            part["pred"] = np.clip(model.predict(test[features]), 0, None)
            part["model"], part["fold"] = label, name
            parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["err"] = out["pred"] - out[TARGET]
    return out


def summarise(data: pd.DataFrame, group: str, fold: str) -> pd.DataFrame:
    g = data.groupby("model", sort=False)["err"]
    res = pd.DataFrame({
        "n_hours": g.size(),
        "MAE": g.apply(lambda e: e.abs().mean()),
        "RMSE": g.apply(lambda e: float(np.sqrt((e ** 2).mean()))),
        "bias": g.mean(),
    })
    res.insert(0, "fold", fold)
    res.insert(0, "group", group)
    return res.reset_index()


def main() -> None:
    df = add_rain_features(pd.read_csv(FEATURES_CSV, parse_dates=["hour_local"]))
    test = predict_folds(df)

    rainy = test["PRECIP_AMOUNT"] > 0
    day = test["hour"].between(DAYTIME_FIRST_HOUR, DAYTIME_LAST_HOUR)
    pooled = [
        summarise(test, "All hours", "Pooled"),
        summarise(test[~rainy], "Dry hours", "Pooled"),
        summarise(test[rainy], "Rainy hours", "Pooled"),
        summarise(test[rainy & day], "Rainy hours, daytime (7 am-9 pm)", "Pooled"),
    ]
    per_fold = [summarise(test[test["fold"] == name], "All hours", name)
                for _, name in ROLLING_FOLDS]
    res = pd.concat(pooled + per_fold, ignore_index=True).round(1)

    RAIN_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(RAIN_RESULTS_CSV, index=False)
    print(res.to_string(index=False))
    print(f"\nWritten: {RAIN_RESULTS_CSV}")


if __name__ == "__main__":
    main()