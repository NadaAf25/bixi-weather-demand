"""
13_error_analysis.py
Where does the simulator model make its errors?

Same model (calendar + temperature + rain) and same rolling-origin folds as
09_validate_models.py (train on the months before, test on June, July, August).
The errors of the three test months are pooled, then summarised by:
    - time of day
    - dry vs rainy hours (all hours, then daytime only, 7 am to 9 pm)
    - temperature band

For each group: number of hours, average real trips, MAE, RMSE, bias
(prediction minus actual) and MAE as a percentage of the average real trips.
As in 09_validate_models.py, negative predictions are clipped to zero.

Output:
    data/processed/error_analysis.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import (DAYTIME_FIRST_HOUR, DAYTIME_LAST_HOUR, ERROR_ANALYSIS_CSV,
                    FEATURES_CSV, GB_PARAMS, ROLLING_FOLDS, SIMULATOR_FEATURES, TARGET)

HOUR_BANDS = {
    "Night (0-5 am)": range(0, 6),
    "Morning (6-9 am)": range(6, 10),
    "Midday (10 am-3 pm)": range(10, 16),
    "Evening rush (4-6 pm)": range(16, 19),
    "Evening (7-11 pm)": range(19, 24),
}
TEMP_BINS = [-np.inf, 10, 15, 20, 25, np.inf]
TEMP_LABELS = ["< 10 C", "10-15 C", "15-20 C", "20-25 C", ">= 25 C"]


def predict_test_months(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample predictions for June, July and August (rolling origin)."""
    parts = []
    for month, _ in ROLLING_FOLDS:
        train, test = df[df["month"] < month], df[df["month"] == month].copy()
        model = HistGradientBoostingRegressor(**GB_PARAMS)
        model.fit(train[SIMULATOR_FEATURES], train[TARGET])
        test["pred"] = np.clip(model.predict(test[SIMULATOR_FEATURES]), 0, None)
        parts.append(test)
    out = pd.concat(parts, ignore_index=True)
    out["actual"] = out[TARGET]
    out["err"] = out["pred"] - out["actual"]
    out["abs_err"] = out["err"].abs()
    out["sq_err"] = out["err"] ** 2
    return out


def summarise(data: pd.DataFrame, column: str, section: str) -> pd.DataFrame:
    g = data.groupby(column, observed=True)
    out = pd.DataFrame({
        "n_hours": g.size(),
        "mean_trips": g["actual"].mean(),
        "MAE": g["abs_err"].mean(),
        "RMSE": g["sq_err"].mean() ** 0.5,
        "bias": g["err"].mean(),
    })
    out["MAE_pct_of_mean"] = 100 * out["MAE"] / out["mean_trips"]
    out.index.name = "group"
    out.insert(0, "section", section)
    return out.reset_index()


def main() -> None:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["hour_local"])
    test = predict_test_months(df)

    hour_to_band = {h: band for band, hours in HOUR_BANDS.items() for h in hours}
    test["time_of_day"] = pd.Categorical(test["hour"].map(hour_to_band),
                                         categories=list(HOUR_BANDS), ordered=True)
    test["rain"] = pd.Categorical(np.where(test["PRECIP_AMOUNT"] > 0, "Rain", "Dry"),
                                  categories=["Dry", "Rain"], ordered=True)
    test["temp_band"] = pd.cut(test["TEMP"], TEMP_BINS, labels=TEMP_LABELS, right=False)
    test["all"] = "All test hours (June-August)"
    daytime = test[test["hour"].between(DAYTIME_FIRST_HOUR, DAYTIME_LAST_HOUR)]

    res = pd.concat([
        summarise(test, "all", "Overall"),
        summarise(test, "time_of_day", "Time of day"),
        summarise(test, "rain", "Rain (all hours)"),
        summarise(daytime, "rain", "Rain (daytime 7 am-9 pm)"),
        summarise(test, "temp_band", "Temperature"),
    ], ignore_index=True).round(1)

    ERROR_ANALYSIS_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(ERROR_ANALYSIS_CSV, index=False)
    print(res.to_string(index=False))
    print(f"\nWritten: {ERROR_ANALYSIS_CSV}")


if __name__ == "__main__":
    main()