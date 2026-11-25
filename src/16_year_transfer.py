"""
16_year_transfer.py
Does the model still work on a year it has never seen?

The model is trained once on ALL of 2026 (April to August), with the same settings as
everywhere else, then frozen and scored on 2025. Nothing from 2025 was used to train,
tune or choose anything in this project.

Three models, scored on the same 2025 hours:
    Naive: average hour x weekday   (from the 2026 training data)
    Gradient Boosting, calendar only
    Gradient Boosting, calendar + temperature + rain   (the simulator model)

Test hours are grouped as:
    May-August 2025       the same months as the training year
    September-October     autumn: months the model has never seen (the temperatures are
                          inside the 2026 range, but cool hours only exist in April 2026)
plus each month, and dry vs rainy hours in May-August.

Diagnostics printed with the results: the average demand of each month in both years
(the network may have grown), the share of hours with no trip (a closed system looks
like zero demand) and the share of hours outside the 2026 temperature range.
As elsewhere, negative predictions are clipped to zero.

Output:
    data/processed/year_transfer_results.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import (CALENDAR, FEATURES_2025_CSV, FEATURES_CSV, GB_PARAMS,
                    SIMULATOR_FEATURES, TARGET, YEAR_TRANSFER_AUTUMN, YEAR_TRANSFER_CSV,
                    YEAR_TRANSFER_MONTHS, YEAR_TRANSFER_SAME_SEASON)

MONTH_NAMES = {4: "April", 5: "May", 6: "June", 7: "July", 8: "August",
               9: "September", 10: "October"}
NAIVE = "Naive: average hour x weekday (2026)"
CALENDAR_ONLY = "Calendar only (trained on 2026)"
SIMULATOR = "Simulator model (trained on 2026)"


def predict_models(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    typical = train.groupby(["day_of_week", "hour"])[TARGET].mean()
    keys = pd.MultiIndex.from_arrays([test["day_of_week"], test["hour"]])
    preds = {NAIVE: pd.Series(typical.reindex(keys).to_numpy(), index=test.index)}
    for label, features in {CALENDAR_ONLY: CALENDAR, SIMULATOR: SIMULATOR_FEATURES}.items():
        model = HistGradientBoostingRegressor(**GB_PARAMS)
        model.fit(train[features], train[TARGET])
        preds[label] = pd.Series(np.clip(model.predict(test[features]), 0, None),
                                 index=test.index)
    return preds


def summarise(test, preds, mask, section, group, temp_range) -> pd.DataFrame:
    sub, y = test[mask], test.loc[mask, TARGET]
    outside = 100 * ((sub["TEMP"] < temp_range[0]) | (sub["TEMP"] > temp_range[1])).mean()
    rows = []
    for label, pred in preds.items():
        err = pred[mask] - y
        rows.append({"section": section, "group": group, "model": label,
                     "n_hours": len(sub), "mean_trips": y.mean(),
                     "MAE": err.abs().mean(), "RMSE": float(np.sqrt((err ** 2).mean())),
                     "bias": err.mean(), "pct_temp_outside_2026_range": outside})
    out = pd.DataFrame(rows)
    out["MAE_pct_of_mean"] = 100 * out["MAE"] / out["mean_trips"]
    return out


def main() -> None:
    train = pd.read_csv(FEATURES_CSV, parse_dates=["hour_local"])
    test = pd.read_csv(FEATURES_2025_CSV, parse_dates=["hour_local"])
    test = test[test["month"].isin(YEAR_TRANSFER_MONTHS)].reset_index(drop=True)
    assert not set(train["hour_local"]) & set(test["hour_local"]), "train and test overlap"

    temp_range = (train["TEMP"].min(), train["TEMP"].max())
    month = test["month"]
    print(f"Training: {len(train):,} hours of 2026 | test: {len(test):,} hours of 2025")
    print(f"2026 temperature range: {temp_range[0]:.1f} to {temp_range[1]:.1f} C\n")

    level = pd.DataFrame({
        "mean_trips_2025": test.groupby("month")[TARGET].mean(),
        "mean_trips_2026": train.groupby("month")[TARGET].mean(),
        "pct_zero_hours_2025": 100 * (test[TARGET] == 0).groupby(month).mean(),
    }).round(1)
    level["ratio_2026_vs_2025"] = (level["mean_trips_2026"] / level["mean_trips_2025"]).round(2)
    print("Demand level by month (is 2026 a bigger or smaller network than 2025?):")
    print(level.to_string(), "\n")

    preds = predict_models(train, test)
    same = month.isin(YEAR_TRANSFER_SAME_SEASON)
    autumn = month.isin(YEAR_TRANSFER_AUTUMN)
    rain = test["PRECIP_AMOUNT"] > 0
    groups = [("Season", "May-August 2025 (same months as 2026)", same),
              ("Season", "September-October 2025 (autumn)", autumn)]
    groups += [("Month", MONTH_NAMES[m], month == m) for m in YEAR_TRANSFER_MONTHS]
    groups += [("Rain, May-August", "Dry", same & ~rain),
               ("Rain, May-August", "Rain", same & rain)]

    res = pd.concat([summarise(test, preds, mask, section, group, temp_range)
                     for section, group, mask in groups], ignore_index=True).round(1)
    YEAR_TRANSFER_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(YEAR_TRANSFER_CSV, index=False)
    print(res.to_string(index=False))
    print(f"\nWritten: {YEAR_TRANSFER_CSV}")


if __name__ == "__main__":
    main()