"""
06_weather_ablation.py
How much does weather really add? Train the SAME model (Gradient Boosting) with
different sets of features and compare the errors on two test months.

    Fold 1: train April - June  ->  test July
    Fold 2: train April - July  ->  test August

Outputs: data/processed/ablation_results.csv and figures/7_weather_ablation.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from config import (ABLATION_CSV, CALENDAR, DAYTIME_FIRST_HOUR, DAYTIME_LAST_HOUR,
                    FEATURES_CSV, FIGURES, GB_PARAMS, TARGET, WEATHER)

FIGURES.mkdir(parents=True, exist_ok=True)

FEATURE_SETS = {
    "Calendar only": CALENDAR,
    "Calendar + temperature": CALENDAR + ["TEMP"],
    "Calendar + temperature + rain": CALENDAR + ["TEMP", "PRECIP_AMOUNT"],
    "Calendar + all weather": CALENDAR + WEATHER,
    "All weather only (no calendar)": WEATHER,
}
TEST_MONTHS = {7: "MAE_July", 8: "MAE_August"}


def fit_predict(df: pd.DataFrame, features: list[str], test_month: int):
    train = df[df["month"] < test_month]
    test = df[df["month"] == test_month]
    model = HistGradientBoostingRegressor(**GB_PARAMS)
    model.fit(train[features], train[TARGET])
    pred = np.clip(model.predict(test[features]), 0, None)
    return test, pred


def main() -> None:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["hour_local"])

    rows, aug_predictions = [], {}
    for name, features in FEATURE_SETS.items():
        row = {"feature_set": name}
        for month, col in TEST_MONTHS.items():
            test, pred = fit_predict(df, features, month)
            row[col] = round(mean_absolute_error(test[TARGET], pred), 1)
            if month == 8:
                aug_predictions[name] = (test, pred)
        row["MAE_mean"] = round((row["MAE_July"] + row["MAE_August"]) / 2, 1)
        rows.append(row)

    results = pd.DataFrame(rows).set_index("feature_set")
    reference = results.loc["Calendar only", "MAE_mean"]
    results["error_reduction_vs_calendar_%"] = (
        100 * (1 - results["MAE_mean"] / reference)).round(1)
    results.to_csv(ABLATION_CSV)
    print("Mean absolute error (trips per hour), lower is better:\n")
    print(results.to_string(), "\n")

    # Where does weather help? Compare rainy vs dry daytime hours in August
    test_cal, pred_cal = aug_predictions["Calendar only"]
    _, pred_all = aug_predictions["Calendar + all weather"]
    day = test_cal["hour"].between(DAYTIME_FIRST_HOUR, DAYTIME_LAST_HOUR).values
    rain = (test_cal["PRECIP_AMOUNT"] > 0).values
    print("August, daytime hours (7 am - 9 pm): MAE by weather condition")
    for label, mask in [("Rainy hours", day & rain), ("Dry hours", day & ~rain)]:
        y = test_cal.loc[mask, TARGET]
        print(f"  {label:<12} (n={mask.sum():>3}): calendar only "
              f"{mean_absolute_error(y, pred_cal[mask]):.0f}  ->  with weather "
              f"{mean_absolute_error(y, pred_all[mask]):.0f}")

    # Chart
    plot = results["MAE_mean"].sort_values(ascending=False)
    colors = ["#e76f51" if i == "Calendar only" else "steelblue" for i in plot.index]
    ax = plot.plot(kind="barh", figsize=(9, 5), color=colors)
    ax.set_title("What does weather add? Mean error by feature set (Gradient Boosting)")
    ax.set_xlabel("Mean absolute error (trips per hour) - lower is better")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES / "7_weather_ablation.png", dpi=150)
    plt.close()
    print(f"\nChart saved in: {FIGURES}")


if __name__ == "__main__":
    main()
