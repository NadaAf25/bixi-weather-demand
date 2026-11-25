"""
09_validate_models.py
Is the model good only in August, or in every month? Rolling-origin validation
plus naive baselines.

Folds (the training window always grows and always ends BEFORE the test month):
    train April-May   -> test June
    train April-June  -> test July
    train April-July  -> test August   (the final, untouched test month)

Models compared on exactly the same test hours:
    Naive: same hour, last week        (what happened 168 h earlier)
    Naive: average hour x day of week  (from the training months)
    Gradient Boosting, calendar only
    Gradient Boosting, calendar + temperature + rain   (the simulator model)
    Gradient Boosting, calendar + all weather

Note on the naive "last week" model: it uses real past trips, which a what-if
simulator cannot know. It is a forecasting yardstick, not a competitor for the
simulator.

Outputs:
    data/processed/validation_results.csv
    figures/9_rolling_validation.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import (CALENDAR, FEATURES_CSV, FIGURES, GB_PARAMS, ROLLING_FOLDS,
                    SIMULATOR_FEATURES, TARGET, VALIDATION_CSV, WEATHER)

FIGURES.mkdir(parents=True, exist_ok=True)

ML_MODELS = {
    "GB: calendar only": CALENDAR,
    "GB: calendar + temp + rain (simulator)": SIMULATOR_FEATURES,
    "GB: calendar + all weather": CALENDAR + WEATHER,
}


def add_last_week(df: pd.DataFrame) -> pd.DataFrame:
    """Trips 168 hours earlier (NaN when that hour is not in the table)."""
    past = df[["hour_local", TARGET]].copy()
    past["hour_local"] = past["hour_local"] + pd.Timedelta(hours=168)
    past = past.rename(columns={TARGET: "trips_last_week"})
    return df.merge(past, on="hour_local", how="left")


def metrics(y, pred) -> dict:
    pred = np.clip(pred, 0, None)
    err = pred - y
    return {"MAE": err.abs().mean(), "RMSE": float(np.sqrt((err ** 2).mean())),
            "bias": err.mean()}


def main() -> None:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["hour_local"])
    df = add_last_week(df)

    rows = []
    for month, name in ROLLING_FOLDS:
        train = df[df["month"] < month]
        test = df[(df["month"] == month) & df["trips_last_week"].notna()]
        y = test[TARGET]
        scores = {}

        scores["Naive: same hour last week"] = test["trips_last_week"]
        typical = train.groupby(["day_of_week", "hour"])[TARGET].mean()
        keys = list(zip(test["day_of_week"], test["hour"]))
        scores["Naive: average hour x weekday"] = pd.Series(
            [typical[k] for k in keys], index=test.index)
        for label, features in ML_MODELS.items():
            model = HistGradientBoostingRegressor(**GB_PARAMS)
            model.fit(train[features], train[TARGET])
            scores[label] = pd.Series(model.predict(test[features]), index=test.index)

        for label, pred in scores.items():
            rows.append({"test_month": name, "model": label, "n_hours": len(test),
                         "train_hours": len(train), **metrics(y, pred)})

    res = pd.DataFrame(rows).round(1)
    res.to_csv(VALIDATION_CSV, index=False)

    table = res.pivot(index="model", columns="test_month", values="MAE")[
        [n for _, n in ROLLING_FOLDS]]
    table["mean"] = table.mean(axis=1).round(1)
    table = table.sort_values("mean")
    print("MAE (trips per hour) by test month, lower is better:\n")
    print(table.to_string(), "\n")
    print("Bias (average prediction minus actual) for the simulator model:")
    bias = res[res["model"].str.contains("simulator")][["test_month", "bias"]]
    print(bias.to_string(index=False))
    print(f"\nWritten: {VALIDATION_CSV}")

    ax = table.drop(columns="mean").plot(kind="bar", figsize=(11, 5.5), width=0.8)
    ax.set_title("Rolling validation: mean error by test month (lower is better)")
    ax.set_ylabel("Mean absolute error (trips per hour)")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "9_rolling_validation.png", dpi=150)
    plt.close()
    print(f"Chart saved in: {FIGURES}")


if __name__ == "__main__":
    main()
