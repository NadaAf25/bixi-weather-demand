"""
07_build_scenarios.py
Build the "what-if" table behind the dashboard simulator.

Why a pre-computed table? Power BI cannot run a scikit-learn model by itself.
So we score the model on a grid of scenarios (temperature x rain x hour x day of
week) and store the predictions. The dashboard then only filters this table.

Model choice: Gradient Boosting on calendar + temperature + rain. In
06_weather_ablation.py this set is almost as accurate as "all weather"
(MAE 401 vs 388 trips/hour) and, unlike humidity or pressure, every input can be
set by a human in a simulator.

Outputs:
    data/processed/scenario_predictions.csv
    figures/8_scenario_curves.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import (DAY_NAMES, FEATURES_CSV, FIGURES, FINAL_TEST_MONTH, GB_PARAMS,
                    MIN_SIMILAR_HOURS, RAIN_LEVELS, SCENARIOS_CSV, SIMULATOR_FEATURES,
                    TARGET, TEMPS)

FIGURES.mkdir(parents=True, exist_ok=True)

FEATURES = SIMULATOR_FEATURES


def main() -> None:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["hour_local"])

    # Same time-based check as the ablation study: train April-July, test August
    train, test = df[df["month"] < FINAL_TEST_MONTH], df[df["month"] == FINAL_TEST_MONTH]
    check = HistGradientBoostingRegressor(**GB_PARAMS)
    check.fit(train[FEATURES], train[TARGET])
    pred = np.clip(check.predict(test[FEATURES]), 0, None)
    mae = np.abs(pred - test[TARGET]).mean()
    print(f"Validation MAE on August: {mae:.0f} trips/hour")

    # Final model: all the data
    model = HistGradientBoostingRegressor(**GB_PARAMS)
    model.fit(df[FEATURES], df[TARGET])

    grid = pd.MultiIndex.from_product(
        [TEMPS, RAIN_LEVELS.items(), range(24), range(7)],
        names=["temp_c", "rain", "hour", "day_of_week"]).to_frame(index=False)
    grid["rain_label"] = grid["rain"].str[0]
    grid["precip_mm"] = grid["rain"].str[1]
    grid = grid.drop(columns="rain")
    grid["is_weekend"] = (grid["day_of_week"] >= 5).astype(int)
    grid["is_holiday"] = 0
    grid["day_name"] = grid["day_of_week"].map(dict(enumerate(DAY_NAMES)))

    X = grid.rename(columns={"temp_c": "TEMP", "precip_mm": "PRECIP_AMOUNT"})[FEATURES]
    grid["predicted_trips"] = np.clip(model.predict(X), 0, None).round().astype(int)

    # Reliability flag: how many real hours look like this scenario (same rain
    # class, temperature within +/- 2 C)? A tree model has nothing to learn from
    # when that number is tiny (e.g. rain at 30 C), so the dashboard warns.
    observed_rain = df["PRECIP_AMOUNT"] > 0
    counts = {}
    for temp in TEMPS:
        near = (df["TEMP"] - temp).abs() <= 2
        counts[(temp, True)] = int((near & observed_rain).sum())
        counts[(temp, False)] = int((near & ~observed_rain).sum())
    is_rain = grid["precip_mm"] > 0
    grid["n_similar_hours"] = [counts[(t, r)] for t, r in zip(grid["temp_c"], is_rain)]
    grid["is_reliable"] = (grid["n_similar_hours"] >= MIN_SIMILAR_HOURS).astype(int)
    print(f"Unreliable scenarios: {(1 - grid['is_reliable']).mean():.0%} of the grid")

    cols = ["temp_c", "rain_label", "precip_mm", "hour", "day_of_week", "day_name",
            "is_weekend", "predicted_trips", "n_similar_hours", "is_reliable"]
    grid[cols].to_csv(SCENARIOS_CSV, index=False)
    print(f"{len(grid):,} scenarios written: {SCENARIOS_CSV}")

    # Headline numbers (also used in the README)
    for temp in (10, 22, 30):
        for label in RAIN_LEVELS:
            day = grid[(grid.temp_c == temp) & (grid.rain_label == label)
                       & (grid.day_of_week == 2)]                  # a Wednesday
            print(f"  {temp:>2} C, {label:<22} Wednesday total: "
                  f"{day['predicted_trips'].sum():>7,} trips")

    # Chart 8: daily trips vs temperature, by rain level (weekday vs weekend)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"No rain": "#e76f51", "Light rain (1 mm/h)": "#2a9d8f",
              "Heavy rain (3 mm/h)": "#264653"}
    for ax, (title, dow) in zip(axes, [("Wednesday", 2), ("Saturday", 5)]):
        sub = grid[grid.day_of_week == dow]
        totals = sub.groupby(["rain_label", "temp_c"])["predicted_trips"].sum()
        reliable = sub.groupby(["rain_label", "temp_c"])["is_reliable"].min()
        for label in RAIN_LEVELS:
            t, y = totals[label].index.values, totals[label].values
            ok = reliable[label].values == 1
            # solid where the data supports the scenario, dotted where it does not
            ax.plot(t, y, color=colors[label], lw=1.5, ls=":", alpha=0.8)
            ax.plot(t[ok], np.where(ok, y, np.nan)[ok], label=label,
                    color=colors[label], lw=2.5)
        ax.set_title(f"{title}: model-predicted trips if it rains ALL day")
        ax.set_xlabel("Temperature (°C)   (dotted = too few similar hours observed)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted trips per day")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "8_scenario_curves.png", dpi=150)
    plt.close()
    print(f"Chart saved in: {FIGURES}")


if __name__ == "__main__":
    main()
