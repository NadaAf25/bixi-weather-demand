"""
05_train_model.py
Predict the number of BIXI trips per hour from weather and calendar features.

Time-based split (no shuffling, to avoid leaking the future into training):
    train = April - July 2026   |   test = August 2026

Models compared:
    1. Baseline: average trips for the same hour of day and weekday/weekend
    2. Ridge regression
    3. Random Forest
    4. Gradient Boosting

Outputs: data/processed/model_results.csv and charts 5_*.png / 6_*.png in figures/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (CALENDAR, FEATURES_CSV, FIGURES, FINAL_TEST_MONTH, GB_PARAMS,
                    MODEL_RESULTS_CSV, SEED, TARGET, WEATHER)

FIGURES.mkdir(parents=True, exist_ok=True)

# The 'month' feature is left out on purpose
FEATURES = WEATHER + CALENDAR


def evaluate(name: str, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "model": name,
        "MAE": round(mae, 1),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
        "R2": round(r2_score(y_true, y_pred), 3),
        "MAE_%_of_mean": round(100 * mae / y_true.mean(), 1),
    }


def main() -> None:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["hour_local"])
    train = df[df["month"] < FINAL_TEST_MONTH]
    test = df[df["month"] == FINAL_TEST_MONTH]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    print(f"Train: {len(train):,} hours ({train['hour_local'].min():%Y-%m-%d} to "
          f"{train['hour_local'].max():%Y-%m-%d})")
    print(f"Test : {len(test):,} hours ({test['hour_local'].min():%Y-%m-%d} to "
          f"{test['hour_local'].max():%Y-%m-%d})\n")

    predictions, results = {}, []

    # 1. Baseline: no weather, just "typical" trips for this hour and day type
    typical = train.groupby(["hour", "is_weekend"])[TARGET].mean()
    keys = list(zip(X_test["hour"], X_test["is_weekend"]))
    predictions["Baseline (hour x weekend)"] = np.array([typical[k] for k in keys])

    # 2. Ridge regression: hour and weekday are one-hot encoded (daily pattern is not linear)
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["hour", "day_of_week"]),
        ("num", StandardScaler(), WEATHER + ["is_weekend", "is_holiday"]),
    ])
    ridge = make_pipeline(preprocess, Ridge(alpha=1.0))
    ridge.fit(X_train, y_train)
    predictions["Ridge regression"] = ridge.predict(X_test)

    # 3. Random Forest
    forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                   n_jobs=-1, random_state=SEED)
    forest.fit(X_train, y_train)
    predictions["Random Forest"] = forest.predict(X_test)

    # 4. Gradient Boosting
    boosting = HistGradientBoostingRegressor(**GB_PARAMS)
    boosting.fit(X_train, y_train)
    predictions["Gradient Boosting"] = boosting.predict(X_test)

    for name, pred in predictions.items():
        results.append(evaluate(name, y_test.values, np.clip(pred, 0, None)))

    results_df = pd.DataFrame(results).set_index("model")
    results_df.to_csv(MODEL_RESULTS_CSV)
    print(results_df.to_string(), "\n")

    # Chart 5: actual vs predicted, two weeks of August, best model
    best = results_df["MAE"].drop("Baseline (hour x weekend)").idxmin()
    window = test["hour_local"].between("2026-08-10", "2026-08-23 23:00")
    plt.figure(figsize=(12, 5))
    plt.plot(test.loc[window, "hour_local"], y_test[window], label="Actual", color="black", lw=1.5)
    plt.plot(test.loc[window, "hour_local"], pd.Series(predictions[best], index=test.index)[window],
             label=f"Predicted ({best})", color="#e76f51", lw=1.2)
    plt.plot(test.loc[window, "hour_local"],
             pd.Series(predictions["Baseline (hour x weekend)"], index=test.index)[window],
             label="Baseline", color="#2a9d8f", lw=1, alpha=0.7)
    plt.title("BIXI trips per hour: actual vs predicted (Aug 10-23, 2026)")
    plt.ylabel("Trips per hour")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "5_actual_vs_predicted.png", dpi=150)
    plt.close()

    # Chart 6: which features matter? (permutation importance on the test set)
    imp = permutation_importance(forest, X_test, y_test, n_repeats=5,
                                 random_state=SEED, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    ax = importance.plot(kind="barh", figsize=(8, 5), color="steelblue")
    ax.set_title("Feature importance (Random Forest, permutation on August)")
    ax.set_xlabel("Drop in R2 when the feature is shuffled")
    plt.tight_layout()
    plt.savefig(FIGURES / "6_feature_importance.png", dpi=150)
    plt.close()

    print("Top features:")
    print(importance.sort_values(ascending=False).head(5).round(3).to_string())
    print(f"\nBest model: {best}")
    print(f"Charts saved in: {FIGURES}")


if __name__ == "__main__":
    main()
