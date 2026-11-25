"""
Automatic checks for the BIXI x weather project.

They run on the processed CSVs and the SQL schema stored in the repository:
no internet, no BIXI raw file and no database are needed.

Run:  py -m pytest
"""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SCHEMA = (ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")


def load_script(name: str):
    """Import a pipeline script whose file name starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "features_hourly.csv", parse_dates=["hour_local"])


@pytest.fixture(scope="module")
def scenarios() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "scenario_predictions.csv")


# --------------------------------------------------------------- features table
def test_features_have_no_missing_values(features):
    assert features.isna().sum().sum() == 0


def test_features_hours_are_unique_and_sorted(features):
    assert features["hour_local"].is_unique
    assert features["hour_local"].is_monotonic_increasing


def test_features_cover_april_to_august(features):
    assert set(features["month"]) == {4, 5, 6, 7, 8}
    assert features["hour_local"].min() >= pd.Timestamp("2026-04-01")
    assert features["hour_local"].max() < pd.Timestamp("2026-09-01")


def test_features_value_ranges(features):
    assert features["trips"].ge(0).all()
    assert features["hour"].between(0, 23).all()
    assert features["day_of_week"].between(0, 6).all()
    assert features["PRECIP_AMOUNT"].ge(0).all()
    assert features["RELATIVE_HUMIDITY"].between(0, 100).all()


def test_calendar_columns_match_the_timestamp(features):
    assert (features["hour"] == features["hour_local"].dt.hour).all()
    assert (features["day_of_week"] == features["hour_local"].dt.dayofweek).all()
    assert (features["is_weekend"] == (features["day_of_week"] >= 5).astype(int)).all()


def test_known_holidays_are_flagged(features):
    holidays = features.loc[features["is_holiday"] == 1, "hour_local"].dt.date.astype(str)
    assert set(holidays) == {"2026-04-03", "2026-05-18", "2026-06-24", "2026-07-01"}


# ------------------------------------------------------------ scenario table
def test_scenario_grid_is_complete(scenarios):
    # 33 temperatures x 3 rain levels x 24 hours x 7 days
    assert len(scenarios) == 33 * 3 * 24 * 7
    key = ["temp_c", "rain_label", "hour", "day_of_week"]
    assert not scenarios.duplicated(key).any()


def test_scenario_predictions_are_sensible(scenarios):
    assert scenarios["predicted_trips"].ge(0).all()
    assert set(scenarios["rain_label"]) == {
        "No rain", "Light rain (1 mm/h)", "Heavy rain (3 mm/h)"}


def test_reliability_flag_follows_the_threshold(scenarios):
    assert ((scenarios["n_similar_hours"] >= 30).astype(int)
            == scenarios["is_reliable"]).all()


def test_rain_lowers_demand_in_reliable_scenarios(scenarios):
    """At 22 degrees on a Wednesday, dry weather must beat rain."""
    day = scenarios[(scenarios["temp_c"] == 22) & (scenarios["day_of_week"] == 2)]
    total = day.groupby("rain_label")["predicted_trips"].sum()
    assert total["No rain"] > total["Light rain (1 mm/h)"] > total["Heavy rain (3 mm/h)"]


# ------------------------------------------------------------- model results
def test_gradient_boosting_is_the_best_model():
    results = pd.read_csv(PROCESSED / "model_results.csv").set_index("model")
    assert results["MAE"].idxmin() == "Gradient Boosting"
    assert results.loc["Gradient Boosting", "MAE"] < results.loc[
        "Baseline (hour x weekend)", "MAE"]


def test_weather_improves_the_forecast():
    ablation = pd.read_csv(PROCESSED / "ablation_results.csv").set_index("feature_set")
    assert (ablation.loc["Calendar + temperature + rain", "MAE_mean"]
            < ablation.loc["Calendar + temperature", "MAE_mean"]
            < ablation.loc["Calendar only", "MAE_mean"])


def test_rolling_validation_beats_naive_baselines():
    results = pd.read_csv(PROCESSED / "validation_results.csv")
    sim = results[results["model"].str.contains("simulator")].set_index("test_month")["MAE"]
    naive = results[results["model"] == "Naive: same hour last week"].set_index(
        "test_month")["MAE"]
    assert (sim < naive).all()


# ------------------------------------------------------------ helper functions
def test_conformal_margin_covers_the_requested_share():
    module = load_script("10_prediction_intervals")
    scores = np.arange(1, 101, dtype=float)          # 1 ... 100
    margin = module.conformal_margin(scores, alpha=0.10)
    assert (scores <= margin).mean() >= 0.90


def test_conformal_margin_never_exceeds_the_largest_score():
    module = load_script("10_prediction_intervals")
    scores = np.array([3.0, 1.0, 2.0])
    assert module.conformal_margin(scores, alpha=0.01) == 3.0


def test_last_week_feature_looks_back_168_hours():
    module = load_script("09_validate_models")
    hours = pd.date_range("2026-04-01", periods=24 * 9, freq="h")
    df = pd.DataFrame({"hour_local": hours, "trips": np.arange(len(hours))})
    out = module.add_last_week(df)
    assert out["trips_last_week"].iloc[:168].isna().all()
    assert (out["trips_last_week"].iloc[168:].to_numpy()
            == out["trips"].iloc[:-168].to_numpy()).all()


# ---------------------------------------------------------------- SQL schema
def table_columns(sql: str, table: str) -> list[str]:
    body = re.search(rf"CREATE TABLE {table} \((.*?)\n\);", sql, re.S).group(1)
    names = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("--", "PRIMARY KEY")):
            continue
        names.append(line.split()[0])
    return names


@pytest.mark.parametrize("table", [
    "weather_hourly", "bixi_hourly", "features_hourly", "model_results",
    "ablation_results", "scenario_predictions"])
def test_csv_columns_match_the_sql_tables(table):
    loader = load_script("08_load_postgres")
    frame = loader.build_jobs()[table]
    assert list(frame.columns) == table_columns(SCHEMA, table)


def test_scenario_view_exposes_rain_order():
    view = re.search(r"CREATE VIEW vw_scenarios AS(.*?);", SCHEMA, re.S).group(1)
    assert "AS rain_order" in view


# ------------------------------------------------- scenario prediction intervals
def test_scenario_intervals_surround_the_forecast():
    intervals = pd.read_csv(PROCESSED / "scenario_intervals.csv")
    assert len(intervals) == 33 * 3 * 24 * 7
    assert (intervals["lower_trips"] >= 0).all()
    assert (intervals["lower_trips"] <= intervals["predicted_trips"]).all()
    assert (intervals["predicted_trips"] <= intervals["upper_trips"]).all()


def test_scenario_intervals_keep_the_point_forecasts():
    intervals = pd.read_csv(PROCESSED / "scenario_intervals.csv")
    base = pd.read_csv(PROCESSED / "scenario_predictions.csv")
    assert (intervals["predicted_trips"].to_numpy()
            == base["predicted_trips"].to_numpy()).all()