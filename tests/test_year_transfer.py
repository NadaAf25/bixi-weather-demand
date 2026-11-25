"""The 2025 test year is clean and never overlaps the training year."""
import pandas as pd

from config import FEATURES_2025_CSV, FEATURES_CSV, HOLIDAYS_2025, YEAR_TRANSFER_MONTHS


def _load(path):
    return pd.read_csv(path, parse_dates=["hour_local"])


def test_2025_features_are_clean():
    df = _load(FEATURES_2025_CSV)
    assert df.isna().sum().sum() == 0
    assert df["hour_local"].is_unique
    assert (df["hour_local"].dt.year == 2025).all()


def test_test_year_never_overlaps_the_training_year():
    train, test = _load(FEATURES_CSV), _load(FEATURES_2025_CSV)
    assert set(train["hour_local"].dt.year) == {2026}
    assert not set(train["hour_local"]) & set(test["hour_local"])


def test_2025_holidays_are_flagged():
    df = _load(FEATURES_2025_CSV)
    flagged = set(df.loc[df["is_holiday"] == 1, "hour_local"].dt.strftime("%Y-%m-%d"))
    present = set(df["hour_local"].dt.strftime("%Y-%m-%d"))
    assert flagged == set(HOLIDAYS_2025) & present


def test_every_month_of_the_comparison_is_present():
    assert set(YEAR_TRANSFER_MONTHS) <= set(_load(FEATURES_2025_CSV)["month"])