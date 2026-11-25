"""
test_lags.py
Lag features must use the past only. No internet, no database, no large file needed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lags import add_lags  # noqa: E402


def make_hours(n=400, missing=None) -> pd.DataFrame:
    """One row per hour; trips = the row number, so every lag is easy to predict."""
    df = pd.DataFrame({"hour_local": pd.date_range("2026-04-01", periods=n, freq="h"),
                       "trips": range(n)})
    return df if missing is None else df.drop(index=missing)


def test_lags_use_only_earlier_hours():
    row = add_lags(make_hours()).iloc[300]
    assert row["lag_1"] == 299
    assert row["lag_24"] == 276
    assert row["lag_168"] == 132
    assert row["mean_3h"] == 298          # mean of hours 297, 298, 299
    assert row["mean_24h"] == 287.5       # mean of hours 276 ... 299
    assert row["mean_24h_lag24"] == 264.5  # mean of hours 253 ... 276


def test_a_missing_hour_gives_nan_not_a_shifted_lag():
    out = add_lags(make_hours(missing=100)).set_index("hour_local")
    start = pd.Timestamp("2026-04-01")
    assert pd.isna(out.loc[start + pd.Timedelta(hours=124), "lag_24"])
    assert out.loc[start + pd.Timedelta(hours=125), "lag_24"] == 101


def test_first_week_has_no_weekly_lag():
    out = add_lags(make_hours())
    assert out["lag_168"].iloc[:168].isna().all()
    assert out["lag_168"].iloc[168:].notna().all()