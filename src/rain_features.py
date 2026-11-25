"""
rain_features.py
Recent-rain features and the monotonic constraint on rain, shared by
15_rain_experiment.py and tests/test_rain_features.py.

Only weather is used (never trips), and only the current and earlier hours,
so nothing from the future leaks into a prediction.
"""
import pandas as pd

from config import RAIN_COLUMNS, RAIN_WINDOW_HOURS

RAIN_FEATURES = ["rain_3h", "rained_prev_hour"]


def add_rain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rain_3h (mm over this hour and the 2 before) and rained_prev_hour (0/1).

    A missing hour in the table gives NaN instead of a value built on fewer hours.
    """
    out = df.sort_values("hour_local").reset_index(drop=True)
    full = out.set_index("hour_local")["PRECIP_AMOUNT"].asfreq("h")
    rain_sum = full.rolling(RAIN_WINDOW_HOURS, min_periods=RAIN_WINDOW_HOURS).sum()
    previous = full.shift(1)
    rained_previous = (previous > 0).astype(float).where(previous.notna())
    out["rain_3h"] = out["hour_local"].map(rain_sum)
    out["rained_prev_hour"] = out["hour_local"].map(rained_previous)
    return out


def monotone_constraints(features: list) -> dict:
    """Decreasing constraint (-1) on every rain column present in `features`."""
    return {f: -1 for f in features if f in RAIN_COLUMNS}