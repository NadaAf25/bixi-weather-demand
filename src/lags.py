"""
lags.py
Lagged-demand features, built from the PAST only.

Every feature of an hour t uses trips of hours strictly before t, so nothing from the
hour being predicted (or later) leaks into the model. The series is put on a complete
hourly grid first, so a missing hour gives a missing lag (NaN) instead of silently
shifting the others.
"""
import pandas as pd

from config import TARGET

TIME = "hour_local"

# Known 24 hours ahead (day-ahead forecast)
DAY_AHEAD = ["lag_24", "lag_168", "mean_24h_lag24"]
# Known one hour ahead only: lag_1 does not exist for a future hour, so these
# features can never be used by the what-if simulator.
ONE_HOUR_AHEAD = ["lag_1", "lag_24", "lag_168", "mean_3h", "mean_24h"]


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling-mean features computed from earlier hours only."""
    s = df.set_index(TIME)[TARGET]
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    past = s.shift(1)                       # the hour before: never the hour itself
    lags = pd.DataFrame({
        "lag_1": past,
        "lag_24": s.shift(24),
        "lag_168": s.shift(168),
        "mean_3h": past.rolling(3).mean(),               # hours t-3 to t-1
        "mean_24h": past.rolling(24).mean(),             # hours t-24 to t-1
        "mean_24h_lag24": s.shift(24).rolling(24).mean(),  # hours t-47 to t-24
    })
    return df.merge(lags, left_on=TIME, right_index=True, how="left")