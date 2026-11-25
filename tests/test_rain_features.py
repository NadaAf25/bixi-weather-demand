"""Rain features use only the current and earlier hours, and the constrained model
never predicts more trips with more rain."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import FEATURES_CSV, GB_PARAMS, SIMULATOR_FEATURES, TARGET
from rain_features import add_rain_features, monotone_constraints


def _frame(hours, rain):
    start = pd.Timestamp("2026-06-01 00:00")
    return pd.DataFrame({"hour_local": [start + pd.Timedelta(hours=h) for h in hours],
                         "PRECIP_AMOUNT": rain})


def test_rain_3h_sums_this_hour_and_the_two_before():
    out = add_rain_features(_frame([0, 1, 2, 3, 4], [0.0, 1.0, 0.0, 2.0, 0.0]))
    assert out["rain_3h"].isna().tolist() == [True, True, False, False, False]
    assert out["rain_3h"].tolist()[2:] == [1.0, 3.0, 2.0]


def test_a_missing_hour_gives_nan_not_a_shorter_window():
    out = add_rain_features(_frame([0, 1, 2, 4, 5], [1.0, 1.0, 1.0, 1.0, 1.0]))
    by_hour = out.set_index("hour_local")
    assert by_hour.loc[pd.Timestamp("2026-06-01 04:00"), "rain_3h"] != by_hour.loc[
        pd.Timestamp("2026-06-01 04:00"), "rain_3h"]          # NaN
    assert np.isnan(by_hour.loc[pd.Timestamp("2026-06-01 04:00"), "rained_prev_hour"])


def test_rained_prev_hour_ignores_the_current_hour():
    a = add_rain_features(_frame([0, 1, 2], [0.0, 1.0, 0.0]))
    b = add_rain_features(_frame([0, 1, 2], [0.0, 1.0, 5.0]))
    assert a["rained_prev_hour"].tolist()[1:] == b["rained_prev_hour"].tolist()[1:] == [0.0, 1.0]


def test_constrained_model_never_predicts_more_trips_with_more_rain():
    df = pd.read_csv(FEATURES_CSV)
    model = HistGradientBoostingRegressor(
        **GB_PARAMS, monotonic_cst=monotone_constraints(SIMULATOR_FEATURES))
    model.fit(df[SIMULATOR_FEATURES], df[TARGET])
    preds = []
    for rain in (0.0, 1.0, 3.0):
        grid = df[SIMULATOR_FEATURES].copy()
        grid["PRECIP_AMOUNT"] = rain
        preds.append(model.predict(grid))
    assert (preds[1] <= preds[0] + 1e-6).all()
    assert (preds[2] <= preds[1] + 1e-6).all()