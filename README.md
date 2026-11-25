# BIXI x Weather: how much does the weather drive bike-share demand in Montreal?

[![tests](https://github.com/NadaAf25/bixi-weather-demand/actions/workflows/tests.yml/badge.svg)](https://github.com/NadaAf25/bixi-weather-demand/actions/workflows/tests.yml)

An end-to-end data project: from a 2 GB open-data file and a weather API to a PostgreSQL
database, a machine-learning model and a Power BI dashboard where you can ask
*"if it's 22 °C and raining, how many trips will there be?"*

![Architecture](docs/architecture.png)

## Key findings

Data: 9.1 million BIXI trips over 153 days (1 April to 31 August 2026), joined to hourly
weather from Montreal-Trudeau airport (3,669 hours).

| Question | Answer |
|---|---|
| Does rain matter? | Yes. Between 7 am and 9 pm there are **1,754 trips/hour when it rains vs 3,589 when it is dry** (-51%). Within the same 5 °C band the drop is still 31% to 49% (e.g. 15 to 20 °C: 1,863 vs 3,326), so it is not just a temperature effect. |
| How much does weather improve a forecast? | A model that knows only the hour, day of week and holidays has a mean error of **925 trips/hour**. Adding **temperature** alone cuts it by **52%**, adding **rain** brings it to **57%**, and all weather variables to **58%** (388 trips/hour). |
| Is weather enough on its own? | No. Weather without calendar features is *worse* than calendar alone (1,332 vs 925): the commute rhythm explains the shape of the day, the weather explains how high the peaks are. |
| Best model? | Gradient Boosting: **MAE 358 trips/hour (11.9% of the average hour), R² 0.94** on August, a month the model never saw. A "typical hour x weekday/weekend" baseline gets 837. |
| Does the weather station matter? | Not materially. Downtown (McTavish) and the airport give the same mean error over three test months: **480.0 vs 479.7 trips/hour**. |

![Weather ablation](figures/7_weather_ablation.png)

### Is the model good only in August?

To check that August was not a lucky month, the model is validated on three successive
test months, each time trained only on the months before (April-May to June, April-June
to July, April-July to August), and compared with two naive baselines.

| Mean error (trips/hour), lower is better | June | July | August |
|---|---|---|---|
| Gradient Boosting, calendar + temperature + rain (the simulator model) | **638** | **405** | **396** |
| Naive: same hour last week | 815 | 521 | 492 |
| Naive: average hour x weekday | 1,223 | 1,076 | 796 |

The model beats both baselines in all three months. June is the weakest month because the
model has seen only two months of training data, including April, when the network was
still ramping up. The average bias of the simulator model is small compared with its error
(-41, +17 and -47 trips/hour). The "same hour last week" baseline uses real past trips,
which a what-if simulator cannot know: it is a yardstick for forecasting, not an
alternative to the simulator.

![Rolling validation](figures/9_rolling_validation.png)

### How uncertain is a prediction?

A single number hides the uncertainty, so the model also gives 90% prediction intervals,
built with conformal prediction (conformalised quantile regression). The model is trained
on April to June, the intervals are sized on July, and their real coverage is checked on
August, a month neither step has seen.

| Hours in August | Actual values inside the interval (target: 90%) | Average width (trips/hour) |
|---|---|---|
| All hours | 90.7% | 2,015 |
| Weekday rush hours | 90.5% | 3,114 |
| Dry daytime hours | 86.8% | 2,346 |
| Rainy hours | **83.3%** | 2,096 |
| Night (0-5 am) | 99.5% | 1,157 |

Overall the intervals deliver what they promise, and their width adapts: narrow at night,
wide at rush hour. During rain they are too optimistic (83% instead of 90%): rain is rare
(8% of hours), so the model has less to learn from. A simpler constant-width interval
reaches 91.9% overall but is just as wide at night (1,599 trips/hour) as it is by day.
The same method gives the intervals shown in the Power BI simulator, as a lower-bound line
and an upper-bound line on the hourly curve (see below).

![Prediction intervals](figures/10_prediction_intervals.png)

### Where does the model make its errors?

The errors of the three validation months (June, July, August; 2,206 hours) are pooled and
split by time of day, rain and temperature (`src/13_error_analysis.py`, simulator model,
trips per hour, bias = prediction minus actual).

| Group | Hours | Average real trips | MAE | RMSE | Bias | MAE as % of average |
|---|---:|---:|---:|---:|---:|---:|
| **All test hours** | 2,206 | 2,966 | 478 | 715 | -24 | 16.1% |
| Night (0-5 am) | 552 | 649 | 166 | 254 | -25 | 25.5% |
| Morning (6-9 am) | 368 | 2,414 | 470 | 710 | +92 | 19.5% |
| Midday (10 am-3 pm) | 552 | 3,575 | 525 | 779 | -61 | 14.7% |
| Evening rush (4-6 pm) | 275 | 6,217 | 752 | 984 | -33 | 12.1% |
| Evening (7-11 pm) | 459 | 3,513 | 640 | 815 | -64 | 18.2% |
| Dry | 2,045 | 3,066 | 438 | 648 | -78 | 14.3% |
| **Rain** | 161 | 1,694 | 992 | 1,290 | **+665** | 58.6% |
| Dry, daytime (7 am-9 pm) | 1,275 | 4,235 | 550 | 758 | -105 | 13.0% |
| **Rain, daytime (7 am-9 pm)** | 103 | 2,278 | 1,341 | 1,568 | **+896** | 58.9% |
| Below 10 C (5 hours only) | 5 | 360 | 144 | 181 | -80 | 40.0% |
| 10-15 C | 66 | 954 | 255 | 426 | -220 | 26.7% |
| 15-20 C | 681 | 1,705 | 472 | 759 | +111 | 27.7% |
| 20-25 C | 982 | 3,028 | 517 | 747 | -75 | 17.1% |
| 25 C and above | 472 | 4,964 | 442 | 609 | -83 | 8.9% |

- **Time of day.** The largest absolute error is at the evening rush (752 trips/hour), where
  demand is highest; relative to demand it is the smallest (12%). At night the error is small
  in absolute terms (166) but represents 25% of the few trips there are.
- **Rain is the weak spot.** On rainy hours the model predicts on average 665 trips/hour
  too many (896 during the day), and its error is about 2.3 times that of dry hours. In other
  words, it underestimates how much rain reduces demand. This matches the intervals that
  cover only 83% of rainy hours. Rain is rare (161 hours here), so this is a limitation of
  the training data, not proof of a systematic effect. The next section tests two fixes.
- **Temperature.** Errors are relatively largest in cool weather (15-20 C: 28% of the
  average, with a bias of +111) and smallest above 25 C (9%). The two coldest bands have
  too few hours (5 and 66) to conclude anything.

### Can the rain error be fixed?

The error analysis shows that the model predicts too many trips when it rains. I tested
two changes (`src/15_rain_experiment.py`) on the same three rolling folds: forcing rain to
never increase demand (a monotonic constraint), and adding recent-rain features
(`rain_3h` and `rained_prev_hour`, built from weather only, never from trips).

| Model (trips/hour) | Bias, rainy hours | Bias, rainy daytime | MAE, rainy hours | MAE, all hours |
|---|---:|---:|---:|---:|
| Simulator model (current) | +665 | +896 | 992 | 478 |
| + monotone rain | +634 | +845 | 981 | 473 |
| + monotone rain + recent rain | +581 | +787 | 924 | 467 |

- **Both changes help a little, neither fixes it.** The bias on rainy hours falls by 5%
  with the constraint and by 13% with recent rain, but the model still predicts about 580
  trips/hour too many when it rains.
- **There is a cost on dry hours.** Their bias goes from -78 to -90 trips/hour, and the
  August bias from -47 to -81.
- **The simulator is unchanged.** The constraint alone improves the overall error by 1%,
  which does not justify regenerating the scenarios, and recent rain cannot be used by a
  what-if simulator, which has no rain history.
- **The cause is not established.** With 161 rainy hours and no significance test, I cannot
  tell whether the model lacks data or misses something else, such as rain intensity.

### Do lagged demand features help?

I tested whether knowing past demand improves the forecast (`src/14_lag_experiment.py`).
Same model, same rolling-origin folds; only the features change. Every lag is computed from
earlier hours only (tested in `tests/test_lags.py`), and the first week of April has no
`lag_168`. Hours without a complete history (263 of 3,669) are dropped for **all** models, so
the numbers below are not comparable with the other tables of this page, only with each other.

| Features (MAE, trips/hour) | June | July | August | Mean | vs simulator model |
|---|---:|---:|---:|---:|---:|
| One hour ahead: + lag_1, lag_24, lag_168, 3 h and 24 h means | 395.2 | 325.3 | 251.6 | 324.0 | -29.8% |
| Simulator model (calendar + temperature + rain) | 598.1 | 408.5 | 378.5 | 461.7 | - |
| Day ahead: + lag_24, lag_168, 24 h mean | 620.6 | 490.1 | 419.9 | 510.2 | +10.5% |
| Day ahead, calendar + lags, **no weather** | 706.8 | 857.8 | 584.7 | 716.4 | +55.2% |

- **One hour ahead, lags help a lot (-30%).** The last hour of demand already contains the
  current weather, so it is a strong predictor. This is a short-term forecasting tool: it
  needs real trips, which do not exist for a future hour.
- **A day ahead, lags do not help.** Adding `lag_24`, `lag_168` and a 24 h mean makes the
  mean error 10% worse. A plausible reason is that yesterday's and last week's demand
  reflect their own weather, not the weather being asked about, and that three months of
  training data are too few to sort this out. I did not test this explanation.
- **Past demand cannot replace the weather.** Without the weather variables, the day-ahead
  model is 55% worse than the simulator model. Over a 24 h horizon, the weather says more
  about demand than yesterday's demand does.
- **The simulator keeps no lag.** It answers "what if it rained?" for hours that have not
  happened, so `lag_1` and the recent means do not exist for it, and a lag would hide the
  very effect of the weather it is meant to measure.

### Does the weather station matter?

The model uses a single station, Montreal-Trudeau airport (about 14 km from downtown).
To test this choice, I downloaded the McTavish station (downtown, about 1 km from the
centre) and compared both stations over April to August 2026
(`src/12_station_comparison.py`).

**Observations.** Hourly temperatures are almost identical (correlation 0.99, mean
absolute difference 0.64 °C, maximum 7.6 °C). Rain differs more: McTavish records 350
rainy hours against 303 at the airport (575 mm against 533 mm), and the two stations
agree on whether it rains in 95.3% of the hours.

**Model.** Same model (calendar + temperature + rain), same rolling-origin folds; only
the weather columns change.

| Station | MAE June | MAE July | MAE August | Mean MAE | Mean RMSE | Mean bias |
|---|---:|---:|---:|---:|---:|---:|
| Airport | 638.4 | 405.3 | 396.3 | 480.0 | 705.4 | -24.0 |
| McTavish | 631.4 | 408.6 | 399.0 | 479.7 | 718.9 | -37.6 |

(trips per hour. The airport row differs by less than 0.5 from the validation table above
because that script clips negative predictions to zero and this one does not.)

**Conclusion.** The station choice does not materially change the performance: the mean
MAE differs by 0.3 trips/hour, and the sign of the difference changes from one month to
the next. The airport station is kept. The error on rainy hours is too noisy to compare
stations (303 to 350 rainy hours in total, fewer in each fold), and there is no
significance test.

### The simulator

The dashboard answers what-if questions from a table of 16,632 pre-computed model
predictions (temperature x rain x hour x day of week). For a Wednesday at 22 °C:

| Scenario (the weather holds all day) | Predicted trips |
|---|---|
| Dry | 72,600 |
| Light rain (1 mm/h) | 40,100 |
| Heavy rain (3 mm/h) | 33,300 |

For reference, the average real day in the data has 59,400 trips. "Rain all day" is an
extreme scenario, so read these as a sensitivity to rain, not as a weather-forecast product.
Scenarios with too few comparable real hours (e.g. rain at 30 °C: it almost never rained
above 25 °C) are flagged as unreliable in the dashboard.

Every scenario also carries a 90% prediction interval (`lower_trips`, `upper_trips`), built
with the same method (quantile models trained on April to June, margin calibrated on July).
The simulator's point forecast comes from a model trained on all the data, so
`11_scenario_intervals.py` stretches an interval when needed so that it always contains that
forecast: 567 of the 16,632 scenarios (3.4%) were adjusted. The average interval is 1,943
trips/hour wide: 1,371 at night (0-5 am) and 2,658 at rush hours (7 to 9 and 16 to 18). The
90.7% coverage above was measured for the April-to-June setup on August; the simulator's own
intervals, once stretched, were not measured again.

For a Wednesday at 8 am and 22 °C:

| Scenario | Lower bound | Predicted trips | Upper bound |
|---|---|---|---|
| Dry | 1,695 | 5,528 | 6,171 |
| Light rain (1 mm/h) | 858 | 2,479 | 5,186 |
| Heavy rain (3 mm/h) | 358 | 2,071 | 4,943 |

In the dashboard, the two bound lines follow the selected rain level, while the "dry" line
always shows the dry scenario. At 8 am the dry forecast (5,528) is therefore above the upper
bound of the light-rain interval (5,186): the two lines belong to different scenarios, not to
the same forecast.

![Scenario curves](figures/8_scenario_curves.png)

## Dashboard (Power BI)

Three pages: **Overview** (daily trips, a typical day, trips by temperature and rain),
**Simulator** (pick temperature, rain and day: get a headline number, an hour-by-hour curve
with its 90% prediction interval and a reliability warning) and **Model** (model comparison
and ablation study).
The report reads the `vw_*` views of the PostgreSQL schema.
The build steps and every DAX measure are in [`powerbi/BUILD_GUIDE.md`](powerbi/BUILD_GUIDE.md).

**Overview**

![Overview](docs/powerbi_overview.png)

**Simulator**

![Simulator](docs/powerbi_simulator.png)

**Model**

![Model](docs/powerbi_model.png)

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `01_extract_weather.py` | Downloads hourly observations from the ECCC GeoMet API (pagination, automatic retries). Rebuilds Montreal local time from UTC, because the API's local timestamp ignores daylight saving. Takes the airport station by default; `--station` and `--out` fetch another one without touching the pipeline's input. |
| 2 | `02_aggregate_bixi.py` | Streams the ~2 GB BIXI file in 1M-row chunks, drops trips under 1 minute or over 24 hours, counts trips per hour. Reads straight from the ZIP. |
| 3 | `03_build_features.py` | Joins trips and weather, adds hour, weekday, weekend and Quebec holiday flags. An hour without any trip becomes 0, not missing. |
| 4 | `04_explore.py` | Exploratory charts 1 to 4. |
| 5 | `05_train_model.py` | Baseline, Ridge, Random Forest, Gradient Boosting. Time-based split (train April to July, test August), no shuffling. |
| 6 | `06_weather_ablation.py` | Same model, different feature sets, two test months (July and August). |
| 7 | `07_build_scenarios.py` | Scores the model on the what-if grid and flags scenarios with little supporting data. |
| 8 | `08_load_postgres.py` | Applies `sql/schema.sql`, bulk-loads every table with `COPY` in a single transaction, then runs quality checks. It loads the scenarios from `scenario_intervals.csv`, so step 11 must have run first. |
| 9 | `09_validate_models.py` | Rolling-origin validation on three test months (June, July, August) against two naive baselines. Does not feed the database. |
| 10 | `10_prediction_intervals.py` | 90% prediction intervals (conformalised quantile regression) and a check of their real coverage on August. Does not feed the database. |
| 11 | `11_scenario_intervals.py` | Adds a 90% prediction interval (low and high bound) to each of the 16,632 scenarios, with the same method as step 10, and writes `scenario_intervals.csv`, which step 8 loads. |
| 12 | `12_station_comparison.py` | Compares the airport and McTavish stations (observations, then model error on the same three rolling folds). Needs `weather_mctavish.csv` (see "Run it"). Does not feed the database. |
| 13 | `13_error_analysis.py` | Pools the out-of-sample errors of the three validation months and summarises MAE, RMSE and bias by time of day, rain and temperature band. Does not feed the database. |
| 14 | `14_lag_experiment.py` | Tests lagged-demand features (day ahead and one hour ahead) on the same rolling folds as step 9, built by `src/lags.py` from earlier hours only. Does not feed the database. |
| 15 | `15_rain_experiment.py` | Tests a monotonic constraint on rain and recent-rain features (`src/rain_features.py`, from weather only) on the same rolling folds as step 9, and reports the bias on rainy hours. Does not feed the database. |

All paths, model settings, test months and the what-if grid are defined once in
`src/config.py`, which every script imports. `src/conformal.py` holds the interval
margin shared by steps 10 and 11, `src/lags.py` the lag features of step 14 and
`src/rain_features.py` the recent-rain features of step 15.

### Database

`sql/schema.sql` creates the `bixi` schema: staging tables (`weather_hourly`, `bixi_hourly`),
the modelling table (`features_hourly`), model outputs (`model_results`, `ablation_results`,
`scenario_predictions`, which includes the interval bounds `lower_trips` and `upper_trips`)
and five reporting views. Primary keys, `CHECK` constraints and
NOT NULL rules reject bad data at load time. The loader is idempotent (re-run it any time) and
rolls back everything if a check fails. Checks include row counts against the CSVs, orphan
hours, and a trip-count reconciliation: 236,330 trips fall before the weather period starts
(January to March) and 11,298 fall in three hours with no weather measurement.

### Tests

`tests/` holds 40 automatic checks. 37 of them run in a few seconds without internet, the
large BIXI file or a database. The 25 checks of `tests/test_pipeline.py` verify, for example,
that `features_hourly.csv` has no missing values and unique hours, that the simulator grid
has exactly 16,632 rows (33 x 3 x 24 x 7), that at 22 °C dry weather beats light rain, which
beats heavy rain, that the columns of every CSV match the tables of `schema.sql`, that the
model beats the naive baselines in all three validation months, and that every scenario
interval contains its point forecast. The 5 checks of `tests/test_config.py` verify that
every setting lives in `src/config.py` (no script copies it) and that every dependency is
pinned to an exact version. The 3 checks of `tests/test_lags.py` verify that lag features
only use earlier hours and that a missing hour gives a missing value instead of a shifted
one. The 4 checks of `tests/test_rain_features.py` verify that the recent-rain features use
only the current and earlier hours, that a missing hour gives a missing value, and that the
constrained model never predicts more trips with more rain.

The 3 checks of `tests/test_postgres_integration.py` load the real CSVs into a PostgreSQL 16
service with `08_load_postgres.py` (twice, to prove that it is idempotent), then verify the
row counts of every table, that the five `vw_*` views return the expected rows, and that the
database rejects a negative trip count and a duplicate key. They run in GitHub Actions and are
skipped locally unless `TEST_DATABASE_URL` is set: the loader drops the `bixi` schema, so it
must never point at a database you care about.

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs everything on every push,
which is what the badge at the top of this page shows.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Run it

Dependency versions are pinned in `requirements.txt` (tested with Python 3.14, the version
the CI uses).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# The processed CSVs are in the repo, so steps 1-2 are optional.
# Step 2 needs the BIXI trips file from BIXI's open-data page in data/raw/.
python src/01_extract_weather.py --years 2026
python src/02_aggregate_bixi.py
python src/03_build_features.py
python src/04_explore.py
python src/05_train_model.py
python src/06_weather_ablation.py
python src/07_build_scenarios.py
python src/09_validate_models.py                       # optional: rolling validation
python src/10_prediction_intervals.py                  # optional: prediction intervals
python src/11_scenario_intervals.py                    # intervals for the simulator (needed by step 8)
python src/13_error_analysis.py                        # optional: errors by hour, rain, temperature
python src/14_lag_experiment.py                        # optional: lagged-demand features
python src/15_rain_experiment.py                       # optional: rain bias experiment

# optional: weather station comparison (downloads the McTavish station first)
python src/01_extract_weather.py --station 7024745 --out data/processed/weather_mctavish.csv
python src/12_station_comparison.py

docker compose up -d                                   # local PostgreSQL 16
python src/08_load_postgres.py                         # or --dsn / DATABASE_URL for another server

pip install -r requirements-dev.txt && python -m pytest   # optional: run the automatic tests
```

Always pass `--out` when you use `--station`: without it, `weather_hourly.csv`, the input
of the whole pipeline, would be overwritten with another station's data.

Already have PostgreSQL installed? Create a `bixi` user and database, then pass the
connection string (adjust the port if yours is not 5432):

```bash
python src/08_load_postgres.py --dsn postgresql://bixi:bixi@localhost:5433/bixi
```

Then open Power BI Desktop and follow [`powerbi/BUILD_GUIDE.md`](powerbi/BUILD_GUIDE.md).
(On Windows, `py` works instead of `python`.)

## Repository layout

```
src/                pipeline scripts 01 to 15, config.py (shared settings), conformal.py, lags.py, rain_features.py
tests/              test_pipeline.py, test_config.py, test_lags.py, test_rain_features.py, test_postgres_integration.py, conftest.py (40 checks)
.github/workflows/  tests.yml (runs the tests on every push, with a PostgreSQL 16 service)
sql/                schema.sql (tables, constraints, views)
powerbi/            build guide, DAX measures, theme, bixi_weather.pbix
data/
  raw/              BIXI file (not in git)
  processed/        all intermediate and result CSVs
figures/            charts 1 to 10
docs/               architecture diagram and Power BI screenshots
docker-compose.yml
requirements.txt
requirements-dev.txt
```

## Design choices worth knowing

- **Time-based validation.** Hours are strongly autocorrelated, so a random split would leak
  the future into training. The model is always tested on a later month.
- **Why the simulator uses temperature and rain only.** In the ablation study, calendar +
  temperature + rain (MAE 401) is almost as good as all weather variables (388), and, unlike
  humidity or pressure, a person can set both in a what-if question.
- **No lagged demand in the simulator.** Lags need real past trips, which a what-if question
  about future hours cannot provide. They are tested separately as a forecasting experiment.
- **Model scored in Python, served in SQL.** Power BI cannot run scikit-learn, so predictions
  are pre-computed and stored. Re-run steps 7, 11 and 8 whenever the model changes.
- **Intervals always contain the forecast.** The interval models are trained on April to June,
  while the point forecast uses all the data. Where the two disagree (3.4% of the scenarios),
  the interval is stretched so that the forecast never falls outside its own band.
- **Sort orders come from the database.** The `vw_scenarios` view exposes a `rain_order`
  column, so Power BI sorts the rain levels (no rain, light, heavy) without a calculated column.
- **One place for every setting, exact versions.** Changing a path, the random seed or a
  model setting means editing `src/config.py` only, and the pinned versions give the same
  results on another machine. A test fails if a script copies a setting.

## Limitations

- **One season, one year.** Only April to August 2026 is covered: nothing about autumn, winter
  or other years. BIXI also ramps up through April (chart 4), so part of the "temperature
  effect" is really seasonality and network growth. A longer history would separate them.
- **One weather station for the whole city.** The pipeline uses the airport, not downtown.
  Comparing it with McTavish (downtown) changed the mean error by 0.3 trips/hour, so the choice
  does not matter for this model, but that was tested on one season and without a significance
  test. Trips are counted for the whole system, not per station or neighbourhood, so local
  weather differences between neighbourhoods are not captured.
- **Feature importance is correlated.** In chart 6 humidity ranks above temperature, which
  probably reflects that it tracks rain and time of day. The ablation study is the cleaner
  test of what each variable adds.
- **Rain is rare** (8% of hours), so rain estimates are noisier than temperature estimates,
  and a tree model cannot extrapolate outside the conditions it has seen.
- **Intervals are too optimistic in the rain** (83% coverage instead of 90%), and they are
  checked on a single month (August). The simulator's own intervals, stretched to contain the
  final model's forecast, were not measured again.
- **The model overestimates demand in the rain.** On rainy hours it predicts 665 trips/hour
  too many on average (896 during the day), and its error is about 2.3 times that of dry
  hours. A monotonic constraint and recent-rain features reduce this bias by only 5% and
  13%, so the simulator's rain scenarios probably still understate the real drop in demand,
  and its intervals are too narrow in the rain (83% coverage instead of 90%). Only 161 rainy
  hours are available for the three validation months.
- **The lag experiment is small.** Three feature sets, no tuning, training windows of 1,295
  to 2,710 hours, and 263 hours dropped for lack of history. The result "lags do not help a
  day ahead" may change with more data; it is not proven for other seasons.
- **Short validation history.** Only three successive test months exist (June to August),
  and June is scored with just two months of training data. More years of data would allow
  a longer rolling validation.
- **The database tests use the project's own CSVs.** The integration test loads the real
  outputs into an empty PostgreSQL, but it does not test a load that fails halfway or a
  different schema version.
- **No hyperparameter tuning.** Every model uses the same fixed Gradient Boosting settings
  (`GB_PARAMS` in `src/config.py`), so a tuned model might do better. The typical error on
  August is 358 trips per hour, which is large for quiet hours and small for peak hours.

## Data sources

- Trips: BIXI Montréal open data, https://bixi.com/fr/donnees-ouvertes/ (not redistributed here; download the 2026 file from this page and put it in `data/raw/`).
- Weather: Environment and Climate Change Canada, GeoMet-OGC API, `climate-hourly`
  collection, station Montréal/Pierre Elliott Trudeau INTL (used by the pipeline) and station
  McTavish (used only for the station comparison).

Please check each provider's terms of use before reusing the data.