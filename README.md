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
| 1 | `01_extract_weather.py` | Downloads hourly observations from the ECCC GeoMet API (pagination, automatic retries). Rebuilds Montreal local time from UTC, because the API's local timestamp ignores daylight saving. |
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

`tests/test_pipeline.py` holds 25 automatic checks that run in a few seconds, without internet,
the large BIXI file or a database. They verify, for example, that `features_hourly.csv` has no
missing values and unique hours, that the simulator grid has exactly 16,632 rows (33 x 3 x 24 x 7),
that at 22 °C dry weather beats light rain, which beats heavy rain, that the columns of every CSV
match the tables of `schema.sql`, that the model beats the naive baselines in all three
validation months, and that every scenario interval contains its point forecast. A GitHub
Actions workflow (`.github/workflows/tests.yml`) runs them on every push, which is what the
badge at the top of this page shows.

```bash
pip install pytest
python -m pytest
```

## Run it

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

docker compose up -d                                   # local PostgreSQL 16
python src/08_load_postgres.py                         # or --dsn / DATABASE_URL for another server

pip install pytest && python -m pytest                 # optional: run the automatic tests
```

Already have PostgreSQL installed? Create a `bixi` user and database, then pass the
connection string (adjust the port if yours is not 5432):

```bash
python src/08_load_postgres.py --dsn postgresql://bixi:bixi@localhost:5433/bixi
```

Then open Power BI Desktop and follow [`powerbi/BUILD_GUIDE.md`](powerbi/BUILD_GUIDE.md).
(On Windows, `py` works instead of `python`.)

## Repository layout

```
src/                pipeline scripts 01 to 11
tests/              test_pipeline.py (25 automatic checks)
.github/workflows/  tests.yml (runs the tests on every push)
sql/                schema.sql (tables, constraints, views)
powerbi/            build guide, DAX measures, theme, bixi_weather.pbix
data/
  raw/              BIXI file (not in git)
  processed/        all intermediate and result CSVs
figures/            charts 1 to 10
docs/               architecture diagram and Power BI screenshots
docker-compose.yml
requirements.txt
```

## Design choices worth knowing

- **Time-based validation.** Hours are strongly autocorrelated, so a random split would leak
  the future into training. The model is always tested on a later month.
- **Why the simulator uses temperature and rain only.** In the ablation study, calendar +
  temperature + rain (MAE 401) is almost as good as all weather variables (388), and, unlike
  humidity or pressure, a person can set both in a what-if question.
- **Model scored in Python, served in SQL.** Power BI cannot run scikit-learn, so predictions
  are pre-computed and stored. Re-run steps 7, 11 and 8 whenever the model changes.
- **Intervals always contain the forecast.** The interval models are trained on April to June,
  while the point forecast uses all the data. Where the two disagree (3.4% of the scenarios),
  the interval is stretched so that the forecast never falls outside its own band.
- **Sort orders come from the database.** The `vw_scenarios` view exposes a `rain_order`
  column, so Power BI sorts the rain levels (no rain, light, heavy) without a calculated column.

## Limitations

- **One season, one year.** Only April to August 2026 is covered: nothing about autumn, winter
  or other years. BIXI also ramps up through April (chart 4), so part of the "temperature
  effect" is really seasonality and network growth. A longer history would separate them.
- **One weather station**, at the airport rather than downtown, and trips are counted for the
  whole system, not per station or neighbourhood.
- **Feature importance is correlated.** In chart 6 humidity ranks above temperature, which
  probably reflects that it tracks rain and time of day. The ablation study is the cleaner
  test of what each variable adds.
- **Rain is rare** (8% of hours), so rain estimates are noisier than temperature estimates,
  and a tree model cannot extrapolate outside the conditions it has seen.
- **Intervals are too optimistic in the rain** (83% coverage instead of 90%), and they are
  checked on a single month (August). The simulator's own intervals, stretched to contain the
  final model's forecast, were not measured again.
- **Short validation history.** Only three successive test months exist (June to August),
  and June is scored with just two months of training data. More years of data would allow
  a longer rolling validation.
- **Tests cover the data and the schema, not the database load.** The automatic tests do not
  start PostgreSQL, so `08_load_postgres.py` is checked by its own load-time quality checks.
- **No hyperparameter tuning.** The typical error is about 358
  trips per hour, which is large for quiet hours and small for peak hours.

## Data sources

- Trips: BIXI Montréal open data, https://bixi.com/fr/donnees-ouvertes/ (not redistributed here; download the 2026 file from this page and put it in `data/raw/`).
- Weather: Environment and Climate Change Canada, GeoMet-OGC API, `climate-hourly`
  collection, station Montréal/Pierre Elliott Trudeau INTL.

Please check each provider's terms of use before reusing the data.