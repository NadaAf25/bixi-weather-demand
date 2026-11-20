# BIXI x Weather: how much does the weather drive bike-share demand in Montreal?

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

![Scenario curves](figures/8_scenario_curves.png)

## Dashboard (Power BI)

Three pages: **Overview** (daily trips, a typical day, trips by temperature and rain),
**Simulator** (pick temperature, rain and day: get a headline number, an hour-by-hour curve
and a reliability warning) and **Model** (model comparison and ablation study).
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
| 8 | `08_load_postgres.py` | Applies `sql/schema.sql`, bulk-loads every table with `COPY` in a single transaction, then runs quality checks. |

### Database

`sql/schema.sql` creates the `bixi` schema: staging tables (`weather_hourly`, `bixi_hourly`),
the modelling table (`features_hourly`), model outputs (`model_results`, `ablation_results`,
`scenario_predictions`) and five reporting views. Primary keys, `CHECK` constraints and
NOT NULL rules reject bad data at load time. The loader is idempotent (re-run it any time) and
rolls back everything if a check fails. Checks include row counts against the CSVs, orphan
hours, and a trip-count reconciliation: 236,330 trips fall before the weather period starts
(January to March) and 11,298 fall in three hours with no weather measurement.

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

docker compose up -d                                   # local PostgreSQL 16
python src/08_load_postgres.py                         # or --dsn / DATABASE_URL for another server
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
src/        pipeline scripts 01 to 08
sql/        schema.sql (tables, constraints, views)
powerbi/    build guide, DAX measures, theme, bixi_weather.pbix
data/
  raw/          BIXI file (not in git)
  processed/    all intermediate and result CSVs
figures/    charts 1 to 8
docs/       architecture diagram and Power BI screenshots
docker-compose.yml
```

## Design choices worth knowing

- **Time-based validation.** Hours are strongly autocorrelated, so a random split would leak
  the future into training. The model is always tested on a later month.
- **Why the simulator uses temperature and rain only.** In the ablation study, calendar +
  temperature + rain (MAE 401) is almost as good as all weather variables (388), and, unlike
  humidity or pressure, a person can set both in a what-if question.
- **Model scored in Python, served in SQL.** Power BI cannot run scikit-learn, so predictions
  are pre-computed and stored. Re-run step 7 and step 8 whenever the model changes.
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
- **No hyperparameter tuning and no prediction intervals.** The typical error is about 358
  trips per hour, which is large for quiet hours and small for peak hours.

## Data sources

- Trips: BIXI Montréal open data (not redistributed here; download it from BIXI's website).
- Weather: Environment and Climate Change Canada, GeoMet-OGC API, `climate-hourly`
  collection, station Montréal/Pierre Elliott Trudeau INTL.

Please check each provider's terms of use before reusing the data.
