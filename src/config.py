"""
config.py
Single source of truth for the whole pipeline: paths, dataset constants, model
settings and the what-if grid. Every script in src/ imports from here, so a
setting is changed in one place only (tests/test_config.py enforces it).

Scripts are run as `py src/05_train_model.py`, so Python puts src/ first on the
import path and `from config import ...` works without any packaging.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
SCHEMA_SQL = ROOT / "sql" / "schema.sql"

WEATHER_CSV = PROCESSED / "weather_hourly.csv"
BIXI_CSV = PROCESSED / "bixi_hourly.csv"
FEATURES_CSV = PROCESSED / "features_hourly.csv"
MODEL_RESULTS_CSV = PROCESSED / "model_results.csv"
ABLATION_CSV = PROCESSED / "ablation_results.csv"
SCENARIOS_CSV = PROCESSED / "scenario_predictions.csv"
SCENARIO_INTERVALS_CSV = PROCESSED / "scenario_intervals.csv"
VALIDATION_CSV = PROCESSED / "validation_results.csv"
INTERVAL_RESULTS_CSV = PROCESSED / "interval_results.csv"

# ---------------------------------------------------------------------------
# Weather source (ECCC GeoMet-OGC API)
# ---------------------------------------------------------------------------
API_URL = "https://api.weather.gc.ca/collections/climate-hourly/items"
STATION_ID = "702S006"   # Montreal/Pierre Elliott Trudeau INTL (automatic station)
API_PAGE_SIZE = 1000
DEFAULT_YEARS = [2026]
DEFAULT_MONTHS = [4, 5, 6, 7, 8]
WEATHER_API_COLUMNS = [
    "ID", "UTC_DATE", "TEMP", "DEW_POINT_TEMP", "HUMIDEX", "WINDCHILL",
    "PRECIP_AMOUNT", "RELATIVE_HUMIDITY", "STATION_PRESSURE", "VISIBILITY",
    "WIND_SPEED", "WIND_DIRECTION", "WEATHER_ENG_DESC",
]

# ---------------------------------------------------------------------------
# BIXI source
# ---------------------------------------------------------------------------
BIXI_COLUMNS = ["STARTTIMEMS", "ENDTIMEMS"]
CHUNK_SIZE = 1_000_000
MIN_TRIP_SECONDS = 60            # trips under 1 minute: false starts / errors
MAX_TRIP_SECONDS = 24 * 3600     # trips over 24 hours: bikes not returned / errors

# Quebec public holidays during the period covered (2026)
HOLIDAYS_2026 = [
    "2026-04-03",  # Good Friday
    "2026-05-18",  # National Patriots' Day
    "2026-06-24",  # Fete nationale du Quebec
    "2026-07-01",  # Canada Day
]

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TARGET = "trips"
SEED = 42
CALENDAR = ["hour", "day_of_week", "is_weekend", "is_holiday"]
WEATHER = ["TEMP", "DEW_POINT_TEMP", "PRECIP_AMOUNT", "RELATIVE_HUMIDITY",
           "STATION_PRESSURE", "WIND_SPEED", "WIND_DIRECTION"]
# What a person can set in the what-if simulator: calendar + temperature + rain
SIMULATOR_FEATURES = CALENDAR + ["TEMP", "PRECIP_AMOUNT"]

# Gradient Boosting settings shared by every script
GB_PARAMS = dict(max_iter=300, learning_rate=0.05, random_state=SEED)

# Time-based protocol (months of 2026; no shuffling anywhere)
FINAL_TEST_MONTH = 8             # August: the final test month
CALIBRATION_MONTH = 7            # July: sizes the prediction intervals
INTERVAL_TRAIN_LAST_MONTH = 6    # intervals are trained on April to June
ROLLING_FOLDS = [(6, "June"), (7, "July"), (8, "August")]   # (test month, name)

ALPHA = 0.10                     # 90% prediction intervals

# Hours of the day used in several analyses
DAYTIME_FIRST_HOUR, DAYTIME_LAST_HOUR = 7, 21     # 7 am to 9 pm
RUSH_HOURS = [7, 8, 9, 16, 17, 18]
NIGHT_HOURS = [0, 1, 2, 3, 4, 5]

# ---------------------------------------------------------------------------
# What-if grid (07_build_scenarios.py). Temperatures stay inside the range
# actually observed (about -5 to 32 C): tree models cannot extrapolate beyond it.
# The rain labels are also used by sql/schema.sql (vw_scenarios.rain_order).
# ---------------------------------------------------------------------------
TEMPS = range(0, 33)
RAIN_LEVELS = {                  # hourly precipitation, in mm
    "No rain": 0.0,
    "Light rain (1 mm/h)": 1.0,
    "Heavy rain (3 mm/h)": 3.0,
}
MIN_SIMILAR_HOURS = 30   # fewer real look-alike hours than this = flagged unreliable
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DEFAULT_DSN = "postgresql://bixi:bixi@localhost:5432/bixi"
