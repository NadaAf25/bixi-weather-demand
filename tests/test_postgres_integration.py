"""
test_postgres_integration.py
Runs the real loader (src/08_load_postgres.py) against a real PostgreSQL, then checks
what ended up in the database.

Skipped unless TEST_DATABASE_URL is set, because the loader DROPS and recreates the
`bixi` schema: never point it at a database you care about. In CI the variable points
at a throwaway postgres:16 service (.github/workflows/tests.yml).

Locally (PowerShell):
    docker compose up -d
    $env:TEST_DATABASE_URL = "postgresql://bixi:bixi@localhost:5432/bixi"
    py -m pytest tests/test_postgres_integration.py
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

psycopg2 = pytest.importorskip("psycopg2")
from psycopg2 import errors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
import config  # noqa: E402

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DSN, reason="set TEST_DATABASE_URL to run (the loader drops the schema `bixi`)")

VIEWS = ["vw_hourly", "vw_daily", "vw_trips_by_hour", "vw_daytime_by_temp_rain",
         "vw_scenarios"]


def run_loader() -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SRC / "08_load_postgres.py"), "--dsn", DSN],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


def count(cur, relation: str) -> int:
    cur.execute(f"SELECT COUNT(*) FROM bixi.{relation}")
    return cur.fetchone()[0]


def csv_rows(path: Path, unique_on: str | None = None) -> int:
    df = pd.read_csv(path)
    return len(df.drop_duplicates(unique_on)) if unique_on else len(df)


@pytest.fixture(scope="module")
def db():
    """Load twice: the loader must be idempotent (it recreates the schema each time)."""
    for attempt in (1, 2):
        result = run_loader()
        assert result.returncode == 0, (
            f"load {attempt} failed:\n{result.stdout}\n{result.stderr}")
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    yield conn
    conn.close()


def test_every_table_has_the_rows_of_its_csv(db):
    expected = {
        "weather_hourly": csv_rows(config.WEATHER_CSV, "hour_local"),
        "bixi_hourly": csv_rows(config.BIXI_CSV),
        "features_hourly": csv_rows(config.FEATURES_CSV),
        "model_results": csv_rows(config.MODEL_RESULTS_CSV),
        "ablation_results": csv_rows(config.ABLATION_CSV),
        "scenario_predictions": csv_rows(config.SCENARIO_INTERVALS_CSV),
    }
    with db.cursor() as cur:
        for table, n_rows in expected.items():
            assert count(cur, table) == n_rows, table
        assert count(cur, "scenario_predictions") == 16_632


def test_views_return_the_expected_rows(db):
    n_days = pd.to_datetime(pd.read_csv(config.FEATURES_CSV)["hour_local"]).dt.date.nunique()
    with db.cursor() as cur:
        for view in VIEWS:
            assert count(cur, view) > 0, view
        assert count(cur, "vw_hourly") == count(cur, "features_hourly")
        assert count(cur, "vw_daily") == n_days
        assert count(cur, "vw_scenarios") == 16_632
        cur.execute("SELECT MIN(rain_order), MAX(rain_order) FROM bixi.vw_scenarios")
        assert cur.fetchone() == (1, 3)


def test_the_database_rejects_bad_data(db):
    with db.cursor() as cur:
        with pytest.raises(errors.CheckViolation):      # negative trip count
            cur.execute("INSERT INTO bixi.bixi_hourly (hour_local, trips) "
                        "VALUES ('2030-01-01 00:00', -1)")
        with pytest.raises(errors.UniqueViolation):     # duplicate primary key
            cur.execute("INSERT INTO bixi.bixi_hourly "
                        "SELECT hour_local, trips FROM bixi.bixi_hourly LIMIT 1")