"""
08_load_postgres.py
Load the project's CSV outputs into PostgreSQL, then run data-quality checks.

Steps
    1. Apply sql/schema.sql (drops and recreates the `bixi` schema)
    2. Bulk-load each CSV with COPY (fast, and the whole load is one transaction)
    3. Check row counts, keys, date ranges and that the views return data

Connection: set DATABASE_URL, or use the default that matches docker-compose.yml
    postgresql://bixi:bixi@localhost:5432/bixi

Usage:
    py src/08_load_postgres.py
    py src/08_load_postgres.py --dsn postgresql://user:pwd@host:5432/dbname
"""
import argparse
import io
import os
from pathlib import Path

import pandas as pd
import psycopg2

from config import (ABLATION_CSV, BIXI_CSV, DEFAULT_DSN, FEATURES_CSV, MODEL_RESULTS_CSV,
                    SCENARIO_INTERVALS_CSV, SCHEMA_SQL, WEATHER_CSV)


def load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["hour_local", "ts_utc"])
    df = df.drop_duplicates("hour_local")
    return df.rename(columns={"ID": "source_id", "WEATHER_ENG_DESC": "weather_desc"}).rename(
        columns=str.lower)[
        ["hour_local", "ts_utc", "source_id", "temp", "dew_point_temp", "humidex",
         "windchill", "precip_amount", "relative_humidity", "station_pressure",
         "visibility", "wind_speed", "wind_direction", "weather_desc"]]


def load_plain(path: Path, rename: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=str.lower)
    return df.rename(columns=rename or {})


# table name -> DataFrame, columns in the order they appear in the table
def build_jobs() -> dict[str, pd.DataFrame]:
    return {
        "weather_hourly": load_weather(WEATHER_CSV),
        "bixi_hourly": load_plain(BIXI_CSV),
        "features_hourly": load_plain(FEATURES_CSV),
        "model_results": load_plain(
            MODEL_RESULTS_CSV, {"mae_%_of_mean": "mae_pct_of_mean"}),
        "ablation_results": load_plain(
            ABLATION_CSV,
            {"error_reduction_vs_calendar_%": "error_reduction_vs_calendar_pct",
             "mae_july": "mae_july", "mae_august": "mae_august"}),
        # scenarios + 90% prediction interval (11_scenario_intervals.py)
        "scenario_predictions": load_plain(SCENARIO_INTERVALS_CSV),
    }


def copy_dataframe(cur, table: str, df: pd.DataFrame) -> None:
    """Bulk insert with COPY. Empty cells become NULL."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, na_rep="")
    buffer.seek(0)
    cols = ", ".join(df.columns)
    cur.copy_expert(f"COPY bixi.{table} ({cols}) FROM STDIN WITH (FORMAT csv)", buffer)


def scalar(cur, sql: str):
    cur.execute(sql)
    return cur.fetchone()[0]


def quality_checks(cur, jobs: dict[str, pd.DataFrame]) -> list[str]:
    problems = []

    # 1. Every CSV row arrived
    for table, df in jobs.items():
        n = scalar(cur, f"SELECT COUNT(*) FROM bixi.{table}")
        status = "ok" if n == len(df) else "MISMATCH"
        print(f"  {table:<22} {n:>7,} rows  (csv: {len(df):,})  {status}")
        if n != len(df):
            problems.append(f"{table}: {n} rows in DB vs {len(df)} in CSV")

    # 2. Foreign-key style checks: every modelling hour must exist in the weather table
    orphans = scalar(cur, """SELECT COUNT(*) FROM bixi.features_hourly f
                             LEFT JOIN bixi.weather_hourly w USING (hour_local)
                             WHERE w.hour_local IS NULL""")
    print(f"  features without matching weather hour: {orphans}")
    if orphans:
        problems.append(f"{orphans} feature rows have no weather observation")

    # 3. Trips must reconcile between staging and modelling tables
    #    (features keeps only hours that have complete weather data, so <= staging)
    staged = scalar(cur, "SELECT SUM(trips) FROM bixi.bixi_hourly")
    modelled = scalar(cur, "SELECT SUM(trips) FROM bixi.features_hourly")
    before_weather = scalar(cur, """SELECT COALESCE(SUM(trips), 0) FROM bixi.bixi_hourly
                                    WHERE hour_local < (SELECT MIN(hour_local)
                                                        FROM bixi.weather_hourly)""")
    missing_weather = staged - modelled - before_weather
    print(f"  trips in staging {staged:,} | in features {modelled:,}")
    print(f"    before the weather period starts : {before_weather:,}")
    print(f"    hours inside it with no weather  : {missing_weather:,}")
    if missing_weather < 0:
        problems.append("features has more trips than the BIXI staging table")

    # 4. Time range
    lo, hi = scalar(cur, "SELECT MIN(hour_local) FROM bixi.features_hourly"), \
        scalar(cur, "SELECT MAX(hour_local) FROM bixi.features_hourly")
    print(f"  modelling period: {lo} -> {hi}")

    # 5. Views work
    for view in ["vw_hourly", "vw_daily", "vw_trips_by_hour",
                 "vw_daytime_by_temp_rain", "vw_scenarios"]:
        n = scalar(cur, f"SELECT COUNT(*) FROM bixi.{view}")
        print(f"  view {view:<26} {n:>7,} rows")
        if n == 0:
            problems.append(f"view {view} is empty")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL", DEFAULT_DSN))
    args = parser.parse_args()

    jobs = build_jobs()
    conn = psycopg2.connect(args.dsn)
    try:
        with conn, conn.cursor() as cur:           # one transaction: all or nothing
            print("Creating schema...")
            cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
            print("Loading tables...")
            for table, df in jobs.items():
                copy_dataframe(cur, table, df)
            print("Quality checks:")
            problems = quality_checks(cur, jobs)
            if problems:
                raise SystemExit("Load rolled back:\n- " + "\n- ".join(problems))
        print("\nDone. Schema `bixi` is ready (connect Power BI to the vw_* views).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
