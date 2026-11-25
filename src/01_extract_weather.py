"""
01_extract_weather.py
Download hourly weather observations from the ECCC GeoMet-OGC API and save them
to a CSV. By default: Montreal-Trudeau airport -> data/processed/weather_hourly.csv.

Usage:
    py src/01_extract_weather.py
    py src/01_extract_weather.py --years 2025 2026 --months 4 5 6 7 8
    py src/01_extract_weather.py --station 7024745 --out data/processed/weather_mctavish.csv

Always pass --out when you use --station, otherwise weather_hourly.csv (the input
of the whole pipeline) is overwritten with another station's data.
"""
import argparse
import calendar
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (API_PAGE_SIZE, API_URL, DEFAULT_MONTHS, DEFAULT_YEARS, STATION_ID,
                    WEATHER_API_COLUMNS, WEATHER_CSV)


def make_session() -> requests.Session:
    """HTTP session that automatically retries on network errors, 429 and 5xx."""
    retry = Retry(total=5, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_month(session: requests.Session, year: int, month: int,
                station: str = STATION_ID) -> list[dict]:
    """Fetch every hourly observation of one month, following pagination."""
    last_day = calendar.monthrange(year, month)[1]
    params = {
        "CLIMATE_IDENTIFIER": station,
        "datetime": f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}",
        "limit": API_PAGE_SIZE,
        "offset": 0,
        "f": "json",
    }
    rows, total = [], None
    while total is None or params["offset"] < total:
        resp = session.get(API_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        total = payload["numberMatched"]
        if not payload["features"]:
            break
        rows.extend(f["properties"] for f in payload["features"])
        params["offset"] += API_PAGE_SIZE
        time.sleep(0.3)  # be polite with the API
    return rows


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, sort, and build a correct Montreal local-time column.

    The API's LOCAL_DATE is in fixed standard time (EST) all year, so it is
    off by one hour during daylight saving time. We start from UTC instead.
    """
    df = df[[c for c in WEATHER_API_COLUMNS if c in df.columns]].drop_duplicates("ID")
    ts_utc = pd.to_datetime(df["UTC_DATE"], utc=True)
    hour_local = (ts_utc.dt.tz_convert("America/Montreal")
                        .dt.tz_localize(None))       # wall-clock time, no timezone
    df = df.assign(ts_utc=ts_utc, hour_local=hour_local).drop(columns=["UTC_DATE"])
    return df.sort_values("ts_utc").reset_index(drop=True)


def report_gaps(df: pd.DataFrame) -> None:
    full = pd.date_range(df["ts_utc"].min(), df["ts_utc"].max(), freq="h")
    missing = len(full) - df["ts_utc"].nunique()
    print(f"{len(df)} observations, {missing} missing hours in the covered range.")
    print("Missing values (%):")
    print((df.isna().mean() * 100).round(1).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    parser.add_argument("--months", type=int, nargs="+", default=DEFAULT_MONTHS)
    parser.add_argument("--station", default=STATION_ID,
                        help="ECCC climate identifier (default: Trudeau airport)")
    parser.add_argument("--out", type=Path, default=WEATHER_CSV)
    args = parser.parse_args()

    session = make_session()
    all_rows = []
    for year in args.years:
        for month in args.months:
            rows = fetch_month(session, year, month, args.station)
            print(f"{year}-{month:02d}: {len(rows)} observations")
            all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No data downloaded.")

    df = clean(pd.DataFrame(all_rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"File written: {args.out}")
    report_gaps(df)


if __name__ == "__main__":
    main()