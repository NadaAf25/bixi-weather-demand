"""
02_aggregate_bixi.py
Read the BIXI open-data file (ZIP or CSV, ~2 GB) in chunks and count trips per
hour (Montreal local time). Output: data/processed/bixi_hourly.csv

Usage:
    py src/02_aggregate_bixi.py                      # looks inside data/raw/
    py src/02_aggregate_bixi.py "C:\\path\\to\\file.zip"
"""
import argparse
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "processed" / "bixi_hourly.csv"

CHUNK_SIZE = 1_000_000
MIN_DURATION_S = 60          # trips under 1 minute: false starts / errors
MAX_DURATION_S = 24 * 3600   # trips over 24 hours: bikes not returned / errors
COLUMNS = ["STARTTIMEMS", "ENDTIMEMS"]


def find_source() -> Path:
    candidates = sorted(RAW_DIR.glob("*.zip")) + sorted(RAW_DIR.glob("*.csv"))
    if not candidates:
        raise SystemExit(f"No BIXI file found in {RAW_DIR}. "
                         "Put the ZIP there or pass its path as an argument.")
    return candidates[0]


def open_source(path: Path):
    """Open the CSV directly inside the ZIP, without extracting 2 GB to disk."""
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return zf.open(name)
    return open(path, "rb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, nargs="?", default=None)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    source = args.source or find_source()
    print(f"Reading {source} (this can take a few minutes)...")

    hourly = pd.Series(dtype="int64")
    stats = {"read": 0, "missing_time": 0, "invalid_duration": 0, "kept": 0}

    with open_source(source) as f:
        for chunk in pd.read_csv(f, usecols=COLUMNS, chunksize=CHUNK_SIZE):
            stats["read"] += len(chunk)

            missing = chunk.isna().any(axis=1)
            stats["missing_time"] += int(missing.sum())
            chunk = chunk[~missing]

            duration = (chunk["ENDTIMEMS"] - chunk["STARTTIMEMS"]) / 1000
            valid = duration.between(MIN_DURATION_S, MAX_DURATION_S)
            stats["invalid_duration"] += int((~valid).sum())
            chunk = chunk[valid]
            stats["kept"] += len(chunk)

            # Epoch in milliseconds (UTC) -> Montreal wall-clock hour, no timezone
            start = (pd.to_datetime(chunk["STARTTIMEMS"], unit="ms", utc=True)
                       .dt.tz_convert("America/Montreal")
                       .dt.tz_localize(None)
                       .dt.floor("h"))
            hourly = hourly.add(start.value_counts(), fill_value=0)

    out = (hourly.astype("int64").sort_index()
                 .rename_axis("hour_local").rename("trips").reset_index())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"Trips read              : {stats['read']:,}")
    print(f"Missing time (excluded) : {stats['missing_time']:,}")
    print(f"Invalid duration (excl.): {stats['invalid_duration']:,}")
    print(f"Trips kept              : {stats['kept']:,}")
    print(f"Hours with trips        : {len(out):,}  ->  {args.out}")


if __name__ == "__main__":
    main()