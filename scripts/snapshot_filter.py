"""
Turns one day's raw flat files (whole-market, both datasets) into filtered,
snapshot-resolved tables for just our 16 tickers, at the T0 and T1 time
windows defined in config.py.

This module does NOT know which snapshot type (T0 or T1) any particular
date is actually needed for - that's a decision for whoever calls
process_date(). It always computes both slices for both datasets, and lets
the caller keep or discard what it doesn't need.

Scaffolding only - see TODOs. Nothing here has been implemented or tested
yet.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd

from config import (
    CLOSE_SNAPSHOT_OFFSET_MINUTES,
    MARKET_CLOSE_TIME,
    MARKET_OPEN_TIME,
    OPEN_SNAPSHOT_OFFSET_MINUTES,
    SNAPSHOT_LOOKBACK_MINUTES,
    TICKERS,
    TIMEZONE,
)

from occ_parser import parse_occ_ticker


@dataclass
class SnapshotWindow:
    start: time
    end: time


def compute_snapshot_windows() -> dict[str, SnapshotWindow]:
    """
    Create the SnapshotWindows for T-1_target and T0_target
    """

    anchor = datetime.combine(datetime.min, MARKET_OPEN_TIME)
    T0_target_start = (anchor + timedelta(minutes=OPEN_SNAPSHOT_OFFSET_MINUTES) - timedelta(minutes=SNAPSHOT_LOOKBACK_MINUTES)).time()
    T0_target_end = (anchor + timedelta(minutes=OPEN_SNAPSHOT_OFFSET_MINUTES)).time()

    anchor = datetime.combine(datetime.min, MARKET_CLOSE_TIME)
    T1_target_start = (anchor - timedelta(minutes=SNAPSHOT_LOOKBACK_MINUTES)).time()
    T1_target_end = (anchor).time()

    T0_target = SnapshotWindow(start=T0_target_start, end=T0_target_end)
    T1_target = SnapshotWindow(start=T1_target_start, end=T1_target_end)

    return {"T0_target":T0_target, "T-1_target":T1_target}



REPO_ROOT = Path(__file__).resolve().parent.parent


def local_path_for(dataset: str, date: datetime) -> Path:
    """
    Where a given (dataset, date)'s raw flat file lives on disk - matches
    download_flat_files.py's local_path_for exactly, since that's what
    actually wrote these files. Duplicated here rather than imported from
    download_flat_files.py to avoid pulling in that module's CLI/dotenv/
    boto3 setup just to compute a path.
    """
    return REPO_ROOT / "data" / "raw" / dataset / f"{date:%Y-%m-%d}.csv.gz"


def cache_path_for(dataset: str, date: datetime) -> Path:
    """Where process_date's filtered/resolved output for (dataset, date) is cached."""
    return REPO_ROOT / "data" / "processed" / dataset / f"{date:%Y-%m-%d}.parquet"


# Explicit dtypes rather than letting pandas guess. window_start is the one
# that actually matters: it's a Unix NANOSECOND timestamp (19 digits), and
# if pandas infers it as float64 instead of int64, values above ~9e15 lose
# precision (float64 only has ~15-17 significant decimal digits) - silently
# corrupting the very timestamps this whole layer depends on.
_RAW_DTYPES = {
    "ticker": "string",
    "volume": "float64",
    "open": "float64",
    "close": "float64",
    "high": "float64",
    "low": "float64",
    "window_start": "int64",
    "transactions": "int64",
}


def read_raw_file(path: Path) -> pd.DataFrame:
    """
    Decompress and load one day's raw .csv.gz (options or equity) into a
    DataFrame. No filtering yet - this is the whole-market file as-is.

    Plain pd.read_csv, no chunking needed yet until dataset becomes huge
    """
    return pd.read_csv(path, dtype=_RAW_DTYPES)


def filter_options_to_tickers(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Given the raw options DataFrame (ticker column = OCC-encoded strings),
    parse each ticker to find its true underlying and keep only rows whose
    underlying is in `tickers`.

    Parses only the *unique* ticker strings once (typically ~10x fewer than
    the row count, since each contract prints many per-minute rows across
    the day) rather than re-parsing the same ticker string on every row.
    """
    ticker_set = set(tickers)

    underlying_by_raw_ticker = {}
    for raw_ticker in df["ticker"].unique():
        try:
            underlying_by_raw_ticker[raw_ticker] = parse_occ_ticker(raw_ticker).underlying
        except ValueError:
            # Not a well-formed OCC ticker - leave it out of the map, so
            # rows with this ticker get NaN below and are dropped rather
            # than crashing the whole batch over one bad row.
            continue

    df = df.copy()
    df["underlying"] = df["ticker"].map(underlying_by_raw_ticker)
    return df[df["underlying"].isin(ticker_set)].reset_index(drop=True)


def filter_equity_to_tickers(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Same idea as filter_options_to_tickers, but the equity file's ticker
    column is already a plain symbol (no OCC encoding).
    """
    ticker_set = set(tickers)
    return df[df["ticker"].isin(ticker_set)].reset_index(drop=True)

def filter_to_window(df: pd.DataFrame, window: SnapshotWindow) -> pd.DataFrame:
    """
    Convert window_start (nanoseconds) to a wall-clock time in TIMEZONE, 
    and keep only rows whose converted time falls within `window`
    (inclusive on both ends).
    """
    ts_utc = pd.to_datetime(df["window_start"], unit="ns", utc=True)
    ts_local = ts_utc.dt.tz_convert(TIMEZONE)

    local_time = ts_local.dt.time

    mask = (local_time >= window.start) & (local_time <= window.end)
    return df[mask].reset_index(drop=True)


def resolve_snapshot_per_key(df: pd.DataFrame, key_column: str) -> pd.DataFrame:
    """
    Within an already-window-filtered DataFrame, collapse to one row per
    `key_column` value (the OCC ticker for options, the plain ticker for
    equity) by keeping the row closest to the window's end (i.e. the latest
    window_start) where multiple candidate rows exist for the same key.
    """
    return df.loc[df.groupby(key_column)["window_start"].idxmax()].reset_index(drop=True)


def _build_snapshots(dataset: str, date: datetime) -> pd.DataFrame:
    """
    One dataset ("options" or "equity"), one date: load, filter to our
    tickers, resolve both T0 and T1 windows, stack them into one tidy
    long-format table tagged with snapshot_type and date. Cached to parquet
    so re-running against a date already processed skips the raw gz
    entirely.
    """
    cache_path = cache_path_for(dataset, date)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = read_raw_file(local_path_for(dataset, date))
    if dataset == "options":
        filtered = filter_options_to_tickers(raw, TICKERS)
        key_column = "ticker"
    else:
        filtered = filter_equity_to_tickers(raw, TICKERS)
        key_column = "ticker"

    windows = compute_snapshot_windows()
    resolved_slices = []
    for snapshot_type, window in windows.items():
        windowed = filter_to_window(filtered, window)
        resolved = resolve_snapshot_per_key(windowed, key_column=key_column)
        resolved = resolved.copy()
        resolved["snapshot_type"] = snapshot_type
        resolved["date"] = pd.Timestamp(date).date()
        resolved_slices.append(resolved)

    result = pd.concat(resolved_slices, ignore_index=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)
    return result


def process_date(date: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrator for one calendar date: loads both raw files (from
    data/raw/options/ and data/raw/equity/, per download_flat_files.py's
    layout), filters each to our 16 tickers, and produces both T0 and T1
    snapshot tables for both datasets.

    Returns (options_snapshots, equity_snapshots), each a tidy long-format
    DataFrame covering both snapshot types for this one date. Each dataset's
    result is cached to data/processed/<dataset>/<date>.parquet, so calling
    this again for the same date reads the cache instead of re-parsing the
    raw gz file.
    """
    options_snapshots = _build_snapshots("options", date)
    equity_snapshots = _build_snapshots("equity", date)
    return options_snapshots, equity_snapshots
