"""
Config constants for the snapshot extraction/filtering layer.

Nothing in this file should ever be hardcoded into snapshot_filter.py or any
other module - if a value here needs tuning (e.g. how much of the open we
skip past, how big the fallback lookback window is), it should only ever
need to change here.
"""

from datetime import date, time
from pathlib import Path
import pandas_market_calendars as mcal

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EARNINGS_CSV_PATH = REPO_ROOT / "demo_earnings_dates_clean.csv"


def get_tickers_from_input(path: Path) -> list[str]:
    """
    Gets the set of tickers used for this analysis from the input file
    (demo_earnings_dates_clean.csv), so this is the single source of truth
    for "which names are we using" - no separate hardcoded list that could
    drift out of sync with the actual earnings-date data.
    """
    df = pd.read_csv(path)
    # groupby's index is sorted by default (sort=True), so this comes back
    # deterministic - same order every run, not dependent on row order in
    # the CSV.
    return df.groupby("ticker").size().index.tolist()


TICKERS = get_tickers_from_input(EARNINGS_CSV_PATH)


# Massive's Options/Stocks Starter plan only serves flat files going back
# this many years from today - confirmed empirically (a 403 Forbidden when
# requesting a date older than this, same "authenticated but not entitled"
# pattern as the earlier equity-dataset access issue). Not upgrading past
# Starter, so this is a hard ceiling on how far back we can ever fetch,
# not just a config nicety.
MAX_HISTORY_YEARS = 2


def _get_clipped_date_range(buffer_months: int = 1) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Shared range-plus-clipping logic used by both get_required_trading_days()
    and get_early_close_days(), so the two guards (2-year plan history limit,
    future-date cutoff) only ever need to be written once.

    Range: `buffer_months` before the earliest earnings_date in the CSV to
    `buffer_months` after the latest.

    Clipped at MAX_HISTORY_YEARS back from today's actual date (computed at
    call time, not hardcoded to any specific date) - the earnings CSV's
    earliest dates can fall outside what the current plan can ever serve,
    and requesting those anyway just gets a 403 partway through a long
    download run instead of a clear, upfront explanation. Symmetric guard on
    the other end clips a future-reaching range end to today, since there's
    no market data for a day that hasn't occurred yet.
    """
    df = pd.read_csv(EARNINGS_CSV_PATH, parse_dates=["earnings_date"])
    range_start = df["earnings_date"].min() - pd.DateOffset(months=buffer_months)
    range_end = df["earnings_date"].max() + pd.DateOffset(months=buffer_months)

    today = pd.Timestamp.now().normalize()

    history_cutoff = today - pd.DateOffset(years=MAX_HISTORY_YEARS)
    if range_start < history_cutoff:
        print(
            f"NOTE: requested range start {range_start.date()} is before the "
            f"~{MAX_HISTORY_YEARS}-year plan history limit (~{history_cutoff.date()} "
            f"as of today). Clipping to {history_cutoff.date()} - some of the "
            f"earliest treatment/control dates will be unavailable under the "
            f"current plan."
        )
        range_start = history_cutoff

    if range_end > today:
        print(
            f"NOTE: requested range end {range_end.date()} is in the future "
            f"(today is {today.date()}). Clipping to today - future dates "
            f"have no market data to fetch regardless of plan tier."
        )
        range_end = today

    return range_start, range_end


def get_required_trading_days(buffer_months: int = 1) -> list:
    """
    Derives the full range of dates to fetch flat files for, from
    demo_earnings_dates_clean.csv - covers both the treatment dates
    themselves and the (much larger) control-day pool, since pooled across
    all tickers, nearly every trading day in the range ends up being some
    ticker's control day outside its own earnings buffer.

    Restricted to actual NYSE trading days - weekends and market holidays
    are excluded here rather than left for download_flat_files.py to
    discover as 404s one at a time. Early-close days (see
    get_early_close_days()) are INTENTIONALLY still included here - they are
    genuine trading days, valid for T0/open purposes and for calendar
    walking (previous_trading_day/next_trading_day). Only their T-1/close
    usage is unreliable, which is handled separately at the point a T-1 date
    gets finalized, not by removing the day from the calendar entirely.

    Unlike TICKERS, this is a function you call when you need it, not a
    module-level constant computed at import time - it pulls in a calendar
    library and computes a few hundred dates, which every `import config`
    shouldn't have to pay for just to reach TICKERS or the time constants.
    """
    range_start, range_end = _get_clipped_date_range(buffer_months)
    nyse = mcal.get_calendar("NYSE")
    trading_days = nyse.valid_days(start_date=range_start, end_date=range_end)
    return [d.date() for d in trading_days]


def get_early_close_days(buffer_months: int = 1) -> set[date]:
    """
    NYSE scheduled early-close days (e.g. day after Thanksgiving, Christmas
    Eve, day before July 4th - typically a 13:00 close instead of 16:00)
    within the same date range get_required_trading_days() covers.

    These days should NOT be used as a T-1 (close) anchor for either
    treatment or control pairs: the 15:55-16:00 target window doesn't exist
    (session already ended at 13:00), and any equity row that does appear to
    "resolve" in that window on such a day cannot be reliably distinguished
    from a stray late/off-exchange print rather than a genuine closing
    trade. They remain perfectly valid as a T0 (open) anchor, or as some
    other day's next_trading_day() target, since early closes only affect
    the close time, never the open - so exclusion needs to happen at
    whichever call site finalizes a T-1 date (build_control_pairs,
    build_treatment_pairs), not here.
    """
    range_start, range_end = _get_clipped_date_range(buffer_months)
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=range_start, end_date=range_end)
    market_close_local = schedule["market_close"].dt.tz_convert(TIMEZONE)
    is_early_close = market_close_local.dt.time != MARKET_CLOSE_TIME
    return set(schedule.index[is_early_close].date)




TIMEZONE = "America/New_York"

# Anchor points. Both flat-file datasets report window_start as Unix
# nanosecond UTC timestamps - these are wall-clock times in TIMEZONE, not UTC.
MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)

# How far past MARKET_OPEN_TIME the T0 snapshot target sits, to dodge the
# noisy first few minutes of open-auction trading. 15 -> target of 9:45.
OPEN_SNAPSHOT_OFFSET_MINUTES = 15

# Same idea for T-1/close, in case the target ever needs to move off the
# exact close. 0 -> target is exactly MARKET_CLOSE_TIME (4:00pm).
CLOSE_SNAPSHOT_OFFSET_MINUTES = 0

# Fallback window size: if a contract has no print in the exact target
# minute, look back up to this many minutes for the most recent one.
# Applies identically to both T0 and T1 targets - see the discussion on why
# a backward-looking window works for both.
SNAPSHOT_LOOKBACK_MINUTES = 5


EXCLUSION_BUFFER = 3