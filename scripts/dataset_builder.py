"""
Builds the labeled manifest of (ticker, T-1 date, T0 date) pairs for both
the treatment set (real earnings events) and the control pool (every other
non-buffer trading day), per the windowing/buffer rules worked out earlier:

- AMC earnings: T-1 = announcement day's close, T0 = next trading day's open
- BMO earnings: T-1 = previous trading day's close, T0 = announcement day's open
- Buffer exclusion (per ticker, per earnings event): the announcement day,
  whichever adjacent day got consumed as that event's other half (previous
  day for BMO, next day for AMC), plus 3 trading days strictly beyond that
  on each side - exclusive on both ends.
- Control days (anything surviving the buffer): plain close(day N) ->
  open(day N+1), no BMO/AMC involved since there's no event.

This module does NOT pull any snapshot data itself - it only decides WHICH
(ticker, T-1, T0) pairs exist and how each is labeled. Actually fetching
each pair's resolved data is process_date()'s job (snapshot_filter.py),
called once this manifest says which dates and tickers matter.

Scaffolding only - see TODOs. Nothing here has been implemented or tested
yet.
"""

from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta

import pandas as pd
import pandas_market_calendars as mcal

from config import EARNINGS_CSV_PATH, TICKERS, EXCLUSION_BUFFER, get_early_close_days, get_required_trading_days

NYSE_CAL = mcal.get_calendar("NYSE")  # built once, reused everywhere below


@dataclass(frozen=True)
class TreatmentPair:
    ticker: str
    t_minus_1: date
    t0: date
    fiscal_period: str
    earnings_date: date
    time_of_day: str  # "BMO" or "AMC"


def previous_trading_day(d: date) -> date:
    """
    Return the NYSE trading day strictly before `d`.
    """
    this_day = pd.Timestamp(d, tz="UTC")

    # is d a valid trading day? 
    valid_days = NYSE_CAL.valid_days(start_date=this_day, end_date=this_day)
    if this_day not in valid_days: 
        raise ValueError("param d is not a valid trading day")

    valid_days = NYSE_CAL.valid_days(start_date=this_day - timedelta(days=20), 
                                     end_date=this_day - timedelta(days=1))

    if valid_days.size == 0:
        raise ValueError("param d is not a valid trading day")

    return valid_days[-1].date()


def next_trading_day(d: date) -> date:
    """
    Return the NYSE trading day strictly after `d`.
    """
    this_day = pd.Timestamp(d, tz="UTC")

    # is d a valid trading day? 
    valid_days = NYSE_CAL.valid_days(start_date=this_day, end_date=this_day)
    if this_day not in valid_days: 
        raise ValueError("param `d` is not a valid trading day")

    valid_days = NYSE_CAL.valid_days(start_date=this_day + timedelta(days=1), 
                                     end_date=this_day + timedelta(days=20))

    if valid_days.size == 0:
        raise ValueError("param `d` is not a valid trading day")

    return valid_days[0].date()
    


def resolve_treatment_window(earnings_date: date, time_of_day: str) -> tuple[date, date]:
    """
    Turn one earnings event into its (T-1, T0) date pair.

    Params:
        - earnings_date: the date of the earnings call
        - time_of_day:      - "bmo": before market open
                            - "amc": after market close
    """
    match time_of_day.lower():
        case "amc":
            return(earnings_date, next_trading_day(earnings_date))
        case "bmo":
            return(previous_trading_day(earnings_date), earnings_date)
        case _:
            raise ValueError("invalid `str` param. Must be either 'amc' or 'bmo'")


def build_treatment_pairs(earnings_df: pd.DataFrame, early_close_days: set[date]) -> list[TreatmentPair]:
    """
    Apply resolve_treatment_window() to every row of the earnings df,
    producing one TreatmentPair per row.

    A row is skipped (logged, not crashed) if its resolved T-1 lands on a
    known NYSE early-close day - see get_early_close_days() in config.py for
    why that close can't be trusted. This is an expected, recurring calendar
    event (5 known dates across our whole range), not a data-quality bug,
    so it gets the skip-and-log treatment rather than the loud-crash policy
    used for genuinely invalid rows (bad time_of_day, unresolvable dates).
    """

    treatment_bucket = []
    for _, row in earnings_df.iterrows():
        earnings_date = row["earnings_date"].date()
        time_of_day = row["time_of_day"]

        t1, t0 = resolve_treatment_window(earnings_date, time_of_day)

        if t1 in early_close_days:
            print(
                f"SKIPPING treatment row: {row['ticker']} {row['fiscal_period']} "
                f"(earnings_date={earnings_date}) - resolved T-1 {t1} is a known "
                f"NYSE early-close day, close snapshot unreliable."
            )
            continue

        treatment_pair = TreatmentPair(ticker=row["ticker"],
                                       t_minus_1=t1,
                                       t0=t0,
                                       fiscal_period=row["fiscal_period"],
                                       earnings_date=earnings_date,
                                       time_of_day=time_of_day)

        treatment_bucket.append(treatment_pair)

    return treatment_bucket

def compute_ticker_exclusions(ticker_rows: pd.DataFrame) -> set[date]:
    """
    Given one ticker's earnings rows (a slice of the earnings CSV, all rows
    where ticker == this ticker), compute the full set of trading days that
    should NOT be used as control days for this ticker.

    For each earnings row:
    - include the earnings announcement date itself
    - include whichever adjacent day got consumed as that event's other
      T-1/T0 half (previous_trading_day for BMO, next_trading_day for AMC)
    - include EXCLUSION_BUFFER trading days strictly beyond the outer edge
      of that pair, on each side
    """
    days_to_exclude = set()

    for _, row in ticker_rows.iterrows():
        earnings_date = row["earnings_date"].date()
        days_to_exclude.add(earnings_date)

        time_of_day = row["time_of_day"].lower()
        if time_of_day == "amc":
            consumed_day = next_trading_day(earnings_date)
            lower_bound, upper_bound = earnings_date, consumed_day
        elif time_of_day == "bmo":
            consumed_day = previous_trading_day(earnings_date)
            lower_bound, upper_bound = consumed_day, earnings_date
        else:
            raise ValueError("invalid `row['time_of_day']` value. Must be either 'amc' or 'bmo'")
        days_to_exclude.add(consumed_day)

        next_trading_days = NYSE_CAL.valid_days(start_date=upper_bound + timedelta(days=1),
                                                  end_date=upper_bound + timedelta(days=20))
        prev_trading_days = NYSE_CAL.valid_days(start_date=lower_bound - timedelta(days=20),
                                                  end_date=lower_bound - timedelta(days=1))

        days_to_exclude.update(d.date() for d in next_trading_days[:EXCLUSION_BUFFER])
        days_to_exclude.update(d.date() for d in prev_trading_days[-EXCLUSION_BUFFER:])

    return days_to_exclude
    


def build_control_pairs(ticker: str, all_trading_days: list[date], exclusions: set[date], early_close_days: set[date]) -> list[tuple[date, date]]:
    """
    For one ticker: every trading day in `all_trading_days` NOT in
    `exclusions` becomes a control day, paired with its own close (T-1) and
    the next trading day's open (T0).

    A day is skipped if its own close (T-1) or its next trading day's open
    (T0) falls inside the exclusion buffer, if its next trading day falls
    outside the fetched date range entirely, or if its own close (T-1)
    lands on a known NYSE early-close day (see get_early_close_days() in
    config.py). early_close_days is checked ONLY against t1 (the close
    leg) - a day whose T0 (open) happens to land on an early-close day is
    still a perfectly valid control pair, since early closes never affect
    the open.
    """
    fetched_days = set(all_trading_days)
    control_pairs = []
    for t1 in all_trading_days:
        if t1 in exclusions:
            continue
        if t1 in early_close_days:
            continue
        t0 = next_trading_day(t1)
        if t0 not in fetched_days:
            continue
        if t0 in exclusions:
            continue
        control_pairs.append((t1, t0))

    return control_pairs

def build_dataset_manifest() -> pd.DataFrame:
    """
    Produces the full labeled manifest for all tickers-
    every treatment pair (from the earnings CSV) and every control pair
    (from the buffer-filtered trading day range), combined into one big
    table.
    """
    earnings_df = pd.read_csv(EARNINGS_CSV_PATH, parse_dates=["earnings_date"])

    early_close_days = get_early_close_days()

    treatment_bucket = build_treatment_pairs(earnings_df, early_close_days)

    all_trading_days = get_required_trading_days()

    control_bucket = []
    for ticker in TICKERS:
        sliced_df = earnings_df.loc[earnings_df["ticker"] == ticker]
        excluded_dates = compute_ticker_exclusions(sliced_df)
        control_bucket.extend((ticker,t1,t0) for t1, t0 in build_control_pairs(ticker = ticker, 
                                                                                   all_trading_days=all_trading_days,
                                                                                   exclusions=excluded_dates,
                                                                                   early_close_days=early_close_days))

    cols = [asdict(treatment_pair) for treatment_pair in treatment_bucket]
    treatment_df = pd.DataFrame(cols)
    treatment_df["label"] = "treatment"

    control_df = pd.DataFrame(control_bucket, columns=["ticker", "t_minus_1", "t0"])
    control_df["label"] = "control"

    manifest = pd.concat([treatment_df, control_df], ignore_index=True)

    return manifest
