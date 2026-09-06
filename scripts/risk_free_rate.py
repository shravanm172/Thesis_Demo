"""
Sources the risk-free rate r used by back_out_implied_vol() (iv_backout.py)
for BAW inversion.

Design decision (2026-09-06): pull the daily Treasury par yield curve from
FRED (https://fred.stlouisfed.org/) - series DGS1MO, DGS3MO, DGS6MO, DGS1,
DGS2 (short tenors only; our options are not expected to run past ~1-2
years given the manifest's earnings-driven date range) - and linearly
interpolate yield vs. days-to-expiry per contract, per date. Chosen over
log-linear-on-discount-factors or a parametric curve fit (Nelson-Siegel/
Svensson): those buy real accuracy only when tenors are sparse/far apart or
rates are volatile between them, neither of which applies here (FRED's
DGS series are already a clean fitted curve, not raw noisy ticks, and our
tenor points are dense in the 1mo-2yr range we actually need). Linear
interpolation on yield directly is the standard textbook BS/BAW treatment
and is simple to implement, verify, and justify in the writeup.

Pipeline position: this module has no dependency on iv_backout.py or
snapshot_filter.py - it only needs a set of dates and, per contract, a
days-to-expiry. back_out_implied_vol() calls into this to get its r input;
this module doesn't know anything about options, contracts, or surfaces.

Nothing in this module is implemented yet - see TODOs.
"""

import os
from datetime import date

import pandas as pd
import requests

from occ_parser import parse_occ_ticker

# FRED series IDs for the Daily Treasury Par Yield Curve Rate, one per
# tenor. Keys are the tenor in calendar days, used directly as the x-axis
# for interpolation against a contract's days-to-expiry.
FRED_TENOR_SERIES = {
    30: "DGS1MO",
    90: "DGS3MO",
    180: "DGS6MO",
    365: "DGS1",
    730: "DGS2",
}

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


def _fetch_one_series(series_id: str, start_date: date, end_date: date, api_key: str) -> pd.Series:
    """
    Call FRED's series/observations endpoint for a single tenor and return it 
    as a pd.Series indexed by date 

    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date.isoformat(),
        "observation_end": end_date.isoformat(),
    }
    response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    observations = payload["observations"]
    dates = [pd.Timestamp(obs["date"]) for obs in observations]
    values = [
        float(obs["value"]) / 100.0 if obs["value"] != "." else float("nan")
        for obs in observations
    ]
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id, dtype="float64")


def fetch_fred_yield_curve(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Pull the daily Treasury par yield curve from FRED for every tenor in
    FRED_TENOR_SERIES, over [start_date, end_date], and return a single
    DataFrame indexed by date with one column per tenor 

    Needs a FRED API key, read from .env

    Missing observations are forward-filled from the most recent prior observation 
    Leading gaps are left as NaN
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY environment variable is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and set it "
            "before calling fetch_fred_yield_curve()."
        )

    series_by_tenor = {
        tenor_days: _fetch_one_series(series_id, start_date, end_date, api_key)
        for tenor_days, series_id in FRED_TENOR_SERIES.items()
    }

    curve = pd.DataFrame(series_by_tenor)
    curve = curve.sort_index()
    curve = curve.ffill()

    return curve


def get_risk_free_rate(curve_row: pd.Series, days_to_expiry: int) -> float:
    """
    Given one row of the DataFrame returned by fetch_fred_yield_curve()
    (i.e. one date's full set of tenor yields, indexed by tenor in
    calendar days) and one contract's days_to_expiry, linearly interpolate
    between the two bracketing tenors to get the DTE-matched risk-free
    rate.

    Below the shortest tenor: linearly extrapolate using the
    line through the two shortest tenors

    Above the longest tenor: clamp to the longest tenor's own yield 
    rather than extrapolate
    """
    tenors = list(FRED_TENOR_SERIES)

    for i in range(len(tenors)):
        if days_to_expiry < tenors[i]:
            if i == 0:
                t_lo, t_hi = tenors[0], tenors[1]
            else:
                t_lo, t_hi = tenors[i - 1], tenors[i]
            y_lo, y_hi = curve_row[t_lo], curve_row[t_hi]
            return y_lo + (y_hi - y_lo) * ((days_to_expiry - t_lo) / (t_hi - t_lo))

        if i == len(tenors) - 1:
            return curve_row[tenors[i]]

    raise ValueError(f"Invalid days_to_expiry: {days_to_expiry!r}")


def attach_risk_free_rate(df: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame produced by build_surface_inputs(), 
    attach a new `r` column: the DTE-matched risk-free rate for each row.

    """
    result = df.copy()

    parsed = result["ticker"].apply(parse_occ_ticker)
    expiry = pd.to_datetime(parsed.apply(lambda p: p.expiry))
    snapshot_date = pd.to_datetime(result["date"])
    result["days_to_expiry"] = (expiry - snapshot_date).dt.days

    def _lookup_rate(row):
        # Use the most recent available date from the curve
        # This is because the bond-market calendar is different from NYSE's
        curve_row = curve.asof(pd.Timestamp(row["date"]))
        return get_risk_free_rate(curve_row, row["days_to_expiry"])

    result["r"] = result.apply(_lookup_rate, axis=1)

    return result
