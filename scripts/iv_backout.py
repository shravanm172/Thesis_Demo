"""
Turns process_date()'s two resolved snapshot tables (options, equity) into
per-contract implied volatilities via Barone-Adesi-Whaley (BAW) inversion,
one (ticker, date, snapshot_type) surface at a time.

Pipeline position: process_date() (snapshot_filter.py) resolves raw flat
files down to one row per option contract and one row per underlying, each
independently snapshotted to its own nearest-available minute within the
5-minute lookback window. This module picks up from there - it does NOT
touch raw flat files or do any minute-level snapshot resolution itself.

Spot price convention: a single canonical spot price per (ticker, date,
snapshot_type) surface, taken from the equity leg's own independently-
resolved snapshot - never a separate price per option contract, even though
individual contracts may have resolved to a different minute than the
equity leg did (that's normal trade staleness, not a problem to solve).
`close` is used on both legs (the option's own traded price and the
equity's spot price): it's the last trade within whichever minute got
resolved, consistent with the "most recent price at or before the target
time" rule the whole snapshot layer is built on. `open` would reach further
from the target time for no benefit, and `high`/`low` aren't tied to any
specific instant at all, so neither is a real candidate.

If a surface's equity leg didn't resolve at all within its own 5-minute
window (no valid spot price to anchor the surface to), every option
contract's row for that (ticker, date, snapshot_type) key gets dropped -
the whole surface, not just some of it - since there is no canonical spot
price to associate with the surface at all.

attach_spot_price() is implemented and verified. Everything else -
build_surface_inputs(), check_no_arbitrage(), back_out_implied_vol(),
build_surface_ivs() - is scaffolding only. See TODOs.
"""

import pandas as pd
from datetime import date, datetime, timedelta

from snapshot_filter import process_date

def attach_spot_price(options_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join the equity (underlying stock) price onto every option contract
    row sharing the same (underlying, date, snapshot_type) key.
    
    Drop the whole surface if there is no valid equity data for this window
    """
    equity_spot = equity_df[["ticker", "date", "snapshot_type", "close"]].rename(
        columns={"ticker": "underlying", "close": "spot_price"}
    )

    return pd.merge(
        options_df,
        equity_spot,
        on=["underlying", "date", "snapshot_type"],
        how="inner",
    )


def build_surface_inputs(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Given the manifest from dataset_builder.build_dataset_manifest(),
    which only names which (ticker, date, snapshot_type) surfaces matter,
    return the tagged and concatenated df with one row per contract, 
    per manifest row, per leg (T-1 or T0)
    """

    unique_days = set()
    for row in manifest[["t_minus_1", "t0"]].itertuples():
        unique_days.add(row.t_minus_1)
        unique_days.add(row.t0)

    surface_pieces = []
    for d in unique_days:
        try:
            options_df, equity_df = process_date(d)
        except FileNotFoundError:   # If there is no data for this date, then log and skip
            print(f"SKIPPING {d}: no downloaded flat file (outside plan history range).")
            continue
        surface_piece = attach_spot_price(options_df=options_df, equity_df=equity_df)
        surface_pieces.append(surface_piece)

    all_surfaces = pd.concat(surface_pieces, ignore_index=True)

    # Reshape the manifest such that rows are split into T-1 and T0
    # so that they can be joined on ["underlying", "date", "snapshot_type"] 
    # with all_surfaces
    t1_leg = manifest.copy()
    t1_leg["date"] = manifest["t_minus_1"]
    t1_leg["snapshot_type"] = "T-1_target"

    t0_leg = manifest.copy()
    t0_leg["date"] = manifest["t0"]
    t0_leg["snapshot_type"] = "T0_target"

    manifest_long = pd.concat([t1_leg, t0_leg], ignore_index=True)
    manifest_long = manifest_long.rename(columns={"ticker": "underlying"})

    # Join the manifest columns to all_surfaces
    tagged = pd.merge(all_surfaces, manifest_long, on=["underlying", "date", "snapshot_type"], how="inner")

    return tagged


def check_no_arbitrage(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: implement. Flag (don't silently drop yet - keep the reason
    visible) any row whose option price falls below its intrinsic value,
    since there's no valid IV solution for those and they'd either fail to
    converge or converge to a meaningless value if forced through the BAW
    solver anyway.

    Intrinsic value depends on option_type - call: max(spot - strike, 0),
    put: max(strike - spot, 0). Needs strike and option_type parsed onto
    each row first (see occ_parser.parse_occ_ticker) - not yet present on
    options_df as of attach_spot_price()'s output.
    """
    raise NotImplementedError


def back_out_implied_vol(row) -> float | None:
    """
    TODO: implement. Barone-Adesi-Whaley inversion for one option contract
    row - given spot_price, strike, time-to-expiry, risk-free rate (FRED),
    dividend yield (static per-ticker), option_type, and the contract's own
    traded price (close), solve for the implied volatility that reprices
    the option to match that traded price.

    Root-finder (e.g. Brent's method) may fail to converge for deep ITM/OTM
    or very short-dated contracts - return None (not an exception) for
    those, so callers can treat "no IV" as a normal, expected outcome (a
    missing grid cell for the interpolation step) rather than a crash.
    """
    raise NotImplementedError


def build_surface_ivs(options_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: implement. Orchestrator tying the above together for one batch of
    surfaces: attach_spot_price() -> check_no_arbitrage() ->
    back_out_implied_vol() per remaining row -> return a tidy DataFrame of
    one implied vol per (contract, date, snapshot_type), with arbitrage
    violations and convergence failures both present as flagged/null rather
    than silently absent, so downstream code can distinguish "no valid IV"
    from "this row was never here."
    """
    raise NotImplementedError
