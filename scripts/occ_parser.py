"""
Parses Massive/Polygon-style OCC option tickers, e.g. "O:AAPL260904C00120000",
into their component fields: underlying, expiry date, option type, strike.

Format: "O:" prefix + root symbol (variable length) + a FIXED 15-character
suffix = 6-digit expiry (YYMMDD) + 1-char option type (C/P) + 8-digit strike
(price in thousandths of a dollar, so fractional strikes are representable).

Because the root symbol's length varies but the suffix length never does,
this parses right-to-left: strip the prefix, take the last 15 characters as
the suffix, and whatever remains is the root - NOT a left-to-right parse,
which would need to know the root's length in advance.
"""

from dataclasses import dataclass
from datetime import date

_PREFIX = "O:"
_SUFFIX_LENGTH = 15  # 6 (expiry) + 1 (type) + 8 (strike)


@dataclass(frozen=True)
class ParsedOccTicker:
    underlying: str
    expiry: date
    option_type: str  # "C" or "P"
    strike: float
    raw: str  # the original ticker string, kept for debugging/traceability


def parse_occ_ticker(ticker: str) -> ParsedOccTicker:
    if not ticker.startswith(_PREFIX):
        raise ValueError(f"Not a Massive/Polygon OCC option ticker (missing '{_PREFIX}' prefix): {ticker!r}")

    body = ticker[len(_PREFIX):]
    if len(body) <= _SUFFIX_LENGTH:
        raise ValueError(f"Ticker too short to contain a root symbol + full OCC suffix: {ticker!r}")

    underlying = body[:-_SUFFIX_LENGTH]
    suffix = body[-_SUFFIX_LENGTH:]

    expiry_str = suffix[0:6]
    option_type = suffix[6]
    strike_str = suffix[7:15]

    if option_type not in ("C", "P"):
        raise ValueError(f"Expected option type 'C' or 'P' at position 7 of suffix, got {option_type!r}: {ticker!r}")

    try:
        # YYMMDD - assume 20xx century, which holds for every date this
        # pipeline will ever touch (demo covers 2024-2026). This assumption
        # would break for legacy contracts from the 1900s, which don't
        # exist in Massive's flat files anyway - not a real-world concern
        # here, just worth naming explicitly rather than leaving implicit.
        yy = int(expiry_str[0:2])
        mm = int(expiry_str[2:4])
        dd = int(expiry_str[4:6])
        expiry = date(2000 + yy, mm, dd)
    except ValueError as e:
        raise ValueError(f"Could not parse expiry date {expiry_str!r} from ticker {ticker!r}: {e}") from None

    try:
        strike = int(strike_str) / 1000
    except ValueError as e:
        raise ValueError(f"Could not parse strike {strike_str!r} from ticker {ticker!r}: {e}") from None

    return ParsedOccTicker(
        underlying=underlying,
        expiry=expiry,
        option_type=option_type,
        strike=strike,
        raw=ticker,
    )
