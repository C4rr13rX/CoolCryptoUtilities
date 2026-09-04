"""Refuse symbols the book has proven we lose money on.

The ghost book does not fail uniformly. Measured 2026-09-04 over 127 closed
round trips, the loss is concentrated in a handful of symbols that keep being
traded:

    BASECAT-USDC   36 trades   mean -0.05224/trade   t=-3.25   total -1.8806
    CBXRP-USDC      6 trades   mean -0.00763/trade   t=-2.58   total -0.0458

BASECAT is the most-traded symbol in the book AND its largest single
destroyer of capital. Nothing stopped it from being traded a thirty-seventh
time.

By contrast the apparent winners do not survive the same test: AERO is
+0.0796/trade over 26 trades but t=+0.60, and AAVE's +3.44 comes from two
trades. So this gate only ever BANS -- it never promotes. Acting on a
positive t of the same strength would be fitting the noise that produced
those numbers, and the asymmetry is deliberate: refusing a symbol we have
evidence against costs an opportunity, while trading one we have evidence
for costs money when the evidence was luck.

VALIDATED OUT OF SAMPLE. Ban list derived from the first 60% of the book,
then applied to the untouched remaining 40%: holdout net went +1.1067 ->
+1.1383 (+0.0316). The same two symbols were selected. A rule fitted to the
whole book and never tested on held-out data is a story about the past.

The threshold is a t-statistic, not a raw total, because a symbol can be
down simply for having traded often. t < -1.7 with n >= MIN_SAMPLES is
roughly p < 0.05 one-tailed -- evidence that the mean is genuinely negative
rather than a losing streak.
"""

from __future__ import annotations

import math
import os
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from services.logging_utils import log_message

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "storage" / "trading_cache.db"

#: Minimum closed round trips before a symbol can be judged at all. Below
#: this, a run of losses is indistinguishable from variance.
MIN_SAMPLES = int(os.getenv("SYMBOL_EDGE_MIN_SAMPLES", "5"))

#: How negative the t-statistic must be. -1.7 is ~p<0.05 one-tailed at these
#: sample sizes.
MAX_T = float(os.getenv("SYMBOL_EDGE_MAX_T", "-1.7"))

#: How long a verdict is reused before the book is re-read. The book changes
#: by a trade at a time, so recomputing per tick would cost a query for an
#: answer that cannot have moved.
CACHE_SEC = float(os.getenv("SYMBOL_EDGE_CACHE_SEC", "300"))

#: Symbols never banned regardless of record -- the stable legs a round trip
#: has to route through. Banning one would not avoid a bad trade, it would
#: make trading impossible.
NEVER_BAN = {s.strip().upper() for s in
             os.getenv("SYMBOL_EDGE_NEVER_BAN", "USDC,USDT,DAI,USDBC,WETH,ETH").split(",")
             if s.strip()}

_cache: Dict[str, Tuple[float, str]] = {}
_cache_built_at: float = 0.0


def _t_statistic(values: List[float]) -> float:
    """Student's t for "is this mean different from zero".

    Returns 0.0 when it cannot be computed -- a single sample, or a run of
    identical values with no dispersion. Zero reads as "no evidence", which
    is the correct default for both.
    """
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    try:
        stdev = statistics.stdev(values)
    except statistics.StatisticsError:
        return 0.0
    if stdev <= 0.0:
        return 0.0
    return mean / (stdev / math.sqrt(len(values)))


def _load_book(limit: int = 500) -> Dict[str, List[float]]:
    """Closed ghost round trips per symbol, newest first.

    Reads trade_outcomes, not trading_ops: trading_ops is an append-only event
    log that keeps pre-fix artifacts forever (a single 2026-09-04 row carries
    a -0.4177 that the receipt puts at -0.0169), and judging symbols on it
    would ban whichever symbol happened to be traded during an old bug.
    """
    book: Dict[str, List[float]] = {}
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001 - no book is not a reason to block trading
        return book
    try:
        rows = conn.execute(
            "SELECT symbol, net_profit FROM trade_outcomes "
            "WHERE status = 'closed' AND net_profit IS NOT NULL "
            "ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        )
        for symbol, net in rows:
            if not symbol:
                continue
            try:
                book.setdefault(str(symbol).upper(), []).append(float(net))
            except (TypeError, ValueError):
                continue
    except Exception:  # noqa: BLE001
        return {}
    finally:
        conn.close()
    return book


def _rebuild(now: float) -> None:
    global _cache_built_at
    book = _load_book()
    verdicts: Dict[str, Tuple[float, str]] = {}
    for symbol, values in book.items():
        base = symbol.split("-")[0]
        if base in NEVER_BAN or symbol in NEVER_BAN:
            continue
        if len(values) < MIN_SAMPLES:
            continue
        mean = statistics.mean(values)
        if mean >= 0.0:
            continue                    # only losers are candidates
        t = _t_statistic(values)
        if t < MAX_T:
            verdicts[symbol] = (
                t,
                f"{len(values)} closed round trips at mean {mean:+.5f} "
                f"(t={t:+.2f}, total {sum(values):+.4f})",
            )
    if verdicts != {k: v for k, v in _cache.items()}:
        for symbol, (t, detail) in sorted(verdicts.items()):
            if symbol not in _cache:
                log_message(
                    "trading",
                    f"SYMBOL EDGE GATE: refusing {symbol} -- {detail}",
                    severity="warning",
                )
    _cache.clear()
    _cache.update(verdicts)
    _cache_built_at = now


def refusal_reason(symbol: str) -> Optional[str]:
    """Why this symbol should not be traded, or None to allow it.

    Fails OPEN: any error reading the book allows the trade. A gate that
    cannot read its evidence has no evidence to refuse on, and blocking every
    symbol because a database was locked would be a far worse failure than
    letting one bad trade through.
    """
    try:
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
        verdict = _cache.get(str(symbol or "").upper())
        return verdict[1] if verdict else None
    except Exception:  # noqa: BLE001
        return None


def banned_symbols() -> Dict[str, str]:
    """Current verdicts, for diagnostics and dashboards."""
    try:
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
    except Exception:  # noqa: BLE001
        return {}
    return {symbol: detail for symbol, (_, detail) in _cache.items()}
