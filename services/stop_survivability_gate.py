"""Refuse symbols whose price jumps further in one tick than the stop can bind.

WHY THIS EXISTS
---------------
Measured 2026-09-06. The live lane was frozen with ES95 tail risk at 0.1241
against a 0.10 guardrail, and the ENTIRE breach was one trade:

    MOONBASE-USDC   -12.41%   27h ago
    every other trade in the 48h window   under 3%

Remove that single trade and ES95 falls to 0.0291 -- comfortably passing. One
position on one symbol was holding the whole live lane shut.

The obvious readings were both wrong:

  * "the stop is too wide" -- it is not. `GHOST_STOP_LOSS_PCT` is 0.02 and
    trading/bot.py records the REALISED loss in the exit reason, so
    "stop_loss:-0.1241" means the position was already down 12.41% by the time
    a tick arrived to evaluate a 2% stop.
  * "the feed was too sparse" -- that is real but insufficient. Sparse feeds
    (<30 ticks/hour) breach by 15.70% on average against 5.00% for dense ones,
    3.1x worse, and the two -56% catastrophes had 0 and 1 ticks in the hour
    before exit. But MOONBASE had **70 ticks** in that hour and still lost
    12.41%, so density alone does not explain it.

What does explain it is the size of a SINGLE TICK JUMP:

    symbol          ticks   p50 jump   p99 jump    2% stop holds?
    MOONBASE-USDC     618     0.000%   99381.16%   NO
    OMARCHY-USDC      155     0.056%      17.55%   NO
    BSTONK-USDC      2640     0.050%      10.91%   NO
    CP-USDC           737     0.063%       3.96%   NO
    AERO-USDC        3418     0.002%       0.76%   yes
    CBBTC-USDC       3437     0.002%       0.47%   yes

A stop is a promise about the worst case. That promise is only keepable if the
price can be observed moving through the stop level -- if one tick can carry
the price from above the stop to far below it, the stop is decoration. Against
MOONBASE's 99,381% p99 jump (feed contamination: a denomination flip, not a
real move) no stop of any width binds.

WHAT THIS GATE ASKS
-------------------
Not "has this symbol lost money" (symbol_edge_gate asks that, and needs closed
round trips it does not have for a new symbol). Not "can this symbol move
enough to pay for a round trip" (symbol_motion_gate asks that). This asks the
question neither does, and the one that MOONBASE failed:

    can this symbol's stop actually be enforced on this symbol's feed?

It is answerable BEFORE the first trade, from the price feed alone, which is
what makes it useful on exactly the symbols the book-based gates cannot judge.

BANS ONLY, NEVER PROMOTES -- the same asymmetry symbol_edge_gate documents. A
tight p99 jump does not mean a symbol is worth trading; it only means a stop
placed on it means something.

FAILS OPEN. A symbol with too few ticks to measure is ALLOWED here, because
"unmeasurable" is not "dangerous" and this gate is not the one that polices
thin feeds -- symbol_motion_gate and the ATF scout's `_feed_is_dense_enough`
already refuse those, and stacking a third refusal on the same condition would
make a quiet feed look like three independent problems.
"""
from __future__ import annotations

import os
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from services.logging_service import log_message
except Exception:  # noqa: BLE001 - logging must never block a trade
    def log_message(*_args, **_kwargs):  # type: ignore[misc]
        return None

DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "trading_cache.db"

#: Ticks needed before a p99 means anything. Below this the gate abstains.
MIN_TICKS = int(os.getenv("STOP_SURVIVE_MIN_TICKS", "200"))

#: How far back to read the feed.
WINDOW_SEC = float(os.getenv("STOP_SURVIVE_WINDOW_SEC", str(7 * 86400)))

#: The stop the p99 jump is measured against. Defaults to the same value
#: trading/bot.py enforces, so the gate and the stop cannot drift apart.
STOP_PCT = float(os.getenv("GHOST_STOP_LOSS_PCT", "0.02"))

#: How much bigger than the stop a p99 jump may be before the symbol is
#: refused. 1.0 would demand the stop survive EVERY tick in the 99th
#: percentile, which is stricter than necessary: a jump slightly larger than
#: the stop overshoots slightly, and the tail guardrail already tolerates
#: that. 2.0 refuses symbols where a single tick can double the intended
#: loss.
MAX_JUMP_RATIO = float(os.getenv("STOP_SURVIVE_MAX_JUMP_RATIO", "2.0"))

#: Symbols never refused regardless of feed. Empty by default.
NEVER_BAN = {
    s.strip().upper()
    for s in os.getenv("STOP_SURVIVE_NEVER_BAN", "").split(",")
    if s.strip()
}

CACHE_SEC = float(os.getenv("STOP_SURVIVE_CACHE_SEC", "600"))

_cache: Dict[str, Tuple[float, str]] = {}
_cache_built_at: float = 0.0


def _tick_jumps(conn: sqlite3.Connection, symbol: str,
                since: float) -> List[float]:
    """Absolute fractional price change between consecutive ticks.

    Fractional rather than absolute because the stop is a fraction: a $0.01
    move means something entirely different on CBBTC than on a sub-cent
    memecoin, and comparing dollars to a percentage is the units error this
    repo has shipped more than once.
    """
    prices: List[float] = []
    try:
        rows = conn.execute(
            "SELECT price FROM market_stream WHERE symbol = ? AND ts > ? "
            "ORDER BY ts",
            (symbol, since),
        )
        for (price,) in rows:
            try:
                value = float(price)
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                prices.append(value)
    except Exception:  # noqa: BLE001
        return []

    jumps: List[float] = []
    for index in range(len(prices) - 1):
        previous = prices[index]
        if previous <= 0.0:
            continue
        jump = abs(prices[index + 1] - previous) / previous
        # A jump is only meaningful if it is finite. Feed contamination can
        # produce inf/nan, and those must not poison a percentile.
        if jump == jump and jump != float("inf"):
            jumps.append(jump)
    return jumps


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def _rebuild(now: float) -> None:
    global _cache_built_at
    verdicts: Dict[str, Tuple[float, str]] = {}
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001 - no feed is not a reason to block trading
        _cache_built_at = now
        return

    try:
        since = now - WINDOW_SEC
        symbols = [
            str(row[0]) for row in conn.execute(
                "SELECT DISTINCT symbol FROM market_stream WHERE ts > ?",
                (since,),
            ) if row and row[0]
        ]
        ceiling = STOP_PCT * MAX_JUMP_RATIO
        for symbol in symbols:
            if symbol.upper() in NEVER_BAN:
                continue
            jumps = _tick_jumps(conn, symbol, since)
            if len(jumps) < MIN_TICKS:
                # Not enough evidence to judge. Abstain -- see the module
                # docstring on why this gate does not police thin feeds.
                continue
            p99 = _percentile(jumps, 0.99)
            if p99 <= ceiling:
                continue
            verdicts[symbol.upper()] = (
                p99,
                f"p99 single-tick jump {p99 * 100:.2f}% over {len(jumps)} "
                f"ticks exceeds {ceiling * 100:.2f}% "
                f"({STOP_PCT * 100:.2f}% stop x {MAX_JUMP_RATIO:.1f}); "
                f"a stop cannot bind on this feed",
            )
    except Exception:  # noqa: BLE001
        _cache_built_at = now
        return
    finally:
        conn.close()

    for symbol, (_p99, detail) in sorted(verdicts.items()):
        if symbol not in _cache:
            log_message(
                "trading",
                f"STOP SURVIVABILITY GATE: refusing {symbol} -- {detail}",
                severity="warning",
            )
    _cache.clear()
    _cache.update(verdicts)
    _cache_built_at = now


def refusal_reason(symbol: str) -> Optional[str]:
    """Why a stop cannot be enforced on this symbol, or None to allow it.

    Fails OPEN on any error: a gate that cannot read the feed has no evidence
    to refuse on, and blocking every symbol because a database was locked
    would be far worse than letting one bad trade through.
    """
    try:
        name = str(symbol or "").strip().upper()
        if not name:
            return None
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
        verdict = _cache.get(name)
        return verdict[1] if verdict else None
    except Exception:  # noqa: BLE001
        return None


def refused_symbols() -> Dict[str, str]:
    """Current verdicts, for diagnostics and dashboards."""
    try:
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
    except Exception:  # noqa: BLE001
        return {}
    return {symbol: detail for symbol, (_p99, detail) in _cache.items()}
