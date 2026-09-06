"""Offer candidates from the symbols we already stream and already trust.

WHY THIS EXISTS
---------------
Measured 2026-09-06, one hour after production came back up: **79% of ghost
candidates were offered on symbols the gates refuse**, and eleven of the
thirteen symbols that passed every gate were never offered a candidate at all.
The pipeline produced two entries in an hour while the feed carried 774 ticks
per ten minutes across 25 symbols.

The cause is structural rather than a bug. ``select_candidates`` in
tools/c0d3rV2/crypto_paper_trade.py builds its list from
``fetch_dexscreener_candidates() + fetch_gecko_new_pool_candidates()`` -- it
hunts NEW pools. That is a real job and worth keeping: new tokens are where
outsized moves live. But it means a symbol we have streamed for a week, whose
edge and motion and stop-survivability we have already measured, can never
become a candidate, because it is not new.

So the discovery engine spends its output where trades cannot happen, while
the symbols that CAN trade sit idle. Over seven days of feed history, these
cleared a 0.65% round trip inside a 15-minute window often enough to be worth
trading:

    AERO-WETH        44.4% of windows        EURC-WETH       33.3%
    CLANKER-USDC     25.0%                   VIRTUAL-WETH    16.7%
    TIBBIR-VIRTUAL   11.1%                   SOL-CBBTC       10.0%
    AERO-USDC         5.5%

Six of those seven had never been offered a candidate.

WHAT THIS DOES AND DOES NOT DO
------------------------------
It proposes symbols. It does not decide anything: every candidate it returns
still passes through the same gates in the same order -- symbol_edge_gate,
symbol_motion_gate, stop_survivability_gate, strategy_edge_gate -- and through
the solver that arbitrates them. This module cannot make a trade happen that
the gates would refuse; it can only stop the pipeline from starving while
eligible symbols go unexamined.

It also does not compete with new-pool discovery. The two answer different
questions -- "what just appeared" and "what have we already proven" -- and a
pipeline that only asks the first will keep rediscovering the same
unprofitable memecoins while ignoring the instruments it has a week of
evidence about.

THE SELECTION RULE
------------------
A symbol is offered when all three hold:

  1. It is ticking NOW (a stale symbol cannot be entered or exited).
  2. Its feed is dense enough that a stop can be evaluated.
  3. It has historically cleared the round-trip cost inside the hold window
     often enough to pay for a trade.

That third condition is the same question ``symbol_motion_gate`` asks, and it
is asked here deliberately rather than left to the gate: a candidate that will
be refused downstream is a wasted slot in the offer list, which is precisely
the failure this module exists to fix.

RANKED, NOT FILTERED. The caller takes the top N, so the ordering is the
product: symbols that clear their cost more often are offered first.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "trading_cache.db"

#: Round-trip cost as a fraction of notional, measured from receipts. The same
#: figure symbol_edge_gate and symbol_motion_gate test against.
ROUND_TRIP_COST = float(os.getenv("STREAMED_CANDIDATE_COST", "0.0065"))

#: The hold window a move has to happen inside. Matches the 15-minute window
#: symbol_motion_gate uses so the two cannot disagree about the same symbol.
WINDOW_SEC = float(os.getenv("STREAMED_CANDIDATE_WINDOW_SEC", "900"))

#: Share of windows that must clear the cost before a symbol is worth offering.
MIN_CLEAR_RATE = float(os.getenv("STREAMED_CANDIDATE_MIN_CLEAR_RATE", "0.03"))

#: How much feed history to judge on.
LOOKBACK_SEC = float(os.getenv("STREAMED_CANDIDATE_LOOKBACK_SEC", str(7 * 86400)))

#: A symbol must have ticked this recently to be offered at all.
FRESH_SEC = float(os.getenv("STREAMED_CANDIDATE_FRESH_SEC", "600"))

#: Ticks needed in the lookback before the clear-rate means anything.
MIN_TICKS = int(os.getenv("STREAMED_CANDIDATE_MIN_TICKS", "50"))

#: Windows needed before a rate is a rate rather than an anecdote.
MIN_WINDOWS = int(os.getenv("STREAMED_CANDIDATE_MIN_WINDOWS", "5"))

CACHE_SEC = float(os.getenv("STREAMED_CANDIDATE_CACHE_SEC", "300"))

_cache: List["StreamedCandidate"] = []
_cache_built_at: float = 0.0


@dataclass
class StreamedCandidate:
    """A symbol worth offering, with the evidence that says so."""

    symbol: str
    clear_rate: float
    windows: int
    ticks: int
    last_tick_age_sec: float

    @property
    def rationale(self) -> str:
        return (
            f"cleared {ROUND_TRIP_COST * 100:.2f}% in "
            f"{self.clear_rate * 100:.1f}% of {self.windows} "
            f"{int(WINDOW_SEC / 60)}-minute windows over {self.ticks} ticks"
        )


def _clear_rate(rows: Sequence[Tuple[float, float]]) -> Tuple[float, int]:
    """Share of non-overlapping windows whose high cleared the round trip.

    Non-overlapping because overlapping windows would count one move many
    times and inflate the rate for a symbol that moved once.

    The high is measured from the window's OPEN, not from its low: an entry
    happens at the price available when the decision is made, and a move that
    only looks profitable measured from a low nobody could have bought at is
    the selection bias this repo has shipped before.
    """
    if len(rows) < 3:
        return 0.0, 0
    cleared = 0
    windows = 0
    index = 0
    total = len(rows)
    while index < total:
        start_ts, start_price = rows[index]
        if start_price <= 0:
            index += 1
            continue
        high = start_price
        cursor = index
        while cursor < total and rows[cursor][0] <= start_ts + WINDOW_SEC:
            if rows[cursor][1] > high:
                high = rows[cursor][1]
            cursor += 1
        if cursor - index >= 3:
            windows += 1
            if (high - start_price) / start_price >= ROUND_TRIP_COST:
                cleared += 1
        index = cursor if cursor > index else index + 1
    if windows == 0:
        return 0.0, 0
    return cleared / windows, windows


def _rebuild(now: float) -> None:
    global _cache_built_at
    found: List[StreamedCandidate] = []
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001 - no feed is not a reason to raise
        _cache_built_at = now
        return
    try:
        symbols = [
            str(row[0]) for row in conn.execute(
                "SELECT DISTINCT symbol FROM market_stream WHERE ts > ?",
                (now - FRESH_SEC,),
            ) if row and row[0]
        ]
        for symbol in symbols:
            rows = [
                (float(ts), float(price))
                for ts, price in conn.execute(
                    "SELECT ts, price FROM market_stream "
                    "WHERE symbol = ? AND ts > ? ORDER BY ts",
                    (symbol, now - LOOKBACK_SEC),
                )
                if price is not None and float(price) > 0.0
            ]
            if len(rows) < MIN_TICKS:
                continue
            rate, windows = _clear_rate(rows)
            if windows < MIN_WINDOWS or rate < MIN_CLEAR_RATE:
                continue
            found.append(StreamedCandidate(
                symbol=symbol,
                clear_rate=rate,
                windows=windows,
                ticks=len(rows),
                last_tick_age_sec=now - rows[-1][0],
            ))
    except Exception:  # noqa: BLE001
        _cache_built_at = now
        return
    finally:
        conn.close()

    found.sort(key=lambda c: c.clear_rate, reverse=True)
    _cache.clear()
    _cache.extend(found)
    _cache_built_at = now


def streamed_candidates(limit: Optional[int] = None) -> List[StreamedCandidate]:
    """Symbols we already stream that have earned a look, best first.

    Fails EMPTY rather than raising: a candidate source that throws would stop
    the trading cycle, and having no suggestions is a normal state (a quiet
    market, a cold feed) rather than an error.
    """
    try:
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
    except Exception:  # noqa: BLE001
        return []
    return list(_cache[:limit] if limit else _cache)


def as_dicts(limit: Optional[int] = None) -> List[Dict[str, object]]:
    """The same list in the shape the bus and dashboards consume."""
    return [
        {
            "symbol": candidate.symbol,
            "clear_rate": round(candidate.clear_rate, 4),
            "windows": candidate.windows,
            "ticks": candidate.ticks,
            "last_tick_age_sec": round(candidate.last_tick_age_sec, 1),
            "rationale": candidate.rationale,
            "source": "streamed_symbol_candidates",
        }
        for candidate in streamed_candidates(limit)
    ]
