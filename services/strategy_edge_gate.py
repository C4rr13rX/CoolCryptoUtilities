"""Refuse STRATEGIES the book has proven cannot pay their own round trip.

``symbol_edge_gate`` bans a symbol once the book proves we lose money on it.
This asks the same question of the other axis: which STRATEGY opened the
trade. A symbol can be perfectly tradable while one strategy's way of
trading it loses money every time.

WHY THIS EXISTS
---------------
Measured 2026-09-05. The graduation bar is 25 ghost trades at >=55% win
(trading/pipeline.py:642-647). Only two strategies clear it -- ``atf_static``
and ``atf_static_scout`` -- and the other 31 share **106 ghost trades, 3.4
each**. None of them failed on win rate. Every one is starved of sample.

That looked like an argument for handing the starved strategies evidence
faster, and it was half an argument. The book says where the evidence is
going:

    atf_static_scout      352 of 458 ghost trades (77%)
    obv_accumulation@1w     9 closed   mean return  -1.668%/trade
    obv_accumulation@5d     7 closed   mean return  -0.556%/trade
    bus_schedule            4 closed   mean return  -1.366%/trade

``atf_static_scout`` hardcodes ``wallet="ghost"``, publishes
``live_execution_enabled: False``, and contains **zero** swap calls -- it is
a simulator that can never place a trade. It consumes three quarters of all
evidence the system generates. Most of the remainder goes to strategies whose
measured return does not cover the ~0.65% round trip.

So the evidence budget is spent almost entirely on participants that either
cannot trade live or lose money when they do, while the strategies that DO
clear costs -- donchian_breakout@5d at +4.893%/trade, the rsi_reversal family
-- have one to four trades each. Feeding the starved strategies faster
without this gate would graduate the losers faster too.

A FALSE START WORTH RECORDING
-----------------------------
The first design gated on FEE DRAG (fee_cost / gross_profit), because
obv_accumulation@1w pays 454% of its gross in fees. That ratio is a trap.
Measured per trade, fee/notional is ~0.65-1.7% for EVERY strategy: it is a
property of the venue, not of the strategy, and it is stable (stdev 0.000% on
most). The 454% came from dividing by a gross that shrinks toward zero, so
the ratio explodes on a WEAK EDGE rather than identifying an EXPENSIVE one.
Gating on it would have banned strategies for the sin of trading quietly.

What actually separates them is mean net return per trade against the round
trip cost, which is precisely the test ``symbol_edge_gate`` already validated.

BANS ONLY, NEVER PROMOTES
-------------------------
The same asymmetry the symbol gate documents, for the same reason. Refusing a
strategy we have evidence against costs an opportunity; promoting one we have
evidence FOR costs money whenever that evidence was luck. And the evidence
here is thin enough that the asymmetry matters more than usual --
``rsi_reversal@1w`` shows +0.7112 on a SINGLE trade, which is noise wearing
the costume of an edge. This module would never act on it.

VALIDATED OUT OF SAMPLE, walk-forward, 67 attributed round trips:

    split  fit  hold  bans                          holdout: all -> gated
     50%    33    34  none (too little fit data)      -21.331% -> -21.331%
     60%    40    27  none (too little fit data)      -13.265% -> -13.265%
     70%    46    21  obv_accumulation@1w              -5.595% ->  +3.607%
     75%    50    17  obv_accumulation@1w              +2.508% ->  +4.970%
     80%    53    14  + obv_accumulation@5d            +6.711% -> +10.067%

Every split where the gate had enough data to fire improved the holdout, and
none was ever made worse. The 70% split is the one that matters: it turns a
LOSING holdout positive by refusing five trades.

MIN_SAMPLES IS 3, NOT 5
-----------------------
The symbol gate uses 5. Here that threshold never fires: only ONE strategy
reaches n>=5 inside a 40-trade fit window, because the book is split 30 ways.
A gate that cannot fire is the same as no gate. Three is the smallest sample
that admits a t-statistic with any dispersion behind it, and it is paired
with a strict t < -1.7 so a strategy must be consistently bad rather than
merely unlucky -- the three strategies it bans today carry t of -8.39, -3.70
and -2.51, nowhere near the boundary.

This does NOT lower the graduation bar. A banned strategy stops consuming
evidence; a strategy that clears this gate still needs its 25 trades at 55%.
"""
from __future__ import annotations

import collections
import json
import math
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

#: Closed round trips a strategy needs before it can be judged. See the
#: module docstring: 5 (the symbol gate's value) never fires on a book split
#: this many ways.
MIN_SAMPLES = int(os.getenv("STRATEGY_EDGE_MIN_SAMPLES", "3"))

#: t-statistic on EXCESS return below which a strategy is refused. Strict
#: because MIN_SAMPLES is small.
MAX_T = float(os.getenv("STRATEGY_EDGE_MAX_T", "-1.7"))

#: FALLBACK round-trip cost, used only when the book holds too few real fees
#: to measure one. The live verdict calls ``_cost()`` -> ``round_trip_cost()``.
#:
#: This literal used to be read DIRECTLY, and its comment claimed it was "the
#: same figure symbol_edge_gate tests against". That stopped being true when
#: symbol_edge_gate moved to the measured cost: it charges 0.4653% while this
#: gate was charging 0.6500%, a 0.185%-of-notional surcharge on every strategy
#: it judged. Two gates answering one question at two different bars is the
#: shape this repo has shipped before.
#:
#: It is not a rounding difference. Re-priced 2026-09-10 against
#: services/round_trip_cost (0.4653%), 2 of the 7 standing bans are entirely
#: an artifact of the surcharge and lift::
#:
#:     rsi_reversal          10 trips  mean -0.625%  t=-1.85 -> -1.44  LIFTS
#:     stochastic_reversal    4 trips  mean -0.168%  t=-1.86 -> -1.29  LIFTS
#:
#:     bus_schedule           4 trips  mean -1.366%  t=-3.70 -> -3.36  stands
#:     donchian_breakout@1d   3 trips  mean -1.082%  t=-2.54 -> -2.27  stands
#:     obv_accumulation@1w    9 trips  mean -1.668%  t=-8.39 -> -7.72  stands
#:     obv_accumulation@3d    3 trips  mean -0.579%  t=-2.04 -> -1.74  stands
#:     obv_accumulation@5d    8 trips  mean -0.598%  t=-2.99 -> -2.55  stands
#:
#: rsi_reversal is the strategy CLOSEST to graduation on tradeable evidence.
#: It was being refused entries for failing to clear a cost it is not charged.
#: A gate refusing a strategy that pays for itself is wrong, not safe -- and
#: nothing here is loosened: MAX_T, MIN_SAMPLES and the ordering are untouched,
#: the bar is simply asked at the price the receipts actually charge.
ROUND_TRIP_COST = float(os.getenv("STRATEGY_EDGE_ROUND_TRIP_COST", "0.0065"))

#: Positions below this notional produce meaningless returns (a $0.0001
#: position divides into a huge fraction).
MIN_NOTIONAL = float(os.getenv("STRATEGY_EDGE_MIN_NOTIONAL", "0.01"))

CACHE_SEC = float(os.getenv("STRATEGY_EDGE_CACHE_SEC", "300"))

#: Never refused however the book reads.
#:
#: ``atf_static`` is the ONLY strategy that has ever placed a profitable live
#: trade (16 closed, +0.1841 net, mean +1.180%/trade). Banning the sole live
#: earner because a ghost sample turned against it would take live trading to
#: zero, which is the failure this whole module exists to prevent. It is also
#: comfortably above cost today; this is a guard against a future sample, not
#: a special pleading for a loser.
NEVER_BAN = {
    strategy.strip()
    for strategy in os.getenv("STRATEGY_EDGE_NEVER_BAN", "atf_static").split(",")
    if strategy.strip()
}

_cache: Dict[str, Tuple[float, str]] = {}
_cache_built_at: float = 0.0


def _t_statistic(values: List[float]) -> float:
    """Student's t for "is this mean different from zero".

    Returns 0.0 for a single sample: one observation carries no evidence
    about dispersion, and zero can never trip a negative threshold.

    ZERO VARIANCE IS NOT ZERO EVIDENCE. Found by test 2026-09-05, and it is a
    genuine hole rather than a technicality: a strategy that loses EXACTLY the
    same fraction on every one of eight round trips has a stdev of 0, and the
    textbook t is undefined (a division by zero). Returning 0.0 there -- which
    reads as "no evidence" -- means the most consistent loser imaginable is
    waved through, while a noisier strategy losing the same amount on average
    is refused. Perfect consistency is the STRONGEST evidence of a real edge,
    not the absence of any.

    So when there is no dispersion the mean alone decides, and it is reported
    at a magnitude past any sane threshold because there is no uncertainty
    left to discount. Sign is preserved: a consistent WINNER returns a large
    positive t, which the caller ignores (this module never promotes).

    ``services/symbol_edge_gate.py`` has the same hole. It is less exposed
    there -- a symbol's returns vary with the market even when a strategy's
    do not -- but it is the same bug and worth fixing when that module is
    next touched.
    """
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    try:
        stdev = statistics.stdev(values)
    except statistics.StatisticsError:
        return 0.0
    if stdev <= 0.0:
        # No dispersion: the sample mean IS the population mean as far as this
        # evidence goes. Exactly zero stays zero -- a strategy that reliably
        # breaks even against cost is not evidence of anything.
        if mean == 0.0:
            return 0.0
        return math.copysign(float("inf"), mean)
    return mean / (stdev / math.sqrt(len(values)))


def _strategy_of(details: Optional[str]) -> str:
    """Which strategy opened this round trip.

    trade_outcomes has no strategy column -- the id lives in the JSON blob
    under any of three keys depending on which executor wrote the row. All
    three are read because missing one silently drops that executor's whole
    history, and a strategy with no history is one this gate cannot judge.
    """
    if not details:
        return ""
    try:
        payload = json.loads(details)
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(payload, dict):
        return ""
    meta = payload.get("meta")
    candidate = (
        payload.get("strategy_id")
        or payload.get("strategy")
        or (meta.get("strategy") if isinstance(meta, dict) else None)
    )
    return str(candidate or "").strip()


def _load_book(limit: int = 500) -> Dict[str, List[float]]:
    """Closed round trips per strategy as RETURNS, newest first.

    ``net_profit / notional``, unitless. Judged on return rather than on
    dollars for the reason symbol_edge_gate documents at length: the ghost
    book's notional varies by more than 100x within a single symbol, so a
    t-statistic over dollars measures sizing policy rather than edge.

    Reads trade_outcomes, never trading_ops: the latter is an append-only log
    that keeps pre-fix artifacts forever and would ban whichever strategy was
    running during an old bug.
    """
    book: Dict[str, List[float]] = collections.defaultdict(list)
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001 - no book is not a reason to block trading
        return {}
    try:
        rows = conn.execute(
            "SELECT net_profit, entry_price, quantity, details FROM trade_outcomes "
            "WHERE status = 'closed' AND net_profit IS NOT NULL "
            "ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        )
        for net, entry_price, quantity, details in rows:
            strategy = _strategy_of(details)
            if not strategy:
                continue
            try:
                notional = float(entry_price) * float(quantity)
                if not (notional >= MIN_NOTIONAL):  # also rejects NaN
                    continue
                ret = float(net) / notional
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if ret != ret or ret in (float("inf"), float("-inf")):
                continue
            book[strategy].append(ret)
    except Exception:  # noqa: BLE001
        return {}
    finally:
        conn.close()
    return dict(book)


def _cost() -> float:
    """What a round trip actually costs, as a fraction of notional.

    Measured from receipts by ``services.round_trip_cost``, falling back to
    ``ROUND_TRIP_COST`` when the book holds too few real fees to measure one.
    Exactly the accessor ``symbol_edge_gate._verdict`` uses, so the two gates
    cannot answer one question at two different bars again.
    """
    try:
        from services.round_trip_cost import round_trip_cost

        measured = float(round_trip_cost(db_path=DB_PATH))
    except Exception:  # noqa: BLE001
        return ROUND_TRIP_COST
    # A non-finite or non-positive measurement is not a free round trip, it is
    # a broken read. Charging zero would ban nothing and switch the gate off.
    if not math.isfinite(measured) or measured <= 0.0:
        return ROUND_TRIP_COST
    return measured


def _rebuild(now: float) -> None:
    global _cache_built_at
    book = _load_book()
    verdicts: Dict[str, Tuple[float, str]] = {}
    # Read the cost ONCE for the whole rebuild. Calling the accessor per
    # strategy would let a cache expiry land mid-sweep and judge two
    # strategies at two different bars in one pass -- the same reason
    # symbol_edge_gate._verdict reads it once per verdict.
    cost = _cost()
    for strategy, values in book.items():
        if strategy in NEVER_BAN:
            continue
        if len(values) < MIN_SAMPLES:
            continue
        mean = statistics.mean(values)
        if mean >= cost:
            continue  # pays for its own trading -- not a candidate
        # Test the EXCESS over what the round trip costs, so the null
        # hypothesis is "this strategy pays for itself" rather than "this
        # strategy is above zero". A strategy returning +0.1% per round trip
        # against the cost is a loser, and comparing to zero would miss it
        # -- the exact shape services/profit_logic_audit.py flags in code.
        excess = [value - cost for value in values]
        t = _t_statistic(excess)
        if t < MAX_T:
            verdicts[strategy] = (
                t,
                f"{len(values)} closed round trips at mean return "
                f"{mean * 100:+.3f}% vs {cost * 100:.3f}% cost "
                f"(t={t:+.2f} on excess return)",
            )
    for strategy, (_t, detail) in sorted(verdicts.items()):
        if strategy not in _cache:
            log_message(
                "trading",
                f"STRATEGY EDGE GATE: refusing {strategy} -- {detail}",
                severity="warning",
            )
    _cache.clear()
    _cache.update(verdicts)
    _cache_built_at = now


def refusal_reason(strategy_id: str) -> Optional[str]:
    """Why this strategy should not open a trade, or None to allow it.

    Fails OPEN: any error reading the book allows the trade. A gate that
    cannot read its evidence has no evidence to refuse on, and blocking every
    strategy because a database was locked would be far worse than letting
    one bad trade through.
    """
    try:
        name = str(strategy_id or "").strip()
        if not name:
            return None
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
        verdict = _cache.get(name)
        return verdict[1] if verdict else None
    except Exception:  # noqa: BLE001
        return None


def banned_strategies() -> Dict[str, str]:
    """Current verdicts, for diagnostics and dashboards."""
    try:
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
    except Exception:  # noqa: BLE001
        return {}
    return {strategy: detail for strategy, (_t, detail) in _cache.items()}
