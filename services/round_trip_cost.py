"""What one round trip actually costs, read from the book instead of assumed.

``symbol_edge_gate.ROUND_TRIP_COST`` has been the literal 0.0065 since it was
written, documented as "the measured median of ``fee_cost / notional`` over
the 143 closed round trips on 2026-09-04". Measured 2026-09-07 over the 196
rows now in ``trade_outcomes``:

    105 of them have fee_cost / notional EXACTLY 0.650000%, min == max,
    zero variance.

Those rows are the constant being written back into the book. The "measured
median" was measuring its own default -- the same shape as every other
self-confirming number this repo has shipped, and invisible because a median
over a population that is half constant still looks like a statistic.

With the echo rows and the sub-``MIN_NOTIONAL`` rows removed, 82 real fees
remain, and they have MOVED:

    2026-09-03 / 09-04 real median   1.2259%   (pre gas-tip fix)
    last 40 real                     0.4667%
    last 20 real            median   0.4621%   p75 0.4738%   p90 0.4971%

So the constant was never conservative in the useful direction. It was 47%
too LOW during the era it claims to have measured, and it is 37% too HIGH
today. Both directions cost money: too low passes losing symbols, too high
bans symbols that genuinely pay, which is a graduation blocker wearing a
safety margin's clothes.

Three rules this module follows, because getting a cost wrong is how this
system has lost money before:

  * **The echo is excluded.** A row whose fee equals the configured fallback
    to the last decimal is treated as unevidenced, not as evidence.
  * **p75, not the median.** The next trade's gas is not the typical one, and
    understating cost is the dangerous direction. A conservative quantile of
    a recent window beats a precise average of a stale one.
  * **Thin evidence falls back to the constant.** Fewer than
    ``MIN_SAMPLES`` real fees is not a measurement, and inventing one from
    three rows is how a 1.2e-14 notional became a -54% return.

Units, stated once: everything here is a FRACTION of notional (0.0065 ==
0.65%), never basis points, never a percentage number, never dollars.
"""
from __future__ import annotations

import os
import sqlite3
import statistics
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

#: The value used when the book cannot evidence a better one. This is also
#: the value whose echo is filtered out of the book, so the two must move
#: together -- see ``_is_echo``.
#:
#: Read through ``fallback_cost()``, never off this module attribute: freezing
#: an env-configurable number at import time means a caller that sets
#: ``SYMBOL_EDGE_ROUND_TRIP_COST`` and reloads ITS module still gets whatever
#: this module saw first. That is not hypothetical -- it broke
#: ``test_a_return_exactly_at_cost_is_not_a_win`` the moment the gate started
#: asking here instead of reading a literal.
FALLBACK_COST = float(os.getenv("SYMBOL_EDGE_ROUND_TRIP_COST", "0.0065"))


def fallback_cost() -> float:
    """The fallback, read from the environment at call time."""
    return float(os.getenv("SYMBOL_EDGE_ROUND_TRIP_COST", str(FALLBACK_COST)))

#: Smallest notional whose fee ratio is meaningful. Below this the division
#: turns rounding into a double-digit "cost" -- the book holds a row at
#: 54237% for exactly this reason.
MIN_NOTIONAL = float(os.getenv("SYMBOL_EDGE_MIN_NOTIONAL", "0.05"))

#: How many recent real fees to read. Long enough to survive one odd block,
#: short enough that a gas regime from four days ago cannot set today's bar.
WINDOW = int(os.getenv("ROUND_TRIP_COST_WINDOW", "20"))

#: Below this many real fees there is no measurement, only a guess.
MIN_SAMPLES = int(os.getenv("ROUND_TRIP_COST_MIN_SAMPLES", "8"))

#: Which quantile of the window to charge. 0.75 is deliberately above the
#: middle: understating cost is the direction that loses money.
QUANTILE = float(os.getenv("ROUND_TRIP_COST_QUANTILE", "0.75"))

#: Hard bounds. A measured cost outside these is not believed -- it means the
#: book is malformed, and a malformed book must not be able to switch the
#: profitability bar off (low) or ban the whole universe (high).
FLOOR = float(os.getenv("ROUND_TRIP_COST_FLOOR", "0.0005"))
CEILING = float(os.getenv("ROUND_TRIP_COST_CEILING", "0.05"))

#: A fee ratio above this is a broken row, not an expensive trade.
ABSURD = float(os.getenv("ROUND_TRIP_COST_ABSURD", "0.20"))

CACHE_SEC = float(os.getenv("ROUND_TRIP_COST_CACHE_SEC", "300"))

_lock = threading.Lock()
#: ``{resolved book path: (expires_at, cost, reason)}``. Keyed on the BOOK,
#: because a caller pointed at a throwaway book must not be served the
#: production book's answer, nor overwrite it.
_cached: dict = {}


def _db_path() -> Path:
    configured = os.getenv("TRADING_CACHE_DB")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "storage" / "trading_cache.db"


def _is_echo(ratio: float, fallback: float = None) -> bool:
    """Is this row the constant written back rather than a fee that was paid?

    Exact to nine decimals: a real fee that happens to land within a
    nanofraction of the constant is indistinguishable from the echo, and
    dropping one true row costs nothing next to trusting 105 false ones.
    """
    fallback = fallback_cost() if fallback is None else fallback
    return abs(ratio - abs(fallback)) < 1e-9


def real_fee_ratios(limit: int = 0, *, db_path: Optional[Path] = None,
                    fallback: Optional[float] = None) -> List[float]:
    """Fee-to-notional ratios that are actual evidence, oldest first.

    Excludes the echo of the constant, rows below ``MIN_NOTIONAL``, rows with
    no fee, and absurd ratios. Returns ``[]`` rather than raising when the
    book is missing -- a cost read must never be able to take the money path
    down with it.
    """
    path = db_path or _db_path()
    if not Path(path).exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True,
                               timeout=5.0)
    except sqlite3.Error:
        return []
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT ts, entry_price, quantity, fee_cost FROM trade_outcomes "
            "ORDER BY ts"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    ratios: List[float] = []
    for row in rows:
        try:
            notional = abs(float(row["entry_price"]) * float(row["quantity"]))
            fee = abs(float(row["fee_cost"]))
        except (TypeError, ValueError):
            continue
        if notional < MIN_NOTIONAL or fee <= 0:
            continue
        ratio = fee / notional
        if ratio > ABSURD or _is_echo(ratio, fallback):
            continue
        ratios.append(ratio)
    return ratios[-limit:] if limit and limit > 0 else ratios


def _quantile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered))))
    return ordered[index]


def measure(*, db_path: Optional[Path] = None) -> Tuple[float, str]:
    """``(cost, reason)`` -- the honest round trip and how it was arrived at.

    The reason is always populated, on the fallback path as much as on the
    measured one, so a gate's refusal can be traced to a number and the
    number to a population.
    """
    fallback = fallback_cost()
    ratios = real_fee_ratios(WINDOW, db_path=db_path)
    if len(ratios) < MIN_SAMPLES:
        return fallback, (
            f"only {len(ratios)} real fees in the book (need {MIN_SAMPLES}); "
            f"using the {fallback:.4%} fallback")
    measured = _quantile(ratios, QUANTILE)
    if not (FLOOR <= measured <= CEILING):
        return fallback, (
            f"measured {measured:.4%} is outside [{FLOOR:.4%}, {CEILING:.4%}]; "
            f"the book is malformed, using the {fallback:.4%} fallback")
    return measured, (
        f"p{QUANTILE * 100:.0f} of the last {len(ratios)} real fees = "
        f"{measured:.4%} (median {statistics.median(ratios):.4%}, "
        f"fallback {fallback:.4%})")


def round_trip_cost(*, db_path: Optional[Path] = None) -> float:
    """The cost every profitability bar should be set against. Cached."""
    key = str(Path(db_path or _db_path()).as_posix())
    now = time.time()
    with _lock:
        hit = _cached.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
    cost, reason = measure(db_path=db_path)
    with _lock:
        _cached[key] = (now + CACHE_SEC, cost, reason)
    return cost


def explain(*, db_path: Optional[Path] = None) -> str:
    return measure(db_path=db_path)[1]


def reset_cache() -> None:
    with _lock:
        _cached.clear()


if __name__ == "__main__":  # pragma: no cover
    ratios = real_fee_ratios()
    cost, reason = measure()
    print(f"real fees in the book : {len(ratios)}")
    if ratios:
        print(f"  median              : {statistics.median(ratios):.4%}")
        print(f"  p75 of last {WINDOW:<3d}     : "
              f"{_quantile(ratios[-WINDOW:], 0.75):.4%}")
    print(f"round trip charged    : {cost:.4%}")
    print(f"  because             : {reason}")
    print(f"  fallback constant   : {fallback_cost():.4%}")
