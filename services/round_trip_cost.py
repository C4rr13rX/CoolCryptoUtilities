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

import json
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


#: PER-TOKEN full round trips (both legs, gas + slippage), measured on chain.
#:
#: This is the ONLY per-token cost table in the repo, and it lives here so a
#: census and the production gate cannot disagree. ``round_trip_cost()`` below
#: takes NO symbol -- it is a single book-wide scalar over all receipts -- and
#: ``services/roundtrip_cost.roundtrip_cost_rate()`` (one underscore apart,
#: different module) takes only a notional. Neither can tell WETH from KTA,
#: and on this venue that is the whole spread: 0.11% against 74.98%.
#:
#: READ THE COVERAGE BEFORE QUOTING A ROW. Six tokens here are MEASURED.
#: Everything else falls through to ``DEFAULT_TOKEN_ROUND_TRIP``, which is an
#: ASSUMPTION, not a measurement -- so an unlisted token is UNPRICED, and
#: ``token_round_trip_cost`` deliberately returns the pessimistic default for
#: it rather than a cheap guess. A caller that refuses above a threshold
#: therefore refuses the unmeasured by default, which is the intended
#: direction: understating cost is the direction that loses money.
TOKEN_ROUND_TRIP = {
    "WETH": 0.0011,
    "ETH": 0.0011,
    "CBETH": 0.0030,
    "USDC": 0.0011,
    "AERO": 0.0435,
    "KTA": 0.7498,
}

#: What an UNLISTED (unmeasured) token costs. Read at call time, like
#: everything else in this module.
DEFAULT_TOKEN_ROUND_TRIP = float(os.getenv("TOKEN_ROUND_TRIP_DEFAULT", "0.0070"))


def default_token_round_trip() -> float:
    """The unmeasured-token cost, read from the environment at call time."""
    return float(os.getenv("TOKEN_ROUND_TRIP_DEFAULT", str(DEFAULT_TOKEN_ROUND_TRIP)))


def token_round_trip_cost(symbol: str) -> float:
    """Full round-trip cost for ``symbol``'s BASE token, as a fraction.

    ``symbol`` is a pair like ``WETH-USDC``; the base is what sets the cost,
    because the quote leg on this venue is the stable side. Unlisted tokens
    get ``default_token_round_trip()`` -- an assumption, and by construction a
    pessimistic one. Callers that need to know the difference should check
    membership in ``TOKEN_ROUND_TRIP`` themselves rather than inferring it
    from the returned number.
    """
    base = str(symbol or "").split("-")[0].strip().upper()
    if base in TOKEN_ROUND_TRIP:
        return float(TOKEN_ROUND_TRIP[base])
    return default_token_round_trip()

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


def _provenance(details: Optional[str]) -> str:
    """``'live'`` or ``'ghost'`` for one ``trade_outcomes`` row.

    A row is live only if it SAYS it is. ``details.mode`` is the field the
    executor writes, and anything unreadable or absent is counted ghost --
    a row that cannot prove it was live must never inflate the live count,
    because the whole point of the count is to say how much of the cost
    floor rests on fills that actually paid gas.
    """
    if not details:
        return "ghost"
    try:
        blob = json.loads(details)
    except (TypeError, ValueError):
        return "ghost"
    if not isinstance(blob, dict):
        return "ghost"
    return "live" if str(blob.get("mode", "")).lower().startswith("live") else "ghost"


def fee_ratio_samples(limit: int = 0, *, db_path: Optional[Path] = None,
                      fallback: Optional[float] = None
                      ) -> List[Tuple[float, str]]:
    """``(ratio, 'live'|'ghost')`` for every fee that is actual evidence.

    Oldest first. Excludes the echo of the constant, rows below
    ``MIN_NOTIONAL``, rows with no fee, and absurd ratios. Returns ``[]``
    rather than raising when the book is missing -- a cost read must never
    be able to take the money path down with it.

    THE PROVENANCE IS THE POINT. Measured 2026-09-17 over the 123 real fees
    in the book: 105 ghost, 18 live -- but the WINDOW that actually sets the
    charged cost, the last 20, was 20 ghost and 0 live. Ghost rows are
    written ~10x more often than live ones, so recency alone buries every
    live fee below the window, and the cost floor that gates entries had
    zero live evidence in it while 18 live measurements sat in the book.

    WHAT THAT IS NOT: it is not currently costing money, and the first
    version of this docstring claimed it was. Measured under each candidate
    rule, every live-inclusive one charges the SAME number -- last 20 by
    recency 0.4613%, last 10 live + last 10 ghost 0.4613%, last 40 by
    recency 0.4652%. Only "all 123 rows" moves it (1.2259%) and that is the
    pre-gas-tip-fix regime ``WINDOW`` exists to exclude. The two populations
    are statistically indistinguishable: Mann-Whitney U=719.0, p=0.1065 over
    n=18 live against 105 ghost. A p75 over 18 rows is one or two rows.

    So the count is here for the DIVERGENCE case, not for today's number:
    the exclusion is structural, so on the day live fees do part company
    with ghost fees, the charged cost will not notice and nobody will be
    told. A caller that cannot see the split cannot see that either.
    """
    path = db_path or _db_path()
    if not Path(path).exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True,
                               timeout=5.0)
    except sqlite3.Error:
        return []
    # A BOOK WITHOUT ``details`` MUST STILL PRICE. Asking for a column that
    # is not there raises, and the except returns [] -- which does not read
    # as "no provenance", it reads as "no fees in the book" and silently
    # drops every gate onto the fallback constant. The provenance is the new
    # thing here; the cost is the old thing, and the old thing must not be
    # able to fail because the new thing is unavailable.
    has_details = True
    try:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT ts, entry_price, quantity, fee_cost, details FROM "
                "trade_outcomes ORDER BY ts"
            ).fetchall()
        except sqlite3.Error:
            has_details = False
            rows = conn.execute(
                "SELECT ts, entry_price, quantity, fee_cost FROM "
                "trade_outcomes ORDER BY ts"
            ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    samples: List[Tuple[float, str]] = []
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
        samples.append(
            (ratio, _provenance(row["details"]) if has_details else "ghost"))
    return samples[-limit:] if limit and limit > 0 else samples


def real_fee_ratios(limit: int = 0, *, db_path: Optional[Path] = None,
                    fallback: Optional[float] = None) -> List[float]:
    """Fee-to-notional ratios that are actual evidence, oldest first.

    The ratios alone, for callers that do not care where they came from.
    ``fee_ratio_samples`` is the same population carrying its provenance;
    use that one if you are about to charge somebody the answer.
    """
    return [ratio for ratio, _ in
            fee_ratio_samples(limit, db_path=db_path, fallback=fallback)]


def provenance_counts(samples: List[Tuple[float, str]]) -> Tuple[int, int]:
    """``(live, ghost)`` over ``fee_ratio_samples`` output."""
    live = sum(1 for _, kind in samples if kind == "live")
    return live, len(samples) - live


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
    samples = fee_ratio_samples(WINDOW, db_path=db_path)
    ratios = [ratio for ratio, _ in samples]
    live, ghost = provenance_counts(samples)
    # THE COUNT TRAVELS WITH THE COST, on every path including the two
    # fallbacks. A cost with no provenance is how this window came to be
    # 20 ghost rows and 0 live ones without anybody noticing.
    provenance = f"[{live} live / {ghost} ghost]"
    if len(ratios) < MIN_SAMPLES:
        return fallback, (
            f"only {len(ratios)} real fees in the book (need {MIN_SAMPLES}) "
            f"{provenance}; using the {fallback:.4%} fallback")
    measured = _quantile(ratios, QUANTILE)
    if not (FLOOR <= measured <= CEILING):
        return fallback, (
            f"measured {measured:.4%} is outside [{FLOOR:.4%}, {CEILING:.4%}] "
            f"{provenance}; the book is malformed, using the "
            f"{fallback:.4%} fallback")
    return measured, (
        f"p{QUANTILE * 100:.0f} of the last {len(ratios)} real fees = "
        f"{measured:.4%} {provenance} (median "
        f"{statistics.median(ratios):.4%}, fallback {fallback:.4%})")


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
    samples = fee_ratio_samples()
    ratios = [r for r, _ in samples]
    live, ghost = provenance_counts(samples)
    cost, reason = measure()
    print(f"real fees in the book : {len(ratios)}  ({live} live / {ghost} ghost)")
    win_live, win_ghost = provenance_counts(samples[-WINDOW:])
    print(f"  the window that sets the cost: {win_live} live / {win_ghost} ghost")
    for kind in ("live", "ghost"):
        only = [r for r, k in samples if k == kind]
        if only:
            print(f"  {kind:<5s} median {statistics.median(only):.4%}  "
                  f"p75 {_quantile(only, 0.75):.4%}  n={len(only)}")
    if ratios:
        print(f"  median              : {statistics.median(ratios):.4%}")
        print(f"  p75 of last {WINDOW:<3d}     : "
              f"{_quantile(ratios[-WINDOW:], 0.75):.4%}")
    print(f"round trip charged    : {cost:.4%}")
    print(f"  because             : {reason}")
    print(f"  fallback constant   : {fallback_cost():.4%}")
