"""What a strategy has actually delivered, for the entry viability gate.

THE DEFECT THIS EXISTS TO FIX
-----------------------------
``bot.py`` sized the entry gate's expected gross return off the directive's
own ``target_price``::

    gross_return = max(0.0, margin)
    if directive is not None and price > 0.0 and directive.target_price > price:
        gross_return = (float(directive.target_price) - price) / price

``atf_static`` builds its target as ``price * 1.05``. So the gate asked "is 5%
more than the cost?", answered yes, and approved the trade -- every time, for
every symbol, in every market. Worse, the target OVERRODE ``margin``, the
brain's actual per-symbol estimate, so the more optimistic a strategy's
advertisement the more certainly it passed. A strategy could not fail this
gate by being wrong; only by advertising less.

Measured 2026-09-05 over all 20 live entries ever taken (``trading_ops``,
status ``live-entry``, ``micro_profit.gross_profit_usd / notional_usd``):

    credited by the gate   +5.00% on 14 of 20, +5.0% to +6.1% on the rest
    actually delivered     median -0.25% gross, net win rate 27.8%
                           (18 settled round trips, ``trade_outcomes``)

Not one of those 20 numbers was a measurement. The book they produced is
-0.186371 net.

The same shape reaches further than the constant. ``target_price`` is derived
from the live price, so a corrupt price makes a spectacular claim: of 1665
recent entry decisions, 11.4% claimed better than +100% and the largest
claimed +1.047e12 (CBETH-USDC quoted at 2.74e-09 against a correct target of
2869.7 -- a decimals shift, not an opportunity). A gate keyed to the claim is
therefore not merely blind, it is INVERTED: the worse the price data, the more
enthusiastically it approves.

WHAT REPLACES IT
----------------
The strategy's own realized gross return, trimmed, from closed round trips.
``estimate_gross_return`` is what the gate now asks.

Three choices worth stating, because each was measured rather than assumed:

1. **The 10% trimmed mean, not the mean or the median.** The mean is the right
   statistic for a repeated additive bet, but this feed puts a fat right tail
   into every sample and the raw mean inherits it. The median is robust and
   wrong -- it throws away the winners the strategy exists to catch, and it
   would have refused every live strategy and stopped trading. Measured on the
   500-row ghost book:

       strategy            n     mean    trim10   median   win%
       atf_static        289  +0.0135  +0.0065  +0.0004  52.9
       atf_static_scout  103  +0.0191  +0.0155  +0.0041  93.2
       rsi_reversal@5h    16  -0.1183  -0.1192  -0.1321  12.5
       obv_accum@1w       12  -0.0098  -0.0092  -0.0094  16.7
       bus_schedule        4  -0.0226  -0.0226  -0.0233   0.0

   Half of ``atf_static``'s raw mean is tail artifact (+1.35% -> +0.65%). The
   trimmed figure still clears the $1.50-clip hurdle of 0.589%, so the two
   strategies that actually trade live keep trading, while every strategy with
   a losing record is refused. A gate that refuses everything is the same as
   being switched off; this one discriminates.

2. **Observations beyond +/-100% are dropped from the sample, not clamped.**
   A round trip that books 1e12 is a feed artifact and averaging it in -- even
   trimmed -- lets one bad tick set a strategy's reputation. (Zero of the
   current 500 ghost exits trip this; it is the guard for the next one, since
   11.4% of ENTRY claims already do.)

3. **A strategy with no track record is credited with the house average, not
   with its claim.** Falling back to the claim would reinstate the tautology
   for every new strategy, which is where the losing ones start. The pooled
   trimmed mean across all strategies (+0.502%, n=500) is a real measurement
   of what this pipeline typically delivers, and it is always capped by the
   claim -- a strategy is never credited with more than it asked for.

THE GHOST LANE IS DELIBERATELY EXEMPT
-------------------------------------
Only entries that spend real money are gated on measured performance. A
simulated entry is how a strategy EARNS a track record, so gating it on one it
does not have yet is a closed loop with no entrance -- the shape that once
refused 385 of 385 ghost entries on the $0.02 dollar floor and starved every
graduation the pipeline depends on. The ghost book keeps using the claim and
keeps producing the evidence that this module reads.

Units: every value in and out of this module is a GROSS price return, a
dimensionless fraction of notional, before costs. That matches the ghost
book's ``profit`` field, verified on BPAD-USDC 2026-09-05 11:02:37 --
``profit`` 0.05425716 against ``(exit - entry) / entry`` 0.05425716 -- and
matches ``evaluate_micro_profit``, which multiplies it by notional and then
subtracts costs itself. It is NOT net of fees; do not compare it to a net
figure.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Round trips beyond this magnitude are feed artifacts, not results. See
#: choice 2 above.
MAX_PLAUSIBLE_RETURN = 1.0

#: Below this many closed round trips a strategy has no track record and is
#: credited with the pooled house average instead. 20 is the smallest sample
#: whose 10% trim removes anything at all.
MIN_SAMPLES = 20

#: How much of each tail the trimmed mean discards.
TRIM_FRACTION = 0.10

#: Ghost exits to read. The whole book is 500 rows today; this is sized to
#: take all of it and still be one indexed query (idx_trading_ops_status_ts).
_GHOST_SAMPLE_LIMIT = 4000
_LIVE_SAMPLE_LIMIT = 400

#: The book moves slowly and the entry path asks per tick, so the sample is
#: cached exactly like the gas reading in ``micro_profit``.
_CACHE: Dict[str, Tuple[float, Dict[str, List[float]]]] = {}
_CACHE_TTL_SEC = 300.0


@dataclass(frozen=True)
class EdgeEstimate:
    """The number the gate should use, and the provenance to justify it."""

    value: float
    source: str
    samples: int
    claimed: float
    strategy_id: str

    def to_dict(self) -> dict:
        return asdict(self)


def _clear_cache() -> None:
    """Test hook. Production relies on the TTL."""
    _CACHE.clear()


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def trimmed_mean(values: Sequence[float], fraction: float = TRIM_FRACTION) -> float:
    """Mean with ``fraction`` of each tail discarded.

    Returns 0.0 for an empty sample: no evidence is not evidence of an edge.
    """
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    cut = int(len(ordered) * max(0.0, min(0.49, float(fraction))))
    core = ordered[cut: len(ordered) - cut] if len(ordered) - 2 * cut > 0 else ordered
    return sum(core) / float(len(core))


def _plausible(value: Optional[float]) -> bool:
    return value is not None and abs(value) <= MAX_PLAUSIBLE_RETURN


def _collect(db: Any) -> Dict[str, List[float]]:
    """strategy_id -> realized gross returns, from the ghost book and the live book.

    Never raises. An estimate that cannot be computed must degrade to "no
    track record", which the caller answers with the house average -- not to
    an exception on the money path.
    """
    by_strategy: Dict[str, List[float]] = {}

    def _add(strategy_id: str, value: Optional[float]) -> None:
        if not strategy_id or not _plausible(value):
            return
        by_strategy.setdefault(strategy_id, []).append(float(value))

    # Ghost book: `profit` is already the gross price return.
    try:
        rows = db.fetch_trades(limit=_GHOST_SAMPLE_LIMIT, statuses=["ghost-exit"])
    except Exception:  # noqa: BLE001 - see docstring
        rows = []
    for row in rows or []:
        details = row.get("details") if isinstance(row, dict) else None
        if not isinstance(details, dict):
            try:
                details = json.loads(details or "{}")
            except Exception:  # noqa: BLE001
                continue
        _add(str(details.get("strategy_id") or ""), _finite(details.get("profit")))

    # Live book: the gross return is recomputed from the fills rather than
    # read from `gross_profit`, which is dollars and would silently mix a
    # dollar amount into a table of rates -- the exact boundary error this
    # repo has shipped before.
    try:
        outcomes = db.fetch_trade_outcomes(wallet="live", limit=_LIVE_SAMPLE_LIMIT)
    except Exception:  # noqa: BLE001
        outcomes = []
    for row in outcomes or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").lower() != "closed":
            continue
        details = row.get("details")
        if not isinstance(details, dict):
            try:
                details = json.loads(details or "{}")
            except Exception:  # noqa: BLE001
                details = {}
        entry = _finite(row.get("entry_price")) or 0.0
        exit_price = _finite(row.get("exit_price"))
        if entry <= 0.0 or exit_price is None:
            continue
        _add(str(details.get("strategy_id") or ""), (exit_price - entry) / entry)

    return by_strategy


def _sample(db: Any, *, now: Optional[float] = None) -> Dict[str, List[float]]:
    stamp = float(now if now is not None else time.time())
    cached = _CACHE.get("book")
    if cached and (stamp - cached[0]) < _CACHE_TTL_SEC:
        return cached[1]
    book = _collect(db)
    _CACHE["book"] = (stamp, book)
    return book


def _min_samples() -> int:
    try:
        return max(1, int(os.getenv("EDGE_MIN_SAMPLES", str(MIN_SAMPLES))))
    except (TypeError, ValueError):
        return MIN_SAMPLES


def estimate_gross_return(
    db: Any,
    strategy_id: Any,
    *,
    claimed_return: float,
    now: Optional[float] = None,
) -> EdgeEstimate:
    """What this strategy has actually delivered, capped by what it claims.

    ``claimed_return`` is the directive's own advertisement -- the number the
    gate used to trust outright. It survives only as a CEILING: a strategy is
    never credited with more than it asked for, so this can refuse an entry
    the old gate allowed but can never allow one it refused.

    Returns a gross price return (dimensionless, before costs), suitable for
    ``evaluate_micro_profit(gross_return=...)``.
    """
    claim = _finite(claimed_return)
    if claim is None:
        claim = 0.0
    sid = str(strategy_id or "").strip()

    book = _sample(db, now=now)
    observations = book.get(sid) or []
    floor_samples = _min_samples()

    if len(observations) >= floor_samples:
        value = trimmed_mean(observations)
        source = "measured"
        samples = len(observations)
    else:
        pooled = [v for values in book.values() for v in values]
        if len(pooled) >= floor_samples:
            value = trimmed_mean(pooled)
            source = "pooled"
            samples = len(pooled)
        else:
            # Nothing has closed anywhere yet. Fall back to the claim so a
            # cold database cannot switch the pipeline off; the cap below is
            # a no-op in this branch and the reason says so.
            value = claim
            source = "unmeasured"
            samples = len(observations)

    value = min(float(value), float(claim))
    return EdgeEstimate(
        value=value,
        source=source,
        samples=int(samples),
        claimed=float(claim),
        strategy_id=sid,
    )
