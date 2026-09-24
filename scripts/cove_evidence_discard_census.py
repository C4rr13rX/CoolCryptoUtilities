"""Where does the ghost engine's closure rate go before it reaches the bar?

THE QUESTION
------------

The graduation banner reports a *tradeable evidence rate* of ~5.6/day, and the
ghost engine records ~25 ``ghost-exit`` rows a day (cross-verified 2026-09-23:
``trading_ops`` ``action='exit'`` and ``trade_outcomes`` agree at exactly 25 over
24h and exactly 2 over 6h). Four fifths of every round trip the engine closes is
discarded before it counts toward a licence, and nobody has counted which arm
does the discarding on a CURRENT window -- ``services.tradeable_evidence.report``
prints the drop counters but only over all recorded history, pooled with years of
pre-fix artifacts.

This script asks the windowed version: over the last N hours, for every recorded
ghost round trip, which is the FIRST predicate that refuses it?

WHY IT REUSES THE MODULE RATHER THAN REIMPLEMENTING
---------------------------------------------------

``services.tradeable_evidence`` exists precisely because a second copy of "is
this symbol tradeable" drifts from the one that decides. So the row source
(``iter_ghost_exits``) and both predicates (``_exceeds_evidence_horizon``,
``_live_tradeable``) are imported, and the filter ORDER here is copied from
``reconstruct`` so that "first arm to refuse" means the same thing in both.

WHAT THIS DOES NOT ESTABLISH
----------------------------

A concentrated discard arm is not by itself a bug. ``_live_tradeable`` refusing
a row is the system correctly saying real money could not have been placed on
that symbol -- that is the arm doing its job. This census says WHERE the rate
goes, not whether it should go there. The per-symbol breakdown is printed so the
next reader can judge that separately.

Usage:
    python scripts/cove_evidence_discard_census.py --hours 24
    python scripts/cove_evidence_discard_census.py --hours 168 --by-symbol
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


#: The filter order is `reconstruct`'s order, not an arbitrary one. A row is
#: attributed to the FIRST arm that refuses it, so moving these around would
#: move the counts without any predicate changing.
ARMS = ("no_profit_field", "implausible", "out_of_horizon", "untradeable", "COUNTED")


def census(hours: float, db_path: str | None = None):
    from services.tradeable_evidence import iter_ghost_exits, _bounds
    from trading.strategies.ledger import _exceeds_evidence_horizon, _live_tradeable

    abs_cap, horizon = _bounds()
    cutoff = time.time() - hours * 3600.0

    arm_counts: Counter[str] = Counter()
    by_symbol: dict[str, Counter[str]] = defaultdict(Counter)
    by_strategy: dict[str, Counter[str]] = defaultdict(Counter)
    counted_net = 0.0
    counted_wins = 0
    # `_live_tradeable` reaches into trading.pipeline per call. Same memo key as
    # the module it is borrowed from: (strategy, symbol), never symbol alone --
    # a symbol-keyed memo lets whichever strategy asked first decide for all.
    tradeable_cache: dict[tuple[str, str], bool] = {}

    for row in iter_ghost_exits(db_path):
        if row["ts"] < cutoff:
            continue
        sid = str(row["strategy_id"])
        symbol = str(row["symbol"] or "")

        arm = None
        profit = row["profit"]
        if profit is None:
            arm = "no_profit_field"
        else:
            try:
                profit = float(profit)
            except (TypeError, ValueError):
                arm = "no_profit_field"
        if arm is None and abs(profit) >= abs_cap:
            arm = "implausible"
        if arm is None and _exceeds_evidence_horizon(row["held_sec"]):
            arm = "out_of_horizon"
        if arm is None:
            key = (sid, symbol)
            if key not in tradeable_cache:
                try:
                    tradeable_cache[key] = bool(_live_tradeable(symbol, sid))
                except Exception:  # noqa: BLE001 - unknown is not tradeable
                    tradeable_cache[key] = False
            if not tradeable_cache[key]:
                arm = "untradeable"
        if arm is None:
            arm = "COUNTED"
            counted_net += profit
            if profit > 0:
                counted_wins += 1

        arm_counts[arm] += 1
        by_symbol[symbol][arm] += 1
        by_strategy[sid][arm] += 1

    return {
        "hours": hours,
        "abs_cap": abs_cap,
        "horizon_sec": horizon,
        "arms": arm_counts,
        "by_symbol": by_symbol,
        "by_strategy": by_strategy,
        "counted_net": counted_net,
        "counted_wins": counted_wins,
    }


def _table(title: str, rows: dict[str, Counter], total_key: str = "COUNTED") -> None:
    print(f"\n  {title}")
    header = f"    {'key':<22s}" + "".join(f"{a:>17s}" for a in ARMS) + f"{'total':>8s}"
    print(header)
    ranked = sorted(rows.items(), key=lambda kv: -sum(kv[1].values()))
    for key, counts in ranked:
        total = sum(counts.values())
        line = f"    {key[:22]:<22s}" + "".join(f"{counts[a]:>17d}" for a in ARMS)
        print(line + f"{total:>8d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--by-symbol", action="store_true")
    ap.add_argument("--by-strategy", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    res = census(args.hours, args.db)
    arms = res["arms"]
    total = sum(arms.values())

    print("=" * 74)
    print(f"  GHOST CLOSURES -> TRADEABLE EVIDENCE, last {args.hours:g}h")
    print("=" * 74)
    print(f"  implausibility cap  {res['abs_cap']}")
    print(f"  evidence horizon    {res['horizon_sec']:.0f}s")
    print(f"  ghost-exit rows     {total}")
    if not total:
        print("\n  NO ROWS IN WINDOW -- widen --hours. Reporting nothing rather than 0%.")
        return 0
    print("\n  first arm to refuse each round trip:")
    for arm in ARMS:
        n = arms[arm]
        print(f"    {arm:<20s} {n:>6d}   {100.0 * n / total:>5.1f}%")
    counted = arms["COUNTED"]
    print(
        f"\n  COUNTED toward the bar: {counted} of {total} "
        f"({100.0 * counted / total:.1f}%), net {res['counted_net']:+.4f}, "
        f"wins {res['counted_wins']}"
        + (f" ({100.0 * res['counted_wins'] / counted:.0f}%)" if counted else "")
    )
    print(f"  rate: {24.0 * counted / args.hours:.1f} tradeable/day "
          f"from {24.0 * total / args.hours:.1f} closures/day")

    if args.by_symbol:
        _table("by symbol", res["by_symbol"])
    if args.by_strategy:
        _table("by strategy", res["by_strategy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
