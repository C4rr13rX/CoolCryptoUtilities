"""Where does the decision budget actually go, per strategy per hour?

THE QUESTION THIS ANSWERS, AND THE PREMISE IT OVERTURNS. The operator's
2026-09-10 brief said the population is starved because ``atf_static`` takes
70% of the evidence, and quoted "decision cycles, last 6h from trading_ops:
atf_static 179, everything else 7 between them". That number is from the wrong
table and it names the wrong bug.

``trading_ops`` is an append-only op LOG, not a cycle table. ``trading/bot.py``
writes a row only when ``action != "hold"``, so a refused cycle leaves no row,
and one tick can emit four rows (measured in ``hold_attribution_census.py``:
96 ban rows against 34 ticks, 23 clusters of size four). The ``atf_static``
rows are ``evaluate_atf_static_entry`` BUS ACTIONS published by the c0d3rv2
strategy publisher -- a different channel from the symbol scheduler -- so
counting them beside scheduler strategies compares a publish rate to a
decision rate.

``organism_snapshots`` is the cycle table. One row is one ``evaluate()`` that
reached a decision, and the snapshot's ``scheduler`` list names, per symbol,
the ``last_directive.strategy_id`` that owns that symbol's slot. Attributing
each cycle to the strategy holding its symbol is therefore the honest
per-strategy budget.

WHAT IT MEASURES, 6h to 2026-09-10T09:45 (this script, ``--hours 6``):

    strategy                 cycles   cyc/h  symbols   actions
    ema_cross                   290    48.3        4   all hold
    <no-directive>              246    41.0        6   all hold
    stochastic_reversal@1w      229    38.2        1   all hold
    rsi_reversal@5h             219    36.5        2   all hold
    tf_forecast                 214    35.7        1   all hold
    rsi_reversal@1d             156    26.0        1   all hold
    ema_cross@1d                151    25.2        1   all hold
    supertrend_follow@1d         63    10.5        1   all hold
    donchian_breakout             5     0.8        1   4 hold, 1 exit
    bollinger_squeeze             1     0.2        1   1 enter

THREE FINDINGS, AND EACH CONTRADICTS A SENTENCE OF THE BRIEF.

1. ``atf_static`` RECEIVES ZERO DECISION CYCLES. It does not appear. It holds
   no symbol slot and never did in the window. It is not over-allocated; on
   this channel it is not allocated at all.

2. THE STARVATION IS SYMBOL-SLOT CONTENTION WITH NEAR-ZERO TURNOVER, which is
   the first of the three mechanisms the brief asked us to distinguish. There
   were NINE symbol slots in six hours and one directive per symbol, so at
   most nine strategies can be live at once; nine distinct strategies held a
   slot all window. Five of the nine symbols never changed strategy at all,
   and the other four changed exactly once. TWENTY-NINE OF 38 STRATEGIES GOT
   EXACTLY ZERO CYCLES -- not few, zero. No per-strategy fix can matter to
   them, and giving every strategy "a floor of cycles" is impossible while a
   symbol carries one directive: the floor has to come from slot ROTATION or
   from more symbols, not from a scheduler weight.

3. IT IS NOT "33 STRATEGIES REFUSING OFFERED CYCLES", the third mechanism --
   but refusal is still where the budget dies, and it is not per-strategy.
   1573 of 1574 cycles decided ``hold``: ONE enter in six hours across every
   strategy that had a slot. A shared downstream gate refuses ~100% of cycles
   regardless of who proposed them (``hold_attribution_census.py`` names it:
   ``net_margin`` max is negative across every symbol for ten hours, so the
   entry conjunct is unsatisfiable by measurement). Redistributing cycles
   between strategies redistributes holds.

So the ranking the operator wants is blocked by two separate walls, and the
allocation one is NOT the one the brief named: 29 strategies cannot be ranked
because they never hold a slot, and the 9 that do cannot be ranked because
their shared entry gate is shut.

Read-only. Touches no trading code and writes nothing.

Usage:
    python -X utf8 scripts/decision_budget_census.py --hours 6
    python -X utf8 scripts/decision_budget_census.py --hours 24 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_DB = os.path.join("storage", "trading_cache.db")

#: A symbol slot with no directive yet. Kept as its own bucket rather than
#: dropped: cycles spent on a symbol nobody has claimed are budget too, and
#: folding them into a strategy would credit it with work it did not do.
NO_DIRECTIVE = "<no-directive>"


def slot_owners(snapshot: Dict[str, Any]) -> Dict[str, str]:
    """Map ``SYMBOL -> strategy_id`` from one snapshot's scheduler list.

    The scheduler carries one ``last_directive`` per symbol, so a symbol's
    slot has exactly one owner at a time -- which is the whole reason the
    budget concentrates.
    """
    owners: Dict[str, str] = {}
    for entry in snapshot.get("scheduler") or []:
        if not isinstance(entry, dict):
            continue
        symbol = str(entry.get("symbol") or "").upper()
        if not symbol:
            continue
        directive = entry.get("last_directive")
        strategy = ""
        if isinstance(directive, dict):
            strategy = str(directive.get("strategy_id") or "")
        owners[symbol] = strategy or NO_DIRECTIVE
    return owners


def census(rows: Iterable[Tuple[float, str]], *, hours: float) -> Dict[str, Any]:
    """Attribute every decision cycle to the strategy holding its symbol."""
    cycles: Counter = Counter()
    actions: Dict[str, Counter] = defaultdict(Counter)
    symbols_touched: Dict[str, set] = defaultdict(set)
    # Per symbol, the ordered sequence of DISTINCT consecutive owners. Its
    # length minus one is how many times that slot changed hands.
    occupancy: Dict[str, List[str]] = defaultdict(list)
    total = 0

    for _ts, payload in rows:
        try:
            snapshot = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
        except Exception:  # noqa: BLE001 - an unparseable snapshot is not evidence
            continue
        if not isinstance(snapshot, dict):
            continue

        owners = slot_owners(snapshot)
        for symbol, owner in owners.items():
            if not occupancy[symbol] or occupancy[symbol][-1] != owner:
                occupancy[symbol].append(owner)

        decision = snapshot.get("decision")
        if not isinstance(decision, dict):
            continue
        symbol = str(decision.get("symbol") or "").upper()
        if not symbol:
            continue
        total += 1
        owner = owners.get(symbol, NO_DIRECTIVE)
        cycles[owner] += 1
        actions[owner][str(decision.get("action") or "")] += 1
        symbols_touched[owner].add(symbol)

    span = max(1e-9, float(hours))
    per_strategy = [
        {
            "strategy": name,
            "cycles": count,
            "cycles_per_hour": round(count / span, 2),
            "symbols": len(symbols_touched[name]),
            "actions": dict(actions[name]),
            "entries": actions[name].get("enter", 0),
        }
        for name, count in cycles.most_common()
    ]
    holds = sum(counter.get("hold", 0) for counter in actions.values())
    slots = {
        symbol: {
            "changes": len(sequence) - 1,
            "distinct_owners": len(set(sequence)),
            "owners": sequence,
        }
        for symbol, sequence in sorted(occupancy.items())
    }
    holders = {name for name in cycles if name != NO_DIRECTIVE}
    return {
        "hours": hours,
        "cycles": total,
        "holds": holds,
        "entries": sum(row["entries"] for row in per_strategy),
        # A hold share at 1.0 means no redistribution of cycles between
        # strategies can produce an entry: the refusal is downstream of who
        # proposed. Reported beside the distribution for exactly that reason.
        "hold_share": (holds / total) if total else 0.0,
        "per_strategy": per_strategy,
        "slots": slots,
        "slot_count": len(slots),
        "strategies_holding_a_slot": len(holders),
        "slot_changes": sum(row["changes"] for row in slots.values()),
    }


def registry_population(path: str = os.path.join("data", "strategy_registry.json")) -> int:
    """How many strategies exist, so 'got zero cycles' has a denominator."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return 0
    if isinstance(data, dict):
        entries = data.get("strategies", data)
        if isinstance(entries, dict):
            return len(entries)
        if isinstance(entries, list):
            return len(entries)
    if isinstance(data, list):
        return len(data)
    return 0


def op_log_multiplicity(
    conn: sqlite3.Connection, *, now: float, hours: float
) -> Dict[str, Any]:
    """The ``trading_ops`` view of the same window, to show why it misleads.

    Rows here are op-log entries, not cycles: several land on one tick and a
    hold writes none. Reported so the next reader does not re-derive the
    per-strategy budget from this table.
    """
    try:
        rows = conn.execute(
            "select status, count(*) from trading_ops where ts>=? group by status "
            "order by count(*) desc limit 8",
            (now - hours * 3600.0,),
        ).fetchall()
        total = conn.execute(
            "select count(*) from trading_ops where ts>=?", (now - hours * 3600.0,)
        ).fetchone()[0]
    except sqlite3.Error:
        return {"rows": 0, "by_status": {}}
    return {"rows": int(total), "by_status": {str(s): int(n) for s, n in rows}}


def _load(db_path: str, hours: float, now: float) -> List[Tuple[float, str]]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "select ts, payload from organism_snapshots where ts>=? order by ts",
            (now - hours * 3600.0,),
        ).fetchall()
    finally:
        conn.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"no database at {args.db}")
        return 2

    now = time.time()
    report = census(_load(args.db, args.hours, now), hours=args.hours)
    population = registry_population()
    report["registry_strategies"] = population
    report["strategies_with_zero_cycles"] = max(
        0, population - report["strategies_holding_a_slot"]
    )
    conn = sqlite3.connect(args.db)
    try:
        report["trading_ops"] = op_log_multiplicity(conn, now=now, hours=args.hours)
    finally:
        conn.close()

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"DECISION BUDGET, last {args.hours:g}h -- one row of organism_snapshots is one cycle")
    print(
        f"  cycles {report['cycles']}   holds {report['holds']} "
        f"({report['hold_share'] * 100:.1f}%)   entries {report['entries']}"
    )
    print()
    print(f"  {'strategy':32s} {'cycles':>7s} {'cyc/h':>7s} {'syms':>5s}  actions")
    for row in report["per_strategy"]:
        print(
            f"  {row['strategy']:32s} {row['cycles']:7d} "
            f"{row['cycles_per_hour']:7.1f} {row['symbols']:5d}  {row['actions']}"
        )
    print()
    print(
        f"  SYMBOL SLOTS {report['slot_count']} "
        f"(one directive each) -- {report['slot_changes']} owner changes in {args.hours:g}h"
    )
    for symbol, slot in report["slots"].items():
        print(
            f"    {symbol:18s} changes={slot['changes']:3d} "
            f"distinct={slot['distinct_owners']:2d}  {slot['owners'][:5]}"
        )
    print()
    print(
        f"  STRATEGIES HOLDING A SLOT {report['strategies_holding_a_slot']} "
        f"of {population} in the registry -- "
        f"{report['strategies_with_zero_cycles']} got ZERO cycles"
    )
    print(
        f"  trading_ops over the same window: {report['trading_ops']['rows']} rows "
        f"-- op-log entries, NOT cycles: {report['trading_ops']['by_status']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
