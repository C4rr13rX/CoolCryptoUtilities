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


def registry_names(
    path: str = os.path.join("data", "strategy_registry.json")
) -> List[str]:
    """Every strategy that EXISTS, so 'got zero cycles' has a denominator."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    entries: Any = data
    if isinstance(data, dict):
        entries = data.get("strategies", data)
    if isinstance(entries, dict):
        return sorted(str(name) for name in entries)
    if isinstance(entries, list):
        return sorted(
            str(item.get("strategy_id") or item.get("name") or item)
            if isinstance(item, dict)
            else str(item)
            for item in entries
        )
    return []


#: Keys under a ``trading_ops`` payload whose value is a LIST of per-strategy
#: records. One row can charge several strategies, so each is harvested.
_LIST_KEYS = ("dropped", "bus_actions", "candidates")


def candidate_channel(
    conn: sqlite3.Connection, *, now: float, hours: float
) -> Dict[str, Any]:
    """Which strategies were PROPOSED at all, on the op-log channel.

    The scheduler slot table is only one of two producers. The c0d3rv2
    publisher emits ``evaluate_atf_static_entry`` bus actions on its own
    channel, and refusal rows name strategies that never reach a slot. A
    strategy absent from BOTH was never proposed -- which is a different
    defect from losing a slot, and the one that turned out to be real.
    """
    appearances: Counter = Counter()
    try:
        rows = conn.execute(
            "select status, details from trading_ops where ts>=?",
            (now - hours * 3600.0,),
        ).fetchall()
    except sqlite3.Error:
        return {"strategies": [], "appearances": {}}

    for _status, raw in rows:
        try:
            details = json.loads(raw) if isinstance(raw, (str, bytes)) else (raw or {})
        except Exception:  # noqa: BLE001 - an unparseable payload is not evidence
            continue
        if not isinstance(details, dict):
            continue
        name = details.get("strategy_id")
        if name:
            appearances[str(name)] += 1
        for key in _LIST_KEYS:
            value = details.get(key)
            if not isinstance(value, list):
                continue
            for entry in value:
                if isinstance(entry, dict) and entry.get("strategy_id"):
                    appearances[str(entry["strategy_id"])] += 1

    return {
        "strategies": sorted(appearances),
        "appearances": dict(appearances.most_common()),
    }


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


def breadth_trend(
    conn: sqlite3.Connection, *, now: float, hours: float, buckets: int = 4
) -> List[Dict[str, Any]]:
    """Symbols and proposable strategies per bucket, oldest first.

    THE SELF-CHECK THAT CHANGED THIS SCRIPT'S CONCLUSION. Read over six hours
    alone, 32 of 42 strategies appear in neither producer and it looks like a
    structural exclusion. Read over 24, only 21 are missing and the strategy
    closest to the graduation bar is among those that DO appear. Nothing is
    permanently excluded; the proposable population is SHRINKING, and a single
    window cannot tell those apart.

    Measured 2026-09-10, 6h buckets over 24h (symbols / slot-holders /
    proposed / cycles / entries):

        -24h..-18h   20 / 16 / 22 / 1151 / 4
        -18h..-12h   18 /  6 /  9 / 1137 / 1
        -12h..-6h    11 /  7 / 12 / 1801 / 0
        -6h..-0h      8 /  9 / 13 / 1572 / 1

    Symbols fall 20 -> 8 and proposable strategies 22 -> 13 in one day, so
    breadth must be reported BESIDE any per-strategy count: a strategy count
    that moves while the symbol count moves under it has measured the feed,
    not the allocation.
    """
    span = max(1e-9, float(hours) / max(1, buckets))
    trend: List[Dict[str, Any]] = []
    for index in range(buckets, 0, -1):
        low, high = now - index * span * 3600.0, now - (index - 1) * span * 3600.0
        rows = conn.execute(
            "select ts, payload from organism_snapshots where ts>=? and ts<? order by ts",
            (low, high),
        ).fetchall()
        report = census(rows, hours=span)
        holders = {
            row["strategy"]
            for row in report["per_strategy"]
            if row["strategy"] != NO_DIRECTIVE
        }
        proposed = holders | set(
            _candidates_between(conn, low=low, high=high)
        )
        trend.append(
            {
                "from_hours_ago": round(index * span, 2),
                "to_hours_ago": round((index - 1) * span, 2),
                "symbols": report["slot_count"],
                "slot_holders": len(holders),
                "proposed": len(proposed),
                "cycles": report["cycles"],
                "entries": report["entries"],
            }
        )
    return trend


def _candidates_between(conn: sqlite3.Connection, *, low: float, high: float) -> set:
    """Strategy ids named by any op-log row in ``[low, high)``."""
    names: set = set()
    try:
        rows = conn.execute(
            "select details from trading_ops where ts>=? and ts<?", (low, high)
        ).fetchall()
    except sqlite3.Error:
        return names
    for (raw,) in rows:
        try:
            details = json.loads(raw) if isinstance(raw, (str, bytes)) else (raw or {})
        except Exception:  # noqa: BLE001 - an unparseable payload is not evidence
            continue
        if not isinstance(details, dict):
            continue
        if details.get("strategy_id"):
            names.add(str(details["strategy_id"]))
        for key in _LIST_KEYS:
            value = details.get(key)
            if not isinstance(value, list):
                continue
            for entry in value:
                if isinstance(entry, dict) and entry.get("strategy_id"):
                    names.add(str(entry["strategy_id"]))
    return names


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
    parser.add_argument(
        "--buckets",
        type=int,
        default=4,
        help="split the window into this many buckets for the breadth trend",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"no database at {args.db}")
        return 2

    now = time.time()
    report = census(_load(args.db, args.hours, now), hours=args.hours)
    registry = registry_names()
    report["registry_strategies"] = len(registry)
    report["strategies_with_zero_cycles"] = max(
        0, len(registry) - report["strategies_holding_a_slot"]
    )
    conn = sqlite3.connect(args.db)
    try:
        report["trading_ops"] = op_log_multiplicity(conn, now=now, hours=args.hours)
        report["candidate_channel"] = candidate_channel(
            conn, now=now, hours=args.hours
        )
        report["breadth_trend"] = breadth_trend(
            conn, now=now, hours=args.hours, buckets=args.buckets
        )
    finally:
        conn.close()

    slot_holders = {
        row["strategy"]
        for row in report["per_strategy"]
        if row["strategy"] != NO_DIRECTIVE
    }
    proposed = slot_holders | set(report["candidate_channel"]["strategies"])
    report["proposed_anywhere"] = sorted(proposed)
    # The finding. A strategy in neither channel did not lose a slot and did
    # not refuse a cycle -- nothing ever put it forward, so no per-strategy
    # fix and no fairer weighting can reach it.
    report["never_proposed"] = sorted(set(registry) - proposed)

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
        f"of {report['registry_strategies']} in the registry -- "
        f"{report['strategies_with_zero_cycles']} got ZERO cycles"
    )
    print(
        f"  trading_ops over the same window: {report['trading_ops']['rows']} rows "
        f"-- op-log entries, NOT cycles: {report['trading_ops']['by_status']}"
    )
    print()
    print(
        f"  PROPOSED ANYWHERE (either producer) "
        f"{len(report['proposed_anywhere'])} of {report['registry_strategies']}: "
        f"{report['proposed_anywhere']}"
    )
    print(
        f"  NEVER PROPOSED IN {args.hours:g}h -- not a lost slot, not a refused "
        f"cycle, NOTHING PUT THEM FORWARD ({len(report['never_proposed'])}):"
    )
    for name in report["never_proposed"]:
        print(f"    {name}")
    print()
    print(
        "  BREADTH TREND, oldest first -- read this BEFORE quoting any "
        "per-strategy count."
    )
    print(
        "  A strategy count that moves while the symbol count moves under it "
        "has measured the feed, not the allocation."
    )
    print(
        f"    {'window':>14s} {'symbols':>8s} {'holders':>8s} "
        f"{'proposed':>9s} {'cycles':>7s} {'entries':>8s}"
    )
    for row in report["breadth_trend"]:
        label = f"-{row['from_hours_ago']:g}h..-{row['to_hours_ago']:g}h"
        print(
            f"    {label:>14s} {row['symbols']:8d} {row['slot_holders']:8d} "
            f"{row['proposed']:9d} {row['cycles']:7d} {row['entries']:8d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
