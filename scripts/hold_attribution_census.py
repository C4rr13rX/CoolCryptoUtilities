"""Why did every decision cycle HOLD? Answer it from recorded snapshots.

THE MEASUREMENT THIS EXISTS TO STOP BEING WRONG. Pass 105 opened on
``[ed0d721e]``, whose headline was "75% of the ghost lane's decision budget
re-decides one banned symbol". That number came from counting rows in
``trading_ops`` -- and ``trading_ops`` rows are not decision cycles. Measured
2026-09-10 over one hour:

    entry-predropped-edge-ban rows, all AERO-USDC          96
    AERO-USDC ticks in market_stream                       34
    AERO-USDC decision cycles in organism_snapshots        34
    ban rows clustered at the same instant (<=3s apart)     25 clusters,
                                                           23 of size FOUR

Four ``evaluate()`` calls land on one tick and each logs its own drop, so the
row count inflates 4x. AERO took 34 of 259 cycles (13.1%) against 34 of 295
ticks (11.5%): it is not over-allocated, and dropping it from the candidate
set frees ~13% of cycles rather than 60%. Worse for the premise, 72 of those
96 rows record ``surviving_enter_candidates=1`` -- the ban did not even empty
the set.

WHAT THE CYCLES ACTUALLY DID. ``organism_snapshots`` records the ``decision``
that ``trading/bot.py`` reached, and ``bot.py`` only writes a ``trading_ops``
row when ``action != "hold"``. So a hold leaves NO row anywhere and the entry
funnel census cannot see it. Same hour: 1606 of 1608 cycles decided ``hold``.
AERO specifically ran 33 of 34 cycles to an ``enter`` DIRECTIVE with an empty
``last_filter_reason`` -- the scheduler proposed an entry every time -- and
the decision was ``hold`` every time.

WHY, AND IT IS NOT THE BAN. ``hold`` is decided on two model heads, and both
had decayed. Per 2h bucket over 24h, oldest first, with the ``dp==0.5 and
nm==0.0`` default pairs excluded because they are the "no prediction"
sentinel rather than a reading:

    direction_prob MAX  1.000 1.000 1.000 0.931 0.954 0.933 0.688 0.598
                        0.428 0.281 0.251 0.313
    net_margin     MAX  +0.588 +1.690 +0.753 +0.632 +1.158 +1.085 +1.033
                        -0.752 -1.331 -0.951 -0.676 -0.180

The entry test needs ``net_margin >= 0``. For the last ten hours the MAXIMUM
net_margin across ~2800 cycles on EVERY symbol is negative, so the conjunct
is unsatisfiable by measurement rather than by inference -- and it was
satisfiable twelve hours earlier (max +1.033). Reporting the MAX is the whole
point: a median says "usually refused", a max says "cannot be satisfied".

Read-only. Touches no trading code and writes nothing.

Usage:
    python -X utf8 scripts/hold_attribution_census.py --hours 6
    python -X utf8 scripts/hold_attribution_census.py --hours 24 --bucket 2
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_DB = os.path.join("storage", "trading_cache.db")

#: ``bot.py`` seeds its prediction summary with these when the model produced
#: no reading at all. A pair at exactly (0.5, 0.0) is therefore the absence of
#: a prediction, not a neutral one, and averaging it into the head series
#: makes a collapsed head look like it still reaches its ceiling.
DEFAULT_DIRECTION_PROB = 0.5
DEFAULT_NET_MARGIN = 0.0


def _is_default_pair(direction_prob: float, net_margin: float) -> bool:
    return direction_prob == DEFAULT_DIRECTION_PROB and net_margin == DEFAULT_NET_MARGIN


def iter_decisions(
    rows: Iterable[Tuple[float, str]]
) -> Iterable[Tuple[float, str, Dict[str, Any]]]:
    """Yield ``(ts, symbol, decision)`` for snapshots that carry a decision.

    A snapshot whose payload will not parse, or that has no ``decision`` dict,
    is skipped rather than counted as a hold: an unreadable cycle is unknown,
    and calling it a hold would inflate the very number this reports.
    """
    for ts, payload in rows:
        try:
            snap = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
        except Exception:  # noqa: BLE001 - an unparseable snapshot is not evidence
            continue
        if not isinstance(snap, dict):
            continue
        decision = snap.get("decision")
        if not isinstance(decision, dict):
            continue
        symbol = str(
            decision.get("symbol")
            or (snap.get("sample") or {}).get("symbol")
            or ""
        ).upper()
        yield float(ts), symbol, decision


def census(
    rows: Iterable[Tuple[float, str]], *, now: float, bucket_hours: float = 2.0
) -> Dict[str, Any]:
    """Count decision CYCLES -- one per snapshot -- and attribute the holds."""
    actions: Counter = Counter()
    by_symbol: Dict[str, Counter] = defaultdict(Counter)
    buckets: Dict[int, List[Tuple[float, float, str]]] = defaultdict(list)
    defaults = 0
    total = 0
    span = max(1.0, float(bucket_hours) * 3600.0)

    for ts, symbol, decision in iter_decisions(rows):
        total += 1
        action = str(decision.get("action") or "")
        actions[action] += 1
        by_symbol[symbol][action] += 1
        try:
            direction_prob = float(decision.get("direction_prob"))
            net_margin = float(decision.get("net_margin"))
        except (TypeError, ValueError):
            continue
        if _is_default_pair(direction_prob, net_margin):
            defaults += 1
            continue
        buckets[int((now - ts) // span)].append((direction_prob, net_margin, action))

    series: List[Dict[str, Any]] = []
    for key in sorted(buckets, reverse=True):
        seen = buckets[key]
        probs = [item[0] for item in seen]
        margins = [item[1] for item in seen]
        series.append(
            {
                "hours_ago": round((key + 1) * float(bucket_hours), 2),
                "n": len(seen),
                "direction_prob_p50": statistics.median(probs),
                "direction_prob_max": max(probs),
                "net_margin_p50": statistics.median(margins),
                "net_margin_max": max(margins),
                "non_hold": sum(1 for item in seen if item[2] != "hold"),
            }
        )

    holds = actions.get("hold", 0)
    return {
        "cycles": total,
        "holds": holds,
        "non_hold": total - holds,
        "hold_share": (float(holds) / total) if total else 0.0,
        "default_pairs": defaults,
        "actions": dict(actions),
        "by_symbol": {sym: dict(cnt) for sym, cnt in by_symbol.items()},
        "series": series,
        # The verdict the entry conjunct actually turns on. A NEGATIVE maximum
        # over a whole bucket means no symbol at any price could have passed
        # `net_margin >= 0` in it -- unsatisfiable, not merely unlikely.
        "buckets_with_negative_net_margin_max": sum(
            1 for row in series if row["net_margin_max"] < 0.0
        ),
    }


def ban_row_multiplicity(
    conn: sqlite3.Connection, *, now: float, hours: float, gap: float = 3.0
) -> Dict[str, Any]:
    """How many ``trading_ops`` ban ROWS land on one tick.

    This is the arithmetic that made [ed0d721e]'s headline wrong, so it is
    reported beside the cycle count rather than left for the next reader to
    rediscover.
    """
    try:
        rows = conn.execute(
            "select ts, symbol from trading_ops "
            "where status='entry-predropped-edge-ban' and ts>=? order by ts",
            (now - hours * 3600.0,),
        ).fetchall()
    except sqlite3.Error:
        return {"rows": 0, "clusters": 0, "rows_per_cluster": 0.0, "by_symbol": {}}
    clusters = 0
    previous: Optional[float] = None
    for ts, _symbol in rows:
        if previous is None or (float(ts) - previous) > gap:
            clusters += 1
        previous = float(ts)
    return {
        "rows": len(rows),
        "clusters": clusters,
        "rows_per_cluster": (len(rows) / clusters) if clusters else 0.0,
        "by_symbol": dict(Counter(str(sym) for _ts, sym in rows).most_common(6)),
    }


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--bucket", type=float, default=2.0, help="bucket size, hours")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    now = time.time()
    if not os.path.exists(args.db):
        print(f"no such database: {args.db}", file=sys.stderr)
        return 2
    rows = _load(args.db, args.hours, now)
    report = census(rows, now=now, bucket_hours=args.bucket)

    conn = sqlite3.connect(args.db)
    try:
        report["ban_rows"] = ban_row_multiplicity(conn, now=now, hours=args.hours)
    finally:
        conn.close()

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(f"DECISION CYCLES, last {args.hours:g}h from {args.db}")
    print(f"  cycles with a decision : {report['cycles']}")
    print(
        f"  hold                   : {report['holds']} "
        f"({report['hold_share'] * 100:.1f}%)"
    )
    print(f"  non-hold               : {report['non_hold']}  {report['actions']}")
    print(
        f"  'no prediction' pairs  : {report['default_pairs']} "
        f"(direction_prob==0.5 and net_margin==0.0; excluded from the series)"
    )
    print()
    ban = report["ban_rows"]
    print("BAN ROWS ARE NOT CYCLES -- rows per tick, the [ed0d721e] arithmetic")
    print(
        f"  entry-predropped-edge-ban rows {ban['rows']} in {ban['clusters']} clusters "
        f"= {ban['rows_per_cluster']:.1f} rows per tick"
    )
    print(f"  by symbol: {ban['by_symbol']}")
    print()
    print("HEADS PER BUCKET, newest last -- the MAX is the one that decides")
    print("  hours_ago     n   dir_prob p50/MAX     net_margin p50/MAX   non-hold")
    for row in reversed(report["series"]):
        print(
            f"  -{row['hours_ago']:>6.1f}h {row['n']:>5d}   "
            f"{row['direction_prob_p50']:.3f} / {row['direction_prob_max']:.3f}      "
            f"{row['net_margin_p50']:+.3f} / {row['net_margin_max']:+.3f}      "
            f"{row['non_hold']}"
        )
    print()
    negative = report["buckets_with_negative_net_margin_max"]
    if negative:
        print(
            f"  VERDICT: {negative} of {len(report['series'])} buckets have a NEGATIVE "
            "net_margin MAXIMUM -- in those, no symbol at any price could satisfy "
            "'net_margin >= 0' and the entry conjunct is unsatisfiable."
        )
    else:
        print(
            "  VERDICT: every bucket reaches a non-negative net_margin maximum, so "
            "'net_margin >= 0' was satisfiable throughout the window."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
