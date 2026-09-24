"""Who the arbitrator can select, at what LABEL horizon, and who it discards.

Read-only. Two questions, both from `trading_ops` rows the scheduler already
writes -- nothing here is inferred from code.

1. THE SELECTABLE POPULATION. `entry-arbitration.details.offered` is the exact
   set of strategies that reached arbitration on a tick. That is the correct
   denominator for "strategies the arbitrator can select" -- the registry FILE
   count (45) is not, because most of those never reach a tick.

2. THE TIEBREAK. One tick produces ONE directive. On a contested tick the
   arbitrator keeps one candidate and DISCARDS the rest. If the kept one
   carries a longer label horizon than a discarded one, the tick was spent on
   a candidate the chaos gate then refuses deterministically -- the horizon
   label is a constant per suffix (p674: six of six suffixes collapse to one
   value equal to their own label), so an @1w candidate is refused on every
   feed this repo has ever measured.

Usage:  python scripts/arbitration_horizon_census.py [hours]
"""
from __future__ import annotations

import collections
import json
import os
import re
import sqlite3
import sys
import time

_SUFFIX = re.compile(r"@(\d+)\s*([mhdw])\b")
_UNIT = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def label_horizon_sec(strategy_id: str):
    """Seconds implied by a strategy id's suffix, or None when it has none.

    None is NOT zero: a suffix-less strategy carries a real, varying horizon
    (20/45/60 min measured at p674) and cannot be ranked against a label.
    """
    m = _SUFFIX.search(strategy_id or "")
    if not m:
        return None
    return int(m.group(1)) * _UNIT[m.group(2)]


def main() -> int:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
    path = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
    conn = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
    cutoff = time.time() - hours * 3600.0

    rows = conn.execute(
        "SELECT details FROM trading_ops WHERE status='entry-arbitration' AND ts>?",
        (cutoff,),
    ).fetchall()

    offered = collections.Counter()
    chosen = collections.Counter()
    via = collections.Counter()
    contested = 0
    long_over_short = 0
    long_over_short_via = collections.Counter()
    discarded_short = collections.Counter()

    for (raw,) in rows:
        try:
            det = json.loads(raw)
        except Exception:  # noqa: BLE001 - a malformed row is not a finding
            continue
        off = det.get("offered") or {}
        for key, count in off.items():
            offered[key] += int(count)
        pick = str(det.get("chosen") or "none")
        chosen[pick] += 1
        via[str(det.get("via"))] += 1
        if len(off) < 2:
            continue
        contested += 1
        pick_h = label_horizon_sec(pick)
        if pick_h is None:
            continue
        shorter = [
            k for k in off
            if k != pick and (label_horizon_sec(k) or 0) < pick_h
        ]
        if shorter:
            long_over_short += 1
            long_over_short_via[str(det.get("via"))] += 1
            for k in shorter:
                discarded_short[k] += 1

    print(f"entry-arbitration rows, last {hours:g}h: {len(rows)}")
    print(f"  via: {dict(via)}")
    print(f"  contested ticks (>1 distinct strategy offered): {contested}")
    print(
        f"  of those, the KEPT candidate had a LONGER label horizon than a "
        f"discarded one: {long_over_short}"
        + (f"  ({100.0 * long_over_short / contested:.1f}% of contested)"
           if contested else "")
    )
    print(f"    by via: {dict(long_over_short_via)}")
    print()
    print("SELECTABLE POPULATION -- strategies that reached arbitration:")
    print(f"  distinct offered: {len(offered)}")
    unpassable = 0
    for key, count in offered.most_common():
        h = label_horizon_sec(key)
        tag = "no suffix (real horizon)" if h is None else f"{h / 60:.0f} min"
        never = h is not None and h > 96.0  # 1.6 min, the widest measured limit
        if never:
            unpassable += 1
        print(f"  {key:34s} offered={count:7d}  label={tag}"
              + ("   CANNOT EVER PASS CHAOS" if never else ""))
    print()
    print(f"  {unpassable} of {len(offered)} OFFERED strategies carry a label "
          f"horizon above the widest decay limit ever measured on this feed "
          f"(1.6 min), so they can never clear the chaos layer.")
    print()
    print("SHORT-HORIZON CANDIDATES DISCARDED in favour of a longer one:")
    for key, count in discarded_short.most_common(15):
        print(f"  {key:34s} discarded={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
