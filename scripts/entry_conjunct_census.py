#!/usr/bin/env python
"""Item d0244ccb criterion 1 -- WHERE the scanner's enter directives stop.

The item asked for the exact function and the exact refusal reason, counted
rather than inferred.  Both are recoverable from the store without reading a
log: ``TradeScheduler._log_no_candidates`` (trading/scheduler.py) writes an
``entry-refused-no-candidates`` row whose ``detail`` IS the entry block, and
the block carries ``failed`` (which conjuncts bound) and ``values`` (what each
one got against the floor it needed).

That row is only written when a live quote-OK atf candidate was standing, so
these rows are directive deaths and not idle ticks -- see the comment at
trading/scheduler.py:1362.

What it prints, and the reason the item exists:

  * how often each conjunct binds, and how often it is the SOLE binder
  * the full observed range of the binding quantity against its floor, which
    is what separates "a strict gate" from "a gate that cannot be satisfied"

    python scripts/entry_conjunct_census.py [--hours 24]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import Counter

STATUS = "entry-refused-no-candidates"


def _pct(a: list, p: float) -> float:
    a = sorted(a)
    return a[int(p * (len(a) - 1))] if a else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()

    path = os.getenv("TRADING_DB_PATH", "storage/trading_cache.db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    since = time.time() - args.hours * 3600.0

    appears: Counter = Counter()
    sole: Counter = Counter()
    got: dict = {}
    need: dict = {}
    fee_rates: Counter = Counter()
    horizons: Counter = Counter()
    n = 0

    for row in conn.execute(
        "SELECT details FROM trading_ops WHERE ts>=? AND status=?", (since, STATUS)
    ):
        try:
            outer = json.loads(row["details"] or "{}")
        except Exception:  # noqa: BLE001
            continue
        block = outer.get("detail")
        if not isinstance(block, dict):
            continue
        n += 1
        failed = block.get("failed") or []
        if isinstance(failed, str):
            failed = [failed]
        for name in failed:
            appears[str(name)] += 1
        if len(failed) == 1:
            sole[str(failed[0])] += 1
        for name, pair in (block.get("values") or {}).items():
            if not isinstance(pair, dict) or "got" not in pair:
                continue
            got.setdefault(str(name), []).append(float(pair["got"]))
            if "need" in pair:
                need.setdefault(str(name), Counter())[float(pair["need"])] += 1
        fee_rates[block.get("fee_rate")] += 1
        horizons[str(block.get("horizon"))] += 1

    print("=" * 78)
    print("d0244ccb criterion 1 -- %s, window %.1fh, n=%d" % (STATUS, args.hours, n))
    print("  written by TradeScheduler._log_no_candidates, trading/scheduler.py")
    print("=" * 78)
    if not n:
        print("no rows in window")
        return 0

    print("\nCONJUNCT BINDS (a row can bind on several):")
    for name, c in appears.most_common():
        print("  %-28s %6d  %5.1f%%   sole binder %5d  %5.1f%%"
              % (name, c, 100.0 * c / n, sole.get(name, 0), 100.0 * sole.get(name, 0) / n))

    print("\nTHE QUANTITY vs ITS FLOOR -- can the gate be satisfied at all?")
    for name in sorted(got, key=lambda k: -appears.get(k, 0)):
        vals = got[name]
        floors = need.get(name) or Counter()
        floor = min(floors) if floors else None
        over = sum(1 for v in vals if floor is not None and v >= floor)
        print("  %-28s n=%-5d min=%.6f p50=%.6f max=%.6f" % (name, len(vals), min(vals), _pct(vals, 0.5), max(vals)))
        if floor is not None:
            span = "  <-- MAX IS %.4f BELOW THE FLOOR" % (floor - max(vals)) if max(vals) < floor else ""
            print("  %-28s floor(s)=%s   rows at-or-above floor: %d of %d%s"
                  % ("", sorted(floors), over, len(vals), span))

    print("\nCONTEXT CARRIED ON THE SAME BLOCK:")
    print("  fee_rate : %s" % dict(fee_rates))
    print("  horizon  : %s" % dict(horizons.most_common(8)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
