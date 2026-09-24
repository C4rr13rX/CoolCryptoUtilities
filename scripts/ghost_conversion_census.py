"""Candidate -> opened position conversion in the ghost lane, over any window.

Written pass 697 by Sage, to make the p697 restart's back-to-back ONE COMMAND
instead of a re-derivation. Read-only; there is no write path in here.

WHY THIS EXISTS
---------------
The ladder prints ``[FAIL] 3 GHOST  no ghost activity in 1h`` and the obvious
reading is a dead or frozen process. Measured p697, that reading is WRONG:
production was writing ``hold`` and ``enter`` rows 0.23h old while its newest
OPENED position was 9.75h old. The lane was evaluating ~952 candidates a day
and converting none of them.

So the quantity that matters is not "is it alive" and not "when was the last
tick". It is CONVERSION:

    ghost_candidate + ghost_candidate_quote_ok   ->   ghost-entry   ->   ghost-exit
              (evaluated)                          (position opened)   (round trip
                                                                        the bar counts)

The graduation bar counts only CLOSED round trips, so a healthy candidate rate
with a dead conversion rate produces exactly zero evidence while looking busy.

THE PRE-RESTART BASELINE, so p698 can compare rather than re-measure
-------------------------------------------------------------------
Captured 2026-09-24 04:20 local, BEFORE the production restart at 04:23
(@Dune, item [6285ff87]). Trailing 24h at that moment::

    enter  rows (all statuses)     978
      ghost_candidate_quote_ok     553
      ghost_candidate              399
      ghost-entry                   26     <- newest 9.75h old
    exit   rows                     25
      ghost-exit                    25     <- newest 9.75h old
    hold   rows                  17748

    conversion candidate -> opened position:  26 / 978 = 2.66%

The last pre-restart ``trading_ops`` row was 04:08 local. **Any row after 04:23
is post-restart evidence**, so the two populations are cleanly separable with no
join and no guessing.

WHAT THE 9.75h GAP IS AND IS NOT
--------------------------------
Inter-``ghost-entry`` gaps over the trailing 14d: median 0.71h, p90 4.15h, max
88.62h, and 5 of 111 gaps were at least 9.76h. So the open gap sat at **p=0.045**
-- uncommon, NOT proof of a break. This tape has produced 10.6h, 12.2h, 26.3h,
58.2h and 88.6h gaps in a fortnight. Quote the 2.66% conversion, which is robust;
do not quote the gap as evidence that something broke. ``--gaps`` reprints that
distribution so the claim can be re-checked rather than believed.

    python scripts/ghost_conversion_census.py
    python scripts/ghost_conversion_census.py --hours 6 --buckets 2
    python scripts/ghost_conversion_census.py --gaps
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: The status a row carries once the lane has actually OPENED a ghost position.
#: Everything else on ``action='enter'`` is an evaluation, not a position --
#: which is the whole distinction this file exists to keep visible.
OPENED = "ghost-entry"
CLOSED = "ghost-exit"

#: The p697 pre-restart baseline, carried in code so a comparison cannot drift
#: from a number somebody remembered. (enter_rows, opened, closed).
BASELINE_24H = (978, 26, 25)
BASELINE_LABEL = "p697 pre-restart, 2026-09-24 04:20 local"


def _db() -> str:
    return os.environ.get("TRADING_DB_PATH", str(PROJECT_ROOT / "storage" / "trading_cache.db"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--buckets", type=float, default=3.0,
                    help="bucket width in hours for the activity table")
    ap.add_argument("--gaps", action="store_true",
                    help="also print the inter-opened-position gap distribution")
    ap.add_argument("--gap-days", type=float, default=14.0)
    args = ap.parse_args()

    now = time.time()
    t0 = now - args.hours * 3600.0
    conn = sqlite3.connect(f"file:{_db()}?mode=ro", uri=True)
    try:
        by_action: Dict[str, Dict[int, int]] = {}
        by_status: Dict[str, int] = {}
        nb = max(1, int(round(args.hours / args.buckets)))
        for action, status, ts in conn.execute(
            "SELECT action, status, ts FROM trading_ops WHERE ts > ?", (t0,)
        ):
            b = min(nb - 1, int((now - float(ts)) // (args.buckets * 3600.0)))
            by_action.setdefault(str(action), {})[b] = \
                by_action.setdefault(str(action), {}).get(b, 0) + 1
            by_status[str(status)] = by_status.get(str(status), 0) + 1

        newest: Dict[str, float] = {}
        for action in sorted(by_action):
            row = conn.execute(
                "SELECT MAX(ts) FROM trading_ops WHERE action = ?", (action,)
            ).fetchone()
            if row and row[0] is not None:
                newest[action] = float(row[0])

        gaps: List[float] = []
        open_ts: List[float] = []
        if args.gaps:
            open_ts = [float(r[0]) for r in conn.execute(
                "SELECT ts FROM trading_ops WHERE action = 'enter' AND status = ? "
                "AND ts > ? ORDER BY ts", (OPENED, now - args.gap_days * 86400.0))]
            gaps = sorted((open_ts[i] - open_ts[i - 1]) / 3600.0
                          for i in range(1, len(open_ts)))
    finally:
        conn.close()

    print("=" * 84)
    print("GHOST CONVERSION CENSUS -- candidates are not positions")
    print("=" * 84)
    print()
    print("ACTIVITY, trailing %.0fh in %.0fh buckets (leftmost = most recent)"
          % (args.hours, args.buckets))
    hdr = "%-22s" % "action (hours ago)" + "".join(
        "%8s" % ("%d-%d" % (b * args.buckets, (b + 1) * args.buckets))
        for b in range(nb))
    print(hdr)
    print("-" * len(hdr))
    for action in sorted(by_action, key=lambda a: -sum(by_action[a].values())):
        buckets = by_action[action]
        total = sum(buckets.values())
        print("%-22s" % action
              + "".join("%8d" % buckets.get(b, 0) for b in range(nb))
              + "   tot %d" % total)
    print()
    print("NEWEST ROW PER ACTION -- 'alive' and 'converting' are different questions")
    for action, ts in sorted(newest.items(), key=lambda kv: kv[1], reverse=True):
        print("    %-24s %8.2f h ago" % (action, (now - ts) / 3600.0))

    print()
    print("ENTER-ACTION STATUS BREAKDOWN, trailing %.0fh" % args.hours)
    enter_rows = 0
    for status, n in sorted(by_status.items(), key=lambda kv: -kv[1]):
        if status in (OPENED, "ghost_candidate", "ghost_candidate_quote_ok"):
            enter_rows += n
            mark = "  <- POSITION OPENED" if status == OPENED else "  (evaluated only)"
            print("    %-30s %6d%s" % (status, n, mark))
    opened = by_status.get(OPENED, 0)
    closed = by_status.get(CLOSED, 0)

    print()
    print("CONVERSION -- the number the graduation bar actually depends on")
    rate = (100.0 * opened / enter_rows) if enter_rows else float("nan")
    print("    candidates evaluated        %6d" % enter_rows)
    print("    positions OPENED            %6d   (%.2f%% conversion)" % (opened, rate))
    print("    round trips CLOSED          %6d   <- the only rows the bar counts"
          % closed)
    b_enter, b_open, b_closed = BASELINE_24H
    b_rate = 100.0 * b_open / b_enter
    print()
    print("    BASELINE (%s), trailing 24h:" % BASELINE_LABEL)
    print("        %d evaluated / %d opened / %d closed  = %.2f%% conversion"
          % (b_enter, b_open, b_closed, b_rate))
    if abs(args.hours - 24.0) < 1e-9:
        print("        DELTA: %+d evaluated, %+d opened, %+d closed, %+.2f pp conversion"
              % (enter_rows - b_enter, opened - b_open, closed - b_closed,
                 (rate - b_rate) if rate == rate else float("nan")))
    else:
        print("        (run with --hours 24 for a like-for-like delta)")

    if args.gaps:
        print()
        print("INTER-POSITION GAP DISTRIBUTION, trailing %.0fd -- the honesty check"
              % args.gap_days)
        if len(gaps) < 2:
            print("    fewer than 2 gaps; nothing quotable")
        else:
            cur = (now - open_ts[-1]) / 3600.0
            at_least = sum(1 for g in gaps if g >= cur)
            print("    %s rows %d   median %.2fh   p90 %.2fh   max %.2fh"
                  % (OPENED, len(open_ts), statistics.median(gaps),
                     gaps[int(0.9 * len(gaps))], gaps[-1]))
            print("    CURRENT OPEN GAP %.2fh   -- %d of %d historical gaps are at "
                  "least this long = p=%.3f"
                  % (cur, at_least, len(gaps), at_least / len(gaps)))
            print("    A LONG GAP IS NOT PROOF OF A BREAK. Quote the conversion rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
