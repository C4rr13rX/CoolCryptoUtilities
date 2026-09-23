"""Frozen and stale tape census -- [edb6da74] criterion 1, re-runnable.

A symbol with fewer than 2 DISTINCT prices over the window is a CONSTANT
series: every variance, spread, momentum, reversal and cost-vs-move statistic
computed on it is a silent zero or a divide-by-zero. Pass 480's h*=5.842 on
DRB-USDC was one instance of that; this counts the rest.

The two failure modes are NOT the same and the distinction is the point:

  STALE   -- the newest tick is old. The feed stopped.
  FROZEN  -- the prices are all identical. The feed is publishing a dead
             price, and it may be publishing it RIGHT NOW.

A staleness-only guard passes every frozen-but-FRESH symbol, which is the
category that actually reaches the gates. Read-only.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time

WINDOW_H = 168.0
STALE_H = 24.0


def main() -> int:
    path = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
    conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    now = time.time()
    rows = conn.execute(
        "select symbol, count(*), count(distinct price), max(ts) "
        "from market_stream where ts >= ? and price is not null "
        "group by symbol order by count(*) desc",
        (now - WINDOW_H * 3600.0,),
    ).fetchall()

    frozen = [r for r in rows if r[2] < 2]
    stale = [r for r in rows if (now - r[3]) > STALE_H * 3600.0]
    both = [r for r in frozen if (now - r[3]) > STALE_H * 3600.0]
    fresh_frozen = [r for r in frozen if (now - r[3]) <= STALE_H * 3600.0]

    n = len(rows) or 1
    print("market_stream over %.0fh, stale threshold %.0fh\n" % (WINDOW_H, STALE_H))
    print("  symbols streamed          : %d" % len(rows))
    print("  FROZEN (<2 distinct price): %d  (%.1f%%)" % (len(frozen), 100.0 * len(frozen) / n))
    print("  STALE  (newest > %.0fh)    : %d  (%.1f%%)" % (STALE_H, len(stale), 100.0 * len(stale) / n))
    print("  both frozen AND stale     : %d" % len(both))
    print("  FROZEN BUT FRESH          : %d   <- a staleness-only guard MISSES these"
          % len(fresh_frozen))

    print("\nFROZEN BUT FRESH -- live feed, dead price:")
    print("  %-18s %6s %8s" % ("symbol", "ticks", "age_h"))
    for sym, cnt, _d, last in sorted(fresh_frozen, key=lambda r: now - r[3]):
        print("  %-18s %6d %8.1f" % (sym, cnt, (now - last) / 3600.0))

    print("\nLargest frozen tapes by tick count:")
    print("  %-18s %6s %8s" % ("symbol", "ticks", "age_h"))
    for sym, cnt, _d, last in frozen[:10]:
        print("  %-18s %6d %8.1f" % (sym, cnt, (now - last) / 3600.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
