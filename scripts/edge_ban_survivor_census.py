"""Does the symbol edge ban EMPTY the candidate set, or only thin it?

MEMORY.md carries, from p719: "entry-predropped-edge-ban (17,476) is the LARGEST
of the 18 statuses => services/symbol_edge_gate.py is the #1 throttle." That is
an inference, not a measurement: the largest status count says how often the ban
FIRES, and says nothing about whether anything survived it. A gate that drops
two of three candidates fires on every tick and blocks nothing.

The rows answer it themselves. Each carries
``details.surviving_enter_candidates`` -- the count left AFTER the ban dropped
what it dropped -- so the question needs no reconstruction:

  zero survivors  -> on this tick the ban IS why nothing was entered
  >=1 survivor    -> the ban thinned the field and entry failed further down

PRE-REGISTERED BY COVE AT p721 BEFORE RUNNING, with the complement named:
hypothesis, the ban is a THROUGHPUT TAX and the block is downstream, which shows
up as the zero-survivor share being small and the rows being spread across
symbols the gate still admits; complement, the ban IS the universe block, which
shows up as >80% of rows sitting on the handful of symbols banned today.

Read-only. Opens the store with mode=ro and PRAGMA query_only.

    python -X utf8 scripts/edge_ban_survivor_census.py
"""

from __future__ import annotations

import collections
import datetime
import json
import os
import sqlite3
import sys

STATUS = "entry-predropped-edge-ban"


def store() -> str:
    return os.getenv("TRADING_DB_PATH") or os.path.join(
        "storage", "trading_cache.db"
    )


def main() -> int:
    path = store()
    if not os.path.exists(path):
        sys.stderr.write("no store at %s\n" % path)
        return 1
    conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    conn.execute("PRAGMA query_only=1")

    surv: collections.Counter = collections.Counter()
    by_symbol: collections.Counter = collections.Counter()
    zero_by_symbol: collections.Counter = collections.Counter()
    by_day: collections.Counter = collections.Counter()
    zero_by_day: collections.Counter = collections.Counter()
    pool_by_day: dict = collections.defaultdict(list)
    dropped_by_day: dict = collections.defaultdict(list)
    missing = total = 0

    rows = conn.execute(
        "SELECT symbol, details, ts FROM trading_ops WHERE status=?", (STATUS,)
    )
    for symbol, details, ts in rows:
        total += 1
        by_symbol[symbol] += 1
        day = datetime.datetime.fromtimestamp(
            ts or 0, datetime.timezone.utc
        ).strftime("%Y-%m-%d")
        by_day[day] += 1
        try:
            payload = json.loads(details or "{}")
        except Exception:  # noqa: BLE001
            payload = {}
        if "surviving_enter_candidates" not in payload:
            missing += 1
            continue
        left = payload["surviving_enter_candidates"]
        surv[left] += 1
        n_dropped = len(payload.get("dropped") or [])
        pool_by_day[day].append(left + n_dropped)
        dropped_by_day[day].append(n_dropped)
        if not left:
            zero_by_symbol[symbol] += 1
            zero_by_day[day] += 1

    judged = total - missing
    if not judged:
        print("no %s rows carry surviving_enter_candidates" % STATUS)
        return 1

    zero = surv.get(0, 0)
    print("%s rows: %d over %d symbols (%d without the field)"
          % (STATUS, total, len(by_symbol), missing))
    print()
    print("surviving_enter_candidates AFTER the ban:")
    for key in sorted(surv):
        print("   %4d -> %7d  (%5.1f%%)"
              % (key, surv[key], 100.0 * surv[key] / judged))
    print()
    print("BAN LEFT ZERO CANDIDATES : %6d of %d  (%.1f%%)"
          % (zero, judged, 100.0 * zero / judged))
    print("SOMETHING SURVIVED IT    : %6d of %d  (%.1f%%)"
          % (judged - zero, judged, 100.0 * (judged - zero) / judged))

    # THE CONTROL THIS CREW IS REQUIRED TO RUN, and here it does not say what it
    # said for the halt rows: 79% of those were ONE DAY, so the pooled figure was
    # a fossil. These are spread over 18 days with no day above 17.9% -- and the
    # zero-survivor share is NOT flat, it CLIMBS. A pooled average over a
    # worsening series understates today, which is the only day that can be fixed.
    print()
    print("by DAY -- the pooled figure above is an average over a TREND:")
    for day in sorted(by_day):
        count = by_day[day]
        print("   %s  %7d  %5.1f%%   zero-survivor %6d  (%5.1f%% of the day)"
              % (day, count, 100.0 * count / total, zero_by_day[day],
                 100.0 * zero_by_day[day] / count))
    worst = by_day.most_common(1)[0]
    print("   largest single day: %s, %d rows, %.1f%% of all %d -- NOT a burst"
          % (worst[0], worst[1], 100.0 * worst[1] / total, total))

    # WHAT THE BAN SUBTRACTS FROM, which is the number that actually explains the
    # zero-survivor share. The rows carry both halves, so the pool needs no
    # reconstruction: pool = surviving_enter_candidates + len(details.dropped).
    #
    # Cove p721 measured the distinct banned (symbol,strategy) pairs per day and
    # found they move INVERSELY to the zero-survivor share -- 78 pairs at 37.4% on
    # 09-23 against 26 pairs at 70.4% on 09-26. So the ban did not get hungrier.
    # A ban that removes two candidates empties a pool of two and not a pool of
    # ten, and this is the column that says which happened.
    print()
    print("the POOL the ban subtracts from -- surviving + dropped, per day:")
    for day in sorted(pool_by_day):
        vals = sorted(pool_by_day[day])
        median = vals[len(vals) // 2]
        print("   %s  n=%-6d median pool %4d   mean %5.2f   median dropped %d"
              % (day, len(vals), median, sum(vals) / len(vals),
                 sorted(dropped_by_day[day])[len(vals) // 2]))
    print("   READ IT AS CONFIRMATION, WHICH IS THE WEAKER KIND: the conclusion")
    print("   came from the pair-count elimination FIRST, and the thinnest day in")
    print("   the series is the same day the headline share comes from. A fresh")
    print("   day is still the test.")

    # The other half of the pre-registered test: are the rows on the symbols the
    # gate refuses TODAY? If they are not, these are per-STRATEGY bans on
    # symbols the gate still admits, which is a different claim entirely.
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import services.symbol_edge_gate as gate  # noqa: PLC0415

        banned = set(gate.banned_symbols())
        pairs = gate.banned_pairs()
    except Exception as exc:  # noqa: BLE001
        print("\n(could not load the gate for the today-split: %s)" % exc)
        return 0

    on_banned = sum(n for s, n in by_symbol.items() if s in banned)
    print()
    print("gate today: %d symbols banned, %d pairs banned" % (len(banned), len(pairs)))
    print("rows on symbols BANNED today   : %6d  (%.1f%%)"
          % (on_banned, 100.0 * on_banned / total))
    print("rows on symbols ADMITTED today : %6d  (%.1f%%)"
          % (total - on_banned, 100.0 * (total - on_banned) / total))
    print()
    print("top 12 symbols by ban rows:")
    for symbol, count in by_symbol.most_common(12):
        print("   %-22s %7d  %5.1f%%   banned_today=%s   zero_survivor_rows=%d"
              % (symbol, count, 100.0 * count / total,
                 symbol in banned, zero_by_symbol[symbol]))
    print()
    print("CAVEAT, and it limits the today-split ONLY: the rows are historical "
          "and the ban set moves, so a symbol admitted today may have been "
          "banned when its row was written. The zero-survivor share above does "
          "NOT depend on that -- it was recorded on the tick.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
