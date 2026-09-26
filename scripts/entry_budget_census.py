#!/usr/bin/env python
"""Census the term that gates the WHOLE entry lane.

Background, so this is readable without the board history:

``trading/bot.py:6549`` reads ``if risk_budget <= 0`` and, when it is true,
skips ``scheduler.evaluate()`` at :6577 entirely.  That is the ONLY caller.  A
zero there writes no arbitration row, no no-candidates row, and no row of any
status -- the entry lane goes silent and every census downstream reads that
silence as "the market offered nothing".  Pass 714 could only read the value by
attaching ``py-spy --locals`` to the live process.

There are TWO terms that can zero it, and they are the same cause:

  * ``bot.py:6510  risk_budget *= ghost_multiplier``, where ghost_multiplier is
    ``risk_flags.ghost_risk_multiplier`` from the transition plan.
  * ``bot.py:6513-6514  if plan_flags.get("halt_ghost") ...: risk_budget = 0.0``

Upstream in ``trading/pipeline.py`` both are driven by one boolean:
``ghost_risk_multiplier`` is a product of five individually-floored factors
(so it cannot reach zero by arithmetic) and is set to ``0.0`` only in the
``else`` of ``if ghost_collection_ready:``; and ``halt_ghost`` is literally
``not ghost_collection_ready``.  Everything else at :6503-6514 is a min/clamp.

Two independent instruments read that state, and this script reports both:

  DIRECT   metrics stage='ghost_trading' category='entry_budget', written
           unconditionally once per evaluated sample by bot.py:6533.  This is
           the uncensored sample -- it reports the value on running ticks as
           well as halted ones.  It only exists in builds at or after ecff145,
           so a production process booted before that commit writes NOTHING
           here and the script says so rather than reporting a misleading zero.

  CENSORED feedback_events source='scheduler' label='halted', written by
           bot.py:6551.  Available in every build, but it fires only INSIDE
           the halt and only on the TRANSITION into it, so it can prove the
           multiplier is zero when the lane is dead and can say nothing about
           its value when the lane is running.  Counting transitions also
           reveals RECOVERIES, which a "is it halted right now" check hides.

Do not read a zero from the DIRECT section as "the budget is zero".  Read it
as "this build never wrote the row".  That distinction is the whole reason
this script exists; see ``--boot``.

Usage:
    python scripts/entry_budget_census.py
    python scripts/entry_budget_census.py --boot "2026-09-25 19:59:47"
    python scripts/entry_budget_census.py --hours 2
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import Counter

DEFAULT_DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")

# The four names bot.py:6533 writes. None of them collided with anything in the
# 22GB store when the instrument landed (all n=0 across the whole metrics
# table), so a non-zero count here is unambiguously the new recorder.
DIRECT_NAMES = (
    "ghost_multiplier",
    "entry_risk_budget",
    "global_risk_budget",
    "entry_lane_halted",
)


def _connect(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise SystemExit(f"no such db: {path}")
    if os.path.getsize(path) == 0:
        raise SystemExit(
            f"{path} is a 0-byte stub and will silently return nothing. "
            "The real store is storage/trading_cache.db (TRADING_DB_PATH)."
        )
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _fmt_ago(now: float, ts: float | None) -> str:
    if not ts:
        return "never"
    return f"{(now - ts) / 3600:.2f}h ago"


def direct_section(conn: sqlite3.Connection, since: float, now: float) -> bool:
    """Report the unconditional per-sample rows. Returns True if any exist."""
    print("=" * 72)
    print("DIRECT -- metrics category='entry_budget' (bot.py:6533, uncensored)")
    print("=" * 72)

    total = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE category='entry_budget'"
    ).fetchone()[0]
    if not total:
        print("  0 rows ALL-TIME.")
        print("  This does NOT mean the risk budget is zero. It means the running")
        print("  build predates ecff145 and never wrote the row. Restart production")
        print("  into a build containing trading/bot.py:6532-6545, then re-run.")
        print("  Until then the CENSORED section below is the only live evidence.")
        return False

    print(f"  {total} rows all-time.")
    for name in DIRECT_NAMES:
        row = conn.execute(
            "SELECT COUNT(*), MIN(value), MAX(value), AVG(value), MAX(ts) "
            "FROM metrics WHERE category='entry_budget' AND name=? AND ts>?",
            (name, since),
        ).fetchone()
        n, lo, hi, avg, last = row
        if not n:
            print(f"  {name:<20} n=0 in window")
            continue
        print(
            f"  {name:<20} n={n:<7} min={lo:<10.6f} max={hi:<10.6f} "
            f"mean={avg:<10.6f} last={_fmt_ago(now, last)}"
        )

    # The deliverable: how often the gate actually tested zero.
    halted = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE category='entry_budget' "
        "AND name='entry_lane_halted' AND value>0 AND ts>?",
        (since,),
    ).fetchone()[0]
    sampled = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE category='entry_budget' "
        "AND name='entry_lane_halted' AND ts>?",
        (since,),
    ).fetchone()[0]
    if sampled:
        pct = 100.0 * halted / sampled
        print(f"\n  ENTRY LANE HALTED ON {halted}/{sampled} EVALUATED SAMPLES ({pct:.1f}%)")
        print("  ^ THIS IS THE DELIVERABLE. Anything below 100% means the lane")
        print("    recovers and the halt is intermittent, not a sealed door.")
    return True


def censored_section(conn: sqlite3.Connection, since: float, now: float) -> None:
    print()
    print("=" * 72)
    print("CENSORED -- feedback_events scheduler/halted (bot.py:6551, transitions only)")
    print("=" * 72)

    total = conn.execute(
        "SELECT COUNT(*) FROM feedback_events WHERE source='scheduler' AND label='halted'"
    ).fetchone()[0]
    rows = conn.execute(
        "SELECT ts, details FROM feedback_events "
        "WHERE source='scheduler' AND label='halted' AND ts>? ORDER BY ts",
        (since,),
    ).fetchall()
    print(f"  {total} rows all-time, {len(rows)} in window.")
    if not rows:
        print("  No halt TRANSITION in this window. Either the lane never halted,")
        print("  or it entered the halt before the window opened and never left")
        print("  (the transition-only recorder cannot tell those apart -- that is")
        print("  exactly what the DIRECT section above is for).")
        return

    mult: Counter = Counter()
    reason: Counter = Counter()
    attributed: list[tuple[object, object]] = []
    for _ts, raw in rows:
        try:
            det = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            det = {}
        mult[det.get("ghost_multiplier")] += 1
        reason[det.get("reason")] += 1
        attributed.append((det.get("symbol"), det.get("halt_transitions")))

    print(f"  ghost_multiplier values : {dict(mult)}")
    print(f"  halt reasons            : {dict(reason)}")
    print(f"  first {_fmt_ago(now, rows[0][0])}, last {_fmt_ago(now, rows[-1][0])}")

    # DO NOT read n transitions as n-1 recoveries. Dune p716 published exactly
    # that and it is wrong: `_scheduler_halted` is initialised False per
    # TradingBot instance (bot.py:545) and `trading/selector.py:1448-1450`
    # builds ONE BOT PER SYMBOL, with :1781 adding more as symbols rotate. So
    # N transitions across N bots is N FIRST halts and ZERO recoveries. The
    # "transition implies a prior un-halted state" inference is valid only
    # WITHIN one object's lifetime, never across the fleet.
    #
    # The give-away is the burst: 13 of the 29 post-boot transitions landed
    # inside about half a minute at startup, which is build() constructing the
    # per-symbol bots, not a lane opening and shutting 13 times in 30 seconds.
    # Nothing in feedback_events carries a bot or symbol id, so this recorder
    # CANNOT separate the two readings -- only the per-sample entry_budget rows
    # in the DIRECT section can, which is one more reason the restart matters.
    n_rec = len(rows) - 1
    print(f"\n  {len(rows)} transitions. This is NOT "
          f"{n_rec} recover{'y' if n_rec == 1 else 'ies'} --")
    print("  _scheduler_halted is PER BOT and selector.py builds one bot per")
    print("  symbol, so a burst of transitions is a fleet booting, not a lane")
    print("  flapping. Use the DIRECT section to count recoveries; it samples")
    print("  every tick and this recorder only sees first-halts.")
    if len(rows) > 1:
        gaps = sorted(rows[i + 1][0] - rows[i][0] for i in range(len(rows) - 1))
        mid = gaps[len(gaps) // 2]
        print(
            f"  gaps between transitions: min={gaps[0]/60:.1f}m "
            f"median={mid/60:.1f}m max={gaps[-1]/60:.1f}m"
        )

    # ...and here is the field that DOES settle it, without waiting for the
    # per-sample rows. Dune p717 added `symbol` and `halt_transitions` to the
    # halt row (bot.py:6551) precisely because the paragraph above had to say
    # "this recorder CANNOT separate the two readings". It can now, from ONE
    # row: `halt_transitions` is a per-bot counter, so a row reading >= 2 means
    # that bot halted, left the halt, and halted again -- and leaving requires
    # `risk_budget > 0` at bot.py:6549, which requires evaluate() to have run.
    # A fleet of first-halts is all 1s. A flapping lane has a 2 somewhere. The
    # two arms are distinguished by the same number, which is what makes it a
    # measurement rather than a story.
    print()
    known = [(s, t) for s, t in attributed if isinstance(t, (int, float))]
    if not known:
        print("  ATTRIBUTION: none of these rows carry `symbol`/`halt_transitions`.")
        print("  They predate Dune p717. Standing-vs-flapping is UNDECIDABLE from")
        print("  this window -- do not claim either arm from the row count alone.")
        return

    repeats = [(s, int(t)) for s, t in known if int(t) >= 2]
    per_symbol: Counter = Counter(str(s or "?") for s, _t in known)
    print(f"  ATTRIBUTION: {len(known)} of {len(rows)} rows attributed, "
          f"{len(per_symbol)} distinct symbol(s).")
    print(f"  worst halt_transitions per symbol: "
          f"{dict(sorted(((str(s or '?'), max(int(t) for s2, t in known if s2 == s)) for s in {s for s, _ in known}), key=lambda kv: -kv[1])[:8])}")
    if repeats:
        print(f"  *** FLAPPING, PROVEN: {len(repeats)} row(s) with "
              f"halt_transitions >= 2 ***")
        print("  Each is a bot that LEFT the halt and re-entered, so risk_budget")
        print("  was > 0 at bot.py:6549 and scheduler.evaluate() DID run between")
        print("  the two halts. The lane is intermittent, not shut.")
    else:
        print("  *** STANDING, PROVEN: every attributed row reads "
              "halt_transitions == 1 ***")
        print("  No bot has ever left the halt once it entered. This is a fleet")
        print("  of first-halts, exactly the reading the caveat above warns is")
        print("  possible -- and here it is the one that holds.")


def clamp_section(conn: sqlite3.Connection) -> None:
    """The metric that is NOT the entry gate, spelled out so nobody reuses it."""
    print()
    print("=" * 72)
    print("NOT THE ENTRY GATE -- metrics name='risk_budget' (bot.py:924)")
    print("=" * 72)
    row = conn.execute(
        "SELECT COUNT(*), MIN(value), MAX(value) FROM metrics WHERE name='risk_budget'"
    ).fetchone()
    n, lo, hi = row
    print(f"  n={n} min={lo} max={hi}")
    print("  This is self.global_risk_budget, the PRE-multiply value, and :913")
    print("  clamps it with max(0.3, ...). Its floor is a property of the clamp,")
    print("  not a measurement -- it is structurally incapable of reading zero.")
    print("  Do NOT cite it as evidence the entry budget is non-zero.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument(
        "--boot",
        help="local timestamp 'YYYY-MM-DD HH:MM:SS'; window starts here",
    )
    ap.add_argument(
        "--hours",
        type=float,
        help="window is the last N hours (ignored if --boot is given)",
    )
    args = ap.parse_args()

    now = time.time()
    if args.boot:
        since = time.mktime(time.strptime(args.boot, "%Y-%m-%d %H:%M:%S"))
        label = f"since boot {args.boot} ({_fmt_ago(now, since)})"
    elif args.hours:
        since = now - args.hours * 3600
        label = f"last {args.hours}h"
    else:
        since = 0.0
        label = "all time"

    conn = _connect(args.db)
    print(f"db={args.db}  window={label}\n")
    have_direct = direct_section(conn, since, now)
    censored_section(conn, since, now)
    clamp_section(conn)

    print()
    if not have_direct:
        print("VERDICT: the uncensored instrument has not run yet. Restart production")
        print("and re-run this script; only then is [5dc802c2] criterion 1 satisfied.")
    conn.close()


if __name__ == "__main__":
    main()
