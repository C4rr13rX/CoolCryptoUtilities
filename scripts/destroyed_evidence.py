"""The ghost lane destroys four round trips for every one it books.

WHY THIS EXISTS
---------------
scripts/hold_time_edge.py -- the pass-100 instrument for [71975c13] -- reads
``trade_outcomes``. That makes it structurally blind to the population it most
needed to see: a ghost position that is RELEASED or ABANDONED never writes an
outcome row, so it is invisible to every tool this loop measures the book with.
The 17.7-hour CRV trip that item was written around was the worst trip that
BOOKED. The worst trip that did not book was held 11.6 days.

This counts the funnel from the entry side instead, out of ``trading_ops``,
where every fate of a ghost position is logged:

    ghost entries          1054
      -> ghost-exit         205   booked; this is the only evidence graduation reads
      -> position-released  784   evicted by a new entry, books nothing  [0f6957e3]
      -> position-abandoned  90   feed went dark, books nothing          [79ad4d0d]

19.4% of ghost entries become evidence. The other 80.6% are work the loop paid
for and then threw away, and the tradeable evidence rate that gates graduation
-- 3.4/day against a 20-trade bar -- is measured on the 19.4%.

THE ABANDONED 90 ARE THE WORST-HELD POSITIONS IN THE SYSTEM
-----------------------------------------------------------
Measured 2026-09-10 over 7 days, from the ``held_sec`` the release op already
records (the instrument exists; do not add logging):

    held_sec   median 9734s = 10.8x stale_exit_secs
               max 1005250s = 11.6 DAYS = 1117x
    ALL 90 of 90 exceed 4x stale_exit_secs

That last line is acceptance criterion 3 of [71975c13] failing at 100%, on a
population no previous pass could see. 78 of the 90 are TRADEABLE symbols --
the exact population the graduation bar counts -- so this is ~11 destroyed
tradeable round trips per day against a measured tradeable evidence rate of
3.4/day.

Every one of the 90 carries ``released_entry_price`` > 0 and ``released_size``
> 0, and 88 of 90 carry a ``released_strategy_id``. Every ingredient needed to
book a round trip is present at the abandon site. The code logs "the slot is
now free" and drops it.

DO NOT FIX THIS BY BOOKING AT THE LAST KNOWN PRICE
--------------------------------------------------
``silent_sec`` median is 3903s, so the last price the bot holds for an abandoned
position is 65 minutes stale, and an outcome booked at it is a fabricated fill.
This repo has already shipped a contaminated tick becoming the next entry basis.
The position is on a real DEX and a quote is one RPC call: price it AT the
abandon, or book it flagged so the tradeable filter can refuse it. A fake
tradeable round trip is worse than a destroyed one, because graduation counts it.

THE TWO DESTROYED POPULATIONS ARE OPPOSITE SHAPES
-------------------------------------------------
Only one of them is evidence, and the tool reports the discriminator
(``inside_horizon``) so nobody has to take that on trust.

The release op records no ``held_sec`` -- but it records ``released_entry_ts``,
and the op's own ``ts`` is when the position was taken away, so hold time is
recoverable from the log that already exists. Reading it, over 7 days:

    EVICTIONS   median held 22 SECONDS; 91.6% taken away INSIDE the 900s
                horizon, only 4.0% past 4x. The price at eviction is FRESH --
                median staleness 3.0s, p90 21.1s, 96.6% under 60s against
                market_stream -- so booking them would not fabricate a fill.
                The objection is different: a position force-closed 22 seconds
                after entry is not a round trip the STRATEGY made. It never
                chose to exit; the harness took its slot. Booking all 784
                injects ~51 tradeable rows/day of pure round-trip cost with no
                direction in them, which pushes win rate and P/L DOWN against a
                55%-and-positive bar -- graduation gets HARDER while the
                throughput number looks better. The fix is to stop evicting at
                22 seconds, not to book the eviction as an exit.

    ABANDONS    the mirror image, and genuine: 0 of 90 inside the horizon,
                median 10.8x, so the exit was truly OWED and no rule could
                reach it -- but the price is 65 minutes stale.

FRESH PRICE + NO REAL TRIP against REAL TRIP + STALE PRICE. Neither is fixed by
booking the row where it is discarded.

WHAT THIS TOOL IS FOR
---------------------
It is the before/after number for [0f6957e3] and [79ad4d0d]. Run it before the
fix and after, and read ``booked_share``, ``over_4x_stale`` and
``inside_horizon``. A rise in ``booked_share`` that comes from booking rows with
a high ``inside_horizon`` share is not progress -- it is the bar being fed rows
the strategy did not make.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DB = ROOT / "storage" / "trading_cache.db"

#: The wall-clock horizon the exit rules promise. A position held past 4x this
#: was not "held", it was lost track of -- that is criterion 3 of [71975c13].
STALE_EXIT_SECS = 900.0

#: ``trading_ops.status`` values, by what they do to the evidence.
BOOKED = ("ghost-exit",)
DESTROYED = {
    "position-released": "evicted by a new entry",
    "position-abandoned-dark-feed": "feed went dark",
}
ENTERED = ("ghost-entry",)


def _rows(con: sqlite3.Connection, status: str, since: float) -> List[Dict[str, Any]]:
    cur = con.execute(
        "SELECT ts, symbol, details FROM trading_ops WHERE status = ? AND ts >= ?",
        (status, since),
    )
    out = []
    for ts, symbol, details in cur.fetchall():
        try:
            d = json.loads(details or "{}")
        except (ValueError, TypeError):
            d = {}
        d.setdefault("symbol", symbol)
        d["ts"] = ts
        out.append(d)
    return out


def _held_secs(row: Dict[str, Any]) -> float:
    """How long the position was actually held, in seconds, or 0.0 if unknowable.

    The abandon op records ``held_sec`` directly. The RELEASE op does not -- but
    it records ``released_entry_ts``, and the op's own ``ts`` is the moment the
    position was taken away, so the hold time is recoverable from the log that
    already exists. Do not add logging for this: 775 of 784 releases over 7 days
    carry the entry timestamp, and reading it is what showed the median eviction
    happens 22 SECONDS after entry.
    """
    direct = float(row.get("held_sec") or 0.0)
    if direct > 0:
        return direct
    entry_ts = float(row.get("released_entry_ts") or 0.0)
    ts = float(row.get("ts") or 0.0)
    return (ts - entry_ts) if (entry_ts > 0 and ts > entry_ts) else 0.0


def _tradeable() -> Optional[Any]:
    """The live lane's own predicate, or None if it cannot be imported.

    Never fall back to "everything counts": the pooled and tradeable books
    differ by a factor of ten and by SIGN, and a tool that quietly reports the
    pooled number as tradeable is how two strategies were once called ready.
    """
    try:
        import scripts.tradeable_book as tb

        return tb._tradeable_predicate()
    except Exception:
        return None


def destroyed_evidence(
    *,
    days: float = 7.0,
    db_path: Optional[Path] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Count what the ghost lane booked against what it threw away."""
    now = time.time() if now is None else float(now)
    since = now - days * 86400.0
    con = sqlite3.connect(str(Path(db_path or DEFAULT_DB)))
    try:
        entries = sum(len(_rows(con, s, since)) for s in ENTERED)
        booked = sum(len(_rows(con, s, since)) for s in BOOKED)
        fates = {s: _rows(con, s, since) for s in DESTROYED}
    finally:
        con.close()

    is_tradeable = _tradeable()
    destroyed_n = sum(len(v) for v in fates.values())
    out: Dict[str, Any] = {
        "days": days,
        "ghost_entries": entries,
        "booked": booked,
        "destroyed": destroyed_n,
        "booked_share": (booked / entries) if entries else 0.0,
        "tradeable_predicate": is_tradeable is not None,
        "fates": {},
    }

    for status, rows in fates.items():
        held = [h for h in (_held_secs(r) for r in rows) if h > 0]
        silent = [float(r.get("silent_sec") or 0.0) for r in rows if r.get("silent_sec")]
        tradeable_n = (
            sum(1 for r in rows if is_tradeable(str(r.get("symbol") or "")))
            if is_tradeable is not None
            else None
        )
        out["fates"][status] = {
            "why": DESTROYED[status],
            "n": len(rows),
            "tradeable": tradeable_n,
            "per_day": len(rows) / days if days else 0.0,
            "tradeable_per_day": (tradeable_n / days) if (tradeable_n is not None and days) else None,
            "held_median_secs": statistics.median(held) if held else None,
            "held_max_secs": max(held) if held else None,
            "held_median_x_stale": (statistics.median(held) / STALE_EXIT_SECS) if held else None,
            "held_max_x_stale": (max(held) / STALE_EXIT_SECS) if held else None,
            "over_4x_stale": sum(1 for h in held if h > 4 * STALE_EXIT_SECS),
            # The discriminator between the two destroyed populations. A position
            # taken away INSIDE the horizon never asked to exit -- booking it
            # writes a round trip the strategy did not make. One past the horizon
            # was owed an exit no rule could reach.
            "inside_horizon": sum(1 for h in held if h < STALE_EXIT_SECS),
            "with_held_secs": len(held),
            # Everything needed to book the round trip, present and unused.
            "bookable": sum(
                1
                for r in rows
                if float(r.get("released_entry_price") or 0.0) > 0
                and float(r.get("released_size") or 0.0) > 0
            ),
            "with_strategy_id": sum(1 for r in rows if r.get("released_strategy_id")),
            "silent_median_secs": statistics.median(silent) if silent else None,
        }
    return out


def _fmt(v: Any, spec: str = "%.1f") -> str:
    return "n/a" if v is None else (spec % v if isinstance(v, float) else str(v))


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    r = destroyed_evidence(days=args.days, db_path=args.db)
    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    print("=" * 74)
    print("GHOST EVIDENCE FUNNEL -- what the lane booked vs what it destroyed")
    print("=" * 74)
    print("  window            %.1f days" % r["days"])
    print("  ghost entries     %d" % r["ghost_entries"])
    print("  BOOKED (ghost-exit)  %d   %.1f%% of entries   <- all graduation reads"
          % (r["booked"], 100.0 * r["booked_share"]))
    print("  DESTROYED            %d" % r["destroyed"])
    if not r["tradeable_predicate"]:
        print("  WARNING: the live lane's tradeable predicate would not import;")
        print("           tradeable counts are unjudgeable and reported as n/a.")
    print()
    for status, f in r["fates"].items():
        print("  %s -- %s" % (status, f["why"]))
        print("      n %d  (%.1f/day)   TRADEABLE %s  (%s/day)"
              % (f["n"], f["per_day"], _fmt(f["tradeable"]),
                 _fmt(f["tradeable_per_day"], "%.1f")))
        print("      held: median %s x stale_exit_secs, max %s x   [%d of %d over 4x]"
              % (_fmt(f["held_median_x_stale"]), _fmt(f["held_max_x_stale"]),
                 f["over_4x_stale"], f["with_held_secs"]))
        if f["inside_horizon"]:
            print("      %d of %d (%.1f%%) were taken away INSIDE the 900s horizon --"
                  % (f["inside_horizon"], f["with_held_secs"],
                     100.0 * f["inside_horizon"] / f["with_held_secs"]))
            print("      those never asked to exit; booking them writes a trip the "
                  "strategy did not make")
        print("      bookable now (entry price AND size present): %d of %d, %d with a strategy_id"
              % (f["bookable"], f["n"], f["with_strategy_id"]))
        if f["silent_median_secs"]:
            print("      last price is stale by a median of %.0fs -- do NOT book a fill at it"
                  % f["silent_median_secs"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
