"""The live-tradeable ghost book is profitable inside 15 minutes and only there.

WHY THIS EXISTS
---------------
Pass 100 retired cost, stop width, clip size and feed regime as explanations for
the live-tradeable ghost book's negative sign (see scripts/stop_width_replay.py
and data/attempts-revenir.md): the book is 0.6331pp short of break-even GROSS at
an infinite clip, and every stop width from 0.25% to 3.00% is worse than what it
booked. That left DIRECTION -- and this is where the direction goes wrong.

Splitting the same population by how long each round trip was actually HELD,
using the tick path recovered by ``stop_width_replay.locate_path``:

    hold time        n    gross$   share of loss  win%   median ticks/min
    under 5 min     16   +0.1903        0.0%      56.2         0.75
    5-15 min         8   -0.1307       11.1%      25.0         0.60
    15-60 min       30   -0.3391       28.7%      40.0         0.45
    1-4 hours       31   -0.2164       18.3%      38.7         0.28
    over 4 hours     3   -0.2012       17.0%       0.0         0.14

    HELD <= 15 min   17 trips   gross +0.0056   +0.0158% of notional   win 58.8%
    HELD  > 15 min   64 trips   gross -0.7566   -0.4350% of notional   win 37.5%

``stale_exit_secs`` is 900s. 64 of 81 placeable trips -- 79% -- outlive it, and
they carry the ENTIRE loss. Inside the horizon this loop says it trades, the
book is positive.

THE MECHANISM, and it is not a threshold that needs widening. Exits are
evaluated only when a TICK ARRIVES: trading/bot.py dispatches the protective
bracket from the sample-handling path, which is why 72 of 73 samples on the live
BSTONK position evaluated no trigger at all (bot.py:6855-6880) and why this repo
already records dark-feed positions as immortal. So ``stale_exit_secs`` is a
WALL-CLOCK promise enforced on a TICK-DRIVEN schedule, and the median tick rate
falls monotonically with hold time in the table above -- 0.75/min under five
minutes down to 0.14/min past four hours. The worst case measured is CRV-USDC:
held 1064.8 minutes (17.7 hours) on 13 ticks, one tick every 82 minutes, closed
by ``stop_loss:-0.0272`` in the same second as two other positions.

READ THE SELECTION EFFECT HONESTLY. Part of this split is definitional: a trip
cannot exit by ``timed-exit`` before 900s, so the slow bucket is enriched in
trades that went nowhere, and a trip that closed fast did so partly BECAUSE a
tick arrived to close it. This table therefore does NOT prove that forcing an
exit at 15 minutes would earn +0.0056. What it does establish, and what does not
depend on the selection, is that the tick rate and the hold time are coupled and
that the losing population is the one the exit rules could not reach on time.

Run:  python -X utf8 scripts/hold_time_edge.py
      python -X utf8 scripts/hold_time_edge.py --days 5 --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "storage" / "trading_cache.db"

# ``stale_exit_secs`` in trading/bot.py, in minutes. The line this whole report
# is drawn either side of.
STALE_EXIT_MINS = 15.0

BUCKETS: Sequence[Tuple[float, float, str]] = (
    (0.0, 5.0, "under 5 min"),
    (5.0, 15.0, "5-15 min"),
    (15.0, 60.0, "15-60 min"),
    (60.0, 240.0, "1-4 hours"),
    (240.0, float("inf"), "over 4 hours"),
)


def _acc(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    gross = sum(float(r.get("cgross", 0.0) or 0.0) for r in rows)
    notional = sum(float(r.get("notional", 0.0) or 0.0) for r in rows)
    wins = sum(1 for r in rows if float(r.get("cgross", 0.0) or 0.0) > 0)
    moved = [r for r in rows if float(r.get("held_mins", 0.0) or 0.0) > 0]
    return {
        "trips": n,
        "gross": gross,
        "notional": notional,
        "gross_pct": (100.0 * gross / notional) if notional > 0 else 0.0,
        "win_rate": (100.0 * wins / n) if n else 0.0,
        "median_ticks": statistics.median([r["ticks"] for r in rows]) if n else 0.0,
        "median_tick_rate": statistics.median(
            [r["ticks"] / r["held_mins"] for r in moved]) if moved else 0.0,
    }


def hold_time_edge(
    *,
    days: float = 7.0,
    db_path: Optional[Path] = None,
    now: Optional[float] = None,
    rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The tradeable book split by how long each round trip was held."""
    import scripts.stop_width_replay as sw
    import scripts.tradeable_book as tb

    now = time.time() if now is None else float(now)
    db = Path(db_path or DEFAULT_DB)

    if rows is None:
        is_tradeable = tb._tradeable_predicate()
        if is_tradeable is None:
            return {"error": "cannot import stop_is_unenforceable; the "
                             "tradeable population is unjudgeable"}
        rows = []
        for r in tb.load_rows(db, now - days * 86400.0):
            if str(r.get("mode", "")).lower() == "live":
                continue
            if not is_tradeable(r["symbol"]):
                continue
            if r.get("strategy_id") in ("", "unclassified"):
                continue
            g = tb.clamped_gross(r)["gross"]
            notional = float(r.get("notional", 0.0) or 0.0)
            if notional <= 0 or abs(g) > 0.5 * notional:
                continue
            r = dict(r)
            r["cgross"] = g
            rows.append(r)

        con = sqlite3.connect(str(db))
        try:
            placed = []
            for r in rows:
                p = sw.locate_path(con, r)
                if p is None:
                    continue
                r["held_mins"] = (r["ts"] - p[0]["ts"]) / 60.0 if p else 0.0
                r["ticks"] = len(p)
                placed.append(r)
        finally:
            con.close()
        unplaceable = len(rows) - len(placed)
        rows = placed
    else:
        unplaceable = 0

    buckets = []
    total_loss = sum(r["cgross"] for r in rows if r["cgross"] < 0)
    for lo, hi, label in BUCKETS:
        v = [r for r in rows if lo <= float(r.get("held_mins", 0.0)) < hi]
        if not v:
            continue
        a = _acc(v)
        a["label"] = label
        a["loss_share"] = (100.0 * min(a["gross"], 0.0) / total_loss + 0.0
                           if total_loss < 0 else 0.0)
        buckets.append(a)

    # The split the loop's own horizon rule draws. Trips with a zero-length path
    # (the entry was the last tick before the exit) are excluded from BOTH sides
    # rather than counted as instant: their hold time is unobserved, not zero.
    timed = [r for r in rows if float(r.get("held_mins", 0.0)) > 0]
    fast = _acc([r for r in timed if r["held_mins"] <= STALE_EXIT_MINS])
    slow = _acc([r for r in timed if r["held_mins"] > STALE_EXIT_MINS])

    longest = max(rows, key=lambda r: r.get("held_mins", 0.0)) if rows else None
    return {
        "days": days,
        "placed": len(rows),
        "unplaceable": unplaceable,
        "stale_exit_mins": STALE_EXIT_MINS,
        "buckets": buckets,
        "within_horizon": fast,
        "outlived_horizon": slow,
        "outlived_pct": (100.0 * slow["trips"] / len(timed)) if timed else 0.0,
        "longest": ({"symbol": longest["symbol"],
                     "held_mins": longest["held_mins"],
                     "ticks": longest["ticks"]} if longest else None),
    }


def render(rep: Dict[str, Any]) -> str:
    if rep.get("error"):
        return f"UNJUDGEABLE: {rep['error']}"
    out = ["=" * 88,
           "HOLD TIME EDGE -- the live-tradeable ghost book by how long it held",
           "=" * 88,
           f"window {rep['days']}d   {rep['placed']} placeable round trips   "
           f"{rep['unplaceable']} unplaceable (excluded, not assumed)",
           "",
           f"  {'hold time':16s} {'n':>3s} {'gross$':>9s} {'loss share':>11s} "
           f"{'win%':>6s} {'ticks/min':>10s}"]
    for b in rep["buckets"]:
        out.append(f"  {b['label']:16s} {b['trips']:3d} {b['gross']:+9.4f} "
                   f"{b['loss_share']:10.1f}% {b['win_rate']:6.1f} "
                   f"{b['median_tick_rate']:10.2f}")
    f, s = rep["within_horizon"], rep["outlived_horizon"]
    out += ["",
            f"  HELD <= {rep['stale_exit_mins']:.0f} min  {f['trips']:3d} trips  "
            f"gross {f['gross']:+8.4f}  {f['gross_pct']:+7.4f}% of notional  "
            f"win {f['win_rate']:4.1f}%",
            f"  HELD  > {rep['stale_exit_mins']:.0f} min  {s['trips']:3d} trips  "
            f"gross {s['gross']:+8.4f}  {s['gross_pct']:+7.4f}% of notional  "
            f"win {s['win_rate']:4.1f}%",
            "",
            f"  stale_exit_secs is {rep['stale_exit_mins'] * 60:.0f}s. "
            f"{rep['outlived_pct']:.0f}% of trips outlive it, and they carry the "
            f"whole loss."]
    if rep.get("longest"):
        lg = rep["longest"]
        out.append(f"  Longest hold: {lg['held_mins']:.1f} min on {lg['symbol']} "
                   f"with {lg['ticks']} ticks -- one every "
                   f"{lg['held_mins'] / max(lg['ticks'], 1):.0f} min.")
    out += ["",
            "  Exits are evaluated only when a tick arrives, so stale_exit_secs is a",
            "  WALL-CLOCK promise on a TICK-DRIVEN schedule. See the module docstring",
            "  for the selection effect this table does and does not survive.",
            "=" * 88]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rep = hold_time_edge(days=a.days, db_path=a.db)
    print(json.dumps(rep, indent=2, default=str) if a.json else render(rep))
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
