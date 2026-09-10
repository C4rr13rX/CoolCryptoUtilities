"""What the live-tradeable ghost book would have paid at a different stop width.

WHY THIS EXISTS
---------------
Item 4af9a51f asks whether the live-tradeable ghost book's negative sign is a
DIRECTION problem or a COST problem. Measured 2026-09-10 over the 7-day
plausible, attributed, live-tradeable population (88 round trips, gross
-0.7043 on 224.04 of notional), the loss is not spread across the book. It is
concentrated in one exit family:

    family                 n    gross$   share of all loss   mean ret%
    stop_loss              8   -0.4655         39.4%          -3.7721
    timed-exit            22   -0.1716         14.5%          -0.1795
    negative_margin        9   -0.1549         13.1%          -0.1937
    confidence_drop       31   -0.0673          5.7%          +0.1352
    ... every other family together              +0.23

Eight round trips -- 9% of the book -- carry 66% of its total gross loss, at a
mean of -3.77% against a winner's mean of +1.04%. That is not a cost problem:
0.3187% of notional cannot explain a -3.77% mean. It is the stop.

Two readings of that were already available and BOTH are refused by the numbers
here, which is why this replays the path rather than arguing from the exit rows:

  * "the stops overshoot their level" -- they barely do. The exit reason records
    the REALISED loss (trading/bot.py:7372 writes `stop_loss:{pnl_pct_held}`),
    and against the configured widths the total recoverable overshoot across all
    eight rows is a few cents of a 0.70 loss. Overshoot is real (see
    services/stop_survivability_gate.py) and it is not the money here.
  * "so tighten the stop" -- unmeasurable from the exit rows alone. A tighter
    stop truncates the losses that breached it AND converts winners that dipped
    through it into losers. You cannot know which without the PATH.

WHAT THIS DOES
--------------
Recovers the path each round trip actually took, from ``market_stream``, and
re-runs the stop against it.

The entry timestamp is not stored on ``trade_outcomes`` -- the row is
{ts, entry_price, exit_price, quantity, ...} and ``ts`` is the EXIT. But the
entry price came from a tick, so the entry is recoverable: the latest tick at or
before the exit whose price matches ``entry_price`` to floating-point tolerance.
Trips with no such tick are SKIPPED AND COUNTED, never assumed -- a replay that
quietly dropped the trips it could not place would report a book made of the
paths that happened to match.

THE COUNTERFACTUAL IS DELIBERATELY ONE-SIDED. For a candidate width ``w`` the
replay exits at the FIRST tick on the path that is at or below ``-w`` from
entry, at THAT TICK'S PRICE -- a stop is a market order and fills where the
price is, not where the level was, which is the correction commit 1135a79 made
for the take-profit clamp and the same rule applies here in reverse. If no tick
on the path breaches ``-w``, the trip keeps the outcome it actually had. So a
tighter width can only ever PRE-EMPT an exit, never extend one, and every
non-stop exit rule stays exactly where it fired. This understates what a tighter
stop would do to the winners' side (a trip whose recorded exit came before its
own path is fully sampled is left alone) and overstates nothing.

WHAT IT CANNOT DO. It cannot re-enter. A trip stopped out early under a tighter
width does not get to take the next signal, so the replay charges the full cost
of every additional stop and credits none of the re-entries a real book would
have taken. Read the result as a FLOOR on a tighter stop, not an estimate.

Run:  python -X utf8 scripts/stop_width_replay.py
      python -X utf8 scripts/stop_width_replay.py --days 5 --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "storage" / "trading_cache.db"

# Candidate stop widths, as a fraction of the entry price. 0.02 is
# GHOST_STOP_LOSS_PCT, so it is the "as configured" column rather than a
# proposal; the tighter ones are the question.
DEFAULT_WIDTHS: Sequence[float] = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)

# A tick matches the recorded entry price when it is within this relative
# distance. The entry price IS a tick price that made a float round trip
# through sqlite, so the tolerance only has to absorb representation, not
# movement -- anything looser would match a genuinely different tick and place
# the entry at the wrong point on the path.
PRICE_TOL = 1e-9

# How far back to look for the entry tick. A round trip that this loop intends
# to resolve in minutes cannot have entered more than a day before it exited,
# and widening this only admits false matches.
MAX_HOLD_SECS = 86400.0


def _ticks(con: sqlite3.Connection, symbol: str, lo: float, hi: float
           ) -> List[Dict[str, float]]:
    return [{"ts": float(r[0]), "price": float(r[1])} for r in con.execute(
        "SELECT ts, price FROM market_stream WHERE symbol = ? AND price > 0 "
        "AND ts >= ? AND ts <= ? ORDER BY ts", (symbol, lo, hi))]


def locate_path(con: sqlite3.Connection, row: Dict[str, Any]
                ) -> Optional[List[Dict[str, float]]]:
    """The ticks this round trip lived through, or None when unplaceable.

    None rather than an empty list is deliberate: "I could not find the entry"
    and "the entry was the last tick before the exit" are different facts, and
    collapsing them would let an unplaceable trip count as a trip no stop could
    have touched.
    """
    exit_ts = float(row.get("ts", 0.0) or 0.0)
    entry = float(row.get("entry_price", 0.0) or 0.0)
    if exit_ts <= 0 or entry <= 0:
        return None
    window = _ticks(con, str(row.get("symbol", "")), exit_ts - MAX_HOLD_SECS,
                    exit_ts)
    entry_i = None
    for i in range(len(window) - 1, -1, -1):
        if abs(window[i]["price"] - entry) <= PRICE_TOL * max(entry, 1e-18):
            entry_i = i
            break
    if entry_i is None:
        return None
    return window[entry_i + 1:]


def replay_row(row: Dict[str, Any], path: Sequence[Dict[str, float]],
               width: float) -> Dict[str, Any]:
    """This trip's gross under a stop of ``width``, and whether it changed.

    ``booked`` is the outcome the trip actually had, already de-contaminated by
    ``tradeable_book.clamped_gross`` before it reaches here.
    """
    booked = float(row.get("cgross", row.get("gross", 0.0)) or 0.0)
    entry = float(row.get("entry_price", 0.0) or 0.0)
    qty = float(row.get("quantity", 0.0) or 0.0)
    out = {"gross": booked, "stopped": False, "ret": 0.0}
    if entry <= 0 or qty <= 0 or width <= 0:
        return out
    for t in path:
        if t["price"] / entry - 1.0 <= -width:
            # Fill at the tick, not at the level: a stop is a market order.
            out["gross"] = (t["price"] - entry) * qty
            out["stopped"] = True
            out["ret"] = t["price"] / entry - 1.0
            return out
    return out


def _book(rows: Sequence[Dict[str, Any]], key: str = "cgross") -> Dict[str, Any]:
    n = len(rows)
    gross = sum(float(r.get(key, 0.0) or 0.0) for r in rows)
    notional = sum(float(r.get("notional", 0.0) or 0.0) for r in rows)
    wins = sum(1 for r in rows if float(r.get(key, 0.0) or 0.0) > 0)
    return {
        "trips": n,
        "gross": gross,
        "notional": notional,
        "gross_pct": (100.0 * gross / notional) if notional > 0 else 0.0,
        "win_rate": (100.0 * wins / n) if n else 0.0,
    }


def replay(
    *,
    days: float = 7.0,
    db_path: Optional[Path] = None,
    now: Optional[float] = None,
    widths: Sequence[float] = DEFAULT_WIDTHS,
    rows: Optional[List[Dict[str, Any]]] = None,
    paths: Optional[Dict[int, List[Dict[str, float]]]] = None,
) -> Dict[str, Any]:
    """The live-tradeable book's gross at each candidate stop width."""
    import scripts.tradeable_book as tb

    now = time.time() if now is None else float(now)
    db = Path(db_path or DEFAULT_DB)

    if rows is None:
        is_tradeable = tb._tradeable_predicate()
        if is_tradeable is None:
            return {"error": "cannot import stop_is_unenforceable; the "
                             "tradeable population is unjudgeable"}
        raw = tb.load_rows(db, now - days * 86400.0)
        rows = []
        for r in raw:
            if str(r.get("mode", "")).lower() == "live":
                continue
            if not is_tradeable(r["symbol"]):
                continue
            # The item's population: attributed and plausible. An unattributed
            # row cannot be credited to a strategy the bar reads, and a row
            # whose |gross| exceeds half its notional is a repricing artifact
            # (see aero-profit-is-one-implausible-row).
            if r.get("strategy_id") in ("", "unclassified"):
                continue
            g = tb.clamped_gross(r)["gross"]
            notional = float(r.get("notional", 0.0) or 0.0)
            if notional <= 0 or abs(g) > 0.5 * notional:
                continue
            r = dict(r)
            r["cgross"] = g
            rows.append(r)

    if paths is None:
        con = sqlite3.connect(str(db))
        try:
            paths = {}
            for i, r in enumerate(rows):
                p = locate_path(con, r)
                if p is not None:
                    paths[i] = p
        finally:
            con.close()

    placed = [i for i in range(len(rows)) if i in paths]
    unplaceable = len(rows) - len(placed)

    baseline = _book(rows)
    baseline_placed = _book([rows[i] for i in placed])

    cols: List[Dict[str, Any]] = []
    for w in widths:
        sim: List[Dict[str, Any]] = []
        changed = 0
        for i in placed:
            res = replay_row(rows[i], paths[i], w)
            if res["stopped"]:
                changed += 1
            d = dict(rows[i])
            d["cgross"] = res["gross"]
            sim.append(d)
        b = _book(sim)
        b["width"] = w
        b["stopped"] = changed
        b["delta_gross"] = b["gross"] - baseline_placed["gross"]
        cols.append(b)

    return {
        "days": days,
        "widths": list(widths),
        "rows": len(rows),
        "placed": len(placed),
        "unplaceable": unplaceable,
        "baseline_all": baseline,
        "baseline_placed": baseline_placed,
        "columns": cols,
    }


def render(rep: Dict[str, Any]) -> str:
    if rep.get("error"):
        return f"UNJUDGEABLE: {rep['error']}"
    out = ["=" * 88,
           "STOP WIDTH REPLAY -- the live-tradeable ghost book against its own ticks",
           "=" * 88,
           f"window {rep['days']}d   population {rep['rows']} plausible attributed "
           f"tradeable round trips"]
    b, bp = rep["baseline_all"], rep["baseline_placed"]
    out.append(f"  path recovered for {rep['placed']}, UNPLACEABLE {rep['unplaceable']} "
               f"(no tick matches the recorded entry price; excluded, not assumed)")
    out.append("")
    out.append(f"  AS BOOKED, whole population   {b['trips']:3d} trips  "
               f"gross {b['gross']:+8.4f}  {b['gross_pct']:+7.4f}% of notional  "
               f"win {b['win_rate']:4.1f}%")
    out.append(f"  AS BOOKED, placed subset      {bp['trips']:3d} trips  "
               f"gross {bp['gross']:+8.4f}  {bp['gross_pct']:+7.4f}% of notional  "
               f"win {bp['win_rate']:4.1f}%")
    out.append("")
    out.append("  Every column below is the SAME trips, re-run against the ticks they")
    out.append("  actually lived through, changing ONLY the stop. No re-entry is")
    out.append("  credited, so each is a FLOOR on that width, not an estimate.")
    out.append("")
    out.append(f"  {'stop':>7s} {'stopped':>8s} {'gross$':>9s} {'vs booked':>10s} "
               f"{'% notional':>11s} {'win%':>6s}")
    for c in rep["columns"]:
        out.append(f"  {100 * c['width']:6.2f}% {c['stopped']:8d} {c['gross']:+9.4f} "
                   f"{c['delta_gross']:+10.4f} {c['gross_pct']:+11.4f} "
                   f"{c['win_rate']:6.1f}")
    out.append("=" * 88)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rep = replay(days=a.days, db_path=a.db)
    print(json.dumps(rep, indent=2) if a.json else render(rep))
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
