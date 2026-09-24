"""Independent cross-verification of p691's 'the toll is 21.6x the peak excursion'.

Bay p691 measured, over the 35 timed-exit rows of the 53 ghost closures in 7d:
median peak favourable excursion +0.0215% against a median toll of 0.4653%,
ratio 21.6x, and a perfect exit oracle still losing on 34 of 35.

This re-derives the excursion FROM market_stream rather than re-reading the
report. It reproduces Bay's join (trade_outcomes -> trading_ops action='enter'
on symbol + entry_price at 1e-6), then pulls every tick between entry and exit
and computes the max favourable move directly.

Usage: python scripts/dune_excursion_verify.py [--days 7]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import time
from typing import List, Sequence

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")


def _q(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[int(pos)] if lo == hi else s[lo] + (s[hi] - s[lo]) * (pos - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--exit-reason", default="timed-exit")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trade_outcomes)")}
    print(f"trade_outcomes columns: {sorted(cols)}")
    cutoff = time.time() - args.days * 86400.0

    ts_col = "exit_ts" if "exit_ts" in cols else ("ts" if "ts" in cols else "timestamp")
    # There is NO exit_reason column -- the reason lives inside the details JSON.
    # Filtering on it matters: the 18 non-timed-exit rows carry nearly all the
    # large excursions, so the unfiltered cohort reads 2.4x too generous.
    reason_col = "details" if "details" in cols else None
    print(f"using exit-time column {ts_col!r}, reason read from {reason_col!r} JSON")

    sql = (f"SELECT symbol, entry_price, exit_price, {ts_col}"
           + (f", {reason_col}" if reason_col else "")
           + (", fee_cost" if "fee_cost" in cols else "")
           + " FROM trade_outcomes WHERE status='closed' AND wallet='ghost'"
           f" AND {ts_col}>=? AND entry_price>0 AND exit_price>0")
    rows = conn.execute(sql, (cutoff,)).fetchall()
    print(f"closed ghost rows in {args.days:g}d: {len(rows)}")
    if reason_col and args.exit_reason:
        def _reason(blob):
            try:
                return json.loads(blob).get("reason")
            except Exception:
                return None
        kept = [r for r in rows if _reason(r[4]) == args.exit_reason]
        print(f"after reason={args.exit_reason!r} filter: {len(kept)} of {len(rows)}")
        if not kept:
            print("  filter matched nothing -- falling back to the full cohort, "
                  "which is NOT the cohort the p691 claim was made on")
        else:
            rows = kept

    joined: List[dict] = []
    misses = multi = 0
    for r in rows:
        sym, entry, exit_px, t_exit = r[0], float(r[1]), float(r[2]), float(r[3])
        # Bay's join: trading_ops action='enter' on symbol + entry_price at 1e-6
        hits = conn.execute(
            "SELECT ts FROM trading_ops WHERE action='enter' AND symbol=? "
            "AND ts<=? AND ts>=? ORDER BY ts",
            (sym, t_exit, t_exit - 86400.0)).fetchall()
        cand = []
        for (t_en,) in hits:
            d = conn.execute(
                "SELECT details FROM trading_ops WHERE action='enter' AND symbol=? AND ts=?",
                (sym, t_en)).fetchone()
            if d and d[0] and f"{entry:.6f}"[:8] in str(d[0]):
                cand.append(t_en)
        if not cand:
            misses += 1
            continue
        if len(cand) > 1:
            multi += 1
        t_entry = cand[-1]
        joined.append({"symbol": sym, "entry": entry, "exit": exit_px,
                       "t_entry": t_entry, "t_exit": t_exit})

    print(f"join: {len(joined)} matched, {misses} missed, {multi} multi")
    if not joined:
        print("NO JOIN -- cannot verify. Reporting that rather than estimating it.")
        return 2

    exc: List[float] = []
    holds: List[float] = []
    no_tape = 0
    for j in joined:
        ticks = conn.execute(
            "SELECT price FROM market_stream WHERE symbol=? AND ts>=? AND ts<=? AND price>0",
            (j["symbol"], j["t_entry"], j["t_exit"])).fetchall()
        holds.append(j["t_exit"] - j["t_entry"])
        if not ticks:
            no_tape += 1
            continue
        hi = max(float(p) for (p,) in ticks)
        exc.append(100.0 * (hi / j["entry"] - 1.0))  # percent, long-side favourable

    conn.close()
    print(f"rows with NO tape inside the hold: {no_tape}")
    if not exc:
        print("no excursions computable")
        return 2
    print(f"\nTRUE HOLD   median {_q(holds,0.5):.0f}s ({_q(holds,0.5)/60:.1f} min)"
          f"  min {min(holds):.0f}s  max {max(holds):.0f}s")
    print(f"PEAK EXCURSION (tape max vs entry, n={len(exc)})")
    print(f"    median {_q(exc,0.5):+.4f}%   p25 {_q(exc,0.25):+.4f}%"
          f"   p75 {_q(exc,0.75):+.4f}%   max {max(exc):+.4f}%")
    toll = 0.4653
    print(f"    vs the claimed toll {toll:.4f}%  ->  ratio {toll/_q(exc,0.5):.1f}x"
          if _q(exc, 0.5) > 0 else "    median excursion is <=0")
    cleared = sum(1 for x in exc if x > toll)
    print(f"    rows whose PEAK cleared the toll: {cleared}/{len(exc)}")
    print(f"\nBAY p691 CLAIMED: median +0.0215%, ratio 21.6x, 1 of 35 cleared the toll")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
