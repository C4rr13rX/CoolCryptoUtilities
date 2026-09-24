"""BSTONK-USDC repeatability census -- item [efe669fc], measurement only.

Three questions, cheapest-kill-first:

  C3  are the booked BSTONK winners three independent events or one ramp
      booked three times?  entry/exit timestamps and the gaps.
  C4  is BSTONK fillable at our clip at all?  tick density and max gap.
  C2  over the full market_stream tape, how many NON-OVERLAPPING windows of
      the booked hold length moved by the booked magnitude?  That is the
      rate three trades cannot estimate.

Read-only.  No file it touches is on the trading path.

    python scripts/bstonk_repeatability_census.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
from datetime import datetime, timezone

SYMBOL = "BSTONK-USDC"
DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB)


def _strategy_index(conn: sqlite3.Connection) -> list[tuple[float, float, str]]:
    """(ts, entry_price, strategy_id) for every BSTONK enter op that names one."""
    out: list[tuple[float, float, str]] = []
    for ts, details in conn.execute(
        "SELECT ts, details FROM trading_ops WHERE symbol=? AND action='enter'",
        (SYMBOL,),
    ):
        try:
            d = json.loads(details) if details else {}
        except Exception:
            continue
        sid = d.get("strategy_id")
        px = d.get("entry_price") or d.get("target_price")
        if sid:
            out.append((ts, float(px) if px else 0.0, str(sid)))
    return out


def outcomes(conn: sqlite3.Connection) -> list[dict]:
    rows = []
    for (
        ts,
        wallet,
        status,
        entry,
        exit_,
        qty,
        gross,
        fee,
        net,
        details,
    ) in conn.execute(
        "SELECT ts, wallet, status, entry_price, exit_price, quantity, "
        "gross_profit, fee_cost, net_profit, details FROM trade_outcomes "
        "WHERE symbol=? ORDER BY ts",
        (SYMBOL,),
    ):
        try:
            d = json.loads(details) if details else {}
        except Exception:
            d = {}
        pos = d.get("position") or {}
        entry_ts = d.get("entry_ts") or pos.get("entry_ts")
        rows.append(
            {
                "exit_ts": ts,
                "entry_ts": float(entry_ts) if entry_ts else None,
                "wallet": wallet,
                "status": status,
                "mode": d.get("mode"),
                "reason": d.get("reason"),
                "entry": entry,
                "exit": exit_,
                "qty": qty,
                "gross": gross,
                "fee": fee,
                "net": net,
                "ret": (exit_ / entry - 1.0) if entry else None,
            }
        )
    return rows


def ticks(conn: sqlite3.Connection) -> list[tuple[float, float]]:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(market_stream)")]
    tcol = "ts" if "ts" in cols else cols[1]
    pcol = next((c for c in ("price", "last_price", "close") if c in cols), None)
    rows = conn.execute(
        f"SELECT {tcol}, {pcol} FROM market_stream WHERE symbol=? "
        f"AND {pcol} IS NOT NULL ORDER BY {tcol}",
        (SYMBOL,),
    ).fetchall()
    return [(float(a), float(b)) for a, b in rows if b]


def main() -> None:
    conn = _conn()
    rows = outcomes(conn)
    idx = _strategy_index(conn)

    print(f"=== {SYMBOL} -- trade_outcomes, all rows ({len(rows)}) ===")
    print(
        f"{'exit (UTC)':21s} {'wal':5s} {'status':9s} {'mode':6s} "
        f"{'entry':>11s} {'exit':>11s} {'ret%':>8s} {'net':>10s}  reason"
    )
    for r in rows:
        print(
            f"{_utc(r['exit_ts']):21s} {r['wallet']:5s} {r['status']:9s} "
            f"{str(r['mode']):6s} {r['entry']:11.8f} {r['exit']:11.8f} "
            f"{(r['ret'] or 0) * 100:8.2f} {r['net']:10.6f}  {r['reason']}"
        )

    closed = [r for r in rows if r["status"] == "closed"]
    wins = [r for r in closed if r["net"] > 0]
    print(f"\nclosed n={len(closed)}  sum(net)={sum(r['net'] for r in closed):+.6f}")
    print(f"  median net {statistics.median(r['net'] for r in closed):+.6f}")
    print(f"  winners n={len(wins)}  sum={sum(r['net'] for r in wins):+.6f}")
    top3 = sorted(closed, key=lambda r: -r["net"])[:3]
    print(f"  TOP 3 net sum={sum(r['net'] for r in top3):+.6f}  "
          f"= {sum(r['net'] for r in top3) / sum(r['net'] for r in closed) * 100:.1f}% "
          f"of the closed book")
    for r in top3:
        print(f"    {_utc(r['exit_ts'])}  net {r['net']:+.6f}  "
              f"entry {r['entry']:.8f} -> {r['exit']:.8f}  ({r['reason']})")

    # --- C3: are the winners one event? -------------------------------------
    print("\n=== C3: are the top rows independent events or ONE ramp? ===")
    ordered = sorted(top3, key=lambda r: r["exit_ts"])
    for a, b in zip(ordered, ordered[1:]):
        gap_h = (b["exit_ts"] - a["exit_ts"]) / 3600.0
        print(f"  gap {_utc(a['exit_ts'])} -> {_utc(b['exit_ts'])} = {gap_h:.2f} h")
    span_h = (ordered[-1]["exit_ts"] - ordered[0]["exit_ts"]) / 3600.0
    print(f"  span of the top 3: {span_h:.2f} h")
    first, last = closed[0], closed[-1]
    print(f"  ENTRY PRICE over the whole book: {first['entry']:.8f} "
          f"-> {last['entry']:.8f}  ({last['entry'] / first['entry']:.2f}x)")
    print(f"  book span: {(last['exit_ts'] - first['exit_ts']) / 86400.0:.2f} days")

    # strategy attribution
    print("\n  strategy attribution via trading_ops enter (nearest ts before exit):")
    from collections import Counter

    seen = Counter()
    for r in closed:
        cands = [s for t, p, s in idx if t <= r["exit_ts"]]
        seen[cands[-1] if cands else "UNATTRIBUTED"] += 1
    print(f"    {dict(seen)}")
    print(f"    distinct strategy_ids ever seen on {SYMBOL} enters: "
          f"{sorted({s for _, _, s in idx})}")

    # --- C4: is it fillable? ------------------------------------------------
    tk = ticks(conn)
    print(f"\n=== C4: tape density ({len(tk)} ticks) ===")
    if tk:
        gaps = [b[0] - a[0] for a, b in zip(tk, tk[1:])]
        print(f"  span {_utc(tk[0][0])} .. {_utc(tk[-1][0])}  "
              f"({(tk[-1][0] - tk[0][0]) / 86400.0:.2f} days)")
        print(f"  median gap {statistics.median(gaps):.1f}s   "
              f"p90 {sorted(gaps)[int(len(gaps) * 0.9)]:.1f}s   "
              f"MAX {max(gaps) / 3600.0:.2f} h")
        print(f"  price {min(p for _, p in tk):.8f} .. {max(p for _, p in tk):.8f}")
        now = datetime.now(timezone.utc).timestamp()
        print(f"  DAYS SINCE LAST TICK: {(now - tk[-1][0]) / 86400.0:.1f}")

    # --- C2: window census ---------------------------------------------------
    # trade_outcomes.details carries no entry_ts, so the hold comes from the
    # matching trading_ops exit rows, which record age_sec.
    holds = []
    for details, in conn.execute(
        "SELECT details FROM trading_ops WHERE symbol=? AND action='exit'", (SYMBOL,)
    ):
        try:
            d = json.loads(details) if details else {}
        except Exception:
            continue
        age = d.get("age_sec") or (d.get("position") or {}).get("age_sec")
        if age:
            holds.append(float(age))
    hold = statistics.median(holds) if holds else 3600.0
    print(f"\n=== C2: non-overlapping window census ===")
    print(f"  hold used: {hold / 60.0:.1f} min "
          f"({'median of ' + str(len(holds)) + ' trading_ops age_sec' if holds else 'DEFAULT 60min'})")
    if not tk:
        print("  no tape -- cannot run")
        return
    for thresh in (0.05, 0.10, 0.15, 0.20):
        n_win = 0
        n_hit = 0
        i = 0
        while i < len(tk):
            t0, p0 = tk[i]
            j = i
            while j < len(tk) and tk[j][0] - t0 < hold:
                j += 1
            if j >= len(tk):
                break
            n_win += 1
            if (tk[j][1] / p0 - 1.0) >= thresh:
                n_hit += 1
            i = j
        days = (tk[-1][0] - tk[0][0]) / 86400.0
        rate = n_hit / days if days else 0.0
        print(f"  move >= +{thresh * 100:4.1f}%: {n_hit:4d} hits of {n_win:4d} windows "
              f"= {n_hit / n_win * 100 if n_win else 0:5.2f}%   "
              f"implied {rate:.2f} hits/day over {days:.1f} days of tape")


if __name__ == "__main__":
    main()
