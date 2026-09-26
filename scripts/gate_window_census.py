"""Does the live ghost gate see the book it is judging?

[73e1c19b] criterion 2, open since p721: "atf_static shows n=144 but gate_n=1 --
the gate judged ONE trade out of 144 inside a 120h GHOST_VALIDATION_LOOKBACK_SEC."
If that holds, no strategy can ever clear an evidence bar and the graduation
funnel is structurally dead rather than merely slow.

Read-only. Calls the SAME function the gate calls -- MetricsCollector.
ghost_trade_snapshot(limit=500, lookback_sec=LOOKBACK, strategy_id=sid), which is
trading/pipeline.py:4850-4852 -- and compares its row count against the all-time
count in trade_outcomes for the same strategy. Nothing is reconstructed: a
disagreement between these two numbers is a disagreement between the gate and
the book it claims to judge.
"""

from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

DB = os.getenv("TRADING_DB_PATH", "storage/trading_cache.db")
LOOKBACK = int(os.getenv("GHOST_VALIDATION_LOOKBACK_SEC", "172800"))


def all_time_counts() -> list[tuple[str, int, float]]:
    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    try:
        # strategy is NOT a column: it lives at details->>'$.strategy_id'.
        # 216 of 312 rows carry it; the other 96 are unattributed.
        rows = con.execute(
            "SELECT json_extract(details,'$.strategy_id') AS sid, COUNT(*), MAX(ts) "
            "FROM trade_outcomes WHERE sid IS NOT NULL AND sid != '' "
            "GROUP BY sid ORDER BY COUNT(*) DESC"
        ).fetchall()
    finally:
        con.close()
    return [(str(r[0]), int(r[1]), float(r[2] or 0.0)) for r in rows]


def main() -> int:
    print("DB            : %s" % DB)
    print("LOOKBACK      : %d sec = %.1f h" % (LOOKBACK, LOOKBACK / 3600.0))

    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    try:
        now = float(con.execute("SELECT MAX(ts) FROM trade_outcomes").fetchone()[0] or 0.0)
    finally:
        con.close()
    print("newest ts     : %.1f" % now)

    from db import get_db
    from trading.metrics import MetricsCollector

    metrics = MetricsCollector(get_db())

    counts = all_time_counts()
    in_window = {}
    con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    try:
        for sid, _n, _mx in counts:
            in_window[sid] = int(
                con.execute(
                    "SELECT COUNT(*) FROM trade_outcomes "
                    "WHERE json_extract(details,'$.strategy_id') = ? AND ts >= ?",
                    (sid, now - LOOKBACK),
                ).fetchone()[0]
            )
    finally:
        con.close()

    print()
    print("%-34s %8s %8s %8s   %s" % ("strategy", "all", "in_win", "gate_n", "verdict"))
    print("-" * 88)
    total_all = total_win = total_gate = 0
    for sid, n_all, _mx in counts:
        try:
            snap = metrics.ghost_trade_snapshot(
                limit=500, lookback_sec=LOOKBACK, strategy_id=sid
            )
            gate_n = len(snap)
        except Exception as exc:  # noqa: BLE001
            print("%-34s %8d %8d %8s   RAISED %s" % (sid, n_all, in_window[sid], "-", exc))
            continue
        n_win = in_window[sid]
        flag = ""
        if n_win > 0 and gate_n < n_win:
            flag = "GATE SEES %d of %d IN WINDOW" % (gate_n, n_win)
        elif gate_n == n_win:
            flag = "agrees"
        total_all += n_all
        total_win += n_win
        total_gate += gate_n
        print("%-34s %8d %8d %8d   %s" % (sid, n_all, n_win, gate_n, flag))

    print("-" * 88)
    print("%-34s %8d %8d %8d" % ("TOTAL", total_all, total_win, total_gate))
    print()
    print("POOLED (strategy_id=None, what the old gate read):")
    pooled = metrics.ghost_trade_snapshot(limit=500, lookback_sec=LOOKBACK, strategy_id=None)
    print("  gate_n = %d" % len(pooled))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
