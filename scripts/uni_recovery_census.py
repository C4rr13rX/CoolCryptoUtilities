#!/usr/bin/env python
"""Item 01137ba1, acceptance criteria 4 and 5 -- the post-365beac measurement.

Criterion 4 asked for a count, after a production restart into 365beac, of
``entry-deferred-label-horizon`` rows and of chaos refusals on UNI-USDC (vs a
168/24h baseline) and WETH-USDC (vs 124).  Criterion 5 asked how many of any
recovered ticks reach ghost-entry and close as TRADEABLE round trips, with
ghost-entry not permitted to fall below 13/day against a 26/day baseline.

Both are pure reads of ``trading_ops`` and ``trade_outcomes``.  Nothing here
touches the registry or the arbitration census -- those are Echo's files this
pass.

The strategy attribution parses ``details`` as JSON and reads ``strategy_id``.
It never substring-matches the name: a LIKE over ``details`` counts
``rsi_reversal@12h`` as ``rsi_reversal`` and has produced a retracted finding
on this board twice (p693, p696).

    python scripts/uni_recovery_census.py [--hours 24]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import Counter

CHAOS_STATUS = "entry-refused-lattice"
LEAD_SYMBOLS = ("UNI-USDC", "WETH-USDC")
BASELINES = {"UNI-USDC": 168, "WETH-USDC": 124}


def _db() -> sqlite3.Connection:
    path = os.getenv("TRADING_DB_PATH", "storage/trading_cache.db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _details(row: sqlite3.Row) -> dict:
    try:
        obj = json.loads(row["details"] or "{}")
    except Exception:  # noqa: BLE001
        return {}
    return obj if isinstance(obj, dict) else {}


def _is_chaos(det: dict) -> bool:
    return str(det.get("detail") or "").strip().lower().startswith("chaos:")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()

    conn = _db()
    now = time.time()
    since = now - args.hours * 3600.0
    scale = 24.0 / args.hours  # report everything per 24h

    print("=" * 74)
    print("01137ba1 criteria 4+5 -- window %.1fh, db=%s"
          % (args.hours, os.getenv("TRADING_DB_PATH", "storage/trading_cache.db")))
    print("=" * 74)

    # ---- criterion 4a: does the 365beac mechanism emit anything at all? ----
    print("\n[4a] entry-deferred-label-horizon rows -- the 365beac signature")
    rows = conn.execute(
        "SELECT status, COUNT(*) n, MAX(ts) last FROM trading_ops "
        "WHERE status LIKE '%label-horizon%' GROUP BY status"
    ).fetchall()
    if not rows:
        print("     ALL-TIME COUNT: 0.  The status string does not exist in the store.")
    else:
        for r in rows:
            print("     %-40s n=%-6d last=%s"
                  % (r["status"], r["n"], time.strftime("%Y-%m-%d %H:%M", time.localtime(r["last"]))))

    # ---- criterion 4b: chaos refusals per lead symbol -----------------------
    print("\n[4b] chaos refusals per symbol (status=%s, detail starts 'chaos:')" % CHAOS_STATUS)
    refusals = conn.execute(
        "SELECT symbol, details FROM trading_ops WHERE ts>=? AND status=?",
        (since, CHAOS_STATUS),
    ).fetchall()
    per_symbol: Counter = Counter()
    per_sym_strat: dict = {}
    for r in refusals:
        det = _details(r)
        if not _is_chaos(det):
            continue
        sym = str(det.get("symbol") or r["symbol"] or "?")
        per_symbol[sym] += 1
        sid = str(det.get("strategy_id") or "?")
        per_sym_strat.setdefault(sym, Counter())[sid] += 1

    total_chaos = sum(per_symbol.values())
    print("     total chaos refusals in window: %d  (%.0f/24h)" % (total_chaos, total_chaos * scale))
    for sym in LEAD_SYMBOLS:
        n = per_symbol.get(sym, 0)
        base = BASELINES[sym]
        print("     %-12s %4d in window -> %6.0f/24h   baseline %d   %s"
              % (sym, n, n * scale, base,
                 "FELL" if n * scale < base else ("HELD" if n else "ZERO -- symbol absent")))
        for sid, c in (per_sym_strat.get(sym) or Counter()).most_common(6):
            print("            %-28s %d  %s" % (sid, c, "SUFFIXLESS" if "@" not in sid else ""))

    print("\n     top chaos-refused symbols in window:")
    for sym, n in per_symbol.most_common(8):
        print("       %-14s %5d" % (sym, n))

    # suffixless-vs-suffixed across the whole chaos population
    suffixless = sum(
        c for sym in per_sym_strat for sid, c in per_sym_strat[sym].items() if "@" not in sid and sid != "?"
    )
    print("\n     chaos refusals carried by a SUFFIXLESS candidate: %d of %d (%.1f%%)"
          % (suffixless, total_chaos, 100.0 * suffixless / total_chaos if total_chaos else 0.0))

    # ---- criterion 5: ghost-entry / ghost-exit / tradeable ------------------
    print("\n[5] the evidence rate -- what any recovered tick would have to become")
    for action, status, label, base in (
        ("enter", "ghost-entry", "ghost-entry", 26),
        ("exit", "ghost-exit", "ghost-exit", 25),
    ):
        n = conn.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE ts>=? AND action=? AND status=?",
            (since, action, status),
        ).fetchone()[0]
        print("     %-12s %4d in window -> %6.1f/24h   baseline %d/day   %s"
              % (label, n, n * scale, base,
                 "BELOW THE 13/day FLOOR" if label == "ghost-entry" and n * scale < 13 else "ok"))

    outc = conn.execute("SELECT COUNT(*) FROM trade_outcomes WHERE ts>=?", (since,)).fetchone()[0]
    print("     trade_outcomes rows written in window: %d -> %.1f/24h" % (outc, outc * scale))

    print("\n     the funnel above chaos, same window:")
    for r in conn.execute(
        "SELECT status, COUNT(*) n FROM trading_ops WHERE ts>=? AND action='hold' "
        "GROUP BY status ORDER BY n DESC LIMIT 8",
        (since,),
    ):
        print("       %-34s %6d" % (r["status"], r["n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
