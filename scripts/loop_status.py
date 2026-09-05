"""What the trading pipeline is actually doing, right now.

Injected into every refinement pass so the agent starts from measured facts
rather than from the last thing it read. The numbers here are the ones that
decide whether a pass succeeded: live trades today, whether positions close,
whether the feed ticks, and whether the processes that have to be up are up.

Money is read from ``trade_outcomes`` and the strategy registry, never from
``trading_ops``. That table is an append-only event log which keeps pre-fix
artifacts forever -- one 2026-09-04 row carries a -0.4177 that the receipt
puts at -0.0169, and summing it reported a losing system that was actually
up +0.14.

Usage:
    python scripts/loop_status.py          human-readable
    python scripts/loop_status.py --json   metrics for the gate
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "storage" / "trading_cache.db"


def _conn() -> Optional[sqlite3.Connection]:
    try:
        return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001
        return None


def _count(conn, sql: str, args: tuple = ()) -> int:
    try:
        row = conn.execute(sql, args).fetchone()
        return int(row[0] or 0) if row else 0
    except Exception:  # noqa: BLE001
        return 0


def _processes() -> Dict[str, bool]:
    """Which of the always-on processes are actually alive.

    A dark pipeline is the highest-priority bug there is, so this is checked
    first and reported first.
    """
    # Matched against the command line, so these are the strings that
    # actually appear there. "wizard" does not: the brain substrate is
    # w1z4rd_node.exe, and looking for the English spelling reported it DOWN
    # while it was running.
    wanted = {
        "production": "start_production",
        "agent_worker": "tradingagent_worker",
        "web": "run_waitress",
        "wizard_node": "w1z4rd_node",
    }
    found = {name: False for name in wanted}
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=60).stdout or ""
    except Exception:  # noqa: BLE001 - cannot check is not the same as down
        return {name: None for name in wanted}  # type: ignore[misc]
    for name, needle in wanted.items():
        found[name] = needle.lower() in out.lower()
    return found


def collect() -> Dict[str, Any]:
    now = time.time()
    out: Dict[str, Any] = {"now": now}
    conn = _conn()
    if conn is None:
        out["error"] = "no database"
        return out

    midnight = now - (now % 86400)

    # THE NUMBER THAT MATTERS. Live round trips settled today.
    out["live_trades_today"] = _count(
        conn, "SELECT COUNT(*) FROM trade_outcomes WHERE wallet='live' "
              "AND status='closed' AND ts >= ?", (midnight,))
    out["live_trades_1h"] = _count(
        conn, "SELECT COUNT(*) FROM trade_outcomes WHERE wallet='live' "
              "AND status='closed' AND ts >= ?", (now - 3600,))
    out["live_swaps_1h"] = _count(
        conn, "SELECT COUNT(*) FROM trading_ops WHERE status='live-swap-settled' "
              "AND ts >= ?", (now - 3600,))

    try:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(net_profit),0) FROM trade_outcomes "
            "WHERE wallet='live' AND status='closed'").fetchone()
        out["live_trades_all_time"] = int(row[0] or 0)
        out["live_net_pl"] = round(float(row[1] or 0.0), 6)
    except Exception:  # noqa: BLE001
        out["live_trades_all_time"] = 0
        out["live_net_pl"] = 0.0

    # Does anything CLOSE? An entry that never resolves is evidence that
    # never arrives, and it holds a symbol slot while it fails to arrive.
    entries = _count(conn, "SELECT COUNT(*) FROM trading_ops WHERE ts>=? "
                           "AND status LIKE '%-entry'", (now - 3600,))
    exits = _count(conn, "SELECT COUNT(*) FROM trading_ops WHERE ts>=? "
                         "AND status LIKE '%-exit'", (now - 3600,))
    out["entries_1h"] = entries
    out["exits_1h"] = exits
    out["close_rate_1h"] = round(exits / entries, 3) if entries else None

    # Is the feed alive at all?
    try:
        row = conn.execute("SELECT MAX(ts) FROM market_stream").fetchone()
        last_tick = float(row[0] or 0.0) if row else 0.0
        out["feed_age_sec"] = round(now - last_tick, 1) if last_tick else None
        out["ticks_10m"] = _count(
            conn, "SELECT COUNT(*) FROM market_stream WHERE ts>=?", (now - 600,))
        out["symbols_1h"] = _count(
            conn, "SELECT COUNT(DISTINCT symbol) FROM market_stream WHERE ts>=?",
            (now - 3600,))
    except Exception:  # noqa: BLE001
        out["feed_age_sec"] = None

    # What is held right now, from the BOOK rather than reconstructed.
    try:
        row = conn.execute(
            "SELECT value FROM kv_store WHERE key='state'").fetchone()
        positions = (json.loads(row[0]) if row and row[0] else {}).get("positions") or {}
        out["open_positions"] = len(positions)
        out["open_symbols"] = sorted(positions)[:10]
    except Exception:  # noqa: BLE001
        out["open_positions"] = None

    # Why entries are being refused -- the usual reason trading is idle.
    refusals: Dict[str, int] = {}
    try:
        for status, count in conn.execute(
                "SELECT status, COUNT(*) FROM trading_ops WHERE ts>=? AND "
                "(status LIKE '%refused%' OR status LIKE '%blocked%') "
                "GROUP BY status ORDER BY COUNT(*) DESC LIMIT 6", (now - 3600,)):
            refusals[str(status)] = int(count)
    except Exception:  # noqa: BLE001
        pass
    out["refusals_1h"] = refusals

    conn.close()
    out["processes"] = _processes()
    return out


def main() -> int:
    data = collect()

    if "--json" in sys.argv:
        print(json.dumps(data, indent=2, default=str))
        return 0

    print("R3V3N!R PIPELINE STATUS\n")

    procs = data.get("processes") or {}
    down = [name for name, up in procs.items() if up is False]
    print("processes:")
    for name, up in procs.items():
        mark = "UP  " if up else ("?   " if up is None else "DOWN")
        print(f"  {mark} {name}")
    if down:
        print(f"  ^^ {len(down)} DOWN -- a dark pipeline is the highest-priority bug")

    today = data.get("live_trades_today", 0)
    print(f"\nLIVE TRADES TODAY: {today}"
          + ("   <- ZERO. This is the failure to fix." if not today else ""))
    print(f"  last hour        : {data.get('live_trades_1h', 0)} closed, "
          f"{data.get('live_swaps_1h', 0)} swaps settled")
    print(f"  all time         : {data.get('live_trades_all_time', 0)} round trips, "
          f"net {data.get('live_net_pl', 0.0):+.6f}")

    rate = data.get("close_rate_1h")
    print(f"\npositions: {data.get('open_positions')} open"
          f"  {data.get('open_symbols') or ''}")
    print(f"  last hour  : {data.get('entries_1h', 0)} in, {data.get('exits_1h', 0)} out"
          + (f" ({rate:.0%} close)" if rate is not None else ""))

    age = data.get("feed_age_sec")
    print(f"\nfeed: {data.get('ticks_10m', 0)} ticks/10m across "
          f"{data.get('symbols_1h', 0)} symbols"
          + (f", newest {age:.0f}s ago" if age is not None else ""))

    refusals = data.get("refusals_1h") or {}
    if refusals:
        print("\nrefusals in the last hour:")
        for status, count in refusals.items():
            print(f"  {count:5d}  {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
