#!/usr/bin/env python
"""Reproduce the [91cfe040] headline: per-strategy closed book on symbols that STILL TICK.

Criterion 1 of [91cfe040] says Echo measured 44 / 5 / 0 at p700 and must not verify it.
This is the independent recompute. Read-only; touches nothing.

    strategies with any closed row  /  with a positive fresh book  /  clearing 20 & 55% & P/L>0

Freshness cut is a market_stream tick inside --days (default 2.0), the cut the item names.
Strategy attribution comes from details.strategy_id on the CLOSING row, which is what
trade_outcomes rows are (trade_id ends ':exit:N'); there is no strategy column anywhere.

    .venv/Scripts/python.exe scripts/fresh_book_census.py
    .venv/Scripts/python.exe scripts/fresh_book_census.py --days 0.25 --days 1 --days 2
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict

# sys.path[0] is scripts/ when run as a file, so `services.` is not importable and
# the censor silently reported "unavailable" on this script's first run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")


def fresh_symbols(conn: sqlite3.Connection, now: float, days: float) -> set[str]:
    cut = now - days * 86400.0
    return {r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM market_stream WHERE ts > ?", (cut,))}


def closed_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = []
    sql = ("SELECT symbol, net_profit, status, details, entry_price, exit_price, "
           "gross_profit, quantity, ts FROM trade_outcomes")
    for (symbol, net, status, details, entry, exit_px, gross, qty, ts) in conn.execute(sql):
        if status != "closed":
            continue
        try:
            d = json.loads(details or "{}")
        except (TypeError, ValueError):
            d = {}
        rows.append({
            "symbol": symbol,
            "net": float(net or 0.0),
            "strategy": d.get("strategy_id") or "unattributed",
            # the censor's own key names -- the dollar arm reads strategy_id and
            # gross_profit, the ratio arm reads entry_price/exit_price. Passing a
            # trimmed dict makes partition() a silent no-op, which is how a first
            # run of this script reported an uncensored book as censored.
            "strategy_id": d.get("strategy_id"),
            "entry_price": entry,
            "exit_price": exit_px,
            "gross_profit": gross,
            "quantity": qty,
            "ts": ts,
            "mode": d.get("mode"),
        })
    return rows


def censor(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop rows the repo's own plausibility censor rejects. Returns (kept, n_dropped)."""
    try:
        from services.outcome_plausibility import partition  # type: ignore
    except Exception:
        return rows, -1
    # NOT wrapped in try/except: a censor that fails open while the header still
    # says "censor APPLIED" is the exact failure this script exists to avoid.
    kept, dropped = partition(rows)
    return list(kept), len(list(dropped))


def census(rows: list[dict], fresh: set[str]) -> dict[str, dict]:
    book: dict[str, dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for r in rows:
        if r["symbol"] not in fresh:
            continue
        b = book[r["strategy"]]
        b["n"] += 1
        b["wins"] += 1 if r["net"] > 0 else 0
        b["pnl"] += r["net"]
    return dict(book)


def report(label: str, book: dict[str, dict]) -> None:
    any_row = len(book)
    positive = [k for k, b in book.items() if b["pnl"] > 0]
    cleared = [k for k, b in book.items()
               if b["n"] >= 20 and b["wins"] / b["n"] >= 0.55 and b["pnl"] > 0]
    print(f"\n=== {label} ===")
    print(f"  strategies with any closed FRESH row : {any_row}")
    print(f"  strategies with a POSITIVE fresh book: {len(positive)}")
    print(f"  strategies clearing 20 / 55% / P/L>0 : {len(cleared)}")
    print(f"  {'strategy':34s} {'n':>4s} {'win':>6s} {'P/L':>10s}")
    for k, b in sorted(book.items(), key=lambda kv: -kv[1]["pnl"]):
        print(f"  {k:34s} {b['n']:4d} {b['wins'] / b['n'] * 100:5.1f}% {b['pnl']:10.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, action="append", default=None)
    args = ap.parse_args()
    cuts = args.days or [2.0]

    now = time.time()
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = closed_rows(conn)
    kept, dropped = censor(rows)
    print(f"db {DB}  closed rows {len(rows)}  "
          f"censor: {'unavailable' if dropped < 0 else f'{dropped} dropped'}")
    newest = conn.execute("SELECT MAX(ts) FROM market_stream").fetchone()[0]
    print(f"newest tick age {now - float(newest):.1f}s")

    for days in cuts:
        fresh = fresh_symbols(conn, now, days)
        traded = {r["symbol"] for r in rows}
        print(f"\n--- freshness cut {days}d: {len(fresh)} symbols ticked; "
              f"{len(traded & fresh)} of {len(traded)} traded symbols are fresh ---")
        report(f"{days}d, censor APPLIED", census(kept, fresh))
        report(f"{days}d, censor NOT applied", census(rows, fresh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
