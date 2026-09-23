"""h* blast radius for [449305ff] criterion 6 -- which symbols a general
``h* >= 1.00`` rule would refuse, and which of those the edge gate ALREADY
refuses on an ex-post basis.

h* = 0.5 + cost / (2 * mean|move|) over the REACHABLE horizon (45 min, per
.env MAX_HOLD_FORCE_SECONDS=2700). h* >= 1.00 means a PERFECT direction call
still loses: the toll is wider than the whole move, so no signal, clip or
horizon pays. Costs are the MINIMUM of each symbol's measured clip curve --
the most generous reading, so a refusal here survives any cost dispute.

Read-only. Names the command for the item's criterion 6 blast-radius number.
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CENSUS = "data/round_trip_quote_census_20260917T190210Z.json"
HORIZON_SEC = 2700.0   # .env:169 MAX_HOLD_FORCE_SECONDS
WINDOW_SEC = 168 * 3600.0


def _db() -> sqlite3.Connection:
    path = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
    return sqlite3.connect("file:%s?mode=ro" % path, uri=True)


def _min_cost_pct(row: dict) -> float | None:
    """Cheapest measured clip on the curve, in percent."""
    pts = [p[1] for p in (row.get("points") or []) if isinstance(p, list) and len(p) == 2]
    if not pts:
        probe = row.get("probe_cost_pct")
        return float(probe) if probe is not None else None
    return float(min(pts))


def _mean_abs_move_bp(conn: sqlite3.Connection, symbol: str, now: float) -> tuple[int, float | None]:
    """Mean |return| over HORIZON_SEC, in bp, from non-overlapping windows.

    Non-overlapping because overlapping windows inflate n without adding
    independent observations -- this repo has already retracted one p-value
    for exactly that.
    """
    cur = conn.execute(
        "select ts, price from market_stream "
        "where upper(symbol) like ? and ts >= ? and price > 0 order by ts",
        (symbol.upper() + "%", now - WINDOW_SEC),
    )
    rows = cur.fetchall()
    if len(rows) < 4:
        return len(rows), None
    moves: list[float] = []
    i = 0
    while i < len(rows):
        t0, p0 = rows[i]
        j = i + 1
        while j < len(rows) and rows[j][0] - t0 < HORIZON_SEC:
            j += 1
        if j >= len(rows):
            break
        t1, p1 = rows[j]
        moves.append(abs(p1 - p0) / p0 * 10000.0)
        i = j
    if not moves:
        return len(rows), None
    return len(rows), statistics.fmean(moves)


def main() -> int:
    if not os.path.exists(CENSUS):
        print("no census at %s" % CENSUS)
        return 2
    census = json.load(open(CENSUS, encoding="utf-8"))
    quoted = census.get("quoted") or {}
    unquotable = list(census.get("unquotable") or {})

    try:
        from services.symbol_edge_gate import refusal_reason
    except Exception as exc:  # noqa: BLE001
        print("cannot import symbol_edge_gate: %s" % exc)
        refusal_reason = None  # type: ignore[assignment]

    now = time.time()
    conn = _db()
    print("h* = 0.5 + cost/(2*mean|move|), horizon %ds, window %dh, "
          "cost = MINIMUM of measured clip curve" % (HORIZON_SEC, WINDOW_SEC / 3600))
    print("h* >= 1.00 means a PERFECT direction call still loses.\n")
    hdr = "%-10s %6s %10s %12s %8s  %-9s %s"
    print(hdr % ("symbol", "rows", "cost_bp", "mean|mv|_bp", "h*", "h*rule", "edge gate today"))
    refused_by_hstar, already_banned, new_refusals = [], [], []
    for sym, row in sorted(quoted.items()):
        cost_pct = _min_cost_pct(row)
        n, move_bp = _mean_abs_move_bp(conn, sym, now)
        if cost_pct is None or move_bp is None or move_bp <= 0:
            print(hdr % (sym, n, "?" if cost_pct is None else "%.1f" % (cost_pct * 100),
                         "?", "?", "UNPRICED", "-"))
            continue
        cost_bp = cost_pct * 100.0
        hstar = 0.5 + cost_bp / (2.0 * move_bp)
        verdict = "REFUSE" if hstar >= 1.0 else "allow"
        gate = "-"
        if refusal_reason is not None:
            r = refusal_reason(sym + "-USDC", None)
            gate = "BANNED" if r else "allows"
        print(hdr % (sym, n, "%.1f" % cost_bp, "%.1f" % move_bp, "%.3f" % hstar, verdict, gate))
        if hstar >= 1.0:
            refused_by_hstar.append(sym)
            (already_banned if gate == "BANNED" else new_refusals).append(sym)

    print("\nBLAST RADIUS over the %d PRICED symbols:" % len(quoted))
    print("  h* >= 1.00 refuses      : %d  %s" % (len(refused_by_hstar), refused_by_hstar))
    print("  ...already edge-banned  : %d  %s" % (len(already_banned), already_banned))
    print("  ...NEW refusals (the    : %d  %s" % (len(new_refusals), new_refusals))
    print("     rule's whole value)")
    print("\nUNPRICED and therefore UNJUDGED by this rule: %d  %s"
          % (len(unquotable), unquotable))
    print("  'unknown' is not 'cheap' -- a rule that defaults these to ALLOWED")
    print("  reproduces the defect on every symbol nobody has measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
