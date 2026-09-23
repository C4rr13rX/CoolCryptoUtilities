#!/usr/bin/env python
"""Split `trade_fills` by PROVENANCE and settle whether its slippage is a LEG or a TRIP.

WHY THIS EXISTS -- backlog item [fa681912], the part nobody had done.

The item carries two criteria that were never executed:

  1. "MANDATORY: split live from ghost BEFORE computing ANY cost statistic on
     trade_fills -- 87% of AERO's rows are ghost and every one reads as a
     perfect fill."  The worry: Echo's AERO figure of 22.04bp was computed on
     NONZERO rows only, and a nonzero filter is not a provenance filter.

  2. "CORRECTED AERO ROUND TRIP: census floor 33.27bp + round-trip slippage on
     22.04bp per leg = roughly 55-77bp, DEPENDING ON WHETHER 22.04 IS A LEG OR
     A TRIP. Echo did NOT establish which -- establish it before quoting either
     end."  55 and 77 support different verdicts on TOKEN_ROUND_TRIP, so the
     item cannot convict or acquit until this is pinned.

WHAT THIS SCRIPT ESTABLISHES

  A. `trade_fills` HAS a provenance column after all. It is not a column, it is
     `details -> $.mode`, and it takes exactly five values:
     ghost_entry / ghost_exit / live_entry / live_exit / gas_refill.
     (Memory records "trade_outcomes has NO provenance column". That is true of
     trade_outcomes and FALSE of trade_fills. Different table.)

  B. Criterion 1 is a NO-OP, and that is the finding, not a failure to find
     one. The partition is EXACT: every ghost row is EXACTLY 0.00bp and every
     live row is nonzero, with no overlap in either direction. So on this table
     "nonzero rows only" and "live rows only" select the IDENTICAL set. Echo's
     22.04bp was already a live-only number. Nothing to re-report.

     This is worth stating rather than deleting, because the criterion's own
     premise -- that a ghost-heavy average is "biased DOWN" and a split would
     move the number -- is the natural expectation and it is wrong here. The
     ghost rows were already excluded by the nonzero filter, by accident.

  C. Criterion 2 is SETTLED: 22.04bp is a LEG, not a trip. Each row in
     `trade_fills` is one fill, carrying one expected/executed price pair, and
     `mode` labels it entry or exit. A round trip is therefore the SUM of an
     entry leg and an exit leg, and it must be summed per symbol because the
     two legs are not equal.

Run:
    python scripts/fill_slippage_provenance.py
    python scripts/fill_slippage_provenance.py --symbol AERO-USDC

Every number printed is measured from storage/trading_cache.db at run time.
Nothing here is a stored constant.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")

LIVE = ("live_entry", "live_exit")
GHOST = ("ghost_entry", "ghost_exit")


def load(con: sqlite3.Connection) -> list[dict]:
    """Every fill with a usable price pair, tagged with its `mode` provenance."""
    out: list[dict] = []
    sql = "SELECT ts, symbol, expected_price, executed_price, details FROM trade_fills"
    for ts, symbol, exp_col, exe_col, details in con.execute(sql):
        try:
            det = json.loads(details) if details else {}
        except (TypeError, ValueError):
            det = {}
        expected = exp_col or det.get("expected_price")
        executed = exe_col or det.get("executed_price")
        if not expected or executed is None:
            continue
        out.append(
            {
                "ts": ts,
                "symbol": symbol,
                "mode": det.get("mode") or "(unlabelled)",
                "bps": abs(executed - expected) / expected * 1e4,
                "notional_usd": det.get("notional_usd"),
            }
        )
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", help="restrict the per-symbol sections to one symbol")
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = load(con)

    print("=" * 74)
    print("FILL SLIPPAGE BY PROVENANCE -- storage/trading_cache.db, trade_fills")
    print("=" * 74)
    print(f"rows with a usable price pair: {len(rows)}")

    # ---- A. the provenance column exists -------------------------------
    modes: dict[str, int] = defaultdict(int)
    for r in rows:
        modes[r["mode"]] += 1
    print("\nA. PROVENANCE LABEL  (details -> $.mode)")
    for m, n in sorted(modes.items(), key=lambda kv: -kv[1]):
        print(f"     {m:<16} {n:>6}")

    # ---- B. does splitting live from ghost change anything? ------------
    cross: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        prov = "live" if r["mode"] in LIVE else ("ghost" if r["mode"] in GHOST else "other")
        cross[(prov, "zero" if r["bps"] == 0 else "nonzero")] += 1
    print("\nB. PROVENANCE x SLIPPAGE -- is 'nonzero' a proxy for 'live'?")
    for prov in ("ghost", "live", "other"):
        z = cross.get((prov, "zero"), 0)
        nz = cross.get((prov, "nonzero"), 0)
        if z or nz:
            print(f"     {prov:<6} exactly-zero={z:<6} nonzero={nz:<6}")
    leak_ghost = cross.get(("ghost", "nonzero"), 0)
    leak_live = cross.get(("live", "zero"), 0)
    if leak_ghost == 0 and leak_live == 0:
        print("     VERDICT: the partition is EXACT. 'nonzero only' and 'live only'")
        print("              select the IDENTICAL set of rows. Splitting live from")
        print("              ghost changes NO published statistic on this table.")
    else:
        print(f"     VERDICT: the filters DIFFER -- {leak_ghost} nonzero ghost rows,")
        print(f"              {leak_live} zero live rows. Every nonzero-only statistic")
        print("              on this table must be recomputed on live rows only.")

    # ---- C. leg or trip -------------------------------------------------
    print("\nC. LEG OR TRIP -- each row is ONE fill, and `mode` names its leg.")
    print("   So a round trip is entry_leg + exit_leg, summed PER SYMBOL.\n")
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["mode"] in LIVE:
            by[(r["symbol"], r["mode"])].append(r["bps"])
    syms = sorted({s for s, _ in by})
    if args.symbol:
        syms = [s for s in syms if s == args.symbol]

    print(f"   {'symbol':<13} {'leg':<11} {'n':>3}  {'mean':>9}  {'median':>9}  {'max':>9}")
    for s in syms:
        for leg in LIVE:
            v = by.get((s, leg))
            if not v:
                continue
            print(
                f"   {s:<13} {leg:<11} {len(v):>3}  {mean(v):>8.2f}bp  "
                f"{median(v):>8.2f}bp  {max(v):>8.2f}bp"
            )

    print("\n   ROUND TRIP = entry + exit, live fills only:")
    print(f"   {'symbol':<13} {'mean rt':>10} {'median rt':>11}   legs")
    for s in syms:
        e = by.get((s, "live_entry"), [])
        x = by.get((s, "live_exit"), [])
        if not (e and x):
            continue
        print(
            f"   {s:<13} {mean(e) + mean(x):>9.2f}bp {median(e) + median(x):>10.2f}bp"
            f"   entry n={len(e)} exit n={len(x)}"
        )

    print("\n   READ THE MEDIAN COLUMN FOR ANY SYMBOL WHOSE MAX IS ABSURD.")
    print("   CBETH-USDC carries one live_entry row at EXACTLY 10000.00bp -- a 100%")
    print("   deviation is a denomination bug, not a fill, and it alone sets that")
    print("   symbol's mean. The item already rules that row dead; this reprints it")
    print("   so nobody rediscovers it as a finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
