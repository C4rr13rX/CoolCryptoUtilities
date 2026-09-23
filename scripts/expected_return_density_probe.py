#!/usr/bin/env python
"""Is atf_static's persisted ``expected_return`` a forecast, or a literal?

Read-only. Item [aca39497], pass 674.

WHAT THIS ANSWERS AND WHY IT IS NOT scripts/expected_return_calibration.py.
The calibration script asks "how well does the persisted number predict the
forward move" and answers 214.4x / rho +0.0453 / p=0.1129 at 15 minutes. That
is a calibration on the wrong object. The producer at
``services/atf_static_strategy.py`` computes a forecast and then throws it
away:

    forecast      = clamp(0, 0.15, (price_change_m5/100)*0.35 + score*0.04)
    target_return = max(0.015, min(0.15, max(forecast, ATF_STATIC_TARGET_RETURN)))
    signal["expected_return"] = round(target_return, 6)     # <- the TARGET

So whenever ``forecast < ATF_STATIC_TARGET_RETURN`` the persisted value is a
LITERAL -- 0.05 from the env default, or 0.015 on the low-confidence branch --
and carries no rank information at all. A rho near zero is then not evidence
that the forecast is bad; it is evidence that the forecast was never stored.

This probe separates the two claims:

  HYPOTHESIS.  The persisted field is dominated by the two literals, so the
  measured rho is an artifact of a constant and says nothing about the
  forecast's skill.

  COMPLEMENT.  The persisted field is mostly the forecast passing through the
  max() unchanged, the literals are a small minority, and the near-zero rho
  IS a real verdict on the forecast -- in which case the forecast should be
  abandoned rather than instrumented.

  THE TEST THAT SEPARATES THEM, pre-registered: the share of priced enter rows
  whose persisted ``expected_return`` is exactly 0.015 or exactly
  ATF_STATIC_TARGET_RETURN (default 0.05). Above ~50% the hypothesis holds and
  the calibration must be re-run on ``forecast_return`` before anyone tunes
  this field. Below ~20% the complement holds and the literal is a footnote.

It also reports how many rows carry ``forecast_return`` at all. That field is
written for the first time by this pass's change and is FORWARD-ONLY: no
historical row has it, so a zero there is the expected answer today and is the
reason the item's calibration criterion cannot yet be satisfied.

Usage:
    python -X utf8 scripts/expected_return_density_probe.py [--days 30]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
# The producer's own two literals. 0.015 is the hard low-confidence branch;
# the other is whatever ATF_STATIC_TARGET_RETURN is set to, read from the
# environment exactly as the producer reads it so a changed env cannot make
# this probe silently measure a stale constant.
LOW_CONFIDENCE_FLOOR = 0.015


def _target_floor() -> float:
    try:
        raw = float(os.getenv("ATF_STATIC_TARGET_RETURN", "0.05"))
    except (TypeError, ValueError):
        raw = 0.05
    return max(0.005, min(0.10, raw))


def _connect(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise SystemExit(f"trading store not found: {path}")
    size = os.path.getsize(path)
    if size < 1024:
        # data/trading.db, trading_bot.db and web/db.sqlite3 are 0-byte stubs
        # that answer every query with nothing. Refuse rather than print zeros.
        raise SystemExit(f"{path} is {size} bytes -- that is a stub, not the store")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _rows(conn: sqlite3.Connection, since: float) -> List[Dict[str, Any]]:
    """Every atf_static enter row carrying a persisted expected_return."""
    out: List[Dict[str, Any]] = []
    sql = (
        "SELECT ts, symbol, status, details FROM trading_ops "
        "WHERE action = 'enter' AND ts > ? AND details LIKE '%expected_return%'"
    )
    for ts, symbol, status, details in conn.execute(sql, (since,)):
        try:
            d = json.loads(details) if details else {}
        except (TypeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        # ghost-entry rows nest the producer's signal under "signal"; the
        # ghost_candidate rows ARE the signal. Take whichever carries the field
        # so the two populations are counted once each, not once and a half.
        sig = d.get("signal") if isinstance(d.get("signal"), dict) else d
        if "expected_return" not in sig:
            continue
        try:
            er = float(sig.get("expected_return"))
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "ts": float(ts or 0.0),
                "symbol": str(symbol or sig.get("symbol") or "").upper(),
                "status": str(status or ""),
                "expected_return": er,
                "forecast_return": sig.get("forecast_return"),
                "forecast_is_floored": sig.get("forecast_is_floored"),
            }
        )
    return out


def _tick_counts(conn: sqlite3.Connection, symbols: List[str], since: float) -> Dict[str, int]:
    """Ticks per symbol over the window -- the tape-density axis."""
    counts: Dict[str, int] = {}
    if not symbols:
        return counts
    # The tape lives in `market_stream` here. Resolve the timestamp column by
    # reading the schema rather than guessing it: a wrong column name raises,
    # a wrong GUESS at a column that happens to exist returns a plausible
    # number that is silently measuring something else.
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(market_stream)")}
    except sqlite3.Error as exc:
        print(f"  (tick census unavailable: {exc})")
        return counts
    if not cols:
        print("  (tick census unavailable: market_stream has no columns)")
        return counts
    ts_col = next((c for c in ("ts", "timestamp", "recorded_at", "time") if c in cols), None)
    sym_col = next((c for c in ("symbol", "pair", "ticker") if c in cols), None)
    if not ts_col or not sym_col:
        print(f"  (tick census unavailable: market_stream columns are {sorted(cols)})")
        return counts
    sql = f"SELECT COUNT(*) FROM market_stream WHERE {sym_col} = ? AND {ts_col} > ?"
    try:
        for sym in symbols:
            row = conn.execute(sql, (sym, since)).fetchone()
            counts[sym] = int(row[0] or 0) if row else 0
    except sqlite3.Error as exc:
        print(f"  (tick census unavailable: {exc})")
        return {}
    return counts


def _median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args()

    floor = _target_floor()
    since = time.time() - args.days * 86400.0
    conn = _connect(args.db)
    rows = _rows(conn, since)

    print("=" * 74)
    print("IS atf_static's PERSISTED expected_return A FORECAST OR A LITERAL?")
    print("=" * 74)
    print(f"  store            : {args.db}")
    print(f"  window           : last {args.days:g} days")
    print(f"  enter rows       : {len(rows)}")
    print(f"  the two literals : {LOW_CONFIDENCE_FLOOR} (low-confidence branch)"
          f" and {floor} (ATF_STATIC_TARGET_RETURN)")
    if not rows:
        print("\n  NO ROWS. Widen --days before reading this as a result: a short")
        print("  window on a slow producer reads as death. Nothing is concluded.")
        return 1

    literals = {round(LOW_CONFIDENCE_FLOOR, 6), round(floor, 6)}
    hist = Counter(round(r["expected_return"], 6) for r in rows)
    on_literal = [r for r in rows if round(r["expected_return"], 6) in literals]
    off_literal = [r for r in rows if round(r["expected_return"], 6) not in literals]
    share = 100.0 * len(on_literal) / len(rows)

    print("\n--- THE PRE-REGISTERED TEST ---")
    print(f"  exactly on a literal : {len(on_literal):6d} / {len(rows)}  ({share:.1f}%)")
    print(f"  carrying a forecast  : {len(off_literal):6d} / {len(rows)}  ({100.0 - share:.1f}%)")
    if share >= 50.0:
        verdict = ("HYPOTHESIS HOLDS -- the field is dominated by a constant. The "
                   "measured\n  rho is an artifact of that constant and is NOT a "
                   "verdict on the forecast.")
    elif share <= 20.0:
        verdict = ("COMPLEMENT HOLDS -- the literals are a minority and the field "
                   "mostly\n  carries the forecast, so the near-zero rho IS a "
                   "verdict on the forecast.")
    else:
        verdict = ("NEITHER -- the share lands between the pre-registered "
                   "thresholds.\n  Report the number; do not pick a side on it.")
    print(f"  VERDICT: {verdict}")

    print("\n--- DISTINCT PERSISTED VALUES, most common first ---")
    for value, n in hist.most_common(10):
        tag = "  <-- LITERAL" if round(value, 6) in literals else ""
        print(f"  {value:<12.6f} {n:6d}  ({100.0 * n / len(rows):5.1f}%){tag}")
    print(f"  distinct values: {len(hist)}")

    # Tape density. The item's claim is that the rows clearing the floor -- the
    # ones carrying a real forecast -- are the THIN-tape rows. On-literal vs
    # off-literal is exactly that split, so it is the cohort comparison to make.
    symbols = sorted({r["symbol"] for r in rows if r["symbol"]})
    ticks = _tick_counts(conn, symbols, since)
    if ticks:
        print("\n--- TAPE DENSITY BY COHORT (ticks per symbol over the window) ---")
        # Report the STREAMED median separately from the all-symbols median.
        # Most researched candidates never reach the tape at all -- the quote
        # step is already known to lose ~87% of symbols -- so an all-symbols
        # median of 0 is a statement about coverage, not about density, and
        # quoting it as "median ticks per symbol" would be the same category
        # error this item was filed about. Both numbers, labelled.
        for label, cohort in (("on a literal", on_literal), ("carrying a forecast", off_literal)):
            syms = sorted({r["symbol"] for r in cohort if r["symbol"]})
            vals = [float(ticks.get(s, 0)) for s in syms]
            streamed = [v for v in vals if v > 0]
            med_all = _median(vals)
            med_str = _median(streamed)
            cov = 100.0 * len(streamed) / len(syms) if syms else 0.0
            print(
                f"  {label:<22} rows={len(cohort):6d}  symbols={len(syms):5d}"
                f"  streamed={len(streamed):5d} ({cov:4.1f}%)"
                f"  median ticks all={med_all if med_all is not None else 0:,.0f}"
                f"  streamed-only={med_str if med_str is not None else 0:,.0f}"
            )

    print("\n--- forecast_return COVERAGE (forward-only; written from pass 674) ---")
    have = [r for r in rows if r.get("forecast_return") is not None]
    print(f"  rows carrying forecast_return : {len(have)} / {len(rows)}")
    if not have:
        print("  ZERO, and that is the EXPECTED answer today, not a failure: the")
        print("  producer only started writing the field this pass and it cannot")
        print("  backfill a value that was never computed on a stored row. Until")
        print("  this is non-zero, the item's calibration criterion CANNOT be")
        print("  satisfied -- re-running the calibration today re-measures the")
        print("  take-profit target and reproduces the same baseline by")
        print("  construction. It needs production restarted into the change and")
        print("  forward enter rows accumulated. Do not report that as a fix that")
        print("  'did not work'; it is a fix that has not been measured yet.")
    else:
        floored = [r for r in have if r.get("forecast_is_floored")]
        fvals = [float(r["forecast_return"]) for r in have]
        print(f"  floored by the max()          : {len(floored)} / {len(have)}"
              f"  ({100.0 * len(floored) / len(have):.1f}%)")
        print(f"  forecast_return median        : {_median(fvals):.6f}")
        print(f"  forecast_return == 0 exactly  : {sum(1 for v in fvals if v == 0.0)}")
        print("  The calibration can now be re-run against forecast_return.")

    by_status: Dict[str, int] = defaultdict(int)
    for r in rows:
        by_status[r["status"]] += 1
    print("\n--- rows by trading_ops status ---")
    for status, n in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"  {status:<34} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
