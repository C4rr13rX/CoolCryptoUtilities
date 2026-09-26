#!/usr/bin/env python
"""Realised slippage measured from SETTLED FILLS, not from quotes.

Pass 710 (Cove).  The cost model carries a fee term and no slippage term at
all, and every toll quoted on the crew board is quote-derived.  This script
measures the term that is missing, from the only rows in the store where
expectation and execution are recorded separately: ``trade_fills``.

Why ghost rows cannot answer this
---------------------------------
For all 1518 ghost fills ``expected_price`` and ``executed_price`` are
identical to the last digit -- a ghost fill is simulated, so its slippage is
structurally zero.  Pooling ghost rows into a slippage census therefore does
not dilute the estimate, it *defines it to be zero*.  Only the non-ghost fills
carry signal, and there are 44 of them (20 live_entry, 18 live_exit, 6
gas_refill with no prices).

The round trip is the unit, NOT the leg
--------------------------------------
The first version of this census reported the median of POOLED LEGS and it was
misleading, badly enough to reverse a per-symbol conclusion.  A median of legs
is not the median of round-trip sums: entry and exit slippage do not cancel
pairwise just because their pooled medians nearly cancel, and 8 of the 20 entry
legs have no matching exit leg, so a pooled median measures unclosed positions
against closed ones.  AERO reads -1.38bp per pooled leg and +25.24bp per
round trip paired on ``trade_id``.  The paired figure is the one to quote.

At book level the two agree -- 31.82bp paired against 27.07bp from doubling the
pooled leg median -- so the aggregate term is robust.  It is the PER-SYMBOL
ranking that the pooled statistic fabricated, and that is exactly the number
somebody would use to choose a pair.

Sign convention -- this is the whole measurement
------------------------------------------------
Reported cost is POSITIVE when money was LOST to slippage:

* ``live_entry`` is a buy, so executing ABOVE the expected price is a cost.
* ``live_exit``  is a sell, so executing BELOW the expected price is a cost.

An unsigned |expected - executed| census reports a larger number that means
nothing, because it charges favourable fills as if they were costs.  AERO in
particular comes out slightly FAVOURABLE under the correct convention and
would look like a 20bp cost under the wrong one.

Censoring
---------
One CBETH-USDC live_entry row records ``expected 2731.12`` against
``executed 2.7397213e-09`` -- a units bug, not slippage.  That single row moves
the mean from +40.12bp to -238.38bp, so a census that does not censor it is
wrong by two orders of magnitude.  The bound is the repo's own
``services.outcome_plausibility.IMPLAUSIBLE_RET`` (0.5) rather than a fresh
constant, so there is one calibration in the tree and not two.

Usage
-----
    python scripts/live_fill_slippage_census.py
    python scripts/live_fill_slippage_census.py --json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import statistics as st
import sys

try:
    from services.outcome_plausibility import IMPLAUSIBLE_RET
except Exception:  # pragma: no cover - keep the census runnable standalone
    IMPLAUSIBLE_RET = 0.5

LIVE_MODES = ("live_entry", "live_exit")


def db_path() -> str:
    return os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")


def load_legs(path: str) -> tuple[list, list]:
    """Return (usable legs, censored legs) from the non-ghost fills."""
    con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    try:
        rows = con.execute(
            "select ts, symbol, expected_amount, executed_amount, "
            "expected_price, executed_price, details from trade_fills "
            "where details not like '%ghost%'"
        ).fetchall()
    finally:
        con.close()

    good: list = []
    censored: list = []
    for ts, sym, exp_amt, exe_amt, exp_px, exe_px, details in sorted(rows, key=lambda r: r[0]):
        try:
            det = json.loads(details) if details else {}
        except Exception:
            det = {}
        mode = det.get("mode", "?")
        if mode not in LIVE_MODES:
            continue
        if not (exp_px and exe_px and exp_px > 0 and exe_px > 0):
            continue
        rel = (exe_px - exp_px) / exp_px
        # positive = money lost: a buy filling high, or a sell filling low
        cost_bp = rel * 1e4 if mode == "live_entry" else -rel * 1e4
        leg = dict(
            ts=ts,
            symbol=sym,
            mode=mode,
            expected_price=exp_px,
            executed_price=exe_px,
            rel=rel,
            cost_bp=cost_bp,
            trade_id=det.get("trade_id"),
        )
        (censored if abs(rel) > IMPLAUSIBLE_RET else good).append(leg)
    return good, censored


def round_trips(good: list, censored: list) -> tuple[list, dict]:
    """Pair legs on trade_id. Returns (complete round trips, leg-census counts).

    A round trip is complete only when BOTH legs are present and neither was
    censored -- a trade whose entry leg carries the units bug has no usable
    round-trip cost even though its exit leg is fine.
    """
    by_trade: dict = collections.defaultdict(dict)
    for leg in good:
        tid = leg.get("trade_id")
        if not tid:
            continue
        by_trade[tid][leg["mode"]] = leg["cost_bp"]
        by_trade[tid]["symbol"] = leg["symbol"]
    for leg in censored:
        tid = leg.get("trade_id")
        if tid:
            by_trade[tid]["censored"] = True
            by_trade[tid]["symbol"] = leg["symbol"]

    complete = []
    counts = {"trade_ids": len(by_trade), "entry_only": 0, "exit_only": 0, "censored": 0}
    for tid, v in by_trade.items():
        if v.get("censored"):
            counts["censored"] += 1
            continue
        has_in, has_out = "live_entry" in v, "live_exit" in v
        if has_in and has_out:
            complete.append(
                dict(trade_id=tid, symbol=v["symbol"], entry_bp=v["live_entry"],
                     exit_bp=v["live_exit"], rt_bp=v["live_entry"] + v["live_exit"])
            )
        elif has_in:
            counts["entry_only"] += 1
        elif has_out:
            counts["exit_only"] += 1
    counts["complete"] = len(complete)
    complete.sort(key=lambda r: (r["symbol"], r["trade_id"]))
    return complete, counts


def describe(values: list, label: str) -> str:
    if not values:
        return "%-22s n=  0" % label
    q = st.quantiles(values, n=4) if len(values) > 3 else None
    return "%-22s n=%3d median=%8.2fbp mean=%9.2fbp p25=%8s p75=%8s" % (
        label,
        len(values),
        st.median(values),
        sum(values) / len(values),
        ("%.2f" % q[0]) if q else "n/a",
        ("%.2f" % q[2]) if q else "n/a",
    )


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = ap.parse_args(argv)

    path = db_path()
    if not os.path.exists(path):
        print("no trading store at %s" % path, file=sys.stderr)
        return 2
    good, censored = load_legs(path)
    if not good:
        print("no usable live fills in %s -- nothing to census" % path)
        return 1

    costs = [leg["cost_bp"] for leg in good]
    entries = [leg["cost_bp"] for leg in good if leg["mode"] == "live_entry"]
    exits = [leg["cost_bp"] for leg in good if leg["mode"] == "live_exit"]
    by_symbol = collections.defaultdict(list)
    for leg in good:
        by_symbol[leg["symbol"]].append(leg["cost_bp"])

    trips, counts = round_trips(good, censored)
    rts = [t["rt_bp"] for t in trips]
    rt_by_symbol = collections.defaultdict(list)
    for t in trips:
        rt_by_symbol[t["symbol"]].append(t["rt_bp"])

    if args.json:
        out = {
            "db": path,
            "legs": len(good),
            "censored": len(censored),
            "implausible_ret": IMPLAUSIBLE_RET,
            "leg_counts": counts,
            # the headline: paired round trips, the correct unit
            "round_trip_paired_n": len(rts),
            "round_trip_paired_median_bp": st.median(rts) if rts else None,
            "round_trip_paired_mean_bp": (sum(rts) / len(rts)) if rts else None,
            "round_trip_by_symbol": {
                sym: {"n": len(v), "median_bp": st.median(v), "mean_bp": sum(v) / len(v)}
                for sym, v in rt_by_symbol.items()
            },
            # kept for comparison only -- see the module docstring on why the
            # pooled-leg statistic fabricates a per-symbol ranking
            "median_leg_bp": st.median(costs),
            "mean_leg_bp": sum(costs) / len(costs),
            "leg_by_symbol": {
                sym: {"n": len(v), "median_bp": st.median(v), "mean_bp": sum(v) / len(v)}
                for sym, v in by_symbol.items()
            },
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print("REALISED SLIPPAGE FROM SETTLED FILLS -- %s" % path)
    print("positive = money LOST to slippage (buy filled high / sell filled low)")
    print()
    print("censored %d leg(s) at |rel| > %.2f (services.outcome_plausibility.IMPLAUSIBLE_RET):"
          % (len(censored), IMPLAUSIBLE_RET))
    for leg in censored:
        print("   %-14s %-11s expected=%.8g executed=%.8g -> %.0fbp"
              % (leg["symbol"], leg["mode"], leg["expected_price"],
                 leg["executed_price"], leg["cost_bp"]))
    if not censored:
        print("   (none)")
    print()
    print("=== HEADLINE: PAIRED ROUND TRIPS (the correct unit) ===")
    print("trade_ids %d | complete %d | entry-only %d | exit-only %d | censored %d"
          % (counts["trade_ids"], counts["complete"], counts["entry_only"],
             counts["exit_only"], counts["censored"]))
    print()
    print("%-14s %10s %10s %10s" % ("symbol", "entry_bp", "exit_bp", "rt_bp"))
    for t in trips:
        print("%-14s %10.2f %10.2f %10.2f"
              % (t["symbol"], t["entry_bp"], t["exit_bp"], t["rt_bp"]))
    print()
    if rts:
        print("PAIRED ROUND TRIP  n=%d  median=%.2fbp  mean=%.2fbp"
              % (len(rts), st.median(rts), sum(rts) / len(rts)))
        print()
        print("%-14s %3s %10s %10s" % ("symbol", "n", "median_rt", "mean_rt"))
        for sym, v in sorted(rt_by_symbol.items(), key=lambda kv: -len(kv[1])):
            print("%-14s %3d %10.2f %10.2f" % (sym, len(v), st.median(v), sum(v) / len(v)))
    print()
    print("=== POOLED LEGS (for comparison only -- do NOT read a per-symbol")
    print("    ranking off this; see the module docstring) ===")
    print(describe(costs, "ALL LEGS"))
    print(describe(entries, "  live_entry"))
    print(describe(exits, "  live_exit"))
    print()
    print("%-14s %3s %10s %10s" % ("symbol", "n", "median_bp", "mean_bp"))
    for sym, v in sorted(by_symbol.items(), key=lambda kv: -len(kv[1])):
        print("%-14s %3d %10.2f %10.2f" % (sym, len(v), st.median(v), sum(v) / len(v)))
    print()
    print("CAVEAT: five of six symbols have exactly ONE complete round trip.")
    print("        No per-symbol slippage number here survives its own sample")
    print("        size except AERO-USDC at n=6.")
    print("CAVEAT: WETH-USDC and ETH-USDC have ZERO live fills, so the pair that")
    print("        decides route (a) is NOT measurable here -- CBBTC/CBETH is a proxy")
    print("        resting on one round trip each.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
