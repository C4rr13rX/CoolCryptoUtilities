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

    if args.json:
        out = {
            "db": path,
            "legs": len(good),
            "censored": len(censored),
            "implausible_ret": IMPLAUSIBLE_RET,
            "median_leg_bp": st.median(costs),
            "mean_leg_bp": sum(costs) / len(costs),
            "round_trip_median_bp": 2 * st.median(costs),
            "round_trip_mean_bp": 2 * sum(costs) / len(costs),
            "by_symbol": {
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
    print(describe(costs, "ALL LEGS"))
    print(describe(entries, "  live_entry"))
    print(describe(exits, "  live_exit"))
    print()
    print("%-14s %3s %10s %10s" % ("symbol", "n", "median_bp", "mean_bp"))
    for sym, v in sorted(by_symbol.items(), key=lambda kv: -len(kv[1])):
        print("%-14s %3d %10.2f %10.2f" % (sym, len(v), st.median(v), sum(v) / len(v)))
    print()
    print("PER ROUND TRIP (2 legs) at the median leg: %.2f bp" % (2 * st.median(costs)))
    print("PER ROUND TRIP (2 legs) at the mean   leg: %.2f bp" % (2 * sum(costs) / len(costs)))
    print()
    print("CAVEAT: four of seven symbols have n=2, i.e. one round trip each.")
    print("CAVEAT: WETH-USDC and ETH-USDC have ZERO live fills, so the pair that")
    print("        decides route (a) is NOT measurable here -- CBBTC/CBETH is a proxy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
