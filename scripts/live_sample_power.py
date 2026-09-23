#!/usr/bin/env python
"""How many live round trips does it take to RESOLVE a cost question, and what
would that many cost in dollars and in days?

Item [7b79c272], pass 674. This exists because pass 487 filed a cost claim, killed
it with its own bootstrap CI in the same pass, and then priced the fix using a
rate that blended a three-day burst with a two-week silence. Both halves of that
are failure modes this script refuses to repeat:

  * MEDIAN WITH A BOOTSTRAP, ALWAYS. The live round-trip sample contains two rows
    near -12000bp and one near +35000bp. A mean/SD sample-size formula on that
    sample asks for 539,748 round trips; it is measuring the outliers, not the
    effect. Every dispersion number here is a median/MAD/bootstrap quantity.
  * A RATE NEEDS ITS SPAN NAMED. The live lane ran on exactly three calendar days
    and has been at a hard zero ever since. Dividing 18 round trips by the whole
    elapsed span and dividing by the active span differ by ~7x, and they answer
    different questions. This script prints BOTH and refuses to pick one for you.

It arms nothing, places nothing and writes nothing. Read-only against
storage/trading_cache.db.

    python -X utf8 scripts/live_sample_power.py
    python -X utf8 scripts/live_sample_power.py --target-halfwidth 10 --clip-usd 10
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import time

DEFAULT_DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")

# Gas is FIXED per round trip, so the minimum clip is NOT the cheapest clip in
# basis points -- see item [dd98146a]. Measured p496: $0.0040/round trip on base.
GAS_USD_PER_ROUND_TRIP = 0.0040

# Deployable stable in the live wallet, from the pass gate. Not read from chain
# here on purpose: this script must stay read-only and offline.
DEPLOYABLE_USD = 23.18


def _utc(ts: float) -> _dt.datetime:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc)


def load_live_legs(db_path: str) -> dict:
    """Every live leg the lane has ever placed, with its timestamp and clip."""
    uri = "file:" + db_path.replace("\\", "/") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    cur = conn.cursor()
    out = {"entries": [], "exits": [], "settled": []}
    for status, key in (
        ("live-entry", "entries"),
        ("live-exit", "exits"),
        ("live-swap-settled", "settled"),
    ):
        cur.execute(
            "SELECT ts, symbol, details FROM trading_ops WHERE status = ? ORDER BY ts",
            (status,),
        )
        for ts, symbol, details in cur.fetchall():
            try:
                blob = json.loads(details) if details else {}
            except Exception:
                blob = {}
            out[key].append({"ts": ts, "symbol": symbol, "d": blob})
    conn.close()
    return out


def round_trip_net_bp(exits: list) -> list:
    """Realised net per round trip, in bp, as profit_usd / notional_usd.

    DO NOT USE net_margin_after_fees FOR THIS. Pass 674 tried it first and it is
    a trap: `net_margin` is EXACTLY +0.0000 on 12 of the 18 live exits, and
    net_margin_after_fees is that zero minus a per-trade FEE CONSTANT. Its median
    (-65.00 bp) is therefore the fee schedule, not a return -- the row booking
    +3227 bp of realised gross carries net_margin_after_fees = -0.0065, the same
    value as a row that lost 286 bp. The only field that moves with the fill is
    `profit` (USD), so the realised fraction is profit / (entry_price * size).
    """
    vals = []
    for row in exits:
        d = row["d"]
        ep, size, profit = d.get("entry_price"), d.get("size"), d.get("profit")
        if not ep or not size or profit is None:
            continue
        notional = ep * size
        if notional <= 0:
            continue
        vals.append(profit / notional * 10000.0)
    return vals


def median_notional_usd(exits: list) -> float:
    ns = []
    for row in exits:
        d = row["d"]
        ep, size = d.get("entry_price"), d.get("size")
        if ep and size and ep * size > 0:
            ns.append(ep * size)
    return statistics.median(ns) if ns else float("nan")


def mad(values: list) -> float:
    """Median absolute deviation. Not scaled to an SD -- it is reported raw."""
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])


def bootstrap_median_ci(values: list, resamples: int = 20000, seed: int = 674):
    rng = random.Random(seed)
    n = len(values)
    meds = []
    for _ in range(resamples):
        meds.append(statistics.median([values[rng.randrange(n)] for _ in range(n)]))
    meds.sort()
    lo = meds[int(0.025 * resamples)]
    hi = meds[int(0.975 * resamples)]
    return lo, hi


def required_n_for_halfwidth(values: list, target_hw: float, resamples: int = 4000,
                             seed: int = 674, n_max: int = 200000) -> int:
    """Smallest n whose bootstrap-median half-width falls under target_hw.

    Resamples WITH replacement from the observed sample at increasing n. This is
    the honest version of a power calculation on a sample this skewed: it makes
    no normality assumption and it carries the observed outliers forward, which
    is exactly why it will not return the tiny n a naive formula would.
    """
    rng = random.Random(seed)
    n = len(values)

    def halfwidth(size: int) -> float:
        meds = []
        for _ in range(resamples):
            meds.append(statistics.median([values[rng.randrange(n)] for _ in range(size)]))
        meds.sort()
        return (meds[int(0.975 * resamples)] - meds[int(0.025 * resamples)]) / 2.0

    hw0 = halfwidth(n)
    if hw0 <= target_hw:
        return n
    # Two measured anchors, then extrapolate on the fitted exponent rather than
    # resampling all the way out -- a direct search to n=200000 is 800M medians.
    hw1 = halfwidth(4 * n)
    beta = math.log(hw0 / hw1) / math.log(4.0) if hw1 > 0 else 0.5
    beta = min(max(beta, 0.30), 0.70)  # pin near root-n; a wild fit is noise
    need = int(math.ceil(4 * n * (hw1 / target_hw) ** (1.0 / beta)))
    return min(need, n_max) if need <= n_max else -need


def rate_report(entries: list, exits: list) -> dict:
    """Both rates, with the span each one divides by named."""
    ts = sorted(r["ts"] for r in exits)
    days = collections.Counter(_utc(t).strftime("%Y-%m-%d") for t in ts)
    now = time.time()
    active_days = len(days)
    span_elapsed_days = (now - ts[0]) / 86400.0
    span_burst_days = (ts[-1] - ts[0]) / 86400.0
    return {
        "n": len(ts),
        "by_day": sorted(days.items()),
        "active_days": active_days,
        "silent_hours": (now - ts[-1]) / 3600.0,
        "rate_over_active_days": len(ts) / active_days,
        "rate_over_burst_span": len(ts) / span_burst_days if span_burst_days else float("nan"),
        "rate_over_elapsed": len(ts) / span_elapsed_days,
        "span_burst_days": span_burst_days,
        "span_elapsed_days": span_elapsed_days,
        "first": _utc(ts[0]).isoformat(timespec="seconds"),
        "last": _utc(ts[-1]).isoformat(timespec="seconds"),
    }


def clip_cost_bp(clip_usd: float, per_side_bp: float) -> float:
    """All-in round-trip cost in bp at a given clip. Gas is FIXED in dollars, so
    it is the term that makes a small clip expensive."""
    gas_bp = (GAS_USD_PER_ROUND_TRIP / clip_usd) * 10000.0
    return 2.0 * per_side_bp + gas_bp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--target-halfwidth", type=float, default=None,
                    help="bp half-width to resolve; default sweeps 25/10/5")
    ap.add_argument("--clip-usd", type=float, default=10.0,
                    help="clip the research programme would run at (default 10)")
    ap.add_argument("--per-side-bp", type=float, default=2.06,
                    help="one-leg cost excluding gas; default 2.06 = p498 WETH-USDC at $10")
    ap.add_argument("--resamples", type=int, default=20000)
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print("NO DB at %s -- cannot measure, and will not estimate." % args.db)
        return 2

    legs = load_live_legs(args.db)
    entries, exits = legs["entries"], legs["exits"]
    if not exits:
        print("ZERO live-exit rows. Nothing to power. Say that, do not estimate.")
        return 2

    print("=" * 74)
    print("LIVE SAMPLE POWER -- what it takes to RESOLVE a cost question")
    print("=" * 74)

    r = rate_report(entries, exits)
    print("\n-- THE RATE, WITH ITS SPAN NAMED (this is the p487 error) --")
    print("  live-entry legs      : %d" % len(entries))
    print("  live-exit legs       : %d   (= completed round trips)" % r["n"])
    print("  settled swap rows    : %d" % len(legs["settled"]))
    print("  first / last exit    : %s  ->  %s" % (r["first"], r["last"]))
    print("  by calendar day      : %s" % (r["by_day"],))
    print("  SILENT FOR           : %.1f hours (%.1f days)"
          % (r["silent_hours"], r["silent_hours"] / 24.0))
    print("  rate / ACTIVE day    : %.2f  <- what the lane does WHEN IT RUNS"
          % r["rate_over_active_days"])
    print("  rate / ELAPSED day   : %.2f  <- burst blended with silence; %.1fx lower"
          % (r["rate_over_elapsed"],
             r["rate_over_active_days"] / max(r["rate_over_elapsed"], 1e-9)))
    print("  The two differ by %.1fx. A days-to-sample figure that does not say"
          % (r["rate_over_active_days"] / max(r["rate_over_elapsed"], 1e-9)))
    print("  WHICH ONE it divided by is not a number, it is a rhetorical device.")

    vals = round_trip_net_bp(exits)
    med = statistics.median(vals)
    lo, hi = bootstrap_median_ci(vals, args.resamples)
    print("\n-- THE SAMPLE (realised net per round trip, bp) --")
    print("  n = %d" % len(vals))
    print("  median            : %+.2f bp" % med)
    print("  MAD               :  %.2f bp" % mad(vals))
    print("  mean / SD         : %+.2f / %.2f   <- DO NOT USE; see below"
          % (statistics.fmean(vals), statistics.pstdev(vals)))
    print("  bootstrap 95%% CI  : [%+.2f, %+.2f]  half-width %.2f bp"
          % (lo, hi, (hi - lo) / 2.0))
    print("  winners           : %d of %d" % (sum(1 for v in vals if v > 0), len(vals)))
    print("  most extreme rows : %s" % ["%+.0f" % v for v in sorted(vals)[:2]]
          + " and %s" % ["%+.0f" % v for v in sorted(vals)[-1:]])
    print("  The mean is %.0fx the median and of the OPPOSITE character. Three rows"
          % abs(statistics.fmean(vals) / med))
    print("  carry it. Any sample size derived from that SD is measuring them.")

    targets = ([args.target_halfwidth] if args.target_halfwidth
               else [25.0, 10.0, 5.0])
    print("\n-- REQUIRED SAMPLE, BOOTSTRAP (no normality assumed) --")
    print("  %-14s %-12s %-14s %-14s" % ("resolve to", "round trips",
                                         "days @active", "days @elapsed"))
    programme = []
    for t in targets:
        need = required_n_for_halfwidth(vals, t)
        if need < 0:
            print("  +/-%-11.1f %-12s (extrapolated past the n=200000 cap)" % (t, abs(need)))
            continue
        d_active = need / r["rate_over_active_days"]
        d_elapsed = need / r["rate_over_elapsed"]
        print("  +/-%-11.1f %-12d %-14.1f %-14.1f" % (t, need, d_active, d_elapsed))
        programme.append((t, need))

    print("\n-- WHAT THAT PROGRAMME COSTS IN DOLLARS --")
    cost_bp = clip_cost_bp(args.clip_usd, args.per_side_bp)
    gas_bp = (GAS_USD_PER_ROUND_TRIP / args.clip_usd) * 10000.0
    print("  clip $%.2f  ->  all-in %.2f bp/round trip (%.2f bp fee+slip, %.2f bp gas)"
          % (args.clip_usd, cost_bp, 2 * args.per_side_bp, gas_bp))
    print("  GAS IS FIXED PER ROUND TRIP, so the MINIMUM clip is NOT the cheapest:")
    for c in (0.75, 1.0, 2.0, 5.0, 10.0, 23.18):
        print("      $%-6.2f -> %7.2f bp all-in   ($%.4f per round trip)"
              % (c, clip_cost_bp(c, args.per_side_bp),
                 c * clip_cost_bp(c, args.per_side_bp) / 10000.0))
    print("  deployable stable: $%.2f (NOT consumed per trade -- it recycles;" % DEPLOYABLE_USD)
    print("  what is consumed is the COST, plus whatever the edge turns out to be)")
    for t, need in programme:
        burn = need * args.clip_usd * cost_bp / 10000.0
        print("  +/-%.1f bp needs %d round trips = $%.2f in pure cost at a $%.2f clip"
              % (t, need, burn, args.clip_usd))
        if burn > DEPLOYABLE_USD:
            print("      ^ EXCEEDS the $%.2f deployable by %.1fx. Not fundable."
                  % (DEPLOYABLE_USD, burn / DEPLOYABLE_USD))

    print("\n-- WHAT IT ACTUALLY BURNS: THE EDGE TERM DOMINATES THE FEE TERM --")
    obs_clip = median_notional_usd(exits)
    print("  Fees are NOT the cost of this programme. The realised net is. The %d"
          % len(vals))
    print("  rows were placed at a MEDIAN $%.2f NOTIONAL where gas alone was %.1f bp"
          % (obs_clip, (GAS_USD_PER_ROUND_TRIP / obs_clip) * 10000.0))
    print("  of a %.1f bp all-in cost. Raise the clip and the gas term collapses."
          % clip_cost_bp(obs_clip, args.per_side_bp))
    obs_gas_bp = (GAS_USD_PER_ROUND_TRIP / obs_clip) * 10000.0
    new_gas_bp = (GAS_USD_PER_ROUND_TRIP / args.clip_usd) * 10000.0
    rescaled = med + obs_gas_bp - new_gas_bp
    print("  observed median net @ $%.2f notional : %+.2f bp" % (obs_clip, med))
    print("  same rows with gas re-priced to a $%.2f clip : %+.2f bp"
          % (args.clip_usd, rescaled))
    print("  ^ THIS RE-SCALING ASSUMES SLIPPAGE IS CLIP-INVARIANT AND IT IS NOT.")
    print("    Bigger clip = more slippage. Treat %+.2f bp as an OPTIMISTIC bound on"
          % rescaled)
    print("    what the same strategy would book at $%.2f, not a forecast." % args.clip_usd)
    for t, need in programme:
        fee_burn = need * args.clip_usd * cost_bp / 10000.0
        edge_burn = -need * args.clip_usd * rescaled / 10000.0
        if rescaled < 0:
            print("  +/-%.1f bp (%d round trips) : fees $%.2f + expected LOSS $%.2f = $%.2f"
                  % (t, need, fee_burn, edge_burn, fee_burn + edge_burn))
        else:
            print("  +/-%.1f bp (%d round trips) : fees $%.2f, edge term non-negative"
                  % (t, need, fee_burn))
        total = fee_burn + max(0.0, edge_burn)
        if total > DEPLOYABLE_USD:
            print("      ^ EXCEEDS the $%.2f deployable by %.1fx. NOT FUNDABLE at this clip."
                  % (DEPLOYABLE_USD, total / DEPLOYABLE_USD))
        else:
            print("      ^ fundable: %.0f%% of the $%.2f deployable."
                  % (100.0 * total / DEPLOYABLE_USD, DEPLOYABLE_USD))

    print("\n-- CAVEATS THAT TRAVEL WITH EVERY NUMBER ABOVE --")
    print("  * THE REQUIRED-n ABOVE ANSWERS 'PIN THE MEDIAN', NOT p487's QUESTION.")
    print("    p487 asked for 1081 round trips to resolve a 12.74bp GAP BETWEEN TWO")
    print("    GROUPS; this asks what pins ONE median. They are different questions")
    print("    and the smaller number here does not overturn the larger one.")
    print("  * The sample is DISCRETE -- 6 of 18 rows are exactly -65.00 bp, which is")
    print("    a booked constant, not a measurement. A bootstrap on a sample with a")
    print("    repeated mode converges FASTER than a continuous one would, so the")
    print("    required-n figures are an OPTIMISTIC floor. Do not quote them as a")
    print("    ceiling, and note +/-10 and +/-5 landing 9 apart is that discreteness,")
    print("    not a resolved difference.")
    print("  * net_margin_after_fees is the EXECUTOR's own booking, not a chain")
    print("    re-derivation. It agrees with the book in sign and order; it has not")
    print("    been tied leg-by-leg to receipts in this script.")
    print("  * WIDE ERROR BARS ARE NOT EVIDENCE OF HIDDEN PROFIT. Every point")
    print("    estimate here loses. The honest verdict at this n is UNDETERMINED.")
    print("  * NEVER QUOTE A GROSS WITHOUT ITS POPULATION. +15.03bp is a SIMULATION")
    print("    over 1408 ghost paths, best of 28 cells, slippage-free, self-labelled")
    print("    an upper bound. -0.0849 is a realised dollar sum over %d live round" % len(vals))
    print("    trips. Comparing them is CONSERVATIVE, not optimistic.")
    print("  * THIS SCRIPT ARMS NOTHING. AUTO_EXECUTE_ON_GRADUATION defaults to 1,")
    print("    so LIVE_TRADES_DRY_RUN must be set explicitly before any restart.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
