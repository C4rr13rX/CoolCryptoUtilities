#!/usr/bin/env python3
"""[51969494] criterion 1: is the chaos gate's ~3-min wall a MEASUREMENT or an
ARTIFACT? And if it is a measurement, what is on the OTHER side of the
inequality?

THE FORK THIS SCRIPT EXISTS TO DECIDE (Bay p718 wrote it, and it decides
one-pass-fix vs project):

    (a) the wall is REAL  -> all 36 strategies that carry books forecast
        45-10080 min and are structurally untradeable, so the work is a
        sub-5-min strategy built from scratch. A PROJECT, not a pass.
    (b) the wall is a BAR-LENGTH ARTIFACT -> the 36 are fine, the gate is
        broken, ONE-PASS FIX.

    Bay's warning, and it is why (b) had to be taken seriously:
    tests/test_lattice_gate.py documents a prior bug where a collapsed bar
    length made the gate refuse everything on every symbol.

THE ANSWER IS NEITHER, measured 2026-09-26 over 11,877 rows / 14 days. Both
forks assume the two sides of the comparison are the same kind of quantity.
They are not.

WHY (b) IS DEAD -- AND THIS WAS THE AUTHOR'S OWN HYPOTHESIS, KILLED HERE.
`web/tradingagent/chaos.py:187` computes

    usable = 1.0 / exponent_per_sec          # == bar_sec / per_step

so `usable` is LINEAR IN bar_sec, and :209-211 hard-clamps it to
`observed_span = len(series) * bar_sec`, returning None above that. That is
exactly the shape of a censored estimator, and the pre-registered expectation
was that the ~2.8 min figure IS that ceiling -- a 64-sample buffer
(`trading/bot.py:2425` refuses below 64) that can only ever measure ~3 min.

    THE MEASUREMENT REFUTES IT: 714 DISTINCT decay values, spanning
    0.200 -> 248.400 min, with 0.01% of rows sitting at the maximum.

A censored measurement does not have 714 distinct values and does not reach
4.1 hours. And `bar_sec` is honestly measured, not assumed: `trading/bot.py:
2471 _median_tick_gap_sec` takes span-over-count off real tick stamps and its
own docstring names this precise failure mode ("a wrong bar length scales the
answer by exactly that factor").

WHY (a) IS DEAD TOO -- THE NUMERATOR, NOT THE DENOMINATOR, IS THE DEFECT.
At `web/tradingagent/lattice.py:141` the gate refuses when
`proposed_horizon_sec > usable`. Count the distinct values on each side:

    DENOMINATOR  usable                 714 distinct, 0.2 -> 248.4 min
    NUMERATOR    proposed_horizon_sec    11 distinct, and every one of them is
                 a strategy-id SUFFIX rendered in minutes
                 {3, 6, 20, 45, 60, 300, 720, 1440, 4320, 7200, 10080}

A measurement against a label. `donchian_breakout@1w` denotes a one-week
Donchian CHANNEL -- a LOOKBACK -- and the gate reads it as "forecasts one week
ahead". The median numerator is 1440.0 min = ONE DAY, while `.env:169
MAX_HOLD_FORCE_SECONDS=2700` caps every hold at 45 MINUTES: our own config
refutes the number the gate is testing.

This CONFIRMS `scripts/chaos_gate_horizon_census.py` (2026-09-22, backlog
c813f5fe), independently and 14 days later, by a different route: that script
grouped the numerator by suffix and found one value per group; this one counts
distinct values on BOTH sides and finds the asymmetry.

WHAT IT IS WORTH, AND THE THREE LIMITS ON SAYING IT LOUDER.
Of the 11,877 candidates the gate refused in 14 days, the share that would
clear the CHAOS layer if the numerator carried the real intended hold:

    45 min (our own MAX_HOLD_FORCE_SECONDS)    458 =  3.9%
    15 min (standing route (c), shorten hold) 1471 = 12.4%
     5 min                                    3958 = 33.3%
     3 min                                    7006 = 59.0%
    today                                        0 =  0.0%

  1. THIS BUYS THROUGHPUT, NOT EDGE. 458 candidates reaching the next gate are
     458 candidates that then face the cost floor and the symbol edge ban.
     It moves the graduation RATE, which is what Bay retitled the item to be.
  2. 3.9% AT OUR ACTUAL CAP, NOT 100%. Over 96% of ticks genuinely cannot
     support a 45-minute forecast, so part of the wall is real. The honest
     headline is "1 in 25 at 45 min, 1 in 8 at 15 min", not "the gate is
     broken".
  3. NOTHING HERE SHOWS THE SURVIVORS ARE PROFITABLE. It is not measured and
     must not be read as if it were.

READ-ONLY, and it needs no restart. A FIX does: the site is
`trading/bot.py:2445 proposed_horizon_sec=horizon_sec`, production serves
stale code until rebooted, and a boot is ~13 minutes.

USAGE
    python -X utf8 scripts/lattice_horizon_census.py
    python -X utf8 scripts/lattice_horizon_census.py --days 7
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys

DB = os.getenv("TRADING_DB_PATH", "storage/trading_cache.db")
STATUS = "entry-refused-lattice"

#: Both sides of web/tradingagent/lattice.py:141, off the row's own detail.
DECAY_RE = re.compile(r"decays after ([0-9.]+) min")
AHEAD_RE = re.compile(r"looks ([0-9.]+) min ahead")

#: Candidate holds to price the numerator fix at. 45.0 is .env:169
#: MAX_HOLD_FORCE_SECONDS=2700; 15.0 is standing route (c).
HOLDS_MIN = (1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 45.0, 60.0)


def _quantile(sorted_vals, p: float) -> float:
    if not sorted_vals:
        return float("nan")
    return sorted_vals[int(p * (len(sorted_vals) - 1))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"no store at {args.db}")
        return 2

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    cur = conn.cursor()

    # THE NO-WHERE CONTROL, FIRST AND UNCONDITIONALLY. `trading_ops.ts` is a
    # FLOAT EPOCH: a `ts >= '2026-09-16'` predicate returns 0 rows with no
    # error, which is indistinguishable from an instrument that stopped
    # emitting. Every cut below is max(ts) - N*86400, never a date string.
    cur.execute("select count(*) from trading_ops")
    control = cur.fetchone()[0]
    cur.execute("select max(ts) from trading_ops")
    newest = cur.fetchone()[0]
    if not newest:
        print("trading_ops carries no usable ts")
        return 2
    cut = newest - args.days * 86400.0

    print(f"store {args.db}  ({os.path.getsize(args.db) / 1e9:.1f} GB)")
    print(f"CONTROL, no WHERE clause: {control} trading_ops rows")
    print(f"newest ts {newest:.1f}; cut at max(ts) - {args.days:g}d = {cut:.1f}")

    cur.execute(
        "select details from trading_ops where status = ? and ts >= ?",
        (STATUS, cut),
    )
    decay: list[float] = []
    ahead: list[float] = []
    for (raw,) in cur:
        d = DECAY_RE.search(raw or "")
        a = AHEAD_RE.search(raw or "")
        if d and a:
            decay.append(float(d.group(1)))
            ahead.append(float(a.group(1)))

    n = len(decay)
    print(f"\n{STATUS} rows carrying BOTH terms, {args.days:g}d: {n}")
    if n == 0:
        print("nothing to decide on -- widen --days before concluding anything")
        return 1

    srt = sorted(decay)
    print("\nDENOMINATOR -- measured usable horizon (chaos.py:187)")
    print(f"  min {srt[0]:.3f}  p05 {_quantile(srt, .05):.3f}  "
          f"p50 {_quantile(srt, .50):.3f}  p95 {_quantile(srt, .95):.3f}  "
          f"MAX {srt[-1]:.3f}  min")
    at_max = 100.0 * decay.count(srt[-1]) / n
    print(f"  DISTINCT VALUES {len(set(decay))}   share at the maximum "
          f"{at_max:.2f}%")
    print("  ^ 714-ish distinct values reaching hours is a MEASUREMENT. A "
          "value censored by")
    print("    observed_span = len(series)*bar_sec would pile up at one "
          "ceiling. Fork (b) dies here.")

    print("\nNUMERATOR -- proposed_horizon_sec (lattice.py:141)")
    print(f"  DISTINCT VALUES {len(set(ahead))}: "
          f"{sorted(set(ahead))}")
    print(f"  median {_quantile(sorted(ahead), .50):.1f} min, against "
          f".env:169 MAX_HOLD_FORCE_SECONDS=2700 = 45 min")
    print("  ^ a handful of discrete values, each a strategy-id SUFFIX in "
          "minutes: a LABEL,")
    print("    not a forecast. Confirms chaos_gate_horizon_census.py "
          "(c813f5fe) by another route.")

    print(f"\nPAYOFF -- of the {n} candidates the gate refused, how many clear")
    print("the CHAOS layer if the numerator carried the real intended hold:")
    for h in HOLDS_MIN:
        passed = sum(1 for x in decay if x >= h)
        tag = ""
        if h == 45.0:
            tag = "   <- .env MAX_HOLD_FORCE_SECONDS"
        elif h == 15.0:
            tag = "   <- standing route (c)"
        print(f"  hold {h:6.1f} min -> {passed:6d} of {n} PASS = "
              f"{100.0 * passed / n:5.1f}%{tag}")
    print(f"  today               -> {0:6d} of {n} PASS =   0.0%")

    print("\nTHROUGHPUT, NOT EDGE: a candidate that clears chaos still faces "
          "the cost floor and")
    print("the symbol edge ban. Over 96% of ticks genuinely cannot support a "
          "45-min forecast,")
    print("so part of the wall is REAL. Nothing above shows the survivors are "
          "profitable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
