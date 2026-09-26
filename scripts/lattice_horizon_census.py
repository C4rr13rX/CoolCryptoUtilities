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

WITHDRAWN BY ITS OWN AUTHOR, SAME PASS, AND THIS IS THE IMPORTANT PART OF THIS
FILE. The first version of this census priced the numerator defect over 14 days
WITHOUT SPLITTING AT THE BOOT, and that is the repo's own standing rule broken:
production booted 2026-09-25 19:59:47 (epoch 1790380787) and pass 680's clamp
(74d5671, 2026-09-23) IS in the running build.

    PRE-BOOT (14d)   n=11556   numerator: 11 distinct, median 1440.0, max 10080
    SINCE BOOT       n=  321   numerator:  1 DISTINCT VALUE, 60.0, max 60.0

One value on every single row: THE CLAMP WORKS, and 97% of the original sample
described code that is no longer running. So the headline "458 of 11,877 at
45 min" is withdrawn -- it priced a defect that was already fixed. What survives
is the DENOMINATOR half above: the decay measurement is real, fork (b) is dead.

WHAT IT IS WORTH, AS A RANGE RATHER THAN A POINT. Share of refused candidates
clearing the CHAOS layer at a given hold ceiling, on both books:

                          since boot (n=321)   pre-boot (n=11556, 37 symbols)
    60 min  (today)              0.0%                    1.91%
    45 min  (MAX_HOLD_FORCE)     0.0%                    3.96%
    15 min  (route (c))          0.0%                   12.73%
     5 min                       5.9%                   34.09%
     3 min                      61.4%                   58.92%

The since-boot decay MAXIMUM is 8.60 min, so every ceiling from 9 min upward is
the same ceiling on the current tape. The defensible statement about standing
route (c) is NOT "shorten the hold to 15 minutes" -- that reaches at best 1
candidate in 8 and at worst none. It is "shorten it to THREE", which is a
different strategy and not a config change.

The p50 is what holds: 3.50 (all pre-boot), 3.00 (pre-boot restricted to the
since-boot symbols), 3.30 (since boot). The p95 tail does NOT reconcile --
38.10 pre-boot against 5.00 since -- and restricting to the same 4 symbols only
takes it to 28.90, so it is not symbol mix. That gap is UNEXPLAINED here.

AND THE THING THIS SCRIPT FOUND WITHOUT LOOKING FOR IT, filed as [a7ec8489]:
distinct symbols reaching this gate went 37 -> 5 across the boot, while the TAPE
carries 47 (17,564 market_stream rows in the same 18.1h, so the feed is alive).
47 on the tape and 5 at the gate is a 9.4x narrowing that is NOT the feed, which
makes standing route (b) -- "restrict to deep pools" -- the wrong shape: the
universe is already restricted, to FLOCK/DRV/HYDX/VELVET/LAPTOP, and WETH-USDT
sits on the tape with 1,015 rows and reaches this gate five times.

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
SYMBOL_RE = re.compile(r"(\S+?): information decays after")

#: Production booted 2026-09-25 19:59:47 -0400. An EXPLICIT EPOCH, because
#: trading_ops.ts is a float and a date-string predicate returns 0 rows with no
#: error. Pass 680's clamp (74d5671, 2026-09-23) is inside this build, so a
#: census that does not split here prices a defect that is already fixed.
BOOT_EPOCH = 1790380787.0

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

    # ---- THE BOOT SPLIT, and it is why the table above is a RANGE ---------
    # Without this the census prices a defect pass 680 already fixed. An
    # explicit epoch, never a date string: trading_ops.ts is a float.
    print(f"\n{'=' * 70}\nBOOT SPLIT -- production booted {BOOT_EPOCH} "
          f"(2026-09-25 19:59:47 -0400)")
    print(f"{(newest - BOOT_EPOCH) / 3600:.1f}h of the current build. Pass 680's "
          "clamp (74d5671) IS in it.")
    for label, lo, hi in (("PRE-BOOT", cut, BOOT_EPOCH),
                          ("SINCE BOOT", BOOT_EPOCH, newest + 1.0)):
        cur.execute(
            "select details from trading_ops where status = ? "
            "and ts >= ? and ts < ?", (STATUS, lo, hi))
        dec: list[float] = []
        ahd: set[float] = set()
        syms: set[str] = set()
        for (raw,) in cur:
            d = DECAY_RE.search(raw or "")
            a = AHEAD_RE.search(raw or "")
            s = SYMBOL_RE.search(raw or "")
            if d:
                dec.append(float(d.group(1)))
            if a:
                ahd.add(float(a.group(1)))
            if s:
                syms.add(s.group(1))
        if not dec:
            print(f"  {label:<11} n=0")
            continue
        ds = sorted(dec)
        print(f"\n  {label:<11} n={len(ds)}  numerator: {len(ahd)} distinct "
              f"{sorted(ahd) if len(ahd) <= 3 else ''}")
        print(f"              decay p50 {_quantile(ds, .50):.2f}  "
              f"p95 {_quantile(ds, .95):.2f}  MAX {ds[-1]:.2f} min   "
              f"SYMBOLS REACHING THE GATE: {len(syms)}")
        for h in (60.0, 45.0, 15.0, 5.0, 3.0):
            p = sum(1 for x in ds if x >= h)
            print(f"              clamp {h:5.1f} min -> {p:6d} clear = "
                  f"{100.0 * p / len(ds):6.2f}%")

    # The tape itself, so "5 symbols at the gate" is read against a
    # denominator rather than alone. Read-only; db.py is not opened.
    try:
        cur.execute("select count(*) from market_stream")
        print(f"\n  CONTROL market_stream total rows: {cur.fetchone()[0]}")
        for label, lo, hi in (("PRE-BOOT", cut, BOOT_EPOCH),
                              ("SINCE BOOT", BOOT_EPOCH, newest + 1.0)):
            cur.execute("select count(distinct symbol), count(*) "
                        "from market_stream where ts >= ? and ts < ?", (lo, hi))
            ns, nr = cur.fetchone()
            print(f"  TAPE {label:<11} distinct symbols {ns:<5} rows {nr}")
        print("  ^ the feed is ALIVE. Symbols on the tape vs symbols reaching "
              "the gate is a")
        print("    SELECTION defect, not a feed defect. Filed as [a7ec8489].")
    except sqlite3.Error as exc:
        print(f"\n  market_stream unreadable ({exc}) -- the gate-side counts "
              "above stand alone")

    print("\nTHROUGHPUT, NOT EDGE: a candidate that clears chaos still faces "
          "the cost floor and")
    print("the symbol edge ban. Most ticks genuinely cannot support a 15-min "
          "forecast, so part")
    print("of the wall is REAL. Nothing above shows the survivors are "
          "profitable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
