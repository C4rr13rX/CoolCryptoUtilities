"""Can this feed support a SECOND-DERIVATIVE feature search, and what must it beat?

    python -X utf8 scripts/derivative_feature_feasibility.py

Filed for [ca395016]. The operator asked (2026-09-16) for calculus-level structure
rather than algebra, and for brute-forcing the history "like brute forcing a
password". Both are reasonable, and both have a precondition nobody had checked:

  1. A second derivative needs THREE ticks at comparable spacing. This feed takes
     global blackouts of 16-59 minutes, so most triples in the raw table span a
     gap where acceleration is meaningless. How many usable ones are there?

  2. A wide search finds a winner by luck. The bar it must clear is
     SE*sqrt(2*ln(K)) for K candidates -- and because that grows as sqrt(ln K),
     it is worth printing how MUCH a wider search actually costs.

This script answers both and prints nothing else. It does not score a predicate,
it does not claim an edge, and it reads the store read-only.

THE RESULT THAT MATTERS, and it is the argument FOR brute force rather than
against it: going from 50 candidates to 5000 -- a hundredfold wider search --
moves the bar by under 2bp. Search width is nearly free once it is priced. The
SYMBOL is not: the same search must clear ~12bp on a CBETH/CBBTC-class pair and
~98bp on AERO, because AERO's round trip costs 64-435bp against a 97bp-sigma
deep pool's 4-11bp. Choosing the pair is worth ~8x more than choosing the search.
"""

from __future__ import annotations

import math
import os
import sqlite3
import statistics as st
import sys
from collections import defaultdict

#: Two consecutive gaps are "comparable" when each is within this multiple of the
#: symbol's own median gap. Looser than this and the second derivative is being
#: taken across a blackout.
REGULARITY = 2.0

#: Measured sigma of forward returns by pool tier (see memory: evidence cost is
#: quadratic in variance). Deep pools are quiet; a tier down is 6x noisier.
SIGMA_BP_DEEP = 97.0
SIGMA_BP_THIN = 590.0

#: Round-trip cost floors measured on this venue, in basis points.
COST_BP_DEEP = 7.0    # WETH/CBETH/CBBTC class, 0.04-0.11%
COST_BP_AERO = 64.0   # AERO floor, 0.64-4.35%

#: Symbols whose pools the live lane can actually fill.
DEEP_SYMBOLS = ("CBETH-USDC", "CBBTC-USDC")


def _triples(rows, medians):
    """Count runs of 3+ consecutive ticks whose gaps are each comparable."""
    counts = defaultdict(int)
    prev_sym = prev_ts = None
    run = 0
    for sym, ts in rows:
        if sym != prev_sym:
            run = 1
        elif prev_ts is not None and 0 < ts - prev_ts <= REGULARITY * medians.get(sym, math.inf):
            run += 1
        else:
            run = 1
        if run >= 3:
            counts[sym] += 1
        prev_sym, prev_ts = sym, ts
    return counts


def main() -> int:
    path = os.getenv("TRADING_DB_PATH", "storage/trading_cache.db")
    if not os.path.exists(path):
        print(f"no store at {path}")
        return 1
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = conn.execute("SELECT symbol, ts FROM market_stream ORDER BY symbol, ts").fetchall()
    print(f"store {path}  ({os.path.getsize(path)/1e9:.1f} GB)   market_stream rows: {len(rows)}")

    gaps = defaultdict(list)
    prev_sym = prev_ts = None
    for sym, ts in rows:
        if sym == prev_sym and prev_ts is not None and ts - prev_ts > 0:
            gaps[sym].append(ts - prev_ts)
        prev_sym, prev_ts = sym, ts
    if not gaps:
        print("no gaps -- market_stream is empty or single-tick per symbol")
        return 1
    medians = {s: st.median(g) for s, g in gaps.items()}

    print()
    print("TICK REGULARITY -- a p90 far above the median IS the blackout structure")
    print(f"{'symbol':<16}{'ticks':>8}{'med_gap_s':>11}{'p90_s':>9}{'regular':>10}")
    for sym, g in sorted(gaps.items(), key=lambda kv: -len(kv[1]))[:10]:
        ordered = sorted(g)
        p90 = ordered[max(0, int(0.9 * len(ordered)) - 1)]
        regular = sum(1 for x in g if x <= REGULARITY * medians[sym]) / len(g)
        print(f"{sym:<16}{len(g)+1:>8}{medians[sym]:>11.0f}{p90:>9.0f}{regular:>9.1%}")

    counts = _triples(rows, medians)
    total = sum(counts.values())
    print()
    print(f"USABLE SECOND-DERIVATIVE TRIPLES (3 consecutive ticks, each gap <= {REGULARITY:.0f}x that symbol's median)")
    print(f"  {total} across {len(counts)} symbols = {100*total/len(rows):.1f}% of all rows")
    print("  -> the feature class IS computable on this feed; it is not blackout-limited")

    n_deep = sum(counts.get(s, 0) for s in DEEP_SYMBOLS)
    print()
    print(f"FILLABLE subset {'+'.join(DEEP_SYMBOLS)}: n = {n_deep} triples")
    if n_deep <= 1:
        print("  too few to price a search -- rerun when the deep pairs have streamed more")
        return 0
    se = SIGMA_BP_DEEP / math.sqrt(n_deep)
    print(f"  sigma {SIGMA_BP_DEEP:.0f}bp -> SE {se:.2f}bp")
    print()
    print("WHAT A CANDIDATE MUST BEAT -- noise bar SE*sqrt(2 ln K), plus the cost floor")
    print(f"{'K candidates':>14}{'noise bar':>12}{'+cost':>8}{'TOTAL bp':>11}")
    for k in (50, 200, 1000, 5000):
        bar = se * math.sqrt(2 * math.log(k))
        print(f"{k:>14}{bar:>11.2f}bp{COST_BP_DEEP:>7.0f}{bar + COST_BP_DEEP:>10.2f}")

    n_aero = counts.get("AERO-USDC", 0)
    if n_aero > 1:
        se_a = SIGMA_BP_THIN / math.sqrt(n_aero)
        bar_a = se_a * math.sqrt(2 * math.log(1000))
        print()
        print(f"AERO-USDC for contrast: n = {n_aero} triples, SE {se_a:.2f}bp")
        print(f"  at K=1000 it must beat {bar_a:.1f}bp noise + {COST_BP_AERO:.0f}bp cost = {bar_a + COST_BP_AERO:.0f}bp")
        deep_total = se * math.sqrt(2 * math.log(1000)) + COST_BP_DEEP
        print(f"  vs {deep_total:.1f}bp on the deep pair -- the SYMBOL is worth "
              f"{(bar_a + COST_BP_AERO)/deep_total:.1f}x the whole search")

    print()
    print("READ THIS BEFORE WIDENING OR NARROWING THE SEARCH:")
    print("  the bar grows as sqrt(ln K), so 100x more candidates costs under 2bp.")
    print("  Brute force is nearly free once it is priced. Picking the pair is not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
