"""How many trades does it take to PROVE an edge, at each pool depth?

WHY THIS EXISTS
---------------
scripts/tradeable_book_repricing.py (31167be) measured gross return per trade
by pool depth and found no tier distinguishable from zero. The useful part was
not the means, it was the SPREADS: >$1M pools carry a 95% interval of +/-22.8bp
where $100k-$1M carries +/-136bp, at comparable n.

That difference is not cosmetic. The trades needed to resolve an edge of size
e from zero goes as

    n  =  (1.96 * sigma / e) ** 2

-- QUADRATIC in sigma. Six times the noise is THIRTY-SIX times the trades. So
"we are 62 days from graduating anything" is not purely an evidence-RATE
problem: it is partly a choice of where we gather the evidence. The same 20
ghost trades that prove nothing in a $50k pool can resolve a real edge in a
$10M one.

WHAT IT PRINTS
  1. sigma of per-trade gross return, in bp, per depth tier, from the book.
  2. the trades required to resolve a 5 / 10 / 20 / 50bp edge in each tier.
  3. where our evidence ACTUALLY sits today, against where it would be cheapest
     to gather.
  4. whether the graduation bar (20 tradeable trades) can detect anything at
     all in the tier we currently trade.

CAVEATS. sigma is estimated from the rows this book has, so a tier with few
rows has a noisy sigma too -- the row count is printed beside every one. This
is a power calculation on GROSS return; it says what is RESOLVABLE, never that
an edge exists. Nothing here is a claim that any tier is profitable.

Run:  python -X utf8 scripts/evidence_power_by_depth.py
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tradeable_book_repricing import TIERS, build  # noqa: E402

EDGES_BP = [5.0, 10.0, 20.0, 50.0]
GRADUATION_N = 20  # the bar: 20 TRADEABLE ghost trades


def stdev(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def n_required(sigma: float, edge_bp: float) -> float:
    if not math.isfinite(sigma) or edge_bp <= 0:
        return float("nan")
    return (1.96 * sigma / edge_bp) ** 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db"))
    ap.add_argument("--rate-per-day", type=float, default=3.9,
                    help="tradeable ghost evidence rate, from graduation_status")
    args = ap.parse_args()

    rows = build(args.db, 0.50, 20.0)["rows"]
    by_tier: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)

    order = [t[0] for t in TIERS] + ["UNRESOLVED"]
    W = 78
    print("=" * W)
    print("WHAT IT COSTS TO PROVE AN EDGE, BY POOL DEPTH")
    print("=" * W)
    print("n = (1.96*sigma/edge)^2 -- QUADRATIC in sigma. Twice the noise is")
    print("four times the trades. This says what is RESOLVABLE, never that an")
    print("edge exists.")
    print()
    print(f"{'TIER':<13}{'n':>5}{'sigma bp':>10}{'   TRADES TO RESOLVE AN EDGE OF':<10}")
    print(f"{'':<13}{'':>5}{'':>10}{'5bp':>10}{'10bp':>10}{'20bp':>10}{'50bp':>10}")
    sigmas = {}
    for t in order:
        g = by_tier.get(t) or []
        if len(g) < 2:
            continue
        s = stdev([r["gross_bp"] for r in g])
        sigmas[t] = (s, len(g))
        cells = "".join(f"{n_required(s, e):>10,.0f}" for e in EDGES_BP)
        print(f"{t:<13}{len(g):>5}{s:>10.1f}{cells}")
    print("  sigma itself is estimated from those n rows -- a tier with few")
    print("  rows has a noisy sigma. The n column is the honest caveat.")
    print()

    print("-" * W)
    print("WHERE OUR EVIDENCE ACTUALLY SITS")
    total = sum(len(v) for v in by_tier.values())
    for t in order:
        g = by_tier.get(t) or []
        if not g:
            continue
        share = len(g) / total * 100.0
        bar = "#" * int(round(share / 2))
        print(f"  {t:<13}{len(g):>5} rows {share:>6.1f}%  {bar}")
    deep_n = len(by_tier.get(">$1M") or [])
    print(f"  {deep_n} of {total} rows ({deep_n/total*100:.1f}%) are in the only tier")
    print("  where the interval is tight enough to resolve a realistic edge.")
    print()

    print("-" * W)
    print(f"CAN THE GRADUATION BAR DETECT ANYTHING? bar = {GRADUATION_N} tradeable trades")
    for t in order:
        if t not in sigmas:
            continue
        s, n = sigmas[t]
        detectable = 1.96 * s / math.sqrt(GRADUATION_N)
        print(f"  {t:<13} smallest edge {GRADUATION_N} trades can resolve: "
              f"{detectable:>8.1f}bp")
    print()
    # The bar's OTHER leg is a win rate, and that one is exactly computable.
    def binom_ge(k: int, n: int, p: float) -> float:
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    print("  THE WIN-RATE LEG IS EXACT -- pure binomial, no estimation. The bar")
    print("  wants 55% of n. What a p=0.5 COIN does against it, at each of the")
    print("  three values this system actually uses:")
    print(f"    {'n':>4}  {'source':<44}{'needs':>8}{'coin passes':>13}{'of 41':>8}")
    for n_bar, src in ((20, "graduation_status.py display"),
                       (25, "MIN_GHOST_TRADES_FOR_PROMOTION, pipeline.py:928"),
                       (30, "GHOST_TRADEABLE_MIN_TRADES, pipeline.py:5101")):
        need = math.ceil(0.55 * n_bar)
        p = binom_ge(need, n_bar, 0.5)
        print(f"    {n_bar:>4}  {src:<44}{need:>5}/{n_bar:<2}{p*100:>12.1f}%"
              f"{41*p:>7.0f}")
    print("  The display and the code DISAGREE on the bar. A bar nobody can")
    print("  quote is a bar nobody can audit -- reconcile that before tuning it.")
    print()
    print("  Compare against this repo's own numbers: the cost floor is ~59bp")
    print("  two-leg at the flat model, WETH round-trips for 4-11bp, and the")
    print("  selection bar on this tape is 5-12bp. An edge the bar cannot")
    print("  resolve is an edge the bar approves or rejects BY NOISE.")
    print()

    print("-" * W)
    print(f"DAYS OF EVIDENCE, at the measured {args.rate_per_day}/day tradeable rate")
    print(f"{'TIER':<13}{'  to resolve 10bp':>20}{'  to resolve 20bp':>20}")
    for t in order:
        if t not in sigmas:
            continue
        s, _ = sigmas[t]
        d10 = n_required(s, 10.0) / args.rate_per_day
        d20 = n_required(s, 20.0) / args.rate_per_day
        print(f"{t:<13}{d10:>18,.0f}d{d20:>19,.0f}d")
    print("  This assumes the WHOLE evidence rate is spent in one tier. Today")
    print("  it is spread across all of them, so every tier is slower than the")
    print("  line above says.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
