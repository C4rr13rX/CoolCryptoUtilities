"""Which symbol can we PROVE an edge on, and how many trades does it take?

WHY THIS EXISTS, and it is not another cost sweep. This repo already chooses
its tradeable universe on COST -- WETH round trips 0.04-0.11%, AERO 0.64-4.35%,
KTA to 74.98% -- and that is the right first cut. It is not the only one. A
symbol also has a DETECTION FLOOR: the smallest per-trade edge that its own
dispersion lets you distinguish from zero in a realistic number of trades. Two
symbols can cost the same and be a hundred times apart on how long it takes to
prove anything about either.

THE ARITHMETIC, stated once. For a per-trade edge ``e`` against a baseline, on
a symbol whose per-trade forward return has standard deviation ``sd``, the
standard error of the mean over ``n`` trades is ``sd/sqrt(n)``. To clear two
standard errors:

    n  =  (1.96 * sd / e) ** 2

n is QUADRATIC in sd. Doubling a symbol's dispersion quadruples the evidence
bill. That is the same law as [evidence-cost-is-quadratic-in-variance] and
[graduation-bar-has-no-statistical-power]; this script prices it per symbol so
a route can be chosen on it rather than argued about.

MEASURED 2026-09-16, pass 430: the L2 held-out arm on AERO-USDC found sd of
2.2-5.1% per 12-bar trade, not the ~2% this repo had been assuming, which made
a 0.65pp edge need ~236 trades to resolve in a 588-bar window that could never
supply them. That is what motivated this census.

THE HORIZON IS FIXED IN WALL CLOCK, NOT IN BARS. ``data/historical_ohlcv``
spans 166s to 345600s bars, so "12 bars" is 33 minutes on one file and 48 days
on another, and dispersion grows with the holding period. Comparing symbols at
a fixed BAR count compares holding periods, not symbols. Every row here is
computed at the same number of SECONDS, and the bar count used per symbol is
printed.

NO NODE, NO NETWORK, NO TRADING STORE. Reads data/historical_ohlcv only.

    python -X utf8 scripts/symbol_resolvability_census.py --horizon-sec 3600
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
from pathlib import Path
from typing import Any, Dict, List

CORPORA = Path("data/historical_ohlcv/base")


def load(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    bars = raw if isinstance(raw, list) else raw.get("bars") or raw.get("data") or []
    bars = [b for b in bars if b.get("close") and b.get("timestamp")]
    bars.sort(key=lambda b: int(b["timestamp"]))
    return bars


def cadence(bars: List[Dict[str, Any]]) -> float:
    gaps = [int(bars[i + 1]["timestamp"]) - int(bars[i]["timestamp"])
            for i in range(min(len(bars) - 1, 4000))]
    gaps = [g for g in gaps if g > 0]
    return st.median(gaps) if gaps else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon-sec", type=int, default=3600,
                    help="holding period in SECONDS, identical for every "
                         "symbol. Bars per symbol are derived from it.")
    ap.add_argument("--edge-bp", type=float, default=20.0,
                    help="the per-trade edge whose evidence bill we are "
                         "pricing, in basis points. 20bp is the top of the "
                         "range any strategy here has ever plausibly earned.")
    ap.add_argument("--quote", default="USDC",
                    help="keep only pairs quoted in this asset -- the live "
                         "book is USDC-quoted, and a WETH-quoted corpus prices "
                         "a different trade than the one we place.")
    ap.add_argument("--peg-floor", type=float, default=0.05,
                    help="per-trade sd below this PERCENT is treated as a peg "
                         "and kept out of the headline: a stablecoin pair has "
                         "no direction to be right about, so an edge that is "
                         "trivially detectable there is also trivially absent.")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--min-bars", type=int, default=2000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    # ONE FILE PER SYMBOL, the largest. Several corpora share a symbol at
    # different cadences; averaging them would mix holding periods again.
    best: Dict[str, Path] = {}
    for p in CORPORA.glob("*.json"):
        sym = p.stem.split("_", 1)[-1].upper()
        if not sym.endswith("-" + args.quote.upper()):
            continue
        prior = best.get(sym)
        if prior is None or p.stat().st_size > prior.stat().st_size:
            best[sym] = p

    picked = sorted(best.items(), key=lambda kv: -kv[1].stat().st_size)[:args.limit]
    print(f"{len(best)} {args.quote}-quoted symbols on disk; measuring the "
          f"{len(picked)} deepest at a fixed {args.horizon_sec}s holding "
          f"period, pricing a {args.edge_bp:.0f}bp edge at 2 SE.\n")

    rows: List[Dict[str, Any]] = []
    for sym, path in picked:
        try:
            bars = load(path)
        except Exception as exc:                      # a corpus we cannot read
            print(f"  {sym:<18} UNREADABLE ({type(exc).__name__})")
            continue
        if len(bars) < args.min_bars:
            continue
        cad = cadence(bars)
        if cad <= 0:
            continue
        h = max(1, round(args.horizon_sec / cad))
        if h >= len(bars) // 4:
            continue
        rets = []
        for i in range(0, len(bars) - h):
            c0 = float(bars[i]["close"])
            c1 = float(bars[i + h]["close"])
            if c0 > 0 and c1 > 0:
                rets.append(c1 / c0 - 1.0)
        if len(rets) < args.min_bars:
            continue
        sd = st.pstdev(rets)
        edge = args.edge_bp / 10000.0
        need = (1.96 * sd / edge) ** 2
        rows.append({
            "symbol": sym, "bars": len(bars), "cadence_sec": cad,
            "horizon_bars": h, "sd": sd, "n_trades_to_resolve": need,
            "days_at_4_per_day": need / 4.0,
            "min_resolvable_edge_bp_at_n30": 1.96 * sd / math.sqrt(30) * 10000,
        })

    rows.sort(key=lambda r: r["n_trades_to_resolve"])
    print(f"{'symbol':<18}{'bars':>8}{'cad':>8}{'h':>5}{'sd/trade':>11}"
          f"{'trades to prove':>17}{'yrs @4/day':>12}{'floor@n=30':>12}")
    print("-" * 91)
    for r in rows:
        print(f"{r['symbol']:<18}{r['bars']:>8}{r['cadence_sec']:>8.0f}"
              f"{r['horizon_bars']:>5}{r['sd']*100:>10.3f}%"
              f"{r['n_trades_to_resolve']:>17,.0f}"
              f"{r['days_at_4_per_day']/365:>12,.1f}"
              f"{r['min_resolvable_edge_bp_at_n30']:>11.0f}bp")

    if rows:
        print(f"\nREAD IT AS A BILL, NOT A RANKING. The last column is the "
              f"SMALLEST per-trade edge that 30 trades can tell from zero on "
              f"that symbol. Every strategy in this repo is judged on ~20-30 "
              f"ghost trades, and every edge it has ever claimed is 5-20bp.")
        # A PEG IS NOT THE CHEAPEST SYMBOL, IT IS A DEGENERATE ONE. USDT-USDC
        # reads sd 0.011% and "0 trades to prove", and naming it as the winner
        # is how a true number becomes a false recommendation: there is no
        # direction to be right about on a stablecoin pair, so an edge that is
        # trivially detectable there is also trivially absent. Excluded from the
        # headline and SAID so, rather than silently dropped from the table.
        real = [r for r in rows if r["sd"] >= args.peg_floor / 100.0]
        pegs = [r for r in rows if r not in real]
        if pegs:
            print(f"EXCLUDED AS PEGS (sd < {args.peg_floor:.2f}%, no direction "
                  f"to be right about): "
                  f"{', '.join(r['symbol'] for r in pegs)}. They are still in "
                  f"the table above; they are not candidates.")
        if real:
            print(f"CHEAPEST REAL EVIDENCE: {real[0]['symbol']} at "
                  f"{real[0]['sd']*100:.3f}% per trade needs "
                  f"{real[0]['n_trades_to_resolve']:,.0f} trades for "
                  f"{args.edge_bp:.0f}bp; the dearest here needs "
                  f"{real[-1]['n_trades_to_resolve']:,.0f}.")
        print(f"POOLED sd HIDES REGIME CHANGE. Measured pass 430: AERO's "
              f"quarters run 1.997%/1.313%/1.298%/1.052%, so its pooled bill "
              f"vs WETH reads 4.53x where the typical quarter is ~2.9x. Quote "
              f"the RANGE, not this table's single figure.")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2),
                                       encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
