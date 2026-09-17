"""Finish [bc7b37c2]: re-price the ghost book per token, and put an ERROR BAR
on the answer.

WHAT THIS ADDS TO scripts/deep_pool_reprice.py (ec1ee5f)
--------------------------------------------------------
That script re-priced the book at each symbol's own measured round-trip cost
and concluded: "DEEP pools are negative on GROSS (-0.0291), before a cent of
cost. DIRECTION sets the sign, not symbol selection."

That conclusion is drawn from SIXTEEN ROUND TRIPS. This repo's own selection
bar on this tape is 5-12bp, and a 16-row sample cannot resolve 5bp from zero.
The claim may still be true; it is not yet MEASURED. So this script does four
things the first one does not:

  1. WIN RATE AND PER-TRADE NET for the restricted universe against the full
     one, with the trade count and the symbol count that survive the cut.

  2. AN ERROR BAR. Every bucket is reported as mean GROSS IN BASIS POINTS OF
     NOTIONAL with a standard error and a 95% interval. Dollar sums across a
     book that mixes clip sizes are not comparable -- this repo has shipped
     that bug ("ghost book prices rows at mixed clips"). bp/trade is.

  3. THE NON-MONOTONE COUNT. The quote curve is not monotone in size: AERO
     reads 0.6395% at $2, 4.3502% at $10 and 0.1330% at $50; CBBTC worsens
     from 0.0617% to 1.4659% between $10 and $50. Every trip whose clip lands
     between an inverted pair is COUNTED AND NAMED, because which side of that
     inversion a trip sits on changes its cost by up to 70x.

  4. THE WHOLE BOOK BY DEPTH, NOT JUST THE 6 QUOTED SYMBOLS. Only 6 symbols
     were ever quoted, which is 68 of 237 rows. But the GROSS question -- does
     direction work in deep pools? -- needs no cost quote at all. It needs a
     DEPTH classification, and data/pool_depth_snapshot.json has one for 153
     symbols. Bucketing the full book by pool liquidity tests the same claim on
     a real sample instead of on 16 rows.

WHICH QUESTION THE NUMBER ANSWERS. Re-pricing THESE trades under a deep-pool
cost answers "what would the trades the current selector took have cost in a
deeper pool". It does NOT answer "what would a deep-pool strategy have earned",
because a deep-pool strategy would have taken DIFFERENT TRADES. The ghost
book's symbol mix was produced by the current selector, so every number here
inherits that selection.

THESE ARE QUOTES, NOT FILLS. Read-only router quotes, USDC in and back.
Slippage and MEV are NOT modelled. The live lane measured ~27.8bp of real
slippage (21.73 buy + 6.04 sell, n=12) that no ghost row carries. Every
"after" figure here is a COST FLOOR -- the best case -- not a realised P/L.

Run:  python -X utf8 scripts/tradeable_book_repricing.py
      python -X utf8 scripts/tradeable_book_repricing.py --json out.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.deep_pool_reprice as dpr  # noqa: E402
from scripts.deep_pool_reprice import (  # noqa: E402
    MAX_CLIP_EXTRAPOLATION,
    CensusArtifactError,
    _base_symbol,
    _move_pct,
    _notional,
    load_quote_census,
    load_rows,
    provenance_lines,
    round_trip_pct,
)

# ONE SOURCE, TWO READERS. The cost table is not defined here and is not
# defined in deep_pool_reprice.py either: both read
# data/round_trip_quote_census_<UTC stamp>.json, written by
# scripts/routing_arbitrage_census.py --round-trip-census.
#
# DO NOT RE-ADD `MEASURED_ROUND_TRIP_PCT = dpr.MEASURED_ROUND_TRIP_PCT` HERE.
# That line stood at this spot until pass 458. It was never read -- every site
# in this file goes through `dpr.` and so does the imported round_trip_pct,
# whose globals are dpr's, which is why --census did work end to end (verified
# with a synthetic artifact: the AERO rate moved 0.6395 -> 0.2222, BASECAT
# entered the priced table, CBBTC/SOL left it). But it was a loaded gun: a
# module-level SNAPSHOT sitting under the exact name the read sites want, taken
# at IMPORT time, i.e. before any --census override at line ~261 can run. Any
# future `MEASURED_ROUND_TRIP_PCT.get(...)` written here would silently read the
# legacy six-symbol literal and every printed number would be the legacy table.
# Read the table through `dpr.MEASURED_ROUND_TRIP_PCT` at CALL time, always.

DEPTH_SNAPSHOT = "data/pool_depth_snapshot.json"

# Depth tiers in USD of pool liquidity. The >$1M bar is the one this repo
# already uses to call a pool fillable (predictability-and-fillability census).
TIERS: List[Tuple[str, float, float]] = [
    (">$1M", 1_000_000.0, float("inf")),
    ("$100k-$1M", 100_000.0, 1_000_000.0),
    ("$10k-$100k", 10_000.0, 100_000.0),
    ("<$10k", 0.0, 10_000.0),
]


def load_depths(path: str) -> Dict[str, Optional[float]]:
    p = Path(path)
    if not p.exists():
        return {}
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {k.upper(): v for k, v in doc.items() if not k.startswith("_")}


def tier_of(depth: Optional[float]) -> str:
    if depth is None:
        return "UNRESOLVED"
    for name, lo, hi in TIERS:
        if lo <= depth < hi:
            return name
    return "<$10k"


def non_monotone_span(symbol: str) -> Optional[Tuple[float, float]]:
    """The size range over which this symbol's quoted cost curve INVERTS.

    Returns (lo_size, hi_size) spanning every adjacent pair where cost falls as
    size rises, or None when the curve is monotone. A trip priced inside this
    span could have been charged a wildly different rate at a nearby clip.
    """
    table = dpr.MEASURED_ROUND_TRIP_PCT.get(_base_symbol(symbol))
    if not table or len(table) < 2:
        return None
    pts = sorted(table)
    lo = hi = None
    for (s0, c0), (s1, c1) in zip(pts, pts[1:]):
        if c1 < c0:  # cost DROPS as size rises -- an inversion
            lo = s0 if lo is None else min(lo, s0)
            hi = s1 if hi is None else max(hi, s1)
    return (lo, hi) if lo is not None else None


def mean_se(xs: List[float]) -> Tuple[float, float, int]:
    n = len(xs)
    if n == 0:
        return 0.0, float("nan"), 0
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n), n


def ci_str(m: float, se: float, n: int) -> str:
    if n < 2 or not math.isfinite(se):
        return "n<2, no interval"
    lo, hi = m - 1.96 * se, m + 1.96 * se
    verdict = "!= 0" if (lo > 0 or hi < 0) else "INDISTINGUISHABLE FROM 0"
    return f"[{lo:+8.1f}, {hi:+8.1f}] {verdict}"


def load_rows_by_mode(db_path: str, mode: Optional[str]) -> List[Dict[str, Any]]:
    """trade_outcomes carries BOTH books. details.mode is 'ghost' or 'live'.

    Of 237 rows, 215 are ghost and 22 are live. Every re-pricing in this repo
    before 2026-09-16 loaded all 237 and called the result the ghost book, so
    the ghost numbers were ~9% live-contaminated. mode=None restores that old
    behaviour for comparison; it is never the default.
    """
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            "SELECT symbol, quantity, entry_price, exit_price, gross_profit, "
            "fee_cost, net_profit, status, ts, details FROM trade_outcomes"
        )
        out = []
        for r in cur.fetchall():
            d = dict(r)
            try:
                meta = json.loads(d.pop("details", None) or "{}")
            except (TypeError, ValueError):
                meta = {}
            d["mode"] = meta.get("mode")
            if mode is not None and d["mode"] != mode:
                continue
            out.append(d)
        return out
    finally:
        con.close()


def build(db_path: str, depth_bar: float, max_move_pct: float,
          mode: Optional[str] = "ghost") -> Dict[str, Any]:
    rows = load_rows_by_mode(db_path, mode)
    depths = load_depths(DEPTH_SNAPSHOT)

    kept: List[Dict[str, Any]] = []
    censored = 0
    for r in rows:
        mv = _move_pct(r)
        notional = _notional(r)
        if mv is None or notional is None:
            continue
        if abs(mv) > max_move_pct:
            censored += 1
            continue
        sym = (r.get("symbol") or "").upper()
        base = _base_symbol(sym)
        gross = float(r.get("gross_profit") or 0.0)
        rate = round_trip_pct(sym, notional)
        fee_meas = (rate / 100.0) * notional if rate is not None else None
        span = non_monotone_span(sym)
        kept.append({
            "symbol": sym,
            "base": base,
            "notional": notional,
            "gross": gross,
            "gross_bp": (gross / notional) * 10_000.0,
            "fee_book": float(r.get("fee_cost") or 0.0),
            "net_before": float(r.get("net_profit") or 0.0),
            "rate": rate,
            "fee_meas": fee_meas,
            "net_after": (gross - fee_meas) if fee_meas is not None else None,
            "depth": depths.get(base),
            "tier": tier_of(depths.get(base)),
            "in_nonmono": bool(span and span[0] <= notional <= span[1]),
            "nonmono_span": span,
        })
    return {"rows": kept, "censored": censored, "depth_bar": depth_bar,
            "n_depths": len(depths)}


def summarise(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    bp = [r["gross_bp"] for r in rows]
    m, se, n = mean_se(bp)
    priced = [r for r in rows if r["net_after"] is not None]
    net_after = sum(r["net_after"] for r in priced)
    net_before = sum(r["net_before"] for r in rows)
    wins_a = sum(1 for r in priced if r["net_after"] > 0)
    wins_b = sum(1 for r in rows if r["net_before"] > 0)
    return {
        "key": key, "n": n, "n_priced": len(priced),
        "symbols": len({r["base"] for r in rows}),
        "gross_bp_mean": m, "gross_bp_se": se,
        "gross": sum(r["gross"] for r in rows),
        "net_before": net_before, "net_after": net_after,
        "win_before": (wins_b / n * 100.0) if n else 0.0,
        "win_after": (wins_a / len(priced) * 100.0) if priced else 0.0,
        "per_trade_before": (net_before / n) if n else 0.0,
        "per_trade_after": (net_after / len(priced)) if priced else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db"))
    ap.add_argument("--depth-bar", type=float, default=0.50)
    ap.add_argument("--max-move-pct", type=float, default=20.0)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--mode", default="ghost",
                    choices=["ghost", "live", "all"],
                    help="which book. trade_outcomes holds BOTH; 'all' is the "
                         "old contaminated behaviour, kept only for comparison")
    ap.add_argument("--census", default=None,
                    help="path to a round_trip_quote_census artifact; default "
                         "is the newest under data/")
    args = ap.parse_args()

    if args.census:
        # Install THROUGH dpr.set_active_census, never by assigning a local
        # name. round_trip_pct's globals are dpr's, and set_active_census also
        # covers the case where dpr was reached as __main__ as well as as
        # scripts.deep_pool_reprice -- two module objects, one source file.
        try:
            dpr.set_active_census(*load_quote_census(args.census))
        except CensusArtifactError as exc:
            print("FATAL: %s" % exc, file=sys.stderr)
            return 2

    mode = None if args.mode == "all" else args.mode
    st = build(args.db, args.depth_bar, args.max_move_pct, mode)
    rows = st["rows"]
    W = 78

    print("=" * W)
    print("GHOST BOOK RE-PRICED PER TOKEN, WITH AN ERROR BAR  [bc7b37c2]")
    print("=" * W)
    print(f"book: {args.mode.upper()}   rows kept {len(rows)}   "
          f"censored |move|>{args.max_move_pct}%: {st['censored']}"
          f"   depth rows available: {st['n_depths']}")
    if mode is None:
        print("  WARNING: --mode all MIXES the ghost and live books. 22 of 237")
        print("  rows are live. This is the pre-2026-09-16 behaviour and is")
        print("  kept only so the contamination can be measured, never quoted.")
    print()
    print("WHICH QUESTION THIS ANSWERS, in one sentence: re-pricing THESE trades")
    print("under a deep-pool cost answers 'what would the trades the current")
    print("selector took have cost in a deeper pool' -- NOT 'what would a")
    print("deep-pool strategy have earned', because a deep-pool strategy would")
    print("have taken DIFFERENT TRADES. Every number below inherits the current")
    print("selector's symbol mix.")
    print()

    for line in provenance_lines(dpr.QUOTE_PROVENANCE):
        print(line)
    print()

    # --- 1. per symbol, measured cost --------------------------------------
    priced = [r for r in rows if r["rate"] is not None]
    print("-" * W)
    print("1. THE %d QUOTED SYMBOLS, PRICED AT THEIR OWN CURVE"
          % len({r["base"] for r in priced}))
    # COVERAGE FIRST. The net below is over PRICED rows only, so the share of
    # the book it speaks for is part of the number, not a footnote. Without
    # this line a reader sees a net and cannot tell it covers a third of the
    # book.
    unpriced = [r for r in rows if r["rate"] is None]
    print(f"  COVERAGE: {len(priced)} of {len(rows)} kept rows carry a quoted "
          f"cost ({len(priced)/max(1,len(rows))*100:.1f}%).")
    print(f"  The other {len(unpriced)} rows are charged NOTHING and cannot "
          f"enter any net below.")
    if unpriced:
        by_un = collections.Counter(r["base"] for r in unpriced)
        print("  unpriced, largest first: " + ", ".join(
            "%s(n=%d)" % (s, c) for s, c in by_un.most_common(8)))
    uq = dpr.QUOTE_PROVENANCE.get("unquotable") or {}
    ur = dpr.QUOTE_PROVENANCE.get("unresolved") or {}
    if uq:
        print("  the census TRIED and could not price: " + ", ".join(
            "%s(rows=%s)" % (s, (d or {}).get("rows")) for s, d in sorted(uq.items())))
    if ur:
        print("  UNRESOLVED (router not answering, verdict withheld): "
              + ", ".join(sorted(ur)))
    print(f"{'SYMBOL':<14}{'n':>4}{'effRATE%':>10}{'rate span':>18}{'gross':>10}"
          f"{'feeBOOK':>10}{'feeMEAS':>10}{'netBEFORE':>11}{'netAFTER':>10}")
    by_sym = collections.defaultdict(list)
    for r in priced:
        by_sym[r["symbol"]].append(r)
    for sym in sorted(by_sym, key=lambda s: -len(by_sym[s])):
        g = by_sym[sym]
        # NOT g[0]['rate']. One symbol's rows draw DIFFERENT rates because they
        # sit at different clips on a quoted curve -- AERO's 39 priced rows drew
        # three. Printing the first row's rate as the symbol's rate made this
        # script disagree with deep_pool_reprice.py on the SAME artifact
        # (0.2222 here, 0.3333 there, measured pass 458). The effective rate is
        # notional-weighted, so it reconciles with feeMEAS by construction.
        notional = sum(x["notional"] for x in g)
        fee_m = sum(x["fee_meas"] for x in g)
        eff = (100.0 * fee_m / notional) if notional > 0 else float("nan")
        rs = sorted({x["rate"] for x in g})
        span = (f"{rs[0]:.4f}-{rs[-1]:.4f} ({len(rs)})" if len(rs) > 1
                else f"{rs[0]:.4f} (1)")
        print(f"{sym:<14}{len(g):>4}{eff:>10.4f}{span:>18}"
              f"{sum(x['gross'] for x in g):>10.4f}"
              f"{sum(x['fee_book'] for x in g):>10.4f}"
              f"{fee_m:>10.4f}"
              f"{sum(x['net_before'] for x in g):>11.4f}"
              f"{sum(x['net_after'] for x in g):>10.4f}")
    tb = sum(r["net_before"] for r in priced)
    ta = sum(r["net_after"] for r in priced)
    print(f"{'TOTAL':<14}{len(priced):>4}{'':>10}{'':>18}"
          f"{sum(r['gross'] for r in priced):>10.4f}"
          f"{sum(r['fee_book'] for r in priced):>10.4f}"
          f"{sum(r['fee_meas'] for r in priced):>10.4f}{tb:>11.4f}{ta:>10.4f}")
    print("  effRATE% is NOTIONAL-WEIGHTED across that symbol's priced rows and")
    print("  is the only single rate that reproduces feeMEAS. 'rate span' is the")
    print("  min-max of the per-row quotes and, in brackets, how many DISTINCT")
    print("  rates the symbol drew: >1 means its rows straddle quoted clips and")
    print("  the symbol HAS no single rate.")
    print(f"  SUBSTITUTION EFFECT: net {tb:+.4f} -> {ta:+.4f}  ({ta - tb:+.4f})")
    print(f"  ...over the {len(priced)} priced rows only, NOT the {len(rows)}-row book.")
    print()

    # --- 2. restricted universe vs full ------------------------------------
    deep = [r for r in priced if r["rate"] < args.depth_bar]
    full = summarise(rows, "FULL BOOK (all rows)")
    dsum = summarise(deep, f"RESTRICTED (quoted rt < {args.depth_bar}%)")
    print("-" * W)
    print("2. RESTRICTED UNIVERSE vs FULL BOOK")
    print(f"{'GROUP':<30}{'n':>5}{'nPRICED':>8}{'syms':>6}{'win%':>7}{'net':>10}{'/trade':>9}")
    print("  net/win/-per-trade are over nPRICED rows only -- the other rows")
    print("  have no quote and are charged NOTHING, so they cannot enter a net.")
    for s in (full, dsum):
        print(f"{s['key']:<30}{s['n']:>5}{s['n_priced']:>8}{s['symbols']:>6}"
              f"{s['win_after']:>7.1f}{s['net_after']:>10.4f}{s['per_trade_after']:>9.4f}")
    print(f"  survive the cut: {dsum['n']} trades, {dsum['symbols']} symbols "
          f"({', '.join(sorted({r['base'] for r in deep}))})")
    print(f"  win rate  full {full['win_before']:.1f}%  restricted {dsum['win_after']:.1f}%")
    print()

    # --- 3. non-monotone exposure ------------------------------------------
    nm = [r for r in priced if r["in_nonmono"]]
    print("-" * W)
    print("3. NON-MONOTONE EXPOSURE -- not smoothed away")
    nm_syms = {s for s in by_sym if non_monotone_span(s)}
    worst = 0.0
    for s in sorted(nm_syms):
        span = non_monotone_span(s)
        tbl = dpr.MEASURED_ROUND_TRIP_PCT.get(_base_symbol(s)) or []
        vals = [b for _a, b in tbl if b > 0]
        if len(vals) >= 2:
            worst = max(worst, max(vals) / min(vals))
        print(f"  {s:<14} curve inverts over ${span[0]:g}-${span[1]:g}: "
              + " ".join(f"${a:g}={b:.4f}%" for a, b in sorted(tbl)))
    if not nm_syms:
        print("  no quoted symbol's cost curve inverts on this grid.")
    print(f"  {len(nm)} of {len(priced)} priced trips ({len(nm)/max(1,len(priced))*100:.1f}%) "
          f"sit at a clip INSIDE an inverted span.")
    print("  Their cost is therefore uncertain by the full width of that")
    print(f"  inversion -- up to {worst:.0f}x across the quoted clips of the worst")
    print("  symbol -- and that is a real uncertainty in the answer, not a")
    print("  rounding detail.")
    print()

    # --- 4. the gross question, on the WHOLE book, by depth -----------------
    print("-" * W)
    print("4. DOES DIRECTION WORK IN DEEP POOLS? -- gross, before ANY cost,")
    print("   on the WHOLE book by pool depth, not just the 6 quoted symbols.")
    print(f"{'TIER':<14}{'n':>5}{'syms':>6}{'grossBP/trade':>15}{'  95% CI (bp)':<34}")
    by_tier = collections.defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)
    order = [t[0] for t in TIERS] + ["UNRESOLVED"]
    tier_out = {}
    for t in order:
        g = by_tier.get(t) or []
        if not g:
            continue
        m, se, n = mean_se([r["gross_bp"] for r in g])
        syms = len({r["base"] for r in g})
        print(f"{t:<14}{n:>5}{syms:>6}{m:>15.1f}  {ci_str(m, se, n)}")
        tier_out[t] = {"n": n, "symbols": syms, "gross_bp": m, "se": se}
    print()
    print("  THE PRIOR PASS'S VERDICT, RE-TESTED. ec1ee5f concluded 'DEEP pools")
    print("  are negative on GROSS, direction sets the sign' from 16 quoted")
    print("  rows. Read the >$1M line above: that is the same claim on every")
    print("  row this book has, and the CI is what says whether it holds.")
    print()

    # --- 5. does DEPTH predict COST? ---------------------------------------
    print("-" * W)
    print("5. DOES POOL DEPTH PREDICT ROUND-TRIP COST? -- checking our own proxy")
    depths = load_depths(DEPTH_SNAPSHOT)
    table = dpr.MEASURED_ROUND_TRIP_PCT
    chk = []
    for s, tbl in sorted(table.items()):
        dep = depths.get(s)
        pts = sorted(tbl)
        if dep is None:
            print(f"  {s:<9}{'(no depth reading)':>20}  "
                  + "  ".join(f"${a:g}={b:.4f}%" for a, b in pts[:4]))
            continue
        print(f"  {s:<9}{dep:>20,.0f}  "
              + "  ".join(f"${a:g}={b:.4f}%" for a, b in pts[:4]))
        chk.append((dep, pts[0][1], s))
    print(f"  Only {len(chk)} of {len(table)} quoted symbols have a")
    print("  depth reading, so this cannot be a correlation -- it is a spot")
    print("  check on whichever symbols the census reached.")
    if len(chk) >= 2:
        ranked = sorted(chk, key=lambda t: -t[0])
        deepest, shallowest = ranked[0], ranked[-1]
        print(f"  deepest pool  {deepest[2]:<9} {deepest[0]:>16,.0f}  "
              f"smallest-clip rt {deepest[1]:.4f}%")
        print(f"  thinnest pool {shallowest[2]:<9} {shallowest[0]:>16,.0f}  "
              f"smallest-clip rt {shallowest[1]:.4f}%")
        if deepest[1] > shallowest[1]:
            print("  IT FAILS on this sample: the DEEPER pool round-trips DEARER.")
        else:
            print("  It is consistent on this sample, which is not the same as")
            print("  established -- two points cannot carry a proxy.")
    print("  CONSEQUENCE: DexScreener pool liquidity is NOT established as a")
    print("  proxy for round-trip cost, and the only rows where we can check it")
    print("  disagree with the assumption. The variance result in section 4 is")
    print("  unaffected -- price VARIANCE and round-trip COST are different")
    print("  quantities and only variance was measured by depth. But no route")
    print("  may assume that selecting deep pools buys cheaper fills until the")
    print("  quote census covers more than 6 symbols.")
    print()

    print("CAVEATS THAT TRAVEL WITH EVERY NUMBER ABOVE")
    print("  1. QUOTES, NOT FILLS. Slippage and MEV are NOT modelled. Every")
    print("     'after' figure is a COST FLOOR, the best case, never a realised")
    print("     P/L. The live lane measured ~27.8bp of round-trip slippage")
    print("     (21.73 buy + 6.04 sell, n=12) that no ghost row carries. A")
    print("     second read of trade_fills puts it at ~13.5bp pooled; the two")
    print("     disagree, so the unmodelled band is 13.5-27.8bp, not one end.")
    print("  2. SELECTION. The symbol mix is the current selector's. See the")
    print("     one-sentence statement at the top.")
    print("  3. Only the symbols in the PROVENANCE block were quoted. Unquoted")
    print("     symbols are charged NOTHING and appear only in the gross/depth")
    print("     table, never in a net. So is any row whose clip sits more than")
    print("     %gx from the nearest quoted clip. A default rate on an"
          % MAX_CLIP_EXTRAPOLATION)
    print("     unquoted pool is an assumption in a measurement's clothes.")
    print("  4. Depth is POINT-IN-TIME (2026-09-16T21:31Z). Pools drain.")
    print("  5. FILLS BEAT QUOTES. trade_fills holds 38 real executed legs.")
    print("     Where a quote here disagrees with a fill for the same symbol,")
    print("     THE QUOTE IS THE SUSPECT.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "full": full, "restricted": dsum, "tiers": tier_out,
            "nonmonotone_trips": len(nm), "priced_trips": len(priced),
            "censored": st["censored"],
        }, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
