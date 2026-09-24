"""Reprice the ghost book at each symbol's OWN census cost, per strategy.

WHY THIS EXISTS
---------------
Item [71cafaca]. The fee BOOKED into ``trade_outcomes.fee_cost`` is
token-blind: nothing in it varies by symbol, so WETH (census pool rate
0.0208%) and VIRTUAL pay the same at the same clip. Graduation (step 5) reads
TRADEABLE ghost NET P/L, which is gross minus that blind fee. So the only
question that matters is not "is the fee wrong" -- it is:

    if every ghost row were charged its own symbol's real round-trip cost at
    its own notional, would ANY strategy cross the bar?

If the answer is no, the cost-repricing route is CLOSED and the loop should
spend its passes on signal or hold time instead. That negative is the most
valuable thing this script can produce and it is printed in those words.

WHAT IT DOES NOT DO
-------------------
It changes no cost model and writes nothing. ``services/roundtrip_cost.py`` is
imported and never reimplemented; the tradeability predicate is imported from
``trading.strategies.ledger`` via ``scripts/tradeable_book.py`` so it cannot
drift from the bar it reports on. Row loading is ``tradeable_book.load_rows``
for the same reason.

THREE GUARDS AGAINST A GOOD NUMBER
----------------------------------
1. OFF-TAPE CENSOR. Four of the closed rows are priced outside their own
   symbol's all-time ``market_stream`` range and carry more than the whole
   book's net. ``--no-censor`` reproduces the uncensored figures on purpose,
   so the effect of the censor is visible rather than assumed.
2. MEDIAN AND n BESIDE EVERY SUM, with the top row's share of the sum named.
   Every positive book found in this repo so far has been a one-row book.
3. BOTH DIRECTIONS. The blind fee OVERCHARGES deep pairs and UNDERCHARGES
   thin ones, so correct pricing makes some strategies WORSE. Symbols that get
   dearer are printed beside the ones that get cheaper, with row counts, and
   census-covered rows are counted separately from uncovered ones (the census
   prices 9 tokens; everything else takes an unpriced default and is therefore
   NOT evidence about that symbol's real cost).

Run:  python -X utf8 scripts/ghost_reprice_census.py
      python -X utf8 scripts/ghost_reprice_census.py --gas-leg
      python -X utf8 scripts/ghost_reprice_census.py --days 3650 --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import services.roundtrip_cost as rtc                      # noqa: E402
from scripts.tradeable_book import (                        # noqa: E402
    DEFAULT_DB,
    _tradeable_predicate,
    load_rows,
)

# The graduation bar, as ``trading/strategies/ledger.py`` enforces it.
BAR_TRADES = 20
BAR_WIN = 0.55
BAR_PNL = 0.0


# --------------------------------------------------------------------------
# the tape, for the off-tape censor
# --------------------------------------------------------------------------
def tape_ranges(db_path: Path) -> Dict[str, Tuple[float, float]]:
    """All-time min/max traded price per symbol from ``market_stream``.

    A row whose entry OR exit price never appeared on its own symbol's tape
    cannot be a real round trip at those prices, whatever the book says.
    Symbols with no tape at all are NOT judged -- absence of a tape is not
    evidence of a bad price, and treating it as such would censor every
    symbol the feed has dropped.
    """
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT symbol, MIN(price), MAX(price), COUNT(*) FROM market_stream "
            "WHERE price > 0 GROUP BY symbol"
        ).fetchall()
    finally:
        con.close()
    out: Dict[str, Tuple[float, float]] = {}
    for sym, lo, hi, n in rows:
        if not sym or not n or lo is None or hi is None:
            continue
        out[str(sym).upper()] = (float(lo), float(hi))
    return out


def is_off_tape(row: Dict[str, Any], ranges: Dict[str, Tuple[float, float]]) -> bool:
    band = ranges.get(str(row.get("symbol") or "").upper())
    if not band:
        return False
    lo, hi = band
    for price in (row.get("entry_price"), row.get("exit_price")):
        try:
            p = float(price or 0.0)
        except Exception:  # noqa: BLE001
            continue
        if p <= 0:
            continue
        if p < lo or p > hi:
            return True
    return False


# --------------------------------------------------------------------------
# repricing
# --------------------------------------------------------------------------
def reprice(row: Dict[str, Any]) -> Dict[str, Any]:
    """Attach this row's census cost at its own notional, and the delta.

    ``notional`` is entry_price * quantity, the row's own two numbers. The
    implied notional (gross / return) is computed alongside ONLY as a check
    that the two agree; it is never used, because gross/return divides by a
    quantity that has already been divided out and amplifies any rounding in
    the recorded prices.
    """
    notional = float(row.get("notional") or 0.0)
    symbol = str(row.get("symbol") or "")
    booked = float(row.get("fees") or 0.0)
    gross = float(row.get("gross") or 0.0)

    out = dict(row)
    out["booked_fee"] = booked
    out["booked_bp"] = 1e4 * booked / notional if notional > 0 else None

    if notional <= 0:
        out["census_fee"] = None
        out["census_bp"] = None
        out["source"] = "no_notional"
        out["net_repriced"] = None
        out["delta"] = None
        return out

    detail = rtc.roundtrip_cost_breakdown(notional, symbol)
    rate = float(detail.get("rate") or 0.0)
    out["census_fee"] = rate * notional
    out["census_bp"] = 1e4 * rate
    out["source"] = str(detail.get("source") or "")
    out["net_repriced"] = gross - out["census_fee"]
    # POSITIVE delta = repricing HELPS this row (it was overcharged).
    out["delta"] = out["net_repriced"] - float(row.get("net") or 0.0)
    return out


def _acc() -> Dict[str, Any]:
    return {"n": 0, "booked_net": [], "repriced_net": []}


def _summarise(acc: Dict[str, Any]) -> Dict[str, Any]:
    def stats(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"n": 0, "sum": 0.0, "median": 0.0, "win": 0.0, "top_share": 0.0}
        total = sum(vals)
        top = max(vals, key=abs)
        return {
            "n": len(vals),
            "sum": total,
            "median": statistics.median(vals),
            "win": sum(1 for v in vals if v > 0) / len(vals),
            # The top row's share of the sum. Named on purpose: a sum whose
            # biggest row is 100%+ of it is one row wearing a book's clothes.
            "top_share": (top / total) if total else float("inf"),
            "top_row": top,
        }

    return {
        "n": acc["n"],
        "booked": stats(acc["booked_net"]),
        "repriced": stats(acc["repriced_net"]),
    }


def clears_bar(s: Dict[str, Any]) -> bool:
    return (s["n"] >= BAR_TRADES and s["win"] >= BAR_WIN and s["sum"] > BAR_PNL)


# --------------------------------------------------------------------------
# the gas-leg question (item criterion 2)
# --------------------------------------------------------------------------
def gas_leg_census(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Is the booked fee ONE modelled gas leg or TWO, and does it vary by size?

    The booked fee is token-blind, so it is compared against the SYMBOLLESS
    model -- ``roundtrip_cost_usd(notional)`` -- which is the only comparison
    that can be right if the writer never saw a symbol. A "second gas leg" is
    ``+ DEFAULT_FIXED_USD`` on top of that.

    Buckets are the ones the item's own criterion names, so the answer lands
    on the 6-15 dollar bucket it says is unresolved.
    """
    buckets = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 15.0), (15.0, 1e9)]
    out: List[Dict[str, Any]] = []
    for lo, hi in buckets:
        sel = [r for r in rows
               if r.get("notional") and lo <= float(r["notional"]) < hi
               and r.get("booked_bp") is not None]
        if not sel:
            out.append({"lo": lo, "hi": hi, "n": 0})
            continue
        one, two, booked = [], [], []
        closer_one = closer_two = 0
        for r in sel:
            n = float(r["notional"])
            m1 = rtc.roundtrip_cost_usd(n)          # one gas leg
            m2 = m1 + rtc.DEFAULT_FIXED_USD          # two gas legs
            b = float(r["booked_fee"])
            one.append(1e4 * m1 / n)
            two.append(1e4 * m2 / n)
            booked.append(1e4 * b / n)
            if abs(b - m1) <= abs(b - m2):
                closer_one += 1
            else:
                closer_two += 1
        out.append({
            "lo": lo, "hi": hi, "n": len(sel),
            "booked_bp": statistics.median(booked),
            "one_leg_bp": statistics.median(one),
            "two_leg_bp": statistics.median(two),
            "closer_one": closer_one,
            "closer_two": closer_two,
        })
    return {"buckets": out}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def collect(*, days: float, db_path: Path, censor: bool) -> Dict[str, Any]:
    now = time.time()
    rows = load_rows(db_path, now - days * 86400.0)
    rows = [r for r in rows if str(r.get("mode", "")).lower() != "live"]

    ranges = tape_ranges(db_path)
    off_tape = [r for r in rows if is_off_tape(r, ranges)]
    if censor:
        rows = [r for r in rows if not is_off_tape(r, ranges)]

    priced = [reprice(r) for r in rows]
    usable = [r for r in priced if r["net_repriced"] is not None]

    is_tradeable = _tradeable_predicate()
    if is_tradeable is None:
        return {"error": "cannot import trading.strategies.ledger._live_tradeable"}

    cache: Dict[Tuple[str, str], bool] = {}

    def tradeable(r: Dict[str, Any]) -> bool:
        key = (str(r.get("symbol") or ""), str(r.get("strategy_id") or ""))
        if key not in cache:
            cache[key] = bool(is_tradeable(key[0], key[1]))
        return cache[key]

    per_strategy: Dict[str, Dict[str, Any]] = {}
    pooled = _acc()
    per_symbol: Dict[str, Dict[str, Any]] = {}
    covered = uncovered = 0

    for r in usable:
        if str(r["source"]) == "census":
            covered += 1
        else:
            uncovered += 1

        sym = str(r.get("symbol") or "")
        ps = per_symbol.setdefault(sym, {"n": 0, "booked_bp": [], "census_bp": [],
                                         "delta": 0.0, "source": r["source"]})
        ps["n"] += 1
        ps["booked_bp"].append(r["booked_bp"])
        ps["census_bp"].append(r["census_bp"])
        ps["delta"] += r["delta"]

        if not tradeable(r):
            continue
        sid = str(r.get("strategy_id") or "unclassified")
        acc = per_strategy.setdefault(sid, _acc())
        for target in (acc, pooled):
            target["n"] += 1
            target["booked_net"].append(float(r["net"] or 0.0))
            target["repriced_net"].append(float(r["net_repriced"]))

    strategies = {sid: _summarise(a) for sid, a in per_strategy.items()}
    sign_flips = [
        sid for sid, s in strategies.items()
        if (s["booked"]["sum"] > 0) != (s["repriced"]["sum"] > 0)
    ]
    crossers_before = [sid for sid, s in strategies.items() if clears_bar(s["booked"])]
    crossers_after = [sid for sid, s in strategies.items() if clears_bar(s["repriced"])]

    return {
        "days": days,
        "censored": censor,
        "off_tape_rows": len(off_tape),
        "off_tape_net": sum(float(r.get("net") or 0.0) for r in off_tape),
        "rows_considered": len(rows),
        "rows_priced": len(usable),
        "census_covered_rows": covered,
        "census_uncovered_rows": uncovered,
        "pooled_tradeable": _summarise(pooled),
        "strategies": strategies,
        "symbols": {
            s: {
                "n": v["n"],
                "source": v["source"],
                "booked_bp": statistics.median([x for x in v["booked_bp"] if x is not None] or [0.0]),
                "census_bp": statistics.median([x for x in v["census_bp"] if x is not None] or [0.0]),
                "delta": v["delta"],
            }
            for s, v in per_symbol.items()
        },
        "sign_flips": sorted(sign_flips),
        "crossers_before": sorted(crossers_before),
        "crossers_after": sorted(crossers_after),
        "gas_leg": gas_leg_census(usable),
    }


def render(res: Dict[str, Any]) -> None:
    if res.get("error"):
        print("ERROR:", res["error"])
        return
    print("=" * 74)
    print("GHOST BOOK REPRICED AT EACH SYMBOL'S OWN CENSUS COST  [71cafaca]")
    print("=" * 74)
    print(f"  window                 {res['days']:.0f} days")
    print(f"  off-tape rows          {res['off_tape_rows']} "
          f"(net {res['off_tape_net']:+.4f})  "
          f"{'STRIPPED' if res['censored'] else 'KEPT (--no-censor)'}")
    print(f"  rows considered        {res['rows_considered']}")
    print(f"  rows priced            {res['rows_priced']}")
    print(f"  census-COVERED rows    {res['census_covered_rows']}")
    print(f"  census-UNCOVERED rows  {res['census_uncovered_rows']}"
          "   <- these take an unpriced DEFAULT, not their real cost")
    print()

    p = res["pooled_tradeable"]
    print("POOLED TRADEABLE BOOK (what graduation reads)")
    for label in ("booked", "repriced"):
        s = p[label]
        print(f"  {label:9s} n={s['n']:4d}  sum {s['sum']:+.4f}  "
              f"median {s['median']:+.6f}  win {100*s['win']:5.1f}%  "
              f"top row {s.get('top_row', 0.0):+.4f} = {100*s['top_share']:.1f}% of sum")
    print()

    print("PER STRATEGY, TRADEABLE ONLY   (bar: n>=20, win>=55%, sum>0)")
    print(f"  {'strategy':28s} {'n':>4s} {'booked sum':>11s} {'reprice sum':>11s} "
          f"{'bk med':>9s} {'rp med':>9s} {'bk win':>7s} {'rp win':>7s}  bar")
    for sid, s in sorted(res["strategies"].items(),
                         key=lambda kv: -kv[1]["repriced"]["sum"]):
        b, r = s["booked"], s["repriced"]
        flag = "CROSSES" if clears_bar(r) else ("was" if clears_bar(b) else "")
        print(f"  {sid[:28]:28s} {s['n']:4d} {b['sum']:+11.4f} {r['sum']:+11.4f} "
              f"{b['median']:+9.5f} {r['median']:+9.5f} "
              f"{100*b['win']:6.1f}% {100*r['win']:6.1f}%  {flag}")
    print()

    print("BOTH DIRECTIONS -- per symbol, median booked bp vs census bp")
    print(f"  {'symbol':16s} {'n':>4s} {'booked bp':>10s} {'census bp':>10s} "
          f"{'delta $':>9s}  source")
    rows = sorted(res["symbols"].items(), key=lambda kv: -kv[1]["delta"])
    for sym, v in rows:
        arrow = "CHEAPER" if v["census_bp"] < v["booked_bp"] else "DEARER"
        print(f"  {sym[:16]:16s} {v['n']:4d} {v['booked_bp']:10.2f} "
              f"{v['census_bp']:10.2f} {v['delta']:+9.4f}  {v['source']:18s} {arrow}")
    print()

    print("GAS LEG: is the booked fee one modelled leg or two?  (criterion 2)")
    print(f"  {'notional':>14s} {'n':>4s} {'booked bp':>10s} {'1 leg bp':>10s} "
          f"{'2 leg bp':>10s} {'closer1':>8s} {'closer2':>8s}")
    for b in res["gas_leg"]["buckets"]:
        if not b["n"]:
            continue
        hi = "inf" if b["hi"] > 1e8 else f"{b['hi']:.0f}"
        print(f"  ${b['lo']:.0f}-{hi:>8s} {b['n']:4d} {b['booked_bp']:10.2f} "
              f"{b['one_leg_bp']:10.2f} {b['two_leg_bp']:10.2f} "
              f"{b['closer_one']:8d} {b['closer_two']:8d}")
    print()

    print("VERDICT")
    print(f"  strategies that CHANGE SIGN on repricing: {len(res['sign_flips'])}"
          f"  {res['sign_flips'] or ''}")
    print(f"  clear the bar BEFORE repricing: {len(res['crossers_before'])}"
          f"  {res['crossers_before'] or ''}")
    print(f"  clear the bar AFTER  repricing: {len(res['crossers_after'])}"
          f"  {res['crossers_after'] or ''}")
    if not res["crossers_after"]:
        print()
        print("  ZERO STRATEGIES CROSS THE BAR EVEN AT CORRECT PER-SYMBOL COST.")
        print("  The cost-repricing route to graduation is CLOSED on this book.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=3650.0)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--no-censor", action="store_true",
                    help="keep the off-tape rows, to show what the censor does")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    res = collect(days=args.days, db_path=args.db, censor=not args.no_censor)
    if args.json:
        print(json.dumps(res, indent=2, default=float))
    else:
        render(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
