#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Why is the live gate shut? Answer it from the gate itself, with a noise bar.

    python -X utf8 scripts/ghost_gate_forensics.py

Step 6 of the graduation ladder reports ``ghost_validation_block`` and a SINGLE
reason string. That string is the first arm of a 14-way ``elif`` cascade
(trading/pipeline.py, ``_ghost_validation``), so it names the first failing
guardrail and hides every other one. Measured 2026-09-16 (pass 433) the pooled
verdict said ``insufficient_samples`` at 20 of 25 trades -- while the win rate
(0.10 v 0.55), the profit factor (0.063 v 0.95) and the net (-0.4929) were ALL
failing too. Read as an evidence-RATE problem it says "collect five more
trades"; read honestly it says reaching 25 only changes the error string.

So this prints the whole picture rather than the headline:

  1. THE VERDICT, pooled and per strategy, with every guardrail that is
     currently failing -- not just the first.
  2. THE HOLD TABLE. Win rate and net under a hard max-hold, because
     "shorten the hold" is a standing route and it should be re-checked
     against the current book rather than a remembered one. On the pass-433
     book all 15 rows held <= 1800s lost, across 11 symbols and 6 strategies.
  3. THE GROSS/NET SPLIT, WITH A NOISE BAR. ``price_book_at_live_clip``
     charges every row a flat, symbol-blind round trip via ``net_profit_usd``
     (``ret * notional - roundtrip_cost_usd(notional)``). Inverting that
     separates what the market did from what the fee model claimed, and the
     t-statistic says whether either is readable at this n.

THE NOISE BAR IS NOT OPTIONAL and it is why this script exists rather than a
one-liner. On the pass-433 book the gross mean was -2.46bp with SE 7.41 --
t = -0.33, a book statistically indistinguishable from zero -- while the fee
was 38.6bp, or 5.2 SEs of the same distribution. The fee was the only component
above the resolution limit. A lift quoted without that bar would have been
reported as "these strategies lose money", which the data does not support.
"""
from __future__ import annotations

import math
import os
import statistics as st
import sys
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOOKBACK_DEFAULT = 432000

# Mirrors the elif cascade in pipeline.py ``_ghost_validation`` but reports EVERY
# failing arm instead of stopping at the first, which is the whole point of this
# file. Keep the order and the comparisons in step with that cascade.
def failing_guardrails(v: Dict[str, Any]) -> List[str]:
    """Every guardrail currently failing, not just the one ``reason`` names."""
    def g(key: str, default: float = 0.0) -> float:
        try:
            return float(v.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    n = int(g("samples"))
    out: List[str] = []
    if v.get("stale"):
        out.append("stale_ghost_book")
    if n < int(g("min_trades")):
        out.append(f"insufficient_samples {n}<{int(g('min_trades'))}")
    if int(g("wins")) <= 0 and n > 0:
        out.append("no_wins")
    if v.get("tradeable_edge_block"):
        out.append("no_tradeable_edge")
    if g("win_rate") < g("min_win_rate"):
        out.append(f"low_win_rate {g('win_rate'):.3f}<{g('min_win_rate'):.2f}")
    if g("avg_profit") < g("min_margin"):
        out.append(f"negative_margin {g('avg_profit'):+.4f}")
    if g("total_net_profit") <= 0.0 and n > 0:
        out.append(f"non_positive_net_profit {g('total_net_profit'):+.4f}")
    if g("profit_factor") < g("min_profit_factor", 0.95):
        out.append(f"weak_profit_factor {g('profit_factor'):.3f}<{g('min_profit_factor', 0.95):.2f}")
    if v.get("tail_unmeasurable"):
        out.append("tail_unmeasurable")
    if g("tail_risk") > g("tail_guardrail", 1e9):
        out.append(f"tail_risk {g('tail_risk'):.4f}>{g('tail_guardrail'):.4f}")
    return out


def describe(label: str, xs: Sequence[float], bar: float | None = None) -> None:
    """Mean, SD, SE and t for a return series in bp. Never a mean alone."""
    n = len(xs)
    if n < 2:
        print(f"  {label:<26} n={n} -- too few rows to put a bar on")
        return
    m = st.mean(xs)
    sd = st.stdev(xs)
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else float("nan")
    print(f"  {label:<26} n={n:3d}  mean={m:+7.2f}bp  sd={sd:6.2f}  SE={se:5.2f}  t={t:+5.2f}")
    print(f"  {'':<26} 95% CI [{m - 1.96 * se:+7.2f}, {m + 1.96 * se:+7.2f}] bp"
          f"{'   <- spans 0, UNMEASURED' if (m - 1.96 * se) * (m + 1.96 * se) < 0 else ''}")
    w = sum(1 for v in xs if v > 0)
    p = w / n
    pse = math.sqrt(p * (1 - p) / n)
    note = ""
    if bar is not None and p - 1.96 * pse <= bar <= p + 1.96 * pse:
        note = f"   <- CI touches the {bar:.2f} bar"
    print(f"  {'':<26} win rate {p:.3f} +/- {1.96 * pse:.3f} -> "
          f"[{max(0.0, p - 1.96 * pse):.3f}, {min(1.0, p + 1.96 * pse):.3f}]{note}")


def main() -> int:
    from db import get_db
    from trading.pipeline import TrainingPipeline, price_book_at_live_clip
    from trading.metrics import MetricsCollector
    from services.roundtrip_cost import (
        ghost_clip_usd,
        roundtrip_cost_usd,
        roundtrip_cost_rate,
        roundtrip_cost_rate_detail,
    )

    db = get_db()
    pipeline = TrainingPipeline(db=db)
    lookback = int(os.getenv("GHOST_VALIDATION_LOOKBACK_SEC", str(LOOKBACK_DEFAULT)))
    clip = ghost_clip_usd()
    fee = roundtrip_cost_usd(clip)

    print("=" * 74)
    print("GHOST GATE FORENSICS -- why step 6 (RISK) is shut, with a noise bar")
    print("=" * 74)
    print(f"  lookback {lookback}s ({lookback / 3600:.0f}h)   clip ${clip:.2f}   "
          f"flat round-trip fee ${fee:.4f} ({fee / clip * 1e4:.1f}bp)")

    print("\n-- 1. THE VERDICT, and EVERY failing guardrail (not just `reason`) --")
    ledger_ids = ["money_button", "atf_static", "atf_static_scout"]
    for sid in [None] + ledger_ids:
        try:
            v = pipeline._ghost_validation(sid)
        except Exception as exc:  # noqa: BLE001 - a probe never takes the process down
            print(f"  {str(sid or 'POOLED'):<20} EXC {type(exc).__name__}: {exc}")
            continue
        name = str(sid or "POOLED")
        ready = "READY" if v.get("ready") else "BLOCKED"
        print(f"  {name:<20} {ready:<8} reason={str(v.get('reason') or '(strict pass)'):<22}"
              f" n={int(v.get('samples') or 0)}")
        fails = failing_guardrails(v)
        if fails:
            print(f"  {'':<20} ALSO FAILING: {'; '.join(fails)}")
        if v.get("ready") and str(v.get("reason")) == "cold_start_bootstrap":
            print(f"  {'':<20} ^^ COLD-START BYPASS on a ZERO-trade book. "
                  f"pipeline.py calls this 'evidence of nothing'.")

    raw = MetricsCollector(db).ghost_trade_snapshot(limit=500, lookback_sec=lookback)
    trades = price_book_at_live_clip(raw)
    rows: List[Tuple[float, float, str]] = []
    for t in trades:
        entry = float(getattr(t, "entry_ts", 0.0) or 0.0)
        exit_ = float(getattr(t, "exit_ts", 0.0) or 0.0)
        if entry <= 0 or exit_ <= entry:
            continue
        rows.append((exit_ - entry, float(getattr(t, "profit", 0.0)),
                     str(getattr(t, "symbol", "?"))))
    if not rows:
        print("\n  no ghost round trips with a usable hold in this window.")
        return 0

    print("\n-- 2. THE HOLD TABLE -- re-check 'shorten the hold' on the CURRENT book --")
    print(f"  {'max_hold':>10} {'n':>4} {'wins':>5} {'win_rate':>9} {'net':>9} {'PF':>8}")
    for cap in (300, 600, 900, 1800, 3600, 7200, 14400, 86400, 10 ** 9):
        sub = [r for r in rows if r[0] <= cap]
        if not sub:
            continue
        wins = sum(1 for r in sub if r[1] > 0)
        net = sum(r[1] for r in sub)
        gains = sum(r[1] for r in sub if r[1] > 0)
        losses = abs(sum(r[1] for r in sub if r[1] <= 0))
        pf = (gains / losses) if losses > 0 else float("inf")
        label = "ALL" if cap > 10 ** 8 else f"{cap}s"
        print(f"  {label:>10} {len(sub):4d} {wins:5d} {wins / len(sub):9.3f} "
              f"{net:+9.4f} {pf:8.3f}")

    print("\n-- 3. GROSS vs NET, PER SYMBOL -- the flat fee against the measured one --")
    try:
        bar = float(pipeline._ghost_validation(None).get("min_win_rate") or 0.55)
    except Exception:  # noqa: BLE001
        bar = 0.55

    # Rebuilt from the RAW snapshot, not from the priced book, because this
    # section compares two costings of the same rows. Inverting one of them out
    # of ``profit`` and calling the remainder "gross" only works while the fee
    # is a constant; it stops working the moment the fee is per symbol, which
    # is the change this section exists to measure.
    book: List[Dict[str, Any]] = []
    frozen: List[Tuple[str, str]] = []
    for t in raw:
        ret = getattr(t, "return_pct", None)
        if ret is None:
            continue
        try:
            ret = float(ret)
        except (TypeError, ValueError):
            continue
        sym = str(getattr(t, "symbol", "") or "?")
        sid = str(getattr(t, "strategy_id", "") or "(unattributed)")
        # FROZEN: exit price == entry price to the bit. That is a feed that did
        # not move between two reads, not a round trip that returned nothing,
        # and it is the same data-quality class as [bd5619eb]. It is excluded
        # from EVERY book below -- gross, net-blind and net-census -- rather
        # than from the gross book only, because a row that is not an outcome
        # is not an outcome at any costing.
        if ret == 0.0:
            frozen.append((sym, sid))
            continue
        cost_blind = roundtrip_cost_usd(clip)
        rate_sym, source = roundtrip_cost_rate_detail(clip, sym)
        cost_sym = roundtrip_cost_usd(clip, sym)
        book.append({
            "symbol": sym,
            "strategy": sid,
            "gross_bp": ret * 1e4,
            "net_blind_bp": (ret * clip - cost_blind) / clip * 1e4,
            "net_census_bp": (ret * clip - cost_sym) / clip * 1e4,
            "source": source,
            "rate_bp": rate_sym * 1e4,
        })

    if frozen:
        print(f"  EXCLUDED {len(frozen)} FROZEN row(s) -- entry price == exit price to the bit.")
        counts: Dict[Tuple[str, str], int] = {}
        for pair in frozen:
            counts[pair] = counts.get(pair, 0) + 1
        for (sym, sid), count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {count:3d}  {sym:<14} {sid}")
        print("    These are a feed that did not move between two reads. Excluded from EVERY")
        print("    book below, not just the gross one.")
    if not book:
        print("  no rows with a usable return in this window.")
        return 0

    describe("net, FLAT fee (was)", [r["net_blind_bp"] for r in book], bar)
    describe("net, PER-SYMBOL fee (is)", [r["net_census_bp"] for r in book], bar)
    describe("gross (market only)", [r["gross_bp"] for r in book], bar)

    flips = [r for r in book
             if (r["net_blind_bp"] > 0) != (r["net_census_bp"] > 0)]
    print(f"\n  SIGN CHANGES under the corrected model: {len(flips)} of {len(book)} rows.")
    for r in flips[:12]:
        direction = "loss -> win" if r["net_census_bp"] > 0 else "win -> loss"
        print(f"    {r['symbol']:<14} {direction}  {r['net_blind_bp']:+8.2f}bp ->"
              f" {r['net_census_bp']:+8.2f}bp   ({r['source']})")
    if len(flips) > 12:
        print(f"    ... and {len(flips) - 12} more")

    print("\n-- 3b. PER SYMBOL, NEVER POOLED -- a pooled book cannot answer a cost question --")
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for r in book:
        by_symbol.setdefault(r["symbol"], []).append(r)
    for sym, group in sorted(by_symbol.items(), key=lambda kv: -len(kv[1])):
        source = group[0]["source"]
        print(f"\n  {sym}  n={len(group)}  fee {group[0]['rate_bp']:.2f}bp"
              f" ({source})   flat was {fee / clip * 1e4:.2f}bp")
        describe("  net, per-symbol fee", [r["net_census_bp"] for r in group], bar)
        if len(group) >= 2:
            wins = sum(1 for r in group if r["net_census_bp"] > 0)
            p = wins / len(group)
            pse = math.sqrt(p * (1 - p) / len(group))
            low = max(0.0, p - 1.96 * pse)
            verdict = ("CLEARS the bar with a CI that excludes it" if low > bar
                       else f"does NOT clear {bar:.2f} with a CI that excludes it")
            print(f"    win rate {p:.3f}, CI low {low:.3f} -> {verdict}")

    if len(book) >= 2:
        gross = [r["gross_bp"] for r in book]
        sd = st.stdev(gross)
        se = sd / math.sqrt(len(gross))
        print(f"\n  THE RATIO THAT MATTERS: the flat fee is {fee / clip * 1e4 / se:.1f} SEs of this"
              f" distribution.")
        print(f"  A signal below ~{1.96 * se:.1f}bp cannot be read at n={len(gross)}."
              f" Anything the book 'says' about")
        print("  strategy quality inside that band is the fee model talking, not the market.")
        for effect in (5.0, 10.0, 20.0):
            print(f"    to resolve a {effect:.0f}bp edge at 2 sigma: n = {(2 * sd / effect) ** 2:.0f} trades")

    print(f"\n  cost curve, flat vs measured (clip on the left):")
    header = "  ".join(f"${c:g}" for c in (0.75, 2, 6, 20, 100))
    print(f"    {'':<10}{header}")
    print(f"    {'blind':<10}" + "  ".join(f"{roundtrip_cost_rate(c) * 1e4:.1f}bp" for c in (0.75, 2, 6, 20, 100)))
    for sym in sorted({r["symbol"] for r in book if r["source"] == "census"}):
        print(f"    {sym:<10}" + "  ".join(
            f"{roundtrip_cost_rate(c, sym) * 1e4:.1f}bp" for c in (0.75, 2, 6, 20, 100)))
    unpriced = sorted({r["symbol"] for r in book if r["source"] != "census"})
    if unpriced:
        print(f"    UNPRICED and therefore UNCHANGED ({len(unpriced)}): {', '.join(unpriced[:12])}"
              + (" ..." if len(unpriced) > 12 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
