"""Does money_button's SHIPPED entry rule find a direction on WETH-USDC?

Item [c967fed1].  Pass 670 (`scripts/weth_direction_census.py`) closed the COST
leg of the WETH deep-pool route and left the DIRECTION leg open: a perfect
oracle pays at every horizon (mean |move| 13.9-31.9bp against a 6.11bp all-in
cost), but the *unconditional* and *trailing-sign* direction calls all fail
h* = 0.5 + c/(2m).  That census deliberately did NOT simulate money_button --
its own docstring says so -- because the prior question was cheaper.

This is the money_button half, and it differs from the census in one way that
matters: it does not re-type money_button's thresholds as literals.  It
imports `MoneyButtonStrategy` and calls its real `evaluate()`, so every gate,
every env default and every discount in `trading/strategies/money_button.py`
runs exactly as shipped.  A rule simulated from hand-copied constants is a
rule nobody has tested; this repo has already shipped three divergent copies
of one cost table (memory `three-copies-of-the-token-cost-table`).

WHAT IS DELIBERATELY GENEROUS, so the fire count is an UPPER BOUND
-----------------------------------------------------------------
  * `market_stream` carries a `volume` column but it is frequently 0, and
    money_button's `volume_faded` gate is a no-op when `baseline_vol <= 0`.
    Every tick this simulation fires on has therefore passed a volume test
    that may never have run.  Reported as `volume_gate_live` so the reader
    knows how many fires were actually screened.
  * The window is the raw tape; nothing upstream is dropping ticks the live
    loop would have missed.  A live `RouteState` sees at most what is here.

If the rule cannot fire even under those conditions, tightening them cannot
rescue it.

THE BAR
-------
Identical to the census, and it is arithmetic rather than a fit:

    h* = 0.5 + c / (2 * mean|forward move|)

computed on the SELECTED sub-tape, not on the whole tape -- a rule that
selects loud ticks faces a *higher* bar than the tape average, and comparing
selected accuracy against the tape-wide h* is how a selection rule flatters
itself.

Significance is claimed only on DISJOINT returns, reusing the census's own
`forward_returns(..., disjoint=True)` walk.  At a 79s median gap a 60-minute
forward return overlaps ~45 of its neighbours, so an overlapping-sample
p-value is anti-conservative by roughly that factor.

Usage:
    python -X utf8 scripts/weth_moneybutton_direction.py --days 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# REUSED, never reimplemented: the tape loader, the pairing walk (including the
# disjoint guard) and the one-sided binomial all come from the pass-670 census
# so the two scripts cannot disagree about what a return is.
from scripts.weth_direction_census import (  # noqa: E402
    COST_ALL_IN_BP,
    _binom_p_one_sided,
    _db,
    forward_returns,
    load_tape,
)
from trading.strategies.base import StrategyContext  # noqa: E402
from trading.strategies.money_button import MoneyButtonStrategy  # noqa: E402


class _Window:
    """The minimum `RouteState` surface `sample_arrays` reads."""

    __slots__ = ("samples", "window_rows_repaired")

    def __init__(self, samples: List[tuple]) -> None:
        self.samples = samples
        self.window_rows_repaired = 0


class _ProbeButton(MoneyButtonStrategy):
    """money_button with its candidate CONSTRUCTOR stubbed, gates untouched.

    `make_candidate` needs a live `RouteState` and a sizing path; none of that
    changes whether the rule FIRES.  Overriding only the final constructor
    keeps every threshold, discount and refusal in the shipped file on the
    code path being measured.
    """

    def make_candidate(self, state, ctx, **kwargs):  # type: ignore[override]
        return dict(kwargs)


def _last_decline(strategy: MoneyButtonStrategy) -> str:
    return strategy.last_decline or "unknown"


def simulate(ts: np.ndarray, px: np.ndarray, vol: np.ndarray,
             fee_rate: float) -> Dict[str, Any]:
    """Run the shipped rule at every tick; return the indices it fires on."""
    strategy = _ProbeButton()
    lookback = float(MoneyButtonStrategy.LOOKBACK_SEC)
    fires: List[int] = []
    declines: Dict[str, int] = {}
    volume_gate_live = 0
    n = ts.size
    start = 0
    for i in range(n):
        while ts[start] < ts[i] - lookback:
            start += 1
        if i - start + 1 < MoneyButtonStrategy.min_samples:
            declines["too_few_samples"] = declines.get("too_few_samples", 0) + 1
            continue
        window = _Window([
            (float(ts[k]), float(px[k]), float(vol[k])) for k in range(start, i + 1)
        ])
        ctx = StrategyContext(
            chain="base",
            last_price=float(px[i]),
            last_volume=float(vol[i]),
            fee_rate=float(fee_rate),
            available_quote=10.0,
            available_base=0.0,
        )
        out = strategy.evaluate(window, ctx)
        if out is None:
            reason = _last_decline(strategy)
            declines[reason] = declines.get(reason, 0) + 1
            continue
        fires.append(i)
        tail = vol[start:i + 1]
        if tail.size >= 6 and float(np.median(tail[:-3])) > 0:
            volume_gate_live += 1
    return {
        "fires": fires,
        "declines": dict(sorted(declines.items(), key=lambda kv: -kv[1])),
        "volume_gate_live": volume_gate_live,
        "evaluations": n,
    }


def _accuracy_block(fwd: np.ndarray, label: str) -> Dict[str, Any]:
    """Long-only accuracy against the h* of the SAME sub-sample."""
    n = int(fwd.size)
    block: Dict[str, Any] = {"label": label, "n": n}
    if n == 0:
        block["verdict"] = "no returns"
        return block
    move_bp = float(np.mean(np.abs(fwd))) * 1e4
    h_star = 0.5 + COST_ALL_IN_BP / (2.0 * move_bp) if move_bp > 0 else float("nan")
    up = int((fwd > 0).sum())
    acc = up / n
    gross_bp = (2.0 * acc - 1.0) * move_bp
    block.update({
        "mean_abs_move_bp": round(move_bp, 2),
        "h_star_required": round(h_star, 4),
        "accuracy_long": round(acc, 4),
        "clears_h_star": bool(acc >= h_star),
        "gross_bp": round(gross_bp, 2),
        "net_bp": round(gross_bp - COST_ALL_IN_BP, 2),
        "p_one_sided_vs_coin": round(_binom_p_one_sided(up, n), 5),
    })
    return block


def _forward_from(ts: np.ndarray, px: np.ndarray, idx: List[int],
                  horizon_sec: float, tolerance: float = 0.5):
    """Forward return at `horizon_sec` for each selected tick index."""
    hi = horizon_sec * (1.0 + tolerance)
    out: List[float] = []
    kept: List[int] = []
    n = ts.size
    for i in idx:
        j = int(np.searchsorted(ts, ts[i] + horizon_sec, side="left"))
        if j >= n or (ts[j] - ts[i]) > hi:
            continue
        out.append(px[j] / px[i] - 1.0)
        kept.append(i)
    return np.array(out), kept


def _disjoint_subset(ts: np.ndarray, kept: List[int], horizon_sec: float) -> List[int]:
    """Greedy left-to-right pick so no two selected windows overlap."""
    chosen: List[int] = []
    guard = -1.0
    for i in kept:
        if ts[i] < guard:
            continue
        chosen.append(i)
        guard = ts[i] + horizon_sec
    return chosen


def decile_test(ts: np.ndarray, px: np.ndarray,
                horizon_min: float) -> Dict[str, Any]:
    """Criteria 3 and 4: re-test the pass-670 trailing-move deciles, DISJOINT.

    Pass 670 bucketed the OVERLAPPING sample.  At 10m it found selection does
    nothing -- all ten deciles fail, 0.4088-0.5000, no trend in trailing size
    (criterion 3 asks only that this be CONFIRMED).  At 60m it found decile 10
    at accuracy 0.5559 against its own h* of 0.5343, net +3.84bp -- on an
    effective n near 7, with a zigzag across deciles that is what noise looks
    like across 20 tests (criterion 4 asks for a DISJOINT re-test).

    The same bucketing is redone here on the disjoint walk, where each return
    is an independent observation.  Ten buckets is ten tests, so a decile is
    only reported as surviving if it clears its OWN h* *and* p < 0.05 --
    one nominal pass in ten is the null, not a lead.
    """
    fwd, trail, _, _ = forward_returns(ts, px, horizon_min * 60.0, disjoint=True)
    out: Dict[str, Any] = {"horizon_min": horizon_min, "disjoint_n": int(fwd.size)}
    if fwd.size < 20:
        out["verdict"] = "disjoint sample too small to decile (n<20)"
        return out
    mag = np.abs(trail)
    order = np.argsort(mag)
    buckets = np.array_split(order, 10)
    rows = []
    for d, b in enumerate(buckets, start=1):
        sub = fwd[b]
        blk = _accuracy_block(sub, "decile%d" % d)
        blk["trailing_move_bp"] = round(float(np.mean(mag[b])) * 1e4, 2)
        rows.append(blk)
    out["deciles"] = rows
    top = rows[-1]
    out["decile10"] = top
    out["decile10_survives_disjoint"] = bool(
        top.get("clears_h_star") and top.get("p_one_sided_vs_coin", 1.0) < 0.05
    )
    out["any_decile_survives_disjoint"] = [
        r["label"] for r in rows
        if r.get("clears_h_star") and r.get("p_one_sided_vs_coin", 1.0) < 0.05
    ]
    out["deciles_nominally_clearing_h_star"] = [
        r["label"] for r in rows if r.get("clears_h_star")
    ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="WETH-USDC")
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--clip-usd", type=float, default=10.0,
                    help="notional the production fee rate is priced at")
    args = ap.parse_args()

    conn = _db()
    ts, px, dropped = load_tape(conn, args.symbol, args.days)
    if ts.size < 3:
        print("no tape for %s" % args.symbol)
        return 1
    rows = conn.execute(
        "select ts, volume from market_stream where symbol = ? and ts > ? "
        "and price > 0 order by ts",
        (args.symbol, float(ts[0]) - 1.0),
    ).fetchall()
    vmap = {round(float(r[0]), 3): float(r[1] or 0.0) for r in rows}
    vol = np.array([vmap.get(round(float(t), 3), 0.0) for t in ts])

    # The fee the LIVE lane would pass in, from the one place that prices it.
    from services.roundtrip_cost import roundtrip_cost_rate
    fee_rate = float(roundtrip_cost_rate(args.clip_usd))

    print("=" * 74)
    print("money_button SHIPPED RULE on %s -- item c967fed1" % args.symbol)
    print("=" * 74)
    print("ticks %d | span %.2fd | median gap %.1fs | dropped(mixed-denom) %d"
          % (ts.size, (ts[-1] - ts[0]) / 86400.0,
             float(np.median(np.diff(ts))), dropped))
    print("rule imported from trading/strategies/money_button.py -- no literal "
          "re-typed; only make_candidate is stubbed")
    print("ctx.fee_rate %.6f (services.roundtrip_cost at $%.2f) | "
          "slippage/margin/hold are the shipped env defaults"
          % (fee_rate, args.clip_usd))
    print("cost bar %.2fbp all-in (same as pass-670 census)" % COST_ALL_IN_BP)
    print("-" * 74)

    result: Dict[str, Any] = {
        "symbol": args.symbol,
        "ticks": int(ts.size),
        "span_days": round((ts[-1] - ts[0]) / 86400.0, 2),
        "ctx_fee_rate": fee_rate,
        "cost_all_in_bp": COST_ALL_IN_BP,
    }

    sim = simulate(ts, px, vol, fee_rate)
    fires = sim["fires"]
    result["fires"] = len(fires)
    result["evaluations"] = sim["evaluations"]
    result["volume_gate_live_on_fires"] = sim["volume_gate_live"]
    result["declines"] = sim["declines"]

    print("FIRES: %d of %d ticks (%.3f%%)"
          % (len(fires), sim["evaluations"],
             100.0 * len(fires) / max(sim["evaluations"], 1)))
    print("volume gate actually screened %d of those fires "
          "(0 means the liquidity test never ran)" % sim["volume_gate_live"])
    print("top declines:")
    for reason, count in list(sim["declines"].items())[:8]:
        print("  %-28s %6d  %5.1f%%"
              % (reason, count, 100.0 * count / max(sim["evaluations"], 1)))

    # THE COMPLEMENT TEST, and it is the one that decides what a 0 means.
    #
    # If the rule declines on `edge_below_cost`, "money_button never fires on
    # WETH" is a statement about the COST GATE and a cheaper token or a bigger
    # clip could change it.  If it declines on `momentum_not_aligned` /
    # `slope_not_positive`, the shape the rule looks for is not IN this tape
    # and no cost parameter can put it there.  Re-running at fee_rate = 0 --
    # cheaper than any venue on earth -- separates the two: whatever still
    # declines with the cost floor at its arithmetic minimum is structural.
    free = simulate(ts, px, vol, 0.0)
    result["zero_fee_fires"] = len(free["fires"])
    result["zero_fee_declines"] = free["declines"]
    print("ZERO-FEE UPPER BOUND (ctx.fee_rate=0, the cheapest venue that could "
          "exist): %d fires" % len(free["fires"]))
    for reason, count in list(free["declines"].items())[:4]:
        print("  %-28s %6d  %5.1f%%"
              % (reason, count, 100.0 * count / max(free["evaluations"], 1)))

    # Criterion 6: is the rule measurable on this tape at all?
    result["measurable"] = bool(len(fires) >= 30)
    print("-" * 74)
    if len(fires) < 30:
        print("UNMEASURABLE FROM THIS TAPE: %d fires < 30. Criterion 6 is the "
              "finding -- no conditional accuracy can be quoted." % len(fires))
    else:
        print("MEASURABLE: %d fires >= 30." % len(fires))

    # Criterion 2: conditional accuracy at 10m and 60m, overlapping and disjoint.
    conditional: Dict[str, Any] = {}
    for h_min in (10.0, 60.0):
        hs = h_min * 60.0
        fwd, kept = _forward_from(ts, px, fires, hs)
        blk = _accuracy_block(fwd, "%.0fm overlapping" % h_min)
        dis_idx = _disjoint_subset(ts, kept, hs)
        dfwd, _ = _forward_from(ts, px, dis_idx, hs)
        dblk = _accuracy_block(dfwd, "%.0fm DISJOINT" % h_min)
        conditional["%.0fm" % h_min] = {"overlapping": blk, "disjoint": dblk}
        for b in (blk, dblk):
            if b["n"] == 0:
                print("%-18s n=0 -- no pairable forward return" % b["label"])
                continue
            print("%-18s n=%-5d acc %.4f vs h* %.4f -> %s | net %+.2fbp | p=%.5f"
                  % (b["label"], b["n"], b["accuracy_long"], b["h_star_required"],
                     "CLEARS" if b["clears_h_star"] else "FAILS",
                     b["net_bp"], b["p_one_sided_vs_coin"]))
    result["conditional"] = conditional

    # Criteria 3 and 4: the pass-670 trailing-move deciles, on disjoint returns.
    deciles: Dict[str, Any] = {}
    for h_min in (10.0, 60.0):
        print("-" * 74)
        dec = decile_test(ts, px, h_min)
        deciles["%.0fm" % h_min] = dec
        if "deciles" in dec:
            print("%.0fm TRAILING-MOVE DECILES on the DISJOINT walk (n=%d):"
                  % (h_min, dec["disjoint_n"]))
            for r in dec["deciles"]:
                print("  %-9s n=%-4d trail %8.2fbp fwd %7.2fbp acc %.4f "
                      "h* %.4f %s net %+7.2fbp p=%.4f"
                      % (r["label"], r["n"], r["trailing_move_bp"],
                         r.get("mean_abs_move_bp", 0.0),
                         r.get("accuracy_long", 0.0),
                         r.get("h_star_required", 0.0),
                         "CLEARS" if r.get("clears_h_star") else "FAILS ",
                         r.get("net_bp", 0.0), r.get("p_one_sided_vs_coin", 1.0)))
            print("  nominally clearing h*: %s | surviving p<0.05: %s"
                  % (dec["deciles_nominally_clearing_h_star"] or "none",
                     dec["any_decile_survives_disjoint"] or "none"))
            print("  decile 10 survives on disjoint returns: %s"
                  % dec["decile10_survives_disjoint"])
        else:
            print("%.0fm deciles: %s" % (h_min, dec.get("verdict")))
    result["deciles"] = deciles
    dec = deciles["60m"]

    # Criterion 5: the plain statement.
    print("-" * 74)
    cleared: List[str] = []
    for h, blocks in conditional.items():
        d = blocks["disjoint"]
        if d["n"] and d.get("clears_h_star") and d.get("p_one_sided_vs_coin", 1.0) < 0.05:
            cleared.append("money_button@%s" % h)
    for h, d in deciles.items():
        for lab in d.get("any_decile_survives_disjoint", []):
            cleared.append("%s trailing-move %s" % (h, lab))
    result["selected_rules_clearing_h_star_on_disjoint"] = cleared
    if cleared:
        print("SELECTED-TICK RULES CLEARING h* ON A DISJOINT SAMPLE: %s"
              % ", ".join(cleared))
    else:
        print("NO selected-tick accuracy clears its own h* on a DISJOINT "
              "sample. The deep-pool WETH route is closed on DIRECTION; the "
              "cost leg is proven CLEAR and must not be reopened.")

    out_path = Path("data/weth_moneybutton_direction.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
