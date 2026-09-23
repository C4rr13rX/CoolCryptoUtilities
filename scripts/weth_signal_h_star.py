"""Do any of the NAMED strategies call WETH-USDC's direction above h*?

Item [854ce89d].  Pass 670 moved the deep-pool blocker from COST to DIRECTION
and pass 675 closed money_button's own rule on WETH (it fires 0 of 3598 ticks,
and only 16 even at a ZERO fee, so its blocker is the momentum SHAPE, not the
cost gate).  That leaves exactly one question standing for this route:

    does ANY strategy that already carries evidence in the ledger call
    WETH-USDC's forward direction at or above h*, on a DISJOINT sample?

THE COST LEG IS NOT RE-OPENED HERE.  ``COST_ALL_IN_BP`` is IMPORTED from
``weth_direction_census`` rather than retyped, so this script cannot silently
disagree with the measured 6.11bp (2.06bp QUOTED round trip, Echo p498
d864991, plus 4.05bp REALISED gas median over the 18 closed live rows).  A
pass that re-measures cost instead of direction has not done this item.

THE BAR.  For a symmetric move of size ``m`` and round-trip cost ``c``, a
directional call breaks even at

    h* = 0.5 + c / (2 * m)

h* is arithmetic, not an edge (memory ``h-star-is-a-tautology-not-an-edge``):
it can never CHOOSE a horizon, only say what accuracy a horizon would need.
It is recomputed here on each signal's OWN disjoint sample, because ``m`` is a
property of the ticks the signal actually selects -- a signal that fires only
in loud windows faces a LOWER bar than the unconditional tape, and scoring it
against the tape-wide h* would flatter or punish it for free.

HYPOTHESIS AND COMPLEMENT, pre-registered before the run:

    HYPOTHESIS  no named strategy clears h* on WETH-USDC disjoint data, and
                the deep-pool WETH route is CLOSED ON DIRECTION.
    COMPLEMENT  at least one named strategy clears h* at some horizon with a
                one-sided p below 0.05, the route is open, and the next step
                is the capital gate [74a12c84] -- NOT a trade from this file.

THE KILL TEST is the disjoint accuracy itself.  Power is not the excuse
either way: the WETH-USDC tape carries 3599 ticks over 654.6 hours, so a
45-minute disjoint sample is bounded by SPAN at 872 windows, where a one-sided
95% test resolves 2.79pp against a 50% null.  Every edge ever claimed on this
symbol is larger than that.  SIGNALS FIRE SPARSELY, so each signal's own n is
printed with its own minimum detectable effect -- read THAT, not the 872.

DISJOINT means no two scored windows share a forward price print: a fire is
accepted only if its entry tick sits at or after the previous accepted
window's exit tick.  The signal's own TRAILING history may overlap a previous
window and that is stated rather than hidden -- what the binomial needs is
independent OUTCOMES, and non-overlapping forward legs give that.  An
OVERLAPPING number is printed alongside for its point estimate only and is
never used for a verdict.

LONG-ONLY is the verdict metric.  Memory
``horizon-axis-closed-at-both-ends-and-the-lane-is-long-only``: the live lane
cannot short, so "accuracy" for this route means "when the rule says ENTER,
does the tape go UP".  A signed/two-sided accuracy is reported as context and
labelled.

This script places no trade, changes no gate, and produces no tx hash.
OMEN_STRATEGY_ENABLED stays 0 -- every rule tested here is a classical one
that already carries ledger evidence.

Usage:
    python -X utf8 scripts/weth_signal_h_star.py [--symbol WETH-USDC] [--days 30]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from typing import Callable, Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

# Criterion: do NOT re-litigate cost.  Import it.
from weth_direction_census import (  # noqa: E402
    COST_ALL_IN_BP,
    MAX_PRICE_SCALE,
    PAIR_TOLERANCE,
    _binom_p_one_sided,
)
from trading.strategies.base import ema, env_float, rsi  # noqa: E402

#: Longest warm-up any rule below needs, so every signal is scored on the same
#: set of candidate entry ticks and a difference in n is a difference in FIRE
#: RATE, not in how far back the loop was allowed to start.
WARMUP_TICKS = 100


def _db() -> sqlite3.Connection:
    path = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
    if not os.path.exists(path):
        raise SystemExit("store not found: %s" % path)
    return sqlite3.connect("file:%s?mode=ro" % path, uri=True)


def load_tape(conn: sqlite3.Connection, symbol: str, days: float):
    """ts, price, volume with the production mixed-denomination guard applied."""
    since = time.time() - days * 86400.0
    rows = conn.execute(
        "select ts, price, volume from market_stream where symbol = ? and ts > ? "
        "and price > 0 order by ts",
        (symbol, since),
    ).fetchall()
    if not rows:
        return np.array([]), np.array([]), np.array([]), 0
    ts = np.array([float(r[0]) for r in rows])
    px = np.array([float(r[1]) for r in rows])
    vol = np.array([float(r[2] or 0.0) for r in rows])
    median = float(np.median(px))
    if median <= 0:
        return np.array([]), np.array([]), np.array([]), 0
    spread = np.maximum(px / median, median / px)
    keep = spread <= MAX_PRICE_SCALE
    return ts[keep], px[keep], vol[keep], int((~keep).sum())


# ---------------------------------------------------------------------------
# The three NAMED rules.  Each takes the WHOLE tape once and returns a causal
# prediction array: +1 (rule says ENTER LONG) / -1 (exit / short side) / 0 (no
# opinion) at every tick.  Parameters are read through env_float with the SAME
# defaults the production strategy reads, so a .env override moves both.
#
# WHY PRECOMPUTED AND WHY THAT IS EXACT, not an approximation:
# ``base.ema`` is a Python for-loop seeded at out[0]=values[0], so
# ``ema(px[:i+1], span)[-1]`` is IDENTICALLY ``ema(px, span)[i]`` -- the
# recursion never looks forward.  Calling it once per tick instead is O(n^2)
# and did not return in 120 seconds on a 3599-tick tape.  Same argument for
# RSI, which reads only the last ``period+1`` prints.  Nothing here uses a
# price the rule could not have seen at tick i.
# ---------------------------------------------------------------------------

def sig_rsi_reversal(ts: np.ndarray, px: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """trading/strategies/rsi_reversal.py:29-65 -- oversold enters, overbought exits."""
    period = int(env_float("RSI_PERIOD", 14, lo=5, hi=50))
    oversold = env_float("RSI_OVERSOLD", 30.0, lo=5.0, hi=45.0)
    overbought = env_float("RSI_OVERBOUGHT", 70.0, lo=55.0, hi=95.0)
    n = px.size
    out = np.zeros(n, dtype=np.int8)
    d = np.diff(px, prepend=px[0])
    gains = np.clip(d, 0.0, None)
    losses = np.clip(-d, 0.0, None)
    cg = np.concatenate(([0.0], np.cumsum(gains)))
    cl = np.concatenate(([0.0], np.cumsum(losses)))
    for i in range(2 * (period + 1), n):
        avg_gain = (cg[i + 1] - cg[i + 1 - period]) / period
        avg_loss = (cl[i + 1] - cl[i + 1 - period]) / period
        if avg_loss <= 1e-12:
            value = 100.0 if avg_gain > 0 else 50.0
        else:
            value = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        if value <= oversold:
            out[i] = 1
        elif value >= overbought:
            out[i] = -1
    return out


def sig_donchian_breakout(ts: np.ndarray, px: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """trading/strategies/donchian_breakout.py:23-47 -- break of the N-bar high,
    with the same 0.8x volume confirmation production applies."""
    channel = int(env_float("DONCHIAN_CHANNEL", 20, lo=10, hi=96))
    n = px.size
    out = np.zeros(n, dtype=np.int8)
    cv = np.concatenate(([0.0], np.cumsum(vol)))
    for i in range(channel + 1, n):
        prior = px[i - channel: i]
        hi = float(prior.max())
        lo = float(prior.min())
        if hi <= 0 or hi - lo <= 1e-15:
            continue
        if px[i] > hi:
            lo3 = max(0, i - 2)
            recent_vol = (cv[i + 1] - cv[lo3]) / float(i + 1 - lo3)
            base_vol = (cv[i] - cv[i - channel]) / float(channel)
            if base_vol > 0 and recent_vol < base_vol * 0.8:
                continue
            out[i] = 1
        elif px[i] < lo:
            out[i] = -1
    return out


def sig_ema_cross(ts: np.ndarray, px: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """trading/strategies/ema_cross.py:18-40 -- golden cross with a rising slow leg."""
    fast_span = int(env_float("EMA_CROSS_FAST_SPAN", 12, lo=3, hi=100))
    slow_span = int(env_float("EMA_CROSS_SLOW_SPAN", 48, lo=8, hi=400))
    n = px.size
    out = np.zeros(n, dtype=np.int8)
    if n < slow_span + 2:
        return out
    fast = ema(px, fast_span)
    slow = ema(px, slow_span)
    f_now, s_now = fast[1:], slow[1:]
    f_prev, s_prev = fast[:-1], slow[:-1]
    golden = (f_prev <= s_prev) & (f_now > s_now) & (s_now >= s_prev)
    death = (f_prev >= s_prev) & (f_now < s_now)
    out[1:][golden] = 1
    out[1:][death & ~golden] = -1
    out[: slow_span + 2] = 0
    return out


def sig_always_long(ts: np.ndarray, px: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """BASELINE, not a strategy: the cheapest possible direction call.  Any named
    rule that cannot beat this is not adding direction, it is adding a filter."""
    return np.ones(px.size, dtype=np.int8)


SIGNALS: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    "rsi_reversal": sig_rsi_reversal,
    "donchian_breakout": sig_donchian_breakout,
    "ema_cross": sig_ema_cross,
    "BASELINE_always_long": sig_always_long,
}


def _mde_pp(n: int) -> float:
    """One-sided 95% minimum detectable effect against a 50% null, in points."""
    if n <= 0:
        return float("nan")
    return 1.645 * (0.5 / math.sqrt(n)) * 100.0


def score(ts, px, vol, horizon_sec: float, preds: np.ndarray, disjoint: bool) -> Dict:
    """Walk the tape, fire the rule, pair the forward leg, tally."""
    hi_tol = horizon_sec * (1.0 + PAIR_TOLERANCE)
    n = ts.size
    guard = -1.0
    fires = 0
    unpairable = 0
    moves: List[float] = []
    long_hits = 0
    long_n = 0
    signed_hits = 0
    signed_n = 0
    long_marks: List = []
    for i in range(WARMUP_TICKS, n):
        if disjoint and ts[i] < guard:
            continue
        pred = int(preds[i])
        if pred == 0:
            continue
        fires += 1
        j = int(np.searchsorted(ts, ts[i] + horizon_sec, side="left"))
        if j >= n or (ts[j] - ts[i]) > hi_tol:
            # A blackout, not a loss.  Dropped and counted, never stretched to
            # the next print (memory ``feed-takes-global-blackouts``).
            unpairable += 1
            continue
        fwd = px[j] / px[i] - 1.0
        moves.append(abs(fwd))
        if np.sign(fwd) != 0:
            signed_n += 1
            if np.sign(fwd) == pred:
                signed_hits += 1
        if pred == 1:
            long_n += 1
            long_marks.append((float(ts[i]), 1 if fwd > 0 else 0))
            if fwd > 0:
                long_hits += 1
        if disjoint:
            guard = ts[j]
    move_bp = float(np.mean(moves)) * 1e4 if moves else 0.0
    h_star = 0.5 + COST_ALL_IN_BP / (2.0 * move_bp) if move_bp > 0 else float("nan")
    out: Dict = {
        "fires": fires,
        "n_unpairable": unpairable,
        "n_long": long_n,
        "n_signed": signed_n,
        "mean_abs_move_bp": round(move_bp, 2),
        "h_star_required_accuracy": None if move_bp <= 0 else round(h_star, 4),
        "perfect_oracle_pays": bool(move_bp > COST_ALL_IN_BP),
    }
    if long_n:
        acc = long_hits / long_n
        out["long_accuracy"] = round(acc, 4)
        out["long_p_one_sided_vs_coin"] = round(_binom_p_one_sided(long_hits, long_n), 5)
        out["long_mde_pp_at_95"] = round(_mde_pp(long_n), 2)
        out["long_clears_h_star"] = bool(move_bp > 0 and acc >= h_star)
        out["long_gross_bp"] = round((2.0 * acc - 1.0) * move_bp, 2)
        out["long_net_bp"] = round((2.0 * acc - 1.0) * move_bp - COST_ALL_IN_BP, 2)
        # SPLIT-HALF BY TIME.  Not a significance test at these n -- it is a
        # STABILITY check.  An edge that lives entirely in one half of the tape
        # is a window, not an edge, and this repo has shipped that mistake
        # (98% of a gross carried by 7 rows).  Reported for every cell so the
        # reader can see it was not chosen after the fact.
        half = len(long_marks) // 2
        if half:
            first = long_marks[:half]
            second = long_marks[half:]
            out["split_first_half_accuracy"] = round(
                sum(m[1] for m in first) / len(first), 4)
            out["split_second_half_accuracy"] = round(
                sum(m[1] for m in second) / len(second), 4)
            out["split_n"] = [len(first), len(second)]
    if signed_n:
        sacc = signed_hits / signed_n
        out["signed_accuracy"] = round(sacc, 4)
        out["signed_p_one_sided_vs_coin"] = round(_binom_p_one_sided(signed_hits, signed_n), 5)
        out["signed_clears_h_star"] = bool(move_bp > 0 and sacc >= h_star)
    return out


def run(symbol: str, days: float, horizons_min) -> Dict:
    conn = _db()
    ts, px, vol, dropped = load_tape(conn, symbol, days)
    out: Dict = {
        "symbol": symbol,
        "days_requested": days,
        "ticks": int(ts.size),
        "ticks_dropped_mixed_denomination": dropped,
        "warmup_ticks": WARMUP_TICKS,
        "cost_all_in_bp": COST_ALL_IN_BP,
        "cost_provenance": "IMPORTED from weth_direction_census (Echo p498 d864991); NOT re-measured here",
        "verdict_metric": "long_accuracy on the DISJOINT sample vs that sample's own h_star",
        "signals": {},
    }
    if ts.size > 1:
        out["tape_span_hours"] = round((ts[-1] - ts[0]) / 3600.0, 2)
        out["median_gap_sec"] = round(float(np.median(np.diff(ts))), 1)
    if ts.size <= WARMUP_TICKS + 10:
        out["verdict"] = "insufficient tape"
        return out

    for name, fn in SIGNALS.items():
        preds = fn(ts, px, vol)
        blocks: Dict = {}
        for h in horizons_min:
            dj = score(ts, px, vol, h * 60.0, preds, disjoint=True)
            ov = score(ts, px, vol, h * 60.0, preds, disjoint=False)
            dj["OVERLAPPING_long_accuracy_point_estimate_only"] = ov.get("long_accuracy")
            dj["OVERLAPPING_n_long"] = ov.get("n_long")
            blocks[str(h) + "m"] = dj
        out["signals"][name] = blocks
    return out


def multiplicity(res: Dict) -> Dict:
    """How many hypotheses were actually tested, and the Bonferroni floor.

    THIS IS THE GUARD THAT DECIDES THIS ITEM.  Four rules x five horizons is
    TWENTY one-sided tests.  Under a pure null, P(at least one p < 0.0106) over
    20 tests is 19.2% -- so an uncorrected 0.0106 is an ordinary draw, not an
    edge.  This repo's history is a list of numbers that looked good until
    somebody counted how many ways they could have looked good, so the family
    is counted here and the verdict is taken at the CORRECTED level.
    """
    ps: List[float] = []
    for blocks in res.get("signals", {}).values():
        for b in blocks.values():
            p = b.get("long_p_one_sided_vs_coin")
            if p is not None:
                ps.append(float(p))
    m = len(ps)
    best_raw = min(ps) if ps else float("nan")
    return {
        "tests_in_family": m,
        "best_raw_p": best_raw,
        "best_bonferroni_p": None if not ps else min(1.0, best_raw * m),
        "family_wise_alpha_per_test": None if not m else round(0.05 / m, 5),
        "p_any_under_null": None if not ps else round(1.0 - (1.0 - best_raw) ** m, 4),
    }


def verdict_lines(res: Dict) -> List[str]:
    lines: List[str] = []
    mult = res.get("multiplicity", {})
    alpha = mult.get("family_wise_alpha_per_test") or 0.05
    any_clear = False
    for name, blocks in res.get("signals", {}).items():
        best = None
        for h, b in blocks.items():
            acc = b.get("long_accuracy")
            hs = b.get("h_star_required_accuracy")
            if acc is None or hs is None:
                continue
            if b.get("long_clears_h_star") and b.get("long_p_one_sided_vs_coin", 1.0) < alpha:
                any_clear = True
            if best is None or acc - hs > best[1]:
                best = (h, acc - hs, b)
        if best is None:
            lines.append("  %-22s NO PAIRED FIRES at any horizon -- DOES NOT CLEAR" % name)
            continue
        h, gap, b = best
        lines.append(
            "  %-22s best %-4s acc %.4f vs h* %.4f  n=%-4d p=%.4f mde=%.2fpp  ->  %s"
            % (name, h, b["long_accuracy"], b["h_star_required_accuracy"],
               b["n_long"], b.get("long_p_one_sided_vs_coin", float("nan")),
               b.get("long_mde_pp_at_95", float("nan")),
               "CLEARS h*" if b.get("long_clears_h_star") else "DOES NOT CLEAR h*")
        )
    lines.append("")
    lines.append("  MULTIPLICITY: %s tests in the family; best RAW p %.4f -> Bonferroni %.4f;"
                 % (mult.get("tests_in_family"), mult.get("best_raw_p", float("nan")),
                    mult.get("best_bonferroni_p", float("nan"))))
    lines.append("  P(at least one p that small under a pure null) = %s. Verdict alpha is %s/test."
                 % (mult.get("p_any_under_null"), alpha))
    lines.append("")
    if any_clear:
        lines.append("  VERDICT: a NAMED signal clears h* at the FAMILY-WISE level.")
        lines.append("  The deep-pool WETH route is OPEN ON DIRECTION.  Do NOT place a trade from")
        lines.append("  this file -- unblock [74a12c84], the capital gate, which needs the operator.")
    else:
        lines.append("  VERDICT: NO named signal clears h* on the DISJOINT sample at the")
        lines.append("  FAMILY-WISE level.  THE DEEP-POOL WETH ROUTE IS CLOSED ON DIRECTION.")
        lines.append("  Cost is measured and the oracle pays at it; direction is what fails.")
        lines.append("  Do not reopen this route on the cost argument alone.")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="WETH-USDC")
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--horizons", default="10,20,30,45,60")
    ap.add_argument("--out", default="data/weth_signal_h_star.json")
    args = ap.parse_args()
    horizons = [float(x) for x in args.horizons.split(",") if x.strip()]

    res = run(args.symbol, args.days, horizons)
    res["multiplicity"] = multiplicity(res)
    res["generated_at"] = time.time()

    print("=" * 78)
    print("NAMED SIGNALS vs h* ON %s -- DISJOINT SAMPLE" % args.symbol)
    print("=" * 78)
    print("ticks %s over %s h, median gap %ss, cost %sbp all-in (IMPORTED, not re-measured)"
          % (res.get("ticks"), res.get("tape_span_hours"), res.get("median_gap_sec"),
             res.get("cost_all_in_bp")))
    print("")
    for name, blocks in res.get("signals", {}).items():
        print("-- %s" % name)
        for h, b in blocks.items():
            if not b.get("n_long"):
                print("   %-4s fires %-4d paired-long 0 (unpairable %d) -- no sample"
                      % (h, b.get("fires", 0), b.get("n_unpairable", 0)))
                continue
            print("   %-4s n=%-4d acc %.4f  h* %.4f  p %.4f  mde %.2fpp  move %.2fbp  net %+.2fbp  %s"
                  % (h, b["n_long"], b["long_accuracy"], b["h_star_required_accuracy"],
                     b["long_p_one_sided_vs_coin"], b["long_mde_pp_at_95"],
                     b["mean_abs_move_bp"], b["long_net_bp"],
                     "CLEARS" if b["long_clears_h_star"] else "fails"))
            print("        OVERLAPPING acc %s over n=%s -- POINT ESTIMATE ONLY, never a verdict"
                  % (b.get("OVERLAPPING_long_accuracy_point_estimate_only"),
                     b.get("OVERLAPPING_n_long")))
            if b.get("split_n"):
                print("        split-half %s / %s over n=%s -- STABILITY, not significance"
                      % (b.get("split_first_half_accuracy"),
                         b.get("split_second_half_accuracy"), b.get("split_n")))
        print("")
    print("=" * 78)
    for line in verdict_lines(res):
        print(line)
    print("=" * 78)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
