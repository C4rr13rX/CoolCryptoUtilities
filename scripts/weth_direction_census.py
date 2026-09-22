"""Does WETH-USDC's DIRECTION clear the bar its now-measured COST sets?

Item [017594f6], criterion 6 -- "The WETH GROSS EDGE has never been measured".
Echo's pass-498 note closed the COST leg at 6.11bp all-in for a $10 clip
(2.06bp quoted round trip + 4.05bp gas, median ``fee_cost`` over the 18 closed
live rows).  That note also says what comes next and that it is a SEPARATE
measurement: the n=99 behind the +0.1769% expectancy contains no WETH fill, so
the expectancy may not be rescaled -- it has to be re-measured on WETH.

This is that measurement, and it is deliberately the WEAKEST form of the claim
the route needs.  It does not simulate money_button.  It asks the prior
question, which is cheaper to answer and which money_button cannot escape:

    over the WETH-USDC tape, at the horizons the lane actually holds, is
    forward direction predictable AT ALL above the rate a 6.11bp cost demands?

The bar is arithmetic, not a fit.  For a symmetric move of size ``m`` and a
round-trip cost ``c``, a directional call is break-even at

    h* = 0.5 + c / (2 * m)

(see memory ``h-star-is-a-tautology-not-an-edge``: h* can never CHOOSE a
horizon, it can only say what accuracy a horizon would need).  So the test is
one-sided and pre-registered:

    HYPOTHESIS  the deep-pool route is now blocked by DIRECTION, not cost.
    COMPLEMENT  direction already clears h* and the route is live-ready.

Either answer is a finished measurement.  A 50% accuracy kills the complement;
an accuracy at or above h* kills the hypothesis and the route should be taken
live at a $10 clip.

QUOTED-vs-REALISED, stated once so no reader has to infer it: the moves here
are REALISED tape, but the 6.11bp cost they are judged against is QUOTED (Echo
p498) plus a REALISED gas median.  No tx hash is produced by this script and it
places no trade and changes no gate.

Pairing rule.  A forward return is only formed when a tick exists inside
[t+H, t+H*(1+TOL)].  The WETH feed takes 16-59 minute global blackouts
(memory ``feed-takes-global-blackouts``), and pairing across one would measure
the blackout rather than the horizon -- so unpairable ticks are DROPPED and
counted, never stretched to the next available print.

Usage:
    python scripts/weth_direction_census.py [--symbol WETH-USDC] [--days 30]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Criterion 7: import the rule's own parameters rather than re-typing literals,
# or the census prices a different rule than production runs.
from trading.strategies.base import env_float  # noqa: E402

#: Echo pass 498, commit d864991.  QUOTED round trip (2.06bp median of
#: 2.34/1.57/2.06) plus REALISED gas (median fee_cost $0.004047 over the 18
#: closed live rows = 4.05bp at a $10 clip).
COST_ALL_IN_BP = 6.11

#: The two thresholds this item names in its own acceptance criteria.
DECISION_THRESHOLDS_BP = (8.1, 12.2)

#: Mixed-denomination guard, same constant and same REFUSE-don't-filter shape
#: as the production lane (money_button.py:170).  WETH-USDT prints both 2450
#: and 0.9996; one such tick decides every statistic computed off the window.
MAX_PRICE_SCALE = env_float("MONEY_BUTTON_MAX_PRICE_SCALE", 50.0, lo=2.0, hi=1e6)

#: How far past the horizon a print may sit and still count as that horizon.
PAIR_TOLERANCE = 0.25


def _db() -> sqlite3.Connection:
    path = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
    if not os.path.exists(path):
        raise SystemExit("store not found: %s" % path)
    return sqlite3.connect("file:%s?mode=ro" % path, uri=True)


def load_tape(conn: sqlite3.Connection, symbol: str, days: float):
    since = time.time() - days * 86400.0
    rows = conn.execute(
        "select ts, price from market_stream where symbol = ? and ts > ? "
        "and price > 0 order by ts",
        (symbol, since),
    ).fetchall()
    if not rows:
        return np.array([]), np.array([]), 0
    ts = np.array([float(r[0]) for r in rows])
    px = np.array([float(r[1]) for r in rows])
    median = float(np.median(px))
    if median <= 0:
        return np.array([]), np.array([]), 0
    spread = np.maximum(px / median, median / px)
    keep = spread <= MAX_PRICE_SCALE
    dropped = int((~keep).sum())
    return ts[keep], px[keep], dropped


def forward_returns(ts: np.ndarray, px: np.ndarray, horizon_sec: float,
                    disjoint: bool = False):
    """Returns (fwd, trailing, paired, unpairable) for every pairable tick.

    ``disjoint=True`` walks forward so that no two returns share a price
    print.  This matters more than it looks: memory
    ``adjacent-returns-share-a-price-print`` is exactly this trap, and at a
    79s median gap a 60-minute forward return overlaps ~45 of its neighbours,
    so the OVERLAPPING sample's p-value is anti-conservative by roughly that
    factor.  Every significance claim here is made on the disjoint sample; the
    overlapping one is reported only for its point estimate.
    """
    hi = horizon_sec * (1.0 + PAIR_TOLERANCE)
    fwd: List[float] = []
    trail: List[float] = []
    unpairable = 0
    n = ts.size
    guard = -1.0
    for i in range(n):
        if disjoint and ts[i] < guard:
            continue
        j = int(np.searchsorted(ts, ts[i] + horizon_sec, side="left"))
        if j >= n or (ts[j] - ts[i]) > hi:
            unpairable += 1
            continue
        # trailing leg: the same horizon backwards, same pairing rule
        k = int(np.searchsorted(ts, ts[i] - horizon_sec, side="left"))
        if k >= i or (ts[i] - ts[k]) > hi:
            continue
        fwd.append(px[j] / px[i] - 1.0)
        trail.append(px[i] / px[k] - 1.0)
        if disjoint:
            # next window may not start before this one's forward leg lands,
            # AND its own trailing leg must not reach back into this window
            guard = ts[j] + horizon_sec
    return np.array(fwd), np.array(trail), len(fwd), unpairable


def _binom_p_one_sided(wins: int, n: int, p0: float = 0.5) -> float:
    """P(X >= wins) under Binomial(n, p0).  No scipy dependency."""
    if n == 0:
        return float("nan")
    total = 0.0
    log_p0, log_q0 = math.log(p0), math.log(1.0 - p0)
    for k in range(wins, n + 1):
        total += math.exp(
            math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * log_p0 + (n - k) * log_q0
        )
    return min(1.0, total)


def census(symbol: str, days: float, horizons_min) -> Dict:
    conn = _db()
    ts, px, dropped = load_tape(conn, symbol, days)
    out: Dict = {
        "symbol": symbol,
        "days_requested": days,
        "ticks": int(ts.size),
        "ticks_dropped_mixed_denomination": dropped,
        "cost_all_in_bp": COST_ALL_IN_BP,
        "cost_provenance": "Echo p498 d864991: 2.06bp QUOTED + 4.05bp gas REALISED at $10",
        "horizons": {},
    }
    if ts.size:
        out["tape_span_days"] = round((ts[-1] - ts[0]) / 86400.0, 2)
        out["median_gap_sec"] = round(float(np.median(np.diff(ts))), 1) if ts.size > 1 else None
    if ts.size < 3:
        return out

    for h_min in horizons_min:
        fwd, trail, paired, unpairable = forward_returns(ts, px, h_min * 60.0)
        block: Dict = {"n_paired": paired, "n_unpairable": unpairable}
        if paired < 30:
            block["verdict"] = "insufficient sample (n<30)"
            out["horizons"][str(h_min) + "m"] = block
            continue

        move_bp = float(np.mean(np.abs(fwd))) * 1e4
        h_star = 0.5 + COST_ALL_IN_BP / (2.0 * move_bp) if move_bp > 0 else float("nan")

        # (a) unconditional long: the cheapest possible "direction call"
        up = int((fwd > 0).sum())
        flat = int((fwd == 0).sum())
        acc_long = up / paired

        # (b) momentum continuation, the shape money_button actually trades:
        #     trailing move up -> predict up, trailing move down -> predict down
        pred = np.sign(trail)
        live = pred != 0
        n_mom = int(live.sum())
        correct = int(((np.sign(fwd) == pred) & live & (np.sign(fwd) != 0)).sum())
        acc_mom = correct / n_mom if n_mom else float("nan")

        block.update({
            "mean_abs_move_bp": round(move_bp, 2),
            "h_star_required_accuracy": round(h_star, 4),
            "flat_returns": flat,
            "unconditional_long_accuracy": round(acc_long, 4),
            "unconditional_long_clears_h_star": bool(acc_long >= h_star),
            "momentum_n": n_mom,
            "momentum_accuracy": None if n_mom == 0 else round(acc_mom, 4),
            "momentum_clears_h_star": bool(n_mom and acc_mom >= h_star),
            "momentum_p_one_sided_vs_coin": (
                None if n_mom == 0 else round(_binom_p_one_sided(correct, n_mom), 5)
            ),
            "perfect_oracle_pays": bool(move_bp > COST_ALL_IN_BP),
        })

        # Expected gross of the momentum call, and its net against the cost.
        # (2*acc - 1) * mean|move| is the gross a symmetric directional call
        # earns; this is a RE-MEASUREMENT on WETH, not a rescale of the n=99.
        if n_mom:
            gross_bp = (2.0 * acc_mom - 1.0) * move_bp
            block["momentum_gross_bp"] = round(gross_bp, 2)
            block["momentum_net_bp"] = round(gross_bp - COST_ALL_IN_BP, 2)
            # the same call taken the other way round, since an accuracy below
            # 0.5 is a contrarian edge of (1 - acc)
            contra_gross = (2.0 * (1.0 - acc_mom) - 1.0) * move_bp
            block["contrarian_gross_bp"] = round(contra_gross, 2)
            block["contrarian_net_bp"] = round(contra_gross - COST_ALL_IN_BP, 2)
            block["either_direction_pays"] = bool(
                max(gross_bp, contra_gross) > COST_ALL_IN_BP
            )

        # The honest sample: no two returns share a price print.
        dfwd, dtrail, dn, _ = forward_returns(ts, px, h_min * 60.0, disjoint=True)
        block["disjoint_n"] = dn
        if dn >= 30:
            dpred = np.sign(dtrail)
            dlive = dpred != 0
            dn_mom = int(dlive.sum())
            dcorrect = int(((np.sign(dfwd) == dpred) & dlive
                            & (np.sign(dfwd) != 0)).sum())
            block["disjoint_momentum_n"] = dn_mom
            block["disjoint_momentum_accuracy"] = (
                round(dcorrect / dn_mom, 4) if dn_mom else None
            )
            block["disjoint_momentum_p_one_sided_vs_coin"] = (
                round(_binom_p_one_sided(dcorrect, dn_mom), 5) if dn_mom else None
            )
            block["disjoint_mean_abs_move_bp"] = round(
                float(np.mean(np.abs(dfwd))) * 1e4, 2
            )
        out["horizons"][str(h_min) + "m"] = block
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="WETH-USDC")
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--horizons", default="10,15,30,45,60")
    ap.add_argument("--out", default="data/weth_direction_census.json")
    args = ap.parse_args()

    horizons = [float(x) for x in args.horizons.split(",") if x.strip()]
    result = census(args.symbol, args.days, horizons)

    print("=" * 74)
    print("WETH DIRECTION CENSUS -- item 017594f6 criterion 6")
    print("=" * 74)
    print("symbol %s | ticks %d | span %sd | median gap %ss | dropped(mixed-denom) %d"
          % (result["symbol"], result["ticks"], result.get("tape_span_days"),
             result.get("median_gap_sec"), result["ticks_dropped_mixed_denomination"]))
    print("cost judged against: %.2fbp all-in (%s)"
          % (COST_ALL_IN_BP, result["cost_provenance"]))
    print("-" * 74)
    hdr = "%-6s %7s %7s %10s %9s %9s %9s" % (
        "horiz", "n", "unpair", "move_bp", "h*", "long_acc", "mom_acc")
    print(hdr)
    for key, blk in result["horizons"].items():
        if "mean_abs_move_bp" not in blk:
            print("%-6s %7d %7d   %s" % (key, blk["n_paired"], blk["n_unpairable"],
                                         blk.get("verdict", "")))
            continue
        print("%-6s %7d %7d %10.2f %9.4f %9.4f %9s" % (
            key, blk["n_paired"], blk["n_unpairable"], blk["mean_abs_move_bp"],
            blk["h_star_required_accuracy"], blk["unconditional_long_accuracy"],
            "n/a" if blk["momentum_accuracy"] is None else "%.4f" % blk["momentum_accuracy"]))
    print("-" * 74)
    for key, blk in result["horizons"].items():
        if "mean_abs_move_bp" not in blk:
            continue
        if not blk["perfect_oracle_pays"]:
            print("%-6s A PERFECT ORACLE LOSES: move %.2fbp < cost %.2fbp"
                  % (key, blk["mean_abs_move_bp"], COST_ALL_IN_BP))
            continue
        verdict = "CLEARS" if blk["momentum_clears_h_star"] else "FAILS"
        print("%-6s momentum %s h*: %.4f vs %.4f required | gross %+.2fbp "
              "net %+.2fbp | contrarian net %+.2fbp | DISJOINT acc %s n=%s p=%s"
              % (key, verdict, blk["momentum_accuracy"] or 0.0,
                 blk["h_star_required_accuracy"], blk.get("momentum_gross_bp", 0.0),
                 blk.get("momentum_net_bp", 0.0), blk.get("contrarian_net_bp", 0.0),
                 blk.get("disjoint_momentum_accuracy"),
                 blk.get("disjoint_momentum_n"),
                 blk.get("disjoint_momentum_p_one_sided_vs_coin")))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
