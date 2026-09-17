"""Replay the MOMENTUM money_button gate over stored ticks. Read-only.

WHY THIS EXISTS. Commit 70fc09c split a shared ``strategy_id``: two rules were
both called ``money_button`` and traded in OPPOSITE directions. The scheduler's
BUY-THE-DIP rule now carries ``money_button_dip`` and has been replayed
(``scripts/money_button_dip_census.py``): 333 resolved, 17.7% win, mean gross
-0.3816% BEFORE cost, mean net -1.1512%.

That leaves ``trading/strategies/money_button.py`` -- the MOMENTUM plugin --
as the sole holder of the bare id, and therefore the sole holder of
``live_approved=true``, granted by the operator at pass 424 on a lifetime of
TWO ghost trades at net -0.0161. The live arm points at a rule nobody has
measured. This measures it.

HOW IT DIFFERS FROM THE DIP CENSUS, deliberately. The dip census reimplements
its gate in the script. That was safe there -- seven thresholds, one code path.
It is NOT safe here: the momentum gate also carries a feed-density guard, a
mixed-denomination guard, a median-gap-scaled window unit, a volatility
discount, a hold-time projection and a volume-fade guard. A hand-copy of that
measures the copy, not the rule. So this script CALLS THE SHIPPED GATE --
``MoneyButtonStrategy().evaluate(state, ctx)`` -- and reports what the real
code decided.

UPPER BOUNDS, stated because they are real:
  * ``market_stream`` stores (ts, price) and no volume, so the volume-fade
    guard cannot fire here. It CAN fire in production. Fires below are an
    upper bound on the true fire count.
  * Nothing downstream of the strategy is replayed -- no arbitration, no
    position cap, no live gate. This is the GATE's verdict, not the lane's.

No threshold is read from anywhere but the shipped defaults, and nothing here
writes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.money_button_dip_census import (  # noqa: E402
    base_of,
    load_ticks,
    round_trip_for,
)
from trading.strategies.base import StrategyContext  # noqa: E402
from trading.strategies.money_button import MoneyButtonStrategy  # noqa: E402


#: How many trailing samples each evaluation sees. The plugin needs
#: ``min_samples`` (10 by default) and scales its own windows off the median
#: gap, so a fixed 40 gives it room without handing it a stale shelf.
WINDOW = 40


def _state(symbol: str, window: List[Tuple[float, float]]):
    """A RouteState-alike carrying REAL timestamps, not synthetic spacing.

    Sample spacing is the thing the plugin scales its windows off
    (``unit = max(300s, 2 * median_gap)``), so faking it uniform would change
    the gate's behaviour. These are the feed's own timestamps.
    """
    state = types.SimpleNamespace()
    state.symbol = symbol
    state.base_token = base_of(symbol)
    state.quote_token = symbol.split("-", 1)[1] if "-" in symbol else "USDC"
    # Volume 0.0: market_stream has none. The plugin skips the volume guard
    # when it cannot see >= 6 volumes, which is the upper bound noted above.
    state.samples = [(float(t), float(p), 0.0) for t, p in window]
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--fee", type=float, default=None,
                    help="one-way fee handed to the GATE; default TRADE_FEE_RATE or 0.003")
    ap.add_argument("--min-evals", type=int, default=50,
                    help="skip symbols thinner than this in the per-token table")
    ap.add_argument("--top", type=int, default=18,
                    help="rows of the per-token table to print, cheapest first")
    ap.add_argument("--horizon-min", type=float, default=15.0,
                    help="forward hold ceiling; MONEY_BUTTON_HOLD_MINUTES ceiling is 15")
    args = ap.parse_args()

    gate_fee = args.fee if args.fee is not None else float(
        os.getenv("TRADE_FEE_RATE", "0.003")
    )
    horizon_sec = float(args.horizon_min) * 60.0

    ticks = load_ticks(args.hours)
    if not ticks:
        print("no ticks in the window -- nothing to replay")
        return 1

    # ONE instance, evaluated repeatedly: that is how production holds it, and
    # a fresh instance per tick would discard any internal debounce.
    strategy = MoneyButtonStrategy()

    # DENSITY declines mean the FEED could not present a window; DIRECTION
    # declines mean the feed was fine and the signal was absent. Telling them
    # apart per token is what decides whether "trade only cheap deep names"
    # is a fixable ingest problem or a dead route.
    DENSITY = {"window_too_short", "feed_frozen", "too_few_samples",
               "feed_too_sparse_for_hold", "window_exceeds_lookback"}
    DIRECTION = {"momentum_not_aligned", "move_exhausted", "slope_not_positive"}

    census: Dict[str, int] = defaultdict(int)
    per_symbol: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    fires: List[dict] = []
    evaluated = 0
    started = time.time()

    for symbol, series in ticks.items():
        n = len(series)
        for i in range(WINDOW, n):
            evaluated += 1
            window = series[i - WINDOW:i + 1]
            last_price = float(window[-1][1])
            ctx = StrategyContext(
                chain="base",
                last_price=last_price,
                last_volume=0.0,
                fee_rate=gate_fee,
                available_quote=1000.0,
                available_base=0.0,
            )
            try:
                result = strategy.evaluate(_state(symbol, window), ctx)
            except Exception as exc:  # a crash is a finding, not a skip
                census[f"RAISED:{type(exc).__name__}"] += 1
                continue
            if result is None:
                reason = str(getattr(strategy, "last_decline", "") or "declined")
                census[reason] += 1
                per_symbol[symbol][reason] += 1
                per_symbol[symbol]["_evals"] += 1
                continue
            census["(candidate produced)"] += 1
            per_symbol[symbol]["FIRED"] += 1
            per_symbol[symbol]["_evals"] += 1
            directive = result["directive"]
            target = float(getattr(directive, "target_price", 0.0) or 0.0)
            fires.append({
                "symbol": symbol, "i": i, "ts": float(window[-1][0]),
                "entry": last_price,
                "target": target if target > 0 else last_price * 1.01,
            })

    # ---- forward outcome: hold until target or the horizon ---------------
    resolved: List[dict] = []
    for f in fires:
        series = ticks[f["symbol"]]
        entry, target = f["entry"], f["target"]
        exit_price, hit, held = entry, False, 0.0
        for ts, price in series[f["i"] + 1:]:
            held = ts - f["ts"]
            exit_price = float(price)
            if price >= target:
                hit = True
                break
            if held >= horizon_sec:
                break
        if held <= 0.0:
            continue  # no forward tape: unresolved, not a win and not a loss
        gross = (exit_price - entry) / entry
        cost = round_trip_for(f["symbol"])
        resolved.append({
            "symbol": f["symbol"], "gross": gross, "net": gross - cost,
            "hit": hit, "held": held, "cost": cost,
        })

    # ---- report ----------------------------------------------------------
    print(f"money_button MOMENTUM census -- the gate at trading/strategies/money_button.py")
    print(f"{args.hours:.0f}h of stored ticks, gate fee {gate_fee:.4f}, "
          f"forward horizon {args.horizon_min:.0f}m, window {WINDOW} samples")
    print(f"{len(ticks)} symbols, {evaluated} evaluations "
          f"({time.time() - started:.0f}s)")
    print("NOTE: market_stream carries no volume, so the volume-fade guard")
    print("      cannot fire here -- these are UPPER BOUNDS on fire count.")
    print()
    print(f"{'decline reason':<32}{'count':>8}{'share':>9}")
    print("-" * 49)
    for reason, count in sorted(census.items(), key=lambda kv: -kv[1]):
        share = 100.0 * count / max(evaluated, 1)
        print(f"{reason:<32}{count:>8}{share:>8.2f}%")
    print()
    print(f"fired: {len(fires)} of {evaluated} evaluations "
          f"({100.0 * len(fires) / max(evaluated, 1):.3f}%)")

    if not resolved:
        print("\nNO RESOLVED ROUND TRIPS. The gate produced no candidate with "
              "forward tape; there is no P/L to report and none is invented.")
        return 0

    nets = np.array([r["net"] for r in resolved], dtype=float)
    grosses = np.array([r["gross"] for r in resolved], dtype=float)
    wins = int((nets > 0).sum())
    print()
    print("--- outcome, held to target or horizon, net of the REAL per-token "
          "round trip ---")
    print(f"{len(resolved)} resolved, {wins}W/{len(resolved) - wins}L "
          f"({100.0 * wins / len(resolved):.1f}% win), "
          f"mean net {100.0 * nets.mean():+.4f}%, "
          f"total net {100.0 * nets.sum():+.4f}% of notional")
    print(f"    target hit: {sum(1 for r in resolved if r['hit'])} of {len(resolved)}"
          f"; mean GROSS {100.0 * grosses.mean():+.4f}%")
    print(f"    median hold {np.median([r['held'] for r in resolved]) / 60.0:.1f} min")

    # SIGNIFICANCE, printed next to the means on purpose. This gate fires
    # rarely, so a mean here rests on a handful of rows and reads as a verdict
    # when it is noise. A t near zero on GROSS means UNMEASURED, not flat --
    # and that is a different finding from a gross edge that is truly negative.
    def _t(sample: np.ndarray) -> float:
        if sample.size < 2:
            return float("nan")
        sd = float(sample.std(ddof=1))
        if sd <= 0.0:
            return float("nan")
        return float(sample.mean() / (sd / np.sqrt(sample.size)))

    t_gross, t_net = _t(grosses), _t(nets)
    print(f"    GROSS t={t_gross:+.3f}   NET t={t_net:+.3f}   (n={len(resolved)})")
    if abs(t_gross) < 2.0:
        print("    -> GROSS is INDISTINGUISHABLE FROM ZERO at this n. Do not")
        print("       report this rule as having, or lacking, a gross edge.")
    if t_net < -2.0 and abs(t_gross) < 2.0:
        print("    -> but NET is significantly negative, and the whole gap")
        print("       between them is the round trip: this is a COST result.")

    # Per token: the dip census showed gross spanning +0.114% to -9.722%
    # INSIDE one cost cohort, so pooling would hide the thing that matters.
    by_symbol: Dict[str, List[dict]] = defaultdict(list)
    for r in resolved:
        by_symbol[r["symbol"]].append(r)
    print()
    print("--- the same signal, split by TOKEN (cost is a token choice) ---")
    print(f"{'symbol':<24}{'n':>5}{'win%':>7}{'mean gross':>12}{'mean net':>11}{'cost':>10}")
    print("-" * 69)
    rows = sorted(by_symbol.items(),
                  key=lambda kv: -float(np.mean([r["net"] for r in kv[1]])))
    for symbol, group in rows:
        g = np.array([r["gross"] for r in group], dtype=float)
        nt = np.array([r["net"] for r in group], dtype=float)
        win = 100.0 * float((nt > 0).sum()) / len(group)
        print(f"{symbol:<24}{len(group):>5}{win:>6.0f}%"
              f"{100.0 * g.mean():>11.3f}%{100.0 * nt.mean():>10.3f}%"
              f"{100.0 * group[0]['cost']:>9.2f}%")
    # ---- density vs direction, bucketed by cost -------------------------
    print()
    print("--- WHY it declines, by token: FEED (density) or SIGNAL (direction) ---")
    print(f"{'symbol':<18}{'cost':>7}{'evals':>7}{'density':>9}{'direction':>11}{'fired':>7}")
    print("-" * 59)
    buckets: Dict[str, List[float]] = defaultdict(list)
    rows_pt = []
    for symbol, tally in per_symbol.items():
        ev = tally.get("_evals", 0)
        if ev < args.min_evals:
            continue
        dens = sum(v for k, v in tally.items() if k in DENSITY)
        dirn = sum(v for k, v in tally.items() if k in DIRECTION)
        rows_pt.append((round_trip_for(symbol), symbol, ev,
                        100.0 * dens / ev, 100.0 * dirn / ev, tally.get("FIRED", 0)))
    for cost, symbol, ev, dens, dirn, fired in sorted(rows_pt)[:args.top]:
        print(f"{symbol:<18}{100.0 * cost:>6.2f}%{ev:>7}{dens:>8.1f}%{dirn:>10.1f}%{fired:>7}")
    print()
    for label, keep in (("CHEAP  (<=0.20% round trip)", lambda c: c <= 0.0020),
                        ("DEARER ( >0.20% round trip)", lambda c: c > 0.0020)):
        grp = [r for r in rows_pt if keep(r[0])]
        if not grp:
            continue
        ev = sum(r[2] for r in grp)
        dens = sum(r[2] * r[3] for r in grp) / ev
        dirn = sum(r[2] * r[4] for r in grp) / ev
        print(f"{label}: {len(grp):>3} symbols, {ev:>6} evals, "
              f"density {dens:>5.1f}%, direction {dirn:>5.1f}%, "
              f"fired {sum(r[5] for r in grp)}")
    print()
    print("A cheap cohort with LOW density and HIGH direction is the dead-route")
    print("answer: the feed served those names best and the signal was absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
