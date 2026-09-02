"""Does money_button's entry TIMING beat entering at random on the same symbols?

The gate census answers "how often does it fire, and what happened after". It
cannot answer the question that decides whether the lane is worth any capital:
is the strategy *selecting* good moments, or is it just along for whatever the
symbol did anyway?

Those come apart badly. Measured 2026-09-02 over 168h of stored ticks,
money_button fired 46 times on exactly two symbols, and the two symbols had
opposite 12-minute drifts:

    BASECAT-USDC   3046 ticks   40.4% of 12-min forward moves up   mean -0.296%
    BSTONK-USDC     630 ticks   52.5% of 12-min forward moves up   mean +0.938%

A lane that fires into a falling market loses without having any opinion, and a
lane that fires into a rising one wins without having any either. Comparing its
fills to the symbol's own forward-return population removes the drift and leaves
only the timing -- which is the only thing the strategy contributes.

The test is a permutation test, not a t-test: forward returns over an
overlapping window are strongly autocorrelated and nowhere near normal, so the
null distribution is built by resampling the symbol's own returns rather than
assumed. `p` is the share of random entry sets of the same size whose mean
return is at least the strategy's.

    python scripts/money_button_edge_vs_baseline.py
    python scripts/money_button_edge_vs_baseline.py --hours 24 --fee 0.0065
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.money_button_gate_census import _State, load_ticks  # noqa: E402


def _forward_return(
    ticks: List[Tuple[float, float, float]], idx: int, hold_sec: float
) -> Optional[float]:
    """Return over `hold_sec` from tick `idx`, or None if the hold runs off the end.

    Counting an unresolved hold as flat would invent a result, so it is dropped
    from both populations equally.
    """
    entry_ts, entry_price = ticks[idx][0], ticks[idx][1]
    if entry_price <= 0:
        return None
    for j in range(idx + 1, len(ticks)):
        if ticks[j][0] >= entry_ts + hold_sec:
            return (ticks[j][1] - entry_price) / entry_price
    return None


def _permutation_p(
    fired: List[float], baseline: List[float], trials: int, rng: random.Random
) -> float:
    """Share of random entry sets whose mean return is at least the strategy's."""
    if not fired or len(baseline) < len(fired):
        return float("nan")
    observed = statistics.mean(fired)
    n = len(fired)
    hits = 0
    for _ in range(trials):
        if statistics.mean(rng.sample(baseline, n)) >= observed:
            hits += 1
    return hits / float(trials)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--fee", type=float, default=None,
                    help="one-way fee rate; default reads TRADE_FEE_RATE or 0.003")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    from trading.strategies.base import StrategyContext
    from trading.strategies.money_button import MoneyButtonStrategy

    fee = args.fee if args.fee is not None else float(os.getenv("TRADE_FEE_RATE", "0.003"))
    hold_sec = float(os.getenv("MONEY_BUTTON_HOLD_MIN", "12")) * 60.0
    slippage = float(os.getenv("MONEY_BUTTON_SLIPPAGE", "0.001"))
    round_trip = 2.0 * fee + slippage
    rng = random.Random(args.seed)

    strat = MoneyButtonStrategy()
    lookback = strat.LOOKBACK_SEC
    by_symbol = load_ticks(args.hours)
    if not by_symbol:
        print("no ticks in the window; nothing to measure.")
        return 0

    fired_by_symbol: Dict[str, List[float]] = defaultdict(list)
    baseline_by_symbol: Dict[str, List[float]] = defaultdict(list)

    for symbol, ticks in by_symbol.items():
        start = 0
        for end in range(len(ticks)):
            cutoff = ticks[end][0] - lookback
            while ticks[start][0] < cutoff:
                start += 1
            ret = _forward_return(ticks, end, hold_sec)
            if ret is None:
                continue
            # Every tick with a resolvable hold is a moment the strategy COULD
            # have entered, so it belongs to the null population.
            baseline_by_symbol[symbol].append(ret)
            ctx = StrategyContext(
                chain="base",
                last_price=ticks[end][1],
                last_volume=ticks[end][2],
                fee_rate=fee,
                available_quote=10.0,
                available_base=0.0,
            )
            if strat.evaluate(_State(ticks[start:end + 1], symbol), ctx) is not None:
                fired_by_symbol[symbol].append(ret)

    print("money_button timing vs random entry -- %.0fh, hold %.0f min, "
          "round trip %.2f%%" % (args.hours, hold_sec / 60.0, 100.0 * round_trip))
    print("permutation test, %d trials, seed %d\n" % (args.trials, args.seed))

    if not fired_by_symbol:
        print("the lane never fired in this window; there is no timing to score.")
        return 0

    header = "%-16s %6s %10s %10s %10s %7s" % (
        "symbol", "fires", "strat%", "baseline%", "edge pp", "p")
    print(header)
    print("-" * len(header))

    all_fired: List[float] = []
    all_baseline: List[float] = []
    for symbol in sorted(fired_by_symbol, key=lambda s: -len(fired_by_symbol[s])):
        fired = fired_by_symbol[symbol]
        baseline = baseline_by_symbol[symbol]
        all_fired.extend(fired)
        all_baseline.extend(baseline)
        s_mean = 100.0 * statistics.mean(fired)
        b_mean = 100.0 * statistics.mean(baseline)
        p = _permutation_p(fired, baseline, args.trials, rng)
        print("%-16s %6d %+9.4f%% %+9.4f%% %+9.4f  %6.3f"
              % (symbol, len(fired), s_mean, b_mean, s_mean - b_mean, p))

    print()
    s_mean = 100.0 * statistics.mean(all_fired)
    b_mean = 100.0 * statistics.mean(all_baseline)
    p = _permutation_p(all_fired, all_baseline, args.trials, rng)
    print("%-16s %6d %+9.4f%% %+9.4f%% %+9.4f  %6.3f"
          % ("POOLED", len(all_fired), s_mean, b_mean, s_mean - b_mean, p))

    print()
    print("strat%    mean 12-min return of the moments money_button chose")
    print("baseline% mean 12-min return of EVERY moment it could have chosen")
    print("edge pp   what the timing added, in percentage points")
    print("p         share of random entry sets that did at least as well;")
    print("          p near 0.5 means the selection is indistinguishable from chance")
    print()
    print("A lane only pays if `strat%%` clears the %.2f%% round trip on its own."
          % (100.0 * round_trip))
    if s_mean < 100.0 * round_trip:
        print("It does not: %+.4f%% gross against %.2f%% of cost."
              % (s_mean, 100.0 * round_trip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
