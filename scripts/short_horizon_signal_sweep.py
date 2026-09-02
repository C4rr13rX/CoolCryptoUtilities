"""Is there ANY short-horizon signal on this feed, or is the lane the problem?

money_button is one hypothesis -- momentum continuation, long only -- and it
has been measured and found to carry no information (see
scripts/money_button_edge_test.py: lift +0.06% +/- 0.61%). That result says
nothing about whether a *different* short-horizon signal would work, and the
plan for this lane is several parallel short-horizon strategies. So before
building any of them, test the candidates against the same control.

This sweep exists to be able to come back NEGATIVE. Testing many signals and
reporting the best one is how noise gets promoted to a strategy: with 8
signals and a 5% test, one false positive is the expected outcome, not a
surprise. So the output reports:

  * every signal tried, not the best one;
  * the lift in standard errors, so a large-but-noisy result cannot masquerade
    as a discovery;
  * a significance bar RAISED for the number of signals tested (Bonferroni),
    stated up front rather than chosen afterwards;
  * and net-of-cost, because a real edge smaller than the round trip is still
    a losing strategy.

A signal must clear all of it -- beat the control by more than the corrected
bar AND cover the round trip -- before it is worth writing as a Strategy.

Usage:
    python scripts/short_horizon_signal_sweep.py
    python scripts/short_horizon_signal_sweep.py --hold-min 30 --fee 0.0065
    python scripts/short_horizon_signal_sweep.py --universe established
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.money_button_edge_test import _drop_mixed_denominations  # noqa: E402
from scripts.money_button_gate_census import load_ticks  # noqa: E402

#: Tokens with an established market, as opposed to the discovery feed's
#: microcaps. The split is not cosmetic: measured 2026-09-02 the two halves
#: have opposite forward distributions (established trimmed mean +0.011% at
#: 12m, discovered -0.152%, with the discovered down-tail 25.6% against an
#: up-tail of 18.1%), which is consistent with the discovery feed selecting
#: tokens after they have already pumped.
ESTABLISHED = {
    "AERO-USDC", "CBBTC-USDC", "COMP-USDC", "AAVE-USDC", "VIRTUAL-USDC",
    "SOL-USDC", "CBETH-USDC", "CBXRP-USDC", "ARB-USDC", "SAND-USDC",
    "VELO-USDC", "WETH-USDC", "LINK-USDC", "UNI-USDC",
}

Window = Sequence[Tuple[float, float, float]]


def _ret(window: Window, seconds: float) -> float:
    """Trailing return over `seconds`, 0.0 when the window cannot express it."""
    if len(window) < 2:
        return 0.0
    cutoff = window[-1][0] - seconds
    anchor = window[0][1]
    for ts, price, _v in window:
        if ts >= cutoff:
            anchor = price
            break
    if anchor <= 0:
        return 0.0
    return window[-1][1] / anchor - 1.0


def _zscore(window: Window) -> float:
    prices = [p for _t, p, _v in window]
    n = len(prices)
    if n < 5:
        return 0.0
    mean = sum(prices) / n
    var = sum((p - mean) ** 2 for p in prices) / (n - 1)
    sd = math.sqrt(var)
    if sd <= 0:
        return 0.0
    return (prices[-1] - mean) / sd


# Each signal takes the trailing window and answers "would you buy here".
# Deliberately simple: the point is to find whether ANY direction of edge
# exists on this feed, not to tune one.
SIGNALS: Dict[str, Callable[[Window], bool]] = {
    "momentum_5m_up": lambda w: _ret(w, 300) > 0.002,
    "momentum_aligned": lambda w: _ret(w, 300) > 0 and _ret(w, 600) > 0 and _ret(w, 1800) > 0,
    "dip_5m": lambda w: _ret(w, 300) < -0.005,
    "dip_30m": lambda w: _ret(w, 1800) < -0.02,
    "deep_dip_30m": lambda w: _ret(w, 1800) < -0.05,
    "zscore_low": lambda w: _zscore(w) < -1.5,
    "zscore_high": lambda w: _zscore(w) > 1.5,
    "breakout_high": lambda w: len(w) >= 8 and w[-1][1] >= max(p for _t, p, _v in w),
    "reversal_up": lambda w: _ret(w, 1800) < -0.02 and _ret(w, 300) > 0.002,
}


def _stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "se": 0.0, "up": 0.0}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    return {
        "n": n,
        "mean": mean,
        "se": math.sqrt(var / n),
        "up": sum(1 for v in values if v > 0) / n,
    }


def _forward(ticks, end: int, hold_sec: float) -> Optional[float]:
    entry_ts, entry_price = ticks[end][0], ticks[end][1]
    if entry_price <= 0:
        return None
    for j in range(end + 1, len(ticks)):
        if ticks[j][0] >= entry_ts + hold_sec:
            return ticks[j][1] / entry_price - 1.0
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--hold-min", type=float, default=12.0)
    ap.add_argument("--fee", type=float, default=0.003)
    ap.add_argument("--slippage", type=float, default=0.001)
    ap.add_argument("--lookback-min", type=float, default=120.0)
    ap.add_argument("--universe", choices=("all", "established", "discovered"),
                    default="all")
    args = ap.parse_args()

    hold_sec = args.hold_min * 60.0
    lookback = args.lookback_min * 60.0
    round_trip = 2.0 * args.fee + args.slippage

    by_symbol = load_ticks(args.hours)
    if args.universe == "established":
        by_symbol = {s: v for s, v in by_symbol.items() if s in ESTABLISHED}
    elif args.universe == "discovered":
        by_symbol = {s: v for s, v in by_symbol.items() if s not in ESTABLISHED}
    if not by_symbol:
        print("no ticks in this universe; nothing to sweep.")
        return 0

    control: List[float] = []
    hits: Dict[str, List[float]] = {name: [] for name in SIGNALS}

    for _symbol, raw in by_symbol.items():
        ticks = _drop_mixed_denominations(raw)
        if len(ticks) < 10:
            continue
        start = 0
        for end in range(len(ticks)):
            cutoff = ticks[end][0] - lookback
            while ticks[start][0] < cutoff:
                start += 1
            window = ticks[start:end + 1]
            if len(window) < 8:
                continue
            fwd = _forward(ticks, end, hold_sec)
            if fwd is None:
                continue
            control.append(fwd)
            for name, fn in SIGNALS.items():
                try:
                    if fn(window):
                        hits[name].append(fwd)
                except Exception:
                    continue

    c = _stats(control)
    k = len(SIGNALS)
    # Bonferroni on a two-sided 5% test: the bar each signal must clear, fixed
    # before looking at any result.
    bar = 2.0 + math.log(k) / 2.0 if k > 1 else 2.0
    bar = round(bar, 2)

    print("short-horizon signal sweep -- %.0fh, universe=%s, hold %.0f min"
          % (args.hours, args.universe, args.hold_min))
    print("round trip %.4f%%  |  %d signals tested, so the bar is %.2f standard "
          "errors (Bonferroni-corrected), fixed before looking\n"
          % (round_trip * 100, k, bar))
    print("control (every reachable tick): n=%d  mean %+.4f%%  se %.4f%%\n"
          % (c["n"], c["mean"] * 100, c["se"] * 100))

    print("%-20s %7s %10s %10s %8s %10s  %s"
          % ("signal", "n", "mean", "lift", "se's", "net", "verdict"))
    print("-" * 88)
    rows = []
    for name in SIGNALS:
        s = _stats(hits[name])
        if not s["n"]:
            print("%-20s %7d %10s" % (name, 0, "(never fired)"))
            continue
        lift = s["mean"] - c["mean"]
        se = math.sqrt(s["se"] ** 2 + c["se"] ** 2)
        sigmas = lift / se if se > 0 else 0.0
        net = s["mean"] - round_trip
        passes = sigmas > bar and net > 0
        verdict = ("WORTH BUILDING" if passes
                   else "no" if abs(sigmas) <= bar
                   else "significant but does not cover costs" if net <= 0
                   else "no")
        rows.append((name, s, lift, sigmas, net, passes))
        print("%-20s %7d %9.4f%% %9.4f%% %8.2f %9.4f%%  %s"
              % (name, s["n"], s["mean"] * 100, lift * 100, sigmas, net * 100, verdict))

    winners = [r for r in rows if r[5]]
    print()
    if winners:
        print("%d signal(s) cleared both bars:" % len(winners))
        for name, s, lift, sigmas, net, _ in winners:
            print("  * %s: %+.4f%% net over %d samples, %.2f se from the control"
                  % (name, net * 100, s["n"], sigmas))
        print("Before building any of these, re-run on a held-out window -- this "
              "sweep chose them, so it cannot also validate them.")
    else:
        print("No signal cleared both bars. On this feed, at this cost, over "
              "this window, none of the %d candidates is worth building." % k)
        print("That is a result about the FEED and the COST, not about any one "
              "strategy: the honest response is to fix what makes the cost "
              "unpayable (routing, fee tier, universe) rather than to keep "
              "searching for a selector.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
