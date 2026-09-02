"""Does the money_button signal know anything, or is it just trading?

The gate census answers "where does the lane die". It does not answer the
question that decides whether this lane should exist: **when money_button
fires, is the next twelve minutes any better than the next twelve minutes
would have been anyway?**

That distinction matters because the two failure modes look identical in a
P&L column. A lane can lose because its costs are too high for a real edge,
which is fixable by routing or by holding longer. Or it can lose because it
has no edge at all, in which case every knob on it is a way of losing money
more slowly, and the only honest move is to say so.

So this replays the strategy exactly as the census does, but scores the
forward return at EVERY tick that got past the feed gates -- the fires and
the declines alike. The declines are the control group. If the fired subset
does not beat that control, the gates are selecting noise, and a strategy
that selects noise cannot be tuned into one that does not.

Three things are reported and each is load-bearing:

  base rate      what a coin flip on this feed earns over the hold. If this
                 is already negative the feed itself is adverse and no
                 long-only lane on it can win.
  lift           fired mean minus base mean, with a standard error. A lift
                 smaller than its own error is not an edge, it is a sample.
  cost line      the round trip the lift has to clear before any of it is
                 real. An edge that exists but does not clear costs is still
                 a losing strategy, and saying "the signal works" about it
                 is how a ledger gets destroyed.

Usage:
    python scripts/money_button_edge_test.py
    python scripts/money_button_edge_test.py --hours 72 --fee 0.0065
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.money_button_gate_census import FEED_GATES, _State, load_ticks  # noqa: E402


def _drop_mixed_denominations(
    ticks: List[Tuple[float, float, float]], factor: float = 50.0
) -> List[Tuple[float, float, float]]:
    """Remove ticks quoted at a different scale from the rest of the symbol.

    The control group needs this more than the strategy does. Scored raw, the
    base rate came back as **+1.26e10%** mean forward return -- a number that
    is obviously not a market, produced by a handful of ticks that cross a
    denomination boundary (SPACEX-USDC 1.5e-9 -> 524.37). Left in, they swamp
    every statistic computed here and would have made the signal look
    catastrophically worse than the feed rather than merely no better.
    """
    if len(ticks) < 3:
        return list(ticks)
    prices = sorted(p for _ts, p, _v in ticks)
    median = prices[len(prices) // 2]
    if median <= 0:
        return list(ticks)
    return [
        t for t in ticks
        if t[1] > 0 and max(t[1] / median, median / t[1]) <= factor
    ]


def _forward_return(
    ticks: List[Tuple[float, float, float]], end: int, hold_sec: float
) -> Optional[float]:
    """Return over the hold starting at ``end``, or None if it runs off the data.

    Counting an unresolved hold as flat would invent a result, and flat is not
    a neutral value here -- it is better than the average trade after costs.
    """
    entry_ts, entry_price = ticks[end][0], ticks[end][1]
    if entry_price <= 0:
        return None
    exit_idx = next(
        (j for j in range(end + 1, len(ticks)) if ticks[j][0] >= entry_ts + hold_sec),
        None,
    )
    if exit_idx is None:
        return None
    return ticks[exit_idx][1] / entry_price - 1.0


def _stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "se": 0.0, "up": 0.0}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    sd = math.sqrt(var)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": sd / math.sqrt(n) if n else 0.0,
        "up": sum(1 for v in values if v > 0) / n,
    }


def _line(label: str, s: Dict[str, float], round_trip: float) -> str:
    if not s["n"]:
        return "%-34s (none)" % label
    clears = "n/a"
    return ("%-34s n=%-5d mean %+.4f%%  sd %.4f%%  se %.4f%%  up %.1f%%  "
            "net of costs %+.4f%%") % (
        label, s["n"], s["mean"] * 100, s["sd"] * 100, s["se"] * 100,
        s["up"] * 100, (s["mean"] - round_trip) * 100,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--fee", type=float, default=None,
                    help="one-way fee rate; default reads TRADE_FEE_RATE or 0.003")
    ap.add_argument("--hold-min", type=float, default=None,
                    help="override the holding period, to test a longer lane")
    args = ap.parse_args()

    from trading.strategies.base import StrategyContext
    from trading.strategies.money_button import MoneyButtonStrategy

    fee = args.fee if args.fee is not None else float(os.getenv("TRADE_FEE_RATE", "0.003"))
    hold_min = (args.hold_min if args.hold_min is not None
                else float(os.getenv("MONEY_BUTTON_HOLD_MIN", "12")))
    hold_sec = hold_min * 60.0
    slippage = float(os.getenv("MONEY_BUTTON_SLIPPAGE", "0.001"))
    round_trip = 2.0 * fee + slippage

    strat = MoneyButtonStrategy()
    lookback = strat.LOOKBACK_SEC
    by_symbol = load_ticks(args.hours)
    if not by_symbol:
        print("no ticks in the window; nothing to test.")
        return 0

    fired: List[float] = []
    control: List[float] = []
    #: The near-misses: everything the cost gate alone rejected. If these do
    #: better than the fires, the cost gate is selecting the wrong tail.
    cost_declined: List[float] = []
    #: What a long-only lane with no signal at all would have earned, i.e.
    #: entering on every tick the feed permitted.
    per_symbol_fired: Dict[str, List[float]] = defaultdict(list)
    per_symbol_control: Dict[str, List[float]] = defaultdict(list)

    dropped_ticks = 0
    for symbol, raw_ticks in by_symbol.items():
        ticks = _drop_mixed_denominations(raw_ticks)
        dropped_ticks += len(raw_ticks) - len(ticks)
        if len(ticks) < 3:
            continue
        start = 0
        for end in range(len(ticks)):
            cutoff = ticks[end][0] - lookback
            while ticks[start][0] < cutoff:
                start += 1
            window = ticks[start:end + 1]
            ctx = StrategyContext(
                chain="base",
                last_price=window[-1][1],
                last_volume=window[-1][2],
                fee_rate=fee,
                available_quote=10.0,
                available_base=0.0,
            )
            result = strat.evaluate(_State(window, symbol), ctx)
            decline = strat.last_decline
            if result is None and decline in FEED_GATES:
                # The lane never looked at the market here, so this tick is not
                # part of the population the gates were choosing from.
                continue
            fwd = _forward_return(ticks, end, hold_sec)
            if fwd is None:
                continue
            control.append(fwd)
            per_symbol_control[symbol].append(fwd)
            if result is not None:
                fired.append(fwd)
                per_symbol_fired[symbol].append(fwd)
            elif decline == "edge_below_cost":
                cost_declined.append(fwd)

    f, c = _stats(fired), _stats(control)
    d = _stats(cost_declined)

    print("money_button edge test -- %.0fh of stored ticks" % args.hours)
    print("one-way fee %.4f + slippage %.4f => round trip %.4f%%; hold %.0f min"
          % (fee, slippage, round_trip * 100, hold_min))
    print("%d tick(s) dropped as mixed-denomination before scoring\n" % dropped_ticks)

    print("forward return over the hold, gross (costs shown separately):")
    print(_line("every tick the lane could see", c, round_trip))
    print(_line("  ...of those, it FIRED on", f, round_trip))
    print(_line("  ...rejected by edge_below_cost", d, round_trip))

    if not f["n"]:
        print("\nThe lane never fired in this window; there is nothing to judge.")
        return 0

    lift = f["mean"] - c["mean"]
    # The fires are a subset of the control, but treating them as independent
    # is the conservative direction here: it makes the error bar wider, not
    # narrower, so a lift that survives it is not an artifact of the overlap.
    se = math.sqrt(f["se"] ** 2 + c["se"] ** 2)
    print("\nlift from the signal: %+.4f%% +/- %.4f%% (1 se)" % (lift * 100, se * 100))
    if se > 0:
        print("                      %.2f standard errors from zero" % (lift / se))
    print("cost it must clear:   %.4f%% round trip" % (round_trip * 100))

    verdict = []
    if c["mean"] < 0:
        verdict.append(
            "The feed itself is adverse over this hold (base rate %+.4f%%): a "
            "long-only lane starts behind before any signal is applied."
            % (c["mean"] * 100))
    if se > 0 and abs(lift) < se:
        verdict.append(
            "The lift is smaller than its own standard error. On this much "
            "data the signal is indistinguishable from selecting at random.")
    elif lift <= 0:
        verdict.append(
            "The signal selects WORSE-than-average ticks. The gates are not "
            "picking the right tail.")
    if f["mean"] < round_trip:
        verdict.append(
            "Even taking the fired mean at face value, %+.4f%% does not cover "
            "the %.4f%% round trip: every fire loses on average."
            % (f["mean"] * 100, round_trip * 100))
    if not verdict:
        verdict.append(
            "The fired subset beats the control by more than its error AND "
            "clears the round trip. This is the only combination under which "
            "the lane should trade live.")
    print("\nverdict:")
    for v in verdict:
        print("  * " + v)

    interesting = sorted(
        per_symbol_fired.items(), key=lambda kv: -len(kv[1]))[:10]
    if interesting:
        print("\nper symbol (fires only, gross):")
        print("  %-22s %6s %10s %10s %10s" % ("symbol", "fires", "mean", "base", "lift"))
        for symbol, vals in interesting:
            fs = _stats(vals)
            cs = _stats(per_symbol_control[symbol])
            print("  %-22s %6d %9.4f%% %9.4f%% %9.4f%%" % (
                symbol, fs["n"], fs["mean"] * 100, cs["mean"] * 100,
                (fs["mean"] - cs["mean"]) * 100))

    # Shape of the distribution any selector on this feed has to work with.
    #
    # The mean alone is misleading on a feed where a few microcaps move
    # hundreds of percent, so report the median and a trimmed mean beside it,
    # and -- decisively for a LONG-ONLY lane -- the two tails separately. If
    # the down tail is fatter than the up tail, buying is the wrong side of
    # this distribution no matter how the entry is timed.
    if control:
        ordered = sorted(control)
        n = len(ordered)
        lo, hi = int(0.05 * n), max(int(0.05 * n) + 1, int(0.95 * n))
        trimmed = sum(ordered[lo:hi]) / max(1, hi - lo)
        up = sum(1 for v in control if v > round_trip) / n
        down = sum(1 for v in control if v < -round_trip) / n
        flat = sum(1 for v in control if v == 0.0) / n
        print("\nshape of the reachable distribution over %.0f min:" % hold_min)
        print("  median %+.4f%%   trimmed mean (5%%) %+.4f%%"
              % (ordered[n // 2] * 100, trimmed * 100))
        print("  clears +%.2f%%: %.2f%%   falls past -%.2f%%: %.2f%%"
              % (round_trip * 100, up * 100, round_trip * 100, down * 100))
        print("  exactly flat: %.2f%%  (a repeated print, not a market)" % (flat * 100))
        if down > up:
            print("  => the down tail is FATTER than the up tail: a long-only "
                  "lane is on the wrong side of this feed before it picks a "
                  "single entry.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
