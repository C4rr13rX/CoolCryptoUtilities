"""Is there a holding period at which money_button's signal beats its cost?

`money_button_gate_census.py` scores every fire at ONE fixed hold (12 minutes)
and reports the lane as gross +0.385% against a 0.70% round trip -- a real
signal that does not cover its costs. But a fixed hold is an assumption, not a
measurement: a 5-minute edge scored at 12 minutes has had seven minutes to
decay, and the lane's one real trade exited at 281 seconds on a confidence
drop, not on a timer.

So this sweeps the hold instead of assuming it, and reports each horizon on the
same terms:

  * GROSS mean return, which is the signal
  * NET after the round trip, which is what the account actually receives
  * the SEQUENTIAL series (no re-entry while a position is open), because
    overlapping fires re-count one price move many times and inflate both the
    sample size and the apparent consistency. Thirteen simultaneous positions
    in one token is one bet, not thirteen.
  * a naive baseline: the same holds, entered at EVERY tick regardless of
    signal. A strategy whose returns match the baseline has no edge -- it is
    reporting the market's drift back to itself, which is how a trending token
    makes any long-only rule look skilful.

Reading it: the lane is worth funding only where sequential NET is positive AND
sequential GROSS beats the baseline gross. Either one alone is a trap -- the
first can come from a rising market, the second from a hold so short the cost
swamps it.

    python scripts/money_button_hold_sweep.py
    python scripts/money_button_hold_sweep.py --hours 168 --fee 0.003
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.money_button_gate_census import _State, load_ticks  # noqa: E402

#: Holds to score, in minutes. Spans the 5-30 minute lane the strategy is for,
#: with 60 included so a decaying edge can be seen decaying rather than assumed.
HOLDS_MIN = [2, 3, 5, 8, 12, 20, 30, 45, 60]

#: Sequential trades required before a horizon may be called profitable.
#:
#: Without this the first run of this script reported "5 holding periods clear
#: both the cost floor and the baseline", topped by 60 min at +4.66% net on a
#: 100% win rate. That was THREE trades. Split by symbol it was two on
#: BASECAT-USDC and one on BSTONK-USDC -- and BSTONK carried 151 ticks across
#: 48 hours, one every nineteen minutes, so its "60-minute hold" was whatever
#: the next tick happened to be. A tool built to test whether an edge is real
#: had produced exactly the shape of record this system has twice had to purge:
#: a perfect win rate, a tiny sample, and one token behind all of it.
MIN_SEQUENTIAL_TRADES = 20

#: A hold is only meaningful if the feed is sampled several times inside it.
#: Below this, the exit price is the next tick whenever it arrives, and the
#: horizon being tested is fiction.
MIN_TICKS_PER_HOLD = 3.0


def _score(
    entries: List[Tuple[str, int]],
    ticks_by_symbol: Dict[str, List[Tuple[float, float, float]]],
    hold_sec: float,
) -> Tuple[List[float], List[float], int]:
    """Returns (all_returns, sequential_returns, unresolved).

    A fire whose holding period runs past the end of the recorded data is left
    unscored rather than counted flat, which would invent a result.
    """
    every: List[float] = []
    sequential: List[float] = []
    unresolved = 0
    open_until: Dict[str, float] = {}
    for symbol, idx in entries:
        ticks = ticks_by_symbol[symbol]
        entry_ts, entry_price = ticks[idx][0], ticks[idx][1]
        exit_idx = None
        for j in range(idx + 1, len(ticks)):
            if ticks[j][0] >= entry_ts + hold_sec:
                exit_idx = j
                break
        if exit_idx is None:
            unresolved += 1
            continue
        ret = ticks[exit_idx][1] / entry_price - 1.0
        every.append(ret)
        if entry_ts >= open_until.get(symbol, 0.0):
            sequential.append(ret)
            open_until[symbol] = ticks[exit_idx][0]
    return every, sequential, unresolved


def _baseline(
    ticks_by_symbol: Dict[str, List[Tuple[float, float, float]]],
    hold_sec: float,
    stride: int,
) -> List[float]:
    """Same hold, entered on a schedule instead of on a signal."""
    out: List[float] = []
    for symbol, ticks in ticks_by_symbol.items():
        for idx in range(0, len(ticks), stride):
            entry_ts, entry_price = ticks[idx][0], ticks[idx][1]
            for j in range(idx + 1, len(ticks)):
                if ticks[j][0] >= entry_ts + hold_sec:
                    out.append(ticks[j][1] / entry_price - 1.0)
                    break
    return out


def _fmt(values: List[float], round_trip: float) -> str:
    if not values:
        return "%9s %8s %7s %8s" % ("-", "-", "-", "-")
    gross = statistics.fmean(values)
    net = gross - round_trip
    wins = sum(1 for v in values if v > round_trip)
    return "%8.4f%% %7.4f%% %6.1f%% %6d" % (
        gross * 100.0, net * 100.0, wins / len(values) * 100.0, len(values),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--fee", type=float, default=None,
                    help="one-way fee rate; default reads TRADE_FEE_RATE or 0.003")
    ap.add_argument("--baseline-stride", type=int, default=25,
                    help="sample every Nth tick for the no-signal baseline")
    args = ap.parse_args()

    from trading.strategies.base import StrategyContext
    from trading.strategies.money_button import MoneyButtonStrategy

    fee = args.fee if args.fee is not None else float(os.getenv("TRADE_FEE_RATE", "0.003"))
    slippage = float(os.getenv("MONEY_BUTTON_SLIPPAGE", "0.001"))
    round_trip = 2.0 * fee + slippage

    strat = MoneyButtonStrategy()
    lookback = strat.LOOKBACK_SEC
    by_symbol = load_ticks(args.hours)
    if not by_symbol:
        print("no ticks in the window; nothing to sweep.")
        return 0

    # Collect the fires ONCE; the hold does not change what the strategy sees
    # at entry, only how the entry is scored.
    entries: List[Tuple[str, int]] = []
    evaluated = 0
    for symbol, ticks in by_symbol.items():
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
            evaluated += 1
            if strat.evaluate(_State(window, symbol), ctx) is not None:
                entries.append((symbol, end))

    print("money_button hold sweep -- %.0fh, one-way fee %.4f, slippage %.4f"
          % (args.hours, fee, slippage))
    print("round trip = %.4f%% ; %d symbols, %d evaluations, %d fires\n"
          % (round_trip * 100.0, len(by_symbol), evaluated, len(entries)))
    if not entries:
        print("the lane never fired in this window; nothing to score.")
        return 0

    print("%5s | %-33s | %-33s | %s" % (
        "hold", "ALL FIRES (overlapping)", "SEQUENTIAL (one at a time)", "NO-SIGNAL BASELINE"))
    print("%5s | %8s %8s %7s %6s | %8s %8s %7s %6s | %8s %6s" % (
        "min", "gross", "net", "win%", "n", "gross", "net", "win%", "n", "gross", "n"))
    print("-" * 108)

    verdicts = []
    for minutes in HOLDS_MIN:
        hold_sec = minutes * 60.0
        every, seq, _unresolved = _score(entries, by_symbol, hold_sec)
        base = _baseline(by_symbol, hold_sec, args.baseline_stride)
        print("%5d | %s | %s | %7.4f%% %6d" % (
            minutes, _fmt(every, round_trip), _fmt(seq, round_trip),
            (statistics.fmean(base) * 100.0) if base else 0.0, len(base),
        ))
        if seq:
            seq_gross = statistics.fmean(seq)
            base_gross = statistics.fmean(base) if base else 0.0
            verdicts.append((minutes, seq_gross - round_trip, seq_gross - base_gross, len(seq)))

    # Where did the fires come from? A horizon that looks profitable across two
    # symbols is two bets, not a strategy, and the per-symbol split is the only
    # place that shows it.
    per_symbol: Dict[str, List[Tuple[str, int]]] = {}
    for symbol, idx in entries:
        per_symbol.setdefault(symbol, []).append((symbol, idx))
    print("\nfires by symbol, and whether the feed can even resolve the hold:")
    print("  %-18s %6s %8s %10s   %s" % ("symbol", "fires", "ticks", "mean gap", "48h price move"))
    for symbol, ents in sorted(per_symbol.items(), key=lambda kv: -len(kv[1])):
        ticks = by_symbol[symbol]
        span = ticks[-1][0] - ticks[0][0]
        gap = span / max(1, len(ticks) - 1)
        move = (ticks[-1][1] / ticks[0][1] - 1.0) * 100.0
        print("  %-18s %6d %8d %8.1fs   %+.1f%%" % (
            symbol, len(ents), len(ticks), gap, move))

    print()
    qualified = [
        v for v in verdicts
        if v[1] > 0 and v[2] > 0 and v[3] >= MIN_SEQUENTIAL_TRADES
    ]
    promising = [v for v in verdicts if v[1] > 0 and v[2] > 0 and v[3] < MIN_SEQUENTIAL_TRADES]

    if qualified:
        print("VERDICT: %d holding period(s) clear the cost floor, the baseline, and"
              % len(qualified))
        print("the %d-trade minimum:" % MIN_SEQUENTIAL_TRADES)
        for minutes, net, edge, n in sorted(qualified, key=lambda v: -v[1]):
            print("  %2d min: net %+.4f%% per trade, %+.4f%% over baseline, %d sequential trades"
                  % (minutes, net * 100.0, edge * 100.0, n))
        print("\n  Still one window. Check the per-symbol split above before funding it:")
        print("  a horizon carried by one token is one bet.")
        return 0

    print("VERDICT: no holding period is supported by enough evidence to act on.")
    if promising:
        print("\n  These cleared cost and baseline but NOT the %d-trade minimum:"
              % MIN_SEQUENTIAL_TRADES)
        for minutes, net, edge, n in sorted(promising, key=lambda v: -v[1]):
            print("    %2d min: net %+.4f%% over %d sequential trade%s -- too few to mean anything"
                  % (minutes, net * 100.0, n, "" if n == 1 else "s"))
        print("\n  A 100% win rate over three trades is not a 100% win rate. This is the")
        print("  shape of record that has twice been purged from this system: a perfect")
        print("  number, a tiny sample, and one token behind all of it.")
    else:
        best_net = max(verdicts, key=lambda v: v[1]) if verdicts else None
        if best_net:
            print("  best sequential net was %+.4f%% at %d min over %d trades"
                  % (best_net[1] * 100.0, best_net[0], best_net[3]))
    print("\n  What IS established: the lane's gating works -- it declined %d of %d"
          % (evaluated - len(entries), evaluated))
    print("  looks -- and its cost floor is %.2f%% a round trip. What is NOT"
          % (round_trip * 100.0))
    print("  established is an edge. The block is sample size, not the strategy;")
    print("  it fired %d times in %.0fh across %d symbols. Collect more before"
          % (len(entries), args.hours, len(per_symbol)))
    print("  concluding either way -- and do not loosen the gates to get there,")
    print("  which buys volume with money instead of information.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
