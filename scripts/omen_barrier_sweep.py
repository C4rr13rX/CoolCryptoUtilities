"""Where does a take/stop pair stop being geared to lose?

``scripts/omen_label_audit.py`` measured that the omen brain's target, with
take == stop == 0.975% against a 0.65% round trip, needs an 83.3% win rate to
break even. A win nets ``T - c``; a loss nets ``-(S + c)``. The cost is paid
on BOTH, which is why a symmetric barrier is not a fair coin -- it is a coin
that pays you 0.325 when it lands heads and takes 1.625 when it lands tails.

    break-even win rate   p* = (S + c) / (T + S)

For p* = 50% you need ``T = S + 2c``: the target must beat the stop by TWO
round trips. Nothing in the current labelling knows that.

This script sweeps (take, stop, horizon) over real bars and reports, for each
combination, the MEASURED outcome mix against that combination's own p*. It
answers three things no single-point measurement can:

  * is there any (T, S, h) where buying indiscriminately is already positive?
    If so the brain's job is only to make it more positive.
  * where is the gap between measured p and required p* smallest? That is the
    target worth teaching, because it needs the least skill to pay.
  * what does it cost in TIME? A scheme that pays 0.4%/trade over 40 bars is
    worse for this loop than one that pays 0.1% over 3.

Every return here is a signed FRACTION of entry, net of one round trip.
``per_hour`` divides by the real time the position was open, using the
corpus's own measured cadence -- never an assumed one.

Usage
-----
  python -X utf8 scripts/omen_barrier_sweep.py --cadence 300 --symbols 10
  python -X utf8 scripts/omen_barrier_sweep.py --cadence 3600 --symbols 12 --report out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.omen_brain import ROUND_TRIP_COST  # noqa: E402
from trading.omen_path import (  # noqa: E402
    PATH_FLAT, PATH_STOP, PATH_WIN, walk_path,
)
from scripts.omen_label_audit import bar_seconds, has_ohlc, load_bars  # noqa: E402


def break_even(take: float, stop: float, cost: float) -> float:
    """The win rate a (take, stop) pair needs just to return zero.

    Derivation, written out because this repo has shipped the version that
    forgets the second term: acting N times, you win pN of them for
    ``take - cost`` each and lose (1-p)N for ``stop + cost`` each. Setting the
    sum to zero gives ``p (take + stop) = stop + cost``.
    """
    denominator = abs(take) + abs(stop)
    if denominator <= 0:
        return 1.0
    return (abs(stop) + abs(cost)) / denominator


def sweep_corpus(bars: Sequence[Mapping[str, Any]], combos: Sequence[tuple],
                 cost: float, stride: int) -> Dict[tuple, Dict[str, Any]]:
    out: Dict[tuple, Dict[str, Any]] = {}
    for take, stop, horizon in combos:
        mix: Counter = Counter()
        nets: List[float] = []
        bars_held: List[int] = []
        for index in range(0, len(bars) - horizon - 1, stride):
            outcome = walk_path(bars, index, horizon_bars=horizon,
                                take=take, stop=stop)
            if outcome is None:
                continue
            mix[outcome.outcome] += 1
            nets.append(outcome.net(cost))
            bars_held.append(outcome.bars_held)
        if not nets:
            continue
        out[(take, stop, horizon)] = {
            "n": len(nets), "mix": dict(mix),
            "sum_net": sum(nets), "sum_bars": sum(bars_held),
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", type=int, default=300,
                        help="only use corpora whose measured bar cadence is this")
    parser.add_argument("--cadence-tol", type=float, default=0.10)
    parser.add_argument("--symbols", type=int, default=10)
    parser.add_argument("--min-bars", type=int, default=1500)
    parser.add_argument("--cost", type=float, default=ROUND_TRIP_COST)
    parser.add_argument("--stride", type=int, default=1,
                        help="sample every Nth bar; 1 walks every bar")
    parser.add_argument("--takes", default="0.0065,0.0098,0.0130,0.0195,0.0260,0.0390")
    parser.add_argument("--stops", default="0.0033,0.0049,0.0065,0.0098,0.0130")
    parser.add_argument("--horizons", default="3,6,12,24")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    takes = [float(v) for v in args.takes.split(",") if v.strip()]
    stops = [float(v) for v in args.stops.split(",") if v.strip()]
    horizons = [int(v) for v in args.horizons.split(",") if v.strip()]
    combos = [(t, s, h) for t in takes for s in stops for h in horizons]

    root = ROOT / "data" / "historical_ohlcv"
    chosen: List[Path] = []
    cadences: List[int] = []
    for path in sorted(root.rglob("*.json")):
        if len(chosen) >= args.symbols:
            break
        bars = load_bars(path)
        if len(bars) < args.min_bars or not has_ohlc(bars):
            continue
        cadence = bar_seconds(bars)
        if abs(cadence - args.cadence) > args.cadence_tol * args.cadence:
            continue
        chosen.append(path)
        cadences.append(cadence)

    if not chosen:
        print(f"no corpus at cadence {args.cadence}s with >= {args.min_bars} "
              f"bars and OHLC -- nothing measured.")
        return 1

    cadence = int(statistics.median(cadences))
    print(f"barrier sweep -- cadence {cadence}s ({cadence/60:.1f} min/bar), "
          f"{len(chosen)} corpora, round trip {100*abs(args.cost):.4f}%")
    print(f"corpora: {', '.join(p.name for p in chosen)}")
    print()

    totals: Dict[tuple, Dict[str, Any]] = {}
    for path in chosen:
        bars = load_bars(path)
        for key, value in sweep_corpus(bars, combos, args.cost, args.stride).items():
            slot = totals.setdefault(
                key, {"n": 0, "mix": Counter(), "sum_net": 0.0, "sum_bars": 0})
            slot["n"] += value["n"]
            slot["mix"].update(value["mix"])
            slot["sum_net"] += value["sum_net"]
            slot["sum_bars"] += value["sum_bars"]

    rows = []
    for (take, stop, horizon), value in totals.items():
        n = value["n"]
        if n < 200:
            continue
        wins = value["mix"].get(PATH_WIN, 0)
        stops_hit = value["mix"].get(PATH_STOP, 0)
        flats = value["mix"].get(PATH_FLAT, 0)
        decided = wins + stops_hit
        measured_p = wins / decided if decided else 0.0
        needed_p = break_even(take, stop, args.cost)
        mean_net = value["sum_net"] / n
        mean_bars = value["sum_bars"] / n
        hours = (mean_bars * cadence) / 3600.0
        rows.append({
            "take": take, "stop": stop, "horizon": horizon, "n": n,
            "win": wins, "stop_hit": stops_hit, "flat": flats,
            "measured_p": measured_p, "needed_p": needed_p,
            "edge_p": measured_p - needed_p,
            "mean_net": mean_net,
            "mean_bars": mean_bars,
            "mean_minutes": mean_bars * cadence / 60.0,
            "net_per_hour": mean_net / hours if hours > 0 else 0.0,
        })

    if not rows:
        print("every combination had fewer than 200 walks -- nothing measured.")
        return 1

    rows.sort(key=lambda r: r["mean_net"], reverse=True)
    header = (f"{'take%':>7} {'stop%':>7} {'h':>3} {'n':>7} "
              f"{'win':>7} {'stop':>7} {'flat':>7} "
              f"{'p':>7} {'p*':>7} {'p-p*':>7} "
              f"{'net%/trade':>11} {'min held':>9} {'net%/hr':>9}")
    print("ALL COMBINATIONS, best net-per-trade first")
    print(header)
    print("-" * len(header))
    for row in rows[:24]:
        print(f"{100*row['take']:7.3f} {100*row['stop']:7.3f} {row['horizon']:3d} "
              f"{row['n']:7d} {row['win']:7d} {row['stop_hit']:7d} {row['flat']:7d} "
              f"{100*row['measured_p']:6.2f}% {100*row['needed_p']:6.2f}% "
              f"{100*row['edge_p']:+6.2f}% "
              f"{100*row['mean_net']:+10.4f}% {row['mean_minutes']:9.1f} "
              f"{100*row['net_per_hour']:+8.4f}%")
    print()

    positive = [r for r in rows if r["mean_net"] > 0]
    print(f"combinations with POSITIVE unconditional net: "
          f"{len(positive)} of {len(rows)}")
    beats = [r for r in rows if r["edge_p"] > 0]
    print(f"combinations where measured p beats break-even p*: "
          f"{len(beats)} of {len(rows)}")
    if beats:
        best = max(beats, key=lambda r: r["edge_p"])
        print(f"  widest margin: take {100*best['take']:.3f}% "
              f"stop {100*best['stop']:.3f}% h={best['horizon']} -> "
              f"p {100*best['measured_p']:.2f}% vs p* {100*best['needed_p']:.2f}% "
              f"({100*best['edge_p']:+.2f} points), "
              f"net {100*best['mean_net']:+.4f}%/trade over "
              f"{best['mean_minutes']:.0f} min")
    print()
    print("THE SMALLEST SKILL REQUIRED, ranked by how few points of edge over")
    print("indiscriminate entry the brain must supply to make the scheme pay:")
    by_gap = sorted(rows, key=lambda r: r["needed_p"] - r["measured_p"])
    for row in by_gap[:8]:
        gap = row["needed_p"] - row["measured_p"]
        print(f"   take {100*row['take']:6.3f}% stop {100*row['stop']:6.3f}% "
              f"h={row['horizon']:3d}: brain must add "
              f"{100*gap:+6.2f} points of win rate "
              f"(decided {row['win']+row['stop_hit']:6d}, "
              f"{row['mean_minutes']:6.1f} min held)")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(
            {"cadence_seconds": cadence, "cost": args.cost,
             "corpora": [p.name for p in chosen], "rows": rows},
            indent=2), encoding="utf-8")
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
