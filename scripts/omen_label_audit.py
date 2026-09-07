"""Is the omen brain being taught the wrong answer?

Three passes have moved the omen brain's train recall from 89.2% to 98.7% by
fixing how a bar is REPRESENTED, and all three ended on the same wall: no
held-out edge. This script asks the question none of them asked -- whether the
TARGET is right -- and it answers it offline, with no node and no retrain.

``trading.omen_brain.label_omen`` labels the endpoint: where is the close
``horizon`` bars from now, against the round-trip cost.
``trading.omen_path.label_path`` labels the path: does a long opened here
reach its take-profit before its stop, and how long did it take.

The four numbers that decide which target is worth learning:

  1. MISSED TRADES   -- of the bars the endpoint label calls ``murk`` (do not
                        trade), what fraction actually completed a winning
                        round trip? Every one of those is a fast profitable
                        trade the brain is being trained to refuse.
  2. FALSE WINS      -- of the bars the endpoint label calls a buy, what
                        fraction hit the stop first? Every one of those is a
                        realised loss the brain is being trained to want.
  3. ORACLE P/L      -- net-of-cost return per trade for a PERFECT predictor
                        of each label. This is the ceiling each target can
                        pay. If the endpoint ceiling is lower, no amount of
                        recall on it can beat the path target.
  4. HOLD TIME       -- bars, and minutes at the corpus's own cadence. The
                        endpoint label always holds exactly ``horizon``. The
                        path label holds until a barrier. This loop's target
                        is single-digit to tens of minutes.

Usage
-----
  python -X utf8 scripts/omen_label_audit.py --symbols 12 --horizon 12
  python -X utf8 scripts/omen_label_audit.py --corpus data/historical_ohlcv/base/0004_AERO-USDC.json
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

from trading.omen_brain import (  # noqa: E402
    OMEN_CLIMB, OMEN_CREST, OMEN_MURK, OMEN_SLIDE, OMEN_TROUGH,
    ROUND_TRIP_COST, label_omen, omen_threshold,
)
from trading.omen_path import (  # noqa: E402
    PATH_FLAT, PATH_STOP, PATH_WIN, omen_take, walk_path,
)

#: Endpoint labels that tell a strategy to buy.
BUY_LABELS = (OMEN_TROUGH, OMEN_CLIMB)


def load_bars(path: Path) -> List[Dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    bars = raw if isinstance(raw, list) else raw.get("bars") or raw.get("data") or []
    bars = [b for b in bars
            if isinstance(b, dict) and b.get("close") and b.get("timestamp")]
    bars.sort(key=lambda b: int(b["timestamp"]))
    return bars


def bar_seconds(bars: Sequence[Mapping[str, Any]]) -> int:
    gaps = [int(bars[i]["timestamp"]) - int(bars[i - 1]["timestamp"])
            for i in range(1, min(len(bars), 400))]
    gaps = [g for g in gaps if g > 0]
    return int(sorted(gaps)[len(gaps) // 2]) if gaps else 3600


def has_ohlc(bars: Sequence[Mapping[str, Any]]) -> bool:
    """A corpus with no high/low cannot be walked -- say so, never guess."""
    sample = bars[: min(len(bars), 200)]
    with_hl = sum(1 for b in sample if b.get("high") and b.get("low"))
    return with_hl >= 0.9 * len(sample)


def audit_corpus(bars: Sequence[Mapping[str, Any]], *, horizon: int,
                 take: float, stop: float, cost: float) -> Dict[str, Any]:
    """Both labels on every bar with a future. One walk, one endpoint, no node."""
    cross: Counter = Counter()
    endpoint_returns: List[float] = []      # net, oracle-on-endpoint
    path_returns: List[float] = []          # net, oracle-on-path
    path_all: List[float] = []              # net of EVERY bar walked long
    hold_bars: List[int] = []
    ambiguous = 0
    walked = 0

    for index in range(0, len(bars) - horizon - 1):
        endpoint = label_omen(bars, index, horizon_bars=horizon)
        outcome = walk_path(bars, index, horizon_bars=horizon,
                            take=take, stop=stop)
        if endpoint is None or outcome is None:
            continue
        walked += 1
        ambiguous += int(outcome.ambiguous)
        cross[(endpoint, outcome.outcome)] += 1
        path_all.append(outcome.net(cost))

        # Oracle on the ENDPOINT label: it buys every bar its own label calls
        # a buy, and exits at the horizon close -- no stop, because the
        # endpoint label has no concept of one.
        if endpoint in BUY_LABELS:
            entry = float(bars[index]["close"])
            exit_price = float(bars[index + horizon]["close"])
            endpoint_returns.append((exit_price - entry) / entry - abs(cost))

        # Oracle on the PATH label: it buys every bar its own label calls a
        # win, and the barrier closes it.
        if outcome.is_win:
            path_returns.append(outcome.net(cost))
            hold_bars.append(outcome.bars_held)

    return {
        "walked": walked,
        "ambiguous": ambiguous,
        "cross": {f"{a}|{b}": n for (a, b), n in sorted(cross.items())},
        "endpoint_oracle": endpoint_returns,
        "path_oracle": path_returns,
        "path_all": path_all,
        "hold_bars": hold_bars,
    }


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean_pct": 100.0 * statistics.fmean(values),
        "median_pct": 100.0 * statistics.median(values),
        "total_pct": 100.0 * sum(values),
        "win_rate": sum(1 for v in values if v > 0) / len(values),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", default=[],
                        help="explicit corpus path; repeatable")
    parser.add_argument("--symbols", type=int, default=12,
                        help="how many corpora to sweep when none given")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--take", type=float, default=None,
                        help="take-profit as a FRACTION; default omen_take()")
    parser.add_argument("--stop", type=float, default=None,
                        help="stop as a FRACTION; default = take")
    parser.add_argument("--cost", type=float, default=ROUND_TRIP_COST)
    parser.add_argument("--min-bars", type=int, default=2000)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    take = args.take if args.take is not None else omen_take()
    stop = args.stop if args.stop is not None else take

    if args.corpus:
        paths = [Path(p) for p in args.corpus]
    else:
        root = ROOT / "data" / "historical_ohlcv"
        paths = sorted(root.rglob("*.json"))

    print(f"omen label audit -- endpoint vs path")
    print(f"  horizon      {args.horizon} bars")
    print(f"  take / stop  {100*take:.4f}% / {100*stop:.4f}%")
    print(f"  round trip   {100*abs(args.cost):.4f}%")
    print(f"  endpoint thr {100*omen_threshold(args.cost):.4f}%")
    print()

    totals: Counter = Counter()
    endpoint_all: List[float] = []
    path_all_win: List[float] = []
    path_every: List[float] = []
    holds: List[int] = []
    ambiguous = 0
    walked = 0
    used: List[str] = []
    cadences: List[int] = []

    for path in paths:
        if len(used) >= args.symbols and not args.corpus:
            break
        bars = load_bars(path)
        if len(bars) < args.min_bars or not has_ohlc(bars):
            continue
        result = audit_corpus(bars, horizon=args.horizon, take=take,
                              stop=stop, cost=args.cost)
        if result["walked"] < 200:
            continue
        used.append(path.name)
        cadences.append(bar_seconds(bars))
        for key, count in result["cross"].items():
            totals[key] += count
        endpoint_all.extend(result["endpoint_oracle"])
        path_all_win.extend(result["path_oracle"])
        path_every.extend(result["path_all"])
        holds.extend(result["hold_bars"])
        ambiguous += result["ambiguous"]
        walked += result["walked"]

    if not walked:
        print("no corpus had both OHLC and enough bars -- nothing measured.")
        return 1

    cadence = int(statistics.median(cadences)) if cadences else 3600
    print(f"corpora {len(used)}  bars walked {walked}  "
          f"cadence {cadence}s ({cadence/60:.0f} min/bar)")
    print()

    # --- 1. missed trades ------------------------------------------------
    murk_total = sum(n for k, n in totals.items() if k.startswith(OMEN_MURK + "|"))
    murk_win = totals.get(f"{OMEN_MURK}|{PATH_WIN}", 0)
    murk_stop = totals.get(f"{OMEN_MURK}|{PATH_STOP}", 0)
    print("1. MISSED TRADES -- bars the endpoint label refuses that actually paid")
    if murk_total:
        print(f"   endpoint says murk : {murk_total}")
        print(f"   path says win      : {murk_win}  ({100*murk_win/murk_total:.2f}%)")
        print(f"   path says stop     : {murk_stop}  ({100*murk_stop/murk_total:.2f}%)")
        print(f"   net of the two     : {murk_win - murk_stop:+d} trades "
              f"({100*(murk_win-murk_stop)/murk_total:+.2f}% of murk)")
    print()

    # --- 2. false wins ---------------------------------------------------
    buy_total = sum(n for k, n in totals.items()
                    if k.split("|")[0] in BUY_LABELS)
    buy_stop = sum(totals.get(f"{lbl}|{PATH_STOP}", 0) for lbl in BUY_LABELS)
    buy_win = sum(totals.get(f"{lbl}|{PATH_WIN}", 0) for lbl in BUY_LABELS)
    print("2. FALSE WINS -- bars the endpoint label calls a buy that stop out first")
    if buy_total:
        print(f"   endpoint says buy  : {buy_total}")
        print(f"   path says win      : {buy_win}  ({100*buy_win/buy_total:.2f}%)")
        print(f"   path says stop     : {buy_stop}  ({100*buy_stop/buy_total:.2f}%)")
    print()

    # --- 3. oracle P/L ---------------------------------------------------
    endpoint_summary = _summary(endpoint_all)
    path_summary = _summary(path_all_win)
    every_summary = _summary(path_every)
    print("3. ORACLE P/L -- what a PERFECT predictor of each label earns, net of cost")
    for name, summary in (("endpoint label", endpoint_summary),
                          ("path label    ", path_summary),
                          ("every bar long", every_summary)):
        if summary["n"]:
            print(f"   {name}: n={summary['n']:6d}  "
                  f"mean {summary['mean_pct']:+.4f}%/trade  "
                  f"win rate {100*summary['win_rate']:.1f}%  "
                  f"total {summary['total_pct']:+.1f}%")
        else:
            print(f"   {name}: n=0")
    if endpoint_summary["n"] and path_summary["n"]:
        delta = path_summary["mean_pct"] - endpoint_summary["mean_pct"]
        print(f"   path - endpoint    : {delta:+.4f}%/trade")
    print()

    # --- 4. hold time ----------------------------------------------------
    print("4. HOLD TIME -- how long a winning trade is actually open")
    if holds:
        ordered = sorted(holds)
        def at(f: float) -> int:
            return ordered[min(len(ordered) - 1, int(f * len(ordered)))]
        print(f"   path wins  : median {at(0.5)} bars "
              f"({at(0.5)*cadence/60:.0f} min), "
              f"p90 {at(0.9)} bars ({at(0.9)*cadence/60:.0f} min)")
    print(f"   endpoint   : always {args.horizon} bars "
          f"({args.horizon*cadence/60:.0f} min)")
    print()

    print(f"ambiguous deciding bars (both barriers in one bar, counted as stops): "
          f"{ambiguous} ({100*ambiguous/walked:.2f}% of walks)")

    payload = {
        "horizon": args.horizon, "take": take, "stop": stop,
        "cost": args.cost, "cadence_seconds": cadence,
        "corpora": used, "walked": walked, "ambiguous": ambiguous,
        "cross": dict(totals),
        "endpoint_oracle": endpoint_summary,
        "path_oracle": path_summary,
        "every_bar": every_summary,
        "hold_bars_median": statistics.median(holds) if holds else None,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
