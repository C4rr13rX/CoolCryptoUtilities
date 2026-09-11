#!/usr/bin/env python3
"""How many held-out windows can the L1 motif rule SCORE, and does it pay?

WHY THIS EXISTS. Pass 113 measured the L1 motif->trough rule on ONE corpus and
found its UP window called zero held-out trades. Read from one corpus that looks
like a property of that corpus. It is not: swept over 102 windows, 83 of them
call nothing, so an ~81% abstention rate is what the rule DOES. A single-corpus
arm cannot see that, and this repo has already paid for a fake +0.9067% read off
one window.

THE CADENCE FILTER IS THE POINT, NOT A DETAIL. ``data/historical_ohlcv`` spans
166s to 345600s bars, so ``--horizon 12`` asks about 33 minutes on one file and
48 days on another ([4d0b539b]). A sweep that mixes cadences is not one
experiment. ``--cadence`` keeps only corpora whose MEDIAN bar spacing matches,
so the horizon means one thing across the whole sweep, and the value used is
printed.

WHAT IT REPORTS, and it deliberately separates the two things a single mean
hides: the ABSTENTION RATE (how many windows called nothing) and the TRADE-
WEIGHTED per-trade net against buy-every-bar, split by whether the window was
UP or DOWN. A long-only rule flatters itself in an up window, so one pooled
number across both is not a scoreboard.

    python -X utf8 scripts/omen_scorable_windows.py --chain base --cadence 3600

MEASURED 2026-09-10, 102 base corpora at 3600s, hysteresis 0.50, train 350 /
purge 12 / test 60:

    UP    11 corpora,  2 called,   6 trades  -1.3176%  vs  +0.6541% baseline
    DOWN  91 corpora, 17 called, 194 trades  -1.5575%  vs  -1.6617% baseline

The DOWN difference is +0.10pp over 194 trades, which is under one standard
error of zero at this feed's ~2% per-trade dispersion. That is a NEGATIVE, and
it is recorded in data/brain_experiments/L1-HYSTERESIS-pass113-cove.md.

NO NODE IS CONTACTED. Production runs :8090 and this never opens a socket.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.omen_layer_probe import heldout_edge, load_bars  # noqa: E402


def median_spacing(bars: List[Dict[str, Any]], sample: int = 200) -> float:
    """Median seconds between bars, over the first ``sample`` of them.

    The MEDIAN rather than the mean: a corpus with one multi-day gap would have
    its mean spacing dragged into a different cadence class by that one hole,
    and cadence classes are what make the horizon flag mean one thing.
    """
    stamps = [int(b["timestamp"]) for b in bars[:sample]]
    if len(stamps) < 3:
        return 0.0
    return statistics.median([b - a for a, b in zip(stamps, stamps[1:])])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", default="base")
    parser.add_argument("--cadence", type=float, default=3600.0,
                        help="keep only corpora whose MEDIAN bar spacing is "
                             "this many seconds, so --horizon means the same "
                             "number of minutes on every one of them")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train", type=int, default=350)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--hysteresis", type=float, default=0.50)
    parser.add_argument("--min-lift", type=float, default=1.3)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many corpora (a smoke run)")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    window = args.train + args.test + args.horizon + 60
    root = Path("data/historical_ohlcv") / args.chain
    if not root.is_dir():
        print("no corpus directory at %s" % root)
        return 2

    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if args.limit is not None and len(rows) >= args.limit:
            break
        try:
            bars = load_bars(path)
        except Exception as exc:              # a corpus we cannot read is data
            rows.append({"corpus": path.name, "error": str(exc)[:60]})
            continue
        if len(bars) < window or median_spacing(bars) != args.cadence:
            continue
        symbol = path.stem.split("_", 1)[-1]
        try:
            edge = heldout_edge(bars[-window:], symbol, args.chain,
                                args.horizon, args.train, args.test,
                                args.min_lift, args.min_support,
                                relative=True, margin=args.hysteresis)
        except Exception as exc:
            rows.append({"corpus": path.name, "error": type(exc).__name__})
            continue
        row = {"corpus": path.name, "symbol": symbol}
        for key in ("called_n", "called_net", "baseline_net",
                    "called_trough_rate", "base_trough_rate", "error"):
            if key in edge:
                value = edge[key]
                row[key] = round(value, 6) if isinstance(value, float) else value
        rows.append(row)

    scored = [r for r in rows if "called_n" in r]
    if not scored:
        print("no corpus in %s matched cadence %ss with >= %d bars"
              % (root, args.cadence, window))
        return 2

    def summarise(subset):
        called = [r for r in subset if r["called_n"] > 0]
        n = sum(r["called_n"] for r in called)
        if not n:
            return len(subset), 0, 0, 0.0, 0.0
        net = sum(r["called_net"] * r["called_n"] for r in called) / n
        base = sum(r["baseline_net"] * r["called_n"] for r in called) / n
        return len(subset), len(called), n, net, base

    up = [r for r in scored if r.get("baseline_net", 0.0) > 0]
    down = [r for r in scored if r.get("baseline_net", 0.0) <= 0]

    print("chain %s  cadence %ss  horizon %d bars (%.1f hours)  hysteresis %.2f"
          % (args.chain, args.cadence, args.horizon,
             args.cadence * args.horizon / 3600.0, args.hysteresis))
    print("train %d / purge %d / test %d, bands fitted on TRAIN and frozen"
          % (args.train, args.horizon, args.test))
    # A corpus that errored is NAMED rather than silently dropped: a sweep that
    # quietly loses corpora reports a smaller denominator as if it were the
    # whole population, which is the exact error the abstention line below
    # exists to prevent.
    failed = [r for r in rows if "called_n" not in r]
    print("corpora scored: %d   (failed to score: %d%s)"
          % (len(scored), len(failed),
             "" if not failed else " -- " + ", ".join(
                 "%s:%s" % (r["corpus"], r.get("error")) for r in failed[:5])))
    print("")
    print("  %-6s %8s %8s %8s %11s %11s %9s"
          % ("window", "corpora", "called", "trades", "per-trade", "baseline",
             "EDGE"))
    for name, subset in (("UP", up), ("DOWN", down), ("BOTH", scored)):
        corpora, called, trades, net, base = summarise(subset)
        print("  %-6s %8d %8d %8d %10.4f%% %10.4f%% %8.2fpp"
              % (name, corpora, called, trades, 100 * net, 100 * base,
                 100 * (net - base)))

    silent = sum(1 for r in scored if r["called_n"] == 0)
    print("\n  ABSTENTION: %d of %d windows called NOTHING (%.0f%%). Every "
          "number above is drawn from the other %d."
          % (silent, len(scored), 100.0 * silent / len(scored),
             len(scored) - silent))
    print("  A pooled mean over both window classes is NOT a scoreboard: a "
          "long-only rule flatters itself in an up window.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"chain": args.chain, "cadence": args.cadence,
             "horizon": args.horizon, "train": args.train, "test": args.test,
             "margin": args.hysteresis, "rows": rows}, indent=2),
            encoding="utf-8")
        print("\njson -> %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
