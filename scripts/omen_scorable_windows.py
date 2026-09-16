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
from scripts.omen_experiment import (  # noqa: E402
    MIN_READABLE_HELDOUT_BARS_AT_3600S, UNREADABLE, readability_cell,
    validate_report_readability,
)
from trading.omen_brain import OMEN_CREST, OMEN_TROUGH  # noqa: E402


def pooled_readability(subset: List[Dict[str, Any]], *,
                       trades: int, net_per_trade: float) -> Dict[str, Any]:
    """The label ceiling of a POOLED window class, and the cell it guards.

    A sweep hides the ceiling better than a single corpus does, because pooling
    194 trades across 17 corpora looks like a sample even when every one of
    those corpora held nine troughs. It is not a different rule here: the
    ceiling is additive, so the pooled cell is scored against the pooled label
    count over the pooled held-out bars, and a pooled per-trade net is
    quotable only when the windows it came from could have produced 30 calls
    between them.

    Corpora that called NOTHING are in the denominator of the bars and the
    labels on purpose -- they are held-out windows this rule was scored on, and
    dropping them would price the ceiling off only the windows that spoke.
    """
    label_n = sum(int(r.get("heldout_trough_labels") or 0) for r in subset)
    crest_n = sum(int(r.get("heldout_crest_labels") or 0) for r in subset)
    window_bars = sum(int(r.get("heldout_window_bars") or 0) for r in subset)
    buy = readability_cell("buy", OMEN_TROUGH, label_n, window_bars,
                           trades=int(trades),
                           net_per_trade=net_per_trade if trades else None)
    sell = readability_cell("sell", OMEN_CREST, crest_n, window_bars,
                            trades=None, net_per_trade=None, scored=False)
    return {
        "readable_label_floor": buy["floor"],
        "min_readable_heldout_bars_at_3600s": MIN_READABLE_HELDOUT_BARS_AT_3600S,
        "heldout_window_bars": window_bars,
        "heldout_trough_labels": label_n,
        "heldout_trough_base_rate": (label_n / window_bars) if window_bars else None,
        "heldout_crest_labels": crest_n,
        "heldout_crest_base_rate": (crest_n / window_bars) if window_bars else None,
        "corpora_pooled": len(subset),
        "readability": {"buy": buy, "sell": sell},
        "buy_net_per_trade": buy["net_per_trade"],
        "sell_net_per_trade": sell["net_per_trade"],
        "heldout_readable": bool(buy["readable"]),
    }


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


def build_parser() -> argparse.ArgumentParser:
    """Split out from ``main`` so the DEFAULT WINDOW is testable without a sweep.

    The default is the measurement here: this harness published a 60-bar
    per-corpus window for two months, and a test that has to run a 102-corpus
    sweep to see that number is a test nobody runs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", default="base")
    parser.add_argument("--cadence", type=float, default=3600.0,
                        help="keep only corpora whose MEDIAN bar spacing is "
                             "this many seconds, so --horizon means the same "
                             "number of minutes on every one of them")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train", type=int, default=350)
    # RAISED FROM 60 IN PASS 423, AND 60 IS WHY THE PUBLISHED SWEEP IS
    # UNREADABLE. Measured node-free on 0004_AERO-USDC: 656 of 656 candidate
    # 60-bar windows hold fewer than 30 trough labels, median 9. The
    # 2026-09-10 sweep in this docstring pooled 194 trades over 17 such
    # windows; pooling does not raise a per-window ceiling, it only hides it.
    parser.add_argument("--test", type=int,
                        default=MIN_READABLE_HELDOUT_BARS_AT_3600S)
    parser.add_argument("--hysteresis", type=float, default=0.50)
    parser.add_argument("--min-lift", type=float, default=1.3)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many corpora (a smoke run)")
    parser.add_argument("--json-out", default=None)
    return parser


def main() -> int:
    parser = build_parser()
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
        # THE CEILING THIS CORPUS IMPOSES, carried per row so the pooled cell
        # is scored against the labels the pooled windows actually held rather
        # than against the 14.46% median of some other corpus.
        counts = edge.get("heldout_label_counts") or {}
        row["heldout_trough_labels"] = int(counts.get(OMEN_TROUGH, 0))
        row["heldout_crest_labels"] = int(counts.get(OMEN_CREST, 0))
        row["heldout_window_bars"] = int(edge.get("test_n") or 0)
        # THE PER-CORPUS CELL IS WHERE THE 9-LABEL CEILING BITES. ``called_net``
        # is this one window's per-trade mean, and at the old 60-bar default no
        # window in this corpus directory held 30 troughs, so every row in
        # every published sweep was a percentage its own window could not
        # carry. The float survives under ``called_net_raw`` for the pooled
        # arithmetic below, which is a different cell with a different ceiling.
        if "called_net" in row:
            row["called_net_raw"] = row["called_net"]
            cell = readability_cell(
                "buy", OMEN_TROUGH, row["heldout_trough_labels"],
                row["heldout_window_bars"], trades=int(row["called_n"]),
                net_per_trade=row["called_net"])
            row["readability"] = {"buy": cell}
            row["called_net"] = cell["net_per_trade"]
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
        # ``called_net_raw``: the row's quotable key is guarded and may be the
        # string UNREADABLE, so the pooled arithmetic reads the float that was
        # deliberately renamed rather than the cell a reader quotes.
        net = sum(r["called_net_raw"] * r["called_n"] for r in called) / n
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
    classes: Dict[str, Dict[str, Any]] = {}
    for name, subset in (("UP", up), ("DOWN", down), ("BOTH", scored)):
        corpora, called, trades, net, base = summarise(subset)
        block = pooled_readability(subset, trades=trades, net_per_trade=net)
        block.update({"corpora": corpora, "called_corpora": called,
                      "trades": trades, "buy_net_per_trade_raw": net,
                      "baseline_net_per_trade": base})
        classes[name] = block
        if block["heldout_readable"]:
            print("  %-6s %8d %8d %8d %10.4f%% %10.4f%% %8.2fpp"
                  % (name, corpora, called, trades, 100 * net, 100 * base,
                     100 * (net - base)))
            continue
        # THE CELL, NOT A FOOTNOTE. A per-trade percentage printed with a
        # caveat underneath it is the percentage that gets quoted.
        print("  %-6s %8d %8d %8d %11s %10.4f%% %9s"
              % (name, corpora, called, trades, UNREADABLE, 100 * base, "--"))
        print("         ^ %d trough labels in %d pooled held-out bars "
              "(%.2f%%), floor %d -- %s"
              % (block["heldout_trough_labels"], block["heldout_window_bars"],
                 100 * (block["heldout_trough_base_rate"] or 0.0),
                 block["readable_label_floor"],
                 "; ".join(block["readability"]["buy"]["unreadable_because"])))

    silent = sum(1 for r in scored if r["called_n"] == 0)
    print("\n  ABSTENTION: %d of %d windows called NOTHING (%.0f%%). Every "
          "number above is drawn from the other %d."
          % (silent, len(scored), 100.0 * silent / len(scored),
             len(scored) - silent))
    print("  A pooled mean over both window classes is NOT a scoreboard: a "
          "long-only rule flatters itself in an up window.")

    if args.json_out:
        report = {"chain": args.chain, "cadence": args.cadence,
                  "horizon": args.horizon, "train": args.train,
                  "test": args.test,
                  "heldout_default_bars": parser.get_default("test"),
                  "margin": args.hysteresis,
                  "window_classes": classes, "rows": rows}
        # The BOTH class is the one a reader quotes off the top of the file, so
        # it is the one whose guarded keys sit at the top level and get checked.
        report.update({k: v for k, v in classes["BOTH"].items()
                       if k in ("readable_label_floor", "heldout_window_bars",
                                "heldout_trough_labels",
                                "heldout_trough_base_rate",
                                "heldout_crest_labels", "heldout_crest_base_rate",
                                "readability", "buy_net_per_trade",
                                "sell_net_per_trade", "heldout_readable")})
        validate_report_readability(report)
        for name, block in classes.items():
            validate_report_readability({**block,
                                         "heldout_default_bars":
                                             report["heldout_default_bars"]})
        Path(args.json_out).write_text(json.dumps(report, indent=2),
                                       encoding="utf-8")
        print("\njson -> %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
