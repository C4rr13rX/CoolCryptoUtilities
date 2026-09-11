#!/usr/bin/env python3
"""The L1 abstention is a SUPPORT famine. Change ONE lever -- train length.

Item [89693706]. Measured by Cove pass 118 (bd609fb) on the train slice only,
at the same bands and margin the scoring path uses: over 102 base corpora at
3600s the MEDIAN corpus has 49 distinct L1 motifs across ~350 labelled train
bars and only TWO of them reach ``min_support=20``, covering 18% of the train
window. That is why 74-82% of windows call nothing at every horizon, and it is
why the whole 300s set produced two trades across four horizons. The rule is not
being selective about markets; it has nothing to be selective WITH.

THE LEVER, AND WHY THIS ONE. Three were available -- encoder granularity, train
length, ``min_support`` -- and the criterion allows exactly one, because changing
two tells you nothing about either. Train length is the one with a measurement
already behind it: Gale's census at 50cef36 found 0 supported groups at 539 train
samples becoming 23 at 4000, on the same encoder. The other two both BUY support
by making the thing being counted weaker (a coarser encoder, or a lower floor),
so a rise in supported groups would be arithmetic rather than evidence. More
train bars raise support without touching what a group IS.

Today's setting is ``--train 350``. The treatment is ``--train-long``. Everything
else is held: the shipped ``L1_HYSTERESIS_MARGIN`` (NOT the pass-120 solver --
that would be a second change), relative banding, ``min_support=20``, the same
encoder, and critically the SAME HELD-OUT TEST WINDOW, because ``heldout_edge``
anchors the test window to the tail of the corpus and grows the train window
backwards from it. So the two arms differ in their train window and in nothing
else, and they are scored on identical bars.

    python -X utf8 scripts/omen_l1_support_floor.py \
        --json-out data/brain_experiments/L1-SUPPORT-FLOOR-pass120-iris.json

NO NODE IS CONTACTED. Production runs :8090 and this never opens a socket.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.omen_brain import LOOKBACK_BARS  # noqa: E402
from scripts.omen_layer_probe import (  # noqa: E402
    build_layer_frames, heldout_edge, label_skew, load_bars,
)

#: Parsed corpora, TRIMMED to the tail each arm can reach, keyed by path.
#: These files run to 26,270 bars and every corpus is read once per arm per
#: horizon, so re-parsing them cost more wall clock than the encoder did. The
#: trim is what makes caching them affordable in memory, and it is safe because
#: both ``heldout_edge`` and ``train_census`` anchor their windows to the END of
#: the corpus and grow backwards -- the head they would drop is head this keeps
#: nothing of, beyond the LOOKBACK_BARS of history the frame builder needs
#: before the earliest train bar.
_BARS: Dict[str, List[Dict[str, Any]]] = {}


def bars_for(path: Path, keep: Optional[int] = None) -> List[Dict[str, Any]]:
    key = str(path)
    if key not in _BARS:
        bars = load_bars(path)
        _BARS[key] = bars[-keep:] if keep and len(bars) > keep else bars
    return _BARS[key]


from trading.omen_layers import L1_HYSTERESIS_MARGIN, relative_bands  # noqa: E402

KEY = "L1_cooccurrence"


def median_spacing(bars: Sequence[Mapping[str, Any]], sample: int = 200) -> float:
    stamps = [int(b["timestamp"]) for b in bars[:sample]]
    if len(stamps) < 3:
        return 0.0
    return float(statistics.median([b - a for a, b in zip(stamps, stamps[1:])]))


def _close(bar: Mapping[str, Any]) -> float:
    return float(bar["close"])


def window_mean_forward(bars: Sequence[Mapping[str, Any]], start: int,
                        stop: int, horizon: int) -> float:
    """Mean forward return over the test window, BEFORE cost.

    Before cost deliberately, as the criterion requires: the window's CLASS is a
    fact about the tape, and charging a round trip to it would make the up/down
    split a function of the fee model rather than of the market.
    """
    moves = []
    for i in range(start, min(stop, len(bars) - horizon)):
        here = _close(bars[i])
        if here:
            moves.append((_close(bars[i + horizon]) - here) / here)
    return sum(moves) / len(moves) if moves else 0.0


def train_census(bars: Sequence[Mapping[str, Any]], symbol: str, chain: str,
                 horizon: int, train: int, test: int,
                 min_support: int) -> Dict[str, Any]:
    """The support census on the TRAIN slice, at the same bands and margin.

    Deliberately the same window arithmetic as ``heldout_edge`` so this census
    describes the exact slice that arm fitted on -- a census over a slightly
    different window is a different number wearing the same name, and that is
    how two agents end up unable to reconcile a median.
    """
    train_stop = len(bars) - test - horizon - 1
    train_start = max(LOOKBACK_BARS, train_stop - train)
    seed = build_layer_frames(bars, symbol, chain, horizon, train_start,
                              train_stop, bands=None, margin=0.0)
    bands = relative_bands(seed) or None
    rows = [r for r in build_layer_frames(bars, symbol, chain, horizon,
                                          train_start, train_stop, bands=bands,
                                          margin=L1_HYSTERESIS_MARGIN)
            if r.get("_label")]
    if not rows:
        return {"labelled": 0, "vocabulary": 0, "supported": 0, "covered": 0.0}
    skew = label_skew(rows, KEY, min_support=min_support)
    return {
        "labelled": len(rows),
        "vocabulary": len({r[KEY] for r in rows}),
        "supported": len(skew.get("groups", [])),
        "covered": skew.get("covered", 0.0),
    }


def between_corpus_stderr(values: Sequence[float]) -> float:
    """Standard error of the corpus-level net, ACROSS CORPORA.

    NAMED FOR WHAT IT IS rather than left to be read as a per-trade standard
    error, because it is not one. ``heldout_edge`` returns a per-corpus mean and
    not the individual trade returns, so the dispersion available here is
    between corpora. That is arguably the more honest denominator anyway -- the
    trades inside one corpus are heavily correlated, so a per-trade SE would be
    optimistically small -- but it must be labelled, and this repo has already
    quoted a +0.10pp difference over 194 trades as if it were a result.
    """
    if len(values) < 2:
        return float("nan")
    return statistics.stdev(values) / math.sqrt(len(values))


def weighted(pairs: Sequence[Tuple[float, int]]) -> float:
    """Trade-weighted mean of per-corpus nets.

    Weighted rather than a flat average of corpus means: a corpus that called
    two trades must not count as much as one that called ninety, which is how a
    single thin corpus ends up setting the headline.
    """
    total = sum(n for _, n in pairs)
    if not total:
        return float("nan")
    return sum(value * n for value, n in pairs) / total


def discover(chain: str, cadence: float, min_bars: int,
             limit: Optional[int]) -> List[Tuple[str, Path]]:
    root = Path("data/historical_ohlcv") / chain
    out: List[Tuple[str, Path]] = []
    for path in sorted(root.glob("*.json")):
        if limit and len(out) >= limit:
            break
        try:
            bars = load_bars(path)
        except Exception:
            continue
        if len(bars) < min_bars:
            continue
        if abs(median_spacing(bars) - cadence) > 1.0:
            continue
        out.append((path.stem.split("_", 1)[-1], path))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", default="base")
    parser.add_argument("--cadence", type=float, default=3600.0)
    parser.add_argument("--horizon", type=int, action="append",
                        help="repeat; defaults to 3, 6 and 12 bars, which at "
                             "3600s is 3, 6 and 12 hours")
    parser.add_argument("--train", type=int, default=350,
                        help="TODAY'S setting; the control arm")
    parser.add_argument("--train-long", type=int, default=2000,
                        help="the treatment, and the ONLY thing that differs")
    parser.add_argument("--test", type=int, default=250)
    parser.add_argument("--min-lift", type=float, default=1.3)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--census-horizon", type=int, default=12,
                        help="the horizon the train support census is taken "
                             "at; matches Cove's pass-118 census so the before "
                             "number is comparable rather than merely similar")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    horizons = sorted(set(args.horizon or (3, 6, 12)))
    min_bars = LOOKBACK_BARS + args.train_long + args.test + max(horizons) + 60
    corpora = discover(args.chain, args.cadence, min_bars, args.limit)
    if not corpora:
        print("no corpus on %s at cadence %.0fs with >=%d bars"
              % (args.chain, args.cadence, min_bars))
        return 1

    print("L1 SUPPORT FLOOR -- ONE lever: train length %d -> %d. %d corpora on "
          "%s at %.0fs, test %d bars (the SAME held-out window in both arms), "
          "margin %.2f, min_support %d, min_lift %.2f\n"
          % (args.train, args.train_long, len(corpora), args.chain,
             args.cadence, args.test, L1_HYSTERESIS_MARGIN, args.min_support,
             args.min_lift))

    arms = {"short": args.train, "long": args.train_long}
    # The tail every arm can reach, plus the frame builder's own history.
    keep = min_bars

    # ---- CRITERION 2: the support census, before and after -----------------
    # Taken at ONE horizon and said so: the census is about how many bars a
    # group has, which the horizon barely moves, and spending it at every
    # horizon would buy four nearly identical medians with the compute this
    # pass needs for the edge table.
    census: Dict[str, List[Dict[str, Any]]] = {"short": [], "long": []}
    for symbol, path in corpora:
        bars = bars_for(path, keep)
        for name, train in arms.items():
            census[name].append(train_census(bars, symbol, args.chain,
                                             args.census_horizon, train,
                                             args.test, args.min_support))

    print("TRAIN SUPPORT CENSUS at horizon %d -- criterion 2" % args.census_horizon)
    print("%-7s %10s %10s %12s %14s" % ("arm", "med bars", "med vocab",
                                        "med supported", "med covered"))
    census_summary: Dict[str, Dict[str, float]] = {}
    for name in ("short", "long"):
        rows = census[name]
        summary = {
            "median_labelled": statistics.median([r["labelled"] for r in rows]),
            "median_vocabulary": statistics.median([r["vocabulary"] for r in rows]),
            "median_supported": statistics.median([r["supported"] for r in rows]),
            "median_covered": statistics.median([r["covered"] for r in rows]),
            "corpora_with_none": sum(1 for r in rows if r["supported"] == 0),
        }
        census_summary[name] = summary
        print("%-7s %10.0f %10.0f %12.1f %13.1f%%"
              % (name, summary["median_labelled"], summary["median_vocabulary"],
                 summary["median_supported"], 100 * summary["median_covered"]))
    for name in ("short", "long"):
        print("   %s: %d of %d corpora have NO supported group at all"
              % (name, census_summary[name]["corpora_with_none"], len(corpora)))
    print("")

    # ---- CRITERION 3 and 4: edge and abstention, per horizon, per class ----
    table: Dict[int, Dict[str, Any]] = {}
    for horizon in horizons:
        per_arm: Dict[str, Dict[str, Any]] = {}
        for name, train in arms.items():
            classes: Dict[str, Dict[str, List[Any]]] = {
                "UP": {"rule": [], "base": []}, "DOWN": {"rule": [], "base": []}}
            called = unspent = abstained = 0
            # per class: rule = [(corpus net, trades)], base = [(net, bars)]
            for symbol, path in corpora:
                bars = bars_for(path, keep)
                result = heldout_edge(bars, symbol, args.chain, horizon,
                                      train, args.test, args.min_lift,
                                      args.min_support, relative=True,
                                      margin=L1_HYSTERESIS_MARGIN, key=KEY)
                if result.get("error"):
                    continue
                if result.get("unspent"):
                    # NOT AN ABSTENTION AND NOT A NEGATIVE. The arm was never
                    # spent: no train group cleared the support floor, so the
                    # rule had nothing to fit and its "edge" would be the
                    # baseline with a minus sign. Counted separately, because
                    # folding it into abstention is exactly the error that made
                    # a support census read as a fact about the market.
                    unspent += 1
                    continue
                start, stop = result["test_window"]
                klass = ("UP" if window_mean_forward(bars, start, stop, horizon) > 0
                         else "DOWN")
                classes[klass]["base"].append(
                    (result["baseline_net"], result["test_n"]))
                if result["called_n"]:
                    classes[klass]["rule"].append(
                        (result["called_net"], result["called_n"]))
                    called += 1
                else:
                    # THE RULE HAD SOMETHING TO SAY AND SAID NOTHING. Distinct
                    # from `unspent` above, where it had nothing to say with.
                    abstained += 1
            per_arm[name] = {
                "called_corpora": called, "abstained_corpora": abstained,
                "unspent_corpora": unspent,
                "abstention_rate": abstained / max(1, called + abstained),
                "classes": {
                    k: {"n": sum(n for _, n in v["rule"]),
                        "corpora": len(v["rule"]),
                        "net": weighted(v["rule"]),
                        "stderr": between_corpus_stderr([x for x, _ in v["rule"]]),
                        "base_n": sum(n for _, n in v["base"]),
                        "base_net": weighted(v["base"])}
                    for k, v in classes.items()},
            }
        table[horizon] = per_arm

        print("HORIZON %d bars" % horizon)
        print("  %-6s %-5s %7s %11s %11s %11s %10s"
              % ("arm", "class", "trades", "net/trade", "stderr",
                 "baseline", "abstain"))
        for name in ("short", "long"):
            arm = per_arm[name]
            for klass in ("UP", "DOWN"):
                cell = arm["classes"][klass]
                print("  %-6s %-5s %7d %10.4f%% %10.4f%% %10.4f%% %9.1f%%"
                      % (name, klass, cell["n"], 100 * cell["net"],
                         100 * cell["stderr"], 100 * cell["base_net"],
                         100 * arm["abstention_rate"]))
            print("         (%d corpora called, %d abstained, %d UNSPENT -- no "
                  "train group cleared n>=%d)"
                  % (arm["called_corpora"], arm["abstained_corpora"],
                     arm["unspent_corpora"], args.min_support))
        print("")

    # ---- THE VERDICT. An arm that wins in ONE class only is a FAIL. --------
    print("VERDICT -- an arm that beats its baseline in only ONE window class "
          "is a FAIL, because a long-only rule flatters itself in an up window")
    verdicts: Dict[int, str] = {}
    for horizon in horizons:
        long_arm = table[horizon]["long"]
        wins = []
        for klass in ("UP", "DOWN"):
            cell = long_arm["classes"][klass]
            beat = (cell["n"] > 0 and not math.isnan(cell["net"])
                    and cell["net"] > cell["base_net"])
            wins.append(beat)
        verdict = ("PASS" if all(wins) else
                   "FAIL -- wins in one class only" if any(wins) else
                   "FAIL -- beats baseline in neither class")
        verdicts[horizon] = verdict
        print("  horizon %2d: long arm %s" % (horizon, verdict))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "chain": args.chain, "cadence": args.cadence,
            "corpora": len(corpora), "train": args.train,
            "train_long": args.train_long, "test": args.test,
            "margin": L1_HYSTERESIS_MARGIN, "min_support": args.min_support,
            "min_lift": args.min_lift, "census": census_summary,
            "horizons": {str(h): table[h] for h in horizons},
            "verdicts": {str(h): verdicts[h] for h in horizons},
        }, indent=2), encoding="utf-8")
        print("\njson -> %s" % args.json_out)

    return 0 if any(v == "PASS" for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
