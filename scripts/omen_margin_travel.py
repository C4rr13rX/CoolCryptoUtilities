#!/usr/bin/env python3
"""Does L1 stickiness TRAVEL? Solve the margin per corpus, then check L2 held out.

Item [746c2ece]. ``L1_HYSTERESIS_MARGIN`` is a fraction of each band's own
WIDTH, and the widths are terciles fitted per stream per corpus -- so the same
0.50 bought 37.6% alphabet turnover on AERO-USDC and 62.3% on ARB-WETH, and L2
cleared the 0.30 identifier ceiling on the first and failed it on the second for
exactly that reason ([41ca68dc], pass 116, 536037e). L1 itself abstracts on
both, so the co-occurrence layer is not what fails to travel; the stickiness is.

WHAT THIS MEASURES, and every one of the four things is a criterion:

  1. THE MARGIN AS A SOLVED QUANTITY. ``solve_hysteresis_margin`` bisects the
     real encoder on the TRAIN frames until the change rate lands on a target,
     so every pair arrives at L2 with comparable turnover. The fixed-0.50 arm is
     measured beside it on the same build, so the spread is one measurement.

  2. HELD OUT, NOT IN SAMPLE. Pass 111's L2 numbers were all in-sample and the
     deciding fact under this thread is a ZERO: with plain relative banding L2
     has NO label-skew group reaching n=20 in either window. Bands, margin and
     nothing else are fitted on train; the distinctness and the skew are read
     on a test window the fit never saw.

  3. AN UP WINDOW AND A DOWN WINDOW PER PAIR. A long-only rule flatters itself
     in an up window and that error has produced a fake +0.9067% here already.

  4. WINDOWS SELECTED ON LABEL INCIDENCE, NOT ONLY DIRECTION. The ARB-WETH DOWN
     window measured in pass 116 has a 0.0% trough base rate -- the omen label
     never fires -- so half of any table built from it is unreadable. A window
     whose base rate is zero is EXCLUDED and named as excluded.

NO NODE IS CONTACTED. This is encoder arithmetic over stored bars, so it is not
blocked by the RAM floor [beec23cc]. Production runs :8090 and this never opens
a socket.

    python -X utf8 scripts/omen_margin_travel.py --pairs 6 \
        --json-out data/brain_experiments/L1-MARGIN-TRAVEL-pass120-iris.json

Exits nonzero when the solve does not close the turnover spread, so it gates the
claim rather than merely printing it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.omen_brain import (  # noqa: E402
    LOOKBACK_BARS, OMEN_TROUGH, build_collections, label_omen,
    measure_bar_seconds,
)
from scripts.omen_layer_probe import label_skew, load_bars  # noqa: E402
from trading.omen_layers import (  # noqa: E402
    IDENTIFIER_CEILING, L1_HYSTERESIS_MARGIN, L1_STREAMS,
    L1_TARGET_CHANGE_RATE, L2_TRANSITION_STEPS, l1_change_rate,
    layer_distinctness, relative_bands, solve_hysteresis_margin,
    sticky_motifs, transition_motif,
)

#: Keys the table reports. L1 is here so the COST of stickiness is visible:
#: a margin that clears the ceiling by coarsening L1 into near-constancy has
#: bought the guard and sold the layer.
KEYS = ("L1_cooccurrence", "L2_transitions")


def _close(bar: Mapping[str, Any]) -> float:
    return float(bar["close"])


def median_spacing(bars: Sequence[Mapping[str, Any]], sample: int = 200) -> float:
    """Median seconds between bars. The MEDIAN because one multi-day gap would
    drag a mean into a different cadence class, and cadence is what makes
    ``--horizon`` mean the same number of minutes on every corpus."""
    stamps = [int(b["timestamp"]) for b in bars[:sample]]
    if len(stamps) < 3:
        return 0.0
    return float(statistics.median([b - a for a, b in zip(stamps, stamps[1:])]))


def window_direction(bars: Sequence[Mapping[str, Any]], start: int, stop: int,
                     horizon: int) -> float:
    """Mean forward return over the window, BEFORE cost.

    Before cost deliberately: the window's CLASS is a fact about the tape, and
    charging a round trip to it would classify a mildly-up window as down and
    make the up/down split a function of the fee model rather than of the
    market.
    """
    moves = []
    for i in range(start, min(stop, len(bars) - horizon)):
        here = _close(bars[i])
        if here:
            moves.append((_close(bars[i + horizon]) - here) / here)
    return sum(moves) / len(moves) if moves else 0.0


def base_trough_rate(bars: Sequence[Mapping[str, Any]], start: int, stop: int,
                     horizon: int) -> float:
    """How often the omen label fires in the window. CRITERION 4's gate.

    Read from ``label_omen`` rather than re-derived, so "the label fires" means
    exactly what the scoring path means by it. A window at 0.0 is unreadable --
    every group's trough rate is zero and every lift is 0/0 -- so it is excluded
    upstream instead of producing a table of zeroes somebody later quotes.
    """
    labels = [label_omen(bars, i, horizon_bars=horizon)
              for i in range(start, min(stop, len(bars) - horizon))]
    firing = [x for x in labels if x]
    if not firing:
        return 0.0
    return sum(1 for x in firing if x == OMEN_TROUGH) / len(firing)


def pick_windows(bars: Sequence[Mapping[str, Any]], horizon: int, train: int,
                 purge: int, test: int) -> Dict[str, Optional[Dict[str, Any]]]:
    """The most-UP and most-DOWN held-out window this corpus can offer.

    Both are chosen from the SAME pair so the up/down comparison is not also a
    comparison of two instruments, and each carries its own ``train`` bars
    immediately before it with a ``purge`` gap -- the gap matters because the
    label at bar i reads ``horizon`` bars forward, so the last ``horizon`` train
    labels would otherwise overlap the test window.

    Selection is on mean forward return AND on label incidence: a candidate
    whose trough base rate is 0.0 is not eligible, however cleanly it is
    directional, because nothing can be read off it.
    """
    first = LOOKBACK_BARS + train + purge
    span = len(bars) - horizon
    candidates: List[Dict[str, Any]] = []
    step = max(1, test // 2)
    for t0 in range(first, span - test, step):
        rate = base_trough_rate(bars, t0, t0 + test, horizon)
        candidates.append({
            "test_start": t0, "test_stop": t0 + test,
            "train_start": t0 - purge - train, "train_stop": t0 - purge,
            "mean_forward": window_direction(bars, t0, t0 + test, horizon),
            "base_trough": rate,
        })
    if not candidates:
        return {"UP": None, "DOWN": None, "excluded": []}

    excluded = [c for c in candidates if c["base_trough"] <= 0.0]
    live = [c for c in candidates if c["base_trough"] > 0.0]
    if not live:
        return {"UP": None, "DOWN": None, "excluded": excluded}

    up = max(live, key=lambda c: c["mean_forward"])
    down = min(live, key=lambda c: c["mean_forward"])
    return {
        "UP": up if up["mean_forward"] > 0 else None,
        "DOWN": down if down["mean_forward"] < 0 else None,
        "excluded": excluded,
    }


def _frames(bars: Sequence[Mapping[str, Any]], start: int, stop: int,
            symbol: str, chain: str, horizon: int,
            cadence: int) -> Tuple[List[Dict[str, str]], List[int]]:
    """L0 frame sets for ``[start, stop)``, with the bar index each came from.

    A bar whose frames cannot be built is DROPPED rather than held as a hole,
    and the index list is what lets the caller line the surviving frames back up
    with their labels -- which a plain list cannot do once anything is missing.
    """
    out: List[Dict[str, str]] = []
    indices: List[int] = []
    for index in range(start, stop):
        try:
            out.append(build_collections(bars, index, horizon_bars=horizon,
                                         bar_seconds=cadence,
                                         symbol=symbol, chain=chain))
        except (ValueError, IndexError):
            continue
        indices.append(index)
    return out, indices


def measure_window(bars: Sequence[Mapping[str, Any]], symbol: str, chain: str,
                   horizon: int, window: Mapping[str, Any], margins: Mapping[str, Any],
                   l2_window: int, steps: int,
                   min_support: int) -> Dict[str, Any]:
    """Encode train and test ONCE, then read every margin arm off that build.

    THE BUILD IS SHARED ON PURPOSE. ``build_collections`` is the expensive call
    and it does not depend on the margin at all, so rebuilding per arm would
    cost minutes and -- worse -- would make the fixed-0.50 arm and the solved
    arm two measurements instead of one. ``sticky_motifs`` is pure arithmetic
    over the already-built frames, so every arm here is genuinely the same
    corpus seen through a different amount of stickiness.

    THE MOTIF STREAM SPANS TRAIN AND TEST IN ONE CALL because hysteresis is a
    fact about a stream: cutting it at the boundary would reset every held band
    and give the first test bars a stickiness they would not have live. No test
    information travels backwards -- the bands and the margin are both solved on
    the train frames alone, and only the held BAND STATE crosses, which is
    exactly what crosses in production.
    """
    cadence = measure_bar_seconds(bars)
    train_frames, _ = _frames(bars, window["train_start"], window["train_stop"],
                              symbol, chain, horizon, cadence)
    test_frames, test_index = _frames(bars, window["test_start"],
                                      window["test_stop"], symbol, chain,
                                      horizon, cadence)
    if len(train_frames) < 50 or len(test_frames) < 50:
        return {"error": "too few buildable bars (train %d, test %d)"
                         % (len(train_frames), len(test_frames))}

    bands = relative_bands(train_frames, streams=L1_STREAMS)
    labels = [label_omen(bars, i, horizon_bars=horizon) or ""
              for i in test_index]

    arms: Dict[str, Any] = {}
    for name, margin in margins.items():
        motifs = sticky_motifs(list(train_frames) + list(test_frames),
                               bands, margin)
        held = motifs[len(train_frames):]
        rows = [{
            "L1_cooccurrence": held[i],
            "L2_transitions": transition_motif(
                held[max(0, i - l2_window + 1):i + 1], steps=steps),
            "_label": labels[i],
        } for i in range(len(held))]
        dist = layer_distinctness(rows, keys=list(KEYS))
        skew = {k: label_skew(rows, k, min_support=min_support) for k in KEYS}
        arms[name] = {
            "margin": margin,
            "train_change_rate": l1_change_rate(motifs[:len(train_frames)]),
            "test_change_rate": l1_change_rate(held),
            "distinct": {k: dist.get(k, 0.0) for k in KEYS},
            "vocab": {k: len({r[k] for r in rows}) for k in KEYS},
            "supported": {k: len(skew[k].get("groups", [])) for k in KEYS},
            "covered": {k: skew[k].get("covered", 0.0) for k in KEYS},
            "best_lift": {k: (skew[k]["groups"][0]["lift"]
                              if skew[k].get("groups") else 0.0) for k in KEYS},
            "l2_identifier": dist.get("L2_transitions", 1.0) > IDENTIFIER_CEILING,
        }
    return {
        "samples": len(test_frames),
        "train_samples": len(train_frames),
        "mean_forward": window["mean_forward"],
        "base_trough": window["base_trough"],
        "arms": arms,
    }


def discover(chains: Sequence[str], cadence: float, per_chain: int,
             min_bars: int) -> List[Tuple[str, str, Path]]:
    """Corpora with enough bars at ONE cadence, spread across chains.

    Cadence is filtered rather than assumed: ``data/historical_ohlcv`` spans
    166s to 345600s bars, so ``--horizon 12`` asks about 33 minutes on one file
    and 48 days on another. A sweep that mixes cadences is not one experiment.
    """
    root = Path("data/historical_ohlcv")
    found: List[Tuple[str, str, Path]] = []
    for chain in chains:
        taken = 0
        seen: set = set()
        for path in sorted((root / chain).glob("*.json")):
            if taken >= per_chain:
                break
            symbol = path.stem.split("_", 1)[-1]
            if symbol in seen:
                continue
            try:
                bars = load_bars(path)
            except Exception:
                continue
            if len(bars) < min_bars:
                continue
            if abs(median_spacing(bars) - cadence) > 1.0:
                continue
            seen.add(symbol)
            found.append((symbol, chain, path))
            taken += 1
    return found


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", action="append", default=None,
                        help="repeat; defaults to base and arbitrum, which is "
                             "the two-chain minimum the criterion asks for")
    parser.add_argument("--per-chain", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=6,
                        help="stop after this many corpora in total")
    parser.add_argument("--cadence", type=float, default=3600.0)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train", type=int, default=400)
    parser.add_argument("--purge", type=int, default=12)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--l2-window", type=int, default=12)
    parser.add_argument("--steps", type=int, default=L2_TRANSITION_STEPS)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--target-rate", type=float,
                        default=L1_TARGET_CHANGE_RATE)
    parser.add_argument("--fixed-margin", type=float,
                        default=L1_HYSTERESIS_MARGIN)
    parser.add_argument("--json-out", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    chains = args.chain or ["base", "arbitrum"]
    min_bars = LOOKBACK_BARS + args.train + args.purge + args.test + args.horizon + 60

    corpora = discover(chains, args.cadence, args.per_chain, min_bars)[: args.pairs]
    if not corpora:
        print("no corpus at cadence %.0fs with >=%d bars" % (args.cadence, min_bars))
        return 1

    print("L1 MARGIN TRAVEL -- fixed %.2f vs solved-to-%.3f, held out, "
          "cadence %.0fs, horizon %d bars, train %d / purge %d / test %d\n"
          % (args.fixed_margin, args.target_rate, args.cadence, args.horizon,
             args.train, args.purge, args.test))

    report: List[Dict[str, Any]] = []
    excluded_notes: List[str] = []

    for symbol, chain, path in corpora:
        bars = load_bars(path)
        picked = pick_windows(bars, args.horizon, args.train, args.purge,
                              args.test)
        entry: Dict[str, Any] = {"symbol": symbol, "chain": chain,
                                 "corpus": str(path), "windows": {}}
        if picked["excluded"]:
            excluded_notes.append(
                "%s/%s: %d candidate window(s) EXCLUDED -- trough base rate "
                "0.0%%, the omen label never fires so nothing is readable"
                % (chain, symbol, len(picked["excluded"])))
        for klass in ("UP", "DOWN"):
            window = picked[klass]
            if window is None:
                excluded_notes.append(
                    "%s/%s %s: no eligible window (either no window of that "
                    "direction, or every one had a 0.0%% base rate)"
                    % (chain, symbol, klass))
                continue

            # SOLVE ON TRAIN ONLY. The margin is a hyperparameter; solving it on
            # the held-out window fits it to the test distribution, which is the
            # same error as refitting relative_bands there.
            cadence = measure_bar_seconds(bars)
            train_frames, _ = _frames(bars, window["train_start"],
                                      window["train_stop"], symbol, chain,
                                      args.horizon, cadence)
            bands = solved = None
            if len(train_frames) >= 50:
                bands = relative_bands(train_frames, streams=L1_STREAMS)
                solved = solve_hysteresis_margin(train_frames, bands,
                                                 args.target_rate)
            if solved is None:
                excluded_notes.append("%s/%s %s: too few buildable train bars"
                                      % (chain, symbol, klass))
                continue

            result = measure_window(
                bars, symbol, chain, args.horizon, window,
                {"fixed": args.fixed_margin, "solved": solved["margin"]},
                args.l2_window, args.steps, args.min_support)
            if "error" in result:
                excluded_notes.append("%s/%s %s: %s"
                                      % (chain, symbol, klass, result["error"]))
                continue
            result["solve"] = solved
            entry["windows"][klass] = result

            print("%-12s %-9s %s  n=%d  mean fwd %+.4f%%  trough base %.1f%%"
                  % (symbol, chain, klass, result["samples"],
                     100 * result["mean_forward"], 100 * result["base_trough"]))
            if not solved["reached"]:
                print("     SOLVE DID NOT REACH the target: best change rate "
                      "%.1f%% at the search ceiling %.2f"
                      % (100 * solved["achieved"], solved["margin"]))
            for name in ("fixed", "solved"):
                arm = result["arms"][name]
                l2 = arm["distinct"]["L2_transitions"]
                print("     %-6s margin=%.4f  L1 change train %.1f%% / test "
                      "%.1f%%   L1 distinct %.4f (vocab %d)   L2 distinct "
                      "%.4f  %s   L2 groups(n>=%d)=%d covering %.1f%% "
                      "best lift %.2fx"
                      % (name, arm["margin"],
                         100 * arm["train_change_rate"],
                         100 * arm["test_change_rate"],
                         arm["distinct"]["L1_cooccurrence"],
                         arm["vocab"]["L1_cooccurrence"], l2,
                         "IDENTIFIER" if arm["l2_identifier"] else "passes",
                         args.min_support, arm["supported"]["L2_transitions"],
                         100 * arm["covered"]["L2_transitions"],
                         arm["best_lift"]["L2_transitions"]))
            print("")
        report.append(entry)

    # THE HEADLINE NUMBER IS A SPREAD, not a mean. What failed to travel is the
    # DISPERSION of turnover across pairs -- a mean change rate would have looked
    # fine at 0.50 while AERO sat at 37.6% and ARB-WETH at 62.3%.
    def rates(name: str) -> List[float]:
        return [w["arms"][name]["train_change_rate"]
                for e in report for w in e["windows"].values()]

    fixed_rates, solved_rates = rates("fixed"), rates("solved")
    if not fixed_rates:
        print("no window survived selection -- nothing to compare")
        return 1

    def spread(values: List[float]) -> float:
        return max(values) - min(values)

    print("TURNOVER SPREAD ACROSS PAIRS (train window, the thing that failed to travel)")
    print("  fixed  margin %.2f : %.1f%% .. %.1f%%   spread %.1f pp"
          % (args.fixed_margin, 100 * min(fixed_rates), 100 * max(fixed_rates),
             100 * spread(fixed_rates)))
    print("  solved to %.3f     : %.1f%% .. %.1f%%   spread %.1f pp"
          % (args.target_rate, 100 * min(solved_rates),
             100 * max(solved_rates), 100 * spread(solved_rates)))

    def l2_verdicts(name: str) -> Tuple[int, int]:
        ok = bad = 0
        for e in report:
            for w in e["windows"].values():
                if w["arms"][name]["l2_identifier"]:
                    bad += 1
                else:
                    ok += 1
        return ok, bad

    print("\nL2 HELD-OUT IDENTIFIER VERDICT (ceiling %.2f), windows passing / "
          "failing" % IDENTIFIER_CEILING)
    for name in ("fixed", "solved"):
        ok, bad = l2_verdicts(name)
        print("  %-6s %d pass, %d IDENTIFIER" % (name, ok, bad))

    def supported_total(name: str) -> int:
        return sum(w["arms"][name]["supported"]["L2_transitions"]
                   for e in report for w in e["windows"].values())

    print("\nSUPPORTED L2 GROUPS (n>=%d) SUMMED OVER WINDOWS -- the zero this "
          "whole thread turns on" % args.min_support)
    for name in ("fixed", "solved"):
        print("  %-6s %d" % (name, supported_total(name)))

    if excluded_notes:
        print("\nEXCLUDED, AND NAMED -- criterion 4")
        for note in excluded_notes:
            print("  " + note)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "ceiling": IDENTIFIER_CEILING, "target_rate": args.target_rate,
            "fixed_margin": args.fixed_margin, "cadence": args.cadence,
            "horizon": args.horizon, "train": args.train, "test": args.test,
            "report": report, "excluded": excluded_notes,
        }, indent=2), encoding="utf-8")
        print("\njson -> %s" % args.json_out)

    # THE GATE. The claim under test is that solving the margin CLOSES the
    # turnover spread; if it does not, the item's premise is wrong and this must
    # say so with a nonzero exit rather than printing a table nobody checks.
    if spread(solved_rates) >= spread(fixed_rates):
        print("\nVERDICT: solving did NOT close the turnover spread "
              "(%.1f pp solved vs %.1f pp fixed). The premise is not supported."
              % (100 * spread(solved_rates), 100 * spread(fixed_rates)))
        return 1
    print("\nVERDICT: the solved margin closes the turnover spread from %.1f pp "
          "to %.1f pp. Whether that buys an L2 that is not an identifier is the "
          "table above, and it is reported per window rather than pooled."
          % (100 * spread(fixed_rates), 100 * spread(solved_rates)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
