#!/usr/bin/env python3
"""Does a motif layer ABSTRACT, or is it a lossy copy? Measure before training.

THE FALSIFICATION TEST, and it is deliberately cheap: it needs no node, no
fabric and no held-out window. ``trading/omen_layers`` states the rule that
governs every layer -- a higher layer earns its place only when its vocabulary
is SMALLER than its input's, because that shrinkage IS the abstraction. A layer
that comes in at or above its input's distinctness has abstracted nothing; it
is a lossy copy costing a consolidation and a query per sample, and it should
be CUT before anyone spends a training run on it.

Run this BEFORE wiring a layer into the train/predict paths. Exits nonzero when
the layer fails to abstract, so it can gate a pass rather than merely inform it.

    python -X utf8 scripts/omen_layer_probe.py \
        --corpus data/brain_experiments/p108_aero_up.json --horizon 12

WHY EXIT CODES RATHER THAN A PRINTED OPINION. The sibling instrument
``scripts/omen_query_path_probe.py`` exits nonzero when the query path does not
fire the pool it claims to fire, and that criterion caught a pass-108 result
that had already been believed. An abstraction claim deserves the same
treatment: a number printed beside a PASS/FAIL that a human has to interpret is
a number that gets quoted without its verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.omen_brain import (  # noqa: E402
    COLLECTIONS, LOOKBACK_BARS, OMEN_CREST, OMEN_TROUGH, ROUND_TRIP_COST,
    build_collections, collection_distinctness, label_omen,
    measure_bar_seconds,
)
from trading.omen_layers import (  # noqa: E402
    IDENTIFIER_CEILING, L1_HYSTERESIS_MARGIN, L1_STREAMS,
    L2_TRANSITION_STEPS, MOTIF_SEQUENCE_STEPS, cooccurrence_motif,
    layer_distinctness, relative_bands, sequence_motif, sticky_motifs,
    transition_motif,
)

#: The L0 streams a motif is built FROM. ``instrument`` and ``horizon`` are
#: excluded by ``L1_STREAMS`` already -- they name which symbol and how far
#: ahead, and are constant on a single-symbol corpus, so including them in the
#: comparison baseline would flatter L1 against two streams at 1/n.
_BASELINE_STREAMS = tuple(L1_STREAMS)

#: The SHIPPED L2 column, and the rejected control kept beside it.
#:
#: ``transition_motif`` won [fa75fa1a] (0.2633 DOWN / 0.2000 UP over a sticky
#: L1) and ``sequence_motif`` lost it (0.5317 / 0.4083 at the same banding).
#: Both are computed here on purpose: a probe that dropped the loser would make
#: the winner's margin unreadable, and a probe that computed ONLY the loser --
#: which is what this file did until [a4ba2028] -- silently scores the rejected
#: scheme in any arm a reader believes is measuring L2.
L2_WINNER = "L2_transitions"
L2_CONTROL = "L2_sequence"

#: How many preceding bars the change-keyed L2 may look back over.
#:
#: TWELVE, AND IT IS NOT A FREE PARAMETER HERE. ``omen_layers.L2_CHURN_CUTS``
#: is measured at a window of 12 and its docstring states plainly that the cut
#: does not travel: the churn rate can only take values k/(n-1), so at a window
#: of 8 "one change" lands above the 0.15 cut, the held band collapses and the
#: frame goes over the ceiling. Changing this without re-sweeping that constant
#: would score a churn symbol whose bands mean something other than what they
#: were fitted to mean. Pinned against ``omen_l2_scheme_probe``'s own default by
#: tests/test_the_l2_probe_scores_the_shipped_scheme_not_the_rejected_one.py.
L2_TRANSITION_WINDOW = 12


def load_bars(path: Path) -> List[Dict[str, Any]]:
    """Same loader as omen_experiment, so the two see an identical corpus."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    bars = raw if isinstance(raw, list) else raw.get("bars") or raw.get("data") or []
    bars = [b for b in bars if b.get("close")]
    bars.sort(key=lambda b: int(b["timestamp"]))
    return bars


def build_layer_frames(bars: Sequence[Mapping[str, Any]], symbol: str,
                       chain: str, horizon: int, start: int, stop: int,
                       bands: Optional[Mapping[str, Any]] = None,
                       margin: float = 0.0
                       ) -> List[Dict[str, str]]:
    """L0 frames plus their L1 motif and L2 motif-path, per bar.

    The L2 path needs the L1 motifs of the PRECEDING bars, so the motifs are
    accumulated in bar order and the path is read off the running list. A bar
    whose L0 frames cannot be built breaks the chain, so its motif is recorded
    as missing rather than silently skipped -- an L2 path that quietly closed
    over a gap would claim an adjacency the corpus does not have.

    ``bands`` are per-stream cut points from ``omen_layers.relative_bands``.
    Passing None reproduces the ORIGINAL sign-banded encoder exactly, which is
    what makes a both-bandings comparison on one corpus possible: the encoder
    fix and the market cannot both move between the two arms if the two arms
    differ only in this argument.
    """
    out: List[Dict[str, str]] = []
    motif_history: List[str] = []

    # HYSTERESIS IS A FACT ABOUT A SEQUENCE, not about one bar, so the whole
    # window is encoded in ONE call rather than bar by bar. margin=0.0 is
    # byte-identical to the old per-bar cooccurrence_motif call, with and
    # without bands, which is what lets the two margins be compared on one
    # fabric -- pinned by tests/test_hysteresis_margin_zero_is_byte_identical.
    #
    # A HOLE DOES NOT RESET THE HELD BAND: the market did not stop, only our
    # view of it did. The hole is still recorded as a break in the L2 path, so
    # no adjacency is claimed that the corpus does not have.
    built: List[Any] = []              # (index, frames) for every buildable bar
    order: List[Optional[int]] = []    # position in `built`, or None for a hole
    cadence = measure_bar_seconds(bars)
    for index in range(max(start, LOOKBACK_BARS), stop):
        try:
            frames = build_collections(bars, index, horizon_bars=horizon,
                                       bar_seconds=cadence,
                                       symbol=symbol, chain=chain)
        except (ValueError, IndexError):
            order.append(None)
            continue
        order.append(len(built))
        built.append((index, frames))

    motifs = sticky_motifs([f for _, f in built], bands, margin)

    for slot in order:
        if slot is None:
            motif_history.append("")      # a hole, not an adjacency
            continue
        index, frames = built[slot]
        motif = motifs[slot]
        motif_history.append(motif)
        fixed = [m for m in motif_history[-MOTIF_SEQUENCE_STEPS:] if m]
        recent = [m for m in motif_history[-L2_TRANSITION_WINDOW:] if m]
        row = dict(frames)
        row["L1_cooccurrence"] = motif
        # THE WINNER AND THE REJECTED CONTROL, SIDE BY SIDE IN ONE ROW.
        # The two read DIFFERENT windows and that is the difference between
        # them rather than a detail: a fixed-length path can only afford
        # MOTIF_SEQUENCE_STEPS bars because it samples one symbol per bar,
        # while a change-keyed scheme can afford L2_TRANSITION_WINDOW because a
        # regime that holds for forty bars costs it one symbol. Handing them
        # the same window would hobble the winner to flatter the loser.
        row["L2_sequence"] = sequence_motif(fixed)
        row["L2_transitions"] = transition_motif(recent,
                                                 steps=L2_TRANSITION_STEPS)
        # The true label rides along so the layer can be asked the only
        # question that matters: does the motif carry information ABOUT THE
        # OUTCOME, or is it merely a tidy compression of the inputs?
        row["_label"] = label_omen(bars, index, horizon_bars=horizon) or ""
        row["_index"] = index
        # The realised forward return over the SAME horizon the label uses.
        # Guarded rather than assumed: the last horizon bars of a corpus have
        # no future, and reading past the end would silently wrap the last
        # price into every one of them.
        if index + horizon < len(bars):
            entry = float(bars[index]["close"])
            row["_forward"] = ((float(bars[index + horizon]["close"]) - entry)
                               / entry) if entry else 0.0
        else:
            row["_forward"] = 0.0
        out.append(row)
    return out


def label_skew(rows: Sequence[Mapping[str, str]], key: str,
               min_support: int = 20,
               target: str = OMEN_TROUGH) -> Dict[str, Any]:
    """Does knowing the motif change what you expect the label to be?

    DISTINCTNESS IS ONLY HALF THE QUESTION, and this repo's dilution law is
    where the other half hides. The law says a low-distinctness stream votes
    for the label *distribution* over every sample that shares its frame -- so
    a coarse stream HURTS a query when that distribution is flat, and helps
    when it is skewed. An abstraction layer is low-distinctness BY DESIGN,
    which puts it under the 0.20 query floor automatically; judging it by
    distinctness alone would cut every layer before it was measured.

    So measure the skew directly. ``lift`` is the trough rate within a motif
    divided by the corpus trough rate: 1.0 means the motif tells you nothing
    about buying low, and the operator's scoreboard leads on trough precision.
    """
    labelled = [r for r in rows if r.get("_label")]
    total = len(labelled)
    if not total:
        return {"total": 0, "groups": []}
    base_trough = sum(1 for r in labelled if r["_label"] == target) / total

    groups: Dict[str, List[str]] = {}
    for row in labelled:
        groups.setdefault(row[key], []).append(row["_label"])

    out = []
    for frame, labels in groups.items():
        n = len(labels)
        if n < min_support:
            continue
        trough = sum(1 for x in labels if x == target) / n
        counts = Counter(labels)
        out.append({
            "frame": frame, "n": n, "share": n / total,
            "trough_rate": trough,
            "lift": (trough / base_trough) if base_trough else 0.0,
            "top_label": counts.most_common(1)[0][0],
            "purity": counts.most_common(1)[0][1] / n,
        })
    out.sort(key=lambda g: -g["lift"])
    return {"total": total, "base_trough": base_trough,
            "covered": sum(g["n"] for g in out) / total, "groups": out}


def heldout_edge(bars: Sequence[Mapping[str, Any]], symbol: str, chain: str,
                 horizon: int, train: int, test: int,
                 min_lift: float, min_support: int,
                 relative: bool = False,
                 margin: float = 0.0,
                 key: str = "L1_cooccurrence") -> Dict[str, Any]:
    """Does the motif in ``key`` carry BUY-LOW information out of sample?

    THE REASON THIS EXISTS BEFORE ANY NODE RUN. A fabric cannot extract from a
    stream what the stream does not contain. Fitting the motif -> trough map on
    the train window and applying it, frozen, to a held-out window asks whether
    the motif generalises AT ALL -- and it costs seconds rather than a training
    run against a node that has no pool for this layer yet.

    A positive here does NOT mean the fabric will find it; a negative means
    there is nothing for the fabric to find, which is the cheaper answer to
    get wrong. Scored on the operator's order: per-trade net against
    buy-every-bar first, trough precision second, exact accuracy nowhere --
    a run can raise 5-class accuracy by calling murk better and place zero
    better trades.

    ``key`` names the layer being scored. It defaulted to L1 and was HARD-WIRED
    to it until [a4ba2028]; any L2 arm run before that was scoring whatever
    column the caller believed it was scoring while the code read L1. Passing
    ``L2_WINNER`` here scores the shipped transition scheme and ``L2_CONTROL``
    the rejected fixed-length one.

    AN ARM OVER A LAYER WITH NO SUPPORTED TRAIN GROUP IS NOT SPENT. A layer
    whose groups are all below ``min_support`` in the train window has nothing
    to fit a motif->trough map ON, and scoring it anyway produces an abstention
    that reads as a negative result. That case returns ``unspent`` with the
    group census rather than a number.
    """
    train_stop = len(bars) - test - horizon - 1
    train_start = max(LOOKBACK_BARS, train_stop - train)
    test_start = train_stop + horizon          # purge: no train future overlaps a test bar
    test_stop = len(bars) - horizon - 1

    def rows_for(start: int, stop: int,
                 bands: Optional[Mapping[str, Any]] = None
                 ) -> List[Dict[str, Any]]:
        out = build_layer_frames(bars, symbol, chain, horizon, start, stop,
                                 bands=bands, margin=margin)
        keep = []
        for row in out:
            idx = row.get("_index")
            if idx is None or not row.get("_label"):
                continue
            keep.append(row)
        return keep

    # THE CUT POINTS ARE FITTED ON THE TRAIN WINDOW AND NOWHERE ELSE. Fitting
    # terciles on the full corpus would put the test window's own distribution
    # inside the frame the test window is scored on -- the same class of leak
    # as fitting the motif->trough map in-sample and reading its lift as an
    # edge, and it would be invisible in the output. The seed pass below is
    # built with bands=None purely to HAVE scores to take terciles of; nothing
    # is scored on it.
    bands: Optional[Mapping[str, Any]] = None
    if relative:
        seed = rows_for(train_start, train_stop)
        bands = relative_bands(seed) or None

    tr = rows_for(train_start, train_stop, bands)
    te = rows_for(test_start, test_stop, bands)
    if not tr or not te:
        return {"error": "empty train or test window"}

    # FIT ON TRAIN ONLY. Any use of a test-window label here would be the
    # leak that makes every one of these numbers meaningless.
    fit = label_skew(tr, key, min_support=min_support)
    base = fit.get("base_trough", 0.0)
    supported = fit.get("groups", [])

    # DO NOT SPEND AN ARM ON A LAYER WITH NOTHING TO FIT. With no train group
    # at or above the support floor, ``buyable`` is empty by construction, the
    # rule abstains on every test bar, and the printed "edge" is the baseline
    # with a minus sign -- a fact about the support census reported as a fact
    # about the market.
    if not supported:
        vocabulary = len({r[key] for r in tr})
        biggest = max(Counter(r[key] for r in tr).values()) if tr else 0
        return {
            "key": key, "banding": "relative" if relative else "sign",
            "margin": margin,
            "train_window": [train_start, train_stop], "train_n": len(tr),
            "test_window": [test_start, test_stop], "test_n": len(te),
            "train_supported_groups": 0,
            "train_vocabulary": vocabulary,
            "train_largest_group": biggest,
            "unspent": True,
            "unspent_reason": (
                f"{key} has NO group with n>={min_support} in the train "
                f"window: {vocabulary} distinct frames over {len(tr)} labelled "
                f"samples, largest {biggest}. There is nothing to fit a "
                f"motif->trough map on, so the arm was not spent"),
        }

    buyable = {g["frame"] for g in supported if g["lift"] >= min_lift}

    called = [r for r in te if r[key] in buyable]
    cost = ROUND_TRIP_COST

    def net(rows: Sequence[Mapping[str, Any]]) -> float:
        if not rows:
            return 0.0
        return sum(float(r["_forward"]) for r in rows) / len(rows) - cost

    def trough_rate(rows: Sequence[Mapping[str, Any]]) -> float:
        if not rows:
            return 0.0
        return sum(1 for r in rows if r["_label"] == OMEN_TROUGH) / len(rows)

    # --- THE SELL-HIGH HALF, which nothing here has ever measured ----------
    # omen_experiment.py:552 opens a position only on a BUY-LOW omen because
    # the live lane is long-only, so a crest is an abstention and its accuracy
    # is scored NOWHERE. That is half the labelled vocabulary going unjudged.
    # A crest that correctly calls a fall is worth money as an EXIT on a held
    # position, so it is scored against FORWARD RETURNS -- not by shorting,
    # which this lane cannot do.
    crest_fit = label_skew(tr, key, min_support=min_support,
                           target=OMEN_CREST)
    sellable = {g["frame"] for g in crest_fit.get("groups", [])
                if g["lift"] >= min_lift}
    crest_called = [r for r in te if r[key] in sellable]

    def fall_rate(rows: Sequence[Mapping[str, Any]]) -> float:
        """Share of bars whose forward return is negative.

        The honest test of an EXIT signal: an exit is right when the price it
        exited ahead of went DOWN. No cost is charged here -- exiting a
        position you already hold does not open a round trip, and billing one
        would be the 'round trip billed twice to one leg' shape the profit
        logic audit exists to catch.
        """
        if not rows:
            return 0.0
        return sum(1 for r in rows if float(r["_forward"]) < 0) / len(rows)

    def mean_forward(rows: Sequence[Mapping[str, Any]]) -> float:
        if not rows:
            return 0.0
        return sum(float(r["_forward"]) for r in rows) / len(rows)

    return {
        # Named on the RESULT, not just in the invocation. Every stale number
        # this item exists to mark was stale because the report did not record
        # which encoder produced it.
        # WHICH LAYER THIS NUMBER IS ABOUT, on the artifact rather than only in
        # the invocation. A JSON file holding a held-out edge with no layer name
        # is exactly how an L2 number and an L1 number get quoted as each other.
        "key": key,
        "unspent": False,
        "train_supported_groups": len(supported),
        "banding": "relative" if relative else "sign",
        "band_streams": sorted(bands) if bands else [],
        "margin": margin,
        # BOTH, and deliberately: ``l1_vocabulary_train`` keeps meaning the L1
        # vocabulary whatever layer is being scored, because an existing test
        # and existing artifacts read it under that name. The scored layer's
        # own vocabulary is a separate field rather than the same name meaning
        # two things depending on an argument.
        "l1_vocabulary_train": len({r["L1_cooccurrence"] for r in tr}),
        "vocabulary_train": len({r[key] for r in tr}),
        "train_window": [train_start, train_stop], "train_n": len(tr),
        "test_window": [test_start, test_stop], "test_n": len(te),
        "train_base_trough": base,
        "buyable_motifs": sorted(buyable),
        "called_n": len(called), "called_share": len(called) / len(te),
        "called_net": net(called), "baseline_net": net(te),
        # A rule that called NOTHING has no per-trade net, and
        # called_net - baseline_net on zero trades manufactures a loss out of
        # an abstention. Measured 2026-09-10: the UP window under relative
        # banding calls 0 bars and the subtraction reads -3.8377%, which is
        # just the baseline with a minus sign. Flagged on the artifact so a
        # number lifted out of the JSON cannot be quoted as a measured edge.
        "unmeasurable": not called,
        "unmeasurable_reason": (
            "" if called else
            "no motif reached the support floor with the required lift, so "
            "the rule abstained; there is no per-trade net to compare"
        ),
        "called_trough_precision": trough_rate(called),
        "baseline_trough_rate": trough_rate(te),
        "cost": cost,
        # sell-high half
        "sellable_motifs": sorted(sellable),
        "crest_called_n": len(crest_called),
        "crest_called_share": len(crest_called) / len(te),
        "crest_fall_precision": fall_rate(crest_called),
        "baseline_fall_rate": fall_rate(te),
        "crest_mean_forward": mean_forward(crest_called),
        "baseline_mean_forward": mean_forward(te),
    }


def build_parser() -> argparse.ArgumentParser:
    """Split out from ``main`` so the DEFAULTS are testable without running.

    The default is the whole measurement here: an L2 number built over a
    margin-0 L1 describes an encoder the live path does not ship, and that is
    exactly the configuration that blocked [fa75fa1a] in pass 111 while every
    individual number in it was correct.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--chain", default="base")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--margin", type=float, default=0.75,
                        help="a layer must come in at or below this FRACTION "
                             "of its input's distinctness to count as "
                             "abstracting. 0.75 is a quarter off the "
                             "vocabulary; anything gentler is a rounding "
                             "difference dressed up as a layer.")
    parser.add_argument("--hysteresis", type=float,
                        default=L1_HYSTERESIS_MARGIN,
                        help="L1 band STICKINESS, as a fraction of each "
                             "stream's own band width ([2a53f971]). A slot "
                             "holds its band until the score is pushed this "
                             "far past the boundary. 0.0 is byte-identical to "
                             "plain banding, so the two are comparable on one "
                             "fabric; 0.50 measured an L1 change rate of "
                             "37.6%% DOWN / 38.2%% UP against 73.1%% / 74.0%% "
                             "at 0.0. Only meaningful with --relative-bands: "
                             "a stream with no fitted cut points has no band "
                             "width to be sticky about. DEFAULTS TO THE "
                             "SHIPPED L1_HYSTERESIS_MARGIN: judging a banding "
                             "the live path does not ship is what produced the "
                             "pass-111 block, and margin 0 is reachable by "
                             "asking for it as the control arm.")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--heldout", action="store_true",
                        help="fit the motif->trough map on a train window and "
                             "score it, frozen, on a held-out window")
    parser.add_argument("--train", type=int, default=350)
    parser.add_argument("--test", type=int, default=180)
    parser.add_argument("--min-lift", type=float, default=1.3)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--relative-bands", action="store_true",
                        help="band each L1 stream against ITS OWN terciles "
                             "rather than absolute token signs. Off by "
                             "default so the original sign-banded encoder is "
                             "still reachable and both can be measured on one "
                             "corpus. Under --heldout the cut points are "
                             "fitted on the TRAIN window only.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    path = Path(args.corpus)
    symbol = path.stem.split("_", 1)[-1]
    bars = load_bars(path)
    stop = args.stop if args.stop is not None else len(bars) - args.horizon - 1
    rows = build_layer_frames(bars, symbol, args.chain, args.horizon,
                              args.start, stop)
    if not rows:
        print("no frames could be built -- corpus too short or all bars bad")
        return 2

    # The distinctness section is a CORPUS-WIDE descriptive measurement with no
    # train/test split, so fitting terciles over the whole of it leaks nothing:
    # there is no held-out number here to leak into. The held-out arm below
    # fits its own bands on its own train window and does not reuse these.
    corpus_bands = None
    if args.relative_bands:
        corpus_bands = relative_bands(rows) or None
        rows = build_layer_frames(bars, symbol, args.chain, args.horizon,
                                  args.start, stop, bands=corpus_bands,
                                  margin=args.hysteresis)

    n = len(rows)
    print(f"corpus {path.name}: {len(bars)} bars, {n} frame sets "
          f"[{max(args.start, LOOKBACK_BARS)}, {stop}), horizon {args.horizon}")
    print(f"encoder banding   : "
          f"{'RELATIVE (per-stream terciles)' if args.relative_bands else 'SIGN (absolute token signs)'}"
          f"{'  live slots %d/%d' % (len(corpus_bands), len(_BASELINE_STREAMS)) if corpus_bands else ''}")
    # THE MARGIN IS PART OF EVERY L2 NUMBER BELOW, so it is printed whether or
    # not it had an effect. A margin with no relative bands to be sticky about
    # is inert, and saying "0.50 (INERT ...)" is the honest form of that --
    # silently printing 0.50 beside sign banding would claim a sticky L1 the
    # run did not have.
    print(f"L1 hysteresis     : {args.hysteresis:.2f}"
          f"{'' if args.relative_bands else '  (INERT: needs --relative-bands; a stream with no fitted cut points has no band width to be sticky about)'}"
          f"{'  <- shipped L1_HYSTERESIS_MARGIN' if args.hysteresis == L1_HYSTERESIS_MARGIN and args.relative_bands else ''}")

    l0 = collection_distinctness([{k: v for k, v in r.items()
                                   if not k.startswith("L")} for r in rows])
    layers = layer_distinctness(rows, keys=["L1_cooccurrence", L2_CONTROL,
                                            L2_WINNER])

    print("\nL0 sensory distinctness (distinct frames / samples):")
    for name in sorted(l0, key=lambda k: -l0[k]):
        mark = "  <- L1 input" if name in _BASELINE_STREAMS else ""
        print(f"  {name:<18} {l0[name]:.4f}{mark}")

    # THE BASELINE IS THE INPUT THE LAYER ACTUALLY READS. Comparing L1 against
    # the sharpest stream in the whole corpus would be easy and meaningless;
    # comparing it against the MEAN of the streams it consumes is the question
    # "did combining these five make something coarser than the five".
    inputs = [l0[s] for s in _BASELINE_STREAMS if s in l0]
    baseline = sum(inputs) / len(inputs) if inputs else 0.0
    l1 = layers["L1_cooccurrence"]
    l2_control = layers[L2_CONTROL]
    l2 = layers[L2_WINNER]

    print(f"\nL1 input streams  : {', '.join(_BASELINE_STREAMS)}")
    print(f"L0 mean (inputs)  : {baseline:.4f}")
    print(f"L1 co-occurrence  : {l1:.4f}   "
          f"({l1 / baseline:.2f}x its input)" if baseline else "")
    print(f"L2 transitions    : {l2:.4f}   "
          f"({l2 / l1:.2f}x L1)   <- SHIPPED" if l1 else "")
    print(f"L2 sequence (CTL) : {l2_control:.4f}   "
          f"({l2_control / l1:.2f}x L1)   <- REJECTED [fa75fa1a]" if l1 else "")
    print(f"L1 vocabulary     : {len({r['L1_cooccurrence'] for r in rows})} "
          f"distinct motifs over {n} samples")
    print(f"L2 vocabulary     : {len({r[L2_WINNER] for r in rows})} "
          f"distinct transition frames over {n} samples "
          f"({L2_TRANSITION_STEPS} kept symbols over a "
          f"{L2_TRANSITION_WINDOW}-bar window)")
    print(f"L2 control vocab  : {len({r[L2_CONTROL] for r in rows})} "
          f"distinct fixed paths over {n} samples "
          f"({MOTIF_SEQUENCE_STEPS} steps)")

    top = Counter(r["L1_cooccurrence"] for r in rows).most_common(5)
    print("\nmost frequent L1 motifs (a motif MANY instants share is the "
          "point; a motif per instant is an identifier):")
    for motif, count in top:
        print(f"  {count:>5}  {count / n:>6.1%}  {motif}")

    # --- does the abstraction carry the OUTCOME, not just compress inputs ---
    skew = label_skew(rows, "L1_cooccurrence")
    if skew.get("total"):
        print(f"\nL1 motif -> label skew (corpus trough rate "
              f"{skew['base_trough']:.1%}, groups with n>=20 cover "
              f"{skew['covered']:.0%} of samples):")
        print(f"  {'n':>5} {'share':>7} {'trough':>7} {'lift':>6} "
              f"{'top':>7} {'purity':>7}  motif")
        for g in skew["groups"]:
            print(f"  {g['n']:>5} {g['share']:>6.1%} {g['trough_rate']:>6.1%} "
                  f"{g['lift']:>5.2f}x {g['top_label']:>7} "
                  f"{g['purity']:>6.1%}  {g['frame']}")
        best = skew["groups"][0] if skew["groups"] else None
        if best:
            print(f"\n  BEST BUY-LOW MOTIF: {best['lift']:.2f}x the corpus "
                  f"trough rate on {best['n']} samples "
                  f"({best['share']:.1%} of the window). A lift near 1.00x "
                  f"means the layer compressed the inputs without carrying "
                  f"the outcome, and it would DILUTE a query rather than "
                  f"sharpen it -- which distinctness alone cannot tell you.")

    # THE SUPPORT CENSUS FOR L2, because the held-out arm below is spent or
    # refused on exactly this number and a refusal nobody can see the reason
    # for reads as a failure of the layer.
    l2_skew = label_skew(rows, L2_WINNER)
    if l2_skew.get("total"):
        groups = l2_skew.get("groups", [])
        print(f"\nL2 transition -> label skew (groups with n>=20 cover "
              f"{l2_skew['covered']:.0%} of samples, {len(groups)} of them):")
        for g in groups[:8]:
            print(f"  {g['n']:>5} {g['share']:>6.1%} {g['trough_rate']:>6.1%} "
                  f"{g['lift']:>5.2f}x {g['top_label']:>7} "
                  f"{g['purity']:>6.1%}  {g['frame']}")
        if not groups:
            print("  NONE. Every L2 frame is below the support floor on this "
                  "corpus, so there is nothing to fit a motif->trough map on "
                  "and a held-out L2 arm must NOT be spent.")

    verdicts = []
    l1_ok = baseline > 0 and l1 <= baseline * args.margin
    verdicts.append(("L1", l1_ok, l1, baseline))
    # L2 is judged against L1, its own input -- not against L0. A sequence of
    # motifs is ALLOWED to be sharper than one motif (order adds information);
    # what it must not be is near-unique, which is the identifier trap that
    # took SEQUENCE_STEPS from 8 to 5 in omen_metacognition.
    #
    # JUDGED ON THE SHIPPED SCHEME. Until [a4ba2028] this exit code read
    # L2_sequence -- the scheme [fa75fa1a] REJECTED -- so the gate refused the
    # layer the live path ships on the number of the one it does not.
    l2_ok = l2 <= IDENTIFIER_CEILING
    verdicts.append(("L2", l2_ok, l2, IDENTIFIER_CEILING))

    print()
    for name, ok, value, against in verdicts:
        state = "ABSTRACTS" if ok else "DOES NOT ABSTRACT"
        print(f"{name}: {state}  ({value:.4f} against {against:.4f})")

    if not l1_ok:
        print("\nVERDICT: CUT L1. It is as distinct as the streams it reads, "
              "so it has abstracted nothing and would cost a consolidation "
              "and a query per sample to re-say what L0 already says.")
    elif not l2_ok:
        print(f"\nVERDICT: L1 ABSTRACTS, L2 DOES NOT. The shipped transition "
              f"frame at {l2:.4f} distinct per sample is approaching an "
              f"identifier -- the trap that maximises train recall and "
              f"destroys generalisation. The lever is NOT the step count: "
              f"[fa75fa1a] measured that L1's change rate is what decides "
              f"this, so re-check --hysteresis (ran at "
              f"{args.hysteresis:.2f}) and --relative-bands before training "
              f"on it.")
    else:
        print("\nVERDICT: BOTH LAYERS ABSTRACT. Distinctness falls layer over "
              "layer, so each layer's vocabulary is smaller than its input's. "
              "This says the layers are WORTH MEASURING -- it does NOT say "
              "they predict. Held-out edge in an UP and a DOWN window is "
              "still the only scoreboard.")

    def run_arm(key: str, title: str) -> Dict[str, Any]:
        edge = heldout_edge(bars, symbol, args.chain, args.horizon,
                            args.train, args.test, args.min_lift,
                            args.min_support, relative=args.relative_bands,
                            margin=args.hysteresis, key=key)
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        if edge.get("error"):
            print(f"  {edge['error']}")
        elif edge.get("unspent"):
            # THE ARM IS REFUSED, NOT FAILED, and the difference is the whole
            # reason this branch prints a census instead of a number.
            print(f"  ARM NOT SPENT: {edge['unspent_reason']}.")
            print(f"  train bars {edge['train_window']} -> {edge['train_n']} "
                  f"samples   (banding {edge['banding'].upper()}, hysteresis "
                  f"{edge['margin']:.2f})")
            print(f"  There is no held-out number here and none should be "
                  f"quoted. Widen the train window or coarsen the layer; do "
                  f"NOT lower --min-support to manufacture a group.")
        else:
            print(f"  encoder banding: {edge['banding'].upper()}  "
                  f"({len(edge['band_streams'])}/{len(_BASELINE_STREAMS)} "
                  f"streams banded, hysteresis {edge['margin']:.2f}, "
                  f"{edge['key']} train vocabulary "
                  f"{edge['vocabulary_train']} frames, "
                  f"{edge['train_supported_groups']} of them with "
                  f"n>={args.min_support})")
            if edge["unmeasurable"]:
                print(f"  *** UNMEASURABLE, NOT NEGATIVE: "
                      f"{edge['unmeasurable_reason']}. Any 'edge' printed "
                      f"below is the baseline with a minus sign.")
            print(f"  train bars {edge['train_window']} -> {edge['train_n']} "
                  f"samples   (fit here ONLY)")
            print(f"  test  bars {edge['test_window']} -> {edge['test_n']} "
                  f"samples   ({args.horizon}-bar purge between)")
            print(f"  motifs called buyable (train lift >= {args.min_lift}): "
                  f"{len(edge['buyable_motifs'])}")
            for m in edge["buyable_motifs"]:
                print(f"      {m}")
            print(f"\n  (a) PER-TRADE NET on called bars : "
                  f"{edge['called_net']:+.4%}  over {edge['called_n']} trades "
                  f"({edge['called_share']:.1%} of the window)")
            print(f"      buy-every-bar baseline        : "
                  f"{edge['baseline_net']:+.4%}  over {edge['test_n']} trades")
            delta = edge["called_net"] - edge["baseline_net"]
            print(f"      EDGE                          : {delta:+.4%} "
                  f"per trade   (round trip cost {edge['cost']:.4%})")
            print(f"\n  (b) TROUGH PRECISION on called   : "
                  f"{edge['called_trough_precision']:.1%}")
            print(f"      window base trough rate       : "
                  f"{edge['baseline_trough_rate']:.1%}")
            print(f"\n  --- THE SELL-HIGH HALF (long-only scores this "
                  f"NOWHERE: a crest is an abstention) ---")
            print(f"  motifs called sellable (train crest lift >= "
                  f"{args.min_lift}): {len(edge['sellable_motifs'])}")
            print(f"  (c) CREST FALL PRECISION         : "
                  f"{edge['crest_fall_precision']:.1%}  over "
                  f"{edge['crest_called_n']} calls "
                  f"({edge['crest_called_share']:.1%} of the window)")
            print(f"      window base fall rate         : "
                  f"{edge['baseline_fall_rate']:.1%}")
            print(f"      mean forward on called        : "
                  f"{edge['crest_mean_forward']:+.4%}  vs window "
                  f"{edge['baseline_mean_forward']:+.4%}")
            if edge["crest_called_n"] < 20:
                print(f"      SAMPLE TOO SMALL TO RANK: "
                      f"{edge['crest_called_n']} calls.")

            if edge["called_n"] < 20:
                print(f"\n  SAMPLE TOO SMALL TO RANK: {edge['called_n']} "
                      f"called trades. Reported, not concluded from.")
            elif delta > 0:
                print(f"\n  POSITIVE in this window. ONE WINDOW IS NOT "
                      f"EVIDENCE -- a long-only rule flatters itself in an UP "
                      f"window. Run the DOWN corpus before believing it.")
            else:
                print(f"\n  AT OR BELOW BASELINE in this window. That is the "
                      f"normal outcome here and it is a finished measurement, "
                      f"not a failed one.")
        return edge

    if args.heldout:
        edge = run_arm("L1_cooccurrence",
                       "HELD-OUT EDGE OF THE L1 MOTIF ALONE -- no node, no "
                       "fabric")
        # THE L2 ARM, AND IT READS THE SHIPPED SCHEME. Run unconditionally at
        # this level because ``heldout_edge`` itself refuses to spend an arm
        # with no supported train group -- the decision belongs next to the
        # census it is made from, not to a caller that would have to
        # re-measure it.
        l2_edge = run_arm(L2_WINNER,
                          f"HELD-OUT EDGE OF THE SHIPPED L2 ({L2_WINNER}) -- "
                          f"no node, no fabric")
        if args.json_out:
            edge_path = str(args.json_out).replace(".json", "-heldout.json")
            Path(edge_path).write_text(json.dumps(edge, indent=2),
                                       encoding="utf-8")
            l2_path = str(args.json_out).replace(".json", "-l2-heldout.json")
            Path(l2_path).write_text(json.dumps(l2_edge, indent=2),
                                     encoding="utf-8")
            print(f"\nwrote {edge_path}\nwrote {l2_path}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "corpus": path.name, "samples": n, "horizon": args.horizon,
            "l0": l0, "l0_mean_inputs": baseline,
            "l1": l1, "l2": l2,
            # The rejected control, named as such on the artifact. Until
            # [a4ba2028] "l2" in this file WAS the rejected scheme under an
            # unqualified name.
            "l2_scheme": L2_WINNER,
            "l2_control_scheme": L2_CONTROL,
            "l2_control": l2_control,
            "hysteresis": args.hysteresis,
            "l1_vocabulary": len({r["L1_cooccurrence"] for r in rows}),
            "l2_vocabulary": len({r[L2_WINNER] for r in rows}),
            "l2_control_vocabulary": len({r[L2_CONTROL] for r in rows}),
            "l2_supported_groups": len(l2_skew.get("groups", [])),
            "l1_abstracts": l1_ok, "l2_abstracts": l2_ok,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    return 0 if (l1_ok and l2_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
