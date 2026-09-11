#!/usr/bin/env python
"""The 'same shape' RELATION as a canonical descriptor -- measured node-free.

WHY THIS EXISTS, AND WHAT IT REPLACES. Item [5c3b2189] asks for the relation
"these two frames are the same shape" to be taught in a ``PoolKind::Internal``
pool. That pool kind cannot carry it: ``PoolKind::Internal`` matches ZERO times
across ``crates/`` in W1z4rDV1510n, and the only behavioural match on
``PoolKind`` anywhere in the engine is ``PoolKind::Action``
(``crates/brain/src/brain.rs:7425``). The ``pool.rs`` sentence the item cites --
"internal learned-route frames re-stimulate atoms grounded by other pools" --
is a comment inside ``InstructionIntentEncoding::atomize``, about the coding
brain's ``@intent:`` encoding, not about the pool kind. Both facts were
established in pass 106 (``TOPOLOGY-DESIGN-pass106-cove.md``) and re-verified
independently in pass 108 (``TOPOLOGY-ASSOCIATION-pass108-gale.md``).

So a relation becomes bindable by being WRITTEN DOWN, not by being labelled --
which is what pass 106 already shipped as the client-computed relation pools
12-14 of ``market_predictor_v3_assoc.identity.toml``.

THE DESIGN. Pass 117 powered the shape-mutation arm and the direction INVERTED
(held-out exact 0.3400 -> 0.2950 UP, 0.2825 -> 0.2375 DOWN). The diagnosis was
that a mutation lands as ANOTHER near-unique key, so more pairs is more
memorisation. That diagnosis has a direct consequence: the mutations only pay
if something SHARED is emitted alongside them. This module computes that shared
thing -- a canonical, coarse descriptor of the lookback path:

    z-normalise the closes      -> invariant to price level and to amplitude
    downsample to K segments    -> invariant to tempo and to bar-level noise
    band each segment into B    -> coarse enough that near-misses collide

and the descriptor of a label-safe mutation is, by design, the descriptor of
its base frame.

WHY IT IS MEASURED HERE BEFORE ANY NODE TIME IS SPENT. A stream is only worth a
pool if it carries held-out label information. That is decidable offline and
far more cheaply than on the node: fit descriptor -> majority-label on the TRAIN
half, read it on the HELD-OUT half, and compare against the majority class. A
frequency lookup is the BEST a substrate could do with this stream and nothing
else, so it is an upper bound: if the lookup cannot beat the majority class out
of sample, no pool topology carrying this stream will either, and the node pass
is saved. A lookup that DOES beat it is a green light and a number to beat --
it is not itself an edge, because it is not the fabric.

FOUR GATES, in the order they can kill the idea:

  1. INVARIANCE -- the share of anchors where descriptor(mutation) ==
     descriptor(base). This is the property eleven flat sensory pools do not
     have. Reported per admitted mutation. Low invariance means the descriptor
     is still keyed to the instant.
  2. NON-DEGENERACY -- the descriptor must not be a constant. Reported as the
     count of distinct descriptors and the share of the window held by the
     single most common one. A descriptor that is always "CCCCCCCC" is
     invariant and worthless.
  3. SUPPORT -- the median anchors per descriptor over the train half and the
     share of the train window covered by descriptors at or above
     ``--min-support``. This is the same famine that closed L1 ([89693706]),
     asked BEFORE the encoder ships rather than after.
  4. HELD-OUT LIFT -- exact accuracy of the train-fitted lookup on the held-out
     window against that window's own majority class, in an UP window AND a
     DOWN window, with the abstention rate at every cell. An arm that wins in
     only one window class is a FAIL and is reported as one.

Usage (the pass-114 two-window protocol, same corpus and same windows so the
numbers sit beside a measured fabric baseline):

    python -X utf8 scripts/omen_shape_relation.py \
        --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
        --horizon-minutes 720 \
        --train-start 19301 --train-end 20501 \
        --up-test 20513 20913 --down-test 21313 21713 \
        --report data/brain_experiments/SHAPE-RELATION-pass120-cove.md

Exit code is nonzero when a gate cannot be evaluated at all (no labelable
anchors). A NEGATIVE RESULT IS EXIT 0: below baseline is the normal outcome
here and this script reports it rather than hiding it.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.omen_brain import (  # noqa: E402
    LOOKBACK_BARS, RANGE_WINDOW, label_omen, measure_bar_seconds,
)
from scripts.omen_shape_mutations import (  # noqa: E402
    ADMITTED, MUTATIONS,
)

#: Band letters. BYTE-DISJOINT by construction -- single distinct characters,
#: so no band's token is a substring of another's. The trap this avoids is the
#: one that cost a pass already: atoms are bytes, so "loss_big" contains
#: "loss" and the frequent class swallows the rare one.
BANDS = "ABCDE"

#: Cuts on the z-normalised scale, one fewer than len(BANDS). Fixed rather
#: than fitted: a cut fitted on the window being scored leaks the window into
#: its own descriptor, and this repo has already paid for a cut fitted on the
#: half it was read on ([c6196eb2], three of four horizons inverted).
Z_CUTS = (-1.0, -0.35, 0.35, 1.0)

#: Band count -> (letters, cuts). Every letter is distinct and single-byte, so
#: no band token is a substring of another. Cuts are fixed constants for each
#: scheme and are never fitted on a scored window.
BAND_SCHEMES: Dict[int, Tuple[str, Tuple[float, ...]]] = {
    2: ("AB", (0.0,)),
    3: ("ABC", (-0.5, 0.5)),
    5: (BANDS, Z_CUTS),
}


def shape_descriptor(closes: Sequence[float], *, segments: int = 8,
                     cuts: Sequence[float] = Z_CUTS,
                     bands: str = BANDS) -> str:
    """The canonical coarse form of one price path.

    ``closes`` is the lookback path with the ANCHOR LAST. The return is a
    string like ``shape8:CCDDCBBA`` -- a token the caller can emit into a
    relation collection and that two different frames of the same shape share.

    Invariance is by construction, in three steps:
      * subtracting the mean and dividing by the standard deviation removes
        price LEVEL and price AMPLITUDE, so the same shape at $0.50 and at
        $1.20, or at half the volatility, descrbes identically;
      * averaging into ``segments`` equal buckets removes TEMPO and bar-level
        noise, so a dilated or jittered path lands on the same buckets;
      * banding with FIXED cuts removes the last of the precision.

    A flat path has zero standard deviation and cannot be z-normalised. It is
    given the middle band everywhere rather than being dropped, because a flat
    run is a real shape and dropping it would silently bias the census toward
    volatile anchors.
    """
    n = len(closes)
    if n < segments or segments < 1:
        raise ValueError(f"need at least {segments} closes, got {n}")
    mean = statistics.fmean(closes)
    try:
        sd = statistics.pstdev(closes)
    except statistics.StatisticsError:  # pragma: no cover - n>=segments>=1
        sd = 0.0
    middle = bands[len(bands) // 2]
    if sd <= 0.0 or not math.isfinite(sd):
        return f"shape{segments}:{middle * segments}"
    out = []
    for s in range(segments):
        lo = (s * n) // segments
        hi = ((s + 1) * n) // segments
        hi = max(hi, lo + 1)
        z = (statistics.fmean(closes[lo:hi]) - mean) / sd
        idx = 0
        while idx < len(cuts) and z >= cuts[idx]:
            idx += 1
        out.append(bands[idx])
    return f"shape{segments}:{''.join(out)}"


def _desc(path: Sequence[float], segments: int, nbands: int) -> str:
    letters, cuts = BAND_SCHEMES[nbands]
    return shape_descriptor(path, segments=segments, cuts=cuts, bands=letters)


def _closes(window: Sequence[Mapping[str, Any]]) -> List[float]:
    return [float(b["close"]) for b in window]


def _load(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    bars = raw["bars"] if isinstance(raw, dict) else raw
    return [dict(b) for b in bars]


def _anchor_window(bars: Sequence[Mapping[str, Any]], index: int,
                   horizon: int) -> List[Dict[str, Any]] | None:
    lo = index - LOOKBACK_BARS
    hi = index + horizon + 1
    if lo < 0 or hi > len(bars):
        return None
    return [dict(b) for b in bars[lo:hi]]


def labelled_anchors(bars: Sequence[Mapping[str, Any]], start: int,
                     stop: int, *, horizon: int) -> List[Dict[str, Any]]:
    """Every labelable anchor in ``[start, stop)`` with its lookback PATH.

    The path stops at the anchor -- the horizon tail is never in it -- so a
    descriptor computed from it cannot see its own label.
    """
    out: List[Dict[str, Any]] = []
    for index in range(start, stop):
        window = _anchor_window(bars, index, horizon)
        if window is None:
            continue
        anchor = LOOKBACK_BARS
        label = label_omen(window, anchor, horizon_bars=horizon)
        if label is None:
            continue
        entry = float(bars[index]["close"])
        forward = (float(bars[index + horizon]["close"]) - entry) / entry
        out.append({
            "index": index,
            "label": label,
            "forward": forward,
            "path": _closes(window[: anchor + 1]),
        })
    return out


def with_descriptors(rows: Sequence[Mapping[str, Any]], *, segments: int,
                     nbands: int) -> List[Dict[str, Any]]:
    return [dict(r, descriptor=_desc(r["path"], segments, nbands))
            for r in rows]


# --------------------------------------------------------------------------
# GATE 1 -- invariance under the admitted, label-safe mutations.
# --------------------------------------------------------------------------
def invariance(bars: Sequence[Mapping[str, Any]], anchors: Sequence[int], *,
               horizon: int, segments: int, nbands: int,
               seed: int) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for kind in ADMITTED:
        fn, strength, _admitted, _why = MUTATIONS[kind]
        rng = random.Random(seed)
        same = 0
        moved_path = 0
        total = 0
        flipped = 0
        for index in anchors:
            window = _anchor_window(bars, index, horizon)
            if window is None:
                continue
            anchor = LOOKBACK_BARS
            base_label = label_omen(window, anchor, horizon_bars=horizon)
            if base_label is None:
                continue
            moved = fn([dict(b) for b in window], anchor, rng, strength)
            if label_omen(moved, anchor, horizon_bars=horizon) != base_label:
                flipped += 1
            base_path = _closes(window[: anchor + 1])
            mut_path = _closes(moved[: anchor + 1])
            total += 1
            if mut_path != base_path:
                moved_path += 1
            if (_desc(base_path, segments, nbands)
                    == _desc(mut_path, segments, nbands)):
                same += 1
        rows[kind] = {
            "n": total,
            "label_flips": flipped,
            "path_changed_rate": (moved_path / total) if total else 0.0,
            "descriptor_same_rate": (same / total) if total else 0.0,
        }
    return rows


# --------------------------------------------------------------------------
# GATES 2 and 3 -- non-degeneracy and support, on the train half only.
# --------------------------------------------------------------------------
def support(rows: Sequence[Mapping[str, Any]], *,
            min_support: int) -> Dict[str, Any]:
    counts = Counter(r["descriptor"] for r in rows)
    n = len(rows)
    supported = {d: c for d, c in counts.items() if c >= min_support}
    covered = sum(supported.values())
    per = sorted(counts.values())
    return {
        "anchors": n,
        "distinct": len(counts),
        "median_per_descriptor": (statistics.median(per) if per else 0.0),
        "top_share": (max(counts.values()) / n) if n else 0.0,
        "supported_descriptors": len(supported),
        "covered_share": (covered / n) if n else 0.0,
        "min_support": min_support,
    }


# --------------------------------------------------------------------------
# GATE 4 -- held-out lift of a train-fitted descriptor -> majority lookup.
# --------------------------------------------------------------------------
def fit_lookup(rows: Sequence[Mapping[str, Any]], *,
               min_support: int) -> Dict[str, str]:
    by: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by[r["descriptor"]][r["label"]] += 1
    out: Dict[str, str] = {}
    for descriptor, labels in by.items():
        if sum(labels.values()) < min_support:
            continue
        out[descriptor] = labels.most_common(1)[0][0]
    return out


def _wilson_se(p: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def score(lookup: Mapping[str, str],
          rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Exact accuracy of the lookup, against this window's own majority class.

    The baseline is computed on the SCORED window, never on the train half:
    a train-half baseline flatters any arm whose held-out window has a
    different class mix, and that error has produced a fake number here before.
    """
    n = len(rows)
    answered = [r for r in rows if r["descriptor"] in lookup]
    hit = sum(1 for r in answered if lookup[r["descriptor"]] == r["label"])
    counts = Counter(r["label"] for r in rows)
    majority_label, majority_n = (counts.most_common(1)[0] if counts
                                  else ("", 0))
    baseline = (majority_n / n) if n else float("nan")
    # The same baseline read on the ANSWERED subset only, because an arm that
    # abstains is not entitled to be compared against the whole window: if it
    # only answers where the majority class is easy, the unconditional
    # baseline understates what abstention alone bought.
    answered_counts = Counter(r["label"] for r in answered)
    answered_baseline = ((answered_counts.most_common(1)[0][1] / len(answered))
                         if answered else float("nan"))
    accuracy = (hit / len(answered)) if answered else float("nan")
    forwards = [r["forward"] for r in rows]
    return {
        "n": n,
        "answered": len(answered),
        "abstention": 1.0 - (len(answered) / n) if n else float("nan"),
        "accuracy": accuracy,
        "accuracy_se": _wilson_se(accuracy, len(answered)) if answered
        else float("nan"),
        "majority_label": majority_label,
        "baseline_all": baseline,
        "baseline_answered": answered_baseline,
        "lift_pp": ((accuracy - answered_baseline) * 100.0
                    if answered else float("nan")),
        "mean_forward": statistics.fmean(forwards) if forwards else 0.0,
        "label_mix": dict(counts),
    }


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--horizon-minutes", type=int, default=720)
    ap.add_argument("--train-start", type=int, required=True)
    ap.add_argument("--train-end", type=int, required=True)
    ap.add_argument("--up-test", type=int, nargs=2, required=True,
                    metavar=("START", "STOP"))
    ap.add_argument("--down-test", type=int, nargs=2, required=True,
                    metavar=("START", "STOP"))
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--sweep", default="2x2,3x2,4x2,2x3,3x3,4x3,6x3,4x5,8x5",
                    help="granularity grid as SEGMENTSxBANDS, comma separated. "
                         "The winner is chosen on the TRAIN half only.")
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--invariance-sample", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    path = Path(args.corpus)
    bars = _load(path)
    cadence = measure_bar_seconds(bars[:512])
    horizon = max(1, round(args.horizon_minutes * 60 / cadence))
    print(f"corpus   : {path.name} -- {len(bars)} bars at {cadence}s")
    print(f"horizon  : {args.horizon_minutes} min = {horizon} bars "
          f"(lookback {LOOKBACK_BARS}, range window {RANGE_WINDOW})")
    print("descriptor: FIXED z cuts per band scheme, never fitted on a "
          "scored window")

    train_raw = labelled_anchors(bars, args.train_start, args.train_end,
                                 horizon=horizon)
    up_raw = labelled_anchors(bars, args.up_test[0], args.up_test[1],
                              horizon=horizon)
    down_raw = labelled_anchors(bars, args.down_test[0], args.down_test[1],
                                horizon=horizon)
    if not train_raw or not up_raw or not down_raw:
        print("NO LABELABLE ANCHORS in one of the windows -- nothing to "
              f"measure (train {len(train_raw)}, up {len(up_raw)}, "
              f"down {len(down_raw)}).")
        return 2

    # THE GRANULARITY SWEEP, AND WHY IT CANNOT SEE A HELD-OUT BAR. Every grid
    # point is scored on the TRAIN half alone and the winner is picked by a
    # rule stated before the numbers were read: among schemes that are not
    # degenerate (the most common descriptor holds at most MAX_TOP of the
    # window, and at least 4 descriptors exist), take the one covering the
    # most of the train window at min_support; break ties toward MORE
    # descriptors, i.e. the finest scheme that still has support. Reading the
    # held-out window to choose granularity would be fitting on the thing
    # being measured -- the exact error that inverted three of four horizons
    # in [c6196eb2].
    MAX_TOP = 0.50
    grid: List[Tuple[int, int]] = []
    for token in args.sweep.split(","):
        seg, _, nb = token.strip().partition("x")
        grid.append((int(seg), int(nb)))
    sweep_rows: List[Dict[str, Any]] = []
    print("\nGRANULARITY SWEEP -- TRAIN HALF ONLY, the held-out windows are "
          "not read here")
    for seg, nb in grid:
        rows = with_descriptors(train_raw, segments=seg, nbands=nb)
        s = support(rows, min_support=args.min_support)
        eligible = s["distinct"] >= 4 and s["top_share"] <= MAX_TOP
        s.update({"segments": seg, "bands": nb, "eligible": eligible})
        sweep_rows.append(s)
        print(f"  {seg}x{nb:<3} distinct {s['distinct']:<5} median/desc "
              f"{s['median_per_descriptor']:<6.1f} top share "
              f"{s['top_share']:.4f}  supported {s['supported_descriptors']:<4} "
              f"covering {s['covered_share']:.4f}"
              f"{'' if eligible else '   [degenerate, not eligible]'}")

    eligible_rows = [s for s in sweep_rows if s["eligible"]]
    if not eligible_rows:
        print("\nNO ELIGIBLE GRANULARITY: every scheme is degenerate or "
              "unsupported. That is the result; the stream is not usable at "
              "any granularity on this grid.")
        chosen = max(sweep_rows, key=lambda s: s["covered_share"])
    else:
        chosen = max(eligible_rows,
                     key=lambda s: (s["covered_share"], s["distinct"]))
    segments, nbands = chosen["segments"], chosen["bands"]
    print(f"\nCHOSEN ON TRAIN: {segments}x{nbands} -- covers "
          f"{chosen['covered_share']:.4f} of the train window with "
          f"{chosen['supported_descriptors']} supported descriptors")

    train = with_descriptors(train_raw, segments=segments, nbands=nbands)
    up = with_descriptors(up_raw, segments=segments, nbands=nbands)
    down = with_descriptors(down_raw, segments=segments, nbands=nbands)

    rng = random.Random(args.seed)
    sample = [r["index"] for r in train]
    if len(sample) > args.invariance_sample:
        sample = rng.sample(sample, args.invariance_sample)
    inv = invariance(bars, sorted(sample), horizon=horizon,
                     segments=segments, nbands=nbands, seed=args.seed)

    sup = support(train, min_support=args.min_support)
    lookup = fit_lookup(train, min_support=args.min_support)
    up_score = score(lookup, up)
    down_score = score(lookup, down)

    print("\nGATE 1 -- invariance under label-safe mutations")
    for kind, row in inv.items():
        print(f"  {kind:<14} n={row['n']:<4} path changed "
              f"{row['path_changed_rate']:.4f}  descriptor SAME "
              f"{row['descriptor_same_rate']:.4f}  label flips "
              f"{row['label_flips']}")

    print("\nGATE 2/3 -- non-degeneracy and support (TRAIN half only)")
    print(f"  anchors {sup['anchors']}  distinct {sup['distinct']}  "
          f"median per descriptor {sup['median_per_descriptor']:.1f}  "
          f"top share {sup['top_share']:.4f}")
    print(f"  clearing min_support={args.min_support}: "
          f"{sup['supported_descriptors']} descriptors covering "
          f"{sup['covered_share']:.4f} of the train window")

    print("\nGATE 4 -- held-out lift, UP and DOWN, lookup fitted on TRAIN only")
    for name, s in (("UP", up_score), ("DOWN", down_score)):
        print(f"  {name:<5} n={s['n']:<4} answered {s['answered']:<4} "
              f"abstention {_fmt(s['abstention'])}  "
              f"exact {_fmt(s['accuracy'])} +/- {_fmt(s['accuracy_se'])}  "
              f"baseline(answered) {_fmt(s['baseline_answered'])}  "
              f"lift {_fmt(s['lift_pp'], 2)}pp  "
              f"mean forward {_fmt(s['mean_forward'], 6)}")

    up_wins = (not math.isnan(up_score["lift_pp"])
               and up_score["lift_pp"] > 0.0)
    down_wins = (not math.isnan(down_score["lift_pp"])
                 and down_score["lift_pp"] > 0.0)
    if up_wins and down_wins:
        verdict = ("BOTH WINDOWS POSITIVE -- worth a node arm, and the lift is "
                   "an offline upper bound, not an edge")
    elif up_wins or down_wins:
        verdict = ("FAIL -- positive in ONE window class only, which is the "
                   "single-window error this loop has already paid for twice")
    else:
        verdict = "NEGATIVE in both windows -- the stream carries no held-out lift"
    print(f"\nVERDICT: {verdict}")

    if args.report:
        _write_report(Path(args.report), args=args, corpus=path,
                      bars=len(bars), cadence=cadence, horizon=horizon,
                      inv=inv, sup=sup, lookup=lookup, sweep=sweep_rows,
                      chosen=chosen, up_score=up_score,
                      down_score=down_score, verdict=verdict)
        print(f"report   : {args.report}")
    return 0


def _write_report(dest: Path, *, args, corpus: Path, bars: int, cadence: int,
                  horizon: int, inv, sup, lookup, sweep, chosen, up_score,
                  down_score, verdict: str) -> None:
    lines: List[str] = []
    a = lines.append
    a("# SHAPE RELATION -- the 'same shape' descriptor, measured node-free")
    a("")
    a(f"Pass 120, Cove. Cove's part of item `[5c3b2189]` (owner Jet).")
    a("")
    a("This is NOT a fabric measurement and does not claim an edge. It is the")
    a("offline upper bound on what a pool carrying this stream could learn:")
    a("a descriptor -> majority-label lookup fitted on the train window and")
    a("read on two held-out windows. A substrate given this stream and nothing")
    a("else cannot beat its own frequency table out of sample, so a lookup")
    a("that fails here fails on the node too, for less than a minute of CPU.")
    a("")
    a("## 1. Corpus and windows")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| corpus | `{corpus.as_posix()}` |")
    a(f"| bars | {bars} at {cadence}s cadence |")
    a(f"| horizon | {args.horizon_minutes} min = {horizon} bars |")
    a(f"| train window | `[{args.train_start}, {args.train_end})` |")
    a(f"| UP held-out | `[{args.up_test[0]}, {args.up_test[1]})` |")
    a(f"| DOWN held-out | `[{args.down_test[0]}, {args.down_test[1]})` |")
    a(f"| descriptor | {chosen['segments']} segments x {chosen['bands']} "
      f"bands, chosen on the TRAIN half (sweep in §2) |")
    a(f"| min_support | {args.min_support} |")
    a("")
    a("Windows are the pass-114 two-window protocol so these numbers sit")
    a("beside a measured fabric baseline on the same corpus and the same bars.")
    a("Window CLASS is not asserted -- it is read from mean forward return")
    a("before cost, printed in §4.")
    a("")
    a("## 2. Granularity sweep -- TRAIN half only")
    a("")
    a("The held-out windows are not read here. The winner is the scheme")
    a("covering the most of the train window at min_support, among schemes")
    a("whose most common descriptor holds at most 50% of the window and that")
    a("have at least four descriptors; ties break toward more descriptors.")
    a("")
    a("| scheme | distinct | median/descriptor | top share | supported | "
      "covered | eligible |")
    a("|---|---|---|---|---|---|---|")
    for s_ in sweep:
        mark = "**<-- chosen**" if (s_["segments"], s_["bands"]) == (
            chosen["segments"], chosen["bands"]) else ""
        a(f"| {s_['segments']}x{s_['bands']} | {s_['distinct']} | "
          f"{s_['median_per_descriptor']:.1f} | {s_['top_share']:.4f} | "
          f"{s_['supported_descriptors']} | {s_['covered_share']:.4f} | "
          f"{'yes' if s_['eligible'] else 'no'} {mark} |")
    a("")
    a("## 3. Gate 1 -- invariance under the label-safe mutations")
    a("")
    a("| mutation | n | path changed | descriptor SAME | label flips |")
    a("|---|---|---|---|---|")
    for kind, row in inv.items():
        a(f"| `{kind}` | {row['n']} | {row['path_changed_rate']:.4f} | "
          f"**{row['descriptor_same_rate']:.4f}** | {row['label_flips']} |")
    a("")
    a("`path changed` is the share of anchors where the mutation really moved")
    a("the price path -- without it a high SAME rate would mean nothing. Label")
    a("flips must read 0: the admitted mutations are deep-prefix only and")
    a("cannot touch entry, future or position-in-range.")
    a("")
    a("## 4. Gates 2 and 3 -- non-degeneracy and support, TRAIN half only")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| labelled anchors | {sup['anchors']} |")
    a(f"| distinct descriptors | {sup['distinct']} |")
    a(f"| median anchors per descriptor | {sup['median_per_descriptor']:.1f} |")
    a(f"| share held by the most common | {sup['top_share']:.4f} |")
    a(f"| descriptors clearing min_support={sup['min_support']} | "
      f"{sup['supported_descriptors']} |")
    a(f"| share of the train window they cover | {sup['covered_share']:.4f} |")
    a(f"| lookup entries fitted | {len(lookup)} |")
    a("")
    a("## 5. Gate 4 -- held-out lift, both window classes")
    a("")
    a("| window | n | answered | abstention | exact | se | baseline (answered) "
      "| lift | mean forward |")
    a("|---|---|---|---|---|---|---|---|---|")
    for name, s in (("UP", up_score), ("DOWN", down_score)):
        a(f"| {name} | {s['n']} | {s['answered']} | {_fmt(s['abstention'])} | "
          f"**{_fmt(s['accuracy'])}** | {_fmt(s['accuracy_se'])} | "
          f"{_fmt(s['baseline_answered'])} | **{_fmt(s['lift_pp'], 2)}pp** | "
          f"{_fmt(s['mean_forward'], 6)} |")
    a("")
    a("The baseline is the majority class OF THE ANSWERED SUBSET, not of the")
    a("whole window: an arm that abstains is not entitled to credit for the")
    a("bars it declined, and comparing an abstaining arm against the")
    a("unconditional baseline is how abstention gets mistaken for skill.")
    a("")
    a("## 6. Verdict")
    a("")
    a(f"**{verdict}**")
    a("")
    a("## 7. What this does NOT establish")
    a("")
    a("* It is not a fabric number. Nothing was trained; the node was not")
    a("  touched and production on `:8090` was not involved.")
    a("* A lookup table is an UPPER bound on this stream alone, not a")
    a("  prediction of what the fabric does with it alongside ten other pools.")
    a("* One corpus. The pass-114 binding limitation is unchanged.")
    a("* `PoolKind::Internal` remains inert (0 matches across `crates/`; the")
    a("  only behavioural `PoolKind` match is `Action` at `brain.rs:7425`), so")
    a("  if this stream ever ships it ships as a client-computed")
    a("  `SensoryInput` relation pool, the pass-106 pattern.")
    a("")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
