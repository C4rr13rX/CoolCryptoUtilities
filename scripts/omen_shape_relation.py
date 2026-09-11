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
    """Bars from one corpus file, or a named refusal.

    Some files under ``data/historical_ohlcv/`` are placeholders holding
    ``["none", []]`` -- a corpus that exists and carries no bars. They are
    refused BY NAME here rather than as a dict-update TypeError three frames
    down, because "unreadable" in a census table hides an empty corpus behind
    what looks like a parser problem.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    bars = raw["bars"] if isinstance(raw, dict) else raw
    rows = [dict(b) for b in bars if isinstance(b, Mapping)]
    if not rows:
        raise ValueError("placeholder corpus -- no bar objects in the file")
    return rows


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


def plan_windows(bars: Sequence[Mapping[str, Any]], *, horizon: int,
                 test_bars: int, train_bars: int,
                 candidates: int) -> Dict[str, Any] | None:
    """Pick an UP and a DOWN held-out window from one corpus, and a train
    window that precedes both with a full horizon gap.

    The windows are chosen by MEAN FORWARD RETURN BEFORE COST -- the criterion
    the item's acceptance names -- over the last ``candidates`` disjoint
    blocks of the corpus: the most positive block is the UP window and the
    most negative is the DOWN window. This is deliberately the HARDEST honest
    pair, not a random pair: a rule that only works in one direction is
    exposed by the two extremes, which is the whole point of running both.

    Returns ``None`` when the corpus is too short, or when its most positive
    block is not actually positive (or its most negative not negative) -- a
    corpus that cannot supply both window classes cannot answer the question
    and is skipped rather than counted.
    """
    last = len(bars) - horizon - 1
    first = LOOKBACK_BARS
    blocks: List[Tuple[int, int, float]] = []
    stop = last
    for _ in range(candidates):
        start = stop - test_bars
        if start < first + train_bars + horizon:
            break
        forwards = []
        for i in range(start, stop):
            entry = float(bars[i]["close"])
            if entry <= 0:
                continue
            forwards.append((float(bars[i + horizon]["close"]) - entry) / entry)
        if forwards:
            blocks.append((start, stop, statistics.fmean(forwards)))
        stop = start
    if len(blocks) < 2:
        return None
    up = max(blocks, key=lambda b: b[2])
    down = min(blocks, key=lambda b: b[2])
    if up[2] <= 0.0 or down[2] >= 0.0 or up[0] == down[0]:
        return None
    train_end = min(up[0], down[0]) - horizon
    train_start = train_end - train_bars
    if train_start < first:
        return None
    return {
        "train": (train_start, train_end),
        "up": (up[0], up[1]),
        "down": (down[0], down[1]),
        "up_forward": up[2],
        "down_forward": down[2],
    }


#: Non-degeneracy bar for the granularity sweep: a scheme whose most common
#: descriptor holds more than this share of the train window is a constant
#: wearing a costume, and its invariance means nothing.
MAX_TOP = 0.50


def parse_grid(spec: str) -> List[Tuple[int, int]]:
    grid: List[Tuple[int, int]] = []
    for token in spec.split(","):
        seg, _, nb = token.strip().partition("x")
        grid.append((int(seg), int(nb)))
    return grid


def choose_granularity(train_raw: Sequence[Mapping[str, Any]],
                       grid: Sequence[Tuple[int, int]], *,
                       min_support: int) -> Tuple[List[Dict[str, Any]],
                                                  Dict[str, Any]]:
    """Pick the descriptor granularity on the TRAIN half, and only there.

    The rule is fixed before any number is read: among schemes that are not
    degenerate (at least four descriptors, most common at most ``MAX_TOP``),
    take the one covering the most of the train window at ``min_support``;
    break ties toward MORE descriptors, i.e. the finest scheme that still has
    support. Reading a held-out window to choose granularity would be fitting
    on the thing being measured -- the error that inverted three of four
    horizons in [c6196eb2].
    """
    rows: List[Dict[str, Any]] = []
    for seg, nb in grid:
        s = support(with_descriptors(train_raw, segments=seg, nbands=nb),
                    min_support=min_support)
        s.update({"segments": seg, "bands": nb,
                  "eligible": s["distinct"] >= 4 and s["top_share"] <= MAX_TOP})
        rows.append(s)
    eligible = [s for s in rows if s["eligible"]]
    chosen = (max(eligible, key=lambda s: (s["covered_share"], s["distinct"]))
              if eligible
              else max(rows, key=lambda s: s["covered_share"]))
    return rows, chosen


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def run_multi(args) -> int:
    """The same protocol over MANY corpora, so the verdict is not one tape.

    Windows are planned per corpus by ``plan_windows`` -- the most positive
    and the most negative disjoint blocks at the end of the corpus, with a
    train window preceding both by a full horizon gap. Granularity is chosen
    on each corpus's OWN train half. A corpus that cannot supply both window
    classes is skipped and counted as skipped, never folded in.
    """
    paths = sorted(Path().glob(args.corpora))[: args.max_corpora]
    if not paths:
        print(f"NO CORPORA matched {args.corpora}")
        return 2
    grid = parse_grid(args.sweep)
    rows: List[Dict[str, Any]] = []
    skipped: List[Tuple[str, str]] = []
    print(f"corpora  : {len(paths)} matching {args.corpora}")
    print(f"protocol : train {args.train_bars} bars, {args.test_bars}-bar UP "
          f"and DOWN windows chosen by mean forward return before cost, "
          f"granularity chosen on each corpus's own train half")
    for path in paths:
        try:
            bars = _load(path)
            cadence = measure_bar_seconds(bars[:512])
        except Exception as exc:  # corpus files vary; a bad one is not a result
            skipped.append((path.name, f"unreadable: {exc}"))
            continue
        horizon = max(1, round(args.horizon_minutes * 60 / cadence))
        plan = plan_windows(bars, horizon=horizon, test_bars=args.test_bars,
                            train_bars=args.train_bars,
                            candidates=args.candidates)
        if plan is None:
            skipped.append((path.name,
                            f"no UP/DOWN pair in {len(bars)} bars"))
            continue
        train_raw = labelled_anchors(bars, *plan["train"], horizon=horizon)
        up_raw = labelled_anchors(bars, *plan["up"], horizon=horizon)
        down_raw = labelled_anchors(bars, *plan["down"], horizon=horizon)
        if not train_raw or not up_raw or not down_raw:
            skipped.append((path.name, "no labelable anchors"))
            continue
        _sweep, chosen = choose_granularity(train_raw, grid,
                                            min_support=args.min_support)
        seg, nb = chosen["segments"], chosen["bands"]
        lookup = fit_lookup(with_descriptors(train_raw, segments=seg,
                                             nbands=nb),
                            min_support=args.min_support)
        up = score(lookup, with_descriptors(up_raw, segments=seg, nbands=nb))
        down = score(lookup, with_descriptors(down_raw, segments=seg,
                                              nbands=nb))
        rows.append({
            "corpus": path.name, "cadence": cadence, "horizon": horizon,
            "scheme": f"{seg}x{nb}", "train_n": len(train_raw),
            "covered": chosen["covered_share"], "up": up, "down": down,
        })
        print(f"  {path.name:<28} {seg}x{nb}  train {len(train_raw):<5} "
              f"UP lift {_fmt(up['lift_pp'], 2):>7}pp (abst "
              f"{_fmt(up['abstention'], 2)})  DOWN lift "
              f"{_fmt(down['lift_pp'], 2):>7}pp (abst "
              f"{_fmt(down['abstention'], 2)})")

    scored = [r for r in rows
              if not math.isnan(r["up"]["lift_pp"])
              and not math.isnan(r["down"]["lift_pp"])]
    both = [r for r in scored
            if r["up"]["lift_pp"] > 0 and r["down"]["lift_pp"] > 0]
    one = [r for r in scored
           if (r["up"]["lift_pp"] > 0) != (r["down"]["lift_pp"] > 0)]
    print(f"\nCORPORA SCORED  : {len(scored)}  (skipped {len(skipped)})")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    if scored:
        up_mean = statistics.fmean(r["up"]["lift_pp"] for r in scored)
        down_mean = statistics.fmean(r["down"]["lift_pp"] for r in scored)
        print(f"MEAN LIFT       : UP {up_mean:+.2f}pp   DOWN "
              f"{down_mean:+.2f}pp")
        print(f"POSITIVE IN BOTH: {len(both)} of {len(scored)}   "
              f"(one window only: {len(one)})")
        verdict = (
            f"NEGATIVE ACROSS THE POPULATION -- {len(both)} of {len(scored)} "
            f"corpora positive in both window classes, mean lift "
            f"{up_mean:+.2f}pp UP and {down_mean:+.2f}pp DOWN"
            if len(both) * 2 <= len(scored) else
            f"POSITIVE IN BOTH WINDOWS ON {len(both)} of {len(scored)} "
            f"corpora -- worth a node arm, still an offline upper bound")
    else:
        up_mean = down_mean = float("nan")
        verdict = "NOT MEASURABLE -- no corpus supplied both window classes"
    print(f"\nVERDICT: {verdict}")
    if args.report:
        _write_multi_report(Path(args.report), args=args, rows=rows,
                            skipped=skipped, scored=len(scored),
                            both=len(both), one=len(one), up_mean=up_mean,
                            down_mean=down_mean, verdict=verdict)
        print(f"report   : {args.report}")
    return 0


def _write_multi_report(dest: Path, *, args, rows, skipped, scored, both,
                        one, up_mean, down_mean, verdict) -> None:
    lines: List[str] = []
    a = lines.append
    a("# SHAPE RELATION ACROSS THE CORPUS POPULATION")
    a("")
    a("Pass 120, Cove. Cove's part of item `[5c3b2189]` (owner Jet). This")
    a("widens the single-corpus result in `SHAPE-RELATION-pass120-cove.md`")
    a("from one tape to many. Still node-free, still an OFFLINE UPPER BOUND:")
    a("a descriptor -> majority-label lookup is the best a substrate could do")
    a("with this stream alone, so a stream that fails here fails on a fabric.")
    a("")
    a("## 1. Protocol")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| corpora | `{args.corpora}`, first {args.max_corpora} by name |")
    a(f"| horizon | {args.horizon_minutes} min, converted per corpus cadence |")
    a(f"| train | {args.train_bars} bars, ending a full horizon before the "
      "earlier test window |")
    a(f"| held-out | {args.test_bars} bars per window class |")
    a(f"| window classes | the most positive and most negative of the last "
      f"{args.candidates} disjoint blocks, by mean forward return BEFORE "
      "cost |")
    a(f"| granularity | chosen on each corpus's OWN train half, sweep "
      f"`{args.sweep}` |")
    a(f"| min_support | {args.min_support} |")
    a("")
    a("Choosing the extreme blocks is deliberate: it is the hardest honest")
    a("pair, and a rule that works in only one direction cannot hide in it.")
    a("")
    a("## 2. Per corpus")
    a("")
    a("| corpus | scheme | train n | UP exact | UP baseline | UP lift | "
      "DOWN exact | DOWN baseline | DOWN lift |")
    a("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        u, d = r["up"], r["down"]
        a(f"| `{r['corpus']}` | {r['scheme']} | {r['train_n']} | "
          f"{_fmt(u['accuracy'])} | {_fmt(u['baseline_answered'])} | "
          f"**{_fmt(u['lift_pp'], 2)}pp** | {_fmt(d['accuracy'])} | "
          f"{_fmt(d['baseline_answered'])} | "
          f"**{_fmt(d['lift_pp'], 2)}pp** |")
    a("")
    if skipped:
        a("Skipped, and why -- a corpus that cannot supply both window")
        a("classes is not folded in:")
        a("")
        for name, why in skipped:
            a(f"* `{name}` -- {why}")
        a("")
    a("## 3. Population result")
    a("")
    a(f"* corpora scored: **{scored}**")
    a(f"* positive in BOTH window classes: **{both}**")
    a(f"* positive in one window class only (a FAIL): {one}")
    a(f"* mean lift: **{up_mean:+.2f}pp** UP, **{down_mean:+.2f}pp** DOWN")
    a("")
    a(f"**{verdict}**")
    a("")
    a("## 4. What this does NOT establish")
    a("")
    a("* Nothing was trained. The node was not touched; production on")
    a("  `:8090` was not involved.")
    a("* A lookup is an upper bound for THIS STREAM ALONE. It does not")
    a("  predict what a fabric does with the stream beside ten other pools.")
    a("* The window classes are the extremes of each corpus's own tail, so")
    a("  they are harder than average windows, not representative ones.")
    a("")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus")
    ap.add_argument("--corpora", help="glob for the multi-corpus census, "
                                      "e.g. 'data/historical_ohlcv/base/*.json'")
    ap.add_argument("--max-corpora", type=int, default=12)
    ap.add_argument("--test-bars", type=int, default=400)
    ap.add_argument("--train-bars", type=int, default=1200)
    ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--horizon-minutes", type=int, default=720)
    ap.add_argument("--train-start", type=int)
    ap.add_argument("--train-end", type=int)
    ap.add_argument("--up-test", type=int, nargs=2,
                    metavar=("START", "STOP"))
    ap.add_argument("--down-test", type=int, nargs=2,
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

    if args.corpora:
        return run_multi(args)
    missing = [name for name, value in (
        ("--corpus", args.corpus), ("--train-start", args.train_start),
        ("--train-end", args.train_end), ("--up-test", args.up_test),
        ("--down-test", args.down_test)) if value is None]
    if missing:
        ap.error("single-corpus mode needs " + ", ".join(missing)
                 + " (or pass --corpora for the population census)")

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

    grid = parse_grid(args.sweep)
    sweep_rows, chosen = choose_granularity(train_raw, grid,
                                            min_support=args.min_support)
    print("\nGRANULARITY SWEEP -- TRAIN HALF ONLY, the held-out "
          "windows are not read here")
    for s_ in sweep_rows:
        print(f"  {s_['segments']}x{s_['bands']:<3} distinct "
              f"{s_['distinct']:<5} median/desc "
              f"{s_['median_per_descriptor']:<6.1f} top share "
              f"{s_['top_share']:.4f}  supported "
              f"{s_['supported_descriptors']:<4} covering "
              f"{s_['covered_share']:.4f}"
              f"{'' if s_['eligible'] else '   [degenerate, not eligible]'}")
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
