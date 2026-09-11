#!/usr/bin/env python3
"""
Layered motif frames: L1 co-occurrence and L2 sequence-of-motifs.

THE PREMISE, in the operator's words: prediction in this model works by "I've
seen it before", but abstraction context is multivariate -- it is looking for
MOTIFS. With one flat layer, "it" is the raw conjunction of every sensory
frame at one instant, which is a nearly unique key. That is why near-unique
streams maximise recall and destroy generalisation, and why this fabric
reproduces 98.7% of its training set while generalising at chance. It has seen
the INSTANT before. It has never seen the MOTIF, because nothing names one.

A motif is an abstraction over instants, so it needs a layer whose vocabulary
is already abstract:

    L0  sensory frames            near-unique per instant
    L1  co-occurrence motif       WHICH variables are extreme together, now
    L2  sequence of L1 motifs     which motifs, in what ORDER, over N bars

L2 is the layer that makes "a sequence that has meant something predictable"
representable at all. ``omen_metacognition.temporal_sequence`` encodes an
ordered path of raw PRICE steps -- the same idea applied one layer too low.
Here the path is over motif names.

THE RULE THAT GOVERNS EVERY LAYER. A higher layer earns its place only when
its vocabulary is SMALLER than its input's -- that shrinkage IS the
abstraction. If L1 is as distinct as L0, it has abstracted nothing and is a
lossy copy that costs a consolidation and a query per sample.
``layer_distinctness`` measures exactly that, and it is meant to be run BEFORE
any held-out number is trusted.

NEVER PREDICT WHAT YOU CAN COMPUTE. Every frame here is computed from settled
facts. The chained stage-1 regime is the counter-example this repo already
paid for: 4 distinct values over 2725 samples, reproduced at 73.3%, at 0.98
confidence when wrong, for a deterministic function of the bars.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    "cooccurrence_motif",
    "relative_bands",
    "sticky_motifs",
    "sequence_motif",
    "layer_distinctness",
    "L1_STREAMS",
    "MOTIF_SEQUENCE_STEPS",
]

#: Which L0 collections the co-occurrence layer reads. Deliberately the
#: market-shape families only: ``instrument`` and ``horizon`` name WHICH symbol
#: and HOW FAR AHEAD, and a motif that encodes the symbol is an identifier, not
#: a motif -- it cannot be shared across instruments, which is the whole point.
L1_STREAMS: Tuple[str, ...] = ("geometry", "temporal", "flow", "volatility", "cross")

#: How many L1 motif names the L2 path carries.
#:
#: MEASURED CONSTRAINT, inherited rather than guessed: an 8-step path over a
#: 3-symbol alphabet produced 0.76 distinct frames per sample in
#: omen_metacognition -- an identifier. The L1 alphabet is larger than 3, so
#: the path must be SHORTER, not longer.
#:
#: 3 IS NOT A MEASURED VALUE ANY MORE. DO NOT CITE IT AS ONE. It was swept
#: 2026-09-10 on p108_aero_up/down under the SIGN-banded encoder, and read
#: 0.1266/0.1530 at 2 steps, 0.2976/0.2962 at 3 and 0.4520/0.4159 at 4 -- so 3
#: was chosen on a 0.8% margin against the 0.30 identifier guard. That sweep is
#: STALE: `relative_bands` (7d2a74e) took the L1 vocabulary from 21/25 motifs
#: to 100/125 on the same two corpora, and a path over a five-times larger
#: alphabet is five times closer to being an identifier. A margin of 0.8% does
#: not survive that, and the operator measured L2 failing the guard at EVERY
#: step count under relative banding -- 0.64/0.83 DOWN and 0.51/0.79 UP, still
#: failing at 0.32 over a deliberately coarse 21-symbol alphabet.
#:
#: The value is left at 3 deliberately rather than re-swept: under relative
#: banding NO step count passes, so there is no number to move it to. L2 is a
#: DESIGN problem, not a parameter one -- a motif changes on 72.3% of bars, so
#: a fixed-length path is near-unique by construction. Keying on motif
#: TRANSITIONS, or run-length encoding a motif until it changes, is the open
#: question ([fa75fa1a]). Until that lands, 3 is the sign-banded default and
#: nothing should train on L2 under relative banding at any step count.
#:
#: The inherited 8-step constraint above still stands and is independent of
#: all this: longer is always worse here.
MOTIF_SEQUENCE_STEPS = 3

#: Bands an L0 stream is bucketed into for the co-occurrence pattern. Three,
#: not more: the motif's job is to say WHICH streams are extreme together, and
#: a finer scale re-introduces the near-uniqueness the layer exists to remove.
_BANDS = ("lo", "mid", "hi")


def _band_of(frame: Optional[str]) -> str:
    """Reduce one L0 frame to lo/mid/hi, or 'na'.

    L0 frames are token strings like ``rmv z6=u12 z24=d8 rngv=r14``. The signed
    bucket tokens already carry direction and magnitude, so the band is read
    off the token letters rather than by re-deriving the underlying float --
    which would mean recomputing what the frame builder already computed.
    """
    if not frame or not isinstance(frame, str):
        return "na"

    up = down = 0
    for token in frame.split():
        _, _, value = token.partition("=")
        if not value or value == "na":
            continue
        head = value[0]
        if head == "u":
            up += 1
        elif head == "d":
            down += 1
        elif head == "r":
            # A positive ratio bucket: r12 is the neutral centre by
            # construction in _bucket_ratio, so read either side of it.
            try:
                level = int(value[1:])
            except ValueError:
                continue
            if level >= 16:
                up += 1
            elif level <= 8:
                down += 1
        elif head == "q":
            # A QUANTILE bucket, 0-19, and the reason this branch exists:
            # without it ``_band_of`` never SAW a q token, so every geometry
            # frame -- which is all q buckets (``geo p24=q5 body=q17 uw=q0``)
            # -- tied 0-0 and returned "mid" on 600 of 600 bars. Measured by
            # Gale on AERO 0004: geometry mid 600/600, so a 5-slot motif had
            # 2 live slots and the whole L1 vocabulary was 13 motifs over
            # 3000 samples. The layer was not degenerate; the encoder was
            # blind to the stream it was reading.
            #
            # The thresholds split the 0-19 range into rough thirds rather
            # than at the midpoint: a quantile is already uniform by
            # construction, so thirds give each band real occupancy, which is
            # the whole point of a band.
            try:
                level = int(value[1:])
            except ValueError:
                continue
            if level >= 13:
                up += 1
            elif level <= 6:
                down += 1

    if up == 0 and down == 0:
        return "mid"
    if up > down:
        return "hi"
    if down > up:
        return "lo"
    return "mid"


def _numeric_of(frame: Optional[str]) -> Optional[float]:
    """One comparable scalar for a frame, or None.

    The bucket tokens already carry magnitude, so this reads THEM rather than
    re-deriving anything: a signed token (u12/d8) becomes +12/-8, a ratio or
    quantile token (r14/q17) becomes its level, and the frame's score is the
    mean over its tokens. Same idea as ``_band_of``'s sign counting, but it
    keeps the MAGNITUDE, which is what makes a relative band possible.
    """
    if not frame or not isinstance(frame, str):
        return None
    values = []
    for token in str(frame).split():
        _, _, value = token.partition("=")
        if not value or value == "na":
            continue
        head, digits = value[0], value[1:]
        try:
            level = float(digits)
        except ValueError:
            continue
        if head == "u":
            values.append(level)
        elif head == "d":
            values.append(-level)
        elif head in ("r", "q"):
            values.append(level)
    if not values:
        return None
    return sum(values) / len(values)


def relative_bands(frame_sets: Sequence[Mapping[str, str]],
                   streams: Sequence[str] = None) -> Dict[str, Tuple[float, float]]:
    """Per-stream (low, high) cut points, from the corpus's OWN distribution.

    WHY THIS EXISTS, and it is a bug this module shipped. ``_band_of`` bands on
    absolute token signs, which is wrong for any stream whose tokens do not
    change sign. Measured on p108_aero_down over 500 bars:

        volatility  vol v24=u10 v168=u11 exp=r11 rng=u10   ->  hi 500/500
        flow        flw v=r7 vt=r10 bs=q9 bs24=q9          ->  mid 427/500

    Volatility is a MAGNITUDE -- it is always "positive", so three ``u`` tokens
    every bar is by construction, not by market state, and no re-banding of
    signs can fix it. Flow's quantiles sit near the middle and its ratios
    straddle the neutral centre, so it reads mid 85% of the time.

    Gale's census on the same corpus: 3 live slots of 5. A motif with two dead
    slots has a vocabulary two slots smaller than it appears, and every
    held-out number measured on it was measured through a partly blind
    encoder.

    The fix is to ask what is HIGH FOR THIS STREAM rather than what is
    positive: terciles of the stream's own scores over the corpus. Returned as
    cut points so a train window can compute them once and a held-out window
    can reuse them WITHOUT refitting -- refitting on the test window would leak
    the test distribution into the frame, which is the same class of error as
    fitting a motif map in-sample and reading its lift as an edge.
    """
    names = list(streams) if streams is not None else list(L1_STREAMS)
    out: Dict[str, Tuple[float, float]] = {}
    for name in names:
        scores = sorted(v for v in
                        (_numeric_of(frames.get(name)) for frames in frame_sets)
                        if v is not None)
        if len(scores) < 3:
            continue
        lo = scores[len(scores) // 3]
        hi = scores[(2 * len(scores)) // 3]
        if lo == hi:
            # A stream that is genuinely constant has no terciles. Say so by
            # omitting it rather than inventing cut points that band nothing.
            continue
        out[name] = (lo, hi)
    return out


def cooccurrence_motif(frames: Mapping[str, str],
                       streams: Sequence[str] = L1_STREAMS,
                       bands: Optional[Mapping[str, Tuple[float, float]]] = None) -> str:
    """L1: which variables are doing something together, right now.

    Not a re-encoding of the bar -- a statement about WHICH L0 streams are
    simultaneously extreme and in which direction. Many instants share one
    motif, which is exactly what a flat sensory conjunction cannot do.

    ``bands`` are per-stream cut points from ``relative_bands``, fitted on the
    TRAIN window and passed in unchanged for the held-out window. Supply them:
    without them a stream whose tokens never change sign lands in one band
    every bar and its slot is dead, which measured 3 live slots of 5 on
    p108_aero_down. Absolute sign banding is kept as the fallback so a caller
    with no corpus still gets a motif rather than an exception.

    Returns a byte-disjoint token: the ``co1`` prefix cannot appear inside an
    L2 name (``co2``) or an omen label, which matters on a substrate whose
    atoms are bytes and where "loss_big" once swallowed "loss".
    """
    parts = []
    for name in streams:
        frame = frames.get(name)
        cuts = (bands or {}).get(name)
        if cuts is not None:
            score = _numeric_of(frame)
            if score is None:
                band = "na"
            elif score <= cuts[0]:
                band = "lo"
            elif score >= cuts[1]:
                band = "hi"
            else:
                band = "mid"
        else:
            band = _band_of(frame)
        parts.append("%s=%s" % (name[:3], band))
    return "co1 " + " ".join(parts)


def sticky_motifs(frame_sets: Sequence[Mapping[str, str]],
                  bands: Optional[Mapping[str, Tuple[float, float]]],
                  margin: float) -> list:
    """L1 motifs with HYSTERESIS: a slot holds its band until it is pushed out.

    THE MEASUREMENT THAT SENT THIS HERE. Neither L2 scheme can work while L1
    changes on 73.1% of bars (DOWN) and 74.0% (UP) -- dropping repeats can only
    remove the ~27% that ARE repeats, so any ordered pair of recent motifs is
    near-unique by construction. The constraint is upstream, so the fix is
    upstream: stop the slot flickering across a tercile boundary.

    ``margin`` is a fraction of the band's own width (hi - lo), so it is in the
    stream's units rather than in absolute score units -- the same knob means
    the same thing on a stream whose scores span 0.01 and one whose scores span
    400. A slot already in ``lo`` stays there until the score climbs past
    ``lo + margin*width``; a slot in ``mid`` needs ``lo - margin*width`` to fall
    into ``lo``.

    margin=0 IS BYTE-IDENTICAL to ``[cooccurrence_motif(f, bands=bands) for f
    in frame_sets]``, including the fallback for a stream ``relative_bands``
    omitted, and that is pinned by
    tests/test_hysteresis_margin_zero_is_byte_identical.py. It has to be: a
    comparison arm on a SIMILAR encoder is a two-change measurement and says
    nothing about either change.

    MEASURED by Gale pass 111, 600 samples per corpus, both computed in one
    process (scripts/omen_l2_scheme_probe.py, 7b67091/2d1d91b):

        margin  change rate DOWN/UP   L2_transitions steps=2 DOWN/UP
        0.00    73.1% / 74.0%         0.6017 / 0.4917   FAIL
        0.25    57.9% / 57.4%         0.5083 / 0.4050   FAIL
        0.50    37.6% / 38.2%         0.2633 / 0.2000   PASS
        1.00    18.9% / 17.9%         0.1483 / 0.1017   PASS

    So an ORDER-CARRYING L2 under the 0.30 ceiling in both windows exists, and
    it needed an L1 change rather than another L2 scheme.

    THE COST, stated because distinctness alone cannot see it: L1 itself
    coarsens (0.1983 -> 0.0817 DOWN, vocabulary 119 -> 49; 0.1383 -> 0.0683 UP,
    83 -> 41). Whether that coarser L1 still carries LABEL SKEW is a separate
    measurement -- ``omen_layer_probe``'s skew test -- and must be made before
    a node arm is spent. A layer that abstracts perfectly and predicts nothing
    is still worthless.

    THE SEQUENCE IS THE INPUT, not one frame: hysteresis is a fact about a
    stream of bars, so this takes the whole corpus and returns one motif per
    bar, in order. ``bands`` must be the TRAIN window's cut points, reused
    unchanged here -- refitting on a held-out window leaks its distribution
    into the frame.
    """
    bands = bands or {}
    scores = {name: [_numeric_of(frames.get(name)) for frames in frame_sets]
              for name in L1_STREAMS}
    columns: Dict[str, list] = {}
    for name in L1_STREAMS:
        cuts = bands.get(name)
        if cuts is None:
            # relative_bands OMITS a stream whose terciles collapse, and
            # cooccurrence_motif falls back to absolute sign banding for it.
            # Emitting "na" here instead would make margin=0 a DIFFERENT
            # encoder from the comparison arm -- which is the whole reason the
            # byte-identity test exists.
            columns[name] = [_band_of(frames.get(name)) for frames in frame_sets]
            continue
        low, high = cuts
        reach = margin * (high - low)
        held: Optional[str] = None
        out: list = []
        for score in scores[name]:
            if score is None:
                out.append("na")
                continue
            if held == "lo":
                band = "lo" if score <= low + reach else (
                    "hi" if score >= high else "mid")
            elif held == "hi":
                band = "hi" if score >= high - reach else (
                    "lo" if score <= low else "mid")
            else:
                band = "lo" if score <= low - reach else (
                    "hi" if score >= high + reach else "mid")
            held = band
            out.append(band)
        columns[name] = out
    return ["co1 " + " ".join("%s=%s" % (name[:3], columns[name][i])
                              for name in L1_STREAMS)
            for i in range(len(frame_sets))]


def sequence_motif(motifs: Sequence[str],
                   steps: int = MOTIF_SEQUENCE_STEPS) -> str:
    """L2: which motifs, in which ORDER, over the last ``steps`` bars.

    ``motifs`` is oldest-first and must END at the bar being decided on.
    Each L1 motif is compressed to its band pattern -- the stream names are
    fixed and identical in every motif, so repeating them would multiply the
    frame's length without adding information.

    A repeat count is carried separately, so "the same motif four times" and
    "four different motifs" are different frames even when the last one
    matches -- persistence is a fact about a sequence and a bare path loses it.
    """
    if not motifs:
        return "co2 path=na rep=na"

    def _compact(motif: str) -> str:
        bands = []
        for token in str(motif).split()[1:]:      # skip the "co1" prefix
            _, _, band = token.partition("=")
            bands.append({"lo": "l", "mid": "m", "hi": "h"}.get(band, "x"))
        return "".join(bands) or "x"

    recent = [_compact(m) for m in motifs[-steps:]]
    path = "|".join(recent)

    last = recent[-1]
    repeat = 0
    for item in reversed(recent):
        if item != last:
            break
        repeat += 1

    return "co2 path=%s rep=%d" % (path, repeat)


def layer_distinctness(frame_sets: Sequence[Mapping[str, str]],
                       keys: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """``distinct frames / samples`` per key. The abstraction test.

    THE NUMBER THAT DECIDES WHETHER A LAYER EARNS ITS PLACE. Distinctness must
    FALL layer over layer: L0 runs 0.4-0.96 on this corpus, and a layer above
    it that does not come in materially lower has abstracted nothing -- it is a
    lossy copy costing a consolidation and a query per sample, and it should be
    cut rather than kept for elegance.

    Deliberately the same arithmetic as ``omen_brain.collection_distinctness``
    so the two are comparable in one table. A key at 1.0 names every sample
    uniquely; one at 1/n is a constant and carries nothing.
    """
    total = len(frame_sets)
    if total <= 0:
        return {}
    names = list(keys) if keys is not None else sorted(
        {k for frames in frame_sets for k in frames})
    out: Dict[str, float] = {}
    for name in names:
        seen = {frames[name] for frames in frame_sets if name in frames}
        out[name] = len(seen) / total
    return out
