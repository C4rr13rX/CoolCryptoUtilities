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
#: SWEPT 2026-09-10 on p108_aero_up/down, 719 samples each, AFTER the _band_of
#: q-token fix -- which matters, because the same sweep against the blind
#: encoder read 0.177 at 4 steps and would have justified keeping it:
#:
#:   steps   L2 distinctness UP / DOWN   verdict
#:     2         0.1266 / 0.1530         comfortably under the guard
#:     3         0.2976 / 0.2962         passes, margin 0.8%
#:     4         0.4520 / 0.4159         FAILS the 0.30 identifier guard
#:
#: Set to 3: it passes and it carries more order than 2. THE MARGIN IS THIN
#: AND THAT IS NOT A ROUNDING DETAIL -- a corpus with a richer L1 vocabulary
#: will push 3 over the line too, and the failure mode is the expensive one
#: (maximises train recall, destroys generalisation). Re-run
#: scripts/omen_layer_probe.py on any new corpus before trusting 3, and drop
#: to 2 rather than arguing with the number.
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


def cooccurrence_motif(frames: Mapping[str, str],
                       streams: Sequence[str] = L1_STREAMS) -> str:
    """L1: which variables are doing something together, right now.

    Not a re-encoding of the bar -- a statement about WHICH L0 streams are
    simultaneously extreme and in which direction. Many instants share one
    motif, which is exactly what a flat sensory conjunction cannot do.

    Returns a byte-disjoint token: the ``co1`` prefix cannot appear inside an
    L2 name (``co2``) or an omen label, which matters on a substrate whose
    atoms are bytes and where "loss_big" once swallowed "loss".
    """
    parts = []
    for name in streams:
        parts.append("%s=%s" % (name[:3], _band_of(frames.get(name))))
    return "co1 " + " ".join(parts)


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
