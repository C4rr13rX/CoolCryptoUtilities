#!/usr/bin/env python3
"""
Metacognition and temporal-structure frames for the omen brain.

WHY THIS EXISTS. Every pool the omen brain has is a view of ONE INSTANT, and
every temporal fact it holds -- ret6, ret24, vol24 -- is a scalar computed
outside and handed in already flattened. So the fabric never sees a sequence,
only a number that happens to summarise one. A trend is an ordered relation
between instants, and nothing in the topology was carrying order.

These frames fill five `Internal` pools (15-19 of
``market_predictor_v4_meta.identity.toml``). `Internal` is the substrate's own
category for "binding, integration, future composite layers", and
``crates/brain/src/pool.rs:683`` describes internal frames as re-stimulating
semantic atoms grounded by other pools. That is what makes these different in
kind from pools 12-14: those were added as relations but declared
``SensoryInput``, so they became three more views of the same instant rather
than a statement about the other pools.

    15 self_outcome       what I predicted, and whether it was right
    16 self_agreement     do my own views agree with each other
    17 temporal_sequence  the ORDER of the last N moves, not their sizes
    18 temporal_scale     the same structure read at several scales at once
    19 self_error_run     how long I have been wrong, and in which direction

WHAT IS DELIBERATELY NOT HERE.

*Confidence.* Measured on this substrate, confidence separates right from
wrong by +0.030 on train recall and -0.002 held out -- i.e. nothing. Agreement
across query sets separates them 99.4% to 73.3%. So pool 16 carries agreement
and no pool carries confidence.

*Anything the caller can compute.* The chained stage-1 regime was the worst
stream ever measured here: 4 distinct values over 2725 samples, reproduced at
73.3%, at 0.98 confidence when wrong -- for a value that is a deterministic
function of the bars. These frames are all computed, never predicted.

*Near-unique frames.* Recall and generalisation are optimised by opposite
things: a frame unique to each sample maximises reproduction and is exactly
what cannot generalise. Every field here is bucketed, and pool 19's frame is
deliberately coarse.

THE FEEDBACK TRAP, AND WHY THIS IS NOT IT. Pools 15/16/19 describe the
brain's own past. That is legitimate only when the past is SETTLED: a frame
may describe predictions whose outcomes are already known, never the
prediction currently being made. Feeding a live prediction back as its own
input is the prediction_error feedback loop that took recall from 100% to 30%
on this substrate, and it stays off. ``self_frames`` therefore takes a history
of RESOLVED predictions and refuses to look at the open one --
``resolved_only=True`` is not a convenience flag, it is the guard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Resolved",
    "temporal_frames",
    "self_frames",
    "metacognition_frames",
    "POOL_SELF_OUTCOME",
    "POOL_SELF_AGREEMENT",
    "POOL_TEMPORAL_SEQUENCE",
    "POOL_TEMPORAL_SCALE",
    "POOL_SELF_ERROR_RUN",
]

POOL_SELF_OUTCOME = 15
POOL_SELF_AGREEMENT = 16
POOL_TEMPORAL_SEQUENCE = 17
POOL_TEMPORAL_SCALE = 18
POOL_SELF_ERROR_RUN = 19

#: Scales read together in ``temporal_scale``, in bars. Chosen to span a
#: decade and a half without adjacent redundancy: 3 and 6 would agree almost
#: always and cost a field for nothing.
SCALES: Tuple[int, ...] = (3, 12, 48)

#: How many recent moves ``temporal_sequence`` encodes as an ordered path.
#:
#: MEASURED, and the first value was wrong. Eight steps over a 3-symbol
#: alphabet is 3**8 = 6561 shapes, and on 2000 random walks that produced
#: 1512 DISTINCT frames -- 0.76 per sample, i.e. an identifier. A frame that
#: near-unique maximises train recall and is exactly what cannot generalise,
#: which is the trap this substrate punishes hardest.
#:
#: Five steps is 3**5 = 243 shapes. On the same 2000 walks that lands around
#: 0.12 per sample: coarse enough to be shared by many samples, long enough
#: that a climb and a spike-then-drift still differ (the whole point of the
#: pool). The run-length field carries the longer-horizon persistence that
#: the shortened path gives up.
SEQUENCE_STEPS = 5

#: Dead-band for calling a step flat, as a fraction of the step's own recent
#: noise. Without it every step is up or down and "flat" never appears, which
#: throws away the distinction between a drift and a chop.
FLAT_Z = 0.25


def _fin(value: Any) -> Optional[float]:
    """float(value) if it is a usable number, else None. Never raises."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _step_token(z: Optional[float]) -> str:
    """One ordered step: up, down, flat, or unknown.

    Single characters so the path stays short. Byte-disjoint from each other,
    which matters on a substrate whose atoms are bytes -- see the LABEL_TOKEN
    lesson where "loss_big" contained "loss" and the frequent class swallowed
    the rare one.
    """
    if z is None:
        return "x"
    if z > FLAT_Z:
        return "u"
    if z < -FLAT_Z:
        return "d"
    return "f"


#: Magnitude edges for ``_bucket_z``, in units of the move's own noise. Four
#: bands, so a scale reads "which way, and is that a nudge, a move, or a
#: dislocation". Coarse on purpose -- see the comment in ``temporal_frames``.
Z_EDGES: Tuple[float, ...] = (0.5, 1.5, 3.0)


def _bucket_z(z: Optional[float]) -> str:
    """How far, in units of the move's own noise, as one digit.

    Magnitude only: the sign already rides on the direction token beside it,
    so encoding it twice would waste resolution on a fact the frame has.
    """
    if z is None:
        return "x"
    size = abs(z)
    for level, edge in enumerate(Z_EDGES):
        if size < edge:
            return str(level)
    return str(len(Z_EDGES))


def _run_length(tokens: Sequence[str]) -> Tuple[str, int]:
    """(token, how many times it repeats at the END of the sequence)."""
    if not tokens:
        return ("x", 0)
    last = tokens[-1]
    n = 0
    for tok in reversed(tokens):
        if tok != last:
            break
        n += 1
    return (last, n)


def _bucket_count(n: int) -> str:
    """Coarse run-length bucket. A run of 9 and a run of 11 are the same fact."""
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2to3"
    if n <= 7:
        return "4to7"
    if n <= 15:
        return "8to15"
    return "16plus"


def _bucket_frac(value: Optional[float], levels: int = 8) -> str:
    """Bucket a 0..1 fraction into ``levels`` bands. 'na' when unknown."""
    if value is None:
        return "na"
    value = max(0.0, min(1.0, value))
    idx = min(levels - 1, int(value * levels))
    return "q%d" % idx


# --------------------------------------------------------------------------
# temporal structure -- pools 17 and 18
# --------------------------------------------------------------------------

def temporal_frames(closes: Sequence[float],
                    noise: Optional[float] = None) -> dict:
    """ORDER and SCALE, from a close series ending at the decision bar.

    ``closes`` is oldest-first and must END at the bar being decided on; it is
    read backwards. ``noise`` is a per-step stdev used to size the flat band --
    when absent it is estimated from the series itself, because a fixed
    percentage dead-band means different things on a stablecoin and a memecoin.

    Returns frames for ``temporal_sequence`` and ``temporal_scale``. Both are
    computed, never predicted.
    """
    series = [c for c in (_fin(c) for c in closes) if c is not None and c > 0]

    if len(series) < 2:
        return {
            "temporal_sequence": "seq path=na run=na",
            "temporal_scale": "scl " + " ".join("s%d=na" % s for s in SCALES) + " agree=na",
        }

    steps = [(series[i + 1] - series[i]) / series[i] for i in range(len(series) - 1)]

    unit = _fin(noise)
    if unit is None or unit <= 0:
        unit = _stdev(steps)
    if not unit or unit <= 0:
        # A perfectly flat series has no noise to normalise by. Every step is
        # flat, which is the honest reading -- not a division by zero.
        unit = None

    tokens = [_step_token(None if unit is None else (s / unit)) for s in steps]
    path = "".join(tokens[-SEQUENCE_STEPS:])
    token, run = _run_length(tokens[-SEQUENCE_STEPS:])
    sequence = "seq path=%s run=%s%s" % (path or "na", token, _bucket_count(run))

    # The same question at several horizons: which way, and HOW FAR, over
    # this many bars?
    #
    # THE MAGNITUDE IS NOT DECORATION. Measured pass 109 on AERO-USDC over 600
    # samples: a direction-token-only frame ("s3=u s12=u s48=d agree=split")
    # scored 0.045 distinct frames per sample against the dilution law's 0.20
    # bar, so ``discriminating_collections`` excluded pool 18 from the query
    # set and the ONE pool aimed at multi-scale regime never fired. Three
    # ternary tokens plus an agreement word cannot say more than a few dozen
    # things, and a stream that coarse votes for the label DISTRIBUTION over
    # everything it matches rather than for a label.
    #
    # The band it has to clear is 0.103-0.260 (the empty band), and the ceiling
    # it must not approach is near-uniqueness: SEQUENCE_STEPS was cut from 8 to
    # 5 for scoring 0.76 distinct frames per sample, which is an IDENTIFIER and
    # maximises recall at the cost of the generalisation we are actually after.
    # So magnitude is added at a DELIBERATELY COARSE resolution -- a signed
    # z-bucket per scale -- rather than a raw number.
    scale_bits, directions = [], []
    for span in SCALES:
        if len(series) <= span:
            scale_bits.append("s%d=na" % span)
            continue
        ret = (series[-1] - series[-1 - span]) / series[-1 - span]
        # Noise over n steps scales as unit*sqrt(n) for independent steps.
        z = None if unit is None else ret / (unit * math.sqrt(span))
        tok = _step_token(z)
        directions.append(tok)
        scale_bits.append("s%d=%s%s" % (span, tok, _bucket_z(z)))

    known = [d for d in directions if d != "x"]
    if not known:
        agree = "na"
    elif all(d == known[0] for d in known):
        agree = "all" + known[0]
    elif "u" in known and "d" in known:
        # The pullback case: short and long disagree in sign. This is the
        # single most useful thing this pool can say, so it gets its own token
        # rather than being folded into "mixed".
        agree = "split"
    else:
        agree = "mixed"

    return {
        "temporal_sequence": sequence,
        "temporal_scale": "scl " + " ".join(scale_bits) + " agree=" + agree,
    }


def _stdev(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in (_fin(v) for v in values) if v is not None]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var) if var > 0 else None


# --------------------------------------------------------------------------
# the brain's own record -- pools 15, 16 and 19
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Resolved:
    """One SETTLED prediction: what was said, and what happened.

    ``resolved`` exists so an open prediction cannot reach a frame by
    accident. ``predicted`` and ``actual`` are label tokens (the byte-disjoint
    omen vocabulary), ``agreed`` is how many query sets voted the same way and
    ``asked`` how many were asked -- the agreement signal that was measured to
    separate right from wrong where confidence does not.
    """
    predicted: str
    actual: Optional[str] = None
    agreed: Optional[int] = None
    asked: Optional[int] = None
    resolved: bool = False

    @property
    def correct(self) -> Optional[bool]:
        if not self.resolved or self.actual is None:
            return None
        return self.predicted == self.actual


def self_frames(history: Iterable[Resolved],
                *,
                window: int = 32,
                resolved_only: bool = True) -> dict:
    """The brain's own track record, as three frames.

    ``history`` is oldest-first. **Only settled predictions may be read**: an
    unresolved entry describes an outcome that has not happened, and feeding
    that back is the prediction_error loop that took recall from 100% to 30%
    on this substrate. ``resolved_only=False`` exists solely so a test can
    prove the guard bites; production must never set it.
    """
    rows = list(history)
    if resolved_only:
        rows = [r for r in rows if r.resolved and r.correct is not None]
    rows = rows[-window:] if window and window > 0 else rows

    if not rows:
        return {
            "self_outcome": "slf hit=na n=0 last=na",
            "self_agreement": "agr unan=na rate=na",
            "self_error_run": "err run=na dir=na",
        }

    hits = [1 if r.correct else 0 for r in rows]
    hit_rate = sum(hits) / len(hits)
    last = "hit" if hits[-1] else "miss"

    outcome = "slf hit=%s n=%s last=%s" % (
        _bucket_frac(hit_rate), _bucket_count(len(rows)), last)

    # Agreement: how often the query sets were unanimous, and how often
    # unanimity coincided with being right. Both bucketed -- a rate is a
    # tendency, not an identifier.
    unanimous = [r for r in rows
                 if r.agreed is not None and r.asked and r.agreed == r.asked]
    if unanimous:
        unan_share = len(unanimous) / len(rows)
        unan_hit = sum(1 for r in unanimous if r.correct) / len(unanimous)
        agreement = "agr unan=%s rate=%s" % (
            _bucket_frac(unan_share), _bucket_frac(unan_hit))
    else:
        agreement = "agr unan=%s rate=na" % _bucket_frac(0.0)

    # The error RUN: not "was I right last time" but how long the current
    # streak has lasted. A model wrong the same way for many bars is in a
    # regime it does not model, which is knowable from its own output alone.
    streak_tokens = ["h" if h else "m" for h in hits]
    token, run = _run_length(streak_tokens)
    if token == "m" and run >= 1:
        # Which way is it wrong? Only meaningful while the streak is misses.
        wrong = [r.predicted for r in rows[-run:]]
        direction = wrong[-1] if all(w == wrong[0] for w in wrong) else "mixed"
    else:
        direction = "na"
    error_run = "err run=%s%s dir=%s" % (token, _bucket_count(run), direction)

    return {
        "self_outcome": outcome,
        "self_agreement": agreement,
        "self_error_run": error_run,
    }


def metacognition_frames(closes: Sequence[float],
                         history: Iterable[Resolved],
                         *,
                         noise: Optional[float] = None,
                         window: int = 32) -> dict:
    """All five frames. The one call a training or predict path needs."""
    frames = temporal_frames(closes, noise=noise)
    frames.update(self_frames(history, window=window))
    return frames
