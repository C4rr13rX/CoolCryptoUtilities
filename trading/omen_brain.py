"""Omens — buy-low / sell-high forecasts from the W1z4rD substrate.

An **omen** is a claim made at bar ``t`` about bar ``t + horizon``:

    trough  buy low   — price sits in the LOW part of its recent range AND
                        the forward move clears the round-trip cost upward
    crest   sell high — price sits in the HIGH part of its recent range AND
                        the forward move clears the round-trip cost downward
    climb   forward move clears cost upward, but not from a low
    slide   forward move clears cost downward, but not from a high
    murk    the forward move does NOT clear the round-trip cost — untradeable

``murk`` is the point of the whole label set. A forecast compared against
zero is the losing shape this repo has shipped more than once (see
``services/profit_logic_audit``); every omen boundary here is drawn at
``ROUND_TRIP_COST`` — the measured median ``fee_cost / notional``, imported
from ``services.symbol_edge_gate`` so it cannot drift from what the books
actually charge. A move that does not pay for its own round trip is not a
weak buy signal, it is *not a signal*.

Specialised collections
-----------------------
The substrate has no tokeniser: atoms are bytes. Flattening every feature
into one frame lets a frequent family's binding mass swallow a rare one.
``brains/market_predictor_v2.identity.toml`` separates the causal families
into their own pools, and this module fires them **together** inside one
learning moment via ``/brain/consolidate/multi``:

    geometry    pool 1   where price sits in its own recent range
    temporal    pool 2   how the returns MUTATED — the temporal pool
    flow        pool 3   volume, and the buy/sell split inside it
    volatility  pool 4   realised range, expanding or contracting
    regime      pool 5   the chained stage-1 output (see below)
    cross       pool 6   the symbol against its own longer baseline
    horizon     pool 9   how far ahead the question is being asked
    instrument  pool 10  which symbol and chain is being asked about
    omen        pool 11  the ACTION pool — what a prediction decodes from

Every collection carries its own byte prefix (``geo`` / ``tmp`` / ...), so
no collection's atoms can be mistaken for another's.

Chaining
--------
Two consolidations per training sample:

    stage 1   geometry+temporal+flow+volatility  ->  regime token (pool 5)
    stage 2   all of the above + the regime frame ->  omen token (pool 11)

At inference stage 1's *predicted* regime is what stage 2 is given, so the
chosen pool integrates the others rather than seeing them flat. Stage 2 is
the only pool a caller reads.

Producing perfectly, before producing profitably
------------------------------------------------
The user's standing ask is that this **produce** correctly first. Three
guards, all of them written because this repo has already been fooled once:

1. **Byte-disjoint labels.** ``trough``/``crest``/``climb``/``slide``/
   ``murk`` — no token is a substring of another. The 2026-07 corpus run
   found every recall miss was ``loss_big`` decoding as ``loss``; disjoint
   tokens took train recall 96% -> 100%.
2. **Degeneracy is a verdict, not a prediction.** A brain that answers the
   same token for every input scores well whenever the market trends. If
   the recent answer stream collapses to one token the omen is emitted with
   ``verdict="degenerate"`` and ``action="hold"``.
3. **A missing signal is never an ambiguous silence.** ``verdict`` is always
   one of ``admitted | below_floor | no_answer | degenerate |
   unsupported_horizon | not_trained``, mirroring ``regime_derived_state``.

Nothing in this module places or sizes a trade. It answers a question.
"""
from __future__ import annotations

import base64
import json
import math
import os
import statistics
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from http.client import HTTPConnection, BadStatusLine, RemoteDisconnected
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from trading.brain_bridge import resolve_node_endpoint

try:  # the cost bar is the books', not ours
    from services.symbol_edge_gate import ROUND_TRIP_COST
except Exception:  # pragma: no cover - import-order safety only
    ROUND_TRIP_COST = float(os.getenv("SYMBOL_EDGE_ROUND_TRIP_COST", "0.0065"))


SCHEMA_VERSION = "omen.v1"

# --- labels ---------------------------------------------------------------
#: The five omens. Equal-ish length and pairwise non-containing so a byte
#: decode of one can never be read as another -- see module docstring.
OMEN_TROUGH = "trough"
OMEN_CREST = "crest"
OMEN_CLIMB = "climb"
OMEN_SLIDE = "slide"
OMEN_MURK = "murk"

OMEN_LABELS: Tuple[str, ...] = (
    OMEN_TROUGH, OMEN_CREST, OMEN_CLIMB, OMEN_SLIDE, OMEN_MURK,
)

#: What a caller is allowed to do with each omen. Derived, never free-form:
#: a strategy reads ``action``, so a new label cannot silently mean "buy".
OMEN_ACTIONS: Dict[str, str] = {
    OMEN_TROUGH: "buy",
    OMEN_CREST: "sell",
    OMEN_CLIMB: "hold",   # up-move already under way; not a buy-LOW
    OMEN_SLIDE: "hold",   # down-move already under way; not a sell-HIGH
    OMEN_MURK: "hold",
}

#: Sign of the move each omen claims, for expected-move reporting.
OMEN_DIRECTION: Dict[str, int] = {
    OMEN_TROUGH: +1, OMEN_CLIMB: +1,
    OMEN_CREST: -1, OMEN_SLIDE: -1,
    OMEN_MURK: 0,
}


def omen_frame(label: str) -> str:
    """The exact bytes the action pool is trained on / decoded from."""
    return f"omen {label}"


def parse_omen(answer: Optional[str]) -> Optional[str]:
    """Map a decoded action frame back to a canonical label, or None.

    Longest-first so a shorter token can never shadow a longer one even if
    the label set later grows a containing pair.
    """
    value = (answer or "").lower()
    for label in sorted(OMEN_LABELS, key=len, reverse=True):
        if label in value:
            return label
    return None


# --- collections ----------------------------------------------------------

@dataclass(frozen=True)
class Collection:
    """One specialised sensory family: a name, a byte prefix, and a pool."""

    name: str
    prefix: str
    pool_id: int


def _pool(env_name: str, default: int) -> int:
    try:
        return int(os.getenv(env_name, str(default)))
    except (TypeError, ValueError):
        return default


#: Pool ids follow brains/market_predictor_v2.identity.toml. Env overrides
#: exist so a differently-configured node can be pointed at without a code
#: change -- the frames are prefix-namespaced either way.
COLLECTIONS: Tuple[Collection, ...] = (
    Collection("geometry",   "geo", _pool("OMEN_POOL_GEOMETRY", 1)),
    Collection("temporal",   "tmp", _pool("OMEN_POOL_TEMPORAL", 2)),
    Collection("flow",       "flw", _pool("OMEN_POOL_FLOW", 3)),
    Collection("volatility", "vol", _pool("OMEN_POOL_VOLATILITY", 4)),
    Collection("cross",      "crs", _pool("OMEN_POOL_CROSS", 6)),
    Collection("horizon",    "hzn", _pool("OMEN_POOL_HORIZON", 9)),
    Collection("instrument", "ins", _pool("OMEN_POOL_INSTRUMENT", 10)),
)

#: The relation collections -- pools 12/13/14 of
#: brains/market_predictor_v3_assoc.identity.toml. OFF BY DEFAULT, and that
#: default is load-bearing rather than timid: a v2 node declares 11 pools, so
#: sending it pool 12 returns ``unknown input pool id 12`` and _consolidate
#: reports the whole sample as a MISS. Enabling these against the wrong node
#: does not degrade training, it silently stops it. Turn on only when the
#: node was started with the v3_assoc identity:
#:   OMEN_RELATION_COLLECTIONS=1
RELATION_COLLECTIONS: Tuple[Collection, ...] = (
    Collection("rel_move_vol",    "rmv", _pool("OMEN_POOL_REL_MOVE_VOL", 12)),
    Collection("rel_shape_flow",  "rsf", _pool("OMEN_POOL_REL_SHAPE_FLOW", 13)),
    Collection("rel_trend_noise", "rtn", _pool("OMEN_POOL_REL_TREND_NOISE", 14)),
)
RELATIONS_ENABLED: bool = os.getenv(
    "OMEN_RELATION_COLLECTIONS", "0") not in ("0", "", "false", "False", "no")
if RELATIONS_ENABLED:
    COLLECTIONS = COLLECTIONS + RELATION_COLLECTIONS

#: The metacognition and temporal-structure collections -- pools 15-19 of
#: brains/market_predictor_v4_meta.identity.toml, and the first pools here
#: declared ``kind="Internal"`` rather than ``SensoryInput``. The frames come
#: from ``trading/omen_metacognition.py``; the reasoning for each is there.
#:
#: WHAT THESE ADD THAT NOTHING ELSE DOES. Every collection above is a view of
#: ONE INSTANT: ret6, ret24 and vol24 are scalars computed outside and handed
#: in already flattened, so the fabric sees a number that summarises a
#: sequence and never the sequence. 17 and 18 carry ORDER and SCALE. 15, 16
#: and 19 carry the brain's own settled track record, which is the abstention
#: signal that measured 99.4% vs 73.3% where confidence measured -0.002.
#:
#: OFF BY DEFAULT, and the default is load-bearing for the same reason the
#: relations' is: a v2 or v3 node returns ``unknown input pool id 15`` and
#: ``_consolidate`` reports the whole sample as a MISS, so enabling these
#: against the wrong node does not degrade training, it SILENTLY STOPS it.
#: Turn on only against a node started with the v4_meta identity:
#:   OMEN_META_COLLECTIONS=1
META_COLLECTIONS: Tuple[Collection, ...] = (
    Collection("self_outcome",      "slf", _pool("OMEN_POOL_SELF_OUTCOME", 15)),
    Collection("self_agreement",    "agr", _pool("OMEN_POOL_SELF_AGREEMENT", 16)),
    Collection("temporal_sequence", "seq", _pool("OMEN_POOL_TEMPORAL_SEQUENCE", 17)),
    Collection("temporal_scale",    "scl", _pool("OMEN_POOL_TEMPORAL_SCALE", 18)),
    Collection("self_error_run",    "err", _pool("OMEN_POOL_SELF_ERROR_RUN", 19)),
)
META_ENABLED: bool = os.getenv(
    "OMEN_META_COLLECTIONS", "0") not in ("0", "", "false", "False", "no")
if META_ENABLED:
    COLLECTIONS = COLLECTIONS + META_COLLECTIONS

#: The SHAPE-CLASS collection -- pool 7 of the ALREADY-SHIPPED
#: ``market_predictor_v2.identity.toml``. It needs no new identity file and no
#: new node build, because pool 7 (``news_entities``) is declared, fully
#: knobbed (recent_atoms_window 65536, max_concept_member_count 24,
#: decay_rate 0.00002, prune_floor 0.001) and has never been fed by any
#: client. docs/BRAIN_POOL_TOPOLOGY.md counted it as dead; this spends it.
#:
#: WHY IT EXISTS, and why it is NOT a PoolKind::Internal pool. The engine
#: matches ``PoolKind::Internal`` NOWHERE -- ``grep -rn 'PoolKind::Internal'``
#: over the W1z4rDV1510n crates returns zero, and the only behavioural match
#: on a pool kind anywhere is ``brain.rs:7425 matches!(ps.kind,
#: PoolKind::Action)``. Declaring a pool Internal is a naming convention. A
#: relation becomes a first-class bindable thing by being SENT, not by being
#: declared, so this is a SensoryInput pool carrying a CLIENT-COMPUTED
#: relation -- the safe side of the trap "never make the substrate guess what
#: the caller can compute".
#:
#: WHAT IT CARRIES. A canonical shape key over the DEEP PREFIX, built to be
#: byte-IDENTICAL for a base frame and for its label-safe mutations. The
#: shape-mutation arm inverted in pass 117 (held-out 0.3400->0.2950 UP,
#: 0.2825->0.2375 DOWN) because a mutation lands as another near-unique key,
#: so more pairs is more memorisation. A key that COLLIDES across a base and
#: its mutants is the one thing that turns those extra pairs into evidence
#: for a shape instead of evidence for an instant.
#:
#: OFF BY DEFAULT for the same load-bearing reason as the others, inverted: a
#: node whose identity does NOT declare pool 7 would report the whole sample
#: as a MISS. v2, v3_assoc and v4_meta all declare it, but the default stays
#: 0 so production's frames do not change underneath it.
#:   OMEN_SHAPE_COLLECTION=1
SHAPE_COLLECTIONS: Tuple[Collection, ...] = (
    Collection("shape_class", "shp", _pool("OMEN_POOL_SHAPE_CLASS", 7)),
)
SHAPE_ENABLED: bool = os.getenv(
    "OMEN_SHAPE_COLLECTION", "0") not in ("0", "", "false", "False", "no")
if SHAPE_ENABLED:
    COLLECTIONS = COLLECTIONS + SHAPE_COLLECTIONS
COLLECTIONS_BY_NAME: Dict[str, Collection] = {c.name: c for c in COLLECTIONS}

#: The chained stage-1 target. It is an input pool at stage 2 and the
#: outcome pool at stage 1 -- that is what makes the chain a chain.
REGIME_POOL = _pool("OMEN_POOL_REGIME", 5)
#: The action pool. The only pool a prediction decodes from.
OMEN_POOL = _pool("OMEN_POOL_OUTCOME", 11)

#: Stage-1 regime tokens. Byte-disjoint from each other AND from the omen
#: labels, so a stage-2 decode can never return a stage-1 token by accident.
REGIME_TOKENS: Tuple[str, ...] = ("bullrun", "bearrun", "chop", "squeeze")

#: Which collections feed stage 1. Deliberately excludes ``instrument`` and
#: ``horizon``: the regime of a market is not a property of which symbol is
#: being asked about, or of how far ahead.
STAGE1_COLLECTIONS: Tuple[str, ...] = ("geometry", "temporal", "flow", "volatility")


# --- what a QUERY is allowed to fire --------------------------------------
# Training binds every collection. A *query* must not, and this is the single
# biggest lever on train recall that has been measured here.
#
# THE DILUTION LAW, measured 2026-09-07 against a fabric trained on 2725
# AERO-USDC pairs (read-only probe, same samples every row, n=200):
#
#   query streams                       distinct frames / 2725   recall
#   temporal + geometry + cross         2725 / 2627 / 709        96.0%
#   ... + flow                          282                      92.5%
#   ... + volatility                    166                      91.0%
#   ... + horizon + instrument          1 / 1                    91.0%
#   all seven                                                    91.5%
#   ... + regime frame (TRUE token)     4                        93.5%
#   ... + regime frame (stage-1 guess)  4                        90.5%
#
# A stream whose frame is shared by many training samples votes for the label
# *distribution* over all of them; a stream unique to one sample votes for one
# label. Fire enough of the former and they out-vote the latter. The ordering
# is monotone in distinctness, which is why the default below is DERIVED from
# a distinctness ratio rather than hand-picked -- see
# ``discriminating_collections``, which the experiment runs and reports so the
# next fabric re-measures this instead of inheriting our list.
#
# Note this is a PREDICT-side rule only. ``train`` still binds all of them:
# the information stays in the fabric, and a multi-symbol corpus makes
# ``instrument`` discriminating rather than constant.

#: Minimum ``distinct frames / samples`` for a collection to be worth firing
#: in a query. 0.20 is the empty band in the measured table above: ``cross``
#: sits at 0.26 and helps, ``flow`` at 0.10 and hurts.
MIN_QUERY_DISTINCTNESS = float(os.getenv("OMEN_MIN_QUERY_DISTINCTNESS", "0.20"))


def _names_from_env(env_name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.getenv(env_name)
    if not raw:
        return default
    picked = tuple(
        name for name in (part.strip() for part in raw.split(","))
        if name in COLLECTIONS_BY_NAME
    )
    return picked or default


#: The collections a prediction fires. Derived from the table above; override
#: with ``OMEN_PREDICT_COLLECTIONS=temporal,geometry,cross``.
PREDICT_COLLECTIONS: Tuple[str, ...] = _names_from_env(
    "OMEN_PREDICT_COLLECTIONS", ("temporal", "geometry", "cross"))

#: Query sets whose AGREEMENT is the abstention signal. Confidence is not
#: one -- measured 2026-09-07, mean confidence was 0.999 when the answer was
#: right and 0.969 when it was wrong on train recall, and 0.965 vs 0.967
#: held-out, i.e. a gap of -0.002. It cannot tell a caller anything.
#:
#: Unanimity across these four can. Same run, same fabric:
#:
#:   train recall   unanimous (85.0% of samples)  99.4%  (169/170)
#:                  split     (15.0%)             73.3%  ( 22/ 30)
#:   held-out       unanimous (19.2%)             33.3%  ( 32/ 96)
#:                  split     (80.8%)             29.7%  (120/404)
#:
#: Read that honestly: unanimity is a REPRODUCTION gate, not an edge gate. It
#: takes "produce perfectly" from 95.5% to 99.4% at the price of abstaining
#: on 15% of asks, and it does NOT make the held-out forecast good -- 33.3%
#: against a 31.2% majority class is not an edge, and the buys it admits
#: still lost 0.2516% per trade. The first member is the primary answer.
CONSENSUS_QUERIES: Tuple[Tuple[str, ...], ...] = (
    ("temporal", "geometry", "cross"),
    ("temporal", "geometry", "cross", "flow"),
    ("temporal", "geometry", "cross", "volatility"),
    ("temporal", "geometry", "cross", "horizon", "instrument"),
)

#: Whether a query also fires the stage-1 regime frame. Off by default: it is
#: the LOWEST-distinctness stream in the whole design (4 values over 2725
#: samples) and it cost 5.5 points of recall on the production path. Training
#: is unaffected -- stage 2 still binds it.
PREDICT_INCLUDE_REGIME = os.getenv("OMEN_PREDICT_INCLUDE_REGIME", "0") not in (
    "0", "", "false", "False", "no")


def collection_distinctness(
    frame_sets: Sequence[Mapping[str, str]],
) -> Dict[str, float]:
    """``distinct frames / samples`` per collection over a training set.

    The number the dilution law is stated in. A collection at 1.0 names every
    sample uniquely; one at ``1/n`` is a constant and carries nothing.
    """
    total = len(frame_sets)
    if total <= 0:
        return {}
    out: Dict[str, float] = {}
    for collection in COLLECTIONS:
        seen = {frames[collection.name] for frames in frame_sets
                if collection.name in frames}
        out[collection.name] = len(seen) / total
    return out


def discriminating_collections(
    frame_sets: Sequence[Mapping[str, str]],
    minimum: float = MIN_QUERY_DISTINCTNESS,
) -> Tuple[str, ...]:
    """Which collections a query should fire, measured from the corpus.

    Returned in ``COLLECTIONS`` order so the choice is reproducible. Falls
    back to the single most discriminating collection rather than an empty
    query, because a query with no streams cannot answer at all.
    """
    scores = collection_distinctness(frame_sets)
    picked = tuple(c.name for c in COLLECTIONS
                   if scores.get(c.name, 0.0) >= minimum)
    if picked:
        return picked
    if not scores:
        return PREDICT_COLLECTIONS
    return (max(scores, key=lambda name: scores[name]),)


# --- bucketing ------------------------------------------------------------
# Coarse buckets cap recall by collision -- the 2026-07 corpus run measured a
# per-feature-majority ceiling well below 1.0 before the buckets were made
# fine. These are deliberately fine-grained; the substrate's job is to find
# which combinations recur, not ours to pre-average them away.

def _bucket_return(value: Optional[float]) -> str:
    """Signed, log-spaced return bucket. Input is a FRACTION (0.01 = 1%)."""
    if value is None or not math.isfinite(value):
        return "na"
    sign = "u" if value > 0 else ("d" if value < 0 else "z")
    magnitude = abs(value)
    if magnitude < 1e-9:
        return "z0"
    # 0.01% .. 100%, ~10 steps per decade -> 20 usable levels.
    level = int(round(math.log10(magnitude / 0.0001) * 5.0))
    level = max(0, min(24, level))
    return f"{sign}{level}"


def _bucket_unit(value: Optional[float], levels: int = 20) -> str:
    """Bucket a value already normalised to [0, 1] into ``levels`` steps."""
    if value is None or not math.isfinite(value):
        return "na"
    clamped = max(0.0, min(1.0, value))
    return f"q{min(levels - 1, int(clamped * levels))}"


def _bucket_ratio(value: Optional[float]) -> str:
    """Bucket a positive ratio around 1.0 (volume vs its own mean, etc.)."""
    if value is None or not math.isfinite(value) or value <= 0:
        return "na"
    level = int(round(math.log10(value) * 6.0)) + 12
    return f"r{max(0, min(24, level))}"


def _bucket_signed(value: Optional[float], span: float = 2.0,
                   levels: int = 40) -> str:
    """Bucket a SIGNED, already-dimensionless quantity (a z-score).

    ``_bucket_return`` is wrong for these: it is log-spaced and calibrated for
    fractions where 0.0001 is the floor, so the band that matters for a
    z-score -- roughly 0.5 to 3 -- lands in about four adjacent levels. This
    maps ``[-span, +span]`` linearly onto ``levels`` buckets instead, so a
    z-score gets even resolution where it is actually informative.

    The 2.0/40 default is MEASURED, not guessed. Distinctness of the relation
    streams over 13219 samples (4 corpus files, horizon 12), against the 0.2
    query floor:

        span/levels   rel_move_vol  rel_shape_flow  rel_trend_noise
        4.0/20            0.089         0.119            0.030
        2.0/20            0.161         0.119            0.072
        1.0/40            0.306         0.208            0.268
        2.0/40            0.314         0.208            0.165

    At 4.0/20 all three were below the floor -- train-only streams by the
    dilution law. 1.0/40 lifts all three, but it saturates every |z| > 1,
    which throws away exactly the large moves the relation exists to flag.
    2.0/40 keeps the tail out to |z| = 2 and still clears the floor on two of
    three. rel_trend_noise stays under it either way; that stream is
    near-constant for its own reasons (slow long-baseline z-scores, plus an
    ``exp`` field duplicated from volatility) and wants redesigning rather
    than re-bucketing.
    """
    if value is None or not math.isfinite(value):
        return "na"
    unit = (value + span) / (2.0 * span)
    clamped = max(0.0, min(1.0, unit))
    return f"s{min(levels - 1, int(clamped * levels))}"


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator is None or not math.isfinite(denominator) or abs(denominator) < 1e-18:
        return None
    result = numerator / denominator
    return result if math.isfinite(result) else None


# --- bar access -----------------------------------------------------------

def _close(bar: Mapping[str, Any]) -> float:
    return float(bar["close"])


#: How far back the feature builder needs to see. A window shorter than this
#: cannot produce a full frame, and a partial frame is a different atom --
#: which is why ``build_collections`` refuses rather than padding.
LOOKBACK_BARS = 168  # 7 days of hourly bars; 168 minutes on a minute feed


def _returns(bars: Sequence[Mapping[str, Any]], index: int,
             spans: Sequence[int]) -> Dict[int, Optional[float]]:
    """Trailing returns over each span, as fractions. Never looks forward."""
    out: Dict[int, Optional[float]] = {}
    now = _close(bars[index])
    for span in spans:
        past_index = index - span
        if past_index < 0:
            out[span] = None
            continue
        out[span] = _safe_div(now - _close(bars[past_index]), _close(bars[past_index]))
    return out


RETURN_SPANS: Tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48, 168)


def measure_bar_seconds(bars: Sequence[Mapping[str, Any]],
                        default: int = 0) -> int:
    """The MODAL gap between consecutive bar timestamps, in seconds.

    A bar index is not a clock. The training corpora here are stored OHLCV at
    3600s; the live path buckets ticks at a nominal 60s and DROPS empty
    buckets, so its measured index step has run at 180s. Both call one step
    "one bar", so anything that wants wall clock must measure it rather than
    take the nominal.

    Returns ``default`` when the series carries no usable timestamps -- which
    is the case for the synthetic bars tests build.
    """
    stamps: List[int] = []
    for bar in bars:
        raw = bar.get("timestamp")
        if raw is None:
            continue
        try:
            stamps.append(int(raw))
        except (TypeError, ValueError):
            continue
    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    if not gaps:
        return int(default)
    return int(Counter(gaps).most_common(1)[0][0])


#: Widths of the two cadence slots in the horizon frame. FIXED WIDTH is not
#: cosmetic: atoms here are bytes, so a variable-width ``c=60`` is a byte
#: prefix of ``c=600`` and the substrate would see one token inside the
#: other. Zero-padding to a constant width makes every cadence token
#: byte-disjoint from every other by construction -- the same rule that
#: "loss_big contains loss" taught this repo the expensive way.
_CADENCE_DIGITS = 6   # up to 999999s, ~11.6 days per bar
_WALLCLOCK_DIGITS = 7  # up to 9999999 minutes ahead, ~19 years


def horizon_frame(horizon_bars: int, bar_seconds: int) -> str:
    """The horizon collection's frame: bar count AND the cadence it counts.

    ``hzn h=12`` alone was the same atom for two questions 60x apart -- 12
    bars of a 3600s corpus is 720 minutes trained, 12 bars of a 60s resample
    is 12 minutes asked. ``c`` is the seconds per bar and ``w`` is the
    wall-clock horizon in minutes, which is the question actually being put.
    ``w`` is redundant with ``h * c`` on purpose: the substrate should never
    be asked to multiply something the caller can compute.

    ``bar_seconds`` of 0 means the cadence could not be measured, and that is
    encoded as its own token rather than silently defaulted -- an unknown
    cadence is a different situation from a known one, and must be a
    different atom.
    """
    seconds = max(0, int(bar_seconds))
    if seconds <= 0:
        return f"hzn h={int(horizon_bars)} c={'x' * _CADENCE_DIGITS} w={'x' * _WALLCLOCK_DIGITS}"
    minutes = int(round(int(horizon_bars) * seconds / 60.0))
    return (f"hzn h={int(horizon_bars)} "
            f"c={seconds:0{_CADENCE_DIGITS}d} "
            f"w={minutes:0{_WALLCLOCK_DIGITS}d}")


#: How many anchor points the deep prefix is resampled onto, and how many
#: bands each point is quantised into. Both are deliberately COARSE. The key
#: has to collide across a base frame and its mutants or it buys nothing, and
#: it has to separate genuinely different shapes or it is a constant -- the
#: census in ``omen_shape_mutations.py shapekey`` reports both numbers and is
#: the thing that decides whether these values are right.
SHAPE_POINTS_FINE: int = 8
SHAPE_BANDS_FINE: int = 5
SHAPE_POINTS_COARSE: int = 4
SHAPE_BANDS_COARSE: int = 3

#: Which resolutions the frame carries. DEFAULT IS THE COARSE KEY ALONE, and
#: that default is a measurement rather than a taste. Censused on
#: 0004_AERO-USDC over 400 anchors (data/brain_experiments/
#: p120-jet-shapekey-census.json):
#:
#:   resolution  distinct  distinctness  supported  anchors in them
#:         k8         307        76.8%         64            39.2%
#:         k4          44        11.0%         42            99.5%
#:
#: The fine key is unique-per-instant for three anchors in four, which is
#: EXACTLY the failure the shape-mutation arm measured in pass 117 -- a key
#: nothing else shares teaches nothing, so carrying it re-imports the
#: memorisation this pool exists to break. The coarse key puts 99.5% of
#: anchors into a class some other anchor also lands in, and holds 92.8% of
#: ``deep_jitter`` mutants on their base's key. Set OMEN_SHAPE_RESOLUTION to
#: "k8" or "both" to carry the fine key anyway; the census above is the
#: argument against it.
SHAPE_RESOLUTION: str = os.getenv("OMEN_SHAPE_RESOLUTION", "k4").lower()


def _resample_mean(values: Sequence[float], points: int) -> List[float]:
    """Average ``values`` down onto ``points`` equal segments.

    The MEAN is what makes the key survive ``deep_jitter``: gaussian noise
    scaled to the prefix's own step size averages towards zero over a segment
    of ~18 bars, while the shape the segment describes does not. Equal
    SEGMENTS, rather than a fixed stride, is what makes it survive
    ``deep_dilate``: a time stretch changes how many source bars land in a
    segment and not which part of the shape the segment covers.
    """
    n = len(values)
    if n == 0 or points <= 0:
        return []
    out: List[float] = []
    for k in range(points):
        lo = (k * n) // points
        hi = max(lo + 1, ((k + 1) * n) // points)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def _quantise(values: Sequence[float], bands: int, alphabet: str) -> str:
    """Min-max normalise then band, so the key is SCALE-free by construction.

    A flat run has no range to normalise against; it gets the middle band
    rather than an arbitrary one, because "flat" is a shape and a divide-by-
    zero is not.
    """
    if not values:
        return ""
    low, high = min(values), max(values)
    span = high - low
    mid = alphabet[bands // 2]
    if span <= 0.0:
        return mid * len(values)
    letters = []
    for v in values:
        idx = int(((v - low) / span) * bands)
        letters.append(alphabet[min(bands - 1, max(0, idx))])
    return "".join(letters)


def shape_class_frame(closes: Sequence[float]) -> str:
    """The canonical SHAPE of the deep prefix, at two resolutions.

    THE POINT OF THIS FUNCTION, in one line: a base frame and every label-safe
    mutation of it must produce the SAME BYTES here, while two genuinely
    different chart shapes must not.

    WHICH BARS. Exactly the bars the mutations are allowed to touch -- the
    deep prefix, everything older than ``RANGE_WINDOW`` back from the anchor.
    That span is derived from the labeller, not hard-coded: ``label_omen``
    reads the entry close, the future close and position-in-range over the
    last ``RANGE_WINDOW`` bars, so a bar older than that cannot move the
    label. Reading only those bars is what makes the key invariant to the
    mutations; reading only those bars is ALSO why the key adds something the
    other frames do not, since geometry/temporal/volatility are all dominated
    by the recent tail.

    THE TWO RESOLUTIONS are not decoration. The fine key discriminates and the
    coarse key gives support: a key nothing else shares teaches nothing, which
    is precisely the failure the shape-mutation arm measured. Their alphabets
    are byte-DISJOINT ('abcde' against 'pqr') so a decode can never confuse a
    fine band for a coarse one -- the same rule that governs the labels.
    """
    want_fine = SHAPE_RESOLUTION in ("k8", "both")
    want_coarse = SHAPE_RESOLUTION in ("k4", "both")
    deep = list(closes[:-RANGE_WINDOW]) if len(closes) > RANGE_WINDOW else []
    parts = ["shp"]
    if want_fine:
        parts.append("k8=" + (_quantise(
            _resample_mean(deep, SHAPE_POINTS_FINE),
            SHAPE_BANDS_FINE, "abcde") if deep else "na"))
    if want_coarse:
        parts.append("k4=" + (_quantise(
            _resample_mean(deep, SHAPE_POINTS_COARSE),
            SHAPE_BANDS_COARSE, "pqr") if deep else "na"))
    return " ".join(parts)


def build_collections(
    bars: Sequence[Mapping[str, Any]],
    index: int,
    *,
    horizon_bars: int,
    bar_seconds: int,
    symbol: str,
    chain: str = "base",
    history: Optional[Sequence[Any]] = None,
) -> Dict[str, str]:
    """Build one frame per specialised collection for the bar at ``index``.

    Strictly causal: only ``bars[:index + 1]`` is read. Raises ``ValueError``
    when there is not enough history, rather than emitting a short frame --
    a short frame is a *different byte string*, so padding would quietly
    create a second atom for the same situation.

    ``bar_seconds`` is the DECLARED seconds per bar and is required, because
    a bar count without a cadence is not a horizon. It is used only when the
    window carries no usable timestamps; when it does, the cadence written
    into the frame is the one MEASURED off those timestamps, because the
    atom must describe the data and not the caller's nominal. The live path
    declares 60s while its measured index step has run at 180s, and taking
    the declared value there is the very crossing this parameter exists to
    close.

    ``history`` is the brain's own SETTLED predictions, oldest-first, as
    ``omen_metacognition.Resolved`` rows. It feeds pools 15/16/19 and is read
    only when ``OMEN_META_COLLECTIONS`` is on. Omitting it is safe and is the
    default: the self frames then carry their ``na`` sentinels, so the pools
    are bound and uninformative rather than absent -- which keeps the
    ``set(build_collections(...)) == {c.name for c in COLLECTIONS}`` invariant
    true in every configuration. It must never be given UNRESOLVED rows;
    ``self_frames`` filters them out, and that filter is the guard against the
    prediction_error feedback loop that took recall from 100% to 30% here.
    """
    if index < LOOKBACK_BARS:
        raise ValueError(
            f"index {index} needs {LOOKBACK_BARS} bars of history, has {index}")
    if index >= len(bars):
        raise IndexError(f"index {index} out of range for {len(bars)} bars")
    if horizon_bars <= 0:
        raise ValueError(f"horizon_bars must be positive, got {horizon_bars}")

    window = bars[index - LOOKBACK_BARS: index + 1]
    closes = [_close(b) for b in window]
    now = closes[-1]
    bar = bars[index]

    # -- geometry: where price sits inside its own recent ranges -----------
    def position_in_range(span: int) -> Optional[float]:
        recent = closes[-span:]
        low, high = min(recent), max(recent)
        return _safe_div(now - low, high - low)

    high_v, low_v, open_v = float(bar["high"]), float(bar["low"]), float(bar["open"])
    body = _safe_div(now - open_v, high_v - low_v)
    upper_wick = _safe_div(high_v - max(now, open_v), high_v - low_v)
    lower_wick = _safe_div(min(now, open_v) - low_v, high_v - low_v)
    geometry = (
        f"geo p24={_bucket_unit(position_in_range(24))} "
        f"p48={_bucket_unit(position_in_range(48))} "
        f"p168={_bucket_unit(position_in_range(LOOKBACK_BARS))} "
        f"body={_bucket_unit(body)} "
        f"uw={_bucket_unit(upper_wick)} lw={_bucket_unit(lower_wick)}"
    )

    # -- temporal: how the returns MUTATED, not what they are --------------
    rets = _returns(bars, index, RETURN_SPANS)
    step_returns = [
        _safe_div(closes[i] - closes[i - 1], closes[i - 1])
        for i in range(1, len(closes))
    ]
    recent_steps = [r for r in step_returns[-24:] if r is not None]
    flips = sum(
        1 for a, b in zip(recent_steps, recent_steps[1:])
        if (a > 0) != (b > 0)
    )
    streak = 0
    for value in reversed(recent_steps):
        if value == 0:
            break
        if streak == 0 or (value > 0) == (recent_steps[-1] > 0):
            streak += 1
        else:
            break
    # Acceleration: the 6-bar return against twice the 3-bar return. Positive
    # means the move is still opening up, negative means it is decaying.
    accel = None
    if rets[6] is not None and rets[3] is not None:
        accel = rets[6] - 2.0 * rets[3]
    temporal = (
        "tmp " + " ".join(f"r{span}={_bucket_return(rets[span])}" for span in RETURN_SPANS)
        + f" flip={min(23, flips)} streak={min(23, streak)}"
        + f" acc={_bucket_return(accel)}"
    )

    # -- flow: volume, and who is doing it ---------------------------------
    volumes = [float(b.get("net_volume") or 0.0) for b in window]
    mean_volume = statistics.fmean(volumes[:-1]) if len(volumes) > 1 else 0.0
    volume_ratio = _safe_div(volumes[-1], mean_volume)
    recent_volume = statistics.fmean(volumes[-24:]) if len(volumes) >= 24 else None
    volume_trend = _safe_div(recent_volume, mean_volume) if recent_volume is not None else None
    buy_v = float(bar.get("buy_volume") or 0.0)
    sell_v = float(bar.get("sell_volume") or 0.0)
    buy_share = _safe_div(buy_v, buy_v + sell_v)
    recent_buy = sum(float(b.get("buy_volume") or 0.0) for b in window[-24:])
    recent_sell = sum(float(b.get("sell_volume") or 0.0) for b in window[-24:])
    buy_share_24 = _safe_div(recent_buy, recent_buy + recent_sell)
    flow = (
        f"flw v={_bucket_ratio(volume_ratio)} vt={_bucket_ratio(volume_trend)} "
        f"bs={_bucket_unit(buy_share)} bs24={_bucket_unit(buy_share_24)}"
    )

    # -- volatility: realised range, expanding or contracting ---------------
    def realised_vol(span: int) -> Optional[float]:
        sample = [r for r in step_returns[-span:] if r is not None]
        if len(sample) < 2:
            return None
        return statistics.pstdev(sample)

    vol24, vol168 = realised_vol(24), realised_vol(LOOKBACK_BARS)
    expansion = _safe_div(vol24, vol168) if (vol24 is not None and vol168) else None
    bar_range = _safe_div(high_v - low_v, now)
    volatility = (
        f"vol v24={_bucket_return(vol24)} v168={_bucket_return(vol168)} "
        f"exp={_bucket_ratio(expansion)} rng={_bucket_return(bar_range)}"
    )

    # -- cross: the symbol against its own longer baseline ------------------
    mean24 = statistics.fmean(closes[-24:])
    mean168 = statistics.fmean(closes)
    cross = (
        f"crs d24={_bucket_return(_safe_div(now - mean24, mean24))} "
        f"d168={_bucket_return(_safe_div(now - mean168, mean168))} "
        f"ma={_bucket_return(_safe_div(mean24 - mean168, mean168))}"
    )

    # -- horizon: how far ahead the question is being asked -----------------
    # Measured off the window that was actually read, falling back to the
    # caller's declared cadence only when the bars carry no timestamps. See
    # ``horizon_frame`` for why the cadence has to be in here at all.
    cadence = measure_bar_seconds(window, default=int(bar_seconds))
    horizon = horizon_frame(horizon_bars, cadence)

    # -- instrument: which market -------------------------------------------
    instrument = f"ins {symbol.strip().lower()} {chain.strip().lower()}"

    # -- RELATIONS between the families above -------------------------------
    # The flat pools can carry "the return is x" and "the range is y", but
    # never "x is large FOR y". That conjunction only exists in the fabric if
    # something writes it down, and PoolKind::Internal will not write it: it
    # is matched 0 times in the engine (see docs/BRAIN_POOL_TOPOLOGY.md), so
    # the relation has to be computed here and fed as its own frame.
    #
    # RELATIONS_ENABLED gates the keys AND the COLLECTIONS entries from one
    # flag, so `set(build_collections(...)) == {c.name for c in COLLECTIONS}`
    # holds either way. That invariant is load-bearing and has a test of its
    # own (test_every_collection_has_its_own_byte_prefix): what this function
    # returns is exactly what gets streamed, with nothing computed that no
    # pool will receive.
    base = {
        "geometry": geometry,
        "temporal": temporal,
        "flow": flow,
        "volatility": volatility,
        "cross": cross,
        "horizon": horizon,
        "instrument": instrument,
    }
    def _finish(frames: Dict[str, str]) -> Dict[str, str]:
        """Append the metacognition frames, if this build streams them.

        Applied at BOTH return paths on purpose: the meta pools are gated by
        their own flag and must appear whether or not the relation pools do.
        Imported here rather than at module scope so a caller with the flag
        off never pays for the module.
        """
        if SHAPE_ENABLED:
            frames["shape_class"] = shape_class_frame(closes)
        if not META_ENABLED:
            return frames
        from trading.omen_metacognition import metacognition_frames
        frames.update(metacognition_frames(closes, history or (),
                                           noise=vol24))
        return frames

    if not RELATIONS_ENABLED:
        return _finish(base)

    # temporal x volatility: the move in units of its OWN noise. vol24 is a
    # per-STEP stdev, so noise over n steps scales as vol24*sqrt(n); dividing
    # a fraction by a fraction leaves these dimensionless, which is what
    # _bucket_signed expects.
    def _z(span: int) -> Optional[float]:
        ret = rets.get(span)
        if ret is None or not vol24:
            return None
        return _safe_div(ret, vol24 * math.sqrt(span))

    rel_move_vol = (
        f"rmv z6={_bucket_signed(_z(6))} z24={_bucket_signed(_z(24))} "
        f"rngv={_bucket_ratio(_safe_div(bar_range, vol24) if vol24 else None)}"
    )

    # geometry x flow: is the shape CONFIRMED by who is trading it? `dir` is
    # positive when the candle's direction agrees with buying pressure;
    # `pos` is positive when buyers are heavier than price position implies
    # (both signed, both dimensionless). `impact` is move per unit of
    # relative volume -- a wide range on thin volume is a book artifact, not
    # a move, and it is the shape that most often fakes an entry here.
    pos24 = position_in_range(24)
    rel_shape_flow = (
        f"rsf dir={_bucket_signed(body * (buy_share - 0.5) * 2.0, span=1.0) if (body is not None and buy_share is not None) else 'na'} "
        f"pos={_bucket_signed(buy_share_24 - pos24, span=1.0) if (buy_share_24 is not None and pos24 is not None) else 'na'} "
        f"impact={_bucket_return(_safe_div(bar_range, volume_ratio) if volume_ratio else None)}"
    )

    # cross x volatility: is the distance from the long baseline BIG for this
    # symbol's noise? A 2% gap means nothing without knowing whether 2% is a
    # normal hour here. Deliberately NOT called symbol-vs-market: this
    # function sees one symbol's bars, so a true cross-sectional relation is
    # not computable at this seam and pretending otherwise would be a fake.
    d168 = _safe_div(now - mean168, mean168)
    d24 = _safe_div(now - mean24, mean24)
    rel_trend_noise = (
        f"rtn t168={_bucket_signed(_safe_div(d168, vol168 * math.sqrt(LOOKBACK_BARS)) if (d168 is not None and vol168) else None)} "
        f"t24={_bucket_signed(_safe_div(d24, vol24 * math.sqrt(24)) if (d24 is not None and vol24) else None)} "
        f"exp={_bucket_ratio(expansion)}"
    )

    base.update({
        "rel_move_vol": rel_move_vol,
        "rel_shape_flow": rel_shape_flow,
        "rel_trend_noise": rel_trend_noise,
    })
    return _finish(base)


# --- labelling ------------------------------------------------------------

#: How high in its recent range price must sit to call the level a high, and
#: how low for a low. 0.35/0.65 splits the range into thirds -- the middle
#: third is neither a trough nor a crest however the market then moves.
LOW_BAND = float(os.getenv("OMEN_LOW_BAND", "0.35"))
HIGH_BAND = float(os.getenv("OMEN_HIGH_BAND", "0.65"))
#: The forward move must clear the round trip by this multiple to count.
#: 1.0 means "exactly pays for itself"; the default demands a real margin.
COST_MULTIPLE = float(os.getenv("OMEN_COST_MULTIPLE", "1.5"))
#: Range window used to decide "low" and "high", in bars.
RANGE_WINDOW = int(os.getenv("OMEN_RANGE_WINDOW", "24"))


def omen_threshold(cost: float = ROUND_TRIP_COST,
                   multiple: float = COST_MULTIPLE) -> float:
    """The forward move an omen must clear, as a fraction. Never zero."""
    return abs(cost) * abs(multiple)


def label_omen(
    bars: Sequence[Mapping[str, Any]],
    index: int,
    *,
    horizon_bars: int,
    cost: float = ROUND_TRIP_COST,
    multiple: float = COST_MULTIPLE,
    range_window: int = RANGE_WINDOW,
) -> Optional[str]:
    """The true omen at ``index`` looking ``horizon_bars`` ahead.

    Returns ``None`` when the future is not in the corpus -- the caller must
    drop the sample rather than label it, because a missing future is not a
    ``murk``.

    The forward return is measured close-to-close. The threshold is the
    round-trip cost, never zero: a +0.1% move on a 0.65% round trip is a
    loss, and a labelling scheme that calls it a win teaches the substrate
    to lose money.
    """
    future_index = index + horizon_bars
    if future_index >= len(bars) or index < 0:
        return None
    entry = _close(bars[index])
    forward = _safe_div(_close(bars[future_index]) - entry, entry)
    if forward is None:
        return None

    threshold = omen_threshold(cost, multiple)
    if abs(forward) < threshold:
        return OMEN_MURK

    window_start = max(0, index - range_window + 1)
    recent = [_close(b) for b in bars[window_start: index + 1]]
    low, high = min(recent), max(recent)
    position = _safe_div(entry - low, high - low)
    # A dead-flat window has no low and no high to be at.
    at_low = position is not None and position <= LOW_BAND
    at_high = position is not None and position >= HIGH_BAND

    if forward > 0:
        return OMEN_TROUGH if at_low else OMEN_CLIMB
    return OMEN_CREST if at_high else OMEN_SLIDE


def label_regime(
    bars: Sequence[Mapping[str, Any]],
    index: int,
    *,
    cost: float = ROUND_TRIP_COST,
) -> str:
    """Stage-1 target: the trailing regime. Strictly causal (no lookahead).

    This is a *summary of the past*, deliberately. Stage 1 exists so that
    stage 2 receives an integrated view of the sensory pools rather than
    seven flat frames; if stage 1 peeked at the future it would leak the
    answer into the chain and the held-out numbers would be fiction.
    """
    if index < 24:
        return "chop"
    closes = [_close(b) for b in bars[max(0, index - LOOKBACK_BARS): index + 1]]
    steps = [
        _safe_div(closes[i] - closes[i - 1], closes[i - 1])
        for i in range(1, len(closes))
    ]
    steps = [s for s in steps if s is not None]
    if len(steps) < 4:
        return "chop"
    drift = _safe_div(closes[-1] - closes[-24], closes[-24]) or 0.0
    vol_recent = statistics.pstdev(steps[-24:]) if len(steps) >= 24 else statistics.pstdev(steps)
    vol_long = statistics.pstdev(steps) if len(steps) >= 4 else vol_recent
    threshold = abs(cost)
    if vol_long > 0 and vol_recent < 0.5 * vol_long:
        return "squeeze"
    if drift >= threshold:
        return "bullrun"
    if drift <= -threshold:
        return "bearrun"
    return "chop"


def regime_frame(token: str) -> str:
    return f"regime {token}"


def parse_regime(answer: Optional[str]) -> Optional[str]:
    value = (answer or "").lower()
    for token in sorted(REGIME_TOKENS, key=len, reverse=True):
        if token in value:
            return token
    return None


# --- the output schema ----------------------------------------------------

@dataclass
class Omen:
    """The schema every caller reads. Fixed fields, validated on build.

    ``verdict`` is never absent and never ambiguous: a brain that has no
    opinion says so with a named reason. ``action`` is derived from the
    label through ``OMEN_ACTIONS`` and is the only field a strategy should
    branch on.
    """

    schema_version: str
    symbol: str
    chain: str
    as_of_ts: int
    price: float
    horizon_bars: int
    bar_seconds: int
    omen: str
    action: str
    confidence: float
    cost_fraction: float
    threshold_fraction: float
    expected_move_fraction: float
    verdict: str
    regime: Optional[str] = None
    regime_confidence: float = 0.0
    collections: Dict[str, str] = field(default_factory=dict)
    support: Dict[str, Any] = field(default_factory=dict)

    VERDICTS = (
        "admitted", "below_floor", "no_answer", "degenerate",
        "unsupported_horizon", "not_trained", "transport_error",
        # The query sets disagreed. Reproduction accuracy on a split is
        # 73.3% against 99.4% on a unanimous answer -- a named abstention,
        # not a weak signal.
        "split",
    )

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        if self.omen not in OMEN_LABELS:
            raise ValueError(f"omen {self.omen!r} not in {OMEN_LABELS}")
        if self.verdict not in self.VERDICTS:
            raise ValueError(f"verdict {self.verdict!r} not in {self.VERDICTS}")
        if self.action not in ("buy", "sell", "hold"):
            raise ValueError(f"action {self.action!r} is not buy/sell/hold")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        if self.threshold_fraction <= 0.0:
            raise ValueError(
                "threshold_fraction must be positive -- an omen measured "
                "against zero instead of against cost is the losing shape")
        # An un-admitted omen must never carry a tradeable action. This is
        # the single invariant that keeps a degenerate or low-confidence
        # brain from reaching the money path.
        if self.verdict != "admitted" and self.action != "hold":
            raise ValueError(
                f"verdict {self.verdict!r} must carry action 'hold', "
                f"got {self.action!r}")

    @property
    def is_actionable(self) -> bool:
        return self.verdict == "admitted" and self.action in ("buy", "sell")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _hold_omen(reason: str, *, symbol: str, chain: str, as_of_ts: int,
               price: float, horizon_bars: int, bar_seconds: int,
               cost: float, threshold: float,
               collections: Optional[Dict[str, str]] = None,
               support: Optional[Dict[str, Any]] = None) -> Omen:
    """A well-formed 'no opinion' answer. Callers never get None."""
    return Omen(
        schema_version=SCHEMA_VERSION,
        symbol=symbol, chain=chain, as_of_ts=int(as_of_ts), price=float(price),
        horizon_bars=int(horizon_bars), bar_seconds=int(bar_seconds),
        omen=OMEN_MURK, action="hold", confidence=0.0,
        cost_fraction=float(cost), threshold_fraction=float(threshold),
        expected_move_fraction=0.0, verdict=reason,
        collections=collections or {}, support=support or {},
    )


# --- transport ------------------------------------------------------------

def _b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> str:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad).decode("utf-8", errors="replace")


def resolve_omen_endpoint(target: Optional[str]) -> Tuple[str, int]:
    """Resolve an omen endpoint to ``(host, port)``, defaulting to :8091.

    One resolver, shared with :class:`trading.brain_bridge.BrainBridge`,
    because the bug it fixes was the same code written twice: a
    scheme-less ``--endpoint 127.0.0.1:8092`` parsed to no host and no
    port, fell back to :8091, and trained the wrong node in silence.
    """
    return resolve_node_endpoint(target, 8091)


class OmenBrain:
    """Client for the omen node: chained multi-pool train and predict.

    Defaults to ``OMEN_BRAIN_ENDPOINT`` (127.0.0.1:8091) -- a node of its
    own, so training omens can never disturb the fabric the live regime
    signal reads on :8090.
    """

    def __init__(self, endpoint: Optional[str] = None, timeout: float = 30.0,
                 degenerate_window: int = 32) -> None:
        target = endpoint or os.getenv("OMEN_BRAIN_ENDPOINT", "http://127.0.0.1:8091")
        self._host, self._port = resolve_omen_endpoint(target)
        self._timeout = timeout
        self._lock = threading.RLock()
        self._conn: Optional[HTTPConnection] = None
        self._multi_supported: Optional[bool] = None
        #: Rolling record of decoded answers -- the degeneracy guard.
        self._recent_answers: deque = deque(maxlen=max(4, degenerate_window))
        self.trained_pairs = 0
        self.failed_pairs = 0

    # -- plumbing ---------------------------------------------------------
    def _reset(self) -> None:
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            pass
        self._conn = HTTPConnection(self._host, self._port, timeout=self._timeout)

    def _post(self, path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        for attempt in (0, 1):
            try:
                if self._conn is None:
                    self._reset()
                self._conn.request("POST", path, body,
                                   {"Content-Type": "application/json"})
                response = self._conn.getresponse()
                raw = response.read()
                if response.status == 404:
                    return {"__status__": 404}
                return json.loads(raw.decode("utf-8", errors="replace"))
            except (BadStatusLine, RemoteDisconnected, ConnectionError,
                    OSError, TimeoutError, ValueError):
                self._conn = None
                if attempt == 1:
                    return None
        return None

    def supports_multi(self) -> bool:
        """Whether the running binary has the multi-stream routes.

        Probed once with a read-only predict. A node built before the routes
        existed answers 404 -- the 2026-07-08 incident was exactly that, a
        stale exe silently lacking the learning surface, so this is checked
        rather than assumed.
        """
        with self._lock:
            if self._multi_supported is None:
                reply = self._post("/brain/predict/multi", {
                    "target_pool": OMEN_POOL,
                    "streams": [{"pool_id": COLLECTIONS[0].pool_id,
                                 "frame": _b64url("geo probe")}],
                })
                self._multi_supported = bool(
                    reply is not None and reply.get("__status__") != 404)
            return self._multi_supported

    # -- training ---------------------------------------------------------
    def _streams(self, frames: Mapping[str, str],
                 include: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        names = include if include is not None else [c.name for c in COLLECTIONS]
        streams = []
        for name in names:
            collection = COLLECTIONS_BY_NAME.get(name)
            if collection is None or name not in frames:
                continue
            streams.append({"pool_id": collection.pool_id,
                            "frame": _b64url(frames[name])})
        return streams

    def _consolidate(self, streams: List[Dict[str, Any]],
                     outcome_pool: int, outcome_frame: str) -> bool:
        """One supervised binding. Honours the node's ingest backpressure."""
        payload = {
            "streams": streams,
            "outcome_pool": outcome_pool,
            "outcome_frame": _b64url(outcome_frame),
        }
        retries = int(os.getenv("WIZARD_BACKPRESSURE_RETRIES", "30"))
        for _ in range(max(1, retries)):
            with self._lock:
                reply = self._post("/brain/consolidate/multi", payload)
            if reply is None:
                return False
            if reply.get("backpressure"):
                time.sleep(min(30.0, float(reply.get("retry_after_ms") or 2000) / 1000.0))
                continue
            return reply.get("consolidated") is True
        return False

    def train(self, frames: Mapping[str, str], omen_label: str,
              regime_token: str) -> bool:
        """Train one chained sample. Both stages must land or it is a miss.

        Stage 1 binds the four market-shape collections to the regime token
        in pool 5; stage 2 binds every collection PLUS that regime frame to
        the omen token in pool 11.
        """
        if omen_label not in OMEN_LABELS:
            raise ValueError(f"unknown omen label {omen_label!r}")
        if regime_token not in REGIME_TOKENS:
            raise ValueError(f"unknown regime token {regime_token!r}")

        stage1 = self._consolidate(
            self._streams(frames, STAGE1_COLLECTIONS),
            REGIME_POOL, regime_frame(regime_token))

        stage2_streams = self._streams(frames)
        stage2_streams.append({"pool_id": REGIME_POOL,
                               "frame": _b64url(regime_frame(regime_token))})
        stage2 = self._consolidate(stage2_streams, OMEN_POOL, omen_frame(omen_label))

        if stage1 and stage2:
            self.trained_pairs += 1
            return True
        self.failed_pairs += 1
        return False

    # -- prediction -------------------------------------------------------
    def _predict(self, streams: List[Dict[str, Any]],
                 target_pool: int) -> Tuple[Optional[str], float]:
        with self._lock:
            reply = self._post("/brain/predict/multi", {
                "target_pool": target_pool, "streams": streams})
        if not reply or reply.get("__status__") == 404:
            return None, 0.0
        encoded = reply.get("answer")
        answer = _b64url_decode(encoded) if encoded else None
        confidence = float(
            reply.get("integrated_confidence") or reply.get("confidence") or 0.0)
        return answer, max(0.0, min(1.0, confidence))

    def _degenerate(self) -> bool:
        """True when the answer stream has collapsed to a single token.

        A brain that says one thing to everything scores well whenever the
        market trends -- that is how a 78% 'accuracy' turned out to be the
        market's own down-drift, inverted. Needs a full window before it
        will call anything degenerate.
        """
        if len(self._recent_answers) < self._recent_answers.maxlen:
            return False
        return len(set(self._recent_answers)) <= 1

    def predict(
        self,
        frames: Mapping[str, str],
        *,
        symbol: str,
        chain: str,
        as_of_ts: int,
        price: float,
        horizon_bars: int,
        bar_seconds: int,
        confidence_floor: float = 0.0,
        cost: float = ROUND_TRIP_COST,
        multiple: float = COST_MULTIPLE,
        regime: Optional[str] = None,
        query_collections: Optional[Sequence[str]] = None,
        consensus: bool = False,
    ) -> Omen:
        """Read-only prediction. Always returns a valid ``Omen``.

        ``regime`` is the caller's own ``label_regime(bars, index)``. Supply
        it whenever the bars are in hand -- which is everywhere the frames
        were built from, since they need the same window. The regime is a
        *deterministic causal function of the past*, so asking the brain to
        guess it is asking a 73.3%-accurate classifier for a number already
        on the caller's desk. Measured: the production path scored 86.0%
        train recall feeding stage 1's guess into stage 2, 90.7% feeding the
        computed one, and 96.0% firing only the discriminating collections.

        Omitting ``regime`` falls back to the stage-1 probe so a caller with
        frames but no bars still gets a regime *reported* -- it is never fed
        back into the stage-2 query unless ``PREDICT_INCLUDE_REGIME`` is set.

        ``consensus`` fires ``CONSENSUS_QUERIES`` instead of one query and
        abstains with ``verdict="split"`` unless they all decode the same
        label. Costs four round trips and buys 95.5% -> 99.4% reproduction;
        it does NOT buy an edge. Off by default so latency is a caller's
        choice.
        """
        threshold = omen_threshold(cost, multiple)
        hold = lambda reason, **extra: _hold_omen(  # noqa: E731 - local alias
            reason, symbol=symbol, chain=chain, as_of_ts=as_of_ts, price=price,
            horizon_bars=horizon_bars, bar_seconds=bar_seconds,
            cost=cost, threshold=threshold, collections=dict(frames), **extra)

        if not self.supports_multi():
            return hold("transport_error",
                        support={"reason": "node lacks /brain/predict/multi"})

        regime_answer: Optional[str] = None
        regime_confidence = 0.0
        if regime is not None and regime in REGIME_TOKENS:
            # Computed, not guessed. Full confidence because it is arithmetic.
            regime_answer, regime_confidence = regime_frame(regime), 1.0
        else:
            # Stage 1 -- integrate the market-shape pools into a regime. Kept
            # for callers that hold frames without bars; its answer is
            # REPORTED, and only fed back into stage 2 behind the env flag.
            regime_answer, regime_confidence = self._predict(
                self._streams(frames, STAGE1_COLLECTIONS), REGIME_POOL)
            regime = parse_regime(regime_answer)

        # Stage 2 -- the discriminating collections only. See the dilution
        # law above ``PREDICT_COLLECTIONS``: every extra low-distinctness
        # stream measurably out-votes the sharp ones.
        names = list(query_collections if query_collections is not None
                     else PREDICT_COLLECTIONS)

        def fire(member: Sequence[str]) -> Tuple[List[Dict[str, Any]], Optional[str], float]:
            streams = self._streams(frames, member)
            if not streams:  # an unknown override must not silence the brain
                streams = self._streams(frames)
            if PREDICT_INCLUDE_REGIME and regime is not None:
                streams.append({"pool_id": REGIME_POOL,
                                "frame": _b64url(regime_frame(regime))})
            reply, conf = self._predict(streams, OMEN_POOL)
            return streams, reply, conf

        if not self._streams(frames, names):
            names = [c.name for c in COLLECTIONS]

        # The first member is always the primary answer, so a consensus read
        # and a plain read agree on WHAT was predicted and differ only on
        # whether it is admitted.
        #
        # Deduped by SET, not by tuple. ``discriminating_collections`` returns
        # the measured query in DISTINCTNESS order, so on AERO-USDC the primary
        # arrives as ('geometry','temporal','cross') while CONSENSUS_QUERIES[0]
        # is ('temporal','geometry','cross') -- the same query, a different
        # tuple. Comparing tuples kept it as a fifth member, and it cost more
        # than a round trip: a member that IS the primary cannot disagree with
        # it on the merits, so it inflated every unanimity rate; and because
        # the node is not perfectly deterministic (an A-vs-A control moved
        # 4/100 held-out predictions, pass 111) it turned node noise into
        # spurious ``split`` abstentions on a query that was never in doubt.
        if consensus:
            members = [tuple(names)]
            seen = {frozenset(names)}
            for candidate in CONSENSUS_QUERIES:
                if frozenset(candidate) not in seen:
                    seen.add(frozenset(candidate))
                    members.append(tuple(candidate))
        else:
            members = [tuple(names)]
        member_labels: List[Optional[str]] = []
        stage2: List[Dict[str, Any]] = []
        answer: Optional[str] = None
        confidence = 0.0
        for position, member in enumerate(members):
            streams, reply, conf = fire(member)
            member_labels.append(parse_omen(reply))
            if position == 0:
                stage2, answer, confidence = streams, reply, conf
        label = parse_omen(answer)
        unanimous = (len(members) == 1
                     or (label is not None
                         and all(m == label for m in member_labels)))

        if label is None:
            self._recent_answers.append("__none__")
            return hold("no_answer", support={"raw_answer": answer})
        self._recent_answers.append(label)

        support = {
            "raw_answer": answer,
            "distinct_recent_answers": len(set(self._recent_answers)),
            "recent_window": len(self._recent_answers),
            "stage1_answer": regime_answer,
            "collections_fired": len(stage2),
            "query_collections": names,
            # Whether the regime was arithmetic or a guess. A held-out number
            # read without this line is not comparable to one read with it.
            "regime_source": "computed" if regime_confidence >= 1.0 else "stage1",
            "consensus": bool(consensus),
            "unanimous": bool(unanimous),
            "member_answers": member_labels,
        }

        if self._degenerate():
            return hold("degenerate", support=support)
        if not unanimous:
            # 73.3% right on a split against 99.4% on a unanimous answer.
            omen = hold("split", support=support)
            omen.regime, omen.regime_confidence = regime, regime_confidence
            return omen
        if confidence < confidence_floor:
            omen = hold("below_floor", support=support)
            omen.regime, omen.regime_confidence = regime, regime_confidence
            return omen

        direction = OMEN_DIRECTION[label]
        return Omen(
            schema_version=SCHEMA_VERSION,
            symbol=symbol, chain=chain, as_of_ts=int(as_of_ts), price=float(price),
            horizon_bars=int(horizon_bars), bar_seconds=int(bar_seconds),
            omen=label, action=OMEN_ACTIONS[label], confidence=confidence,
            cost_fraction=float(cost), threshold_fraction=threshold,
            # The floor of the band the label asserts, signed. Never the
            # midpoint: a strategy sizing on this must not be told to expect
            # more than the label actually claims.
            expected_move_fraction=direction * threshold,
            verdict="admitted", regime=regime, regime_confidence=regime_confidence,
            collections=dict(frames), support=support,
        )
