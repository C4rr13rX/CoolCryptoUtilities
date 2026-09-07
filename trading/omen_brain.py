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
from collections import deque
from dataclasses import dataclass, field, asdict
from http.client import HTTPConnection, BadStatusLine, RemoteDisconnected
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

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


def build_collections(
    bars: Sequence[Mapping[str, Any]],
    index: int,
    *,
    horizon_bars: int,
    symbol: str,
    chain: str = "base",
) -> Dict[str, str]:
    """Build one frame per specialised collection for the bar at ``index``.

    Strictly causal: only ``bars[:index + 1]`` is read. Raises ``ValueError``
    when there is not enough history, rather than emitting a short frame --
    a short frame is a *different byte string*, so padding would quietly
    create a second atom for the same situation.
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
    horizon = f"hzn h={int(horizon_bars)}"

    # -- instrument: which market -------------------------------------------
    instrument = f"ins {symbol.strip().lower()} {chain.strip().lower()}"

    return {
        "geometry": geometry,
        "temporal": temporal,
        "flow": flow,
        "volatility": volatility,
        "cross": cross,
        "horizon": horizon,
        "instrument": instrument,
    }


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


class OmenBrain:
    """Client for the omen node: chained multi-pool train and predict.

    Defaults to ``OMEN_BRAIN_ENDPOINT`` (127.0.0.1:8091) -- a node of its
    own, so training omens can never disturb the fabric the live regime
    signal reads on :8090.
    """

    def __init__(self, endpoint: Optional[str] = None, timeout: float = 30.0,
                 degenerate_window: int = 32) -> None:
        target = endpoint or os.getenv("OMEN_BRAIN_ENDPOINT", "http://127.0.0.1:8091")
        parsed = urlparse(target)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 8091
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
        stage2 = self._streams(frames, names)
        if not stage2:  # an unknown override must not silence the brain
            stage2 = self._streams(frames)
            names = [c.name for c in COLLECTIONS]
        if PREDICT_INCLUDE_REGIME and regime is not None:
            stage2.append({"pool_id": REGIME_POOL,
                           "frame": _b64url(regime_frame(regime))})
        answer, confidence = self._predict(stage2, OMEN_POOL)
        label = parse_omen(answer)

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
        }

        if self._degenerate():
            return hold("degenerate", support=support)
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
