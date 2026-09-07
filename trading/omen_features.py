"""Omen features as NUMBERS, and bins fitted to the data they must generalise over.

Why this module exists
----------------------
``trading.omen_brain.build_collections`` emits byte frames for the substrate.
Its bucket boundaries are fixed constants, chosen -- the comment there says so
-- to be "deliberately fine-grained" because coarse buckets "cap recall by
collision". That objective is backwards for money. Measured on the 2725-pair
AERO-USDC training set those frames produce **2725 distinct signatures out of
2725 samples**: no two bars in three thousand ever land on the same key. A
learner keyed on those frames can memorise every training bar perfectly and
has, by construction, nothing to say about a bar it has not seen. That is the
exact shape of the 2026-09-07 result: 100% train recall beside 26.6% held-out
accuracy against a 31.2% majority class.

Generalisation needs the opposite property: bins coarse enough that *similar
situations share a bin*, so a bin carries a population whose forward returns
can be averaged into an estimate. This module provides that, and keeps two
rules the fixed-constant approach cannot:

1. **Bin edges are fitted, never assumed.** Edges are the empirical quantiles
   of the TRAINING window only. A log-spaced constant is right for one
   volatility regime and wrong for the next; a quantile is right for whatever
   the data is, and guarantees every bin holds mass. ``fit_bins`` never sees a
   forward return, so no edge can encode an answer.
2. **Strictly causal.** Every scalar here reads ``bars[:index + 1]`` and
   nothing else, and ``features`` raises rather than padding a short window --
   a padded window is a different situation wearing the same key.

The scalars mirror ``build_collections`` quantity-for-quantity so anything
learned here transfers back to the substrate frames; the difference is the
representation, not the sensing.

Nothing here predicts, labels, or sizes anything. It turns bars into numbers
and numbers into bins.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: How far back the feature builder must see, in bars. Same as the substrate
#: frames' lookback so the two representations sense the same history.
LOOKBACK_BARS = 168

#: Trailing return spans, in bars.
RETURN_SPANS: Tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48, 168)

#: Every scalar this module produces, in a fixed order. Fixed so a bin table
#: fitted on one run can never be applied to a differently-ordered vector.
FEATURE_NAMES: Tuple[str, ...] = (
    # geometry -- where price sits inside its own recent range
    "p24", "p48", "p168", "body", "upper_wick", "lower_wick",
    # temporal -- what the returns did, and how that mutated
    "r1", "r2", "r3", "r6", "r12", "r24", "r48", "r168",
    "flips", "streak", "accel",
    # flow -- volume, and who is doing it
    "volume_ratio", "volume_trend", "buy_share", "buy_share_24",
    # volatility -- realised range, expanding or contracting
    "vol24", "vol168", "expansion", "bar_range",
    # cross -- the bar against its own longer baselines
    "d24", "d168", "ma_spread",
)


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    """Division that returns ``None`` instead of raising or emitting inf/nan."""
    if denominator is None or not math.isfinite(denominator) or abs(denominator) < 1e-18:
        return None
    result = numerator / denominator
    return result if math.isfinite(result) else None


def _close(bar: Mapping[str, Any]) -> float:
    return float(bar["close"])


def features(
    bars: Sequence[Mapping[str, Any]],
    index: int,
) -> Dict[str, Optional[float]]:
    """Every scalar for the bar at ``index``. Reads only ``bars[:index + 1]``.

    Raises ``ValueError`` when there is not enough history and ``IndexError``
    when ``index`` is past the end -- never returns a short or padded vector,
    because a padded window is a different situation and would share a key
    with the real one.

    A scalar that cannot be computed (a flat window has no position inside its
    own range) is ``None``, which ``digitize`` turns into its own bin. Missing
    is a state, not a zero.
    """
    if index < LOOKBACK_BARS:
        raise ValueError(
            f"index {index} needs {LOOKBACK_BARS} bars of history, has {index}")
    if index >= len(bars):
        raise IndexError(f"index {index} out of range for {len(bars)} bars")

    window = bars[index - LOOKBACK_BARS: index + 1]
    closes = [_close(b) for b in window]
    now = closes[-1]
    bar = bars[index]

    out: Dict[str, Optional[float]] = {}

    # -- geometry ----------------------------------------------------------
    def position_in_range(span: int) -> Optional[float]:
        recent = closes[-span:]
        low, high = min(recent), max(recent)
        return _safe_div(now - low, high - low)

    out["p24"] = position_in_range(24)
    out["p48"] = position_in_range(48)
    out["p168"] = position_in_range(LOOKBACK_BARS)

    high_v, low_v, open_v = float(bar["high"]), float(bar["low"]), float(bar["open"])
    out["body"] = _safe_div(now - open_v, high_v - low_v)
    out["upper_wick"] = _safe_div(high_v - max(now, open_v), high_v - low_v)
    out["lower_wick"] = _safe_div(min(now, open_v) - low_v, high_v - low_v)

    # -- temporal ----------------------------------------------------------
    rets: Dict[int, Optional[float]] = {}
    for span in RETURN_SPANS:
        past = index - span
        rets[span] = (None if past < 0 else
                      _safe_div(now - _close(bars[past]), _close(bars[past])))
        out[f"r{span}"] = rets[span]

    steps = [_safe_div(closes[i] - closes[i - 1], closes[i - 1])
             for i in range(1, len(closes))]
    recent_steps = [s for s in steps[-24:] if s is not None]
    out["flips"] = float(sum(
        1 for a, b in zip(recent_steps, recent_steps[1:]) if (a > 0) != (b > 0)))
    streak = 0
    for value in reversed(recent_steps):
        if value == 0:
            break
        if streak == 0 or (value > 0) == (recent_steps[-1] > 0):
            streak += 1
        else:
            break
    out["streak"] = float(streak)
    out["accel"] = (None if rets[6] is None or rets[3] is None
                    else rets[6] - 2.0 * rets[3])

    # -- flow --------------------------------------------------------------
    volumes = [float(b.get("net_volume") or 0.0) for b in window]
    mean_volume = statistics.fmean(volumes[:-1]) if len(volumes) > 1 else 0.0
    out["volume_ratio"] = _safe_div(volumes[-1], mean_volume)
    recent_volume = statistics.fmean(volumes[-24:]) if len(volumes) >= 24 else None
    out["volume_trend"] = (None if recent_volume is None
                           else _safe_div(recent_volume, mean_volume))
    buy_v = float(bar.get("buy_volume") or 0.0)
    sell_v = float(bar.get("sell_volume") or 0.0)
    out["buy_share"] = _safe_div(buy_v, buy_v + sell_v)
    recent_buy = sum(float(b.get("buy_volume") or 0.0) for b in window[-24:])
    recent_sell = sum(float(b.get("sell_volume") or 0.0) for b in window[-24:])
    out["buy_share_24"] = _safe_div(recent_buy, recent_buy + recent_sell)

    # -- volatility --------------------------------------------------------
    def realised_vol(span: int) -> Optional[float]:
        sample = [s for s in steps[-span:] if s is not None]
        return statistics.pstdev(sample) if len(sample) >= 2 else None

    vol24, vol168 = realised_vol(24), realised_vol(LOOKBACK_BARS)
    out["vol24"], out["vol168"] = vol24, vol168
    out["expansion"] = (_safe_div(vol24, vol168)
                        if vol24 is not None and vol168 else None)
    out["bar_range"] = _safe_div(high_v - low_v, now)

    # -- cross -------------------------------------------------------------
    mean24 = statistics.fmean(closes[-24:])
    mean168 = statistics.fmean(closes)
    out["d24"] = _safe_div(now - mean24, mean24)
    out["d168"] = _safe_div(now - mean168, mean168)
    out["ma_spread"] = _safe_div(mean24 - mean168, mean168)

    return {name: out.get(name) for name in FEATURE_NAMES}


# --- binning --------------------------------------------------------------

#: The bin a ``None`` scalar lands in. Its own value, never folded into a
#: numeric bin: "this window was flat so it has no position in its range" is a
#: distinguishable state, and averaging it in with "position 0.5" invents data.
MISSING_BIN = -1


@dataclass(frozen=True)
class BinTable:
    """Fitted bin edges, one list per feature. Built by ``fit_bins`` only.

    ``edges[name]`` holds ``bins - 1`` interior cut points, so a value falls in
    bin ``bisect_right(edges, value)`` in ``0 .. bins - 1``. A feature with too
    few distinct values gets fewer bins than asked for rather than duplicate
    edges -- duplicate edges create bins that can never be occupied, which
    silently reduces the model's capacity without saying so.
    """

    bins: int
    edges: Dict[str, Tuple[float, ...]]

    def bin_count(self, name: str) -> int:
        return len(self.edges.get(name, ())) + 1


def fit_bins(rows: Sequence[Mapping[str, Optional[float]]],
             bins: int) -> BinTable:
    """Fit quantile bin edges on ``rows``. Never reads a label or a future.

    Quantiles rather than fixed constants: every bin then holds roughly equal
    training mass, which is the property that lets a bin's forward returns be
    averaged into an estimate at all. A log-spaced constant grid puts most of
    a quiet symbol's bars into one bin and most of a volatile one's into
    another, and the model's capacity silently follows the volatility regime.
    """
    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins}")
    edges: Dict[str, Tuple[float, ...]] = {}
    for name in FEATURE_NAMES:
        values = sorted(v for v in (row.get(name) for row in rows)
                        if v is not None and math.isfinite(v))
        if len(values) < bins:
            edges[name] = ()
            continue
        cuts: List[float] = []
        for k in range(1, bins):
            candidate = values[min(len(values) - 1, (k * len(values)) // bins)]
            # Strictly increasing only: a repeated quantile means a value mass
            # wider than one bin, and emitting it twice would create an empty
            # bin rather than splitting that mass.
            if not cuts or candidate > cuts[-1]:
                cuts.append(candidate)
        edges[name] = tuple(cuts)
    return BinTable(bins=bins, edges=edges)


def digitize(row: Mapping[str, Optional[float]],
             table: BinTable) -> Dict[str, int]:
    """Map a scalar row to its bin indices. ``None`` -> ``MISSING_BIN``."""
    out: Dict[str, int] = {}
    for name in FEATURE_NAMES:
        value = row.get(name)
        if value is None or not math.isfinite(value):
            out[name] = MISSING_BIN
            continue
        cuts = table.edges.get(name, ())
        low, high = 0, len(cuts)
        while low < high:  # bisect_right, inlined to avoid a tuple copy
            mid = (low + high) // 2
            if value < cuts[mid]:
                high = mid
            else:
                low = mid + 1
        out[name] = low
    return out


def signature(binned: Mapping[str, int]) -> str:
    """The joint key across every feature -- the thing that must NOT be unique.

    Provided so callers can measure their own collision rate. A corpus whose
    signatures are all distinct cannot generalise however good the learner is,
    and that is a property of the binning, measurable before any training.
    """
    return "|".join(f"{name}={binned.get(name, MISSING_BIN)}"
                    for name in FEATURE_NAMES)
