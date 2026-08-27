"""Crypto SELECTION features, searchable by the GA and scored on history.

Separate from direction. Two different questions:

    which crypto is worth trading right now   <- here
    which way will it go                      <- the brain

C0d3rV2's ``score_pair`` answers the first one badly. Measured 2026-08-27 over
131,200 real bars, its top-scoring decile returned **-0.00086 per trade against
a -0.00023 baseline** with a 46.3% win rate versus 48.3%: its picks did WORSE
than choosing at random. It weights momentum and buy pressure, which mean-revert
at this horizon, so it reliably buys the top of a move.

Rather than hand-tune another set of weights, every feature here is a gene the
GA can weight, invert, or zero out, and ``evaluate_selector`` scores a weight
vector on withheld data using the same lift-over-baseline measure that caught
score_pair. A selector that cannot beat random selection scores zero.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Features. Each maps a window of bars -> a scale-free number.
# Sign convention: POSITIVE means "more attractive to buy".
# --------------------------------------------------------------------------


def _closes(bars: Sequence[Dict[str, Any]]) -> List[float]:
    return [float(b.get("close") or 0.0) for b in bars]


def f_reversion(bars: Sequence[Dict[str, Any]]) -> float:
    """How far BELOW its recent mean the price sits (z-score, inverted).

    The opposite of score_pair's momentum term, and the reason this module
    exists: at a 6-bar horizon the measured edge is in buying weakness, not
    chasing strength.
    """
    closes = _closes(bars)
    if len(closes) < 8:
        return 0.0
    window = closes[-24:] if len(closes) >= 24 else closes
    mean = statistics.mean(window)
    std = statistics.pstdev(window)
    if std <= 1e-12 or mean <= 0:
        return 0.0
    return float(-((closes[-1] - mean) / std))


def f_momentum(bars: Sequence[Dict[str, Any]]) -> float:
    """Recent return. Kept so the GA can decide its sign empirically."""
    closes = _closes(bars)
    if len(closes) < 7:
        return 0.0
    prev = closes[-7]
    if prev <= 0:
        return 0.0
    return float((closes[-1] / prev) - 1.0)


def f_buy_pressure(bars: Sequence[Dict[str, Any]]) -> float:
    """Buy share of volume, centred on zero."""
    buys = sum(float(b.get("buy_volume") or 0.0) for b in bars[-6:])
    sells = sum(float(b.get("sell_volume") or 0.0) for b in bars[-6:])
    total = buys + sells
    if total <= 0:
        return 0.0
    return float((buys / total) - 0.5) * 2.0


def f_volume_surge(bars: Sequence[Dict[str, Any]]) -> float:
    """Recent volume against its own baseline -- attention, not direction."""
    vols = [float(b.get("buy_volume") or 0.0) + float(b.get("sell_volume") or 0.0) for b in bars]
    if len(vols) < 24:
        return 0.0
    recent = statistics.mean(vols[-6:])
    base = statistics.mean(vols[-24:])
    if base <= 0:
        return 0.0
    return float(math.log10(max(0.1, recent / base)))


def f_volatility(bars: Sequence[Dict[str, Any]]) -> float:
    """Realised volatility. Range is opportunity; the GA sets the sign."""
    closes = _closes(bars)
    if len(closes) < 12:
        return 0.0
    rets = [
        (closes[i] / closes[i - 1]) - 1.0
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    if len(rets) < 4:
        return 0.0
    return float(statistics.pstdev(rets[-24:]) if len(rets) >= 24 else statistics.pstdev(rets))


def f_drawdown(bars: Sequence[Dict[str, Any]]) -> float:
    """Distance below the recent high -- 'on sale' relative to itself."""
    closes = _closes(bars)
    if len(closes) < 12:
        return 0.0
    window = closes[-24:] if len(closes) >= 24 else closes
    peak = max(window)
    if peak <= 0:
        return 0.0
    return float((peak - closes[-1]) / peak)


def f_trend_slope(bars: Sequence[Dict[str, Any]]) -> float:
    """Normalised least-squares slope over the window."""
    closes = _closes(bars)
    if len(closes) < 12:
        return 0.0
    window = closes[-24:] if len(closes) >= 24 else closes
    n = len(window)
    xs = np.arange(n, dtype=float)
    ys = np.asarray(window, dtype=float)
    mean = float(ys.mean())
    if mean <= 0:
        return 0.0
    slope = float(np.polyfit(xs, ys, 1)[0])
    return slope / mean * n


def f_range_position(bars: Sequence[Dict[str, Any]]) -> float:
    """Where in its own recent range the price sits, inverted (low = good)."""
    closes = _closes(bars)
    if len(closes) < 12:
        return 0.0
    window = closes[-24:] if len(closes) >= 24 else closes
    lo, hi = min(window), max(window)
    if hi <= lo:
        return 0.0
    return float(1.0 - ((closes[-1] - lo) / (hi - lo)))


FEATURES: Dict[str, Callable[[Sequence[Dict[str, Any]]], float]] = {
    "reversion": f_reversion,
    "momentum": f_momentum,
    "buy_pressure": f_buy_pressure,
    "volume_surge": f_volume_surge,
    "volatility": f_volatility,
    "drawdown": f_drawdown,
    "trend_slope": f_trend_slope,
    "range_position": f_range_position,
}

#: Weight range per feature. Spans negative so the GA can INVERT a feature --
#: the honest way to discover that momentum should be sold, not bought.
SELECTOR_GENE_SPACE: Dict[str, Tuple[float, float]] = {
    "w_%s" % name: (-1.0, 1.0) for name in FEATURES
}


def feature_vector(bars: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, fn in FEATURES.items():
        try:
            value = float(fn(bars))
        except Exception:
            value = 0.0
        out[name] = value if math.isfinite(value) else 0.0
    return out


def selector_score(bars: Sequence[Dict[str, Any]], weights: Dict[str, float]) -> float:
    """Weighted sum of features. Higher = more attractive to buy."""
    vector = feature_vector(bars)
    total = 0.0
    for name, value in vector.items():
        total += float(weights.get("w_%s" % name, 0.0)) * value
    return total


# --------------------------------------------------------------------------
# Scoring a selector on history
# --------------------------------------------------------------------------


@dataclass
class SelectorResult:
    lift: float = 0.0
    top_mean: float = 0.0
    baseline_mean: float = 0.0
    top_win_rate: float = 0.0
    baseline_win_rate: float = 0.0
    samples: int = 0
    selected: int = 0
    passed: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lift": self.lift, "top_mean": self.top_mean,
            "baseline_mean": self.baseline_mean, "top_win_rate": self.top_win_rate,
            "baseline_win_rate": self.baseline_win_rate, "samples": self.samples,
            "selected": self.selected, "passed": self.passed, "reason": self.reason,
        }


def evaluate_selector(
    weights: Dict[str, float],
    bars_by_symbol: Dict[str, Sequence[Dict[str, Any]]],
    *,
    window: int = 24,
    horizon: int = 6,
    top_fraction: float = 0.10,
    split: float = 0.7,
    fee_rate: float = 0.0065,
    min_samples: int = 200,
) -> SelectorResult:
    """Lift of the selector's top picks over picking at random, out-of-sample.

    The same measure that caught score_pair. Scored ONLY on the held-out tail,
    because a selection rule tuned on the data it is scored against will always
    look good -- that is how a losing rule survives review.
    """
    scored: List[Tuple[float, float]] = []      # (score, forward_return)
    for _symbol, bars in bars_by_symbol.items():
        if len(bars) < window + horizon + 10:
            continue
        cut = int(len(bars) * split)
        for i in range(max(window, cut), len(bars) - horizon):
            history = bars[i - window:i + 1]
            entry = float(bars[i].get("close") or 0.0)
            future = float(bars[i + horizon].get("close") or 0.0)
            if entry <= 0 or future <= 0:
                continue
            scored.append((selector_score(history, weights), (future / entry) - 1.0 - fee_rate))

    result = SelectorResult(samples=len(scored))
    if len(scored) < min_samples:
        result.reason = "insufficient_samples"
        return result

    returns = [r for _s, r in scored]
    result.baseline_mean = float(statistics.mean(returns))
    result.baseline_win_rate = sum(1 for r in returns if r > 0) / len(returns)

    scored.sort(key=lambda pair: -pair[0])
    k = max(1, int(len(scored) * top_fraction))
    top = [r for _s, r in scored[:k]]
    result.selected = len(top)
    result.top_mean = float(statistics.mean(top))
    result.top_win_rate = sum(1 for r in top if r > 0) / len(top)
    result.lift = result.top_mean - result.baseline_mean

    if result.lift <= 0:
        result.reason = "no lift over random selection"
        return result
    if result.top_mean <= 0:
        result.reason = "top picks lose money after fees"
        return result
    result.passed = True
    result.reason = "top %.0f%% beats random by %+.5f/trade" % (top_fraction * 100, result.lift)
    return result


def selector_fitness(
    weights: Dict[str, float],
    bars_by_symbol: Dict[str, Sequence[Dict[str, Any]]],
    **kwargs: Any,
) -> float:
    """GA fitness: lift x profitability, zero unless BOTH are positive.

    A selector that picks winners but loses money after fees is worthless, and
    so is one that makes money by accident without beating random selection.
    """
    result = evaluate_selector(weights, bars_by_symbol, **kwargs)
    if not result.passed:
        return 0.0
    confidence = math.sqrt(min(1.0, result.selected / 500.0))
    return result.lift * result.top_mean * 10000.0 * confidence
