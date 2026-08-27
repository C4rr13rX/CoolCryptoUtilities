"""Scale-invariant chart-shape extraction and matching.

The premise, in Adam's words: a pattern that "to the human eye would look
generally like the same structure on a crypto chart over time/history, but
could appear as different sizes". Two moves are the same SHAPE when their
normalised trajectories agree within a margin, regardless of absolute price,
absolute duration, or absolute amplitude.

So a window is reduced to a scale-free signature:

  * resampled to a fixed number of points   -> duration invariance
  * z-normalised across the window          -> amplitude/price invariance
  * compared by mean absolute deviation     -> a single distance in sigma units

The DYNAMIC MARGIN is the tolerance on that distance. A tight margin finds few,
very similar shapes; a loose one finds many, vaguer ones. It is a GA-tunable
parameter rather than a constant, because the right tolerance is an empirical
question and differs per regime.

Shapes are only useful if they PREDICT. Every discovered shape carries the
forward-return distribution of its historical occurrences, and a shape is
retained only when that distribution is consistent -- a shape whose outcomes
are a coin flip is a pattern in the noise, which is exactly what an
unconstrained search over 3 years of bars will otherwise produce in bulk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def resample(series: Sequence[float], points: int) -> Optional[np.ndarray]:
    """Linearly resample to a fixed length -> duration invariance."""
    arr = np.asarray([float(x) for x in series], dtype=float)
    if arr.size < 2 or points < 2:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    src = np.linspace(0.0, 1.0, arr.size)
    dst = np.linspace(0.0, 1.0, points)
    return np.interp(dst, src, arr)


def normalize(window: Sequence[float], points: int = 16) -> Optional[np.ndarray]:
    """Scale-free signature: resample, then z-normalise.

    Returns None for a flat window: zero variance has no shape, and dividing
    by its std would manufacture one out of floating-point noise.
    """
    resampled = resample(window, points)
    if resampled is None:
        return None
    mean = float(np.mean(resampled))
    std = float(np.std(resampled))
    if not math.isfinite(std) or std <= 1e-12:
        return None
    return (resampled - mean) / std


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute deviation in sigma units. 0.0 == identical shape."""
    if a is None or b is None or a.shape != b.shape:
        return float("inf")
    return float(np.mean(np.abs(a - b)))


def matches(a: np.ndarray, b: np.ndarray, margin: float) -> bool:
    return distance(a, b) <= margin


@dataclass
class ShapeCluster:
    """A recurring chart shape and what happened next, historically."""

    signature: np.ndarray
    occurrences: int = 0
    forward_returns: List[float] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)

    def absorb(self, sig: np.ndarray, forward_return: float, symbol: str) -> None:
        # Running mean keeps the centroid representative as members join.
        n = self.occurrences
        self.signature = (self.signature * n + sig) / (n + 1)
        self.occurrences = n + 1
        self.forward_returns.append(float(forward_return))
        if symbol not in self.symbols:
            self.symbols.append(symbol)

    # -- what makes a shape worth trading -----------------------------------

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.forward_returns)) if self.forward_returns else 0.0

    @property
    def median_return(self) -> float:
        return float(np.median(self.forward_returns)) if self.forward_returns else 0.0

    @property
    def hit_rate(self) -> float:
        """Share of occurrences that moved in the mean direction."""
        if not self.forward_returns:
            return 0.0
        sign = 1.0 if self.mean_return >= 0 else -1.0
        return sum(1 for r in self.forward_returns if r * sign > 0) / len(self.forward_returns)

    @property
    def consistency(self) -> float:
        """|mean| / std -- effect size, not just direction.

        This is the guard against pattern-mining noise. A shape that averages
        +2% with a 20% spread is not a signal; one that averages +2% with a
        1% spread is. Scale-free, so it is comparable across symbols.
        """
        if len(self.forward_returns) < 2:
            return 0.0
        std = float(np.std(self.forward_returns))
        if std <= 1e-12:
            # Zero spread is PERFECT consistency, not absent consistency.
            # Returning 0.0 here ranked a shape whose outcomes never varied
            # below one that was merely noisy, which inverts the whole point
            # of the measure. A non-zero mean with no spread is the strongest
            # signal the data can express; a zero mean with no spread is a
            # shape that reliably does nothing, and stays worthless.
            # Large but finite: inf breaks the ranking sort and does not
            # survive JSON serialisation into the run/model registries.
            return 999.0 if abs(self.mean_return) > 1e-12 else 0.0
        return abs(self.mean_return) / std

    @property
    def symbol_dominance(self) -> float:
        """Guard against a 'shape' that is really one symbol's quirk."""
        if not self.forward_returns:
            return 1.0
        return 1.0 / max(1, len(self.symbols))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": [round(float(x), 5) for x in self.signature],
            "occurrences": self.occurrences,
            "mean_return": self.mean_return,
            "median_return": self.median_return,
            "hit_rate": self.hit_rate,
            "consistency": self.consistency,
            "symbols": list(self.symbols),
        }


def extract_windows(
    bars: Sequence[Dict[str, Any]],
    *,
    window: int,
    horizon: int,
    stride: int = 1,
    price_key: str = "close",
) -> List[Tuple[np.ndarray, float]]:
    """(signature, forward_return) pairs from one symbol's bar history.

    The forward return is measured AFTER the window closes, so a signature
    never contains any information about the outcome it is labelled with.
    """
    out: List[Tuple[np.ndarray, float]] = []
    prices = [float(b.get(price_key, 0.0) or 0.0) for b in bars]
    n = len(prices)
    for i in range(0, n - window - horizon, max(1, stride)):
        seg = prices[i : i + window]
        entry = prices[i + window - 1]
        exit_px = prices[i + window - 1 + horizon]
        if entry <= 0 or exit_px <= 0 or any(p <= 0 for p in seg):
            continue
        sig = normalize(seg)
        if sig is None:
            continue
        out.append((sig, (exit_px / entry) - 1.0))
    return out


def cluster_shapes(
    samples: Sequence[Tuple[np.ndarray, float]],
    *,
    margin: float,
    symbol: str = "",
    max_clusters: int = 400,
) -> List[ShapeCluster]:
    """Greedy single-pass clustering under the dynamic margin.

    Greedy rather than k-means on purpose: the number of distinct shapes is
    not known in advance, and the margin -- not a cluster count -- is the
    parameter the GA is meant to tune.
    """
    clusters: List[ShapeCluster] = []
    for sig, fwd in samples:
        best: Optional[ShapeCluster] = None
        best_d = float("inf")
        for c in clusters:
            d = distance(sig, c.signature)
            if d < best_d:
                best_d, best = d, c
        if best is not None and best_d <= margin:
            best.absorb(sig, fwd, symbol)
        elif len(clusters) < max_clusters:
            c = ShapeCluster(signature=np.array(sig, dtype=float))
            c.absorb(sig, fwd, symbol)
            clusters.append(c)
    return clusters


def select_predictive(
    clusters: Sequence[ShapeCluster],
    *,
    min_occurrences: int = 8,
    min_consistency: float = 0.35,
    min_abs_return: float = 0.002,
    min_symbols: int = 1,
) -> List[ShapeCluster]:
    """Keep only shapes whose history is consistent enough to act on.

    Without this a search over 3 years of bars returns hundreds of "patterns"
    that are pure sampling noise. Requiring repeat occurrences, a real effect
    size, and (optionally) presence across multiple symbols is what separates
    a shape from a coincidence.
    """
    keep = [
        c
        for c in clusters
        if c.occurrences >= min_occurrences
        and c.consistency >= min_consistency
        and abs(c.mean_return) >= min_abs_return
        and len(c.symbols) >= min_symbols
    ]
    keep.sort(key=lambda c: c.consistency * math.sqrt(c.occurrences), reverse=True)
    return keep


def best_match(
    sig: np.ndarray,
    clusters: Sequence[ShapeCluster],
    *,
    margin: float,
) -> Optional[Tuple[ShapeCluster, float]]:
    """Closest shape within the margin, or None when nothing is close."""
    best: Optional[ShapeCluster] = None
    best_d = float("inf")
    for c in clusters:
        d = distance(sig, c.signature)
        if d < best_d:
            best_d, best = d, c
    if best is None or best_d > margin:
        return None
    return best, best_d
