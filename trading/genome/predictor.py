"""Turn a champion genome into a bus-scheduler placement.

Closes the loop Adam described: shape + sentiment + brain are cycled back as a
single prediction metric that decides whether to buy low now, what sell-high
margin to expect, and by when -- and refuses the trade when volume cannot
support exiting at that margin before the profit is gone.

The volume check is the part that is easy to leave out and expensive to omit.
A predicted +4% means nothing if the pool cannot absorb the exit: the fill
walks the book down and the margin evaporates. So a placement carries the size
it was validated for, and is refused when the recent volume cannot clear it.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from trading.genome.shapes import best_match, normalize, ShapeCluster


@dataclass
class Placement:
    """A buy-low/sell-high intent, sized and time-boxed."""

    symbol: str
    action: str                  # "enter" | "hold" | "skip"
    reason: str = ""
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_price: float = 0.0
    expected_return: float = 0.0
    confidence: float = 0.0
    horizon_sec: float = 0.0
    size_usd: float = 0.0
    shape_occurrences: int = 0
    shape_consistency: float = 0.0
    sentiment: float = 0.0
    brain_agreement: Optional[bool] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_directive(self) -> Dict[str, Any]:
        """Shape the scheduler consumes (see trading/scheduler.py)."""
        return {
            "action": self.action,
            "symbol": self.symbol,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "expected_return": self.expected_return,
            "confidence": self.confidence,
            "horizon": "%ds" % int(self.horizon_sec),
            "size_usd": self.size_usd,
            "reason": self.reason,
            "strategy_id": "genome_shape",
            "meta": dict(self.meta),
        }


def _volume_supports(
    recent_volume_usd: float,
    size_usd: float,
    min_ratio: float,
) -> bool:
    """Can the pool absorb this size without eating the expected margin?

    ``min_ratio`` is the multiple of the trade size that recent volume must
    show. At 0 the check is disabled; the GA tunes it because the right
    threshold differs by venue and regime.
    """
    if min_ratio <= 0.0:
        return True
    if size_usd <= 0.0:
        return False
    return recent_volume_usd >= size_usd * min_ratio


def predict(
    *,
    symbol: str,
    prices: Sequence[float],
    genes: Dict[str, Any],
    clusters: Sequence[ShapeCluster],
    recent_volume_usd: float = 0.0,
    sentiment: float = 0.0,
    brain_answer: Optional[str] = None,
    brain_confidence: float = 0.0,
    size_usd: float = 0.0,
    bar_seconds: float = 3600.0,
) -> Placement:
    """One prediction, with every refusal reason stated explicitly."""
    window = int(genes.get("window", 24))
    margin = float(genes.get("shape_margin", 0.35))
    target_margin = float(genes.get("target_margin", 0.02))
    stop_margin = float(genes.get("stop_margin", 0.02))
    horizon = int(genes.get("horizon", 6))
    entry_pct = float(genes.get("entry_percentile", 0.25))
    sent_w = float(genes.get("sentiment_weight", 0.0))
    sent_margin = float(genes.get("sentiment_margin", 0.2))
    brain_w = float(genes.get("brain_weight", 0.0))
    brain_min_conf = float(genes.get("brain_min_confidence", 0.0))
    min_vol_ratio = float(genes.get("min_volume_ratio", 0.0))

    if len(prices) < window:
        return Placement(symbol=symbol, action="skip", reason="insufficient_history")

    seg = list(prices[-window:])
    price = float(seg[-1])
    if price <= 0:
        return Placement(symbol=symbol, action="skip", reason="no_price")

    sig = normalize(seg, points=int(genes.get("shape_points", 16)))
    if sig is None:
        return Placement(symbol=symbol, action="skip", reason="flat_window_no_shape")

    match = best_match(sig, clusters, margin=margin)
    if match is None:
        return Placement(symbol=symbol, action="skip", reason="no_matching_shape")
    cluster, dist = match

    expected = float(cluster.mean_return)
    if expected <= 0:
        return Placement(
            symbol=symbol, action="skip", reason="shape_predicts_down",
            expected_return=expected, shape_occurrences=cluster.occurrences,
        )

    # "Buy low": only enter in the lower part of the window's own range, so a
    # bullish shape does not become a chase at the top of the move.
    lo, hi = float(np.min(seg)), float(np.max(seg))
    if hi > lo:
        position = (price - lo) / (hi - lo)
        if position > entry_pct + 0.5:
            return Placement(
                symbol=symbol, action="skip",
                reason="not_low_in_range(%.2f)" % position,
                expected_return=expected,
            )

    # News sentiment as a dynamic-margin veto.
    if sent_w > 0.0 and abs(sentiment) >= sent_margin and sentiment < 0:
        return Placement(
            symbol=symbol, action="skip", reason="sentiment_contradicts",
            sentiment=sentiment, expected_return=expected,
        )

    # Brain read, cycled back in.
    agreement: Optional[bool] = None
    if brain_w > 0.0 and brain_answer and brain_confidence >= brain_min_conf:
        agreement = brain_answer in ("win", "win_big")
        if not agreement:
            return Placement(
                symbol=symbol, action="skip", reason="brain_disagrees",
                brain_agreement=False, expected_return=expected,
            )

    # The margin must clear fees, or the trade is work for nothing.
    fee = float(os.getenv("GENOME_FEE_RATE", "0.0065"))
    effective_target = max(target_margin, expected)
    if effective_target <= fee:
        return Placement(
            symbol=symbol, action="skip",
            reason="margin_below_fees(%.4f<=%.4f)" % (effective_target, fee),
            expected_return=expected,
        )

    # Volume must support exiting at that margin.
    if not _volume_supports(recent_volume_usd, size_usd, min_vol_ratio):
        return Placement(
            symbol=symbol, action="skip",
            reason="volume_too_thin(%.0f<%.0fx%.2f)" % (
                recent_volume_usd, size_usd, min_vol_ratio),
            expected_return=expected, size_usd=size_usd,
        )

    confidence = min(0.95, max(0.05, cluster.consistency / 2.0)) if cluster.consistency < 999 else 0.95
    if sent_w > 0.0 and sentiment > 0:
        confidence = min(0.95, confidence * (1.0 + sent_w * sentiment))

    return Placement(
        symbol=symbol,
        action="enter",
        reason="shape n=%d consistency=%.2f dist=%.3f" % (
            cluster.occurrences, min(cluster.consistency, 999.0), dist),
        entry_price=price,
        target_price=price * (1.0 + effective_target),
        stop_price=price * (1.0 - stop_margin),
        expected_return=effective_target,
        confidence=confidence,
        horizon_sec=horizon * bar_seconds,
        size_usd=size_usd,
        shape_occurrences=cluster.occurrences,
        shape_consistency=min(cluster.consistency, 999.0),
        sentiment=sentiment,
        brain_agreement=agreement,
        meta={
            "shape_distance": dist,
            "shape_hit_rate": cluster.hit_rate,
            "shape_symbols": list(cluster.symbols)[:6],
            "recent_volume_usd": recent_volume_usd,
        },
    )
