from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class OpportunitySignal:
    symbol: str
    kind: str  # "buy-low" or "sell-high"
    zscore: float
    price: float
    mean_price: float
    std_price: float
    timestamp: float

    def to_dict(self) -> Dict[str, float]:
        payload = asdict(self)
        return payload


class OpportunityTracker:
    def __init__(self, *, min_points: int = 40, zscore_threshold: float = 1.5) -> None:
        # Allow env override of warmup gate so buy-low/sell-high can
        # fire after fewer samples when starting fresh.  The math is
        # still trustworthy with 16 points — z-score over a 16-tick
        # window catches 2-3% fluctuations the user is targeting.
        env_min = int(os.getenv("OPPORTUNITY_MIN_POINTS", str(min_points)))
        self.min_points = max(16, env_min)
        env_z = float(os.getenv("OPPORTUNITY_ZSCORE_THRESHOLD", str(zscore_threshold)))
        self.zscore_threshold = max(0.5, env_z)
        #: How long a same-kind signal may suppress its own successor.
        #:
        #: The consumer DELETES what we send it. TradeScheduler stores the
        #: signal in ``_opportunity_bias`` and drops any entry older than
        #: 1800s (scheduler.py:1505-1506), while ``_money_button_candidate``
        #: refuses outright when that entry is missing (scheduler.py:1665).
        #: The debounce below, left unbounded, never decays its reference
        #: point -- so a symbol whose z-score holds still emits ONCE and
        #: then goes quiet, and half an hour later the consumer has nothing
        #: and cannot get it back. A mean-reverting 2-5% drawdown is exactly
        #: the shape that holds its z-score still, which makes the two
        #: halves disagree about lifetime precisely on the condition the
        #: money button exists to trade.
        #:
        #: Measured, 168h over 150 symbols
        #: (scripts/opportunity_precondition_census.py): 327 refusals landed
        #: while the consumer held nothing, and bounding this recovers 61 of
        #: 1564 dip ticks -- 45.6% armed to 49.5%. Small, real, and in the
        #: right direction; it is NOT the reason the lane is quiet.
        #:
        #: Default matches the consumer's expiry so the producer cannot be
        #: the reason the bias is absent.
        self.refresh_sec = max(0.0, float(os.getenv("OPPORTUNITY_REFRESH_SEC", "1800")))
        self._last_signal: Dict[str, OpportunitySignal] = {}

    def evaluate(self, symbol: str, prices: np.ndarray) -> Optional[OpportunitySignal]:
        if prices.size < self.min_points:
            return None
        series = np.asarray(prices, dtype=np.float64)
        mean_price = float(np.mean(series))
        std_price = float(np.std(series))
        if std_price < 1e-9:
            return None
        current = float(series[-1])
        zscore = (current - mean_price) / std_price
        if abs(zscore) < self.zscore_threshold:
            return None
        kind = "buy-low" if zscore < 0 else "sell-high"
        existing = self._last_signal.get(symbol)
        if existing and existing.kind == kind and abs(existing.zscore - zscore) < 0.2:
            # Suppress the duplicate only while the consumer still holds the
            # original. Once it is older than the window in which the
            # scheduler keeps it, silence stops being deduplication and
            # starts being starvation: the condition is still true and the
            # only component that can say so has muted itself.
            age = time.time() - float(existing.timestamp)
            if self.refresh_sec <= 0.0 or age < self.refresh_sec:
                return None
        signal = OpportunitySignal(
            symbol=symbol,
            kind=kind,
            zscore=float(zscore),
            price=current,
            mean_price=mean_price,
            std_price=std_price,
            timestamp=time.time(),
        )
        self._last_signal[symbol] = signal
        return signal
