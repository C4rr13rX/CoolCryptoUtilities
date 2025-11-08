from __future__ import annotations

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
        self.min_points = max(16, int(min_points))
        self.zscore_threshold = max(0.5, float(zscore_threshold))
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
