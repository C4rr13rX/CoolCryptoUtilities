from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ArbSignal:
    spread: float
    implied_edge: float
    action: str  # "buy_eth" / "sell_eth" / "hold"
    confidence: float


class KalmanFilter:
    def __init__(self, process_var: float = 1e-5, measurement_var: float = 1e-2) -> None:
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.estimate = 0.0
        self.variance = 1.0

    def update(self, measurement: float) -> float:
        self.variance += self.process_var
        kalman_gain = self.variance / (self.variance + self.measurement_var)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.variance = (1 - kalman_gain) * self.variance
        return self.estimate


class VolatilityArbCell:
    """
    Focused brain cell for ETH ↔ stablecoin oscillations.

    Uses dual Kalman filters to estimate fair value and volatility, generating
    small opportunistic trades during whipsaw periods.
    """

    def __init__(self) -> None:
        self.price_filter = KalmanFilter()
        self.vol_filter = KalmanFilter(process_var=1e-4, measurement_var=1e-2)
        self.spread_history: List[float] = []

    def observe(self, eth_price: float, usdc_price: float) -> ArbSignal:
        ratio = eth_price / max(usdc_price, 1e-6)
        fair = self.price_filter.update(ratio)
        spread = ratio - fair
        self.spread_history.append(spread)
        if len(self.spread_history) > 200:
            self.spread_history.pop(0)
        vol = np.std(self.spread_history) if len(self.spread_history) > 10 else 0.0
        vol_est = abs(self.vol_filter.update(vol))
        threshold = vol_est * 1.2 + 1e-4
        if spread > threshold:
            action = "sell_eth"
            implied = spread
        elif spread < -threshold:
            action = "buy_eth"
            implied = -spread
        else:
            action = "hold"
            implied = 0.0
        confidence = float(np.clip(implied / (vol_est + 1e-6), 0.0, 1.0))
        return ArbSignal(spread=spread, implied_edge=implied, action=action, confidence=confidence)
