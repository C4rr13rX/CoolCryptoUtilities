"""Bollinger squeeze strategy.

Volatility contraction (bandwidth in the low quantile of its own history)
followed by a band break signals expansion: break up enters, break down exits.
Bandwidth history is kept per symbol because one scheduler tracks many routes.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from trading.strategies.base import Strategy, StrategyContext, env_float, sample_arrays


class BollingerSqueezeStrategy(Strategy):
    strategy_id = "bollinger_squeeze"
    default_horizon = "45m"
    min_samples = 40

    def __init__(self) -> None:
        self._bandwidth_history: Dict[str, Deque[float]] = {}

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        window = int(env_float("BOLLINGER_WINDOW", 20, lo=10, hi=100))
        n_std = env_float("BOLLINGER_STD", 2.0, lo=1.0, hi=4.0)
        squeeze_q = env_float("BOLLINGER_SQUEEZE_QUANTILE", 0.25, lo=0.05, hi=0.6)
        min_net = env_float("BOLLINGER_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)

        _, prices, _ = sample_arrays(state)
        if prices.size < max(self.min_samples, window + 2) or ctx.last_price <= 0:
            return None
        tail = prices[-window:]
        mid = float(np.mean(tail))
        std = float(np.std(tail))
        if mid <= 0:
            return None
        upper = mid + n_std * std
        lower = mid - n_std * std
        bandwidth = (upper - lower) / mid

        history = self._bandwidth_history.setdefault(state.symbol, deque(maxlen=512))
        history.append(bandwidth)
        if len(history) < 30:
            return None
        squeeze_floor = float(np.quantile(np.asarray(history, dtype=np.float64), squeeze_q))
        # Squeeze = the bandwidth of the bars *before* the break was compressed.
        prev_tail = prices[-window - 1:-1]
        prev_mid = float(np.mean(prev_tail))
        prev_std = float(np.std(prev_tail))
        prev_bandwidth = (2.0 * n_std * prev_std) / max(prev_mid, 1e-12)
        in_squeeze = prev_bandwidth <= squeeze_floor
        if not in_squeeze:
            return None

        # Expansion move expectation: half the band width.
        expected = max(bandwidth * 0.5, 0.0)
        if expected - ctx.fee_rate < min_net:
            return None
        confidence = min(0.8, 0.55 + max(0.0, (squeeze_floor - prev_bandwidth) / max(squeeze_floor, 1e-9)) * 0.3)

        if ctx.last_price > upper and ctx.available_quote > 0:
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=ctx.last_price * (1.0 + expected),
                confidence=confidence,
                direction_prob=confidence,
                reason=f"squeeze break up (bw {prev_bandwidth:.3%} <= q{squeeze_q:.0%} {squeeze_floor:.3%})",
                extra_meta={"bandwidth": bandwidth, "squeeze_floor": squeeze_floor},
            )
        if ctx.last_price < lower and ctx.available_base > 0:
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"squeeze break down, avoiding ~{expected:.2%}",
                extra_meta={"bandwidth": bandwidth, "squeeze_floor": squeeze_floor},
            )
        return None
