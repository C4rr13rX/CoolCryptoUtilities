"""EMA cross strategy.

Golden cross (fast EMA over slow, slow rising) enters; death cross exits.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from trading.strategies.base import Strategy, StrategyContext, ema, env_float, sample_arrays


class EmaCrossStrategy(Strategy):
    strategy_id = "ema_cross"
    default_horizon = "1h"
    min_samples = 40

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        fast_span = int(env_float("EMA_CROSS_FAST_SPAN", 12, lo=3, hi=100))
        slow_span = int(env_float("EMA_CROSS_SLOW_SPAN", 48, lo=8, hi=400))
        min_net = env_float("EMA_CROSS_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)

        _, prices, _ = sample_arrays(state)
        if prices.size < max(self.min_samples, slow_span) or ctx.last_price <= 0:
            return None
        fast = ema(prices, fast_span)
        slow = ema(prices, slow_span)
        f_now, s_now = float(fast[-1]), float(slow[-1])
        f_prev, s_prev = float(fast[-2]), float(slow[-2])
        if s_now <= 0:
            return None
        spread = (f_now - s_now) / s_now
        # Slow must not be falling at the cross tick — a rising-slow-only
        # filter would reject V-bottom golden crosses, which are exactly the
        # buy-low moments this strategy exists for.
        slow_rising = s_now >= s_prev

        golden = f_prev <= s_prev and f_now > s_now
        death = f_prev >= s_prev and f_now < s_now

        if golden and slow_rising and ctx.available_quote > 0:
            # Expected: momentum carries roughly 2x the crossover spread.
            expected = max(abs(spread) * 2.0, min_net + ctx.fee_rate)
            expected = min(expected, 0.05)
            confidence = min(0.8, 0.55 + abs(spread) * 20.0)
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=ctx.last_price * (1.0 + expected),
                confidence=confidence,
                direction_prob=confidence,
                reason=f"golden cross, spread {spread:.3%}",
                extra_meta={"ema_spread": spread},
            )

        if death and ctx.available_base > 0:
            expected = max(abs(spread) * 2.0, min_net + ctx.fee_rate)
            expected = min(expected, 0.05)
            confidence = min(0.8, 0.55 + abs(spread) * 20.0)
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"death cross, spread {spread:.3%}",
                extra_meta={"ema_spread": spread},
            )
        return None
