"""Swarm consensus strategy.

Promotes the MultiResolutionSwarm's Borda-vote consensus (trading/brain/swarm.py)
from a small additive bias into a standalone strategy: when the horizon cells
agree strongly enough (low disagreement entropy, decent confidence), the
consensus expected return becomes a directive of its own. This gives the
system a fully TF-independent entry/exit signal.

The bot publishes the latest consensus into the scheduler each tick via
``scheduler.external_signals["swarm_consensus"]`` (a plain dict) — this
strategy is a lookup + arithmetic, never a computation over raw windows.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from trading.strategies.base import Strategy, StrategyContext, env_float


class SwarmConsensusStrategy(Strategy):
    strategy_id = "swarm_consensus"
    default_horizon = "swarm"
    min_samples = 12

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        consensus = (ctx.extras or {}).get("swarm_consensus")
        if not consensus:
            return None
        # Accept either a ConsensusResult-like object or a plain dict.
        get = consensus.get if isinstance(consensus, dict) else lambda k, d=None: getattr(consensus, k, d)
        try:
            expected_raw = float(get("expected_return", 0.0) or 0.0)
            confidence = float(get("confidence", 0.0) or 0.0)
            direction_prob = float(get("direction_prob", 0.5) or 0.5)
            entropy = float(get("entropy", 1.0) or 1.0)
            horizon_count = int(get("horizon_count", 0) or 0)
        except (TypeError, ValueError):
            return None

        min_conf = env_float("SWARM_STRATEGY_MIN_CONFIDENCE", 0.55, lo=0.0, hi=1.0)
        max_entropy = env_float("SWARM_STRATEGY_MAX_ENTROPY", 0.7, lo=0.0, hi=1.0)
        min_net = env_float("SWARM_STRATEGY_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)
        cap = env_float("SWARM_STRATEGY_RETURN_CAP", 0.08, lo=0.01, hi=0.5)

        if horizon_count < 2 or confidence < min_conf or entropy > max_entropy:
            return None
        expected = max(-cap, min(cap, expected_raw))
        magnitude = abs(expected)
        if magnitude - ctx.fee_rate < min_net or ctx.last_price <= 0:
            return None

        if expected > 0 and ctx.available_quote > 0:
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=magnitude,
                target_price=ctx.last_price * (1.0 + magnitude),
                confidence=confidence,
                direction_prob=direction_prob,
                reason=f"consensus +{magnitude:.2%} across {horizon_count} horizons (entropy {entropy:.2f})",
                extra_meta={"entropy": entropy, "horizon_count": horizon_count,
                            "dominant_horizon": get("dominant_horizon", "")},
            )
        if expected < 0 and ctx.available_base > 0:
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=magnitude,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=direction_prob,
                reason=f"consensus {expected:.2%} across {horizon_count} horizons (entropy {entropy:.2f})",
                extra_meta={"entropy": entropy, "horizon_count": horizon_count,
                            "dominant_horizon": get("dominant_horizon", "")},
            )
        return None
