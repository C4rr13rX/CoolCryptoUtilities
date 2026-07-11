from __future__ import annotations

import os
from typing import Any, Dict, Optional

from trading.strategies.base import Strategy, StrategyContext, env_float


class ATFStaticStrategy(Strategy):
    """Consume C0D3R/ATF research signals as normal BusScheduler candidates."""

    strategy_id = "atf_static"
    default_horizon = "atf"
    min_samples = 4

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        enabled = os.getenv("ATF_STATIC_STRATEGY_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
        if not enabled:
            return None
        try:
            from services.atf_static_strategy import latest_signals
        except Exception:
            return None

        max_age = env_float("ATF_STATIC_SIGNAL_MAX_AGE_SEC", 1800.0, lo=30.0, hi=24 * 3600.0)
        symbol = str(getattr(state, "symbol", "") or "").upper()
        match = None
        for signal in latest_signals(max_age):
            if str(signal.get("symbol") or "").upper() == symbol:
                match = signal
                break
        if not match:
            return None

        quote_probe = match.get("quote_probe") if isinstance(match.get("quote_probe"), dict) else {}
        require_quote = os.getenv("ATF_STATIC_REQUIRE_QUOTE_OK", "0").strip().lower() in {"1", "true", "yes", "on"}
        if require_quote and not quote_probe.get("ok"):
            return None

        try:
            expected = float(match.get("expected_return") or 0.0)
            target_price = float(match.get("target_price") or 0.0)
            confidence = float(match.get("confidence") or 0.0)
        except Exception:
            return None
        min_edge = env_float("ATF_STATIC_MIN_EDGE", 0.006, lo=0.0, hi=0.25)
        cap_edge = env_float("ATF_STATIC_EDGE_CAP", 0.10, lo=0.005, hi=0.5)
        expected = max(0.0, min(cap_edge, expected))
        if expected - ctx.fee_rate < min_edge:
            return None
        if ctx.last_price > 0:
            target_price = max(target_price, ctx.last_price * (1.0 + expected))

        return self.make_candidate(
            state,
            ctx,
            action="enter",
            expected_return=expected,
            target_price=target_price,
            confidence=max(0.05, min(0.9, confidence)),
            direction_prob=max(0.51, min(0.9, 0.5 + expected * 2.0)),
            reason=f"ATF researched candidate score={match.get('score')} quote_ok={bool(quote_probe.get('ok'))}",
            horizon=self.default_horizon,
            extra_meta={
                "token_address": match.get("token_address"),
                "pair_address": match.get("pair_address"),
                "source": match.get("source"),
                "quote_probe_ok": bool(quote_probe.get("ok")),
                "liquidity_usd": match.get("liquidity_usd"),
                "volume_h1": match.get("volume_h1"),
                "url": match.get("url"),
            },
        )
