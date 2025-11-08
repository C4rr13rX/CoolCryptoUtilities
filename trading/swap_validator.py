from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from db import TradingDatabase, get_db
from trading.metrics import FeedbackSeverity, MetricStage, MetricsCollector


class SwapValidator:
    """
    Lightweight guard that scores a proposed swap against recent liquidity,
    execution quality, and volatility before allowing it to proceed.
    """

    def __init__(
        self,
        *,
        db: Optional[TradingDatabase] = None,
        lookback_sec: Optional[int] = None,
        max_liquidity_ratio: Optional[float] = None,
        min_execution_ratio: Optional[float] = None,
        max_slippage: Optional[float] = None,
        max_volatility: Optional[float] = None,
    ) -> None:
        self.db = db or get_db()
        self.metrics = MetricsCollector(self.db)
        self.lookback_sec = lookback_sec or int(os.getenv("SWAP_GUARD_LOOKBACK_SEC", "7200"))
        self.max_liquidity_ratio = max_liquidity_ratio or float(os.getenv("SWAP_GUARD_LIQUIDITY_RATIO", "0.35"))
        self.min_execution_ratio = min_execution_ratio or float(os.getenv("SWAP_GUARD_MIN_EXEC_RATIO", "0.82"))
        self.max_slippage = max_slippage or float(os.getenv("SWAP_GUARD_MAX_SLIPPAGE", "0.045"))
        self.max_volatility = max_volatility or float(os.getenv("SWAP_GUARD_MAX_VOLATILITY", "0.18"))

    def validate(
        self,
        *,
        symbol: str,
        route: Sequence[str],
        trade_size: float,
        price: float,
        volume: float,
        prediction: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, Dict[str, float], List[str]]:
        symbol_u = symbol.upper()
        samples = self.db.fetch_market_samples_for(symbol_u, limit=360)
        trade_usd = abs(trade_size * price)
        avg_volume_usd = self._average_volume_usd(samples)
        liquidity_ratio = trade_usd / max(avg_volume_usd, 1e-6)

        fills = self.db.fetch_trade_fills(limit=100)
        exec_ratio, avg_slippage = self._execution_stats(fills)

        volatility = self._estimate_volatility(samples)

        metrics = {
            "trade_value_usd": trade_usd,
            "avg_volume_usd": avg_volume_usd,
            "liquidity_ratio": liquidity_ratio,
            "execution_ratio": exec_ratio,
            "avg_slippage": avg_slippage,
            "volatility": volatility,
        }
        if prediction:
            metrics.update(
                {
                    "pred_direction_prob": float(prediction.get("direction_prob", 0.0)),
                    "pred_margin": float(prediction.get("net_margin", 0.0)),
                }
            )

        allowed = True
        reasons: List[str] = []
        if liquidity_ratio > self.max_liquidity_ratio:
            allowed = False
            reasons.append("liquidity")
        if exec_ratio < self.min_execution_ratio:
            allowed = False
            reasons.append("execution")
        if avg_slippage > self.max_slippage:
            allowed = False
            reasons.append("slippage")
        if volatility > self.max_volatility:
            allowed = False
            reasons.append("volatility")

        metrics["allowed"] = 1.0 if allowed else 0.0
        self.metrics.record(
            MetricStage.LIVE_TRADING,
            metrics,
            category="swap_guard",
            meta={
                "symbol": symbol_u,
                "route": list(route),
                "reasons": reasons,
            },
        )
        if not allowed:
            self.metrics.feedback(
                "swap_guard",
                severity=FeedbackSeverity.WARNING,
                label="swap_blocked",
                details={
                    "symbol": symbol_u,
                    "route": list(route),
                    "reasons": reasons,
                    "metrics": metrics,
                },
            )
        return allowed, metrics, reasons

    def _average_volume_usd(self, samples: Sequence[Dict[str, float]]) -> float:
        now = time.time()
        window = []
        for sample in samples:
            ts = float(sample.get("ts") or 0.0)
            if now - ts > self.lookback_sec:
                continue
            price = float(sample.get("price") or 0.0)
            volume = float(sample.get("volume") or 0.0)
            if price > 0 and volume > 0:
                window.append(price * volume)
        if not window:
            return 0.0
        return float(np.mean(window))

    def _execution_stats(self, fills: Sequence[Dict[str, float]]) -> Tuple[float, float]:
        ratios: List[float] = []
        slippages: List[float] = []
        now = time.time()
        for fill in fills:
            ts = float(fill.get("ts") or 0.0)
            if now - ts > self.lookback_sec:
                continue
            expected_amount = float(fill.get("expected_amount") or 0.0)
            executed_amount = float(fill.get("executed_amount") or 0.0)
            expected_price = float(fill.get("expected_price") or 0.0)
            executed_price = float(fill.get("executed_price") or 0.0)
            if expected_amount > 0:
                ratios.append(executed_amount / expected_amount)
            if expected_price > 0 and executed_price > 0:
                slippages.append(abs(executed_price - expected_price) / expected_price)
        exec_ratio = float(np.mean(ratios)) if ratios else 1.0
        avg_slippage = float(np.mean(slippages)) if slippages else 0.0
        return exec_ratio, avg_slippage

    def _estimate_volatility(self, samples: Sequence[Dict[str, float]]) -> float:
        prices = [float(sample.get("price") or 0.0) for sample in samples]
        if len(prices) < 3:
            return 0.0
        returns = np.diff(prices) / np.array(prices[:-1], dtype=float)
        returns = returns[np.isfinite(returns)]
        if returns.size == 0:
            return 0.0
        return float(np.std(returns) * np.sqrt(min(len(returns), 60)))

    def plan_transition(
        self,
        *,
        positions: Dict[str, Dict[str, Any]],
        exposure: Dict[str, float],
        readiness: Dict[str, Any],
        risk_budget: float,
        pending_decision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        gross_exposure = float(sum(abs(val) for val in exposure.values()))
        max_pending = max(1, int(os.getenv("LIVE_MAX_PENDING_POSITIONS", "4")))
        precision = float(readiness.get("precision", 0.0))
        recall = float(readiness.get("recall", 0.0))
        confidence_floor = float(os.getenv("LIVE_CONFIDENCE_FLOOR", "0.65"))
        confidence_margin = min(precision, recall) - confidence_floor
        budget_scale = max(0.2, min(1.0, 0.6 + confidence_margin))
        adjusted_budget = max(0.05, risk_budget * budget_scale)
        allowed = gross_exposure <= adjusted_budget and len(positions) <= max_pending
        snapshot = {
            "allowed": allowed,
            "gross_exposure": gross_exposure,
            "risk_budget": risk_budget,
            "adjusted_risk_budget": adjusted_budget,
            "confidence_margin": confidence_margin,
            "pending_positions": len(positions),
            "readiness": readiness,
        }
        if pending_decision:
            snapshot["pending_decision"] = {
                "action": pending_decision.get("action"),
                "symbol": pending_decision.get("symbol"),
                "size": pending_decision.get("size"),
            }
        return snapshot
