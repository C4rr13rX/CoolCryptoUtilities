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

    The liquidity clause compares the trade against observed per-sample volume.
    That comparison is only meaningful when the feed reports volume at all.
    Every tick this stack records carries ``volume=0`` -- the DexScreener REST
    consensus path publishes price only -- so dividing by the observed mean
    turned "we never measured volume" into "this pair has no liquidity", the
    most extreme violation the clause can express. A $0.35 trade on AERO
    (1.26M pooled, 386k/h traded, in USD) scored a liquidity ratio of 348,862
    against a limit of 0.35 and was refused, as was every other live entry on
    every pair, forever. Absent volume is now carried as *unmeasured* and
    answered from an independent per-symbol reading (DexScreener pair volume,
    via the ATF signal feed or a discovery swap probe); when even that is
    missing the ratio stays undefined and an absolute notional cap stands in
    for it, because a number you never measured cannot be a violation.
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
        # Ceiling on a single trade when no volume basis exists at all. Small
        # enough that no realistically tradeable pair could be moved by it, so
        # the guard still refuses to size blind into an unknown book.
        self.unknown_liquidity_max_usd = float(
            os.getenv("SWAP_GUARD_UNKNOWN_LIQUIDITY_MAX_USD", "25")
        )

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

        observed_volume_usd = self._average_volume_usd(samples)
        volume_basis = "observed"
        avg_volume_usd = observed_volume_usd
        if avg_volume_usd is None:
            interval = self._sample_interval_sec(samples)
            avg_volume_usd = self._reference_volume_usd(symbol_u, interval)
            volume_basis = "reference" if avg_volume_usd is not None else "unmeasured"
        liquidity_ratio: Optional[float] = None
        if avg_volume_usd is not None and avg_volume_usd > 0:
            liquidity_ratio = trade_usd / avg_volume_usd

        fills = self.db.fetch_trade_fills(limit=100)
        exec_ratio, avg_slippage = self._execution_stats(fills)

        volatility = self._estimate_volatility(samples)

        metrics = {
            "trade_value_usd": trade_usd,
            "avg_volume_usd": float(avg_volume_usd) if avg_volume_usd is not None else -1.0,
            "liquidity_ratio": float(liquidity_ratio) if liquidity_ratio is not None else -1.0,
            "liquidity_measured": 1.0 if liquidity_ratio is not None else 0.0,
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
        if liquidity_ratio is not None:
            if liquidity_ratio > self.max_liquidity_ratio:
                allowed = False
                reasons.append("liquidity")
        elif trade_usd > self.unknown_liquidity_max_usd:
            # No volume basis anywhere: the ratio is undefined, so fall back to
            # the only bound that survives without a measurement -- an absolute
            # notional the book cannot plausibly notice.
            allowed = False
            reasons.append("liquidity_unmeasured")
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
                "volume_basis": volume_basis,
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
                    "volume_basis": volume_basis,
                    "reasons": reasons,
                    "metrics": metrics,
                },
            )
        return allowed, metrics, reasons

    def _average_volume_usd(self, samples: Sequence[Dict[str, float]]) -> Optional[float]:
        """Mean per-sample USD volume, or None when the feed reported none.

        None means *unmeasured*, not zero. Returning 0.0 here is what made the
        liquidity ratio unbounded on a feed that never carries volume.
        """
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
            return None
        return float(np.mean(window))

    def _sample_interval_sec(self, samples: Sequence[Dict[str, float]]) -> float:
        """Median spacing between samples, so an hourly volume can be scaled
        onto the same per-sample basis the ratio is defined against."""
        stamps = sorted(
            float(s.get("ts") or 0.0)
            for s in samples
            if float(s.get("ts") or 0.0) > 0
        )
        gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
        if not gaps:
            return 300.0
        return float(min(3600.0, max(30.0, float(np.median(gaps)))))

    def _reference_volume_usd(self, symbol: str, interval_sec: float) -> Optional[float]:
        """Per-sample USD volume from an independent per-symbol reading.

        The tick feed publishes price only, but DexScreener pair volume reaches
        this stack two other ways: the ATF signal rows (``volume_h1``) and the
        discovery swap probes (``volume_24h_usd``). Either is scaled down to the
        sample cadence so it means the same thing as the observed mean.
        """
        scale = max(30.0, float(interval_sec)) / 3600.0
        try:
            from services.atf_static_strategy import latest_signals

            best: Optional[float] = None
            for row in latest_signals(float(os.getenv("SWAP_GUARD_REFERENCE_MAX_AGE_SEC", "3600"))):
                if not isinstance(row, dict):
                    continue
                if str(row.get("symbol") or "").upper() != symbol:
                    continue
                hourly = float(row.get("volume_h1") or 0.0)
                if hourly > 0 and (best is None or hourly < best):
                    best = hourly
            if best is not None:
                return best * scale
        except Exception:
            pass
        try:
            hourly = self.db.reference_hourly_volume_usd(symbol)
        except Exception:
            hourly = None
        if hourly and hourly > 0:
            return float(hourly) * scale
        return None

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
        wallet_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        readiness = readiness or {}
        wallet_state = wallet_state or readiness.get("wallet_state") or {}
        ghost_meta = readiness.get("ghost_validation") or {}
        ghost_ready = bool(
            readiness.get(
                "ghost_ready",
                ghost_meta.get("ready", True) if isinstance(readiness, dict) else True,
            )
        )
        ghost_reason = str(readiness.get("ghost_reason") or ghost_meta.get("reason") or "")
        tail_risk = float(ghost_meta.get("tail_risk", readiness.get("ghost_tail_risk", 0.0)))
        tail_guard = float(
            ghost_meta.get(
                "tail_guardrail",
                ghost_meta.get("tail_guard", float(os.getenv("GHOST_TAIL_GUARDRAIL", os.getenv("GHOST_TAIL_GUARD", "0.0")))),
            )
        )
        ghost_samples = int(readiness.get("ghost_samples", ghost_meta.get("samples", 0)))
        ghost_min_trades = int(
            ghost_meta.get(
                "min_trades",
                int(os.getenv("MIN_GHOST_TRADES_FOR_PROMOTION", os.getenv("MIN_GHOST_TRADES_OVERRIDE", "0"))),
            )
        )
        if ghost_samples < ghost_min_trades and ghost_min_trades > 0:
            ghost_ready = False
            ghost_reason = ghost_reason or "ghost_sample_gap"
        tail_limit_hit = tail_guard > 0 and tail_risk > tail_guard
        if tail_limit_hit:
            ghost_ready = False
            ghost_reason = ghost_reason or "tail_risk"
        gross_exposure = float(sum(abs(val) for val in exposure.values()))
        max_pending = max(1, int(os.getenv("LIVE_MAX_PENDING_POSITIONS", "4")))
        precision = float(readiness.get("precision", 0.0))
        recall = float(readiness.get("recall", 0.0))
        confidence_floor = float(os.getenv("LIVE_CONFIDENCE_FLOOR", "0.65"))
        confidence_margin = min(precision, recall) - confidence_floor
        budget_scale = max(0.2, min(1.0, 0.6 + confidence_margin))
        tail_headroom = 1.0
        if tail_guard > 0:
            tail_headroom = max(0.25, min(1.0, (tail_guard - tail_risk) / max(tail_guard, 1e-9)))
        adjusted_budget = max(0.05, risk_budget * budget_scale * tail_headroom)
        capital_deficit = max(
            0.0,
            float(wallet_state.get("min_capital_usd", 0.0)) - float(wallet_state.get("stable_usd", 0.0)),
        )
        sparse_wallet = bool(wallet_state.get("sparse"))
        fragmented_wallet = bool(wallet_state.get("fragmented"))
        fragment_ratio = float(wallet_state.get("fragment_ratio", 0.0))
        native_starved = bool(wallet_state.get("native_starved", False))
        native_gap = float(wallet_state.get("native_buffer_gap_usd", 0.0))
        reasons: List[str] = []
        if not ghost_ready:
            reasons.append("ghost_not_ready")
        if ghost_samples < ghost_min_trades and ghost_min_trades > 0:
            reasons.append("ghost_sample_gap")
        if tail_limit_hit and "tail_risk" not in reasons:
            reasons.append("tail_risk")
        if sparse_wallet:
            reasons.append("sparse_wallet")
        if fragmented_wallet:
            reasons.append("fragmented_wallet")
        if capital_deficit > 0:
            reasons.append("capital_deficit")
        if native_starved:
            reasons.append("native_starved")
        if gross_exposure > adjusted_budget:
            reasons.append("exposure_limit")
        if len(positions) > max_pending:
            reasons.append("pending_limit")
        if not ghost_ready:
            adjusted_budget = 0.0
        allowed = (
            ghost_ready
            and not sparse_wallet
            and not fragmented_wallet
            and not native_starved
            and capital_deficit <= 0
            and gross_exposure <= adjusted_budget
            and len(positions) <= max_pending
        )
        snapshot = {
            "allowed": allowed,
            "gross_exposure": gross_exposure,
            "risk_budget": risk_budget,
            "adjusted_risk_budget": adjusted_budget,
            "confidence_margin": confidence_margin,
            "pending_positions": len(positions),
            "readiness": readiness,
            "wallet_state": {
                "sparse": sparse_wallet,
                "capital_deficit": capital_deficit,
                "stable_usd": float(wallet_state.get("stable_usd", 0.0)),
                "fragmented": fragmented_wallet,
                "fragment_ratio": fragment_ratio,
                "native_starved": native_starved,
                "native_buffer_gap_usd": native_gap,
            },
            "block_reasons": reasons,
        }
        bus_swap_plan = None
        if sparse_wallet and capital_deficit > 0:
            bus_swap_plan = {
                "action": "swap_to_stable",
                "reduce_position": float(capital_deficit),
                "reason": "sparse_wallet",
            }
        if native_starved and bus_swap_plan is None:
            target_usd = native_gap if native_gap > 0 else float(os.getenv("GAS_MIN_REFILL_USD", "5"))
            if float(wallet_state.get("stable_usd", 0.0)) > 0:
                bus_swap_plan = {
                    "action": "swap_stable_to_native",
                    "reason": "native_starved",
                    "target_usd": target_usd,
                }
            else:
                bus_swap_plan = {"action": "freeze_live", "reason": "native_starved", "target_usd": target_usd}
        if fragmented_wallet and bus_swap_plan is None:
            bus_swap_plan = {
                "action": "consolidate_fragments",
                "reason": "fragmented_wallet",
                "dust_tokens": list(wallet_state.get("dust_tokens", []))[:8],
            }
        if not ghost_ready and bus_swap_plan is None:
            bus_swap_plan = {"action": "freeze_live", "reason": ghost_reason or "ghost_not_ready"}
        if bus_swap_plan is None and tail_guard > 0 and tail_risk >= tail_guard * 0.9:
            bus_swap_plan = {
                "action": "freeze_live",
                "reason": "tail_risk_headroom" if tail_risk < tail_guard else "tail_risk",
                "tail_risk": tail_risk,
                "tail_guardrail": tail_guard,
            }
        if not allowed and exposure and bus_swap_plan is None:
            try:
                symbol, value = max(exposure.items(), key=lambda kv: abs(kv[1]))
                bus_swap_plan = {
                    "symbol": symbol,
                    "action": "rebalance_to_stable",
                    "reduce_position": float(abs(value) * 0.5),
                    "reason": "exposure_above_budget",
                }
            except Exception:
                bus_swap_plan = {"reason": "exposure_above_budget"}
        snapshot["bus_swap_plan"] = bus_swap_plan
        if pending_decision:
            snapshot["pending_decision"] = {
                "action": pending_decision.get("action"),
                "symbol": pending_decision.get("symbol"),
                "size": pending_decision.get("size"),
            }
        snapshot["risk_flags"] = {
            "ghost_ready": ghost_ready,
            "ghost_reason": ghost_reason,
            "capital_deficit": capital_deficit,
            "sparse_wallet": sparse_wallet,
            "ghost_samples": ghost_samples,
            "ghost_min_trades": ghost_min_trades,
            "tail_risk": tail_risk,
            "tail_guardrail": tail_guard,
            "fragmented_wallet": fragmented_wallet,
            "fragment_ratio": fragment_ratio,
            "native_starved": native_starved,
            "native_buffer_gap_usd": native_gap,
        }
        return snapshot
