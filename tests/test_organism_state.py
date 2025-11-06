from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List

import pytest

from services.organism_state import build_snapshot


class _FakeHolding:
    def __init__(self, quantity: float, usd: float) -> None:
        self.quantity = quantity
        self.usd = usd


class _FakePortfolio:
    def __init__(self) -> None:
        self.holdings: Dict[tuple[str, str], _FakeHolding] = {
            ("base", "USDC"): _FakeHolding(56.0, 56.0),
        }
        self.native_balances: Dict[str, float] = {"base": 0.42}

    def summary(self) -> Dict[str, float]:
        return {
            "wallet": "0x123",
            "stable_usd": 56.0,
            "native_eth": 0.42,
            "holdings": len(self.holdings),
            "last_refresh": time.time(),
        }


class _FakeScheduler:
    def snapshot(self) -> List[Dict[str, Any]]:
        return [
            {
                "symbol": "ETH-USDC",
                "price": 3200.0,
                "history_points": 32,
            }
        ]


class _FakeGraph:
    def __init__(self) -> None:
        self._ts = time.time()

    def nodes_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            "ETH-USDC": {
                "kind": "asset",
                "value": 1.1,
                "last_update": self._ts,
                "metadata": {"volatility": 0.03},
            },
            "ETH-USDC:momentum": {
                "kind": "volatility",
                "value": 0.2,
                "last_update": self._ts,
                "metadata": {},
            },
        }

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            "ETH-USDC": {"ETH-USDC:momentum": 0.18},
            "ETH-USDC:momentum": {},
        }


class _FakeBot:
    def __init__(self) -> None:
        self.live_trading_enabled = False
        self.ghost_session_id = 7
        self.positions = {
            "ETH-USDC": {
                "entry_price": 3150.0,
                "size": 1.25,
                "entry_ts": time.time() - 90,
                "target_price": 3300.0,
                "brain_snapshot": {"graph_confidence": 1.05},
            }
        }
        self.active_exposure = {"ETH-USDC": 3937.5}
        self.queue: List[Dict[str, Any]] = [{"action": "enter", "symbol": "ETH-USDC"}]
        self._pending_queue = deque([{"symbol": "ETH-USDC"}], maxlen=8)
        self.stable_bank = 120.0
        self.total_profit = 45.5
        self.realized_profit = 21.0
        self.total_trades = 88
        self.wins = 61
        self.portfolio = _FakePortfolio()
        self.scheduler = _FakeScheduler()
        self.graph = _FakeGraph()
        self._last_windows = {
            "ETH-USDC": {
                "prices": {"fast": [3100.0, 3120.0, 3150.0]},
                "sentiment": {"fast": [-0.1, 0.05, 0.12]},
            }
        }
        self._latency_window = deque([0.012, 0.017, 0.009], maxlen=12)

    def current_equity(self) -> float:
        return 256.75

    def latency_stats(self) -> Dict[str, float]:
        # optional helper; build_snapshot falls back to internal window
        return {}


def test_build_snapshot_compiles_organism_view() -> None:
    bot = _FakeBot()
    sample = {
        "symbol": "ETH-USDC",
        "chain": "base",
        "price": 3200.0,
        "volume": 180000.0,
        "ts": time.time(),
    }
    pred_summary = {
        "direction_prob": 0.72,
        "net_margin": 0.061,
        "exit_conf": 0.69,
    }
    brain_summary = {
        "graph_confidence": 1.08,
        "swarm_bias": 0.12,
        "memory_bias": 0.04,
        "scenario_spread": 0.03,
        "swarm_votes": [
            {"horizon": "5m", "expected": 0.018, "confidence": 0.66}
        ],
    }
    decision = {"action": "enter", "symbol": "ETH-USDC", "status": "ghost-entry"}
    discovery = {"status_counts": {"validated": 4}, "recent_events": []}

    snapshot = build_snapshot(
        bot=bot,
        sample=sample,
        pred_summary=pred_summary,
        brain_summary=brain_summary,
        directive=None,
        decision=decision,
        latency_s=0.015,
        latency_window=list(bot._latency_window),
        pending_depth=len(bot._pending_queue),
        discovery_snapshot=discovery,
        last_windows=bot._last_windows,
    )

    assert snapshot["mode"] == "ghost"
    assert snapshot["ghost_session"] == bot.ghost_session_id
    assert snapshot["exposure"]["ETH-USDC"] == pytest.approx(3937.5)
    assert snapshot["positions"]["ETH-USDC"]["entry_price"] == pytest.approx(3150.0)
    assert snapshot["queue_depth"] == len(bot.queue)
    assert snapshot["pending_samples"] == len(bot._pending_queue)
    assert snapshot["discovery"]["status_counts"]["validated"] == 4
    assert snapshot["totals"]["equity"] == pytest.approx(bot.current_equity())
    assert snapshot["latency_stats"]["count"] == 3
    graph_nodes = snapshot["organism_graph"]["nodes"]
    assert any(node["id"] == "brain" for node in graph_nodes)
    assert any(node["id"] == "asset:ETH-USDC" for node in graph_nodes)
