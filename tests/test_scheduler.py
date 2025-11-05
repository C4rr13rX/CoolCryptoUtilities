from __future__ import annotations

import math
import time
from typing import Dict

import numpy as np

from trading.scheduler import BusScheduler


class StubPortfolio:
    def __init__(self, quote_balance: float, base_balance: float, native_balance: float) -> None:
        self.quote_balance = quote_balance
        self.base_balance = base_balance
        self.native_balance = native_balance

    def get_quantity(self, symbol: str, chain: str = "base") -> float:
        if symbol.upper() == "USDC":
            return self.quote_balance
        if symbol.upper() == "ETH":
            return self.base_balance
        return 0.0

    def get_native_balance(self, chain: str = "base") -> float:
        return self.native_balance


def _pump_samples(scheduler: BusScheduler, symbol: str, start_price: float, steps: int = 20) -> None:
    ts = time.time() - steps * 60.0
    for i in range(steps):
        sample = {
            "symbol": symbol,
            "ts": ts + i * 60.0,
            "price": start_price * (1.0 + 0.002 * i),
            "volume": 1_000 + 10 * i,
            "chain": "base",
        }
        scheduler._update_state(sample)


def test_scheduler_produces_enter_directive(monkeypatch) -> None:
    scheduler = BusScheduler(horizons=[("5m", 300), ("30m", 1800)], min_profit=0.01)
    symbol = "ETH-USDC"
    _pump_samples(scheduler, symbol, start_price=100.0, steps=32)

    portfolio = StubPortfolio(quote_balance=1_000.0, base_balance=0.0, native_balance=1.0)
    pred_summary: Dict[str, float] = {
        "direction_prob": 0.75,
        "exit_conf": 0.72,
        "net_margin": 0.02,
    }

    directive = scheduler.evaluate(
        {
            "symbol": symbol,
            "ts": time.time(),
            "price": 107.0,
            "volume": 1_200.0,
            "chain": "base",
        },
        pred_summary,
        portfolio,
    )
    assert directive is not None
    assert directive.action == "enter"
    assert directive.base_token == "ETH"
    assert directive.quote_token == "USDC"
    assert directive.size > 0.0
    assert directive.expected_return > 0.0


def test_scheduler_exit_directive_when_expected_return_negative() -> None:
    scheduler = BusScheduler(horizons=[("5m", 300)], min_profit=0.01)
    symbol = "ETH-USDC"
    _pump_samples(scheduler, symbol, start_price=150.0, steps=40)

    # Tilt slope negative by manually adjusting history
    state = scheduler.routes[symbol]
    times = np.array([row[0] for row in state.samples])
    prices = np.array([row[1] for row in state.samples])
    prices -= np.linspace(0, 5.0, len(prices))
    state.samples = type(state.samples)(list(zip(times, prices, np.ones_like(prices))))

    portfolio = StubPortfolio(quote_balance=0.0, base_balance=10.0, native_balance=1.0)
    pred_summary: Dict[str, float] = {
        "direction_prob": 0.3,
        "exit_conf": 0.8,
        "net_margin": -0.05,
    }

    directive = scheduler.evaluate(
        {
            "symbol": symbol,
            "ts": time.time(),
            "price": prices[-1],
            "volume": 900.0,
            "chain": "base",
        },
        pred_summary,
        portfolio,
    )
    assert directive is not None
    assert directive.action == "exit"
    assert directive.size > 0.0
