from __future__ import annotations

import json

import pytest

from trading.bot import TradingBot


def test_trading_bot_state_rehydration() -> None:
    bot = TradingBot.__new__(TradingBot)
    # wipe any attributes that would normally be initialised
    if hasattr(bot, "_pair_adjustments"):
        delattr(bot, "_pair_adjustments")
    TradingBot._ensure_runtime_state(bot)  # type: ignore[attr-defined]
    assert hasattr(bot, "_pair_adjustments")
    assert isinstance(bot._pair_adjustments, dict)  # type: ignore[attr-defined]
    assert hasattr(bot, "savings")


def test_trading_bot_timeline_append(tmp_path) -> None:
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)  # type: ignore[attr-defined]
    bot._timeline_path = tmp_path / "timeline.json"  # type: ignore[attr-defined]
    bot._append_organism_timeline({"index": 0}, limit=3)  # type: ignore[attr-defined]
    data = json.loads(bot._timeline_path.read_text())  # type: ignore[attr-defined]
    assert data["snapshots"][0]["index"] == 0
    for idx in range(5):
        bot._append_organism_timeline({"index": idx}, limit=3)  # type: ignore[attr-defined]
    data = json.loads(bot._timeline_path.read_text())  # type: ignore[attr-defined]
    assert len(data["snapshots"]) == 3
    assert data["snapshots"][-1]["index"] == 4


def test_base_allocation_respects_pair_multiplier() -> None:
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)  # type: ignore[attr-defined]

    class StubDB:
        def __init__(self, record):
            self.record = record

        def get_pair_adjustment(self, symbol):
            return dict(self.record)

    bot.db = StubDB({"allocation_multiplier": 2.0})  # type: ignore[attr-defined]
    bot.max_symbol_share = 0.1  # type: ignore[attr-defined]
    bot.primary_chain = "base"  # type: ignore[attr-defined]
    bot.live_trading_enabled = False  # type: ignore[attr-defined]
    bot.sim_quote_balances = {("base", "USDC"): 1000.0}  # type: ignore[attr-defined]
    bot.stable_bank = 0.0  # type: ignore[attr-defined]
    bot.active_exposure = {}  # type: ignore[attr-defined]
    bot._pair_adjustments = {}  # type: ignore[attr-defined]

    sample = {"symbol": "ETH-USD", "chain": "base"}
    allocation = bot._compute_base_allocation(sample)  # type: ignore[attr-defined]
    assert allocation["ETH-USD"] == pytest.approx(200.0)

    bot.active_exposure["ETH-USD"] = 150.0  # type: ignore[index]
    allocation = bot._compute_base_allocation(sample)  # type: ignore[attr-defined]
    assert allocation["ETH-USD"] == pytest.approx(50.0)


def test_plan_gas_replenishment_prefers_stable_swaps() -> None:
    bot = TradingBot.__new__(TradingBot)
    TradingBot._ensure_runtime_state(bot)  # type: ignore[attr-defined]

    class Holding:
        def __init__(self, symbol: str, token: str, quantity: float, usd: float) -> None:
            self.symbol = symbol
            self.token = token
            self.quantity = quantity
            self.usd = usd

    class Portfolio:
        def __init__(self) -> None:
            self.holdings = {("base", "USDC"): Holding("USDC", "0xusdc", 120.0, 120.0)}
            self.native_balances = {"base": 0.0, "ethereum": 0.25}

    bot.portfolio = Portfolio()  # type: ignore[attr-defined]
    bot.stable_tokens = {"USDC"}  # type: ignore[attr-defined]
    bot.gas_buffer_multiplier = 1.2  # type: ignore[attr-defined]
    bot.gas_roundtrip_fee_ratio = 0.0  # type: ignore[attr-defined]
    bot.gas_bridge_flat_fee = 0.0  # type: ignore[attr-defined]
    bot.gas_profit_guard = 1.0  # type: ignore[attr-defined]
    bot.gas_force_refill = True  # type: ignore[attr-defined]

    strategy = bot._plan_gas_replenishment(  # type: ignore[attr-defined]
        chain="base",
        route=["ETH", "USDC"],
        native_balance=0.0,
        gas_required=0.01,
        trade_size=1.0,
        price=2000.0,
        margin=0.02,
        pnl=0.5,
        available_quote=50.0,
        symbol="ETH-USDC",
    )

    assert strategy is not None
    assert strategy["stable_swap_plan"]
    assert strategy["force_rebalance"] is True
    assert strategy["remaining_native_gap"] >= 0
