from __future__ import annotations

import json

import pytest

import trading.bot as bot_module
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
    bot.gas_force_refill = False  # type: ignore[attr-defined]

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


@pytest.mark.parametrize(
    ("confidence", "floor", "accepted"),
    [(0.29, 0.30, False), (0.30, 0.30, True), (0.49, None, False), (0.50, None, True)],
)
def test_brain_entry_supports_separate_explore_and_live_floors(
    monkeypatch, confidence: float, floor: float | None, accepted: bool
) -> None:
    class Bridge:
        def query_confidence(self, features: str):
            return "UP", confidence

    bot = TradingBot.__new__(TradingBot)
    bot._brain_conf_ema = 0.0
    monkeypatch.setattr(bot_module, "_brain_bridge", lambda: Bridge())
    monkeypatch.setattr(bot_module, "_brain_features_text", lambda **kwargs: "features")
    monkeypatch.setenv("BRAIN_CONFIDENCE_FLOOR", "0.5")
    decision = {}

    result = bot._brain_record_entry(
        decision,
        side="buy",
        symbol="ETH-USDC",
        chain_name="base",
        price=100.0,
        min_confidence=floor,
    )

    assert (result > 0.0) is accepted
    assert decision["brain"]["bridge_confidence_rejected"] is (not accepted)


def test_brain_graduation_requires_positive_net_profit(monkeypatch) -> None:
    class Metrics:
        def feedback(self, *args, **kwargs):
            raise AssertionError("loss-making ghost strategy must not graduate")

    bot = TradingBot.__new__(TradingBot)
    bot.live_trading_enabled = False
    bot.auto_promote_live = True
    bot.total_trades = 30
    bot.wins = 25
    bot.total_profit = -0.01
    bot._brain_conf_ema = 0.9
    bot.required_live_trades = 50
    bot.required_live_win_rate = 0.7
    bot.required_live_profit = 0.0
    bot.metrics = Metrics()
    monkeypatch.setenv("BRAIN_GRADUATION_MIN_TRADES", "20")
    monkeypatch.setenv("BRAIN_GRADUATION_MIN_WINRATE", "0.55")
    monkeypatch.setenv("BRAIN_GRADUATION_MIN_CONF_EMA", "0.20")

    bot._maybe_promote_to_live()

    assert bot.live_trading_enabled is False
