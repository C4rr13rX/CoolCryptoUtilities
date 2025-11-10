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
