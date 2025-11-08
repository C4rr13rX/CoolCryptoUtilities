from __future__ import annotations

import json

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
