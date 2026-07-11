"""Tests for the per-strategy ghost→live graduation ledger."""
from __future__ import annotations

import pytest

from trading.strategies.ledger import StrategyLedger


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_TRADES", "5")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_WINRATE", "0.6")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_PROFIT", "0.0")
    monkeypatch.setenv("STRATEGY_DEMOTE_MAX_LIVE_LOSSES", "3")
    return StrategyLedger(path=tmp_path / "ledger.json")


def test_starts_unapproved(ledger):
    assert not ledger.is_live_approved("mean_reversion")
    assert not ledger.any_live_approved()


def test_graduates_on_profitable_ghost_record(ledger):
    for _ in range(4):
        ledger.record("mean_reversion", profit=1.0, mode="ghost", confidence=0.7)
    assert not ledger.is_live_approved("mean_reversion")  # 4 < 5 trades
    ledger.record("mean_reversion", profit=1.0, mode="ghost", confidence=0.7)
    assert ledger.is_live_approved("mean_reversion")
    assert ledger.any_live_approved()
    assert "mean_reversion" in ledger.approved_ids()


def test_low_winrate_never_graduates(ledger):
    # loss-first alternation keeps the running winrate at or below 0.5
    for i in range(8):
        ledger.record("ema_cross", profit=-1.0 if i % 2 == 0 else 1.0, mode="ghost")
    assert not ledger.is_live_approved("ema_cross")


def test_unprofitable_never_graduates(ledger):
    # an early deep loss keeps cumulative profit negative despite later wins
    ledger.record("volume_spike", profit=-10.0, mode="ghost")
    for _ in range(6):
        ledger.record("volume_spike", profit=0.01, mode="ghost")
    assert not ledger.is_live_approved("volume_spike")


def test_live_loss_streak_demotes_and_resets_ghost(ledger):
    for _ in range(5):
        ledger.record("rsi_reversal", profit=1.0, mode="ghost")
    assert ledger.is_live_approved("rsi_reversal")
    for _ in range(3):
        ledger.record("rsi_reversal", profit=-0.5, mode="live")
    assert not ledger.is_live_approved("rsi_reversal")
    stats = ledger.stats("rsi_reversal")
    assert stats["demotions"] == 1
    assert stats["ghost"]["trades"] == 0  # must re-prove from scratch


def test_manual_demote(ledger):
    for _ in range(5):
        ledger.record("vwap_reversion", profit=1.0, mode="ghost")
    ledger.demote("vwap_reversion", "circuit breaker")
    assert not ledger.is_live_approved("vwap_reversion")
    assert ledger.stats("vwap_reversion")["demote_reason"] == "circuit breaker"


def test_persistence_across_instances(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_TRADES", "2")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_WINRATE", "0.5")
    path = tmp_path / "ledger.json"
    first = StrategyLedger(path=path)
    first.record("momentum_breakout", profit=1.0, mode="ghost")
    first.record("momentum_breakout", profit=1.0, mode="ghost")
    assert first.is_live_approved("momentum_breakout")

    second = StrategyLedger(path=path)
    assert second.is_live_approved("momentum_breakout")
    assert second.stats("momentum_breakout")["ghost"]["trades"] == 2


def test_blank_strategy_id_maps_to_unclassified(ledger):
    ledger.record("", profit=1.0, mode="ghost")
    assert ledger.stats("unclassified")["ghost"]["trades"] == 1
