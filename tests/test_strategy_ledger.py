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


def test_live_loss_streak_demotes_and_snapshots_ghost(ledger):
    for _ in range(5):
        ledger.record("rsi_reversal", profit=1.0, mode="ghost")
    assert ledger.is_live_approved("rsi_reversal")
    for _ in range(3):
        ledger.record("rsi_reversal", profit=-0.5, mode="live")
    assert not ledger.is_live_approved("rsi_reversal")
    stats = ledger.stats("rsi_reversal")
    assert stats["demotions"] == 1
    # The ghost book SURVIVES a demotion, and is snapshotted rather than wiped.
    #
    # This assertion used to read `ghost["trades"] == 0` -- "must re-prove from
    # scratch" -- and had been failing since _demote_locked stopped wiping the
    # record, because wiping it made demotion permanent within a session: the
    # strategy faced a 20-trade bar from zero and nothing could trade meanwhile.
    #
    # "Re-prove from scratch" is still enforced, but by measuring the ghost book
    # against the snapshot instead of by destroying it. Re-arming needs a full
    # graduation-grade book gathered AFTER `ghost_at_demotion`, so the evidence
    # has to be fresh without the history being thrown away.
    assert stats["ghost"]["trades"] == 5
    assert stats["ghost_at_demotion"]["trades"] == 5


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


# ---------------------------------------------------------------------------
# Loss accounting
#
# `losses` was never incremented and was not even a key in _blank_mode(), so
# the ledger reported zero losses for every strategy forever -- atf_static read
# 120 trades / 79 wins / 0 losses, and rsi_reversal@5h read 1 trade, 0 wins,
# 0 losses, -0.1370. Graduation was unaffected (it scores wins/trades), but a
# flawless record with no losses is this project's own signature for a
# FABRICATED one, so honest losers were indistinguishable from invented winners.
# ---------------------------------------------------------------------------


def test_losses_are_counted(ledger):
    ledger.record("s", profit=-0.5, mode="ghost")
    ledger.record("s", profit=-0.25, mode="ghost")
    ledger.record("s", profit=1.0, mode="ghost")
    ghost = ledger.stats("s")["ghost"]
    assert ghost["trades"] == 3
    assert ghost["wins"] == 1
    assert ghost["losses"] == 2


def test_wins_and_losses_account_for_every_trade(ledger):
    """The invariant that was silently false on disk: wins + losses == trades."""
    for i in range(10):
        ledger.record("s", profit=(0.02 if i % 3 == 0 else -0.01), mode="ghost")
    ghost = ledger.stats("s")["ghost"]
    assert ghost["wins"] + ghost["losses"] == ghost["trades"] == 10


def test_a_blank_mode_exposes_the_losses_field(ledger):
    """Readers used .get('losses', 0), so a missing key read as a clean record."""
    ledger.record("s", profit=0.01, mode="ghost")
    assert "losses" in ledger.stats("s")["ghost"]
    assert "losses" in ledger.stats("s")["live"]


def test_ledger_and_registry_agree_on_loss_count(ledger, tmp_path, monkeypatch):
    """The two files are each other's only independent check, so a flat
    outcome must not be a loss in one and not the other."""
    import services.strategy_registry as registry

    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "registry.json")
    profits = [0.05, -0.02, 0.0, -0.10, 0.03]
    for p in profits:
        ledger.record("agree", profit=p, mode="ghost")
        registry.record_outcome("agree", profit=p, mode="ghost", symbol="A-USDC")

    led = ledger.stats("agree")["ghost"]
    reg = registry.get_strategy("agree")["lifetime"]["ghost"]
    assert (led["trades"], led["wins"], led["losses"]) == (
        reg["trades"], reg["wins"], reg["losses"]
    )
