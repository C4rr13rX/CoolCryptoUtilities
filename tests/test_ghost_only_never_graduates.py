"""A ghost-only executor must never hold live approval.

``atf_static_scout`` is services/atf_static_strategy.py's ghost quote scout. It
hardcodes ``wallet="ghost"`` and has no live branch, so its ghost record cannot
be spent. It was deliberately demoted for that reason on 2026-09-02 and
re-graduated 83 minutes later, because graduation re-ran on every recorded
outcome and only ever asked whether the strategy had traded well -- never
whether it could trade at all.

That is not a cosmetic flag. ``approved_ids()`` selects whose ghost book the
live gate judges, so the sole approved strategy was one structurally incapable
of spending; it was judged on 9 paired trades against a 25-trade minimum and
the entire live path reported ``ghost_validation_block``.
"""
from __future__ import annotations

import json

import pytest

from trading.strategies.ledger import StrategyLedger


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_TRADES", "5")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_WINRATE", "0.6")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_PROFIT", "0.0")
    monkeypatch.setenv("GHOST_ONLY_STRATEGY_IDS", "atf_static_scout")
    return StrategyLedger(path=tmp_path / "ledger.json")


def test_ghost_only_executor_never_graduates(ledger):
    """A flawless ghost record still buys no live approval."""
    for _ in range(30):
        ledger.record("atf_static_scout", profit=0.05, mode="ghost", confidence=0.9)

    stats = ledger.stats("atf_static_scout")
    assert stats["ghost"]["trades"] == 30       # the record is kept...
    assert stats["ghost"]["wins"] == 30
    assert not ledger.is_live_approved("atf_static_scout")   # ...but not spendable
    assert "atf_static_scout" not in ledger.approved_ids()
    assert not ledger.any_live_approved()
    assert stats["graduation_blocked"] is True


def test_spendable_strategy_still_graduates_normally(ledger):
    """The bar is on one executor, not on graduation itself."""
    for _ in range(6):
        ledger.record("atf_static", profit=0.05, mode="ghost", confidence=0.9)
    assert ledger.is_live_approved("atf_static")
    assert ledger.approved_ids() == ["atf_static"]


def test_existing_approval_is_revoked_on_load(tmp_path, monkeypatch):
    """The flag already sitting in the file is stripped when it is read.

    The live ledger held ``live_approved: true`` for the scout when this was
    written. Revoking only at demotion time would have left it there, because
    nothing was going to demote it again.
    """
    monkeypatch.setenv("GHOST_ONLY_STRATEGY_IDS", "atf_static_scout")
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps(
            {
                "atf_static_scout": {
                    "ghost": {"trades": 137, "wins": 96, "losses": 41,
                              "total_profit": 4.55},
                    "live": {"trades": 0, "wins": 0, "losses": 0,
                             "total_profit": 0.0},
                    "live_approved": True,
                    "graduated_ts": 1788366844.0,
                }
            }
        )
    )

    ledger = StrategyLedger(path=path)
    assert not ledger.is_live_approved("atf_static_scout")
    assert ledger.approved_ids() == []


def test_demotion_reason_survives_a_fresh_ghost_book(ledger):
    """Re-earning the evidence does not re-open a structural bar.

    A performance demotion is meant to be recoverable. This one is not: the
    reason it was demoted never stops being true.
    """
    ledger.block_graduation("signal_only_publisher", "no live branch")
    for _ in range(40):
        ledger.record("signal_only_publisher", profit=0.1, mode="ghost")
    assert not ledger.is_live_approved("signal_only_publisher")
    assert ledger.stats("signal_only_publisher")["graduation_blocked"] is True


def test_performance_demotion_remains_recoverable(ledger):
    """Guards against over-correcting: ordinary demotion must still be undoable."""
    for _ in range(6):
        ledger.record("mean_reversion", profit=0.05, mode="ghost")
    assert ledger.is_live_approved("mean_reversion")

    ledger.demote("mean_reversion", "3 consecutive live losses")
    assert not ledger.is_live_approved("mean_reversion")

    # Demotion blanks the ghost book; earning it again re-approves.
    for _ in range(6):
        ledger.record("mean_reversion", profit=0.05, mode="ghost")
    assert ledger.is_live_approved("mean_reversion")
