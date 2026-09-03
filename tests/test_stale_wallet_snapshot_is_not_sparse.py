"""An old wallet READING is not an empty wallet.

``TrainingPipeline._wallet_state`` scored the age of its measurement as a
wallet fault: ``sparse`` included ``not balance_fresh``. That single term is
load-bearing far past its name -- ``sparse`` raises a ``freeze_live`` bus
action, and it feeds ``halt_live``, which trading/bot.py maps to
``risk_budget = 0.0``, halting the whole scheduler including ghost evaluation.

Measured 2026-09-03 06:12 against the running production process
(``feedback_events``, trailing 90 minutes):

  * 50 x ``bus_scheduler:bus_actions_pending`` carrying
    ``{"action": "freeze_live", "reason": "sparse_wallet",
       "min_capital_usd": 3.0, "reasons": ["wallet_snapshot_stale"]}``
  * 6 x ``scheduler:halted -> {"reason": "wallet_sparse"}``

while ``reconciled_wallet_snapshot('guardian')`` reported, on base,
7.0138 USD of ETH and 6.9773 USD of USDC -- $13.99 against a $3.00 minimum,
4.7x clear, the whole time. ``wallet_snapshot_stale`` was the ONLY reason in
the list. Nothing refreshes the snapshot on a timer, and the refresh was
requested from this function only once the reading had ALREADY expired, so
every cycle sawtoothed through a window with trading frozen against funds
that never moved.

The rule these tests pin:

  fresh                      -> not sparse, no reason
  stale (past max_age)       -> not sparse on its own, reported, refreshed
  unknown (past hard bound)  -> sparse; we cannot stand behind the number
  no timestamp at all        -> unknown by the same rule

Capital, native-gas and empty-focus faults are untouched: they are properties
of the wallet, not of the reading.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

from trading.pipeline import TrainingPipeline


BASE_BALANCES = [
    {
        "wallet": "0x291c854811e92906a658Fb94Aa511bF919f968ad",
        "chain": "base",
        "token": "0x0000000000000000000000000000000000000000",
        "symbol": "ETH",
        "quantity": "0.002796601250781071",
        "usd_amount": 7.0138,
    },
    {
        "wallet": "0x291c854811e92906a658Fb94Aa511bF919f968ad",
        "chain": "base",
        "token": "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
        "symbol": "USDC",
        "quantity": "6.977334",
        "usd_amount": 6.977258,
    },
]


def _snapshot(*, age_seconds, max_age_seconds: float = 1800.0) -> Dict[str, Any]:
    """The real shape of ``reconciled_wallet_snapshot`` output.

    Measured against the live wallet on 2026-09-03: ``fresh`` is a bool,
    ``age_seconds`` a float or None, ``max_age_seconds`` a float, and
    ``usd_amount`` a float in human USD (not raw base units).
    """
    fresh = age_seconds is not None and float(age_seconds) <= max_age_seconds
    return {
        "wallet_alias": "guardian",
        "wallet": "0x291c854811e92906a658Fb94Aa511bF919f968ad",
        "updated_at": "2026-09-03T06:07:09Z",
        "age_seconds": age_seconds,
        "max_age_seconds": max_age_seconds,
        "fresh": fresh,
        "status": "current" if fresh else "stale_refresh_required",
        "cached_total_usd": 13.991058,
        "balances": list(BASE_BALANCES),
    }


@pytest.fixture(autouse=True)
def _wallet_env(monkeypatch: pytest.MonkeyPatch):
    """The production configuration this bug was measured under."""
    monkeypatch.setenv("TRADING_WALLET", "guardian")
    monkeypatch.setenv("LIVE_FOCUS_CHAIN", "base")
    monkeypatch.setenv("LIVE_FOCUS_CHAINS", "base")
    monkeypatch.setenv("LIVE_MIN_CAPITAL_USD", "3.00")
    monkeypatch.setenv("LIVE_MICRO_MIN_CAPITAL_USD", "2.00")
    monkeypatch.setenv("LIVE_NATIVE_BUFFER_USD", "5")
    monkeypatch.setenv("WALLET_DUST_USD", "0.50")
    monkeypatch.delenv("WALLET_SNAPSHOT_HARD_MAX_AGE_SEC", raising=False)
    monkeypatch.delenv("WALLET_SNAPSHOT_REFRESH_FRACTION", raising=False)


def _wallet_state(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: Dict[str, Any],
    refreshes: List[bool],
) -> Dict[str, Any]:
    import services.wallet_reconciliation as recon

    monkeypatch.setattr(recon, "reconciled_wallet_snapshot", lambda alias="guardian": snapshot)
    monkeypatch.setattr(recon, "request_wallet_refresh", lambda **kw: refreshes.append(True))
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    return TrainingPipeline._wallet_state(pipeline)


def test_a_fresh_snapshot_is_not_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    refreshes: List[bool] = []
    state = _wallet_state(monkeypatch, _snapshot(age_seconds=43.2), refreshes)

    assert state["sparse"] is False
    assert state["sparse_reasons"] == []
    assert state["balance_fresh"] is True
    assert state["balance_stale"] is False
    assert state["balance_unknown"] is False
    # $6.98 + $7.01 on base, well clear of the $3.00 floor.
    assert state["capital_total_usd"] == pytest.approx(13.991058, rel=1e-6)


def test_a_stale_snapshot_reports_but_does_not_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact production condition: 1900s old, $13.99 present, $3.00 needed."""
    refreshes: List[bool] = []
    state = _wallet_state(monkeypatch, _snapshot(age_seconds=1900.0), refreshes)

    assert state["balance_fresh"] is False
    assert state["balance_stale"] is True
    assert state["balance_unknown"] is False
    # Visible on the plan and the dashboard...
    assert "wallet_snapshot_stale" in state["sparse_reasons"]
    # ...but NOT a fault. This is the term that raised freeze_live and
    # zeroed the risk budget for the whole scheduler.
    assert state["sparse"] is False
    assert state["capital_total_usd"] == pytest.approx(13.991058, rel=1e-6)


def test_a_snapshot_past_the_hard_bound_is_unknown_and_does_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old enough that the number is not evidence of anything any more."""
    refreshes: List[bool] = []
    # default hard bound = max(4 x 1800, 3600) = 7200s
    state = _wallet_state(monkeypatch, _snapshot(age_seconds=7201.0), refreshes)

    assert state["balance_unknown"] is True
    assert state["balance_stale"] is False
    assert state["sparse"] is True
    assert "wallet_snapshot_unknown" in state["sparse_reasons"]
    assert "wallet_snapshot_stale" not in state["sparse_reasons"]


def test_a_snapshot_that_was_never_timestamped_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refreshes: List[bool] = []
    state = _wallet_state(monkeypatch, _snapshot(age_seconds=None), refreshes)

    assert state["balance_unknown"] is True
    assert state["sparse"] is True
    assert "wallet_snapshot_unknown" in state["sparse_reasons"]


def test_the_hard_bound_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WALLET_SNAPSHOT_HARD_MAX_AGE_SEC", "2400")
    refreshes: List[bool] = []

    assert _wallet_state(monkeypatch, _snapshot(age_seconds=2399.0), refreshes)["sparse"] is False
    assert _wallet_state(monkeypatch, _snapshot(age_seconds=2401.0), refreshes)["sparse"] is True


def test_refresh_is_requested_before_the_snapshot_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waiting for expiry to ask for a refresh guarantees a stale window."""
    refreshes: List[bool] = []
    # Half of the 1800s permitted age: still fresh, already renewing.
    state = _wallet_state(monkeypatch, _snapshot(age_seconds=950.0), refreshes)

    assert state["balance_fresh"] is True
    assert state["sparse"] is False
    assert refreshes, "a snapshot past half its permitted age must be renewed"


def test_a_young_snapshot_is_not_refreshed(monkeypatch: pytest.MonkeyPatch) -> None:
    refreshes: List[bool] = []
    _wallet_state(monkeypatch, _snapshot(age_seconds=43.2), refreshes)

    assert refreshes == []


def test_real_wallet_faults_still_set_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capital and gas are properties of the wallet, not of the reading."""
    refreshes: List[bool] = []
    snapshot = _snapshot(age_seconds=10.0)
    snapshot["balances"] = [
        {**BASE_BALANCES[0], "quantity": "0.0000004", "usd_amount": 0.001},
        {**BASE_BALANCES[1], "quantity": "0.1", "usd_amount": 0.1},
    ]

    state = _wallet_state(monkeypatch, snapshot, refreshes)

    assert state["balance_fresh"] is True
    assert state["sparse"] is True
    assert "stable_below_min" in state["sparse_reasons"]


def test_an_empty_focus_chain_still_sets_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    refreshes: List[bool] = []
    snapshot = _snapshot(age_seconds=10.0)
    snapshot["balances"] = []

    state = _wallet_state(monkeypatch, snapshot, refreshes)

    assert state["sparse"] is True
    assert "focus_empty" in state["sparse_reasons"]


def test_an_unreadable_snapshot_is_unknown_not_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exception path must name itself the same way the age bounds do."""
    import services.wallet_reconciliation as recon

    def boom(alias: str = "guardian"):
        raise RuntimeError("wallet state unreadable")

    monkeypatch.setattr(recon, "reconciled_wallet_snapshot", boom)
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    state = TrainingPipeline._wallet_state(pipeline)

    assert state["sparse"] is True
    assert state["balance_unknown"] is True
    assert state["sparse_reasons"] == ["wallet_snapshot_unreadable"]


def test_the_contract_the_plan_consumes_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every consumer reads `sparse` as a bool and `sparse_reasons` as a list
    of str -- trading/pipeline.py `_build_transition_plan` and
    `live_readiness_report`, trading/swap_validator.py `plan_transition`, and
    web/frontend/src/views/PipelineView.vue. Both types survive the change."""
    refreshes: List[bool] = []
    for age in (43.2, 1900.0, 7201.0, None):
        state = _wallet_state(monkeypatch, _snapshot(age_seconds=age), refreshes)
        assert isinstance(state["sparse"], bool)
        assert isinstance(state["sparse_reasons"], list)
        assert all(isinstance(reason, str) for reason in state["sparse_reasons"])
        assert isinstance(state["balance_stale"], bool)
        assert isinstance(state["balance_unknown"], bool)
        assert isinstance(state["capital_total_usd"], float)
