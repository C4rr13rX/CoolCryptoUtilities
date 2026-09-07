"""A strategy that has never traded must still appear in the population.

The pipeline page and ``scripts/readiness_report.py`` both enumerated the
LEDGER, and the ledger only gets a row when a strategy records an outcome. So
the strategies with no outcomes -- the ones most worth looking at, because
something is stopping them producing any evidence at all -- were the ones
guaranteed to be invisible.

Measured 2026-09-07 on the real files: ``data/strategy_registry.json`` holds 41
strategies and ``data/strategy_ledger.json`` holds 37. The four that exist only
in the registry are ``mean_reversion``, ``momentum_breakout``, ``volume_spike``
and ``vwap_reversion``. All four were commissioned 10.9 days earlier and have
closed zero ghost round trips between them. Nothing in the dashboard could show
that, so "we have 37 strategies collecting evidence" read as a healthy pipeline
while four commissioned strategies sat dead and unwatched.

The rule pinned here: the population is registry UNION ledger, and a strategy
present in only one of them is reported with the stage that says so.
"""
from __future__ import annotations

import json

import pytest

from services import strategy_registry
from services.strategy_population import collect


def _registry(tmp_path, monkeypatch, rows):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"strategies": rows}), encoding="utf-8")
    monkeypatch.setattr(strategy_registry, "REGISTRY_PATH", path)
    return path


def _ledger(tmp_path, rows):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _bar(monkeypatch):
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_TRADES", "20")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_WINRATE", "0.55")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_PROFIT", "0.0")
    monkeypatch.setenv("GHOST_ONLY_STRATEGY_IDS", "")


def test_a_registry_only_strategy_appears_as_a_candidate(tmp_path, monkeypatch):
    """The four never-run strategies must be in the payload, staged honestly."""
    _registry(
        tmp_path,
        monkeypatch,
        {
            "traded_one": {
                "strategy_id": "traded_one",
                "name": "traded_one",
                "commissioned": True,
                "created_at": 1000.0,
            },
            "never_run": {
                "strategy_id": "never_run",
                "name": "never_run",
                "commissioned": True,
                "created_at": 1000.0,
            },
        },
    )
    led = _ledger(
        tmp_path,
        {"traded_one": {"ghost": {"trades": 3, "wins": 2, "total_profit": 0.1}}},
    )

    out = collect(now=2000.0, ledger_path=led)
    by_id = {r["id"]: r for r in out["strategies"]}

    # The whole point: the strategy with no ledger row is present.
    assert "never_run" in by_id, "a registry-only strategy vanished from the population"
    assert out["totals"]["strategies"] == 2

    row = by_id["never_run"]
    assert row["stage"] == "candidate"
    assert row["status"] == "never-run"
    assert row["in_registry"] is True
    assert row["in_ledger"] is False
    # Time at stage runs from when it was commissioned, which is the number
    # that makes "commissioned 10.9 days ago and never ran" visible.
    assert row["stage_age_sec"] == pytest.approx(1000.0)
    # An empty book has an UNKNOWN win rate, not a 0% one: rendering zero puts
    # an untested strategy below genuine losers in a sorted column.
    assert row["ghost"]["win_rate"] is None
    assert row["last_trade_age_sec"] is None

    assert by_id["traded_one"]["stage"] == "ghost"


def test_the_candidate_stage_is_reported_even_when_empty(tmp_path, monkeypatch):
    """Every stage gets a row, so an empty stage reads as a finding.

    ``backtest`` is empty on the real data -- zero registry rows carry metrics
    or experiments -- and an omitted stage would render as "no such stage"
    rather than "nothing is being backtested".
    """
    _registry(
        tmp_path,
        monkeypatch,
        {"only": {"strategy_id": "only", "name": "only", "created_at": 1.0}},
    )
    led = _ledger(tmp_path, {"only": {"ghost": {"trades": 1, "wins": 1}}})

    out = collect(now=2000.0, ledger_path=led)
    stages = {s["stage"]: s["count"] for s in out["stages"]}

    assert set(stages) == {"candidate", "backtest", "ghost", "live", "rejected"}
    assert stages["ghost"] == 1
    assert stages["backtest"] == 0
    assert stages["live"] == 0


def test_a_ledger_only_strategy_is_not_dropped(tmp_path, monkeypatch):
    """The union runs both ways: a ledger row with no registry entry counts."""
    _registry(tmp_path, monkeypatch, {})
    led = _ledger(tmp_path, {"orphan": {"ghost": {"trades": 4, "wins": 1}}})

    out = collect(now=2000.0, ledger_path=led)
    by_id = {r["id"]: r for r in out["strategies"]}

    assert "orphan" in by_id
    assert by_id["orphan"]["in_registry"] is False
    assert by_id["orphan"]["in_ledger"] is True
    assert by_id["orphan"]["kind"] == "ledger-only"
