"""The genome strategy must refuse to trade anything it cannot justify.

Every refusal here maps to a way the system previously produced a confident
number from a model nobody validated: a partial feature vector, a genome
below the objective, or a profit factor measured on one fold.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from trading.strategies.genome_champion import GenomeChampionStrategy


def context(**overrides):
    base = dict(
        chain="base", last_price=100.0, last_volume=1000.0, fee_rate=0.002,
        available_quote=1000.0, available_base=10.0, risk_budget=1.0,
        live_trading=False, direction_prob=0.5, confidence=0.5,
        net_margin=0.0, opportunity=None, extras={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def state(symbol="AAVE-USDC", base_token="AAVE", quote_token="USDC"):
    return SimpleNamespace(symbol=symbol, base_token=base_token,
                           quote_token=quote_token, samples=[])


def signal(**overrides):
    base = {
        "genome_id": "701a3dfda8afcf6e", "profit_factor": 1.1965,
        "coverage": 0.754, "evaluated_folds": 3, "meets_objective": True,
        "scorable": True, "direction": 1, "confidence": 0.75,
        "direction_prob": 0.7, "expected_return": 0.02,
    }
    base.update(overrides)
    return base


def test_abstains_without_signals():
    assert GenomeChampionStrategy().evaluate(state(), context()) is None


def test_abstains_when_feature_vector_was_incomplete():
    """A partial vector must never be scored against defaulted zeros."""
    ctx = context(extras={"genome_signals": {"AAVE": signal(scorable=False)}})
    assert GenomeChampionStrategy().evaluate(state(), ctx) is None


def test_abstains_when_genome_is_below_the_objective():
    """Backtest profit under the bar does not earn ghost trades."""
    ctx = context(extras={"genome_signals": {
        "AAVE": signal(meets_objective=False, profit_factor=1.0567)}})
    assert GenomeChampionStrategy().evaluate(state(), ctx) is None


def test_abstains_on_a_flat_direction():
    ctx = context(extras={"genome_signals": {"AAVE": signal(direction=0)}})
    assert GenomeChampionStrategy().evaluate(state(), ctx) is None


def test_abstains_below_its_own_confidence_band():
    """The genome abstains by design; forcing coverage is what lost money."""
    ctx = context(extras={"genome_signals": {"AAVE": signal(confidence=0.10)}})
    assert GenomeChampionStrategy().evaluate(state(), ctx) is None


def test_abstains_when_the_edge_cannot_cover_fees():
    ctx = context(fee_rate=0.05,
                  extras={"genome_signals": {"AAVE": signal(expected_return=0.001)}})
    assert GenomeChampionStrategy().evaluate(state(), ctx) is None


def test_emits_a_candidate_for_a_qualifying_genome():
    ctx = context(extras={"genome_signals": {"AAVE": signal()}})
    candidate = GenomeChampionStrategy().evaluate(state(), ctx)
    assert candidate is not None
    assert candidate["directive"].action == "enter"
    assert candidate["meta"]["genome_id"] == "701a3dfda8afcf6e"
    assert candidate["meta"]["source"] == "market_evolution"


def test_ledger_id_is_per_genome():
    """A new champion must earn its own record, never inherit graduation."""
    strategy = GenomeChampionStrategy()
    ctx_a = context(extras={"genome_signals": {"AAVE": signal()}})
    first = strategy.evaluate(state(), ctx_a)
    ctx_b = context(extras={"genome_signals": {
        "AAVE": signal(genome_id="ffffffffffff9999")}})
    second = strategy.evaluate(state(), ctx_b)
    assert first is not None and second is not None
    assert first["directive"].strategy_id != second["directive"].strategy_id
    assert first["directive"].strategy_id.startswith("genome_")


def test_can_be_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("STRATEGY_GENOME_CHAMPION_ENABLED", "0")
    ctx = context(extras={"genome_signals": {"AAVE": signal()}})
    assert GenomeChampionStrategy().evaluate(state(), ctx) is None


def test_registered_in_the_default_strategy_set():
    import inspect

    import trading.strategies as strategies

    source = inspect.getsource(strategies)
    assert "GenomeChampionStrategy()" in source
