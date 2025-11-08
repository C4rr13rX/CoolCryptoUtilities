from __future__ import annotations

import time

import numpy as np

from trading.brain.arb_cell import VolatilityArbCell
from trading.brain.event_engine import EventEngine, ReflexRule
from trading.brain.memory import PatternMemory
from trading.brain.neuro_graph import NeuroGraph
from trading.brain.scenario import ScenarioReactor
from trading.brain.swarm import MultiResolutionSwarm


def test_neuro_graph_reinforcement_and_decay() -> None:
    graph = NeuroGraph(decay=0.9, learning_rate=0.5)
    now = time.time()
    graph.upsert_node("ETH-USDC", "asset", 1800.0, now)
    graph.upsert_node("ETH-USDC:momentum", "volatility", 0.0, now)

    graph.reinforce("ETH-USDC", "ETH-USDC:momentum", now, 0.8)
    assert graph.strength("ETH-USDC", "ETH-USDC:momentum") > 0.0

    initial_weight = graph.strength("ETH-USDC", "ETH-USDC:momentum")
    graph.decay_all()
    decayed_weight = graph.strength("ETH-USDC", "ETH-USDC:momentum")
    assert decayed_weight < initial_weight
    assert decayed_weight > 0.0


def test_multi_resolution_swarm_vote_and_learn() -> None:
    horizons = [("fast", 5), ("slow", 10)]
    swarm = MultiResolutionSwarm(horizons)

    prices = np.linspace(100.0, 102.0, 10, dtype=np.float64)
    sentiments = np.linspace(-0.2, 0.4, 10, dtype=np.float64)

    price_windows = {
        "fast": prices[-5:],
        "slow": prices,
    }
    sentiment_windows = {
        "fast": sentiments[-5:],
        "slow": sentiments,
    }

    votes = swarm.vote(price_windows, sentiment_windows)
    assert len(votes) == len(horizons)
    assert all(np.isfinite(v.expected_return) for v in votes)

    realized = {"fast": 0.01, "slow": 0.02}
    pre_total = swarm.stats["fast"]["total"]
    swarm.learn(price_windows, sentiment_windows, realized)
    assert swarm.stats["fast"]["total"] == pre_total + 1.0


def test_multi_resolution_swarm_weights_normalised() -> None:
    swarm = MultiResolutionSwarm([("fast", 5), ("medium", 8)])
    weights = swarm.weights()
    assert weights
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9


def test_pattern_memory_match() -> None:
    memory = PatternMemory(dim=3, max_entries=8)
    vec_a = np.array([1.0, 0.5, 0.25], dtype=np.float32)
    vec_b = np.array([-0.5, 0.2, 0.1], dtype=np.float32)
    memory.add(vec_a, profit=5.0, duration=120.0)
    memory.add(vec_b, profit=-2.0, duration=45.0)

    query = np.array([0.9, 0.48, 0.26], dtype=np.float32)
    result = memory.match(query)
    assert result is not None
    score, meta = result
    assert score > 0.0
    assert "duration" in meta


def test_scenario_reactor_divergence() -> None:
    reactor = ScenarioReactor(tolerance=0.01)
    scenarios = reactor.analyse(base_expected=0.001, confidence=0.7, volatility=0.0002)
    spread = reactor.divergence(scenarios)
    assert spread > 0.0
    # small volatility keeps divergence below tolerance
    assert reactor.should_defer(scenarios) is False

    wide_scenarios = reactor.analyse(base_expected=0.0, confidence=0.5, volatility=0.5)
    assert reactor.should_defer(wide_scenarios) is True


def test_volatility_arb_cell_observe() -> None:
    cell = VolatilityArbCell()
    # prime filters with neutral observations
    for _ in range(20):
        cell.observe(eth_price=1800.0, usdc_price=1.0)

    signal = cell.observe(eth_price=1850.0, usdc_price=1.0)
    assert signal.action in {"buy_eth", "sell_eth", "hold"}
    assert np.isfinite(signal.spread)
    assert np.isfinite(signal.implied_edge)


def test_event_engine_process_with_cooldown() -> None:
    engine = EventEngine()
    triggered: list[str] = []

    def condition(ctx: dict[str, float]) -> bool:
        return ctx.get("drawdown", 0.0) < -0.1

    def action(ctx: dict[str, float]) -> None:
        triggered.append(ctx.get("reflex_rule", "rule"))

    engine.register(ReflexRule(name="drawdown_guard", condition=condition, action=action, cooldown=1.0))
    context = {"drawdown": -0.2, "volatility": 0.5}
    now = time.time()

    first = engine.process(context.copy(), now)
    assert first == ["drawdown_guard"]
    assert triggered == ["drawdown_guard"]

    # Cooldown prevents immediate retrigger.
    second = engine.process(context.copy(), now + 0.1)
    assert second == []

    # After cooldown, rule can trigger again.
    third = engine.process(context.copy(), now + 2.0)
    assert third == ["drawdown_guard"]
