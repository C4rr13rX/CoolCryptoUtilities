"""Unit tests for the independent CPU strategy plugins (trading/strategies/).

Each strategy gets a synthetic pattern that must emit a candidate, and flat
data must emit nothing (no false positives on dead markets).
"""
from __future__ import annotations

import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from trading.strategies import (
    BollingerSqueezeStrategy,
    EmaCrossStrategy,
    DustMicroSwingStrategy,
    MeanReversionStrategy,
    MomentumBreakoutStrategy,
    RsiReversalStrategy,
    StrategyContext,
    SwarmConsensusStrategy,
    VolumeSpikeStrategy,
    VwapReversionStrategy,
    build_default_registry,
)

FEE = 0.005


def make_state(prices, volumes=None, t0=1_700_000_000.0, step=60.0):
    volumes = volumes if volumes is not None else [10.0] * len(prices)
    samples = deque(
        (t0 + i * step, float(p), float(v)) for i, (p, v) in enumerate(zip(prices, volumes))
    )
    return SimpleNamespace(
        symbol="TEST-USDC",
        base_token="TEST",
        quote_token="USDC",
        samples=samples,
    )


def make_ctx(state, *, quote=100.0, base=0.0, extras=None):
    ts, price, vol = state.samples[-1]
    return StrategyContext(
        chain="base",
        last_price=price,
        last_volume=vol,
        fee_rate=FEE,
        available_quote=quote,
        available_base=base,
        extras=extras or {},
    )


def flat_state(n=80):
    return make_state([100.0] * n)


ALL_STRATEGIES = [
    MeanReversionStrategy(),
    MomentumBreakoutStrategy(),
    EmaCrossStrategy(),
    RsiReversalStrategy(),
    BollingerSqueezeStrategy(),
    VolumeSpikeStrategy(),
    VwapReversionStrategy(),
    SwarmConsensusStrategy(),
]


class TestNoFalsePositivesOnFlatData:
    @pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.strategy_id)
    def test_flat_market_yields_nothing(self, strategy):
        state = flat_state()
        assert strategy.evaluate(state, make_ctx(state, quote=100.0, base=10.0)) is None


class TestMeanReversion:
    def test_dip_below_mean_enters(self, monkeypatch):
        monkeypatch.setenv("MEAN_REVERSION_Z_ENTRY", "1.5")
        # The dip descent itself would trip the knife guard (tested separately
        # below); relax it here to exercise the z-score/reversion math.
        monkeypatch.setenv("MEAN_REVERSION_SLOPE_FLOOR", "-0.005")
        # long base at 100, controlled dip to ~96.5 that then stabilises
        prices = [100.0] * 60 + list(np.linspace(100, 96.5, 10)) + [96.5, 96.52, 96.55, 96.53]
        state = make_state(prices)
        cand = MeanReversionStrategy().evaluate(state, make_ctx(state))
        assert cand is not None
        d = cand["directive"]
        assert d.action == "enter"
        assert d.target_price > d.target_price * 0 and d.expected_return > FEE
        assert cand["meta"]["strategy"] == "mean_reversion"

    def test_stretch_above_mean_exits_when_holding(self, monkeypatch):
        monkeypatch.setenv("MEAN_REVERSION_Z_ENTRY", "1.5")
        prices = [100.0] * 60 + list(np.linspace(100, 103.5, 10)) + [103.5, 103.52]
        state = make_state(prices)
        cand = MeanReversionStrategy().evaluate(state, make_ctx(state, base=5.0))
        assert cand is not None and cand["directive"].action == "exit"

    def test_freefall_is_rejected_by_knife_guard(self, monkeypatch):
        monkeypatch.setenv("MEAN_REVERSION_Z_ENTRY", "1.0")
        # relentless steep decline — no stabilisation
        prices = list(np.linspace(110, 90, 60))
        state = make_state(prices)
        assert MeanReversionStrategy().evaluate(state, make_ctx(state)) is None


class TestMomentumBreakout:
    def test_breakout_with_volume_enters(self):
        rng = np.random.default_rng(7)
        base = list(97.0 + rng.uniform(0, 3.0, size=32))  # consolidation 97-100
        prices = base + [100.1, 100.3, 100.6]
        volumes = [10.0] * 32 + [12.0, 15.0, 30.0]
        state = make_state(prices, volumes)
        cand = MomentumBreakoutStrategy().evaluate(state, make_ctx(state))
        assert cand is not None
        assert cand["directive"].action == "enter"
        assert cand["directive"].expected_return > FEE

    def test_breakdown_exits_when_holding(self):
        rng = np.random.default_rng(7)
        base = list(97.0 + rng.uniform(0, 3.0, size=32))
        prices = base + [96.9, 96.7, 96.4]
        volumes = [10.0] * 35
        state = make_state(prices, volumes)
        cand = MomentumBreakoutStrategy().evaluate(state, make_ctx(state, base=5.0))
        assert cand is not None and cand["directive"].action == "exit"


class TestEmaCross:
    def test_golden_cross_enters(self):
        prices = [100.0] * 50 + [99.0] * 5 + [103.0]
        state = make_state(prices)
        cand = EmaCrossStrategy().evaluate(state, make_ctx(state))
        assert cand is not None
        assert cand["directive"].action == "enter"


class TestRsiReversal:
    def test_oversold_enters(self):
        prices = [105.0] * 40 + list(np.linspace(105, 97.5, 16))
        state = make_state(prices)
        cand = RsiReversalStrategy().evaluate(state, make_ctx(state))
        assert cand is not None
        assert cand["directive"].action == "enter"
        assert cand["meta"]["rsi"] <= 30.0

    def test_overbought_exits_when_holding(self):
        prices = [95.0] * 40 + list(np.linspace(95, 102.5, 16))
        state = make_state(prices)
        cand = RsiReversalStrategy().evaluate(state, make_ctx(state, base=5.0))
        assert cand is not None and cand["directive"].action == "exit"


class TestBollingerSqueeze:
    def test_break_up_after_squeeze_enters(self):
        strat = BollingerSqueezeStrategy()
        prices = [100.0] * 75
        # accrue 30+ bandwidth-history entries over repeated flat evaluations
        for i in range(42, 76):
            state = make_state(prices[:i])
            strat.evaluate(state, make_ctx(state))
        state = make_state(prices + [110.0])
        cand = strat.evaluate(state, make_ctx(state))
        assert cand is not None
        assert cand["directive"].action == "enter"


class TestVolumeSpike:
    def test_ignition_up_enters(self):
        prices = [100.0] * 30 + [104.0]
        volumes = [10.0] * 30 + [45.0]
        state = make_state(prices, volumes)
        cand = VolumeSpikeStrategy().evaluate(state, make_ctx(state))
        assert cand is not None
        assert cand["directive"].action == "enter"

    def test_ignition_down_exits_when_holding(self):
        prices = [100.0] * 30 + [96.0]
        volumes = [10.0] * 30 + [45.0]
        state = make_state(prices, volumes)
        cand = VolumeSpikeStrategy().evaluate(state, make_ctx(state, base=5.0))
        assert cand is not None and cand["directive"].action == "exit"


class TestVwapReversion:
    def test_below_vwap_enters(self):
        prices = [100.0] * 40 + [97.0]
        state = make_state(prices)
        cand = VwapReversionStrategy().evaluate(state, make_ctx(state))
        assert cand is not None
        assert cand["directive"].action == "enter"
        assert math.isclose(cand["directive"].target_price, cand["meta"]["vwap"])


class TestSwarmConsensus:
    def _extras(self, expected):
        return {
            "swarm_consensus": {
                "expected_return": expected,
                "confidence": 0.7,
                "direction_prob": 0.8,
                "entropy": 0.3,
                "horizon_count": 3,
                "dominant_horizon": "medium",
            }
        }

    def test_bullish_consensus_enters(self):
        state = flat_state(20)
        cand = SwarmConsensusStrategy().evaluate(
            state, make_ctx(state, extras=self._extras(0.03))
        )
        assert cand is not None
        assert cand["directive"].action == "enter"

    def test_bearish_consensus_exits_when_holding(self):
        state = flat_state(20)
        cand = SwarmConsensusStrategy().evaluate(
            state, make_ctx(state, base=5.0, extras=self._extras(-0.03))
        )
        assert cand is not None and cand["directive"].action == "exit"

    def test_high_entropy_disagreement_blocks(self):
        state = flat_state(20)
        extras = self._extras(0.03)
        extras["swarm_consensus"]["entropy"] = 0.95
        assert SwarmConsensusStrategy().evaluate(state, make_ctx(state, extras=extras)) is None


class TestDustMicroSwing:
    def _pattern(self):
        # Thirty-minute oscillation, sharp dip, then confirmed five-minute bounce.
        return [100.0] * 24 + [98.5, 96.0, 95.5, 95.8, 96.2, 97.0, 97.5]

    def test_cost_covered_dust_targets_usdc_pair_in_ghost(self, monkeypatch):
        monkeypatch.setenv("DUST_CONVERSION_COST_RATIO", "0.002")
        state = make_state(self._pattern())
        extras = {"dust_micro": {
            "enabled": True,
            "budget_usdc": 1.0,
            "source_tokens": ["OLD", "TINY"],
        }}
        cand = DustMicroSwingStrategy().evaluate(state, make_ctx(state, quote=0.0, extras=extras))
        assert cand is not None
        directive = cand["directive"]
        assert directive.strategy_id == "dust_micro_swing"
        assert directive.horizon in {"5m", "15m", "30m"}
        assert math.isclose(directive.size * state.samples[-1][1], 1.0, rel_tol=1e-6)
        assert cand["meta"]["expected_net_usdc"] >= 0.01
        assert cand["meta"]["target_token"] == "TEST"
        assert cand["meta"]["ghost_only"] is True

    def test_never_emits_in_live_mode(self):
        state = make_state(self._pattern())
        ctx = make_ctx(state, quote=100.0, extras={"dust_micro": {
            "enabled": True, "budget_usdc": 1.0, "source_tokens": ["OLD"],
        }})
        ctx.live_trading = True
        assert DustMicroSwingStrategy().evaluate(state, ctx) is None

    def test_rejects_flat_or_uncosted_cent_profit(self, monkeypatch):
        extras = {"dust_micro": {"enabled": True, "budget_usdc": 1.0, "source_tokens": ["OLD"]}}
        flat = flat_state(31)
        assert DustMicroSwingStrategy().evaluate(flat, make_ctx(flat, extras=extras)) is None
        monkeypatch.setenv("DUST_CONVERSION_COST_RATIO", "0.20")
        state = make_state(self._pattern())
        assert DustMicroSwingStrategy().evaluate(state, make_ctx(state, extras=extras)) is None


class TestRegistry:
    def test_default_registry_contains_core_and_multihorizon_strategies(self):
        reg = build_default_registry()
        ids = set(reg.ids())
        core = {
            "mean_reversion",
            "momentum_breakout",
            "ema_cross",
            "rsi_reversal",
            "bollinger_squeeze",
            "volume_spike",
            "vwap_reversion",
            "swarm_consensus",
        }
        assert core <= ids
        assert len(ids) > len(core)

    def test_evaluate_all_tags_strategy_meta(self, monkeypatch):
        monkeypatch.setenv("MEAN_REVERSION_Z_ENTRY", "1.5")
        prices = [100.0] * 60 + list(np.linspace(100, 96.5, 10)) + [96.5, 96.52, 96.55, 96.53]
        state = make_state(prices)
        cands = build_default_registry().evaluate_all(state, make_ctx(state))
        assert cands, "at least one strategy should fire on the dip pattern"
        for cand in cands:
            assert cand["meta"].get("strategy") in build_default_registry().ids()

    def test_env_disable_gates_strategy(self, monkeypatch):
        monkeypatch.setenv("STRATEGY_VOLUME_SPIKE_ENABLED", "0")
        prices = [100.0] * 30 + [104.0]
        volumes = [10.0] * 30 + [45.0]
        state = make_state(prices, volumes)
        cands = build_default_registry().evaluate_all(state, make_ctx(state))
        assert all(c["meta"]["strategy"] != "volume_spike" for c in cands)

    def test_candidates_pass_cdcl_clauses(self, monkeypatch):
        """Emitted candidates must survive the solver's clause DB."""
        from trading.cdcl_solver import CDCLTradingSolver
        monkeypatch.setenv("MEAN_REVERSION_Z_ENTRY", "1.5")
        prices = [100.0] * 60 + list(np.linspace(100, 96.5, 10)) + [96.5, 96.52, 96.55, 96.53]
        state = make_state(prices)
        cands = build_default_registry().evaluate_all(state, make_ctx(state))
        assert cands
        solver = CDCLTradingSolver()
        context = {
            "native_balance": 0.05,
            "min_native": 0.01,
            "fee_rate": FEE,
            "risk_budget": 1.0,
            "trade_history": deque(),
        }
        chosen = solver.select(cands, context)
        assert chosen is not None, f"solver rejected all: {solver.last_unsat}"
