"""A discovered rule must earn its way to money like anything else.

These guard the boundary between "the search found a pattern" and "real
capital is committed to it". Every test here is about a way that boundary
could be crossed without evidence.
"""

from __future__ import annotations

import json

import pytest

from trading.strategies import discovered as mod
from trading.strategies.discovered import (DiscoveredRuleStrategy,
                                           MIN_HOLDOUT_T, _condition_holds,
                                           build_discovered_strategies)


class _Ctx:
    """Minimal StrategyContext stand-in."""

    def __init__(self, **kw):
        self.chain = kw.get("chain", "base")
        self.last_price = kw.get("last_price", 1.0)
        self.last_volume = kw.get("last_volume", 100.0)
        self.fee_rate = kw.get("fee_rate", 0.0065)
        self.available_quote = kw.get("available_quote", 10.0)
        self.available_base = kw.get("available_base", 0.0)
        self.risk_budget = kw.get("risk_budget", 1.0)
        self.live_trading = kw.get("live_trading", False)
        self.direction_prob = 0.5
        self.confidence = 0.5
        self.net_margin = 0.0
        self.opportunity = None
        self.extras = {}


class TestConditions:
    def test_a_missing_feature_is_none_not_false(self):
        """Unmeasurable must not read as "condition met" OR as "refuted".

        Both wrong answers are dangerous in different directions: treating it
        as met trades on nothing, and treating it as refuted silently retires
        a rule on symbols where its feature is merely unavailable.
        """
        condition = {"feature": "hurst", "op": ">", "value": 0.5}
        assert _condition_holds(condition, {}) is None
        assert _condition_holds(condition, {"hurst": 0.9}) is True
        assert _condition_holds(condition, {"hurst": 0.1}) is False

    def test_an_unparseable_value_is_none(self):
        condition = {"feature": "hurst", "op": ">", "value": "not a number"}
        assert _condition_holds(condition, {"hurst": 0.9}) is None


class TestGating:
    def _rule(self, **kw):
        record = {
            "id": "test-rule",
            "rule": "hurst > 0.5",
            "conditions": [{"feature": "hurst", "op": ">", "value": 0.5}],
            "holdout_t": 2.5,
            "holdout_mean_excess": 0.03,
        }
        record.update(kw)
        return DiscoveredRuleStrategy(record)

    def test_a_rule_below_the_bar_cannot_run(self):
        """There must be no path to money without a number behind it."""
        assert not self._rule(holdout_t=1.0).enabled()
        assert self._rule(holdout_t=2.5).enabled()

    def test_a_rule_with_no_statistic_is_refused(self):
        strategy = self._rule(holdout_t=None)
        assert strategy.holdout_t == 0.0
        assert not strategy.enabled()

    def test_an_unmeasurable_feature_does_not_fire(self, monkeypatch):
        """The discipline: no measurement, no trade."""
        monkeypatch.setattr(mod, "_live_features", lambda state, ctx: {})
        assert self._rule().evaluate(object(), _Ctx()) is None

    def test_a_rule_whose_edge_cannot_clear_cost_does_not_fire(self, monkeypatch):
        monkeypatch.setattr(mod, "_live_features",
                            lambda state, ctx: {"hurst": 0.9})
        strategy = self._rule(holdout_mean_excess=0.001)
        assert strategy.evaluate(object(), _Ctx(fee_rate=0.0065)) is None

    def test_a_failed_condition_does_not_fire(self, monkeypatch):
        monkeypatch.setattr(mod, "_live_features",
                            lambda state, ctx: {"hurst": 0.1})
        assert self._rule().evaluate(object(), _Ctx()) is None


class TestCandidateProduction:
    def test_a_satisfied_rule_produces_an_entry(self, monkeypatch):
        """The end of the chain: search -> publish -> registry -> directive.

        The state must carry base_token and quote_token. make_candidate
        refuses any pair not quoted in a stable, and a fake state missing
        those fields returns None for a reason that has nothing to do with
        the rule -- which is how an earlier version of this test passed while
        proving nothing.
        """
        record = {
            "id": "hurst-rule", "rule": "hurst > 0.5",
            "conditions": [{"feature": "hurst", "op": ">", "value": 0.5}],
            "holdout_t": 2.5, "holdout_mean_excess": 0.05,
        }
        monkeypatch.setattr(mod, "_live_features",
                            lambda state, ctx: {"hurst": 0.72})

        class _State:
            symbol = "BSTONK-USDC"
            base_token = "BSTONK"
            quote_token = "USDC"
            samples = [(float(i), 1.0 + i * 0.001, 1.0) for i in range(120)]

        candidate = DiscoveredRuleStrategy(record).evaluate(_State(), _Ctx())
        assert candidate is not None, "a satisfied rule produced no candidate"
        assert candidate["directive"].action == "enter"
        assert candidate["directive"].size > 0

    def test_a_non_stable_pair_is_refused(self, monkeypatch):
        """Base/base pairs carry a token ratio, not a USD price."""
        record = {
            "id": "hurst-rule", "rule": "hurst > 0.5",
            "conditions": [{"feature": "hurst", "op": ">", "value": 0.5}],
            "holdout_t": 2.5, "holdout_mean_excess": 0.05,
        }
        monkeypatch.setattr(mod, "_live_features",
                            lambda state, ctx: {"hurst": 0.72})

        class _State:
            symbol = "CBETH-CBBTC"
            base_token = "CBETH"
            quote_token = "CBBTC"
            samples = [(float(i), 1.0 + i * 0.001, 1.0) for i in range(120)]

        assert DiscoveredRuleStrategy(record).evaluate(_State(), _Ctx()) is None


class TestLoading:
    def test_rules_below_the_bar_are_never_loaded(self, tmp_path, monkeypatch):
        path = tmp_path / "discovered_rules.json"
        path.write_text(json.dumps([
            {"id": "good", "rule": "a", "holdout_t": MIN_HOLDOUT_T + 0.5,
             "conditions": [{"feature": "hurst", "op": ">", "value": 0.5}]},
            {"id": "weak", "rule": "b", "holdout_t": 0.4,
             "conditions": [{"feature": "hurst", "op": ">", "value": 0.5}]},
            {"id": "no-stat", "rule": "c",
             "conditions": [{"feature": "hurst", "op": ">", "value": 0.5}]},
            {"id": "no-conditions", "rule": "d", "holdout_t": 9.0},
        ]), encoding="utf-8")
        monkeypatch.setattr(mod, "DISCOVERED_PATH", path)

        strategies = build_discovered_strategies()
        assert len(strategies) == 1, [s.rule_id for s in strategies]
        assert strategies[0].rule_id == "good"

    def test_a_missing_file_is_no_rules_not_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "DISCOVERED_PATH", tmp_path / "absent.json")
        assert build_discovered_strategies() == []

    def test_a_corrupt_file_is_no_rules_not_a_crash(self, tmp_path, monkeypatch):
        path = tmp_path / "discovered_rules.json"
        path.write_text("{ not json", encoding="utf-8")
        monkeypatch.setattr(mod, "DISCOVERED_PATH", path)
        assert build_discovered_strategies() == []

    def test_similar_rules_get_distinct_ids(self, tmp_path, monkeypatch):
        """The registry is keyed by strategy_id.

        Two rules sharing a long prefix collided when the id was a truncation,
        and the second silently replaced the first -- one discovered rule
        simply vanished from a 74-strategy registry.
        """
        shared = "notional_usd < 1.99994 AND hurst > 0.521123"
        path = tmp_path / "discovered_rules.json"
        path.write_text(json.dumps([
            {"id": shared, "rule": shared, "holdout_t": 2.5,
             "conditions": [{"feature": "hurst", "op": ">", "value": 0.52}]},
            {"id": shared + " AND return_autocorrelation < 0.1",
             "rule": shared + " AND return_autocorrelation < 0.1",
             "holdout_t": 2.4,
             "conditions": [{"feature": "hurst", "op": ">", "value": 0.52}]},
        ]), encoding="utf-8")
        monkeypatch.setattr(mod, "DISCOVERED_PATH", path)

        ids = {s.strategy_id for s in build_discovered_strategies()}
        assert len(ids) == 2, ids
