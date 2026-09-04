"""The agent must not be able to spend more than its tier allows.

Every rule here corresponds to a way this system has already lost money, so
each is pinned rather than trusted:

  * a ghost hypothesis must cost nothing -- experiments exist to be wrong
    cheaply, and one that can spend is not an experiment;
  * a ceiling must be enforced BEFORE the trade, because a limit audited
    afterwards is a report;
  * a tier must be earned by measured results, never set by configuration
    alone.
"""

from __future__ import annotations

from django.test import TestCase

from .engine import _clamp_trades, _record_decision, performance_snapshot
from .models import (AgentConfig, AgentRun, Constraint, Experiment,
                     LossRecovery, RiskTier)


def _greedy():
    """What an agent asks for when nothing stops it."""
    return [
        {"symbol": "A-USDC", "side": "enter", "size_usd": 500.0},
        {"symbol": "B-USDC", "side": "enter", "size_usd": 0.10},
        {"symbol": "C-USDC", "side": "enter", "size_usd": 50.0},
        {"symbol": "D-USDC", "side": "enter", "size_usd": 1.0},
        {"symbol": "E-USDC", "side": "enter", "size_usd": 1.0},
    ]


class SpendCeilingTest(TestCase):
    def setUp(self):
        self.config = AgentConfig.load()

    def test_ghost_spends_nothing_however_much_is_asked_for(self):
        self.config.tier = RiskTier.GHOST
        out = _clamp_trades(_greedy(), self.config)
        self.assertTrue(out, "ghost still records intent")
        self.assertEqual(sum(t["size_usd"] for t in out), 0.0,
                         "no real money may move in ghost")

    def test_each_tier_caps_the_clip(self):
        for tier, ceiling in (("micro", 0.25), ("small", 0.75), ("normal", 2.0)):
            self.config.tier = tier
            for trade in _clamp_trades(_greedy(), self.config):
                self.assertLessEqual(
                    trade["size_usd"], ceiling,
                    f"{tier}: {trade['symbol']} exceeded its ${ceiling} ceiling")

    def test_a_smaller_request_is_left_alone(self):
        """The ceiling is a cap, not a target."""
        self.config.tier = RiskTier.SMALL
        out = _clamp_trades([{"symbol": "B-USDC", "side": "enter",
                              "size_usd": 0.10}], self.config)
        self.assertAlmostEqual(out[0]["size_usd"], 0.10)

    def test_position_count_is_capped(self):
        self.config.tier = RiskTier.SMALL
        self.config.max_open_positions = 2
        self.assertEqual(len(_clamp_trades(_greedy(), self.config)), 2)

    def test_a_clamped_trade_records_what_was_asked(self):
        """So an agent repeatedly asking for too much is visible."""
        self.config.tier = RiskTier.MICRO
        out = _clamp_trades([{"symbol": "A-USDC", "side": "enter",
                              "size_usd": 500.0}], self.config)
        self.assertEqual(out[0]["clamped_from"], 500.0)
        self.assertEqual(out[0]["size_usd"], 0.25)

    def test_a_non_numeric_size_is_dropped_not_defaulted(self):
        """Defaulting a bad size to something tradeable is how money leaks."""
        self.config.tier = RiskTier.SMALL
        out = _clamp_trades([{"symbol": "X-USDC", "side": "enter",
                              "size_usd": "lots"}], self.config)
        self.assertEqual(out, [])


class GhostFirstTest(TestCase):
    def setUp(self):
        self.config = AgentConfig.load()
        self.run = AgentRun.objects.create(agent="test")

    def test_a_new_experiment_always_starts_in_ghost(self):
        """Even when the agent asks for live, and whatever the config says."""
        self.config.tier = RiskTier.NORMAL
        _record_decision(self.run, {"new_experiments": [
            {"hypothesis": "buying dips pays", "metric": "net_pl",
             "target": 0.5, "min_trades": 20, "status": "live",
             "tier": "normal"}]}, self.config)
        experiment = Experiment.objects.get()
        self.assertEqual(experiment.status, Experiment.Status.GHOST)
        self.assertEqual(experiment.tier, RiskTier.GHOST)

    def test_an_experiment_loss_cap_cannot_exceed_the_daily_limit(self):
        self.config.max_daily_loss_usd = 2.0
        _record_decision(self.run, {"new_experiments": [
            {"hypothesis": "swing for it", "max_loss_usd": 999.0}]}, self.config)
        self.assertLessEqual(Experiment.objects.get().max_loss_usd, 2.0)

    def test_a_new_constraint_is_proposed_not_active(self):
        """A rule earns its way in by being tested, like an experiment."""
        _record_decision(self.run, {"new_constraints": [
            {"kind": "entry", "rule": "only trade bulls", "rationale": "hunch"}]},
            self.config)
        self.assertEqual(Constraint.objects.get().status,
                         Constraint.Status.PROPOSED)

    def test_graduation_needs_a_sample_and_a_profit(self):
        experiment = Experiment.objects.create(
            hypothesis="h", min_trades=20, status=Experiment.Status.GHOST)

        experiment.ghost_trades, experiment.ghost_net_pl = 5, 1.0
        self.assertFalse(experiment.ready_to_graduate, "5 trades is not a sample")

        experiment.ghost_trades, experiment.ghost_net_pl = 25, -0.5
        self.assertFalse(experiment.ready_to_graduate, "a loser must not graduate")

        experiment.ghost_trades, experiment.ghost_net_pl = 25, 0.4
        self.assertTrue(experiment.ready_to_graduate)

    def test_the_ladder_never_skips_a_rung(self):
        self.assertEqual(RiskTier.next_tier("ghost"), "micro")
        self.assertEqual(RiskTier.next_tier("micro"), "small")
        self.assertEqual(RiskTier.next_tier("small"), "normal")
        self.assertEqual(RiskTier.next_tier("normal"), "normal")


class LossRecoveryTest(TestCase):
    def test_recovery_is_judged_on_what_it_saved(self):
        """A rule turning -8% into -3% works, though every trade under it lost."""
        rule = LossRecovery.objects.create(
            trigger="down 3% in 10 minutes", action="exit half",
            avg_loss_without=-0.08, avg_loss_with=-0.03, times_triggered=12)
        # POSITIVE means the loss got smaller. Losses are stored negative, so
        # a naive without-minus-with inverts the answer and reports a rule
        # that made things worse as though it helped.
        self.assertAlmostEqual(rule.saved_per_trigger, 0.05)

    def test_a_recovery_that_costs_money_shows_negative(self):
        rule = LossRecovery.objects.create(
            trigger="any dip", action="double down",
            avg_loss_without=-0.03, avg_loss_with=-0.09)
        self.assertLess(rule.saved_per_trigger, 0.0)


class DisabledAgentTest(TestCase):
    def test_a_disabled_agent_does_nothing(self):
        from .engine import run_once

        config = AgentConfig.load()
        config.enabled = False
        config.save()
        run = run_once(config)
        self.assertEqual(run.status, AgentRun.Status.COMPLETED)
        self.assertEqual(run.trades_opened, 0)
        self.assertIn("disabled", run.report)


class PerformanceSnapshotTest(TestCase):
    def test_it_counts_only_completed_runs(self):
        AgentRun.objects.create(agent="t", status=AgentRun.Status.COMPLETED,
                                net_pl=0.5, trades_opened=2, trades_closed=1)
        AgentRun.objects.create(agent="t", status=AgentRun.Status.FAILED,
                                net_pl=-99.0, trades_opened=9)
        snapshot = performance_snapshot()
        self.assertEqual(snapshot["runs"], 1)
        self.assertAlmostEqual(snapshot["net_pl"], 0.5)
        self.assertEqual(snapshot["trades_opened"], 2)
