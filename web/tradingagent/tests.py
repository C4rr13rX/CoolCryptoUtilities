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


class DataLabBridgeTest(TestCase):
    """The agent may fetch data, but not whatever it likes whenever it likes.

    An agent that can start any job will start them constantly, and this box
    already had a download flood saturate its connection budget.
    """

    def test_only_allowlisted_jobs_can_start(self):
        from .datalab_bridge import request_job

        for job in ("shell", "rm -rf /", "flush", "", "dumpdata"):
            result = request_job(job)
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "job_not_allowed")

    def test_the_allowlist_is_the_three_real_jobs(self):
        from .datalab_bridge import available_jobs

        self.assertEqual(set(available_jobs()),
                         {"download2000", "make2000index", "make_assignments"})

    def test_junk_symbols_are_dropped_not_watched(self):
        from .datalab_bridge import add_symbols

        result = add_symbols(["", "   ", "HAS SPACE", "A" * 40])
        self.assertFalse(result["ok"])
        self.assertEqual(result["added"], [])

    def test_symbol_adds_are_capped(self):
        """Watching everything thins the feed for every symbol."""
        from .datalab_bridge import MAX_ADDS_PER_PASS, add_symbols

        many = [f"TOK{i}-USDC" for i in range(MAX_ADDS_PER_PASS + 20)]
        result = add_symbols(many)
        if result.get("ok"):
            self.assertLessEqual(len(result["added"]), MAX_ADDS_PER_PASS)

    def test_candidates_come_from_the_stream_the_executor_prices(self):
        """A candidate the executor cannot price cannot be exited."""
        from .datalab_bridge import candidate_tokens

        for candidate in candidate_tokens(limit=5):
            self.assertIn("symbol", candidate)
            self.assertIn("volatility_1h_pct", candidate)
            self.assertGreaterEqual(candidate["ticks_1h"], 5,
                                    "a symbol with no ticks is not a candidate")


class PromotionRequiresEvidenceTest(TestCase):
    """Spending real money is earned, never configured.

    A tier is the ceiling on what one mistake can cost, so raising it on
    request rather than on results would make the whole ladder decorative.
    """

    def setUp(self):
        from django.contrib.auth import get_user_model

        user = get_user_model().objects.create_user("promo", password="x")
        self.client.force_login(user)

    def test_tier_cannot_be_set_through_config(self):
        import json

        response = self.client.post(
            "/api/trading-agent/config/",
            data=json.dumps({"tier": "normal"}),
            content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(AgentConfig.load().tier, RiskTier.GHOST)

    def test_promotion_is_refused_without_a_sample(self):
        import json

        response = self.client.post(
            "/api/trading-agent/promote/",
            data=json.dumps({"direction": "up"}),
            content_type="application/json")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(AgentConfig.load().tier, RiskTier.GHOST)

    def test_promotion_is_refused_when_the_account_shrank(self):
        import json

        for _ in range(25):
            AgentRun.objects.create(agent="t", status=AgentRun.Status.COMPLETED,
                                    trades_closed=1, net_pl=-0.02)
        response = self.client.post(
            "/api/trading-agent/promote/",
            data=json.dumps({"direction": "up"}),
            content_type="application/json")
        self.assertEqual(response.status_code, 409)
        self.assertIn("net P/L", response.json()["detail"])
        self.assertEqual(AgentConfig.load().tier, RiskTier.GHOST)

    def test_promotion_succeeds_on_measured_results(self):
        import json

        for _ in range(25):
            AgentRun.objects.create(agent="t", status=AgentRun.Status.COMPLETED,
                                    trades_closed=1, net_pl=0.05)
        response = self.client.post(
            "/api/trading-agent/promote/",
            data=json.dumps({"direction": "up"}),
            content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(AgentConfig.load().tier, RiskTier.MICRO)

    def test_demotion_never_needs_justifying(self):
        """Reducing risk is always allowed, whatever the record says."""
        import json

        config = AgentConfig.load()
        config.tier = RiskTier.SMALL
        config.save()
        response = self.client.post(
            "/api/trading-agent/promote/",
            data=json.dumps({"direction": "down"}),
            content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(AgentConfig.load().tier, RiskTier.MICRO)


class MathAuditTest(TestCase):
    """The audit must refuse to answer rather than guess.

    A confident number computed from nothing is more dangerous than an
    admitted gap: this repo shipped four strategies whose entire records were
    invented, and every one would have failed a sample-size test.
    """

    def test_no_trades_is_reported_as_insufficient_not_as_zero_edge(self):
        from .mathaudit import probability, statistics

        self.assertFalse(probability([])["sufficient"])
        self.assertFalse(statistics([])["sufficient"])

    def test_a_small_sample_is_unproven_however_good_it_looks(self):
        from .mathaudit import statistics

        result = statistics([0.05] * 5)
        self.assertFalse(result["significant_at_05"])
        self.assertIn("UNPROVEN", result["verdict"])

    def test_an_interval_containing_zero_is_not_an_edge(self):
        """Alternating wins and losses average to nothing."""
        from .mathaudit import statistics

        result = statistics([0.05, -0.05] * 15)
        self.assertFalse(result["significant_at_05"])
        self.assertIn("NOT SIGNIFICANT", result["verdict"])

    def test_a_real_edge_is_reported_as_significant(self):
        """Real returns vary. Identical values have zero variance, so the
        t-test cannot run on them at all -- the sample must look like trading."""
        from .mathaudit import statistics

        result = statistics([0.02, 0.021, 0.019, 0.022, 0.018] * 6)
        self.assertTrue(result["significant_at_05"])
        self.assertIn("POSITIVE EDGE", result["verdict"])

    def test_zero_variance_is_not_mistaken_for_significance(self):
        """Identical outcomes are a data artifact, never an edge."""
        from .mathaudit import statistics

        result = statistics([0.02] * 30)
        self.assertFalse(result["significant_at_05"])

    def test_a_losing_strategy_is_named_as_losing(self):
        from .mathaudit import statistics

        result = statistics([-0.02] * 30)
        self.assertIn("NEGATIVE EDGE", result["verdict"])

    def test_expectancy_is_reported_net_of_the_round_trip(self):
        """A gross edge smaller than the fee is not an edge."""
        from .mathaudit import probability

        # Wins slightly more often than not, but by less than the fee costs.
        result = probability([0.003] * 12 + [-0.002] * 8, round_trip_cost=0.0065)
        self.assertGreater(result["expectancy_gross"], 0.0)
        self.assertLess(result["expectancy_net"], 0.0)
        self.assertFalse(result["edge_survives_fees"])

    def test_kelly_is_halved_because_the_measured_edge_is_not_the_true_one(self):
        from .mathaudit import probability

        result = probability([0.05] * 15 + [-0.02] * 5)
        self.assertAlmostEqual(result["kelly_half"],
                               result["kelly_fraction"] / 2.0, places=6)

    def test_calculus_sees_a_strategy_handing_back_its_peak(self):
        """A total says where we are; its derivative says if we still are."""
        from .mathaudit import calculus

        rising_then_falling = [0.05] * 10 + [-0.04] * 10
        result = calculus(rising_then_falling)
        self.assertTrue(result["giving_back"])
        self.assertGreater(result["max_drawdown"], 0.0)

    def test_calculus_distinguishes_improving_from_decaying(self):
        from .mathaudit import calculus

        # A perfectly straight rising line must not read as decaying: the
        # second derivative is ~-1e-18 from float subtraction, not a trend.
        self.assertEqual(calculus([0.01] * 25)["direction"], "improving")
        self.assertEqual(calculus([-0.01] * 25)["direction"], "worsening")
        # A genuinely fading strategy still reads as decaying.
        fading = [0.05 - 0.002 * i for i in range(25)]
        self.assertEqual(calculus(fading)["direction"], "decaying")

    def test_algebra_counts_refusals_as_loudly_as_fills(self):
        """A pipeline refusing 26 entries and filling 2 made 28 decisions."""
        from .mathaudit import algebra

        actions = ([{"status": "live-swap-settled"}] * 2
                   + [{"status": "entry-refused-duplicate"}] * 26)
        result = algebra(actions)
        self.assertEqual(result["settled_swaps"], 2)
        self.assertEqual(result["refused_or_failed"], 26)
        self.assertEqual(result["actions_total"], 28)

    def test_unclosed_positions_are_surfaced(self):
        """Entries without exits are capital that went out and stayed out."""
        from .mathaudit import algebra

        actions = ([{"status": "live-entry"}] * 7 + [{"status": "live-exit"}] * 2)
        self.assertEqual(algebra(actions)["unclosed"], 5)


class BusSchedulerAwarenessTest(TestCase):
    """The agent shares a wallet with the scheduler.

    Buying a symbol the scheduler is mid-route on does not open a separate
    position -- it moves the shared balance out from under a plan already in
    flight. The agent may still choose to, but not unknowingly, and the plan
    to return the capital travels with the decision.
    """

    def test_a_deadline_inside_a_round_trip_is_not_feasible(self):
        """Settlement takes minutes, so a 2-minute window cannot be used."""
        from .bus_bridge import return_plan

        plan = return_plan({"symbol": "PEPE-USDC", "horizon": "30m",
                            "due_in_sec": 120.0}, clip_usd=0.75)
        self.assertFalse(plan["feasible"])
        self.assertIn("miss its window", plan["detail"])

    def test_a_comfortable_deadline_is_feasible(self):
        from .bus_bridge import return_plan

        plan = return_plan({"symbol": "ARB-USDC", "horizon": "1d",
                            "due_in_sec": 43000.0}, clip_usd=0.75)
        self.assertTrue(plan["feasible"])

    def test_an_overdue_position_says_to_close_it_first(self):
        """Capital the scheduler is waiting on is worth more than a new entry."""
        from .bus_bridge import return_plan

        plan = return_plan({"symbol": "CBETH-USDC", "horizon": "10m",
                            "due_in_sec": -600.0}, clip_usd=0.75)
        self.assertFalse(plan["feasible"])
        self.assertIn("OVERDUE", plan["detail"])

    def test_a_missing_horizon_is_admitted_not_assumed_on_schedule(self):
        """No deadline means it cannot be checked, not that it is fine."""
        from .bus_bridge import return_plan

        plan = return_plan({"symbol": "VIRTUAL-USDC", "horizon": "",
                            "due_in_sec": None}, clip_usd=0.75)
        self.assertIsNone(plan["feasible"])
        self.assertIn("no recorded horizon", plan["detail"])

    def test_horizon_labels_resolve_to_seconds(self):
        from .bus_bridge import _horizon_seconds

        self.assertEqual(_horizon_seconds("30m"), 1800)
        self.assertEqual(_horizon_seconds("1d"), 86400)
        self.assertEqual(_horizon_seconds("1w"), 604800)
        self.assertIsNone(_horizon_seconds("nonsense"))

    def test_the_briefing_speaks_even_with_nothing_riding(self):
        """Silence would read as 'no scheduler', not 'nothing committed'."""
        from .bus_bridge import bus_briefing
        from unittest import mock

        with mock.patch("tradingagent.bus_bridge.scheduled_commitments",
                        return_value=[]):
            lines = bus_briefing(0.75)
        self.assertTrue(lines)
        self.assertIn("no open commitments", lines[0])
