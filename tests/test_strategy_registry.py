"""Strategy lifecycle: objectives, commissioning, and model provenance.

A strategy screen is only trustworthy if the things it lets you switch on have
actually earned it. Choosing "prioritise net profit" must reweight what a
search rewards -- it must never let a strategy with no out-of-sample edge be
commissioned.
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from services import strategy_registry as reg


def _genome(edge=0.05, expectancy=0.002, trades=300, fitness=0.5):
    return types.SimpleNamespace(
        oos_edge=edge, oos_expectancy=expectancy,
        oos_trades=trades, fitness=fitness,
    )


class ObjectiveScoringTest(unittest.TestCase):
    def test_no_edge_scores_zero_under_every_objective(self):
        """The objective reweights real performance; it never substitutes."""
        g = _genome(edge=-0.01, expectancy=5.0, trades=10000, fitness=9.9)
        for name in reg.OBJECTIVES:
            self.assertEqual(reg.score(g, name), 0.0, name)

    def test_net_profit_prefers_volume_of_profitable_trades(self):
        few = _genome(expectancy=0.01, trades=10)
        many = _genome(expectancy=0.01, trades=1000)
        self.assertGreater(reg.score(many, "net_profit"), reg.score(few, "net_profit"))

    def test_expectancy_prefers_per_trade_quality(self):
        fat = _genome(expectancy=0.05, trades=10)
        thin = _genome(expectancy=0.001, trades=1000)
        self.assertGreater(reg.score(fat, "expectancy"), reg.score(thin, "expectancy"))

    def test_accuracy_credits_only_edge_over_baseline(self):
        self.assertEqual(reg.score(_genome(edge=0.0), "accuracy"), 0.0)
        self.assertGreater(reg.score(_genome(edge=0.08), "accuracy"), 0.0)

    def test_consistency_discounts_small_samples(self):
        small = _genome(edge=0.10, trades=10)
        large = _genome(edge=0.10, trades=1000)
        self.assertGreater(reg.score(large, "consistency"), reg.score(small, "consistency"))

    def test_unknown_objective_falls_back_to_balanced(self):
        g = _genome(fitness=0.42)
        self.assertEqual(reg.score(g, "not_an_objective"), reg.score(g, "balanced"))


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(reg, "REGISTRY_PATH", Path(self.tmp.name) / "reg.json")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_commission_requires_out_of_sample_edge(self):
        s = reg.register_strategy(name="overfit", metrics={"oos_edge": -0.02})
        out = reg.set_commissioned(s["strategy_id"], True)
        self.assertFalse(out["commissioned"])
        self.assertIn("no out-of-sample edge", out["commission_error"])

    def test_commission_succeeds_with_edge(self):
        s = reg.register_strategy(name="good", metrics={"oos_edge": 0.04})
        out = reg.set_commissioned(s["strategy_id"], True)
        self.assertTrue(out["commissioned"])
        self.assertNotIn("commission_error", out)

    def test_decommission_is_always_allowed(self):
        """Turning something OFF must never be blocked."""
        s = reg.register_strategy(name="x", metrics={"oos_edge": 0.04}, commissioned=True)
        out = reg.set_commissioned(s["strategy_id"], False)
        self.assertFalse(out["commissioned"])

    def test_strategy_records_its_model_provenance(self):
        s = reg.register_strategy(name="x", model_id="brain-7", model_name="Wizard 7")
        self.assertEqual(s["model_id"], "brain-7")
        self.assertEqual(s["model_name"], "Wizard 7")

    def test_missing_model_is_labelled_not_blank(self):
        s = reg.register_strategy(name="x")
        self.assertEqual(s["model_name"], "(no brain)")

    def test_delete_removes_strategy(self):
        s = reg.register_strategy(name="x")
        self.assertTrue(reg.delete_strategy(s["strategy_id"]))
        self.assertIsNone(reg.get_strategy(s["strategy_id"]))


class CrossModelExperimentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(reg, "REGISTRY_PATH", Path(self.tmp.name) / "reg.json")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_experiments_accumulate_rather_than_overwrite(self):
        """Testing on a second brain must not erase the first result."""
        s = reg.register_strategy(
            name="shape-a", model_id="brain-1", model_name="Wizard 1",
            metrics={"oos_edge": 0.03},
        )
        sid = s["strategy_id"]
        reg.add_experiment(sid, run_id="r2", model_id="brain-2",
                           model_name="Wizard 2", objective="accuracy",
                           metrics={"oos_edge": 0.06})
        reg.add_experiment(sid, run_id="r3", model_id="brain-3",
                           model_name="Wizard 3", objective="accuracy",
                           metrics={"oos_edge": 0.01})
        rows = reg.compare_experiments(sid)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["model_name"], "Wizard 2")   # ranked by edge
        names = {r["model_name"] for r in rows}
        self.assertEqual(names, {"Wizard 1", "Wizard 2", "Wizard 3"})

    def test_origin_row_is_marked(self):
        s = reg.register_strategy(name="x", model_id="b1", model_name="B1",
                                  metrics={"oos_edge": 0.9})
        rows = reg.compare_experiments(s["strategy_id"])
        self.assertTrue(rows[0]["origin"])

    def test_compare_unknown_strategy_is_empty(self):
        self.assertEqual(reg.compare_experiments("nope"), [])



class LifetimeMetricsTest(unittest.TestCase):
    """Lifetime is append-only and must survive a ledger reset."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(reg, "REGISTRY_PATH", Path(self.tmp.name) / "reg.json")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_outcomes_accumulate(self):
        for p in (0.10, -0.05, 0.20, -0.05):
            reg.record_outcome("s1", profit=p, mode="ghost", symbol="A-USDC")
        m = reg.lifetime_metrics("s1")["ghost"]
        self.assertEqual(m["trades"], 4)
        self.assertEqual(m["wins"], 2)
        self.assertAlmostEqual(m["total_profit"], 0.20, places=6)
        self.assertAlmostEqual(m["profit_factor"], 3.0, places=6)

    def test_unregistered_strategy_is_auto_registered(self):
        """Pre-existing strategies must not be dropped on the floor."""
        reg.record_outcome("legacy_strategy", profit=0.05, mode="ghost")
        entry = reg.get_strategy("legacy_strategy")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.get("auto_registered"))

    def test_ghost_and_live_are_never_summed(self):
        """Simulated and realised profit are different claims."""
        reg.record_outcome("s2", profit=1.0, mode="ghost")
        reg.record_outcome("s2", profit=-0.5, mode="live")
        m = reg.lifetime_metrics("s2")
        self.assertEqual(m["ghost"]["trades"], 1)
        self.assertEqual(m["live"]["trades"], 1)
        self.assertAlmostEqual(m["ghost"]["total_profit"], 1.0)
        self.assertAlmostEqual(m["live"]["total_profit"], -0.5)

    def test_tracks_worst_case_risk(self):
        for p in (0.1, -0.2, -0.3, -0.1, 0.05):
            reg.record_outcome("s3", profit=p, mode="ghost")
        m = reg.lifetime_metrics("s3")["ghost"]
        self.assertEqual(m["max_consecutive_losses"], 3)
        self.assertAlmostEqual(m["worst"], -0.3, places=6)
        self.assertGreater(m["max_drawdown"], 0.0)

    def test_empty_strategy_reports_zero_not_error(self):
        m = reg.lifetime_metrics("never_traded")
        self.assertEqual(m["ghost"]["trades"], 0)


if __name__ == "__main__":
    unittest.main()
