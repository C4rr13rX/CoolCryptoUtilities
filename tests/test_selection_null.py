"""A search that keeps the best of N must be judged against the best of N.

Pinning the error that produced four positive-looking runs here on
2026-09-16: every seed showed a positive held-out edge, a naive binomial
z-test called two of them significant at p<=0.002, and against the correct
null the two largest-sample runs scored BELOW the median of pure chance.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.genome import selection_null as N


class TheNullItself(unittest.TestCase):

    def test_screening_more_genomes_raises_the_bar(self):
        """The whole point: the more you look, the luckier the best looks."""
        _, few, _ = N.best_of_n_null(genomes_screened=2, trades=300,
                                     baseline=0.5, trials=200, seed=1)
        _, many, _ = N.best_of_n_null(genomes_screened=64, trades=300,
                                      baseline=0.5, trials=200, seed=1)
        self.assertGreater(many, few)

    def test_a_bigger_sample_lowers_the_bar(self):
        """Luck has less room on more trades -- which is why an artifact's
        edge decays as n grows while a real one holds."""
        _, small, _ = N.best_of_n_null(genomes_screened=32, trades=100,
                                       baseline=0.5, trials=200, seed=2)
        _, large, _ = N.best_of_n_null(genomes_screened=32, trades=2000,
                                       baseline=0.5, trials=200, seed=2)
        self.assertGreater(small, large)

    def test_it_is_deterministic_for_a_given_seed(self):
        a = N.best_of_n_null(genomes_screened=8, trades=200, baseline=0.5,
                             trials=100, seed=7)[1]
        b = N.best_of_n_null(genomes_screened=8, trades=200, baseline=0.5,
                             trials=100, seed=7)[1]
        self.assertEqual(a, b)

    def test_degenerate_inputs_do_not_raise(self):
        self.assertEqual(N.best_of_n_null(genomes_screened=0, trades=10,
                                          baseline=0.5)[0], 0.0)
        self.assertEqual(N.best_of_n_null(genomes_screened=4, trades=0,
                                          baseline=0.5)[0], 0.0)


class TheRealRuns(unittest.TestCase):
    """The four runs that prompted this, by their measured numbers."""

    def _judge(self, acc, base, trades):
        return N.judge(observed_accuracy=acc, baseline=base, trades=trades,
                       population=8, generations=4, trials=300, seed=5)

    def test_the_largest_sample_run_fails(self):
        """seed 404: +1.27pp on 645 trades, below the null median."""
        v = self._judge(0.5178, 0.5052, 645)
        self.assertFalse(v.passed)
        self.assertLessEqual(v.observed_edge, v.null_median)

    def test_the_p_equals_0002_run_is_merely_marginal(self):
        """seed 77: +8.67pp on 320 trades.

        A binomial z-test calls this p=0.002 -- a one-in-five-hundred result.
        Against best-of-32 it lands at the 95th-97th percentile, flipping
        either side of the threshold with simulation noise. That is the whole
        correction: not "impossible by chance" but "about one in twenty",
        found by a search that draws far more than twenty times. Asserting
        the PERCENTILE rather than passed/failed, because a boundary case
        that flips with the simulation seed is exactly what it is, and a test
        that pretended otherwise would be the same over-claiming this module
        exists to stop."""
        v = self._judge(0.5875, 0.5008, 320)
        self.assertGreater(v.percentile_of_null, 93.0)
        self.assertLess(v.percentile_of_null, 99.0)

    def test_a_genuinely_large_edge_on_a_real_sample_passes(self):
        """The guard must not reject everything, or it says nothing. A 15pp
        edge over 3000 trades is far outside best-of-32 luck."""
        v = self._judge(0.65, 0.50, 3000)
        self.assertTrue(v.passed, v.reason)

    def test_every_verdict_explains_itself(self):
        for args in ((0.5178, 0.5052, 645), (0.5875, 0.5008, 320),
                     (0.65, 0.50, 3000)):
            self.assertTrue(self._judge(*args).reason)

    def test_no_trades_is_not_a_pass(self):
        v = self._judge(0.99, 0.50, 0)
        self.assertFalse(v.passed)
        self.assertIn("nothing was actually tested", v.reason)

    def test_the_screening_count_is_an_upper_bound(self):
        """population*generations over-counts, because elites are re-used
        rather than re-drawn. Over-counting makes the null HARDER, which is
        the safe direction -- a pass cannot be an artefact of it."""
        v = self._judge(0.5875, 0.5008, 320)
        self.assertEqual(v.genomes_screened, 32)


class SampleSizeCorrelation(unittest.TestCase):

    def test_the_observed_artifact_shape_is_strongly_negative(self):
        """The four real runs: -0.842."""
        runs = [{"trades": 132, "edge": 0.2041},
                {"trades": 320, "edge": 0.0867},
                {"trades": 276, "edge": 0.0546},
                {"trades": 645, "edge": 0.0127}]
        corr = N.edge_sample_correlation(runs)
        self.assertIsNotNone(corr)
        self.assertLess(corr, -0.7)

    def test_a_real_effect_holds_as_the_sample_grows(self):
        runs = [{"trades": 100, "edge": 0.05}, {"trades": 500, "edge": 0.05},
                {"trades": 2000, "edge": 0.05}]
        self.assertIsNone(N.edge_sample_correlation(runs))

    def test_two_runs_cannot_manufacture_a_correlation(self):
        """Two points always correlate perfectly."""
        self.assertIsNone(N.edge_sample_correlation(
            [{"trades": 100, "edge": 0.2}, {"trades": 900, "edge": 0.01}]))

    def test_runs_with_no_trades_are_ignored(self):
        self.assertIsNone(N.edge_sample_correlation(
            [{"trades": 0, "edge": 0.5}, {"trades": 0, "edge": 0.1},
             {"trades": 0, "edge": 0.3}]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
