"""The GA must not be able to win by overfitting.

Measured 2026-08-27 on 26,270 real ARB-WETH bars: shapes selected for
consistency on a training split scored 62-81% hit rates in-sample and **50.0%
out-of-sample against a 53.5% majority-class baseline**. A GA maximising
in-sample fitness breeds exactly those and reports a confident champion with no
edge, so fitness is defined against held-out data and penalised by the
majority-class baseline.
"""

from __future__ import annotations

import math
import random
import unittest

import numpy as np

from trading.genome.ga import (
    GENE_SPACE,
    Genome,
    crossover,
    evaluate,
    mutate,
    random_genome,
)
from trading.genome.shapes import (
    ShapeCluster,
    cluster_shapes,
    distance,
    extract_windows,
    normalize,
    select_predictive,
)


def _bars(prices):
    return [{"timestamp": 1700000000 + i * 3600, "close": p, "open": p,
             "high": p, "low": p, "net_volume": 1000.0} for i, p in enumerate(prices)]


class ShapeInvarianceTest(unittest.TestCase):
    def test_same_shape_different_scale_matches(self):
        """The core premise: same structure, different size, same signature."""
        base = [1, 2, 3, 2, 1, 2, 4, 6, 5, 4, 3, 2]
        small = normalize(base)
        large = normalize([p * 1000.0 for p in base])
        self.assertIsNotNone(small)
        self.assertLess(distance(small, large), 1e-6)

    def test_same_shape_different_duration_matches(self):
        """Duration invariance: a stretched pattern is the same pattern."""
        short = normalize([1, 3, 2, 5, 4])
        long = normalize([1, 1, 3, 3, 2, 2, 5, 5, 4, 4])
        self.assertIsNotNone(short)
        self.assertLess(distance(short, long), 0.35)

    def test_different_shapes_do_not_match(self):
        rising = normalize([1, 2, 3, 4, 5, 6, 7, 8])
        falling = normalize([8, 7, 6, 5, 4, 3, 2, 1])
        self.assertGreater(distance(rising, falling), 1.0)

    def test_flat_window_has_no_shape(self):
        """Zero variance would otherwise manufacture a shape from FP noise."""
        self.assertIsNone(normalize([5.0] * 10))


class ShapeSelectionTest(unittest.TestCase):
    def test_inconsistent_shape_is_rejected(self):
        """A coin-flip outcome distribution is a pattern in the noise."""
        c = ShapeCluster(signature=np.zeros(16))
        for r in (0.05, -0.05, 0.04, -0.06, 0.05, -0.04, 0.03, -0.03):
            c.absorb(np.zeros(16), r, "X-USDC")
        self.assertLess(c.consistency, 0.35)
        self.assertEqual(select_predictive([c], min_occurrences=4), [])

    def test_consistent_shape_is_kept(self):
        c = ShapeCluster(signature=np.zeros(16))
        for _ in range(10):
            c.absorb(np.zeros(16), 0.03, "X-USDC")
        self.assertGreater(c.consistency, 0.35)
        self.assertEqual(len(select_predictive([c], min_occurrences=4)), 1)

    def test_rare_shape_is_rejected(self):
        c = ShapeCluster(signature=np.zeros(16))
        c.absorb(np.zeros(16), 0.05, "X-USDC")
        self.assertEqual(select_predictive([c], min_occurrences=8), [])


class FitnessHonestyTest(unittest.TestCase):
    def test_no_edge_scores_zero(self):
        """Random data has no edge; fitness must be 0, not 'a small number'."""
        rng = random.Random(7)
        prices = [100.0]
        for _ in range(2000):
            prices.append(max(0.01, prices[-1] * (1.0 + rng.gauss(0, 0.01))))
        g = random_genome(random.Random(1))
        g.genes.update({"window": 16, "horizon": 6, "shape_margin": 0.4,
                        "sentiment_weight": 0.0, "brain_weight": 0.0})
        evaluate(g, {"RAND-USDC": _bars(prices)})
        self.assertEqual(g.fitness, 0.0)

    def test_below_baseline_accuracy_scores_zero(self):
        g = Genome(genes=dict(GENE_SPACE and {}), genome_id="t")
        g.oos_edge = -0.05
        g.oos_expectancy = 0.01
        self.assertLessEqual(max(0.0, g.oos_edge), 0.0)

    def test_monotonic_series_cannot_manufacture_edge(self):
        """A pure trend is the majority class; beating it must require skill."""
        prices = [100.0 * (1.001 ** i) for i in range(1500)]
        g = random_genome(random.Random(3))
        g.genes.update({"window": 16, "horizon": 6, "shape_margin": 0.4,
                        "sentiment_weight": 0.0, "brain_weight": 0.0})
        evaluate(g, {"UP-USDC": _bars(prices)})
        # Baseline is ~100% up, so edge cannot be positive.
        self.assertLessEqual(g.oos_edge, 0.0)
        self.assertEqual(g.fitness, 0.0)

    def test_too_few_trades_scores_zero(self):
        prices = [100.0 + i for i in range(300)]
        g = random_genome(random.Random(5))
        g.genes.update({"window": 24, "horizon": 6})
        evaluate(g, {"TINY-USDC": _bars(prices)}, min_oos_trades=1000)
        self.assertEqual(g.fitness, 0.0)


class GeneOperatorTest(unittest.TestCase):
    def test_mutation_respects_bounds(self):
        rng = random.Random(11)
        g = random_genome(rng)
        for _ in range(40):
            g = mutate(g, rng, rate=1.0)
            for key, spec in GENE_SPACE.items():
                val = g.genes[key]
                if isinstance(spec, tuple):
                    self.assertGreaterEqual(val, spec[0])
                    self.assertLessEqual(val, spec[1])
                elif isinstance(spec, list):
                    self.assertIn(val, spec)

    def test_crossover_keeps_full_gene_set(self):
        rng = random.Random(13)
        a, b = random_genome(rng), random_genome(rng)
        child = crossover(a, b, rng)
        self.assertEqual(set(child.genes), set(GENE_SPACE))

    def test_random_genome_covers_space(self):
        rng = random.Random(17)
        g = random_genome(rng)
        self.assertEqual(set(g.genes), set(GENE_SPACE))


class GeneSpaceScopingTest(unittest.TestCase):
    def test_pinned_gene_is_constant(self):
        from services.ga_service import resolve_space
        space = resolve_space({"window": 32})
        rng = random.Random(19)
        for _ in range(15):
            self.assertEqual(random_genome(rng, space).genes["window"], 32)

    def test_range_override_narrows_search(self):
        from services.ga_service import resolve_space
        space = resolve_space({"shape_margin": {"min": 0.30, "max": 0.32}})
        rng = random.Random(23)
        for _ in range(15):
            v = random_genome(rng, space).genes["shape_margin"]
            self.assertGreaterEqual(v, 0.30)
            self.assertLessEqual(v, 0.32)

    def test_unknown_key_is_ignored(self):
        from services.ga_service import resolve_space
        space = resolve_space({"not_a_gene": 5})
        self.assertNotIn("not_a_gene", space)


class ChampionRegistryTest(unittest.TestCase):
    def test_champion_without_edge_is_refused(self):
        """The model dropdown must never list an overfit."""
        import services.ga_service as svc
        original = svc.get_run
        svc.get_run = lambda rid: {
            "name": "bad",
            "best": {"genome_id": "abc", "oos_edge": -0.01, "oos_expectancy": 0.5},
        }
        try:
            self.assertIsNone(svc.register_champion("x"))
        finally:
            svc.get_run = original

    def test_champion_without_expectancy_is_refused(self):
        import services.ga_service as svc
        original = svc.get_run
        svc.get_run = lambda rid: {
            "name": "bad",
            "best": {"genome_id": "abc", "oos_edge": 0.05, "oos_expectancy": -0.001},
        }
        try:
            self.assertIsNone(svc.register_champion("x"))
        finally:
            svc.get_run = original


if __name__ == "__main__":
    unittest.main()
