"""A search that stopped discovering must say so and do something about it.

The failure being pinned is silent by construction: the generation counter
climbs, on_generation keeps firing, the log keeps scrolling, and the best
fitness has not moved in forty generations. Nothing in a stale run looks
different from a productive one from the outside, which is why every branch
here is tested without running a real search -- a bug in the restart path
would otherwise first appear deep into a long run, which is exactly where
nobody is watching.
"""
import random
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.genome import stagnation as S


@dataclass
class FakeGenome:
    """Enough of a Genome to be measured: genes and a fitness."""
    genes: Dict[str, Any] = field(default_factory=dict)
    fitness: float = 0.0


def pop_of(fitnesses, distinct=True):
    return [FakeGenome(genes={"a": i if distinct else 0}, fitness=f)
            for i, f in enumerate(fitnesses)]


class Diversity(unittest.TestCase):

    def test_identical_genes_are_one_search_point(self):
        """Two genomes with the same genes are the same point, whatever
        their ids or scores -- counting them as distinct is precisely the
        error that lets a collapsed population look healthy."""
        pop = [FakeGenome(genes={"a": 1}, fitness=0.5),
               FakeGenome(genes={"a": 1}, fitness=0.9)]
        self.assertEqual(S.diversity(pop), 0.5)

    def test_all_distinct_is_one(self):
        self.assertEqual(S.diversity(pop_of([1, 2, 3, 4])), 1.0)

    def test_floats_are_bucketed(self):
        """14.0000001 and 14.0 are not a different search point."""
        pop = [FakeGenome(genes={"a": 14.0}), FakeGenome(genes={"a": 14.0000000001})]
        self.assertEqual(S.diversity(pop), 0.5)

    def test_empty_population_is_not_a_crash(self):
        self.assertEqual(S.diversity([]), 0.0)


class Improvement(unittest.TestCase):

    def test_a_better_best_resets_the_clock(self):
        m = S.StagnationMonitor(patience=3)
        m.observe(0, pop_of([0.1]))
        m.observe(1, pop_of([0.2]))
        self.assertEqual(m.stale_generations, 0)
        self.assertAlmostEqual(m.best_fitness, 0.2)

    def test_an_unchanged_best_is_not_progress(self):
        """The strict rule: equalling the best is not improving on it."""
        m = S.StagnationMonitor(patience=3)
        m.observe(0, pop_of([0.5]))
        for gen in (1, 2):
            m.observe(gen, pop_of([0.5]))
        self.assertEqual(m.stale_generations, 2)

    def test_a_rising_mean_with_a_flat_best_is_not_progress(self):
        """A stale run's favourite way to justify itself: the mean drifts up
        as elites accumulate while the frontier has not moved at all."""
        m = S.StagnationMonitor(patience=10)
        m.observe(0, pop_of([0.5, 0.0, 0.0, 0.0]))
        m.observe(1, pop_of([0.5, 0.4, 0.3, 0.2]))
        self.assertEqual(m.stale_generations, 1)

    def test_noise_below_min_delta_is_not_improvement(self):
        m = S.StagnationMonitor(patience=10, min_delta=1e-4)
        m.observe(0, pop_of([0.5]))
        m.observe(1, pop_of([0.50000001]))
        self.assertEqual(m.stale_generations, 1)


class Escapes(unittest.TestCase):

    def test_within_patience_it_leaves_the_search_alone(self):
        m = S.StagnationMonitor(patience=5)
        m.observe(0, pop_of([0.5]))
        v = m.observe(1, pop_of([0.5]))
        self.assertEqual(v["action"], "continue")

    def test_collapsed_diversity_takes_the_cheap_escape_first(self):
        m = S.StagnationMonitor(patience=2, diversity_floor=0.5)
        same = [FakeGenome(genes={"a": 0}, fitness=0.5) for _ in range(4)]
        m.observe(0, same)
        m.observe(1, same)
        v = m.observe(2, same)
        self.assertEqual(v["action"], "immigrate")
        self.assertIn("collapsed", v["reason"])

    def test_spread_out_and_going_nowhere_widens_the_box(self):
        """Diversity is fine and fitness still is not moving: the population
        is spread out and none of it is any good, so the answer is not in the
        box being searched."""
        m = S.StagnationMonitor(patience=2, diversity_floor=0.5)
        varied = pop_of([0.5, 0.4, 0.3, 0.2])
        m.observe(0, varied)
        m.observe(1, varied)
        v = m.observe(2, varied)
        self.assertEqual(v["action"], "widen")

    def test_restart_is_the_last_resort_not_the_first(self):
        """Only after the space has been widened its maximum number of
        times, because a restart discards every evaluation already paid for."""
        m = S.StagnationMonitor(patience=1, diversity_floor=0.5, max_widenings=2)
        varied = pop_of([0.5, 0.4, 0.3, 0.2])
        actions = []
        for gen in range(8):
            actions.append(m.observe(gen, varied)["action"])
        self.assertEqual(actions.count("widen"), 2)
        self.assertIn("restart", actions)
        self.assertLess(actions.index("widen"), actions.index("restart"))

    def test_an_escape_resets_the_clock(self):
        """Otherwise a run that immigrated once would immigrate every
        generation forever, never giving the immigrants a chance to show
        whether they helped."""
        m = S.StagnationMonitor(patience=2, diversity_floor=0.5)
        same = [FakeGenome(genes={"a": 0}, fitness=0.5) for _ in range(4)]
        for gen in range(3):
            m.observe(gen, same)
        self.assertEqual(m.stale_generations, 0)
        self.assertEqual(m.observe(3, same)["action"], "continue")

    def test_every_escape_is_recorded(self):
        """A run that ends with no champion must be able to say what it tried."""
        m = S.StagnationMonitor(patience=1, diversity_floor=0.5, max_widenings=1)
        varied = pop_of([0.5, 0.4, 0.3, 0.2])
        for gen in range(6):
            m.observe(gen, varied)
        self.assertTrue(m.history)
        self.assertTrue(all("reason" in h and h["reason"] for h in m.history))


class Immigrants(unittest.TestCase):

    def setUp(self):
        self.rng = random.Random(7)
        self.counter = iter(range(1000))
        self.make = lambda: FakeGenome(genes={"new": next(self.counter)},
                                       fitness=0.0)

    def test_the_elites_are_never_what_gets_thrown_away(self):
        pop = pop_of([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        before = [g.fitness for g in pop[:4]]
        S.inject_immigrants(pop, self.rng, self.make, fraction=0.5, keep_elite=4)
        self.assertEqual([g.fitness for g in pop[:4]], before)

    def test_it_replaces_from_the_back(self):
        pop = pop_of([0.9, 0.8, 0.7, 0.6])
        S.inject_immigrants(pop, self.rng, self.make, fraction=0.5, keep_elite=2)
        self.assertIn("new", pop[-1].genes)

    def test_it_never_replaces_the_whole_population(self):
        """That is a restart, and a restart is a separate, later, more
        expensive decision."""
        pop = pop_of([0.9, 0.8, 0.7, 0.6])
        n = S.inject_immigrants(pop, self.rng, self.make, fraction=1.0, keep_elite=2)
        self.assertEqual(n, 2)

    def test_an_empty_population_is_not_a_crash(self):
        self.assertEqual(S.inject_immigrants([], self.rng, self.make), 0)


class Widening(unittest.TestCase):

    def test_numeric_ranges_stretch_around_their_midpoint(self):
        out = S.widen_space({"x": (10.0, 20.0)}, factor=2.0)
        self.assertEqual(out["x"], (5.0, 25.0))

    def test_integer_ranges_stay_integers(self):
        """A lookback of 14.7 bars is not a thing, and rounding it later
        would collapse several widened values back onto one point."""
        out = S.widen_space({"lookback": (10, 20)}, factor=1.5)
        self.assertTrue(all(isinstance(v, int) for v in out["lookback"]))

    def test_choice_lists_are_left_alone(self):
        """Inventing a categorical value the evaluator has never seen is a
        crash in a worker thread, not a wider search."""
        space = {"mode": ["ema", "rsi"], "flag": True}
        out = S.widen_space(space, factor=2.0)
        self.assertEqual(out["mode"], ["ema", "rsi"])
        self.assertEqual(out["flag"], True)

    def test_counts_never_widen_below_one(self):
        """A negative lookback reaches the evaluator as a slice that
        silently returns nothing rather than raising."""
        out = S.widen_space({"lookback": (2, 6)}, factor=4.0)
        self.assertGreaterEqual(out["lookback"][0], 1)

    def test_an_integer_range_never_inverts_or_collapses(self):
        out = S.widen_space({"n": (5, 6)}, factor=0.01)
        self.assertLess(out["n"][0], out["n"][1])

    def test_widening_does_not_mutate_the_original(self):
        space = {"x": (10.0, 20.0)}
        S.widen_space(space, factor=2.0)
        self.assertEqual(space["x"], (10.0, 20.0))


class RealGeneSpace(unittest.TestCase):
    """Widening must survive the ACTUAL space the GA ships with, not just
    the tidy fabricated one above."""

    def test_the_shipped_space_widens_without_raising(self):
        from trading.genome.ga import GENE_SPACE
        out = S.widen_space(GENE_SPACE, factor=1.5)
        self.assertEqual(set(out), set(GENE_SPACE))

    def test_no_shipped_count_widens_below_one(self):
        from trading.genome.ga import GENE_SPACE
        out = S.widen_space(GENE_SPACE, factor=3.0)
        for key, val in out.items():
            if (isinstance(val, (list, tuple)) and len(val) == 2
                    and all(isinstance(v, int) for v in val)):
                self.assertGreaterEqual(
                    val[0], 1, "%s widened to %r" % (key, val))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class WiredIntoEvolve(unittest.TestCase):
    """The monitor is worth nothing if evolve() does not act on it."""

    def setUp(self):
        import trading.genome.ga as ga
        self.ga = ga
        self._real = ga.evaluate

    def tearDown(self):
        self.ga.evaluate = self._real

    def _flat(self, score=0.25):
        def ev(g, *a, **k):
            g.fitness = score
            g.oos_trades = 5
        self.ga.evaluate = ev

    def test_a_search_that_never_improves_escapes_rather_than_grinding(self):
        self._flat()
        seen = []
        self.ga.evolve({}, population=10, generations=25, elite=3, seed=1,
                       stagnation_patience=3, on_stagnation=seen.append)
        self.assertTrue(seen, "25 generations with a flat score took no escape")

    def test_the_escapes_climb_the_ladder_cheapest_first(self):
        self._flat()
        seen = []
        self.ga.evolve({}, population=10, generations=25, elite=3, seed=1,
                       stagnation_patience=3, on_stagnation=seen.append)
        actions = [e["action"] for e in seen]
        self.assertIn("restart", actions)
        self.assertLess(actions.index("widen"), actions.index("restart"))

    def test_a_restart_never_loses_the_best_genome(self):
        """Otherwise a longer search reports a WORSE champion than a shorter
        one, and nothing anywhere says why."""
        state = {"n": 0}

        def ev(g, *a, **k):
            state["n"] += 1
            g.fitness = 0.9 if state["n"] == 3 else 0.1
            g.oos_trades = 5
        self.ga.evaluate = ev
        pop = self.ga.evolve({}, population=8, generations=30, elite=2,
                             seed=5, stagnation_patience=2)
        self.assertAlmostEqual(pop[0].fitness, 0.9)

    def test_patience_zero_is_exactly_the_old_behaviour(self):
        self._flat()
        seen = []
        self.ga.evolve({}, population=6, generations=12, elite=2, seed=1,
                       stagnation_patience=0, on_stagnation=seen.append)
        self.assertEqual(seen, [])

    def test_a_raising_callback_cannot_kill_the_search(self):
        """A reporting callback must never be able to take down a search
        that is already in trouble."""
        self._flat()

        def boom(_):
            raise RuntimeError("reporting exploded")
        pop = self.ga.evolve({}, population=6, generations=12, elite=2,
                             seed=1, stagnation_patience=2, on_stagnation=boom)
        self.assertTrue(pop)
