"""Tests for the genetic search.

Written around the ways a search like this lies to you: fitting the sample it
was scored on, converging on a lucky outlier, polishing one local optimum
forever, and reporting a winner without an audit trail.
"""

from __future__ import annotations

import random

from django.test import SimpleTestCase

from .evolve import (MIN_SELECTED, Condition, Genome, evolve)


def _rows(n: int = 200, seed: int = 20260905) -> list:
    """Trades where ONE feature genuinely predicts return, and others do not.

    A search that cannot recover a planted signal cannot be trusted to find a
    real one, and a search that finds signal in the noise features is
    overfitting.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        hurst = rng.uniform(0.2, 0.8)
        # The planted relationship: high hurst pays, low hurst does not.
        base = 0.02 if hurst > 0.55 else -0.01
        out.append({
            "return": base + rng.gauss(0, 0.01),
            "hurst": hurst,
            "ticks_1h": rng.uniform(1, 20),          # pure noise
            "notional_usd": rng.uniform(0.5, 3.0),   # pure noise
            "hour_of_day": rng.randrange(24),        # pure noise
        })
    return out


class ConditionTests(SimpleTestCase):
    def test_a_missing_feature_answers_none_not_false(self):
        """None is not evidence against a rule.

        Counting an unanswerable row as False would make every rule look
        better on symbols with sparse data -- exactly the symbols where a
        rule is least trustworthy.
        """
        condition = Condition("hurst", ">", 0.5)
        self.assertIsNone(condition.holds({"other": 1}))
        self.assertTrue(condition.holds({"hurst": 0.9}))
        self.assertFalse(condition.holds({"hurst": 0.1}))

    def test_a_genome_is_unanswerable_if_any_clause_is(self):
        genome = Genome([Condition("hurst", ">", 0.5),
                         Condition("missing", ">", 1.0)], "test")
        self.assertIsNone(genome.selects({"hurst": 0.9}))


class SearchTests(SimpleTestCase):
    def test_it_recovers_a_planted_signal(self):
        result = evolve(_rows(), cost=0.0, generations=10, seed=1)
        self.assertTrue(result["ok"], result.get("reason"))
        self.assertTrue(result["survivors"], "found nothing in planted data")
        # The planted feature should appear in the best rule.
        self.assertIn("hurst", result["survivors"][0]["rule"])

    def test_it_finds_nothing_in_pure_noise(self):
        """The test that matters most.

        A search that reports survivors on noise is worse than useless: every
        run would produce confident rules, and none would hold up on money.
        """
        rng = random.Random(7)
        noise = [{"return": rng.gauss(0, 0.01),
                  "hurst": rng.uniform(0.2, 0.8),
                  "ticks_1h": rng.uniform(1, 20),
                  "notional_usd": rng.uniform(0.5, 3.0)}
                 for _ in range(200)]
        result = evolve(noise, cost=0.0, generations=10, seed=1)
        self.assertTrue(result["ok"])
        self.assertEqual(result["survivors"], [],
                         f"found signal in noise: {result['survivors']}")

    def test_judgment_is_on_data_the_search_never_fitted(self):
        result = evolve(_rows(), cost=0.0, generations=6, seed=1)
        self.assertGreater(result["holdout_rows"], 0)
        self.assertGreater(result["fit_rows"], 0)
        self.assertEqual(result["fit_rows"] + result["holdout_rows"],
                         result["trades"])

    def test_too_little_data_refuses_rather_than_guessing(self):
        result = evolve([{"return": 0.01, "hurst": 0.6}] * 4)
        self.assertFalse(result["ok"])
        self.assertIn("need", result["reason"])

    def test_stale_lineages_are_retired_and_reported(self):
        """An unauditable search is indistinguishable from overfitting."""
        result = evolve(_rows(), cost=0.0, generations=14, seed=3)
        self.assertTrue(result["lineages"])
        retired = [l for l in result["lineages"] if l["retired"]]
        self.assertTrue(retired, "no lineage ever went stale in 14 generations")
        self.assertTrue(result["attention_events"])


class AdvisorEscalationTests(SimpleTestCase):
    def test_the_search_does_not_ask_for_help_while_it_can_help_itself(self):
        """Escalating on every stall would waste the agent's attention."""
        calls = []

        def advisor(ctx):
            calls.append(ctx)
            return ["hurst"]

        evolve(_rows(), cost=0.0, generations=12, seed=777, advisor=advisor)
        self.assertEqual(calls, [],
                         "advisor consulted while unexplored features remained")

    def test_it_asks_once_every_feature_is_exhausted(self):
        calls = []

        def advisor(ctx):
            calls.append(ctx)
            return ["hurst"]

        # One feature only: the search runs out of unexplored territory at
        # the first stall and has no move of its own left.
        thin = [{"return": r["return"], "hurst": r["hurst"]} for r in _rows()]
        evolve(thin, cost=0.0, generations=12, seed=777, advisor=advisor)
        self.assertTrue(calls, "advisor was never consulted despite exhaustion")
        self.assertIn("stalled_lineage", calls[0])
        self.assertIn("features_exhausted", calls[0])

    def test_nonsense_advice_is_ignored(self):
        """Advice is a hint, not an instruction.

        A search that could be steered into features that do not exist would
        be worse than one with no advisor at all.
        """
        def advisor(ctx):
            return ["not_a_feature", "also_fake"]

        thin = [{"return": r["return"], "hurst": r["hurst"]} for r in _rows()]
        result = evolve(thin, cost=0.0, generations=10, seed=777,
                        advisor=advisor)
        self.assertTrue(result["ok"])

    def test_an_advisor_that_raises_does_not_end_the_search(self):
        def advisor(ctx):
            raise RuntimeError("agent unavailable")

        thin = [{"return": r["return"], "hurst": r["hurst"]} for r in _rows()]
        result = evolve(thin, cost=0.0, generations=10, seed=777,
                        advisor=advisor)
        self.assertTrue(result["ok"])
