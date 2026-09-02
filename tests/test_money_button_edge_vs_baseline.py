"""The tool that decides whether money_button's timing is worth anything.

scripts/money_button_edge_vs_baseline.py exists because the gate census cannot
separate a strategy's SELECTION from the drift of whatever it selected. Measured
2026-09-02 over 168h, the two symbols money_button fired on had opposite drifts
(BSTONK +0.938% mean 12-min move, BASECAT -0.296%), so "it lost money" and "it
chose badly" are different claims and only the second is about the strategy.

If this comparison is wrong, every conclusion drawn from it about whether to
fund the lane is wrong too, so the machinery is pinned here: an unresolved hold
must be dropped rather than counted flat, and the permutation test must actually
discriminate a good entry set from a random one.
"""

from __future__ import annotations

import random
import unittest

from scripts.money_button_edge_vs_baseline import _forward_return, _permutation_p


class ForwardReturn(unittest.TestCase):
    def test_uses_the_first_tick_at_or_past_the_hold(self) -> None:
        ticks = [(0.0, 100.0, 0.0), (300.0, 105.0, 0.0), (720.0, 110.0, 0.0)]
        # 12-minute hold from t=0 lands on the t=720 tick, not the t=300 one.
        self.assertAlmostEqual(_forward_return(ticks, 0, 720.0), 0.10, places=9)

    def test_unresolved_hold_is_dropped_not_counted_flat(self) -> None:
        """A hold running off the end of the data has no outcome.

        Returning 0.0 here would quietly add a flat trade to whichever
        population the walk happened to be filling, biasing both the strategy
        mean and the null toward zero.
        """
        ticks = [(0.0, 100.0, 0.0), (60.0, 101.0, 0.0)]
        self.assertIsNone(_forward_return(ticks, 0, 720.0))

    def test_non_positive_entry_price_is_refused(self) -> None:
        ticks = [(0.0, 0.0, 0.0), (720.0, 110.0, 0.0)]
        self.assertIsNone(_forward_return(ticks, 0, 720.0))

    def test_a_loss_is_reported_as_a_loss(self) -> None:
        ticks = [(0.0, 100.0, 0.0), (720.0, 90.0, 0.0)]
        self.assertAlmostEqual(_forward_return(ticks, 0, 720.0), -0.10, places=9)


class PermutationTest(unittest.TestCase):
    def test_selection_that_takes_only_winners_is_significant(self) -> None:
        baseline = [-0.02] * 90 + [0.05] * 10
        fired = [0.05] * 10          # picked every winner
        p = _permutation_p(fired, baseline, trials=2000, rng=random.Random(1))
        self.assertLess(p, 0.05, "picking every winner must not look like chance")

    def test_selection_drawn_from_the_baseline_looks_like_chance(self) -> None:
        rng = random.Random(7)
        baseline = [rng.gauss(0.0, 0.01) for _ in range(1000)]
        fired = rng.sample(baseline, 40)   # a genuinely random entry set
        p = _permutation_p(fired, baseline, trials=4000, rng=random.Random(3))
        self.assertGreater(
            p, 0.05,
            "a random entry set must not be reported as skill -- this is the "
            "false positive that would fund a lane with no edge",
        )

    def test_selection_worse_than_random_reports_a_high_p(self) -> None:
        """The BSTONK case: fired into the worst moments of a rising symbol."""
        baseline = [0.01] * 90 + [-0.03] * 10
        fired = [-0.03] * 10
        p = _permutation_p(fired, baseline, trials=2000, rng=random.Random(11))
        self.assertGreater(p, 0.95)

    def test_no_fires_is_not_a_verdict(self) -> None:
        p = _permutation_p([], [0.01, 0.02], trials=100, rng=random.Random(0))
        self.assertNotEqual(p, p, "an empty fire set must return nan, not a score")

    def test_baseline_smaller_than_the_fire_set_is_not_a_verdict(self) -> None:
        p = _permutation_p([0.01, 0.02, 0.03], [0.01], trials=100, rng=random.Random(0))
        self.assertNotEqual(p, p)


if __name__ == "__main__":
    unittest.main()
