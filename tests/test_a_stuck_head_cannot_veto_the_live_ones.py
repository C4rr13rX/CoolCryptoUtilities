"""A conjunct reading NOTHING must not veto the three conjuncts reading something.

``trading/scheduler.py``'s entry test is a four-way conjunction
(:1054-1071): ``expected > min_profit``, ``direction_prob >= floor``,
``confidence >= floor``, ``net_margin >= floor``. It is the single largest
refusal in the system and the one that decides the GHOST ENTRY RATE, which is
what the graduation bar's "20 tradeable ghost trades" is made of.

WHAT IT WAS DOING. ``confidence`` is bound at :778 to
``pred_summary["exit_conf"]``. ``0.5`` is the value ``trading/bot.py`` writes
into that key when the head cannot be read (:6038, :6073), and what
``_neutral_pred_summary`` returns when TensorFlow is absent. ``bot.py:8088``
already classifies ``abs(exit_conf - 0.5) < 0.02`` as ``model_neutral``. So a
0.5 arriving at the entry test is indistinguishable BY CONSTRUCTION from
"there was no reading" -- and the test read it as a measured 0.5 and refused it
against a 0.6 floor.

MEASURED 2026-09-23 over the 1805 ``entry-refused-no-candidates`` rows of the
last 14 days, from the ``values`` the scheduler already stores. Confidence
histogram to 2dp: ``{0.49: 3, 0.50: 1778, 0.51: 23, 0.62: 1}`` -- 1804 of 1805
inside the neutral band. ``confidence_vs_floor`` bound on 1799 of 1800 and was
the ONLY failing conjunct on 134; 134 of those 134 are inside the band and
ZERO are "informative but below 0.6". On the same rows the siblings have real
range -- direction_prob 0.0097..0.9892 across 1668 distinct values, net_margin
-1.7401..+2.9515, expected -0.0625..+5.4669.

DIRECTION, STATED: THE GATE OPENS, and the measured upper bound is those 134
of 1805 refusals (7.4%) -- only ticks where all three informative conjuncts
had ALREADY passed. NO THRESHOLD MOVED: ``SCHEDULER_MIN_CONFIDENCE`` is
untouched at 0.6 and an informative confidence below it still refuses.

These tests exercise ``_confidence_abstains`` directly rather than driving
``evaluate()``, which needs a live db, a quote asset, a price history and a
horizon forecast to reach. The predicate IS the decision -- its caller passes
the result straight into both the conjunction and the logged row.
"""

from __future__ import annotations

import unittest

from trading.scheduler import MODEL_NEUTRAL_BAND, _confidence_abstains


class AStuckHeadAbstains(unittest.TestCase):
    def test_the_value_production_actually_emits_abstains(self):
        """0.5000 against a live direction read. 1778 of 1805 rows are this."""
        self.assertTrue(_confidence_abstains(0.5, 0.82))

    def test_the_whole_observed_band_abstains(self):
        """Every confidence value seen in 14 days except the single 0.6164."""
        for confidence in (0.4911, 0.49, 0.5, 0.5013, 0.51, 0.5199):
            self.assertTrue(
                _confidence_abstains(confidence, 0.82),
                f"{confidence} is inside bot.py:8088's own no-information band",
            )


class AReadingThatSaysSomethingStillBinds(unittest.TestCase):
    """The floor is not lowered -- these are the cases that must still refuse."""

    def test_an_informative_confidence_below_the_floor_does_not_abstain(self):
        """0.55 is a real reading under a 0.6 floor: refuse it, as before.

        No row in 14 days took this path. The path exists so a head that
        starts emitting again lands on the floor rather than on the
        abstention.
        """
        self.assertFalse(_confidence_abstains(0.55, 0.82))

    def test_a_confident_reading_does_not_abstain(self):
        self.assertFalse(_confidence_abstains(0.6164, 0.82))
        self.assertFalse(_confidence_abstains(0.95, 0.82))

    def test_a_silent_model_still_refuses(self):
        """BOTH heads neutral is bot.py:8088's model_neutral -- the model
        declined to speak, and nothing may be entered on that."""
        self.assertFalse(_confidence_abstains(0.5, 0.5))
        self.assertFalse(_confidence_abstains(0.5, 0.505))

    def test_an_unreadable_value_never_abstains(self):
        """Abstention is a claim about the reading; no reading, no claim."""
        self.assertFalse(_confidence_abstains(float("nan"), 0.82))
        self.assertFalse(_confidence_abstains(0.5, float("nan")))
        self.assertFalse(_confidence_abstains(None, 0.82))
        self.assertFalse(_confidence_abstains("0.5", object()))


class TheBandMatchesTheOneTheBotAlreadyUses(unittest.TestCase):
    """Two copies of one rule that drifted is a defect shape this repo has
    had four times. bot.py:8088 inlines 0.02; this must equal it."""

    def test_the_band_is_the_bots_band(self):
        self.assertEqual(MODEL_NEUTRAL_BAND, 0.02)

    def test_the_bot_still_uses_that_number(self):
        import pathlib

        src = (
            pathlib.Path(__file__).resolve().parents[1] / "trading" / "bot.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "abs(direction_prob - 0.5) < 0.02 and abs(exit_conf_val - 0.5) < 0.02",
            src,
            "bot.py's model_neutral band moved; MODEL_NEUTRAL_BAND must move with it",
        )


class TheDecisionAndTheLoggedRowCannotDisagree(unittest.TestCase):
    """The conjunction and the `failed` list must read the same predicate.

    An earlier shape of this defect in the same file -- two env reads of one
    clock -- let the sweep offer a position the chain then refused.
    """

    def test_the_predicate_is_evaluated_once_and_used_twice(self):
        import pathlib

        src = (
            pathlib.Path(__file__).resolve().parents[1] / "trading" / "scheduler.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            src.count("confidence_abstains = _confidence_abstains("), 1,
            "the predicate must be computed once, not per consumer",
        )
        self.assertIn(
            'and not (name == "confidence_vs_floor" and confidence_abstains)', src
        )
        self.assertIn(
            "and (confidence_abstains or confidence >= min_confidence)", src
        )
        self.assertIn('"confidence_abstains": bool(confidence_abstains),', src)


if __name__ == "__main__":
    unittest.main()
