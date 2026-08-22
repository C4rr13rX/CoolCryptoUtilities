"""
Calibration for the crypto brain's regime confidence gate.

Written in the style the senior engineer brain's recall gate was calibrated:
measure the whole set of cases and pick a floor that separates them, rather
than asserting a threshold from a handful of hand-picked examples.

The bug this pins down
----------------------
Confidence used to be keyword *density*, which measures verbosity, not
conviction:

    "bull"                                        -> 1.000   (one word!)
    a hedged reply naming both sides              -> 0.615
    a thorough 200-word analysis                  -> 0.120

So the tersest, least reasoned reply carried the most weight. That is the
same class of error as the senior brain's "margin does not measure what it
looks like" -- a gate whose number does not mean what its name claims.

Confidence is now one-sidedness x evidence, and both have to hold.
"""

from __future__ import annotations

import unittest

from trading.wizard_trainer import (
    REGIME_CONFIDENCE_FLOOR,
    _parse_regime_text,
)

# (reply, description, should_pass_gate)
CASES = [
    # -- genuine readings: agreeing keywords, one-sided ------------------
    ("bullish breakout above resistance, strong accumulation",
     "on-topic bullish", True),
    ("bearish, breakdown, weak, distribution, falling",
     "on-topic bearish", True),
    ("Strong bullish momentum: breakout confirmed with heavy accumulation "
     "and rising support as buyers absorb supply",
     "thorough bullish", True),
    ("future_direction=down with distribution and a confirmed breakdown "
     "below support",
     "structured bearish", True),

    # -- not readings: must not carry conviction -------------------------
    ("bull", "single token", False),
    ("bullish", "single word (also: must not double-count bull+bullish)",
     False),
    ("bearish", "single word bear", False),
    ("The capital of France is Paris.", "unrelated", False),
    ("I cannot answer that question.", "refusal", False),
    ("", "empty", False),
    ("The market may go up or down; support and resistance and both "
     "accumulation and distribution are visible so it is unclear",
     "hedged, both sides named", False),
]


class RegimeConfidenceCalibration(unittest.TestCase):
    def test_gate_separates_readings_from_noise(self):
        """Every case falls on the correct side of the floor."""
        failures = []
        for text, label, should_pass in CASES:
            _direction, confidence = _parse_regime_text(text)
            passes = confidence >= REGIME_CONFIDENCE_FLOOR
            if passes != should_pass:
                failures.append(
                    f"{label!r}: confidence {confidence:.3f} "
                    f"{'passed' if passes else 'rejected'}, expected "
                    f"{'pass' if should_pass else 'reject'}"
                )
        self.assertFalse(failures, "\n".join(failures))

    def test_floor_sits_between_the_populations(self):
        """
        The floor must have headroom on both sides.

        A threshold that only just separates the measured cases will not
        survive a phrasing it has not seen. This is the check the senior
        brain's calibration added after 0.35 admitted an untrained prompt.
        """
        passing = [
            _parse_regime_text(t)[1] for t, _l, ok in CASES if ok
        ]
        rejected = [
            _parse_regime_text(t)[1] for t, _l, ok in CASES if not ok
        ]
        self.assertLess(max(rejected), REGIME_CONFIDENCE_FLOOR,
                        f"a rejected case scored {max(rejected):.3f}")
        self.assertGreaterEqual(min(passing), REGIME_CONFIDENCE_FLOOR,
                                f"a valid reading scored {min(passing):.3f}")

    def test_a_single_keyword_carries_no_conviction(self):
        """The headline regression: one word must not mean certainty."""
        direction, confidence = _parse_regime_text("bull")
        self.assertEqual(confidence, 0.0)
        # The lean is still reported -- it is the conviction that is withheld.
        self.assertEqual(direction, 1.0)

    def test_bull_and_bullish_count_once(self):
        """'bullish' contains 'bull'; one word must not be two votes."""
        _d1, one_word = _parse_regime_text("bullish")
        _d2, two_words = _parse_regime_text("bullish breakout")
        self.assertEqual(one_word, 0.0, "one word scored as corroborated")
        self.assertGreater(two_words, one_word)

    def test_verbosity_does_not_reduce_confidence(self):
        """
        A longer reply saying the same thing must not score lower.

        Under the old density formula it did -- padding an answer with
        explanation actively lowered its confidence, which is why a
        one-word reply beat a full analysis.
        """
        short_text = "bullish breakout accumulation"
        long_text = (
            "Looking at the higher timeframe I would say this is bullish: "
            "there is a clear breakout and steady accumulation, and while "
            "nothing is ever certain the structure has held for some time "
            "and continues to look constructive to me overall right now"
        )
        _d1, short_conf = _parse_regime_text(short_text)
        _d2, long_conf = _parse_regime_text(long_text)
        self.assertGreaterEqual(
            long_conf, short_conf * 0.9,
            f"verbosity penalised: {short_conf:.3f} -> {long_conf:.3f}",
        )

    def test_direction_is_reported_independently_of_confidence(self):
        """
        Direction and confidence must not be aliases of one quantity.

        The senior brain found `margin` was literally equal to `score` on its
        recall route, so gating on both double-counted one signal. Here they
        are genuinely different: a dead-heat reply has a direction (0.5) and
        no confidence, and a one-sided reply has both.
        """
        hedged_dir, hedged_conf = _parse_regime_text(
            "support resistance accumulation distribution")
        oneside_dir, oneside_conf = _parse_regime_text(
            "support accumulation breakout")
        self.assertAlmostEqual(hedged_dir, 0.5, places=2)
        self.assertEqual(hedged_conf, 0.0)
        self.assertGreater(oneside_dir, 0.9)
        self.assertGreater(oneside_conf, 0.0)


if __name__ == "__main__":
    unittest.main()
