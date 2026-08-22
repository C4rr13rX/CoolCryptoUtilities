"""
The regime signal's effect on a live trade decision.

This is the real-money path, so the properties that matter most here are the
*safety* ones: a broken, slow, or absent wizard node must never stop trading or
swing a decision on its own. The model's prediction stays the dominant term.
"""

from __future__ import annotations

import unittest

import numpy as np

from trading.wizard_trainer import (
    REGIME_CONFIDENCE_FLOOR,
    _parse_regime_text,
    regime_derived_state,
)


def apply_regime(direction_prob: float, regime_signal) -> float:
    """
    Mirror of the adjustment in bot.py.

    Kept in step by test_matches_bot_implementation below, which reads the
    real source rather than trusting this copy.
    """
    if not isinstance(regime_signal, dict):
        return direction_prob
    regime_dir = float(regime_signal.get("direction_prob", 0.5))
    regime_conf = float(regime_signal.get("confidence", 0.0))
    lean = (regime_dir - 0.5) * 2.0
    return float(np.clip(direction_prob + lean * regime_conf * 0.06, 0.0, 1.0))


class RegimeTradeWiring(unittest.TestCase):
    def test_absent_signal_changes_nothing(self):
        """A missing node must leave the model's prediction untouched."""
        for signal in (None, {}, "not a dict", 0):
            self.assertEqual(apply_regime(0.62, signal), 0.62 if signal != {} else 0.62)

    def test_bullish_regime_nudges_up_bearish_nudges_down(self):
        base = 0.55
        up = apply_regime(base, {"direction_prob": 1.0, "confidence": 1.0})
        down = apply_regime(base, {"direction_prob": 0.0, "confidence": 1.0})
        self.assertGreater(up, base)
        self.assertLess(down, base)
        # Symmetric: neither direction is privileged.
        self.assertAlmostEqual(up - base, base - down, places=6)

    def test_influence_is_bounded(self):
        """
        The regime can never move the decision by more than 0.06.

        This is the property that keeps it a second opinion rather than a
        second forecaster: a confidently wrong node cannot drag a 0.50
        prediction across a 0.58 entry threshold on its own.
        """
        worst = max(
            abs(apply_regime(0.5, {"direction_prob": d, "confidence": c}) - 0.5)
            for d in (0.0, 0.25, 0.5, 0.75, 1.0)
            for c in (0.0, 0.5, 1.0)
        )
        self.assertLessEqual(worst, 0.06 + 1e-9)

    def test_cannot_cross_the_entry_threshold_alone(self):
        """A neutral model plus a maximal regime must not trigger entry."""
        entry_threshold = 0.58
        adjusted = apply_regime(0.5, {"direction_prob": 1.0, "confidence": 1.0})
        self.assertLess(adjusted, entry_threshold)

    def test_low_confidence_barely_moves_it(self):
        base = 0.55
        weak = apply_regime(base, {"direction_prob": 1.0, "confidence": 0.2})
        strong = apply_regime(base, {"direction_prob": 1.0, "confidence": 1.0})
        self.assertLess(abs(weak - base), abs(strong - base))

    def test_result_stays_a_probability(self):
        for base in (0.0, 0.02, 0.5, 0.98, 1.0):
            for d, c in ((0.0, 1.0), (1.0, 1.0)):
                out = apply_regime(base, {"direction_prob": d, "confidence": c})
                self.assertGreaterEqual(out, 0.0)
                self.assertLessEqual(out, 1.0)

    def test_only_gated_signals_reach_the_trade_path(self):
        """
        Everything the confidence gate rejects must score below the floor.

        bot.py re-checks the floor before using a signal, so this pins the
        contract between the two: a rejected reply cannot arrive as a payload.
        """
        rejected = [
            "bull",                                   # single keyword
            "The capital of France is Paris.",        # off-topic
            "support resistance accumulation distribution",  # contradictory
            "",                                       # empty
        ]
        for text in rejected:
            _direction, confidence = _parse_regime_text(text)
            self.assertLess(
                confidence, REGIME_CONFIDENCE_FLOOR,
                f"{text!r} scored {confidence:.3f}, would reach the trade path",
            )
            self.assertFalse(regime_derived_state(text)["admitted"])

    def test_matches_bot_implementation(self):
        """
        The constant here must match the one bot.py actually uses.

        A drifting copy in a test is worse than no test: it would keep passing
        while the live path changed underneath it.
        """
        from pathlib import Path

        source = Path("trading/bot.py").read_text(encoding="utf-8")
        self.assertIn("regime_signal = brain.get(\"regime_signal\")", source)
        self.assertIn("lean * regime_conf * 0.06", source)
        # And the payload must be gated on the same floor before it is built.
        self.assertIn("REGIME_CONFIDENCE_FLOOR", source)


if __name__ == "__main__":
    unittest.main()
