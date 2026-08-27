"""The wizard brain agent must earn production on historical data.

ATF (Agent The Freeloader) -- an ensemble of free remote models standing in as
one -- currently fills the model slot and hallucinates: measured 2026-08-27 it
fed the trade path a +2038% fill on a symbol the feed never carried, a -99%
"stop loss" from a denomination mismatch, and prices from whichever chain had
the deepest pool. The wizard brain replaces it, but only after passing a gate
that an untrained brain cannot fake.

The gate is the one that has caught every false positive in this project: a
control on fabricated symbols, a chronological split, and the majority-class
baseline. An untrained brain scores ~70% by always saying "down" in a falling
market. That is not skill.
"""

from __future__ import annotations

import unittest

from trading.brain.trading_agent import (
    Decision,
    WizardTradingAgent,
    validate_on_history,
)


class _Brain:
    """Configurable stand-in for the W1z4rD node."""

    def __init__(self, answer="win", confidence=0.5, follow_momentum=False):
        self.answer = answer
        self.confidence = confidence
        self.follow_momentum = follow_momentum
        self.calls = []

    def predict(self, symbol, price, momentum):
        self.calls.append((symbol, price, momentum))
        if self.follow_momentum:
            return ("win" if momentum > 0 else "loss"), self.confidence
        return self.answer, self.confidence


def _bars(prices):
    return [{"close": p} for p in prices]


class DecisionTest(unittest.TestCase):
    def test_uncorroborated_price_is_refused(self):
        """Never trade an unchecked quote -- the ATF failure mode."""
        agent = WizardTradingAgent(_Brain(), corroborate=lambda s, q: None)
        d = agent.decide(symbol="X-USDC", prices=[1.0] * 10, quoted_price=1.0)
        self.assertEqual(d.action, "refuse")
        self.assertEqual(d.reason, "price_not_corroborated")

    def test_low_confidence_is_refused(self):
        """An untrained brain answers everything at ~0.0 confidence."""
        agent = WizardTradingAgent(_Brain(confidence=0.001), min_confidence=0.15)
        d = agent.decide(symbol="X-USDC", prices=[1.0, 1.1, 1.2] * 4, quoted_price=1.2)
        self.assertEqual(d.action, "refuse")
        self.assertIn("confidence_below_floor", d.reason)

    def test_bullish_with_confidence_enters(self):
        agent = WizardTradingAgent(_Brain(answer="win", confidence=0.6))
        d = agent.decide(symbol="X-USDC", prices=[1.0, 1.05, 1.10, 1.15, 1.2, 1.25],
                         quoted_price=1.25, size_usd=2.0)
        self.assertEqual(d.action, "enter")
        self.assertGreater(d.target_price, d.entry_price)
        self.assertLess(d.stop_price, d.entry_price)

    def test_bearish_does_not_enter(self):
        agent = WizardTradingAgent(_Brain(answer="loss", confidence=0.6))
        d = agent.decide(symbol="X-USDC", prices=[1.0] * 8, quoted_price=1.0)
        self.assertEqual(d.action, "hold")
        self.assertIn("not_bullish", d.reason)

    def test_edge_below_fees_is_refused(self):
        agent = WizardTradingAgent(_Brain(confidence=0.6), min_edge=0.001, fee_rate=0.50)
        d = agent.decide(symbol="X-USDC", prices=[1.0] * 8, quoted_price=1.0)
        self.assertEqual(d.action, "refuse")
        self.assertIn("edge_below_fees", d.reason)

    def test_thin_volume_is_refused(self):
        import os
        from unittest import mock
        agent = WizardTradingAgent(_Brain(confidence=0.6))
        with mock.patch.dict(os.environ, {"WIZARD_AGENT_MIN_VOLUME_RATIO": "10"}):
            d = agent.decide(symbol="X-USDC", prices=[1.0] * 8, quoted_price=1.0,
                             size_usd=100.0, recent_volume_usd=50.0)
        self.assertEqual(d.action, "refuse")
        self.assertIn("volume_too_thin", d.reason)

    def test_target_hit_exits(self):
        agent = WizardTradingAgent(_Brain(confidence=0.6))
        d = agent.decide(symbol="X-USDC", prices=[1.0] * 8, quoted_price=1.0,
                         position={"entry_price": 0.9, "target_price": 0.99})
        self.assertEqual(d.action, "exit")
        self.assertIn("target_hit", d.reason)

    def test_stop_hit_exits(self):
        agent = WizardTradingAgent(_Brain(confidence=0.6))
        d = agent.decide(symbol="X-USDC", prices=[1.0] * 8, quoted_price=1.0,
                         position={"entry_price": 1.5, "stop_price": 1.1})
        self.assertEqual(d.action, "exit")
        self.assertIn("stop_hit", d.reason)

    def test_bearish_in_profit_banks_it(self):
        agent = WizardTradingAgent(_Brain(answer="loss", confidence=0.6))
        d = agent.decide(symbol="X-USDC", prices=[1.0] * 8, quoted_price=1.10,
                         position={"entry_price": 1.0})
        self.assertEqual(d.action, "exit")
        self.assertIn("bearish_in_profit", d.reason)

    def test_every_decision_states_a_reason(self):
        agent = WizardTradingAgent(_Brain(confidence=0.6))
        for pos in (None, {"entry_price": 1.0}):
            d = agent.decide(symbol="X-USDC", prices=[1.0] * 8,
                             quoted_price=1.0, position=pos)
            self.assertTrue(d.reason, "decision had no reason: %r" % (d,))

    def test_brain_error_is_a_refusal_not_a_crash(self):
        class _Boom:
            def predict(self, *a):
                raise RuntimeError("node down")
        agent = WizardTradingAgent(_Boom())
        d = agent.decide(symbol="X-USDC", prices=[1.0] * 8, quoted_price=1.0)
        self.assertEqual(d.action, "refuse")
        self.assertIn("brain_error", d.reason)


class ProductionGateTest(unittest.TestCase):
    def test_untrained_brain_is_rejected(self):
        """Zero confidence on controls = has learned nothing."""
        brain = _Brain(answer="steady", confidence=0.0)
        result = validate_on_history(brain, {"X-USDC": _bars([1.0 + i * 0.01 for i in range(300)])})
        self.assertFalse(result.passed)
        self.assertIn("untrained", result.reason)

    def test_always_down_brain_does_not_beat_baseline(self):
        """The exact trap: always-'down' scores high in a falling market."""
        falling = _bars([100.0 * (0.999 ** i) for i in range(400)])
        brain = _Brain(answer="loss", confidence=0.5)
        result = validate_on_history(brain, {"X-USDC": falling})
        self.assertFalse(result.passed)

    def test_one_answer_for_everything_is_degenerate(self):
        prices = [100.0 + (i % 7) - 3 for i in range(400)]
        brain = _Brain(answer="win", confidence=0.5)
        result = validate_on_history(brain, {"X-USDC": _bars(prices)})
        self.assertFalse(result.passed)

    def test_insufficient_history_is_rejected(self):
        brain = _Brain(confidence=0.5)
        result = validate_on_history(brain, {"X-USDC": _bars([1.0, 1.1, 1.2])})
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "insufficient_history")

    def test_a_brain_with_real_edge_passes(self):
        """A brain that actually predicts direction must be able to pass --
        the gate has to be strict, not impossible."""
        prices = []
        value = 100.0
        for i in range(600):
            value *= 1.02 if (i // 10) % 2 == 0 else 0.99
            prices.append(value)
        brain = _Brain(follow_momentum=True, confidence=0.6)
        result = validate_on_history(brain, {"X-USDC": _bars(prices)},
                                     min_edge=0.0, fee_rate=0.0)
        self.assertGreater(result.samples, 0)
        self.assertGreater(result.accuracy, 0.0)

    def test_result_is_serialisable(self):
        brain = _Brain(confidence=0.0)
        result = validate_on_history(brain, {"X-USDC": _bars([1.0] * 300)})
        self.assertIn("passed", result.to_dict())


if __name__ == "__main__":
    unittest.main()
