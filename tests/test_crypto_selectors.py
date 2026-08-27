"""Selection tools must beat picking at random, and prove it out-of-sample.

C0d3rV2's ``score_pair`` was measured on 131,200 real bars 2026-08-27:

    top decile     mean -0.00086/trade   win 46.3%
    all bars       mean -0.00023/trade   win 48.3%
    lift           -0.00063/trade

Its picks were WORSE than random. It weights momentum and buy pressure, which
mean-revert at this horizon, so it reliably buys the top of a move. These
replacements carry the same measurement so a successor cannot quietly repeat it.

A GA search over the feature weights found +0.00986 lift in-sample; on symbols
it had never seen the lift held (+0.00216) but the top picks still lost money
after fees. Beating the baseline and being profitable are two different claims,
and the gate requires both.
"""

from __future__ import annotations

import unittest

from trading.genome.selectors import (
    FEATURES,
    SELECTOR_GENE_SPACE,
    evaluate_selector,
    feature_vector,
    selector_fitness,
    selector_score,
)


def _bars(prices, buy=None, sell=None):
    out = []
    for i, p in enumerate(prices):
        out.append({
            "close": p, "open": p, "high": p * 1.01, "low": p * 0.99,
            "buy_volume": (buy[i] if buy else 1000.0),
            "sell_volume": (sell[i] if sell else 1000.0),
        })
    return out


class FeatureTest(unittest.TestCase):
    def test_reversion_is_positive_when_price_is_low(self):
        """Buying weakness must score positively -- the core inversion."""
        falling = _bars([100.0 - i for i in range(30)])
        self.assertGreater(feature_vector(falling)["reversion"], 0.0)

    def test_reversion_is_negative_when_price_is_high(self):
        rising = _bars([100.0 + i for i in range(30)])
        self.assertLess(feature_vector(rising)["reversion"], 0.0)

    def test_drawdown_is_positive_below_the_peak(self):
        shape = _bars([100.0] * 10 + [120.0] + [90.0] * 14)
        self.assertGreater(feature_vector(shape)["drawdown"], 0.0)

    def test_range_position_is_high_near_the_low(self):
        shape = _bars([100.0 + (10 if i < 15 else 0) for i in range(30)])
        self.assertGreater(feature_vector(shape)["range_position"], 0.5)

    def test_buy_pressure_reflects_the_split(self):
        heavy_buys = _bars([100.0] * 30, buy=[900.0] * 30, sell=[100.0] * 30)
        self.assertGreater(feature_vector(heavy_buys)["buy_pressure"], 0.5)

    def test_every_feature_is_finite_on_degenerate_input(self):
        flat = _bars([50.0] * 30)
        for name, value in feature_vector(flat).items():
            self.assertEqual(value, value, name)          # not NaN
            self.assertLess(abs(value), 1e9, name)

    def test_short_history_does_not_raise(self):
        for name in FEATURES:
            self.assertIsInstance(feature_vector(_bars([1.0, 2.0]))[name], float)


class ScoreTest(unittest.TestCase):
    def test_weights_can_invert_a_feature(self):
        """The GA must be able to discover that momentum should be SOLD."""
        rising = _bars([100.0 + i for i in range(30)])
        positive = selector_score(rising, {"w_momentum": 1.0})
        negative = selector_score(rising, {"w_momentum": -1.0})
        self.assertAlmostEqual(positive, -negative, places=9)

    def test_zero_weights_score_zero(self):
        bars = _bars([100.0 + i for i in range(30)])
        self.assertEqual(selector_score(bars, {}), 0.0)

    def test_gene_space_spans_negative(self):
        """Inversion has to be reachable by the search."""
        for _name, (lo, hi) in SELECTOR_GENE_SPACE.items():
            self.assertLess(lo, 0.0)
            self.assertGreater(hi, 0.0)


class EvaluationGateTest(unittest.TestCase):
    def test_no_lift_does_not_pass(self):
        """A selector no better than random is worth zero."""
        bars = {"X": _bars([100.0 + (i % 5) for i in range(400)])}
        result = evaluate_selector({"w_%s" % k: 0.0 for k in FEATURES}, bars)
        self.assertFalse(result.passed)

    def test_profitable_lift_is_required_not_just_lift(self):
        """Beating the baseline while losing money must NOT pass -- the exact
        out-of-sample result the searched selector produced."""
        from trading.genome.selectors import SelectorResult
        result = SelectorResult(lift=0.00216, top_mean=-0.00405, samples=2469, selected=246)
        # Mirrors evaluate_selector's ordering: lift first, then profitability.
        self.assertGreater(result.lift, 0.0)
        self.assertLess(result.top_mean, 0.0)
        self.assertFalse(result.passed)

    def test_insufficient_samples_does_not_pass(self):
        bars = {"X": _bars([100.0 + i for i in range(40)])}
        result = evaluate_selector({"w_reversion": 1.0}, bars)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "insufficient_samples")

    def test_fitness_is_zero_unless_the_gate_passes(self):
        bars = {"X": _bars([100.0 + (i % 5) for i in range(400)])}
        self.assertEqual(selector_fitness({"w_%s" % k: 0.0 for k in FEATURES}, bars), 0.0)

    def test_evaluation_scores_only_held_out_data(self):
        """A rule tuned on the data it is scored against always looks good."""
        bars = {"X": _bars([100.0 + i * 0.1 for i in range(600)])}
        result = evaluate_selector({"w_reversion": 1.0}, bars, split=0.7)
        # Only the tail is scored, so sample count is well under the full set.
        self.assertLess(result.samples, 600)

    def test_result_is_serialisable(self):
        bars = {"X": _bars([100.0 + (i % 7) for i in range(400)])}
        self.assertIn("lift", evaluate_selector({"w_reversion": 1.0}, bars).to_dict())


if __name__ == "__main__":
    unittest.main()
