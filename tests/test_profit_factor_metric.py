"""``aggregate_trade_metrics`` must actually report profit factor.

It never returned the key, so every caller using
``summary.get("profit_factor", 1.0)`` read the constant 1.0 regardless of the
trades. The ghost gate compares that value against MIN_GHOST_PROFIT_FACTOR to
decide whether a strategy may trade real money, so a strategy with a true
profit factor of 2.077 was judged on a placeholder -- and no threshold above
1.0 could ever have been met by anything.
"""

from __future__ import annotations

import unittest

from trading.metrics import MetricsCollector, TradePerformance


def _trade(profit: float) -> TradePerformance:
    return TradePerformance(
        symbol="TEST-USDC",
        entry_ts=0.0,
        exit_ts=60.0,
        profit=profit,
        expected_delta=0.0,
        realized_delta=0.0,
        reason="test",
        route=[],
    )


class ProfitFactorMetricTest(unittest.TestCase):
    def setUp(self):
        self.collector = MetricsCollector.__new__(MetricsCollector)

    def test_profit_factor_is_reported(self):
        summary = self.collector.aggregate_trade_metrics(
            [_trade(2.0), _trade(1.0), _trade(-1.0)]
        )
        self.assertIn("profit_factor", summary)
        self.assertAlmostEqual(summary["profit_factor"], 3.0)

    def test_profit_factor_below_one_when_losing(self):
        summary = self.collector.aggregate_trade_metrics(
            [_trade(1.0), _trade(-2.0), _trade(-2.0)]
        )
        self.assertAlmostEqual(summary["profit_factor"], 0.25)

    def test_asymmetric_strategy_scores_above_one(self):
        """A 40% win rate with 3x payoff is profitable and must read as such."""
        trades = [_trade(0.30)] * 4 + [_trade(-0.10)] * 6
        summary = self.collector.aggregate_trade_metrics(trades)
        self.assertLess(summary["win_rate"], 0.5)
        self.assertAlmostEqual(summary["profit_factor"], 2.0)

    def test_no_losses_is_finite(self):
        """inf does not survive JSON or guardrail comparison."""
        summary = self.collector.aggregate_trade_metrics([_trade(1.0), _trade(2.0)])
        self.assertTrue(summary["profit_factor"] > 1.0)
        self.assertNotEqual(summary["profit_factor"], float("inf"))

    def test_all_losses_is_zero(self):
        summary = self.collector.aggregate_trade_metrics([_trade(-1.0), _trade(-2.0)])
        self.assertEqual(summary["profit_factor"], 0.0)

    def test_empty_is_safe(self):
        summary = self.collector.aggregate_trade_metrics([])
        self.assertEqual(summary.get("profit_factor", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
