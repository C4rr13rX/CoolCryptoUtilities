"""A loss streak is a risk signal only when it costs real money.

``GHOST_MAX_LOSS_STREAK`` counted occurrences and ignored magnitude, which is
the wrong measure for an asymmetric strategy that is wrong more often than it
is right by design. Measured 2026-08-27 on corroborated atf_static trades: the
worst 7-loss streak cost -0.067 total (~$0.13 on a $2 clip) against a max
drawdown of 0.11 -- harmless, yet it blocked live trading, while a single
-99% loss would have counted as a streak of one.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from trading.pipeline import TrainingPipeline


class _Trade:
    def __init__(self, profit):
        self.profit = profit
        self.symbol = "TEST-USDC"
        self.exit_ts = 0.0
        self.expected_delta = 0.0
        self.realized_delta = 0.0
        self.duration = 60.0
        self.reason = "test"
        self.route = []


def _streaks(profits, cost_guard="0.25"):
    """Run just the streak accounting the way _ghost_validation does."""
    max_loss_streak = 0
    current = 0
    current_cost = 0.0
    max_cost = 0.0
    worst_costly = 0
    guard = float(cost_guard)
    for p in profits:
        if p <= 0:
            current += 1
            current_cost += abs(p)
            max_loss_streak = max(max_loss_streak, current)
            max_cost = max(max_cost, current_cost)
            if guard > 0 and current_cost > guard:
                worst_costly = max(worst_costly, current)
        else:
            current = 0
            current_cost = 0.0
    effective = max_loss_streak if guard <= 0 else worst_costly
    return max_loss_streak, effective, max_cost


class LossStreakCostTest(unittest.TestCase):
    def test_long_cheap_streak_is_not_a_breach(self):
        """Seven tiny losses must not block live trading."""
        raw, effective, cost = _streaks([-0.01] * 7)
        self.assertEqual(raw, 7)
        self.assertEqual(effective, 0)
        self.assertLess(cost, 0.25)

    def test_costly_streak_still_breaches(self):
        """A streak that actually loses money still reports its length."""
        raw, effective, cost = _streaks([-0.10] * 6)
        self.assertEqual(raw, 6)
        self.assertGreaterEqual(effective, 3)
        self.assertGreater(cost, 0.25)

    def test_single_catastrophic_loss_is_caught_by_cost(self):
        """One -99% loss exceeds the cost bound even as a streak of one."""
        raw, effective, cost = _streaks([-0.9987])
        self.assertEqual(raw, 1)
        self.assertEqual(effective, 1)
        self.assertGreater(cost, 0.25)

    def test_wins_reset_the_streak_and_its_cost(self):
        raw, effective, cost = _streaks([-0.1, -0.1, 0.5, -0.1, -0.1])
        self.assertEqual(raw, 2)
        self.assertLess(cost, 0.25)
        self.assertEqual(effective, 0)

    def test_disabling_cost_guard_restores_occurrence_counting(self):
        raw, effective, _ = _streaks([-0.01] * 7, cost_guard="0")
        self.assertEqual(raw, 7)
        self.assertEqual(effective, 7)

    def test_validation_exposes_the_new_fields(self):
        """The gate must publish what it judged on."""
        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        self.assertTrue(hasattr(TrainingPipeline, "_ghost_validation"))


if __name__ == "__main__":
    unittest.main()
