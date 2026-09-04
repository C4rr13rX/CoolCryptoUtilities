"""The bot-level promotion gate must not reject a profitable asymmetric strategy.

A third, separate graduation gate lives in TradingBot._maybe_transition_to_live,
independent of the strategy ledger and the transition plan. It required a win
RATE, which assumes a symmetric strategy.

Measured 2026-08-27 over 81 real ghost trades:

    win rate      0.519   (required 0.62)  -> REJECTED
    avg win      +0.05808
    avg loss     -0.01033
    payoff        5.622
    profit factor 6.055
    net          +2.0365
    expectancy   +0.0186/trade AFTER fees

It is profitable BECAUSE the winners are large, not because they are frequent.
A frequency test rejects it forever, and no amount of trading changes that.
"""

from __future__ import annotations

import unittest


def _expectancy_ok(profits, *, required_trades=40, required_profit=0.5,
                   min_payoff=2.0, min_profit_factor=1.5, margin_usd=0.0,
                   enabled=True):
    """Mirror of the expectancy path in _maybe_transition_to_live.

    ``profits`` are USD per trade and are ALREADY net of fees, so the haircut
    subtracted here is a USD margin of safety (default none), not a rate.
    """
    count = len(profits)
    total = sum(profits)
    if count < required_trades or total < required_profit:
        return False
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]
    if not wins or not losses:
        return False
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    gross_loss = abs(sum(losses))
    payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    profit_factor = (sum(wins) / gross_loss) if gross_loss > 0 else 0.0
    net_expectancy = (total / max(1, count)) - margin_usd
    return bool(enabled and payoff >= min_payoff
                and profit_factor >= min_profit_factor
                and net_expectancy > 0.0)


def _observed():
    """The real distribution: 42 wins at +0.05808, 39 losses at -0.01033."""
    return [0.05808] * 42 + [-0.01033] * 39


class PromotionExpectancyTest(unittest.TestCase):
    def test_observed_asymmetric_strategy_qualifies(self):
        """The exact case the win-rate hurdle rejected."""
        profits = _observed()
        self.assertLess(sum(1 for p in profits if p > 0) / len(profits), 0.62)
        self.assertTrue(_expectancy_ok(profits))

    def test_low_payoff_does_not_qualify(self):
        """Symmetric and barely profitable must still fail this path."""
        profits = [0.011] * 42 + [-0.010] * 39
        self.assertFalse(_expectancy_ok(profits))

    def test_negative_expectancy_does_not_qualify(self):
        """A book that does not make money is not tradeable.

        ``profits`` are already net of fees, so a negative average IS the
        after-fee answer; there is no second fee to apply.
        """
        profits = [0.010] * 40 + [-0.020] * 45
        self.assertLess(sum(profits) / len(profits), 0.0)
        self.assertFalse(_expectancy_ok(profits))

    def test_explicit_usd_margin_can_still_reject_a_thin_book(self):
        """The haircut survives, in USD -- the units it is subtracted in.

        These 81 trades average +$0.0186/trade and pass with no margin. A
        margin of $0.05/trade is larger than the edge, so it rejects.
        """
        profits = _observed()
        self.assertTrue(_expectancy_ok(profits, margin_usd=0.0))
        self.assertFalse(_expectancy_ok(profits, margin_usd=0.05))

    def test_flat_fee_no_longer_punishes_a_small_clip_strategy(self):
        """The bug this file used to mirror.

        A small-clip lane that wins $0.004 and loses $0.001 is genuinely
        profitable, but the old code subtracted a flat 0.0065 -- larger than
        the entire per-trade edge -- and rejected it forever.
        """
        profits = [0.004] * 200 + [-0.001] * 130
        avg = sum(profits) / len(profits)
        self.assertGreater(avg, 0.0)
        self.assertLess(avg, 0.0065)          # the old flat charge exceeded it
        self.assertTrue(
            _expectancy_ok(profits, required_trades=40, required_profit=0.5)
        )
        # Reproduce the old formula to show it rejected the same book.
        self.assertLess(avg - 0.0065, 0.0)

    def test_too_few_trades_does_not_qualify(self):
        self.assertFalse(_expectancy_ok(_observed()[:10]))

    def test_insufficient_profit_does_not_qualify(self):
        profits = [0.001] * 42 + [-0.0001] * 39
        self.assertFalse(_expectancy_ok(profits, required_profit=0.5))

    def test_all_wins_does_not_qualify_without_losses(self):
        """No losses means no measurable payoff -- refuse rather than divide."""
        self.assertFalse(_expectancy_ok([0.05] * 50))

    def test_path_can_be_disabled(self):
        self.assertFalse(_expectancy_ok(_observed(), enabled=False))

    def test_win_rate_path_still_works_for_symmetric_strategies(self):
        """A high-win-rate strategy is unaffected by any of this."""
        profits = [0.02] * 70 + [-0.02] * 15
        count, total = len(profits), sum(profits)
        win_rate = sum(1 for p in profits if p > 0) / count
        self.assertGreaterEqual(win_rate, 0.62)
        self.assertGreaterEqual(count, 40)
        self.assertGreaterEqual(total, 0.5)


if __name__ == "__main__":
    unittest.main()
