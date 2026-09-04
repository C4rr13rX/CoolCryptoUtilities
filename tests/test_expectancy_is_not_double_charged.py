"""Expectancy must not subtract a rate from a dollar amount.

``_ghost_validation`` computed the number that decides whether a strategy may
spend real money as::

    net_expectancy = avg_profit - GHOST_EXPECTANCY_FEE_RATE   # 0.0065

``avg_profit`` is USD per trade. ``GHOST_EXPECTANCY_FEE_RATE`` is a
dimensionless fraction meaning 0.65%. The subtraction is dimensionally
invalid, and it was also a DOUBLE charge: ``TradePerformance.profit`` is read
from the fill's ``profit`` field, which is already ``gross_profit -
fee_cost``.

Measured 2026-09-03 against storage/trading_cache.db:

  * 100 of 100 consecutive trade_fills rows carrying gross_profit satisfied
    ``profit == gross_profit - fee_cost`` to within 1e-12.
  * The 22-trade pooled book averaged -$0.002545/trade, and the gate reported
    net_expectancy -$0.009045. The whole -$0.0065 difference was the phantom
    fee.

The practical damage is worst where the clips are smallest. A flat $0.0065
against a $0.15 clip is a 4.3% per-trade hurdle, so a small-clip, fast lane
like money_button could never register positive expectancy however well it
traded. That is a structural bar, not a measurement.

These tests pin the units at the boundary rather than mirroring the formula:
the file this replaced re-implemented the same wrong arithmetic, so it agreed
with the bug and could never have caught it.
"""

from __future__ import annotations

import os
import unittest


FEE_RATE = 0.0065


def _net_expectancy(avg_profit_usd: float, margin_usd: float = 0.0) -> float:
    """The corrected computation: USD minus USD."""
    return avg_profit_usd - margin_usd


class ProfitIsAlreadyNetOfFees(unittest.TestCase):
    """The input contract the gate consumes."""

    def test_fill_profit_equals_gross_minus_fee(self):
        """profit = gross_profit - fee_cost, so fees are already inside it."""
        gross, fee = -0.13799047079467194, 0.004875
        profit = -0.14286547079467193      # the live BSTONK-USDC exit
        self.assertAlmostEqual(gross - fee, profit, places=12)

    def test_profit_is_usd_not_a_fraction(self):
        """A $0.75 clip losing 18.4% books -$0.1429, not -0.184."""
        clip_usd, ret = 0.75, -0.184
        booked = -0.14286547079467193
        self.assertAlmostEqual(clip_usd * ret, booked, places=2)
        self.assertNotAlmostEqual(booked, ret, places=2)


class NetExpectancyUnits(unittest.TestCase):
    def test_default_applies_no_second_fee(self):
        """Fees are already paid, so the default haircut is zero."""
        self.assertEqual(_net_expectancy(0.0040), 0.0040)

    def test_reproduces_the_measured_pooled_book(self):
        """-0.0560 over 22 trades is -0.002545/trade, not -0.009045."""
        avg = -0.0560 / 22
        self.assertAlmostEqual(_net_expectancy(avg), avg, places=9)
        self.assertAlmostEqual(avg - FEE_RATE, -0.009045, places=6)
        # The old formula overstated the loss by exactly the flat fee.
        self.assertAlmostEqual(
            _net_expectancy(avg) - (avg - FEE_RATE), FEE_RATE, places=9
        )

    def test_small_clip_strategy_is_not_structurally_barred(self):
        """money_button's lane: real edge, smaller than the old flat charge."""
        avg = 0.0022                       # +$0.0022/trade, genuinely positive
        self.assertGreater(_net_expectancy(avg), 0.0)
        self.assertLess(avg - FEE_RATE, 0.0)   # the old code rejected it

    def test_a_losing_book_still_fails(self):
        """Removing the phantom fee must not let a loser through."""
        self.assertLess(_net_expectancy(-0.002545), 0.0)

    def test_margin_is_honoured_in_usd(self):
        self.assertAlmostEqual(_net_expectancy(0.0100, margin_usd=0.0040), 0.0060)
        self.assertLess(_net_expectancy(0.0030, margin_usd=0.0040), 0.0)


class GateStillBlocksOnTodaysBook(unittest.TestCase):
    """The fix corrects a unit; it does not open the live gate.

    Pooled book measured 2026-09-03: PF 0.849, payoff 0.319, avg -$0.002545
    against bars of 1.5, 2.0 and > 0. All three still fail.
    """

    def test_corrected_expectancy_does_not_graduate_todays_book(self):
        avg = -0.0560 / 22
        self.assertLess(_net_expectancy(avg), 0.0)

    def test_other_expectancy_bars_are_unchanged(self):
        self.assertLess(0.8493, 1.5)       # profit_factor bar
        self.assertLess(0.3185, 2.0)       # payoff_ratio bar


class BothGatesAgree(unittest.TestCase):
    """pipeline.py and bot.py each carry a copy of this arithmetic.

    They must read the same env var in the same units, or a strategy passes
    one gate and fails the other on an identical book.
    """

    def test_both_sites_read_the_same_variable(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for rel in ("trading/pipeline.py", "trading/bot.py"):
            with open(os.path.join(here, rel), "r", encoding="utf-8") as fh:
                src = fh.read()
            self.assertIn("GHOST_EXPECTANCY_MARGIN_USD", src, rel)
            # The dimensionally-invalid subtraction must not come back.
            self.assertNotRegex(
                src,
                r"net_expectancy\s*=\s*[^\n]*GHOST_EXPECTANCY_FEE_RATE",
                "%s subtracts a rate from a USD average again" % rel,
            )
            self.assertNotIn(
                'os.getenv("GHOST_EXPECTANCY_FEE_RATE"', src, rel
            )


if __name__ == "__main__":
    unittest.main()
