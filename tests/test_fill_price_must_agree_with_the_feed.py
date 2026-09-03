"""A fill price twelve orders of magnitude out is not slippage.

Measured 2026-09-03, from the live book and confirmed against base:

    CBETH-USDC  entry_price 2.739721277650459e-09   feed 2731.12   ratio 1e-12
    tx 0x076978740803789cd40564cb150753bc075a5f6600d8422add58a4720822b82b

The receipt for that transaction says the wallet sent 750000 raw USDC and
received 273750474589586 raw cbETH, and USDC's own contract (eth_call
0x313ce567 on mainnet.base.org) says 6 decimals -- so the trade really cost
$0.75 and the real price was $2739.72. The parser read USDC with 18 decimals
and booked the spend as 7.5e-13.

``tests/test_fill_decimals_are_never_guessed.py`` closes the specific hole:
USDC on base now comes from the table and an unreadable token refuses instead
of answering 18. This file closes the general one. The table is a list of
tokens, and the next corruption will be a token that is not on it. We already
know what the asset costs -- a fill claiming otherwise by orders of magnitude
is not reporting a trade, whatever the reason.

Why it is worth a guard of its own: the damage lands after the money has
moved. With a cost basis of 7.5e-13 the exit computes

    gross_profit = quote_received - cost_portion  ~=  0.75 - 7.5e-13

on a notional of 1e-9 -- a ~10^12 return, booked into live P/L and into the
ledger that decides graduation. Four strategies have already been purged from
this repo for fabricated records; this one would have been written by the live
lane itself, out of a real on-chain transaction.

The threshold is calibrated rather than guessed -- see
``TheThresholdIsCalibratedTest``.
"""

from __future__ import annotations

import unittest

from trading.bot import TradingBot


#: The corrupted trade, exactly as the book recorded it.
CBETH_BOOKED_PRICE = 2.739721277650459e-09
CBETH_FEED_PRICE = 2731.12
#: From the receipt: 750000 raw USDC / 273750474589586 raw cbETH at 6/18.
CBETH_USD_SPENT = 0.75
CBETH_QTY = 0.000273750474589586

#: The AERO-USDC entry one minute later, which was correct and must stay
#: allowed: 0.75 USDC -> 1.5391564987245243 AERO against a feed of 0.48747.
AERO_BOOKED_PRICE = 0.48727988389842986
AERO_FEED_PRICE = 0.48747


class TheCorruptedTradeIsCaughtTest(unittest.TestCase):
    def test_the_booked_cbeth_price_disagrees_with_the_feed(self):
        self.assertTrue(
            TradingBot._fill_price_disagrees_with_feed(
                CBETH_BOOKED_PRICE, CBETH_FEED_PRICE
            )
        )

    def test_the_price_the_receipt_actually_supports_is_accepted(self):
        """The same trade, parsed with the right decimals, must go through."""
        real_price = CBETH_USD_SPENT / CBETH_QTY
        self.assertAlmostEqual(real_price, 2739.72, places=1)
        self.assertFalse(
            TradingBot._fill_price_disagrees_with_feed(real_price, CBETH_FEED_PRICE)
        )

    def test_the_error_is_the_decimals_difference(self):
        self.assertAlmostEqual(
            CBETH_BOOKED_PRICE * 10 ** (18 - 6),
            CBETH_USD_SPENT / CBETH_QTY,
            places=6,
        )


class RealTradesAreNotRefusedTest(unittest.TestCase):
    """The availability half. A guard that blocks real fills stops the lane."""

    def test_the_good_aero_entry_from_the_same_minute_passes(self):
        self.assertFalse(
            TradingBot._fill_price_disagrees_with_feed(
                AERO_BOOKED_PRICE, AERO_FEED_PRICE
            )
        )

    def test_ordinary_slippage_passes(self):
        for bps in (0, 25, 75, 300, 900):
            with self.subTest(bps=bps):
                self.assertFalse(
                    TradingBot._fill_price_disagrees_with_feed(
                        100.0 * (1 + bps / 10_000.0), 100.0
                    )
                )

    def test_a_position_whose_price_genuinely_doubled_passes(self):
        """UNI-USDC, open since 2026-08-17, sat at a 0.47 ratio legitimately."""
        self.assertFalse(TradingBot._fill_price_disagrees_with_feed(2.859, 6.0825))
        self.assertFalse(TradingBot._fill_price_disagrees_with_feed(6.0825, 2.859))


class TheThresholdIsCalibratedTest(unittest.TestCase):
    """Both sides of 10x, so the number cannot drift without a test failing."""

    def test_just_inside_ten_x_passes_both_directions(self):
        self.assertFalse(TradingBot._fill_price_disagrees_with_feed(9.9, 1.0))
        self.assertFalse(TradingBot._fill_price_disagrees_with_feed(1.0, 9.9))

    def test_beyond_ten_x_is_refused_both_directions(self):
        self.assertTrue(TradingBot._fill_price_disagrees_with_feed(10.1, 1.0))
        self.assertTrue(TradingBot._fill_price_disagrees_with_feed(1.0, 10.1))

    def test_the_factor_is_the_one_the_docstring_calibrates(self):
        self.assertEqual(TradingBot.FILL_PRICE_SANITY_FACTOR, 10.0)


class UnusableNumbersAreRefusedTest(unittest.TestCase):
    """RANGE: what a price can be besides wrong."""

    def test_zero_and_negative_implied_prices_are_refused(self):
        for bad in (0.0, -1.0, -1e-12):
            with self.subTest(bad=bad):
                self.assertTrue(TradingBot._fill_price_disagrees_with_feed(bad, 100.0))

    def test_nan_and_inf_implied_prices_are_refused(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                self.assertTrue(TradingBot._fill_price_disagrees_with_feed(bad, 100.0))

    def test_a_non_numeric_price_is_refused_rather_than_raising(self):
        self.assertTrue(TradingBot._fill_price_disagrees_with_feed("nonsense", 100.0))
        self.assertTrue(TradingBot._fill_price_disagrees_with_feed(None, 100.0))


class NoFeedIsNotADisagreementTest(unittest.TestCase):
    """A settled trade must not go unrecorded because the feed was dark.

    This repo has already stranded real money by refusing to book: 1.5462814520
    AERO and 19.4882431740 BASECAT were held on-chain on 2026-09-03 with no
    position of any kind, after two settled swaps recorded no fill.
    """

    def test_a_missing_feed_price_lets_the_fill_through(self):
        for feed in (0.0, -1.0, float("nan"), float("inf"), None, "n/a"):
            with self.subTest(feed=feed):
                self.assertFalse(
                    TradingBot._fill_price_disagrees_with_feed(2739.72, feed)
                )

    def test_an_unreadable_feed_does_not_refuse_but_an_unreadable_fill_does(self):
        """The asymmetry is the point: absence blocks nothing, garbage blocks."""
        self.assertFalse(TradingBot._fill_price_disagrees_with_feed(2739.72, None))
        self.assertTrue(TradingBot._fill_price_disagrees_with_feed(None, 2739.72))


class BothLegsConsultTheGuardTest(unittest.TestCase):
    """A guard on the entry alone still lets the exit book a fake profit."""

    def _source(self):
        import inspect

        return inspect.getsource(TradingBot)

    def test_the_entry_checks_the_receipt_before_taking_its_numbers(self):
        source = self._source()
        self.assertIn("receipt_price_insane = self._fill_price_disagrees_with_feed", source)
        self.assertIn("and not receipt_price_insane", source)

    def test_the_exit_checks_the_receipt_before_taking_its_numbers(self):
        source = self._source()
        self.assertIn("exit_price_insane = self._fill_price_disagrees_with_feed", source)
        self.assertIn("and not exit_price_insane", source)

    def test_both_legs_have_a_backstop_after_the_fallback(self):
        """The wallet delta can be wrong the same way the receipt was."""
        source = self._source()
        self.assertIn(
            "if self._fill_price_disagrees_with_feed(executed_entry_price, price):",
            source,
        )
        self.assertIn(
            "if self._fill_price_disagrees_with_feed(exit_price_effective, price):",
            source,
        )

    def test_an_estimated_basis_is_flagged_where_it_is_booked(self):
        source = self._source()
        self.assertIn('"basis_estimated": basis_estimated', source)
        self.assertIn('"proceeds_estimated": exit_proceeds_estimated', source)


if __name__ == "__main__":
    unittest.main()
