"""A price that never moved is not a flat market, and it must not book a trade.

Pins ``services.feed_freeze`` against the three windows measured on
``storage/trading_cache.db`` for item e44d880d (pass 690). Each case below is a
real ``market_stream`` window that a real ghost round trip opened and closed
inside, booking ``gross_profit`` of exactly 0.0 and a net loss of pure cost.

The prices and cadences are transcribed from the database, not invented:

    PUMP-USDC    09-10 20:30 -> 09-11 12:45 UTC  153 rows,  2 distinct,
                 125 of them exactly 1.022046762e-07, cadence ~6.4 min
    JITOSOL-USDC 09-07 00:30 -> 09-07 02:50 UTC   65 rows,  1 distinct,
                 100.937663393985 throughout
    DRB-USDC     09-16 03:15 -> 09-16 04:05 UTC   10 rows,  1 distinct,
                 0.0001663 throughout

The three DRB round trips are tied out by outcome_id in the item:
``2:DRB-USDC:878e2d6474d64d1d83a6ef35c65d9ec4:exit:1`` and two siblings, all
entry == exit == 0.0001663, gross 0.0, net approx -0.02534, reason timed-exit.
"""

import unittest

from services.feed_freeze import window_is_frozen


class FrozenFeedIsNotAFlatMarket(unittest.TestCase):
    def test_jitosol_sixty_five_identical_prints_over_two_hours_is_frozen(self):
        # 65 prints of one value over 2.3h. The span clause must catch this:
        # max == min, well past both floors.
        px = 100.937663393985
        prices = [px] * 65
        stamps = [1788741617.0 + 130.0 * i for i in range(65)]
        v = window_is_frozen(prices, stamps)
        self.assertTrue(v.frozen, v.diagnostics)
        self.assertEqual(v.reason, "feed_frozen")

    def test_drb_ten_identical_prints_over_fifty_minutes_is_frozen(self):
        # The window the three tied-out DRB round trips opened and closed in.
        prices = [0.0001663] * 10
        stamps = [1789529142.0 + 300.0 * i for i in range(10)]
        v = window_is_frozen(prices, stamps)
        self.assertTrue(v.frozen, v.diagnostics)
        self.assertEqual(v.reason, "feed_frozen")

    def test_pump_two_distinct_prices_across_one_hundred_fifty_three_prints(self):
        # max != min here, so the span clause CANNOT fire -- the window holds
        # 1.022046762e-07 and one other value. This is exactly the case the
        # distinct clause exists for, and it is why lifting only the span
        # clause would have missed four of the fifteen booked rows.
        prices = [1.022046762e-07] * 125 + [2.64241291858668e-09] * 28
        stamps = [1789073119.0 + 384.0 * i for i in range(153)]
        v = window_is_frozen(prices, stamps)
        self.assertNotEqual(max(prices), min(prices))
        self.assertTrue(v.frozen, v.diagnostics)
        self.assertEqual(v.reason, "feed_frozen")

    def test_a_moving_feed_is_not_refused(self):
        # The complement. A guard that cannot pass a live window is not a
        # guard, it is an outage.
        prices = [2500.0 + i * 0.37 for i in range(40)]
        stamps = [1789529142.0 + 60.0 * i for i in range(40)]
        v = window_is_frozen(prices, stamps)
        self.assertFalse(v.frozen, v.diagnostics)
        self.assertEqual(v.reason, "ok")

    def test_a_calm_minute_is_not_called_frozen(self):
        # Four identical prints 10 seconds apart is a quiet on-chain pair
        # between trades, not a dead feed. Below FROZEN_MIN_SPAN_SEC the span
        # clause must abstain rather than refuse -- refusing here is how a
        # guard bans a working strategy.
        prices = [1.5] * 4
        stamps = [1789529142.0 + 10.0 * i for i in range(4)]
        v = window_is_frozen(prices, stamps)
        self.assertFalse(v.frozen, v.diagnostics)
        self.assertEqual(v.reason, "undecidable")

    def test_no_price_is_reported_as_its_own_fault(self):
        # An absent price is a different defect with a different remedy.
        # Calling it "frozen" would hide it.
        v = window_is_frozen([], [])
        self.assertFalse(v.frozen)
        self.assertEqual(v.reason, "no_price")

    def test_without_timestamps_only_the_distinct_clause_runs(self):
        # A caller with no stamps still gets the money_button floor.
        v = window_is_frozen([7.0] * 12)
        self.assertTrue(v.frozen, v.diagnostics)
        self.assertEqual(v.reason, "feed_frozen")
        self.assertNotIn("span_sec", v.diagnostics)


if __name__ == "__main__":
    unittest.main()
