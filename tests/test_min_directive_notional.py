"""A trade too small to clear its own costs must be resized, not skipped.

Observed 2026-08-27, the last thing blocking live trading: atf_static emitted a
directive for 0.3946 units of BASECAT-USDC at $0.02776 = $0.011 notional, 32x
under the $0.35 the transition plan had approved. At a 5% target that nets
$0.00048 against the $0.02 SMALL_PROFIT_FLOOR, so micro_profit correctly
refused it and every live entry was blocked as
"micro-profit-blocked:net_profit_below_dollar_floor".

The guard was right; the size was wrong. And $0.35 itself is too small -- it
nets $0.0152, still under the floor. The minimum viable notional at a 5% target
is $0.46, which is why the clip moved to $0.75.
"""

from __future__ import annotations

import unittest


def _resize(directive_size, price, available_quote, min_notional=0.75,
            live=True, has_position=False):
    """Mirror of the resize block in TradingBot._evaluate_entry."""
    trade_size = float(directive_size)
    if live and not has_position and price > 0.0:
        if min_notional > 0.0 and trade_size * price < min_notional:
            affordable = max(0.0, available_quote * 0.5)
            target = min(min_notional, affordable)
            if target > 0.0:
                trade_size = target / price
    return trade_size


def _net_profit(notional, gross_return=0.05, fee_rate=0.0065):
    return notional * gross_return - notional * fee_rate


class MinDirectiveNotionalTest(unittest.TestCase):
    def test_observed_undersized_directive_is_raised(self):
        """The exact BASECAT case: $0.011 -> viable notional."""
        price = 0.02776
        size = _resize(0.39460784313725494, price, available_quote=6.977258)
        self.assertAlmostEqual(size * price, 0.75, places=6)

    def test_resized_trade_clears_the_profit_floor(self):
        """The whole point: the new size must actually be viable."""
        price = 0.02776
        size = _resize(0.39460784313725494, price, available_quote=6.977258)
        self.assertGreater(_net_profit(size * price), 0.02)

    def test_thirty_five_cent_clip_does_NOT_clear_the_floor(self):
        """Documents why the clip had to move: $0.35 nets $0.0152 < $0.02."""
        self.assertLess(_net_profit(0.35), 0.02)

    def test_adequate_directive_is_left_alone(self):
        price = 1.0
        self.assertAlmostEqual(_resize(5.0, price, available_quote=100.0), 5.0)

    def test_never_spends_more_than_half_the_wallet(self):
        price = 1.0
        size = _resize(0.001, price, available_quote=0.40, min_notional=10.0)
        self.assertLessEqual(size * price, 0.20 + 1e-9)

    def test_ghost_mode_is_untouched(self):
        """Only live sizing is governed by the real-money floor."""
        price = 0.02776
        size = _resize(0.3946, price, available_quote=6.98, live=False)
        self.assertAlmostEqual(size, 0.3946)

    def test_held_position_is_untouched(self):
        """Exits size off the held position, not the entry floor."""
        price = 0.02776
        size = _resize(0.3946, price, available_quote=6.98, has_position=True)
        self.assertAlmostEqual(size, 0.3946)

    def test_empty_wallet_leaves_size_unchanged(self):
        price = 1.0
        self.assertAlmostEqual(_resize(0.001, price, available_quote=0.0), 0.001)

    def test_zero_price_does_not_divide(self):
        self.assertAlmostEqual(_resize(0.3946, 0.0, available_quote=6.98), 0.3946)

    def test_disabled_when_min_notional_is_zero(self):
        price = 0.02776
        size = _resize(0.3946, price, available_quote=6.98, min_notional=0.0)
        self.assertAlmostEqual(size, 0.3946)


if __name__ == "__main__":
    unittest.main()
