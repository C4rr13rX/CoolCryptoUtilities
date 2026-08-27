"""A zero-volume tick must not make live trading impossible.

Measured 2026-08-27: 100% of market_stream ticks in the last hour reported
volume=0. These are price updates from book changes, not trades, so volume is
legitimately absent. ``trade_size`` is derived from volume, so it was ALWAYS
zero in live mode, and every live directive bailed at 'insufficient_quote'
before the entry logic ran.

Ghost mode already had a floor for exactly this reason; live mode did not. Zero
live trades were possible regardless of what any gate said -- all ten risk
gates PASSed with block_reason empty while this silently blocked every entry.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock


def _size_with_floor(volume, price, available_quote, max_trade_share=0.12,
                     use_sim=False, live_min_usd="0.35"):
    """Mirror of the sizing block in TradingBot._evaluate_entry."""
    trade_size = max(min(volume * max_trade_share, volume), 0.0)
    trade_size = min(trade_size, 100.0)
    if use_sim and price > 0.0:
        floor = float(os.getenv("GHOST_MIN_TRADE_USD", "2.0")) / price
        if trade_size < floor:
            trade_size = floor
    elif price > 0.0 and trade_size <= 0.0:
        clip = float(live_min_usd)
        clip = min(clip, max(0.0, available_quote * 0.5))
        if clip > 0.0:
            trade_size = clip / price
    return trade_size


class ZeroVolumeSizingTest(unittest.TestCase):
    def test_zero_volume_no_longer_zeroes_live_size(self):
        """The exact observed failure."""
        size = _size_with_floor(volume=0.0, price=0.5201, available_quote=6.977258)
        self.assertGreater(size, 0.0)

    def test_floor_respects_the_risk_approved_clip(self):
        """Real money is sized by the transition plan, not by a ghost default."""
        price = 2.0
        size = _size_with_floor(volume=0.0, price=price, available_quote=100.0,
                                live_min_usd="0.35")
        self.assertAlmostEqual(size * price, 0.35, places=6)

    def test_floor_never_exceeds_half_the_wallet(self):
        """A thin wallet must not be spent in one trade."""
        price = 1.0
        size = _size_with_floor(volume=0.0, price=price, available_quote=0.40,
                                live_min_usd="10.0")
        self.assertLessEqual(size * price, 0.20 + 1e-9)

    def test_empty_wallet_still_sizes_zero(self):
        size = _size_with_floor(volume=0.0, price=1.0, available_quote=0.0)
        self.assertEqual(size, 0.0)

    def test_real_volume_still_drives_size(self):
        """The floor must not override genuine depth information."""
        size = _size_with_floor(volume=1000.0, price=1.0, available_quote=100.0)
        self.assertAlmostEqual(size, 120.0 if False else min(1000.0 * 0.12, 100.0))

    def test_ghost_floor_is_unchanged(self):
        with mock.patch.dict(os.environ, {"GHOST_MIN_TRADE_USD": "2.0"}):
            size = _size_with_floor(volume=0.0, price=1.0, available_quote=0.0,
                                    use_sim=True)
        self.assertAlmostEqual(size, 2.0)

    def test_zero_price_does_not_divide(self):
        self.assertEqual(_size_with_floor(volume=0.0, price=0.0, available_quote=10.0), 0.0)

    def test_tiny_price_produces_large_but_bounded_notional(self):
        """A sub-cent token must still cost only the clip in USD."""
        price = 0.009503
        size = _size_with_floor(volume=0.0, price=price, available_quote=6.977258)
        self.assertAlmostEqual(size * price, 0.35, places=6)


if __name__ == "__main__":
    unittest.main()
