"""A wrong valuation is worse than a missing one.

Observed 2026-08-27: a native holding was persisted at $0.1428 for 0.00285556
ETH -- an implied $50.00/ETH against a real $2,499.61, exactly 1/50th. The
wallet then read as gas-starved and the live gate blocked on
``native_gas_starved`` while it actually held $7.14 of ETH. Nothing was
broken except the number, and the number silently stopped trading.
"""

from __future__ import annotations

import unittest
from unittest import mock

from services.wallet_bootstrap import _sane_native_usd


class _Row(dict):
    """Stands in for a sqlite3.Row (supports .keys() and ['usd'])."""


class _DB:
    def __init__(self, price=2499.61):
        self._price = price

    def fetch_price(self, chain, token):
        if self._price is None:
            return None
        return _Row(usd=self._price)


class NativeValuationTest(unittest.TestCase):
    def test_fifty_dollar_eth_is_corrected(self):
        """The exact observed failure."""
        out = _sane_native_usd(
            {"symbol": "ETH", "quantity": 0.00285556, "usd": 0.1428}, "base", _DB()
        )
        self.assertAlmostEqual(out, 0.00285556 * 2499.61, places=4)

    def test_correct_value_is_left_alone(self):
        reported = 0.00285556 * 2499.61
        out = _sane_native_usd(
            {"symbol": "ETH", "quantity": 0.00285556, "usd": reported}, "base", _DB()
        )
        self.assertAlmostEqual(out, reported, places=6)

    def test_small_disagreement_is_tolerated(self):
        """Feeds differ slightly; only a real divergence is a bug."""
        reported = 0.00285556 * 2400.0        # ~4% off
        out = _sane_native_usd(
            {"symbol": "ETH", "quantity": 0.00285556, "usd": reported}, "base", _DB()
        )
        self.assertAlmostEqual(out, reported, places=6)

    def test_missing_value_is_filled(self):
        out = _sane_native_usd(
            {"symbol": "ETH", "quantity": 0.00285556, "usd": 0.0}, "base", _DB()
        )
        self.assertGreater(out, 7.0)

    def test_non_native_token_passes_through(self):
        """Only native tokens are cross-checked; USDC is its own reference."""
        out = _sane_native_usd(
            {"symbol": "USDC", "quantity": 6.98, "usd": 6.98}, "base", _DB()
        )
        self.assertAlmostEqual(out, 6.98, places=6)

    def test_no_reference_price_keeps_the_report(self):
        """Without a reference we cannot judge; do not invent a number."""
        out = _sane_native_usd(
            {"symbol": "ETH", "quantity": 0.00285556, "usd": 0.1428}, "base", _DB(price=None)
        )
        self.assertAlmostEqual(out, 0.1428, places=6)

    def test_zero_quantity_is_safe(self):
        out = _sane_native_usd(
            {"symbol": "ETH", "quantity": 0.0, "usd": 0.0}, "base", _DB()
        )
        self.assertEqual(out, 0.0)

    def test_malformed_input_does_not_raise(self):
        out = _sane_native_usd(
            {"symbol": "ETH", "quantity": "not-a-number", "usd": None}, "base", _DB()
        )
        self.assertEqual(out, 0.0)

    def test_db_failure_does_not_raise(self):
        class _Broken:
            def fetch_price(self, chain, token):
                raise RuntimeError("db down")
        out = _sane_native_usd(
            {"symbol": "ETH", "quantity": 0.00285556, "usd": 0.1428}, "base", _Broken()
        )
        self.assertAlmostEqual(out, 0.1428, places=6)


if __name__ == "__main__":
    unittest.main()
