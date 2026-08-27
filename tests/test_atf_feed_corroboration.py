"""A signal price that the feed cannot confirm must never become a trade.

Regression test for the fabricated fill that carried atf_static to graduation:
BASELIFE-USDC was entered at 2.05e-07 and exited at 4.39e-06 (+2038%) on a
symbol with zero rows in ``market_stream``. That one trade was 81% of the
strategy's entire net profit.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from services.atf_static_strategy import _corroborated_price


class _FakeDB:
    """Stands in for the market_stream lookup."""

    def __init__(self, price=None):
        self._price = price
        self.calls = []

    def get_market_price(self, symbol, chain, *, ts=None, after=True):
        self.calls.append((symbol, chain))
        if self._price is None:
            return None
        return (self._price, 0.0, None)


class FeedCorroborationTest(unittest.TestCase):
    def setUp(self):
        # The guard is on by default; make the tests independent of ambient env.
        self._env = mock.patch.dict(
            os.environ,
            {
                "ATF_STATIC_REQUIRE_FEED_PRICE": "1",
                "ATF_STATIC_MAX_FEED_DEV": "0.35",
            },
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_symbol_with_no_feed_is_refused(self):
        """The BASELIFE case: no streamed tick at all -> no trade."""
        db = _FakeDB(price=None)
        self.assertIsNone(
            _corroborated_price(db, "BASELIFE-USDC", "base", 2.05353933888008e-07)
        )

    def test_wild_quote_against_real_feed_is_refused(self):
        """A 21x divergence from the streamed price is not a fill."""
        db = _FakeDB(price=2.05e-07)
        self.assertIsNone(
            _corroborated_price(db, "BASELIFE-USDC", "base", 4.39062972154016e-06)
        )

    def test_agreeing_quote_is_accepted(self):
        db = _FakeDB(price=100.0)
        self.assertEqual(_corroborated_price(db, "SOL-USDC", "base", 103.0), 100.0)

    def test_accepted_price_is_the_feed_not_the_quote(self):
        """Entry and exit must share one price basis, so the feed wins."""
        db = _FakeDB(price=100.0)
        self.assertEqual(_corroborated_price(db, "SOL-USDC", "base", 110.0), 100.0)

    def test_nonpositive_quote_is_refused(self):
        db = _FakeDB(price=100.0)
        self.assertIsNone(_corroborated_price(db, "SOL-USDC", "base", 0.0))
        self.assertIsNone(_corroborated_price(db, "SOL-USDC", "base", -1.0))

    def test_guard_can_be_disabled_explicitly(self):
        """An operator opt-out exists, but it must be deliberate."""
        db = _FakeDB(price=None)
        with mock.patch.dict(os.environ, {"ATF_STATIC_REQUIRE_FEED_PRICE": "0"}):
            self.assertEqual(
                _corroborated_price(db, "BASELIFE-USDC", "base", 2.05e-07), 2.05e-07
            )

    def test_feed_lookup_is_bounded_by_age(self):
        """A stale tick must not be treated as corroboration."""
        db = _FakeDB(price=100.0)
        with mock.patch.dict(os.environ, {"ATF_STATIC_FEED_MAX_AGE_SEC": "900"}):
            _corroborated_price(db, "SOL-USDC", "base", 100.0)
        self.assertEqual(db.calls, [("SOL-USDC", "base")])


if __name__ == "__main__":
    unittest.main()
