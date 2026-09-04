"""A wrong valuation is worse than a missing one.

Observed 2026-08-27: a native holding was persisted at $0.1428 for 0.00285556
ETH -- an implied $50.00/ETH against a real $2,499.61, exactly 1/50th. The
wallet then read as gas-starved and the live gate blocked on
``native_gas_starved`` while it actually held $7.14 of ETH. Nothing was
broken except the number, and the number silently stopped trading.

Observed 2026-09-04, the same bug on the other leg: the check was applied to
NATIVE symbols only. Measured through ``scan_wallet_holdings`` on the live
wallet:

    ETH   qty=0.002773371878944287  usd=6.938      -> $2501.65  ok
    USDC  qty=18.190627             usd=3.687393   -> $0.2027   WRONG
    AERO  qty=1.5462814519601997    usd=0.0        -> $0.00     WRONG

18.19 USDC carried at $3.69 is the number ``pipeline._wallet_state`` sums into
``stable_usd``, from which the live clip is sized -- the live gate reported
deployable_stable_usd $3.687393 against a wallet holding $18.19.
"""

from __future__ import annotations

import time
import unittest

from services.wallet_bootstrap import _reference_price_usd, _sane_holding_usd


class _Row(dict):
    """Stands in for a sqlite3.Row (supports .keys() and ['usd'])."""


class _DB:
    """Prices keyed by token, as ``fetch_price`` really is.

    The previous stub answered every token with the same price, which is what
    let ``test_non_native_token_passes_through`` assert that USDC needs no
    reference: with a token-blind stub, checking USDC would have "corrected"
    6.98 USDC to $17,447. The real table is keyed, and USDC really is in it.
    """

    def __init__(self, prices=None, ticks=None, age_sec=0.0):
        self._prices = {"eth": 2499.61} if prices is None else prices
        self._ticks = ticks or {}
        self._age = age_sec

    def fetch_price(self, chain, token):
        if token not in self._prices:
            return None
        # `usd` is stored as TEXT in the real table; keep the stub honest.
        return _Row(usd=str(self._prices[token]), ts=time.time() - self._age)

    def recent_market_prices(self, symbol, chain, *, since_ts=None, limit=25):
        return [(p, time.time()) for p in self._ticks.get(symbol, [])]


class NativeValuationTest(unittest.TestCase):
    def test_fifty_dollar_eth_is_corrected(self):
        """The exact observed failure."""
        out = _sane_holding_usd(
            {"symbol": "ETH", "quantity": 0.00285556, "usd": 0.1428}, "base", _DB()
        )
        self.assertAlmostEqual(out, 0.00285556 * 2499.61, places=4)

    def test_correct_value_is_left_alone(self):
        reported = 0.00285556 * 2499.61
        out = _sane_holding_usd(
            {"symbol": "ETH", "quantity": 0.00285556, "usd": reported}, "base", _DB()
        )
        self.assertAlmostEqual(out, reported, places=6)

    def test_small_disagreement_is_tolerated(self):
        """Feeds differ slightly; only a real divergence is a bug."""
        reported = 0.00285556 * 2400.0        # ~4% off
        out = _sane_holding_usd(
            {"symbol": "ETH", "quantity": 0.00285556, "usd": reported}, "base", _DB()
        )
        self.assertAlmostEqual(out, reported, places=6)

    def test_missing_value_is_filled(self):
        out = _sane_holding_usd(
            {"symbol": "ETH", "quantity": 0.00285556, "usd": 0.0}, "base", _DB()
        )
        self.assertGreater(out, 7.0)

    def test_no_reference_price_keeps_the_report(self):
        """Without a reference we cannot judge; do not invent a number."""
        out = _sane_holding_usd(
            {"symbol": "ETH", "quantity": 0.00285556, "usd": 0.1428},
            "base",
            _DB(prices={}),
        )
        self.assertAlmostEqual(out, 0.1428, places=6)

    def test_zero_quantity_is_safe(self):
        out = _sane_holding_usd(
            {"symbol": "ETH", "quantity": 0.0, "usd": 0.0}, "base", _DB()
        )
        self.assertEqual(out, 0.0)

    def test_malformed_input_does_not_raise(self):
        out = _sane_holding_usd(
            {"symbol": "ETH", "quantity": "not-a-number", "usd": None}, "base", _DB()
        )
        self.assertEqual(out, 0.0)

    def test_db_failure_does_not_raise(self):
        class _Broken:
            def fetch_price(self, chain, token):
                raise RuntimeError("db down")

            def recent_market_prices(self, *a, **k):
                raise RuntimeError("db down")

        out = _sane_holding_usd(
            {"symbol": "ETH", "quantity": 0.00285556, "usd": 0.1428}, "base", _Broken()
        )
        self.assertAlmostEqual(out, 0.1428, places=6)


class StableAndTokenValuationTest(unittest.TestCase):
    """The 2026-09-04 half: the leg that was returned verbatim."""

    def test_eighteen_usdc_is_not_worth_three_dollars(self):
        """The exact observed failure: 18.190627 USDC carried at $3.687393."""
        out = _sane_holding_usd(
            {"symbol": "USDC", "quantity": 18.190627, "usd": 3.687393},
            "base",
            _DB(prices={"usdc": 0.999773}),
        )
        self.assertAlmostEqual(out, 18.190627 * 0.999773, places=6)

    def test_a_correct_stable_value_is_left_alone(self):
        out = _sane_holding_usd(
            {"symbol": "USDC", "quantity": 6.98, "usd": 6.98},
            "base",
            _DB(prices={"usdc": 0.999773}),
        )
        self.assertAlmostEqual(out, 6.98, places=6)

    def test_a_stable_with_no_quote_falls_back_to_one_dollar(self):
        """$1.00 is a definition, so it is reached only when nothing quoted."""
        out = _sane_holding_usd(
            {"symbol": "USDC", "quantity": 18.190627, "usd": 3.687393},
            "base",
            _DB(prices={}),
        )
        self.assertAlmostEqual(out, 18.190627, places=6)

    def test_a_held_token_priced_at_zero_is_filled_from_the_tick_feed(self):
        """AERO is absent from the price table; market_stream is the reference.

        Left at $0.00 it falls under MIN_HOLDING_USD and
        ``generate_pairs_from_holdings`` drops the stream for a token the
        wallet is actually holding.
        """
        out = _sane_holding_usd(
            {"symbol": "AERO", "quantity": 1.5462814519601997, "usd": 0.0},
            "base",
            _DB(prices={}, ticks={"AERO-USDC": [0.5106, 0.5104, 0.5105]}),
        )
        self.assertAlmostEqual(out, 1.5462814519601997 * 0.5105, places=6)

    def test_the_tick_reference_is_a_median_not_the_newest_tick(self):
        """market_stream interleaves denominations; one sample is a coin flip.

        AERO has been observed publishing 0.5138 and 1.14 alternately. The
        median survives that; ``[-1]`` or ``[0]`` does not.
        """
        px = _reference_price_usd(
            "AERO", "base", _DB(prices={}, ticks={"AERO-USDC": [1.14, 0.5138, 0.5140, 1.14, 0.5139]})
        )
        self.assertIsNotNone(px)
        self.assertAlmostEqual(px, 0.5140, places=6)

    def test_a_stale_reference_does_not_overrule_a_report(self):
        """A dead feed must stop correcting, not correct to a dead number."""
        out = _sane_holding_usd(
            {"symbol": "CBBTC", "quantity": 1.0, "usd": 81000.0},
            "base",
            _DB(prices={"cbbtc": 40000.0}, age_sec=90 * 86400),
        )
        self.assertAlmostEqual(out, 81000.0, places=6)

    def test_an_unknown_token_with_no_reference_is_untouched(self):
        out = _sane_holding_usd(
            {"symbol": "WHOKNOWS", "quantity": 5.0, "usd": 1.23}, "base", _DB(prices={})
        )
        self.assertAlmostEqual(out, 1.23, places=6)


if __name__ == "__main__":
    unittest.main()
