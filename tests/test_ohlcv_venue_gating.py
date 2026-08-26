"""
Startup must not spend its time on a venue that always refuses us.

Pair selection downloads OHLCV history for every candidate pair, synchronously,
on the main thread, BEFORE any market stream starts. A py-spy dump of live
production caught startup parked exactly there:

    ssl_wrap_socket (urllib3/util/ssl_.py:461)
    _fetch_binance_klines (services/cex_ohlcv_fallback.py:147)
    download_pair -> _ensure_ohlcv -> try_add_candidate -> select_pairs
    build (trading/selector.py:754) -> start (production.py:185)

Binance answers **HTTP 451** ("Service unavailable from a restricted location")
from this host -- which is why it is already named in MARKET_ENDPOINT_EXCLUDE --
but the download path never consulted that and tried it once per pair anyway.

The two config lists mean different things here, and conflating them would
break the working path:

  * `MARKET_ENDPOINT_INCLUDE` is the allowlist for live price STREAMING.
    Coinbase is absent from it because its ticker endpoints 403 here -- yet its
    candle API answers 200 and is the first and best OHLCV source. Gating
    downloads on that list would disable the venue that actually works.
  * `MARKET_ENDPOINT_EXCLUDE` names venues that refuse us outright.

So this consults the exclude list only.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from services.cex_ohlcv_fallback import _venue_enabled


class VenueGating(unittest.TestCase):
    def test_an_excluded_venue_is_skipped(self):
        with mock.patch.dict(os.environ, {"MARKET_ENDPOINT_EXCLUDE": "binance"}):
            self.assertFalse(_venue_enabled("binance"))

    def test_coinbase_survives_the_streaming_allowlist(self):
        """
        The regression this guards.

        Coinbase is deliberately absent from the streaming allowlist but its
        candle API works, so it must remain enabled for downloads.
        """
        with mock.patch.dict(os.environ, {
            "MARKET_ENDPOINT_INCLUDE": "mexc,kucoin,coingecko,dexscreener",
            "MARKET_ENDPOINT_EXCLUDE": "binance",
        }):
            self.assertTrue(_venue_enabled("coinbase"))
            self.assertTrue(_venue_enabled("coingecko"))

    def test_matching_is_case_insensitive_and_tolerates_spacing(self):
        with mock.patch.dict(os.environ, {"MARKET_ENDPOINT_EXCLUDE": " Binance , OKX "}):
            self.assertFalse(_venue_enabled("BINANCE"))
            self.assertFalse(_venue_enabled("okx"))

    def test_a_download_specific_override_is_honoured(self):
        """A venue may be fine for streaming but useless for candles."""
        with mock.patch.dict(os.environ, {"OHLCV_VENUE_EXCLUDE": "coingecko"}):
            self.assertFalse(_venue_enabled("coingecko"))

    def test_nothing_is_disabled_when_no_lists_are_set(self):
        """A host that configured nothing must keep every venue."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MARKET_ENDPOINT_EXCLUDE", None)
            os.environ.pop("OHLCV_VENUE_EXCLUDE", None)
            for venue in ("binance", "coinbase", "coingecko"):
                self.assertTrue(_venue_enabled(venue))


if __name__ == "__main__":
    unittest.main()
