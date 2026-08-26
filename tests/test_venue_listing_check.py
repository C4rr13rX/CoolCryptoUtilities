"""
A stream must not be handed a websocket for a pair the venue does not list.

This is the mechanism that kept the feed frozen even after DexScreener was
wired in. MEXC -- and okx, kucoin and bybit alike -- was offered to every token
whose quote was USDT/USDC/BTC/ETH, with no check that the venue carries it. For a Base-chain token the socket connects,
the subscription is accepted, and no data ever arrives -- so the stream looks
healthy, holds its seed price forever, and never rotates to the REST fallbacks
where the on-chain sources live.

`_select_next_endpoint` returns the first endpoint that renders a websocket
URL, and MEXC rendered one unconditionally. DexScreener has no websocket, so it
sat behind a WS endpoint that never failed and was therefore never reached.

The check fails OPEN by design: a probe failure must never cost a feed that
would otherwise work.
"""

from __future__ import annotations

import unittest
from unittest import mock

import trading.data_stream as ds


class VenueListingCheck(unittest.TestCase):
    def setUp(self):
        self._saved = dict(ds._VENUE_SYMBOLS)
        ds._VENUE_SYMBOLS.clear()

    def tearDown(self):
        ds._VENUE_SYMBOLS.clear()
        ds._VENUE_SYMBOLS.update(self._saved)
        ds._VENUE_PROBE_RETRY_AT.clear()

    def test_a_listed_pair_is_allowed(self):
        ds._VENUE_SYMBOLS["mexc"] = {"ETHUSDC", "BTCUSDT"}
        self.assertTrue(ds._venue_lists("mexc", "ETH", "USDC"))
        self.assertTrue(ds._venue_lists("mexc", "btc", "usdt"))

    def test_an_unlisted_pair_is_refused(self):
        """The Base-chain case: this is what must stop being given a socket."""
        ds._VENUE_SYMBOLS["mexc"] = {"ETHUSDC", "BTCUSDT"}
        self.assertFalse(ds._venue_lists("mexc", "BASECAT", "USDC"))
        self.assertFalse(ds._venue_lists("mexc", "CBXRP", "USDC"))

    def test_a_probe_failure_fails_open(self):
        """
        Losing the symbol list must not lose the endpoint.

        A network blip should degrade to the old behaviour, not silently
        disable a venue that works.
        """
        with mock.patch("urllib.request.urlopen", side_effect=OSError("no network")):
            self.assertTrue(ds._venue_lists("mexc", "ETH", "USDC"))
        self.assertIsNone(
            ds._VENUE_SYMBOLS.get("mexc"), "a failed probe must not be cached"
        )


    def test_a_failing_probe_is_not_retried_on_every_call(self):
        """
        Fail open, but stop asking.

        bybit answers 403 from this host. Without a backoff the probe reran on
        every call, so a check meant to be in-process made a network round
        trip each time -- it turned a 0.06ms lookup into a 13ms one and showed
        up as a 26s test suite.
        """
        ds._VENUE_PROBE_RETRY_AT.pop("bybit", None)
        calls = []

        def boom(venue):
            calls.append(venue)
            raise OSError("403")

        with mock.patch.object(ds, "_fetch_venue_symbols", side_effect=boom):
            for _ in range(5):
                self.assertTrue(ds._venue_lists("bybit", "ETH", "USDC"))
        self.assertEqual(len(calls), 1, f"probed {len(calls)} times, expected 1")
        ds._VENUE_PROBE_RETRY_AT.pop("bybit", None)

    def test_the_check_can_be_disabled(self):
        ds._VENUE_SYMBOLS["mexc"] = {"ETHUSDC"}
        with mock.patch.dict("os.environ", {"VENUE_LISTING_CHECK": "0"}):
            self.assertTrue(ds._venue_lists("mexc", "BASECAT", "USDC"))


class EndpointSelectionForUnlistedPairs(unittest.TestCase):
    """The behaviour that actually matters: what the stream ends up using."""

    def setUp(self):
        self._saved = dict(ds._VENUE_SYMBOLS)
        # Every websocket venue lists ETH/BTC only, so no live call is made
        # and no venue can hand an unlisted Base token a socket.
        listed = {"ETHUSDC", "ETHUSDT", "BTCUSDT", "BTCUSDC"}
        for venue in ("mexc", "okx", "kucoin", "bybit"):
            ds._VENUE_SYMBOLS[venue] = set(listed)

    def tearDown(self):
        ds._VENUE_SYMBOLS.clear()
        ds._VENUE_SYMBOLS.update(self._saved)

    def _endpoints(self, symbol):
        stream = ds.MarketDataStream(symbol=symbol, chain="base")
        return stream._build_endpoints()

    def test_an_unlisted_pair_gets_no_websocket_at_all(self):
        """
        With no WS endpoint, the stream must fall through to REST.

        That fall-through is the only path that reaches the on-chain sources.
        """
        names = [e.name for e in self._endpoints("BASECAT-USDC")]
        ws_capable = [e.name for e in self._endpoints("BASECAT-USDC") if e.ws_template]
        self.assertNotIn("mexc", names)
        self.assertEqual(ws_capable, [], f"unlisted pair still got a socket: {ws_capable}")

    def test_an_unlisted_pair_can_still_be_priced(self):
        """Refusing the socket is only correct if something else can price it."""
        names = [e.name for e in self._endpoints("BASECAT-USDC")]
        self.assertTrue(
            {"dexscreener", "geckoterminal"} & set(names),
            f"no on-chain source available for an unlisted pair: {names}",
        )

    def test_a_listed_pair_keeps_its_websocket(self):
        """The check must not cost working feeds their streaming price."""
        ws_capable = [e.name for e in self._endpoints("ETH-USDC") if e.ws_template]
        self.assertIn("mexc", ws_capable)


if __name__ == "__main__":
    unittest.main()
