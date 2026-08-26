"""
Discovery must actually find the tokens that are moving.

`fetch_trending_tokens` called `https://api.dexscreener.com/latest/dex/tokens`
with `?limit=&filter=`. That endpoint requires a token address and has no
trending variant, so it answered **HTTP 404 every single time**. The 404 was
caught and turned into an empty list, so the failure was completely silent:
`discovery_discoveredtoken` sat at **0 rows** for the life of the deployment
and a token that started moving could never enter the streamed universe.

That matters because pair selection scores candidates from *stored* OHLCV
(`avg_volume * (1 + volatility) + liquidity`), which describes how a token
behaved in the past. Discovery is the only path by which something moving
*right now* gets picked up -- and it was dead.

GeckoTerminal publishes a real per-network trending list. Verified live: 20
Base pools including Basecat +8.02%/1h, NVDAc +3.85%, STONKEX +3.83%.
"""

from __future__ import annotations

import unittest
from unittest import mock

from services.discovery.trending_fetcher import (
    _GECKO_NETWORKS,
    fetch_trending_tokens,
)


def _gecko_payload():
    """One pool, shaped the way GeckoTerminal actually returns them."""
    return {
        "data": [
            {
                "attributes": {
                    "name": "Basecat / USDC 0.9%",
                    "address": "0xpool",
                    "base_token_price_usd": "0.02487",
                    "reserve_in_usd": "780101.0",
                    "fdv_usd": "1000000",
                    "price_change_percentage": {"h1": "8.022", "h6": "12.0", "h24": "20.0"},
                    "volume_usd": {"h24": "2416702.4"},
                },
                "relationships": {"dex": {"data": {"id": "uniswap"}}},
            }
        ]
    }


class TrendingFetcher(unittest.TestCase):
    def test_a_pool_name_is_split_into_base_and_quote(self):
        """
        GeckoTerminal names pools "Basecat / USDC 0.9%".

        The fee suffix must not end up in the quote symbol.
        """
        with mock.patch("requests.get") as get:
            get.return_value = mock.Mock(
                status_code=200, json=_gecko_payload, raise_for_status=lambda: None
            )
            tokens = fetch_trending_tokens(limit=10, chains=["base"])
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].symbol, "Basecat-USDC")

    def test_price_movement_survives_the_conversion(self):
        """The whole point is finding movers, so the change must come through."""
        with mock.patch("requests.get") as get:
            get.return_value = mock.Mock(
                status_code=200, json=_gecko_payload, raise_for_status=lambda: None
            )
            token = fetch_trending_tokens(limit=10, chains=["base"])[0]
        self.assertAlmostEqual(token.price_change_1h, 8.022, places=3)
        self.assertAlmostEqual(token.liquidity_usd, 780101.0, places=1)

    def test_an_unknown_chain_is_skipped_not_fatal(self):
        self.assertEqual(fetch_trending_tokens(chains=["not-a-chain"]), [])

    def test_a_failing_request_yields_no_tokens_rather_than_raising(self):
        """
        Discovery is best-effort; it must never take the worker down.

        It should, however, be loud -- the silent 404 is what hid this bug.
        """
        with mock.patch("requests.get", side_effect=OSError("network down")):
            self.assertEqual(fetch_trending_tokens(chains=["base"]), [])

    def test_the_limit_is_respected_across_chains(self):
        with mock.patch("requests.get") as get:
            get.return_value = mock.Mock(
                status_code=200, json=_gecko_payload, raise_for_status=lambda: None
            )
            tokens = fetch_trending_tokens(limit=1, chains=["base", "ethereum"])
        self.assertLessEqual(len(tokens), 1)

    def test_the_chains_we_trade_are_all_mapped(self):
        """A missing mapping silently disables discovery for that chain."""
        for chain in ("base", "ethereum", "arbitrum", "optimism", "polygon", "bsc"):
            self.assertIn(chain, _GECKO_NETWORKS)


class PromotionToStreamWatchlist(unittest.TestCase):
    """
    Discovery must put movers where pair selection will actually see them.

    Fixing the fetcher was necessary but not sufficient: discovery wrote only
    to its own `DiscoveredToken`/`DiscoveryEvent` tables, and nothing under
    `trading/` reads those. A token could be discovered, recorded, and still
    never streamed. Pair selection promotes symbols from the `stream`
    watchlist, so that is the link that was missing.
    """

    @staticmethod
    def _token(symbol, change, liquidity):
        from services.discovery.trending_fetcher import TrendingToken
        return TrendingToken(
            symbol=symbol, chain="base", pair_address="0x", dex_id="uniswap",
            price_usd=1.0, volume_24h_usd=1e6, liquidity_usd=liquidity,
            price_change_1h=change, price_change_6h=change, price_change_24h=change,
            metadata={},
        )

    def _promote(self, tokens):
        from services.discovery.coordinator import DiscoveryCoordinator
        captured = {}

        def fake_save(watchlists, db=None):
            captured.update(watchlists)
            return watchlists

        with mock.patch("services.watchlists.load_watchlists", return_value={"stream": [], "ghost": [], "live": []}),              mock.patch("services.watchlists.save_watchlists", side_effect=fake_save):
            DiscoveryCoordinator(chains=["base"])._promote_to_stream_watchlist(tokens)
        return captured.get("stream", [])

    def test_a_rising_liquid_usd_pair_is_promoted(self):
        """The case this exists for: a mover that was not being streamed."""
        result = self._promote([self._token("NVDAC-USDC", 2.65, 861780.0)])
        self.assertIn("NVDAC-USDC", result)

    def test_a_falling_pair_is_not_promoted(self):
        """A trending list includes things trending DOWNWARD."""
        self.assertEqual(self._promote([self._token("VELVET-USDC", -5.5, 1400912.0)]), [])

    def test_an_illiquid_pair_is_not_promoted(self):
        """In a shallow pool the quoted move is noise, not opportunity."""
        self.assertEqual(self._promote([self._token("TINY-USDC", 9.0, 1200.0)]), [])

    def test_token_token_pairs_are_refused(self):
        """
        A token/token pair prices as a RATIO, not a dollar value.

        Streaming one poisons every strategy's arithmetic -- the same class of
        bug that produced fantasy P&L before.
        """
        for symbol in ("TIBBIR-VIRTUAL", "VVV-WETH", "BASECAT-ETH"):
            with self.subTest(symbol):
                self.assertEqual(self._promote([self._token(symbol, 5.0, 900000.0)]), [])

    def test_promotion_is_capped(self):
        """A burst of discoveries must not crowd out the existing universe."""
        many = [self._token(f"TOK{i}-USDC", 5.0, 900000.0) for i in range(80)]
        self.assertLessEqual(len(self._promote(many)), 48)

    def test_a_watchlist_failure_never_breaks_discovery(self):
        """Best effort: promotion is an enhancement, not a dependency."""
        from services.discovery.coordinator import DiscoveryCoordinator
        with mock.patch("services.watchlists.load_watchlists", side_effect=OSError("db gone")):
            DiscoveryCoordinator(chains=["base"])._promote_to_stream_watchlist(
                [self._token("NVDAC-USDC", 2.65, 861780.0)]
            )   # must not raise


if __name__ == "__main__":
    unittest.main()
