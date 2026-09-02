"""A live entry must not be refused for a token the feed already identified.

Link 9 (LIVE) failed with ``reason=token_unresolved``. The bot could resolve a
contract address from only two places -- the wallet's current holdings and the
static core catalog, EIGHT symbols on base -- while it trades whatever the feed
surfaces. Measured 2026-09-02 against the recorded outcomes: nine of the ten
traded symbols had a base token outside the catalog, so a live entry on any of
them was structurally impossible.

The address was never missing. GeckoTerminal returns it on every trending pool
as ``relationships.base_token.data.id`` = ``"base_0x..."`` and the fetcher kept
only the symbol. What discovery *did* store was the POOL address, which cannot
be swapped and, for a Uniswap v4 pair, is a 32-byte pool id rather than an
address at all.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import services.token_address_book as book
from services.discovery.trending_fetcher import _gecko_address, fetch_trending_tokens

# A real base-chain pool as GeckoTerminal returns it (Basecat / WETH 1%).
GECKO_POOL = {
    "attributes": {
        "name": "Basecat / WETH 1%",
        "address": "0xf79478d5a6bae4546f7e489e80b2fc690b558944",
        "base_token_price_usd": "0.0285",
        "volume_usd": {"h24": "1696506.4"},
        "reserve_in_usd": "738515.5",
        "price_change_percentage": {"h1": "-1.0", "h6": "-5.0", "h24": "-26.5"},
        "fdv_usd": "1000000",
    },
    "relationships": {
        "base_token": {"data": {"id": "base_0xb2000000000000000000004c27f6523082f41d01"}},
        "quote_token": {"data": {"id": "base_0x4200000000000000000000000000000000000006"}},
        "dex": {"data": {"id": "uniswap-v3-base"}},
    },
}

BASECAT = "0xb2000000000000000000004c27f6523082f41d01"
WETH_BASE = "0x4200000000000000000000000000000000000006"
#: 32 bytes. A real Uniswap v4 pool id from data/base_pair_provider_assignment.json.
V4_POOL_ID = "0x8930762cccc36040f25fc29db58a8ec22e872a347260d992f39666a3cdce7e5a"


class AddressShapeTest(unittest.TestCase):
    def test_a_v4_pool_id_is_not_a_token_address(self):
        """The distinction that keeps a swap from being pointed at a pool."""
        self.assertTrue(book.is_token_address(BASECAT))
        self.assertFalse(book.is_token_address(V4_POOL_ID))
        self.assertFalse(book.is_token_address("0xdeadbeef"))
        self.assertFalse(book.is_token_address(""))
        self.assertFalse(book.is_token_address(None))

    def test_gecko_ids_become_addresses(self):
        self.assertEqual(_gecko_address("base_" + BASECAT), BASECAT)
        self.assertEqual(_gecko_address(""), "")
        self.assertEqual(_gecko_address("base_" + V4_POOL_ID), "")


class AddressBookTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(book, "BOOK_PATH", Path(self.tmp.name) / "tok.json")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_records_and_looks_up(self):
        self.assertTrue(book.record("base", "TOAD", BASECAT))
        self.assertEqual(book.lookup("base", "TOAD"), BASECAT)

    def test_lookup_is_case_and_pair_insensitive(self):
        book.record("base", "TOAD", BASECAT)
        for probe in ("TOAD", "toad", "Toad", "TOAD-USDC", "toad-usdc"):
            self.assertEqual(book.lookup("base", probe), BASECAT, probe)
        self.assertEqual(book.lookup("BASE", "TOAD"), BASECAT)

    def test_a_pool_id_is_refused(self):
        """Storing this would be storing something that cannot be swapped."""
        self.assertFalse(book.record("base", "BASECAT", V4_POOL_ID))
        self.assertIsNone(book.lookup("base", "BASECAT"))

    def test_unknown_symbols_stay_unknown(self):
        """No guessing: an unresolvable token must keep blocking the trade."""
        self.assertIsNone(book.lookup("base", "NEVERSEEN"))

    def test_chains_do_not_bleed_into_each_other(self):
        book.record("base", "TOAD", BASECAT)
        self.assertIsNone(book.lookup("ethereum", "TOAD"))

    def test_an_unreadable_book_is_never_blanked(self):
        book.record("base", "TOAD", BASECAT)
        before = book.BOOK_PATH.read_text(encoding="utf-8")
        with mock.patch.object(book, "read_json", return_value=(None, False)):
            self.assertFalse(book.record("base", "OTHER", WETH_BASE))
        self.assertEqual(book.BOOK_PATH.read_text(encoding="utf-8"), before)


class FetcherKeepsAddressesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(book, "BOOK_PATH", Path(self.tmp.name) / "tok.json")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fetch(self):
        resp = mock.Mock()
        resp.json.return_value = {"data": [GECKO_POOL]}
        resp.raise_for_status.return_value = None
        with mock.patch("services.discovery.trending_fetcher.requests.get", return_value=resp):
            return fetch_trending_tokens(limit=10, chains=["base"])

    def test_token_addresses_survive_the_fetch(self):
        tok = self._fetch()[0]
        self.assertEqual(tok.symbol, "Basecat-WETH")
        self.assertEqual(tok.base_token_address, BASECAT)
        self.assertEqual(tok.quote_token_address, WETH_BASE)

    def test_the_pool_address_is_not_mistaken_for_the_token(self):
        tok = self._fetch()[0]
        self.assertEqual(tok.pair_address, "0xf79478d5a6bae4546f7e489e80b2fc690b558944")
        self.assertNotEqual(tok.pair_address, tok.base_token_address)

    def test_fetching_teaches_the_address_book(self):
        """Held only in memory, the address is as good as dropped: the resolver
        runs in another process, later."""
        self._fetch()
        self.assertEqual(book.lookup("base", "BASECAT"), BASECAT)
        self.assertEqual(book.lookup("base", "WETH"), WETH_BASE)


class ResolverTest(unittest.TestCase):
    """The bot's own resolver, which is what actually gates the live entry."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(book, "BOOK_PATH", Path(self.tmp.name) / "tok.json")
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _bot():
        from trading.bot import TradingBot

        bot = object.__new__(TradingBot)
        bot.portfolio = mock.Mock(holdings={})
        return bot

    def test_a_discovered_token_now_resolves(self):
        bot = self._bot()
        self.assertIsNone(bot._resolve_token_address("base", "BASECAT"))
        book.record("base", "BASECAT", BASECAT)
        self.assertEqual(bot._resolve_token_address("base", "BASECAT"), BASECAT)

    def test_mixed_case_catalog_symbols_resolve(self):
        """The catalog keys 'cbETH' and 'USDbC' in mixed case while the lookup
        upper-cases, so an exact .get() never matched them."""
        bot = self._bot()
        with mock.patch(
            "trading.bot.core_tokens_for_chain",
            return_value={"cbETH": WETH_BASE, "USDbC": BASECAT},
        ):
            self.assertEqual(bot._resolve_token_address("base", "CBETH"), WETH_BASE)
            self.assertEqual(bot._resolve_token_address("base", "USDBC"), BASECAT)

    def test_holdings_still_win(self):
        bot = self._bot()
        holding = mock.Mock(token=WETH_BASE)
        bot.portfolio = mock.Mock(holdings={("base", "TOAD"): holding})
        book.record("base", "TOAD", BASECAT)
        self.assertEqual(bot._resolve_token_address("base", "TOAD"), WETH_BASE)

    def test_a_pool_id_is_never_returned_as_a_token(self):
        """`len(raw) >= 42` accepted a 66-char v4 pool id as an address."""
        bot = self._bot()
        with mock.patch("trading.bot.core_tokens_for_chain", return_value={}):
            _, swap = bot._resolve_live_trade_asset("base", V4_POOL_ID)
        self.assertIsNone(swap)

    def test_a_literal_address_still_resolves(self):
        bot = self._bot()
        with mock.patch("trading.bot.core_tokens_for_chain", return_value={}):
            _, swap = bot._resolve_live_trade_asset("base", BASECAT)
        self.assertEqual(swap, BASECAT)

    def test_an_unknown_token_still_blocks(self):
        """The gate must keep holding for anything genuinely unidentified."""
        bot = self._bot()
        with mock.patch("trading.bot.core_tokens_for_chain", return_value={}):
            _, swap = bot._resolve_live_trade_asset("base", "NEVERSEEN")
        self.assertIsNone(swap)


if __name__ == "__main__":
    unittest.main()


class NativeSentinelTest(unittest.TestCase):
    """Correctly shaped addresses that are not ERC-20 contracts.

    Caught on live data 2026-09-02: a real GeckoTerminal fetch reported base
    ETH as the zero address, which passed a pure hex-shape check. Recording it
    would have pointed a swap at the burn address.
    """

    ZERO = "0x0000000000000000000000000000000000000000"
    EEEE = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(book, "BOOK_PATH", Path(self.tmp.name) / "tok.json")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_native_placeholders_are_not_token_addresses(self):
        self.assertFalse(book.is_token_address(self.ZERO))
        self.assertFalse(book.is_token_address(self.EEEE))
        self.assertFalse(book.is_token_address(self.EEEE.lower()))
        self.assertTrue(book.is_token_address(BASECAT))

    def test_they_are_never_recorded(self):
        self.assertFalse(book.record("base", "ETH", self.ZERO))
        self.assertIsNone(book.lookup("base", "ETH"))

    def test_they_never_reach_a_swap(self):
        bot = ResolverTest._bot()
        with mock.patch("trading.bot.core_tokens_for_chain", return_value={}):
            _, swap = bot._resolve_live_trade_asset("base", self.ZERO)
        self.assertIsNone(swap)
