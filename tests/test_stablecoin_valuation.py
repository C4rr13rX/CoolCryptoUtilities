"""
A stablecoin with no market price is worth ~$1, never $0.

Observed live on 2026-08-26: a real 8.378 USDC balance on base was scanned
correctly from chain and then valued at **0**, because the `prices` table held
no base rows at all and `_lookup_cached_price` returned None. Downstream that
is not read as "price unknown" -- it is read as "no money". The trading
pipeline reported `stable_usd: 0.1071` (only some arbitrum dust) and refused to
trade for want of capital the wallet actually held.

Two things had to be true for the fix to work, and both are pinned here:

  * the fallback matches on **contract address**, not symbol. The cached row
    for base USDC carried a NULL symbol, which is exactly why a symbol-keyed
    check silently did nothing.
  * the check runs **before** the `if not self.cp: return None` cache guard. A
    stablecoin is worth a dollar whether or not a price cache is configured.

Deliberately narrow: only coins whose entire premise is holding a USD peg. A
depegged stable is mispriced here by a few percent, against a 100% error the
other way.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

import balances
from balances import MultiChainTokenPortfolio


BASE_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71B54bda02913"
BASE_WETH = "0x4200000000000000000000000000000000000006"


def _portfolio() -> MultiChainTokenPortfolio:
    """A portfolio with no price cache — the state that produced the bug."""
    portfolio = MultiChainTokenPortfolio(
        wallet_address="0x" + "1" * 40,
        tokens=[],
        default_chain="base",
        verbose=False,
    )
    portfolio.cp = None
    return portfolio


class StablecoinAddressSet(unittest.TestCase):
    def test_the_address_set_is_derived_from_the_token_catalog(self):
        """
        Derived, not hardcoded, so the two lists cannot drift apart.

        A stablecoin added to the shared catalog is covered here with no
        second edit.
        """
        self.assertTrue(balances._STABLE_USD_ADDRESSES)
        self.assertIn(BASE_USDC.lower(), balances._STABLE_USD_ADDRESSES)

    def test_non_stablecoins_are_not_in_the_set(self):
        """WETH must never be assumed to be worth a dollar."""
        self.assertNotIn(BASE_WETH.lower(), balances._STABLE_USD_ADDRESSES)


class StablecoinPriceFallback(unittest.TestCase):
    def test_usdc_prices_at_one_without_any_cache(self):
        """The exact failure: no price cache, real balance, valued at zero."""
        price = _portfolio()._lookup_cached_price("base", BASE_USDC, None)
        self.assertEqual(price, Decimal("1"))

    def test_it_matches_on_address_when_the_symbol_is_missing(self):
        """
        The cached row for base USDC had a NULL symbol.

        A symbol-keyed fallback looked correct and did nothing at all, so this
        pins the address path specifically.
        """
        price = _portfolio()._lookup_cached_price("base", BASE_USDC, {"symbol": None})
        self.assertEqual(price, Decimal("1"))

    def test_address_matching_is_case_insensitive(self):
        """Checksummed and lowercase addresses must behave identically."""
        portfolio = _portfolio()
        self.assertEqual(
            portfolio._lookup_cached_price("base", BASE_USDC.lower(), None),
            portfolio._lookup_cached_price("base", BASE_USDC.upper(), None),
        )

    def test_a_volatile_token_still_returns_no_price(self):
        """
        The fallback must not leak into non-stable assets.

        Valuing WETH at $1 would be far worse than valuing it at 0.
        """
        self.assertIsNone(_portfolio()._lookup_cached_price("base", BASE_WETH, None))

    def test_an_unknown_token_still_returns_no_price(self):
        self.assertIsNone(
            _portfolio()._lookup_cached_price("base", "0x" + "9" * 40, None)
        )


if __name__ == "__main__":
    unittest.main()
