"""A price from another chain is not this chain's price.

The DexScreener search endpoint returns every chain's pools for a ticker, and
the extractor selected by liquidity alone. Measured live 2026-08-27:

    MAMO/USDC on solana = 0.1723  with $171,338,732 liquidity
    MAMO/USDC on base   = 0.0103  with $362,801 liquidity

Base is what this bot trades, but Solana won on depth every time. The feed then
alternated between 0.0103 and 0.1723 (16x) and AERO between 0.5138 and 1.14
(2.2x), which made every ATF signal fail feed corroboration -- 100% of live
candidates refused, so no position could open at all.
"""

from __future__ import annotations

import unittest

from trading.data_stream import _extract_rest_price


def _pair(chain, base, quote, price, liquidity):
    return {
        "chainId": chain,
        "baseToken": {"symbol": base},
        "quoteToken": {"symbol": quote},
        "priceUsd": str(price),
        "priceNative": str(price),
        "liquidity": {"usd": liquidity},
    }


class DexscreenerChainFilterTest(unittest.TestCase):
    def test_deeper_foreign_chain_pool_is_ignored(self):
        """The real MAMO case: Solana is 470x deeper and must still lose."""
        payload = {"pairs": [
            _pair("solana", "MAMO", "USDC", 0.1723, 171_338_732),
            _pair("base", "MAMO", "USDC", 0.0103, 362_801),
        ]}
        self.assertAlmostEqual(
            _extract_rest_price("dexscreener", payload, "MAMO", "USDC", "base"),
            0.0103, places=6,
        )

    def test_aero_case(self):
        payload = {"pairs": [
            _pair("ethereum", "AERO", "USDC", 1.14, 90_000_000),
            _pair("base", "AERO", "USDC", 0.5153, 29_421_774),
        ]}
        self.assertAlmostEqual(
            _extract_rest_price("dexscreener", payload, "AERO", "USDC", "base"),
            0.5153, places=6,
        )

    def test_without_chain_the_bug_reproduces(self):
        """Pins the old behaviour so the regression is unambiguous."""
        payload = {"pairs": [
            _pair("solana", "MAMO", "USDC", 0.1723, 171_338_732),
            _pair("base", "MAMO", "USDC", 0.0103, 362_801),
        ]}
        self.assertAlmostEqual(
            _extract_rest_price("dexscreener", payload, "MAMO", "USDC"),
            0.1723, places=6,
        )

    def test_deepest_pool_on_the_right_chain_still_wins(self):
        payload = {"pairs": [
            _pair("base", "AERO", "USDC", 0.5153, 29_000_000),
            _pair("base", "AERO", "USDC", 0.4000, 1_000),
        ]}
        self.assertAlmostEqual(
            _extract_rest_price("dexscreener", payload, "AERO", "USDC", "base"),
            0.5153, places=6,
        )

    def test_pair_without_chain_id_is_not_dropped(self):
        """Absent chainId must not silently discard an otherwise valid pool."""
        payload = {"pairs": [{
            "baseToken": {"symbol": "AERO"}, "quoteToken": {"symbol": "USDC"},
            "priceUsd": "0.5153", "priceNative": "0.5153",
            "liquidity": {"usd": 1_000_000},
        }]}
        self.assertAlmostEqual(
            _extract_rest_price("dexscreener", payload, "AERO", "USDC", "base"),
            0.5153, places=6,
        )

    def test_no_pool_on_the_requested_chain_returns_nothing(self):
        """Better to go silent than to hand back another chain's price."""
        payload = {"pairs": [_pair("solana", "MAMO", "USDC", 0.1723, 171_338_732)]}
        self.assertIsNone(
            _extract_rest_price("dexscreener", payload, "MAMO", "USDC", "base")
        )

    def test_peer_median_guard_still_applies_within_a_chain(self):
        """A fabricated-liquidity pool on the right chain is still outvoted."""
        payload = {"pairs": [
            _pair("base", "CBXRP", "USDC", 1.41, 10_000),
            _pair("base", "CBXRP", "USDC", 1.40, 12_000),
            _pair("base", "CBXRP", "USDC", 1.42, 11_000),
            _pair("base", "CBXRP", "USDC", 0.001177, 117_000_000),
        ]}
        price = _extract_rest_price("dexscreener", payload, "CBXRP", "USDC", "base")
        self.assertGreater(price, 1.0)


if __name__ == "__main__":
    unittest.main()
