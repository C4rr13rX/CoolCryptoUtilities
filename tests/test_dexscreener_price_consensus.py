"""
DexScreener prices must survive a spoofed liquidity number.

Most symbols this bot trades are Base-chain DEX tokens that no centralised
exchange lists, so DexScreener is the only feed that can price them. Its search
endpoint returns every pool matching a ticker, and anyone can deploy a pool and
report whatever `liquidity.usd` they like.

The parser used to pick the pool with the highest reported liquidity and hand
that price straight to the trade path. Observed live on 2026-08-26:

    cbXRP/USDC     1.41 on three DEXes, and a pool claiming $117M at 0.001177
    BASECAT/USDC   0.025 with $500K liquidity and $2.4M daily volume,
                   and a pool quoting 0.00005

A 1,200x and a 500x error respectively, both of which the old selection rule
would have accepted as the live price. This is not a hypothetical: the guard
fired on real market data the first time it ran.
"""

from __future__ import annotations

import unittest

from trading.data_stream import _extract_rest_price


def _pair(base: str, quote: str, price: str, liquidity: float) -> dict:
    return {
        "baseToken": {"symbol": base},
        "quoteToken": {"symbol": quote},
        "priceUsd": price,
        "liquidity": {"usd": liquidity},
    }


class DexScreenerPriceConsensus(unittest.TestCase):
    def test_a_deep_outlier_pool_cannot_outvote_the_market(self):
        """The cbXRP case: one huge fake pool against three honest ones."""
        payload = {
            "pairs": [
                _pair("cbXRP", "USDC", "1.42", 34210.0),
                _pair("cbXRP", "USDC", "0.001177", 117664909.0),  # spoofed
                _pair("cbXRP", "USDC", "1.41", 20481.0),
                _pair("cbXRP", "USDC", "1.40", 8629.0),
            ]
        }
        price = _extract_rest_price("dexscreener", payload, "CBXRP", "USDC")
        self.assertIsNotNone(price)
        self.assertAlmostEqual(price, 1.41, delta=0.05)

    def test_the_honest_deepest_pool_is_still_preferred(self):
        """
        The guard must not fight normal price dispersion.

        When pools broadly agree, the deepest one is the best estimate and
        should be returned untouched -- otherwise this trades a rare spoof for
        a permanent loss of precision on every other symbol.
        """
        payload = {
            "pairs": [
                _pair("BASECAT", "USDC", "0.02514", 500375.0),
                _pair("BASECAT", "USDC", "0.02497", 16868.0),
                _pair("BASECAT", "USDC", "0.02473", 9585.0),
            ]
        }
        price = _extract_rest_price("dexscreener", payload, "BASECAT", "USDC")
        self.assertAlmostEqual(price, 0.02514, places=5)

    def test_too_few_pools_to_form_a_consensus_is_left_alone(self):
        """
        With fewer than three pools there is no majority to appeal to.

        Two pools disagreeing gives no way to tell which one is lying, so the
        liquidity heuristic stands rather than inventing a false consensus.
        """
        payload = {
            "pairs": [
                _pair("OBSCURE", "USDC", "1.00", 5000.0),
                _pair("OBSCURE", "USDC", "0.001", 900000.0),
            ]
        }
        price = _extract_rest_price("dexscreener", payload, "OBSCURE", "USDC")
        self.assertAlmostEqual(price, 0.001, places=6)

    def test_mismatched_symbols_are_never_priced(self):
        """A pool for a different token must not price this symbol."""
        payload = {
            "pairs": [
                _pair("NOTOURTOKEN", "USDC", "42.0", 999999.0),
            ]
        }
        self.assertIsNone(
            _extract_rest_price("dexscreener", payload, "BASECAT", "USDC")
        )


if __name__ == "__main__":
    unittest.main()
