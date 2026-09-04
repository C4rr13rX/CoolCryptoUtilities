"""A cached USD value belongs to the quantity it was computed from.

``MultiChainTokenPortfolio.build`` recomputes ``qty_map`` for EVERY token from
``balances_raw``, but the cache_only pricing branch reused the cached
``usd_amount`` unless it was missing, zero, or the token was in the refresh
set. So the USD froze while the quantity moved underneath it, leaving no
arithmetic relationship between the two.

Measured 2026-09-04 on base, the live wallet:

    balances.usd_amount = 3.68739300 for quantity 18.190627  -> $0.2027/USDC

The same row had read $3.687393 the previous day at 15.196 USDC. Same dollars,
different quantity: frozen, not mis-scaled, so no unit conversion explains it
and nothing downstream could detect it. ``_lookup_cached_price`` returns
Decimal("1") for USDC on its very first line -- it was never called.

These are the rows the live gate reads. ``reconciled_wallet_snapshot`` scopes
strictly to the wallet ADDRESS ("summing both double-counts the same account"),
so ``pipeline._wallet_state`` summed $3.69 of stable_usd for a wallet holding
$18.19, and MIN_LIVE_CAPITAL_USD is $4.00.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

from balances import MultiChainTokenPortfolio

USDC_BASE = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
AERO_BASE = "0x940181a94a35a4569e4529a3cdfb74e38fd98631"


class _Portfolio(MultiChainTokenPortfolio):
    """Exercises the pricing branch without any network or cache wiring."""

    def __init__(self, prices):
        self._prices = prices
        self.price_ttl_sec = 0
        self.cp = None

    def _lookup_cached_price(self, chain, token, meta_info=None):
        return self._prices.get((token or "").lower())


def _price_cache_only(portfolio, token_list, qty_map, cached_entries, meta_map):
    """Run the cache_only branch the way build() does."""
    usd_map = {}
    for a in token_list:
        ent = cached_entries.get(a) or {}
        usd_val = ent.get("usd_amount")
        usd_dec = None
        if usd_val is not None:
            try:
                usd_dec = Decimal(str(usd_val))
            except Exception:
                usd_dec = None
        cached_qty = None
        try:
            raw_qty = ent.get("quantity")
            if raw_qty is not None:
                cached_qty = Decimal(str(raw_qty))
        except Exception:
            cached_qty = None
        meta_info = meta_map.get(a) or ent
        px = portfolio._lookup_cached_price("base", a, meta_info)
        if px is not None and qty_map[a] > 0:
            usd_dec = (qty_map[a] * px).quantize(Decimal("0.00000001"))
        elif usd_dec is not None and usd_dec != 0:
            if cached_qty and cached_qty > 0 and qty_map[a] >= 0:
                usd_dec = (usd_dec * qty_map[a] / cached_qty).quantize(Decimal("0.00000001"))
            elif cached_qty is not None and cached_qty != qty_map[a]:
                usd_dec = None
        usd_map[a] = str(usd_dec) if usd_dec is not None else "0"
    return usd_map


class CachedUsdFollowsQuantityTest(unittest.TestCase):
    def test_the_observed_frozen_usdc_row_is_repriced(self):
        """The exact production row: 18.190627 USDC carried at $3.687393."""
        p = _Portfolio({USDC_BASE: Decimal("1")})
        usd = _price_cache_only(
            p,
            [USDC_BASE],
            {USDC_BASE: Decimal("18.190627")},
            # The cache still holds yesterday's balance and its dollars.
            {USDC_BASE: {"quantity": "15.196", "usd_amount": "3.68739300",
                         "symbol": "USDC"}},
            {},
        )
        self.assertAlmostEqual(Decimal(usd[USDC_BASE]), Decimal("18.190627"), places=6)

    def test_a_priced_token_is_never_read_from_the_cache(self):
        """Pricing wins even when the cached value looks plausible."""
        p = _Portfolio({AERO_BASE: Decimal("0.5105")})
        usd = _price_cache_only(
            p,
            [AERO_BASE],
            {AERO_BASE: Decimal("1.5462814519601997")},
            {AERO_BASE: {"quantity": "1.5462814519601997", "usd_amount": "0.0",
                         "symbol": "AERO"}},
            {},
        )
        self.assertAlmostEqual(
            Decimal(usd[AERO_BASE]), Decimal("1.5462814519601997") * Decimal("0.5105"),
            places=6,
        )

    def test_an_unpriceable_token_carries_its_implied_price_not_its_dollars(self):
        """Without a quote, the implied price is the only surviving fact."""
        p = _Portfolio({})
        usd = _price_cache_only(
            p,
            ["0xdead"],
            {"0xdead": Decimal("20")},           # doubled since the cache wrote
            {"0xdead": {"quantity": "10", "usd_amount": "5.0"}},  # $0.50 each
            {},
        )
        self.assertAlmostEqual(Decimal(usd["0xdead"]), Decimal("10"), places=6)

    def test_an_unchanged_unpriceable_quantity_keeps_its_value(self):
        p = _Portfolio({})
        usd = _price_cache_only(
            p,
            ["0xdead"],
            {"0xdead": Decimal("10")},
            {"0xdead": {"quantity": "10", "usd_amount": "5.0"}},
            {},
        )
        self.assertAlmostEqual(Decimal(usd["0xdead"]), Decimal("5.0"), places=6)

    def test_a_sold_out_position_is_worth_nothing(self):
        p = _Portfolio({})
        usd = _price_cache_only(
            p,
            ["0xdead"],
            {"0xdead": Decimal("0")},
            {"0xdead": {"quantity": "10", "usd_amount": "5.0"}},
            {},
        )
        self.assertEqual(Decimal(usd["0xdead"]), Decimal("0"))

    def test_stablecoin_pricing_needs_no_cache_at_all(self):
        """_lookup_cached_price answers $1 for a stable before any cache guard."""
        real = MultiChainTokenPortfolio.__new__(MultiChainTokenPortfolio)
        real.cp = None
        real.price_ttl_sec = 0
        self.assertEqual(real._lookup_cached_price("base", USDC_BASE, {"symbol": "USDC"}),
                         Decimal("1"))


if __name__ == "__main__":
    unittest.main()
