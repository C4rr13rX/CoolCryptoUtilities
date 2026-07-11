"""Regression tests for the stablecoin decimals bug.

Historical failure: a metadata miss defaulted decimals to 18, deflating
6-decimal stables (USDC/USDT) by 10^12 and poisoning the balance cache.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from token_decimals import known_token_decimals, KNOWN_TOKEN_DECIMALS


BASE_USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
ETH_USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"


class TestKnownTokenDecimals:
    def test_usdc_by_address_is_6(self):
        assert known_token_decimals("base", BASE_USDC) == 6
        assert known_token_decimals("ethereum", ETH_USDC) == 6

    def test_address_lookup_is_case_insensitive(self):
        assert known_token_decimals("Base", BASE_USDC.upper()) == 6

    def test_symbol_fallback_for_unknown_chain(self):
        # bridged deployment on a chain we don't map by address
        assert known_token_decimals("bsc", "0x" + "1" * 40, "USDC") == 6
        assert known_token_decimals("bsc", "0x" + "1" * 40, "usdt") == 6

    def test_dai_is_18_and_wbtc_is_8(self):
        assert known_token_decimals(None, None, "DAI") == 18
        assert known_token_decimals(None, None, "WBTC") == 8

    def test_unknown_token_returns_none_not_18(self):
        assert known_token_decimals("base", "0x" + "9" * 40, "RANDOMCOIN") is None

    def test_every_mapped_stable_is_not_18(self):
        for chain, tokens in KNOWN_TOKEN_DECIMALS.items():
            for addr, dec in tokens.items():
                assert 0 < dec <= 18, f"{chain}:{addr} has implausible decimals {dec}"


class TestHexToDecimal:
    def test_usdc_quantity_with_correct_decimals(self):
        from balances import MultiChainTokenPortfolio
        # 50.113501 USDC = 50113501 raw units at 6 decimals
        raw_hex = hex(50113501)
        qty = MultiChainTokenPortfolio._hex_to_decimal(raw_hex, 6)
        assert qty == Decimal("50.113501")

    def test_wrong_default_18_demonstrates_the_bug(self):
        from balances import MultiChainTokenPortfolio
        raw_hex = hex(50113501)
        qty = MultiChainTokenPortfolio._hex_to_decimal(raw_hex, 18)
        # the historical bug: 10^12 deflation
        assert qty == Decimal("50.113501") / Decimal(10) ** 12


class TestCacheUpsertDecimals:
    @pytest.fixture()
    def cache(self, tmp_path):
        from db import TradingDatabase
        from cache import CacheBalances
        db = TradingDatabase(path=str(tmp_path / "test.db"))
        return CacheBalances(db=db)

    def test_missing_decimals_resolves_known_stable(self, cache):
        cache.upsert_many(
            "0xwallet", "base",
            {BASE_USDC: {"balance_hex": hex(50113501), "symbol": "USDC",
                         "quantity": "50.113501", "usd_amount": "50.11"}},
        )
        ent = cache.get_token("0xwallet", "base", BASE_USDC)
        assert ent["decimals"] == 6

    def test_missing_decimals_unknown_token_defaults_18(self, cache):
        addr = "0x" + "9" * 40
        cache.upsert_many(
            "0xwallet", "base",
            {addr: {"balance_hex": "0x1", "symbol": "MYSTERY", "quantity": "1"}},
        )
        ent = cache.get_token("0xwallet", "base", addr)
        assert ent["decimals"] == 18

    def test_explicit_decimals_preserved(self, cache):
        addr = "0x" + "8" * 40
        cache.upsert_many(
            "0xwallet", "base",
            {addr: {"balance_hex": "0x1", "decimals": 9, "quantity": "1"}},
        )
        ent = cache.get_token("0xwallet", "base", addr)
        assert ent["decimals"] == 9
