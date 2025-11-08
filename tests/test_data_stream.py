from __future__ import annotations

import asyncio
from types import SimpleNamespace

from trading.data_stream import MarketDataStream, _split_symbol


class DummyOfflineStore:
    def __init__(self, mapping):
        self.mapping = {key.upper(): value for key, value in mapping.items()}

    def get_price(self, symbol: str):
        return self.mapping.get(symbol.upper())


def test_split_symbol_normalizes_usdbc():
    base, quote = _split_symbol("WETH-USDbC")
    assert base == "ETH"
    assert quote == "USDC"


def test_market_stream_fallback_uses_base_alias(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDBC")
    stream._offline_enabled = True
    snapshot = SimpleNamespace(price=1234.5, ts=1700000000.0, volume=42.0, source="stub")
    stream._offline_store = DummyOfflineStore({"ETH": snapshot})
    stream._recent_price_by_source.clear()
    stream.reference_price = None
    price = stream._fallback_consensus_price()
    assert price == 1234.5


def test_market_stream_offline_failover_sample_includes_alias():
    stream = MarketDataStream(symbol="WETH-USDBC")
    stream._offline_enabled = True
    snapshot = SimpleNamespace(price=321.0, ts=1700001111.0, volume=11.0, source="stub")
    stream._offline_store = DummyOfflineStore({"WETH": snapshot})
    sample = asyncio.run(stream._offline_failover_sample())
    assert sample is not None
    assert sample["price"] == snapshot.price
    assert sample["source"] == snapshot.source
    assert sample.get("alias") in {"WETH", "ETH", "ETH-USDC", "USDC-ETH"}
