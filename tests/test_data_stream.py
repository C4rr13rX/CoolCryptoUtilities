from __future__ import annotations

import asyncio
from types import SimpleNamespace
import statistics
import time

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


def test_endpoint_failure_applies_backoff(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    stream._endpoint_backoff_until.clear()
    target_endpoint = next(iter(stream._endpoint_scores))
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._record_endpoint_failure(target_endpoint)
    assert target_endpoint in stream._endpoint_backoff_until
    assert stream._endpoint_backoff_until[target_endpoint] > base_time


def test_price_rejection_recenters_reference(monkeypatch):
    monkeypatch.setenv("PRICE_REJECTION_BUFFER", "4")
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    tick = {"value": 0}

    def fake_time():
        return base_time + tick["value"]

    monkeypatch.setattr("trading.data_stream.time.time", fake_time)
    stream.reference_price = 100.0
    stream._global_price_ema = 100.0
    stream._global_price_var = 0.0
    prices = [130.0, 129.6, 129.9, 130.1]
    accepted = False
    for idx, price in enumerate(prices):
        tick["value"] = idx * 5
        accepted = stream._validate_price(price)
    assert accepted is True
    assert abs(stream.reference_price - statistics.median(prices)) < 1e-6
