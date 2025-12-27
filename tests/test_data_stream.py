from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
import statistics
import time

from trading.data_stream import MarketDataStream, _split_symbol


class DummyOfflineStore:
    def __init__(self, mapping):
        self.mapping = {key.upper(): value for key, value in mapping.items()}

    def get_price(self, symbol: str):
        return self.mapping.get(symbol.upper())

    def snapshots(self, *, symbols=None, limit=25):
        records = []
        wanted = {sym.upper() for sym in symbols} if symbols else None
        for key, value in self.mapping.items():
            if wanted and key not in wanted:
                continue
            if hasattr(value, "__dict__"):
                payload = dict(value.__dict__)
            else:
                payload = {"price": getattr(value, "price", None)}
            payload.setdefault("symbol", key)
            records.append(SimpleNamespace(**payload))
            if len(records) >= limit:
                break
        return records


def test_split_symbol_normalizes_usdbc():
    base, quote = _split_symbol("WETH-USDbC")
    assert base == "ETH"
    assert quote == "USDC"


def test_rest_pair_prefers_market_order_for_stable_base():
    stream = MarketDataStream(symbol="USDC-WETH")
    assert stream._rest_base == "ETH"
    assert stream._rest_quote == "USDC"
    assert stream._rest_inverted is True


def test_market_stream_fallback_uses_base_alias(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDBC")
    stream._offline_enabled = True
    snapshot = SimpleNamespace(price=1234.5, ts=1700000000.0, volume=42.0, source="stub")
    stream._offline_store = DummyOfflineStore({"ETH": snapshot})
    stream._recent_price_by_source.clear()
    stream.reference_price = None
    price = stream._fallback_consensus_price()
    assert price == 1234.5


def test_fallback_uses_recent_live_source(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    stream._offline_enabled = False
    stream._offline_store = None
    stream._recent_price_by_source.clear()
    stream.reference_price = None
    base_time = time.time()
    stream._last_emitted_by_source = {"onchain": (1234.0, base_time - 10.0)}
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    price = stream._fallback_consensus_price()
    assert price == 1234.0
    assert stream._recent_price_by_source.get("onchain") == (1234.0, base_time - 10.0)


def test_market_stream_offline_failover_sample_includes_alias():
    stream = MarketDataStream(symbol="WETH-USDBC")
    stream._offline_enabled = True
    snapshot = SimpleNamespace(price=321.0, ts=1700001111.0, volume=11.0, source="stub")
    stream._offline_store = DummyOfflineStore({"WETH": snapshot})
    sample = asyncio.run(stream._offline_failover_sample())
    assert sample is not None
    assert sample["price"] == snapshot.price
    assert sample["source"] in {snapshot.source, f"{snapshot.source}+stable_assumed"}
    assert sample.get("alias") in {"WETH", "ETH", "ETH-USDC", "USDC-ETH"}


def test_offline_snapshot_accepts_reversed_pair():
    stream = MarketDataStream(symbol="WETH-USDBC")
    stream._offline_enabled = True
    snapshot = SimpleNamespace(price=2000.0, ts=1700002222.0, volume=3.0, source="stub")
    stream._offline_store = DummyOfflineStore({"USDC-WETH": snapshot})
    offline = stream._offline_snapshot()
    assert offline is not None
    alias, snap = offline
    assert alias == "ETH-USDC"
    assert snap.price == snapshot.price


def test_offline_snapshot_inverts_stable_base():
    stream = MarketDataStream(symbol="USDC-WETH")
    stream._offline_enabled = True
    snapshot = SimpleNamespace(price=2000.0, ts=1700003333.0, volume=1.0, source="stub")
    stream._offline_store = DummyOfflineStore({"USDC-WETH": snapshot})
    offline = stream._offline_snapshot()
    assert offline is not None
    alias, snap = offline
    assert alias == "USDC-ETH"
    assert abs(snap.price - 0.0005) < 1e-12


def test_market_stream_fallback_pair_ratio():
    stream = MarketDataStream(symbol="USDC-WETH")
    stream._offline_enabled = True
    base_snapshot = SimpleNamespace(
        price=1.0,
        ts=1700000000.0,
        volume=10.0,
        source="stub",
        symbol="USDC",
        name="USD Coin",
    )
    quote_snapshot = SimpleNamespace(
        price=2000.0,
        ts=1700000200.0,
        volume=5.0,
        source="stub",
        symbol="ETH",
        name="Ethereum",
    )
    stream._offline_store = DummyOfflineStore({"USDC": base_snapshot, "ETH": quote_snapshot})
    stream._recent_price_by_source.clear()
    stream.reference_price = None
    price = stream._fallback_consensus_price()
    assert abs(price - 0.0005) < 1e-9


def test_fallback_uses_local_snapshot_when_offline_disabled(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "market_snapshots.json"
    snapshot_path.write_text(
        json.dumps({"data": [{"symbol": "USDC-WETH", "price_usd": 2000.0}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCAL_MARKET_CACHE", str(snapshot_path))
    stream = MarketDataStream(symbol="WETH-USDBC")
    stream._offline_enabled = False
    stream._offline_store = None
    stream._recent_price_by_source.clear()
    stream.reference_price = None
    price = stream._fallback_consensus_price()
    assert price == 2000.0


def test_fallback_uses_snapshot_tokens_for_pair(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "market_snapshots.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "data": [
                    {"symbol": "ETH", "price_usd": 2500.0},
                    {"symbol": "USDC", "price_usd": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCAL_MARKET_CACHE", str(snapshot_path))
    stream = MarketDataStream(symbol="WETH-USDBC")
    stream._offline_enabled = False
    stream._offline_store = None
    stream._recent_price_by_source.clear()
    stream.reference_price = None
    price = stream._fallback_consensus_price()
    assert price is not None
    assert abs(price - 2500.0) < 1e-9


def test_consensus_accepts_single_live_source_with_stale_reference(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream.reference_price = 100.0
    stream._recent_price_by_source = {"reference": (100.0, base_time)}
    accepted, pending, consensus, confidence = stream._confirm_consensus("coingecko", 250.0)
    assert accepted is True
    assert pending is False
    assert confidence > 0.0
    assert consensus == 250.0


def test_accept_consensus_single_source_override_when_stale(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream.reference_price = 100.0
    stream._global_price_ema = 100.0
    stream._global_price_var = 0.0
    stream._last_consensus_ts = base_time - stream.consensus_timeout - 1.0
    stream._consensus_relax_until = 0.0
    accepted = stream._accept_consensus_price(250.0, confidence=0.8, live_sources=1)
    assert accepted is True
    assert stream.reference_price == 250.0


def test_accept_consensus_single_source_override_bootstrap(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream.reference_price = 100.0
    stream._global_price_ema = 100.0
    stream._global_price_var = 0.0
    stream._last_consensus_ts = base_time
    stream._consensus_relax_until = 0.0
    stream._consensus_initialized = False
    accepted = stream._accept_consensus_price(250.0, confidence=0.8, live_sources=1)
    assert accepted is True
    assert stream.reference_price == 250.0


def test_endpoint_failure_applies_backoff(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    stream._endpoint_backoff_until.clear()
    target_endpoint = next(iter(stream._endpoint_scores))
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._record_endpoint_failure(target_endpoint)
    assert target_endpoint in stream._endpoint_backoff_until
    assert stream._endpoint_backoff_until[target_endpoint] > base_time


def test_cooldown_sleep_interval_waits_for_backoff(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._endpoint_backoff_until = {
        "alpha": base_time + 12.0,
        "beta": base_time + 5.0,
    }
    sleep_for = stream._cooldown_sleep_interval()
    assert abs(sleep_for - 5.0) < 1e-6


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


def test_reference_override_accepts_multi_source_consensus(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream.reference_price = 100.0
    stream._global_price_ema = 100.0
    stream._global_price_var = 0.0
    stream._recent_price_by_source = {
        "binance": (250.0, base_time),
        "coingecko": (252.0, base_time),
    }
    live_sources = stream._live_source_count()
    assert live_sources == 2
    accepted = stream._accept_consensus_price(251.0, confidence=0.8, live_sources=live_sources)
    assert accepted is True
    assert stream.reference_price == 251.0
