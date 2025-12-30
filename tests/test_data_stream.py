from __future__ import annotations

import asyncio
import json
import socket
import aiohttp
from types import SimpleNamespace
import statistics
import time

from aiohttp.client_reqrep import ConnectionKey

from trading.data_stream import Endpoint, MarketDataStream, RestFetchResult, _classify_network_error, _split_symbol


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


def _restrict_endpoints(stream: MarketDataStream, names: set[str]) -> None:
    stream.endpoints = [ep for ep in stream.endpoints if ep.name in names]
    assert stream.endpoints


def _force_endpoints(stream: MarketDataStream, endpoints: list[Endpoint]) -> None:
    stream.endpoints = endpoints
    assert stream.endpoints


def _dns_blocking_endpoint() -> Endpoint:
    return Endpoint(
        name="mirror",
        ws_template="wss://mirror.example.com/ws/{symbol}",
        subscribe_template=None,
        rest_template="https://mirror.example.com/api",
    )


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


def test_endpoint_network_failure_applies_backoff(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    target_endpoint = next(iter(stream._endpoint_scores))
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._record_endpoint_network_failure(target_endpoint, reason="dns")
    assert stream._endpoint_network_backoff_until[target_endpoint] > base_time
    eligible = {endpoint.name for endpoint in stream._eligible_rest_endpoints()}
    assert target_endpoint not in eligible


def test_endpoint_network_backoff_clears_on_success(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    target_endpoint = next(iter(stream._endpoint_scores))
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._record_endpoint_network_failure(target_endpoint, reason="network")
    stream._record_endpoint_success(target_endpoint, price=123.0, consensus=123.0)
    assert target_endpoint not in stream._endpoint_network_backoff_until
    assert stream._endpoint_network_failures.get(target_endpoint, 0) == 0


def test_websocket_backoff_sets_cooldown(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    stream._endpoint_backoff_until.clear()
    endpoint = stream.current_endpoint
    if endpoint not in stream._endpoint_scores:
        endpoint = next(iter(stream._endpoint_scores))
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._register_ws_backoff(reason="network", endpoint=endpoint, backoff=5.0)
    assert stream._endpoint_backoff_until[endpoint] >= base_time + stream._network_outage_base
    assert stream._network_outage_active(base_time) is False


def test_rest_only_mode_activates_after_threshold(monkeypatch):
    monkeypatch.setenv("REST_ONLY_WS_FAILURES", "2")
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._register_rest_only(reason="dns")
    assert stream._rest_only_mode is False
    stream._register_rest_only(reason="dns")
    assert stream._rest_only_mode is True
    assert stream._rest_only_reason == "dns"


def test_rest_only_mode_expires(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    stream._rest_only_mode = True
    stream._rest_only_last_check = base_time - stream._rest_only_retry - 1.0
    assert stream._rest_only_active(base_time) is False
    assert stream._rest_only_mode is False


def test_dns_ws_error_skips_rest_only_mode(monkeypatch):
    monkeypatch.setenv("REST_ONLY_WS_FAILURES", "1")
    stream = MarketDataStream(symbol="WETH-USDC")
    stream.url = "wss://example.com/ws"
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    key = ConnectionKey(
        host="stream.binance.com",
        port=9443,
        is_ssl=True,
        ssl=None,
        proxy=None,
        proxy_auth=None,
        proxy_headers_hash=None,
    )
    dns_error = aiohttp.ClientConnectorDNSError(
        key,
        socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
    )
    called = {"rest": 0, "refresh": 0}

    async def fake_poll(_: float) -> None:
        called["rest"] += 1
        stream._stop_event.set()

    async def fake_refresh() -> None:
        called["refresh"] += 1

    async def fake_ws() -> None:
        raise dns_error

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(stream, "_poll_rest_data", fake_poll)
    monkeypatch.setattr(stream, "_refresh_reference_price", fake_refresh)
    monkeypatch.setattr(stream, "_consume_ws", fake_ws)
    monkeypatch.setattr("trading.data_stream.asyncio.sleep", fast_sleep)

    async def runner() -> None:
        try:
            await stream.start()
        finally:
            await stream.stop()

    asyncio.run(runner())
    assert stream._rest_only_mode is False
    assert stream._rest_only_failures == 0
    assert called["rest"] >= 1


def test_classify_network_error_handles_dns_errno():
    exc = OSError(socket.EAI_AGAIN, "Temporary failure in name resolution")
    assert _classify_network_error(exc) == "dns"


def test_classify_network_error_handles_nested_dns_cause():
    try:
        raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
    except socket.gaierror as exc:
        try:
            raise RuntimeError("connector failure") from exc
        except RuntimeError as wrapped:
            assert _classify_network_error(wrapped) == "dns"


def test_classify_network_error_handles_dns_args():
    exc = RuntimeError("connector failure", socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"))
    assert _classify_network_error(exc) == "dns"


def test_classify_network_error_handles_aiohttp_dns():
    key = ConnectionKey(
        host="stream.binance.com",
        port=9443,
        is_ssl=True,
        ssl=None,
        proxy=None,
        proxy_auth=None,
        proxy_headers_hash=None,
    )
    exc = aiohttp.ClientConnectorDNSError(
        key,
        socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
    )
    assert _classify_network_error(exc) == "dns"


def test_classify_network_error_handles_exception_group():
    exc = ExceptionGroup(
        "connector failure",
        [socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")],
    )
    assert _classify_network_error(exc) == "dns"


def test_network_outage_blocks_rest_on_dns(monkeypatch):
    monkeypatch.setenv("NETWORK_OUTAGE_BLOCK_REST_ON_DNS", "1")
    stream = MarketDataStream(symbol="WETH-USDC")
    _force_endpoints(stream, [_dns_blocking_endpoint()])
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._register_network_outage(reason="dns", endpoint="mirror", source="websocket")
    assert stream._network_outage_block_rest is True
    assert stream._network_outage_blocks_rest(base_time) is True


def test_register_network_outage_reclassifies_dns(monkeypatch):
    monkeypatch.setenv("NETWORK_OUTAGE_BLOCK_REST_ON_DNS", "1")
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    key = ConnectionKey(
        host="stream.binance.com",
        port=9443,
        is_ssl=True,
        ssl=None,
        proxy=None,
        proxy_auth=None,
        proxy_headers_hash=None,
    )
    dns_error = aiohttp.ClientConnectorError(
        key,
        socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
    )
    stream._register_network_outage(
        reason="network",
        endpoint="binance",
        source="websocket",
        exc=dns_error,
    )
    assert stream._network_outage_reason == "dns"


def test_network_outage_unknown_ws_endpoint_allows_rest(monkeypatch):
    monkeypatch.setenv("NETWORK_OUTAGE_BLOCK_REST_ON_DNS", "1")
    stream = MarketDataStream(symbol="WETH-USDC")
    _force_endpoints(stream, [_dns_blocking_endpoint()])
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._register_network_outage(reason="dns", endpoint="bootstrap", source="websocket")
    assert stream._network_outage_block_rest is False
    assert stream._network_outage_blocks_rest(base_time) is False


def test_network_outage_unblocks_rest_after_non_dns(monkeypatch):
    monkeypatch.setenv("NETWORK_OUTAGE_BLOCK_REST_ON_DNS", "1")
    stream = MarketDataStream(symbol="WETH-USDC")
    _force_endpoints(stream, [_dns_blocking_endpoint()])
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._register_network_outage(reason="dns", endpoint="mirror", source="websocket")
    assert stream._network_outage_block_rest is True
    stream._register_network_outage(reason="network", endpoint="mirror", source="websocket")
    assert stream._network_outage_block_rest is False
    assert stream._network_outage_blocks_rest(base_time) is False


def test_network_outage_allows_rest_same_domain_subdomain(monkeypatch):
    monkeypatch.setenv("NETWORK_OUTAGE_BLOCK_REST_ON_DNS", "1")
    stream = MarketDataStream(symbol="WETH-USDC")
    _restrict_endpoints(stream, {"binance"})
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._register_network_outage(reason="dns", endpoint="binance", source="websocket")
    assert stream._network_outage_block_rest is False
    assert stream._network_outage_blocks_rest(base_time) is False


def test_network_outage_blocks_rest_same_domain_when_env_enabled(monkeypatch):
    monkeypatch.setenv("NETWORK_OUTAGE_BLOCK_REST_ON_DNS", "1")
    monkeypatch.setenv("NETWORK_OUTAGE_BLOCK_REST_SAME_DOMAIN", "1")
    stream = MarketDataStream(symbol="WETH-USDC")
    _restrict_endpoints(stream, {"binance"})
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._register_network_outage(reason="dns", endpoint="binance", source="websocket")
    assert stream._network_outage_block_rest is True
    assert stream._network_outage_blocks_rest(base_time) is True


def test_network_outage_allows_rest_with_alternate_domains(monkeypatch):
    monkeypatch.setenv("NETWORK_OUTAGE_BLOCK_REST_ON_DNS", "1")
    stream = MarketDataStream(symbol="WETH-USDC")
    _restrict_endpoints(stream, {"binance", "coinbase"})
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._register_network_outage(reason="dns", endpoint="binance", source="websocket")
    assert stream._network_outage_block_rest is False
    assert stream._network_outage_blocks_rest(base_time) is False


def test_dns_outage_skips_rest_endpoints_on_ws_domain(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    _restrict_endpoints(stream, {"binance"})
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._register_network_outage(reason="dns", endpoint="binance", source="websocket")
    eligible = {endpoint.name for endpoint in stream._eligible_rest_endpoints()}
    assert "binance" not in eligible


def test_websocket_outage_keeps_rest_fallback(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._network_outage_until = base_time + 60.0
    stream._network_outage_source = "websocket"
    stream._network_outage_reason = "dns"
    stream._network_outage_endpoint = "binance"
    called = {"rest": 0, "refresh": 0}

    async def fake_poll(duration: float) -> None:
        called["rest"] += 1
        stream._stop_event.set()

    async def fake_refresh() -> None:
        called["refresh"] += 1

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(stream, "_poll_rest_data", fake_poll)
    monkeypatch.setattr(stream, "_refresh_reference_price", fake_refresh)
    monkeypatch.setattr("trading.data_stream.asyncio.sleep", fast_sleep)

    async def runner() -> None:
        try:
            await stream.start()
        finally:
            await stream.stop()

    asyncio.run(runner())
    assert called["rest"] >= 1
    assert called["refresh"] >= 1


def test_rest_outage_allows_websocket_attempt(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._network_outage_until = base_time + 60.0
    stream._network_outage_source = "rest"
    stream._network_outage_reason = "dns"
    stream._network_outage_endpoint = "kucoin"
    stream.url = "wss://example.com/ws"
    called = {"rest": 0, "refresh": 0, "ws": 0}

    async def fake_poll(duration: float) -> None:
        called["rest"] += 1
        stream._stop_event.set()

    async def fake_refresh() -> None:
        called["refresh"] += 1

    async def fake_ws() -> None:
        called["ws"] += 1
        stream._stop_event.set()

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(stream, "_poll_rest_data", fake_poll)
    monkeypatch.setattr(stream, "_refresh_reference_price", fake_refresh)
    monkeypatch.setattr(stream, "_consume_ws", fake_ws)
    monkeypatch.setattr("trading.data_stream.asyncio.sleep", fast_sleep)

    async def runner() -> None:
        try:
            await stream.start()
        finally:
            await stream.stop()

    asyncio.run(runner())
    assert called["ws"] >= 1
    assert called["rest"] == 0


def test_rest_outage_does_not_override_websocket_outage(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._register_network_outage(reason="dns", endpoint="binance", source="websocket")
    expected_source = stream._network_outage_source
    expected_reason = stream._network_outage_reason
    expected_endpoint = stream._network_outage_endpoint
    expected_block_rest = stream._network_outage_block_rest
    expected_until = stream._network_outage_until

    stream._register_network_outage(reason="dns", endpoint="kucoin", source="rest")

    assert stream._network_outage_source == expected_source
    assert stream._network_outage_reason == expected_reason
    assert stream._network_outage_endpoint == expected_endpoint
    assert stream._network_outage_block_rest == expected_block_rest
    assert stream._network_outage_until >= expected_until


def test_rest_success_does_not_clear_websocket_outage(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._network_outage_until = base_time + 60.0
    stream._network_outage_failures = 2
    stream._network_outage_source = "websocket"
    stream._network_outage_reason = "dns"
    stream._network_outage_endpoint = "binance"
    endpoint = next(iter(stream._endpoint_scores))
    stream._record_endpoint_success(endpoint, price=100.0, consensus=100.0)
    assert stream._network_outage_until == base_time + 60.0
    assert stream._network_outage_failures == 2
    assert stream._network_outage_source == "websocket"


def test_rest_success_clears_rest_outage(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._network_outage_until = base_time + 60.0
    stream._network_outage_failures = 2
    stream._network_outage_source = "rest"
    endpoint = next(iter(stream._endpoint_scores))
    stream._record_endpoint_success(endpoint, price=100.0, consensus=100.0)
    assert stream._network_outage_until == 0.0
    assert stream._network_outage_failures == 0
    assert stream._network_outage_source is None


def test_refresh_reference_price_does_not_register_partial_outage(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    _restrict_endpoints(stream, {"binance", "coinbase"})
    stream._http_session = SimpleNamespace()

    async def fake_fetch(endpoint: Endpoint, base: str, quote: str) -> RestFetchResult:
        if endpoint.name == "binance":
            return RestFetchResult(None, "dns")
        return RestFetchResult(None, "invalid")

    monkeypatch.setattr(stream, "_fetch_rest_price", fake_fetch)
    asyncio.run(stream._refresh_reference_price())
    assert stream._rest_outage_failures == 0
    assert stream._network_outage_failures == 0


def test_refresh_reference_price_registers_outage_on_all_network_failures(monkeypatch):
    stream = MarketDataStream(symbol="WETH-USDC")
    _restrict_endpoints(stream, {"binance", "coinbase"})
    stream._http_session = SimpleNamespace()

    async def fake_fetch(endpoint: Endpoint, base: str, quote: str) -> RestFetchResult:
        if endpoint.name == "coinbase":
            return RestFetchResult(None, "dns")
        return RestFetchResult(None, "network")

    monkeypatch.setattr(stream, "_fetch_rest_price", fake_fetch)
    asyncio.run(stream._refresh_reference_price())
    assert stream._rest_outage_failures == 1
    assert stream._network_outage_failures == 1
    assert stream._network_outage_source == "rest"
    assert stream._network_outage_reason == "dns"


def test_refresh_reference_price_uses_snapshot_during_outage(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "market_snapshots.json"
    snapshot_path.write_text(
        json.dumps({"data": [{"symbol": "WETH-USDC", "price_usd": 2500.0}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCAL_MARKET_CACHE", str(snapshot_path))
    stream = MarketDataStream(symbol="WETH-USDC")
    stream._http_session = SimpleNamespace()
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._network_outage_until = base_time + 60.0
    stream._network_outage_source = "rest"
    stream._network_outage_reason = "dns"
    stream._network_outage_block_rest = True

    asyncio.run(stream._refresh_reference_price())

    assert stream.reference_price == 2500.0


def test_ws_disabled_env_uses_rest_fallback(monkeypatch):
    monkeypatch.setenv("MARKET_WEBSOCKET_DISABLED", "1")
    stream = MarketDataStream(symbol="WETH-USDC")
    assert stream._ws_disabled is True
    called = {"rest": 0, "refresh": 0, "ws": 0}

    async def fake_poll(duration: float) -> None:
        called["rest"] += 1
        stream._stop_event.set()

    async def fake_refresh() -> None:
        called["refresh"] += 1

    async def fake_ws() -> None:
        called["ws"] += 1
        stream._stop_event.set()

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(stream, "_poll_rest_data", fake_poll)
    monkeypatch.setattr(stream, "_refresh_reference_price", fake_refresh)
    monkeypatch.setattr(stream, "_consume_ws", fake_ws)
    monkeypatch.setattr("trading.data_stream.asyncio.sleep", fast_sleep)

    async def runner() -> None:
        try:
            await stream.start()
        finally:
            await stream.stop()

    asyncio.run(runner())
    assert called["rest"] >= 1
    assert called["refresh"] >= 1
    assert called["ws"] == 0


def test_rest_only_mode_uses_rest_fallback(monkeypatch):
    monkeypatch.setenv("REST_ONLY_WS_FAILURES", "1")
    stream = MarketDataStream(symbol="WETH-USDC")
    base_time = time.time()
    monkeypatch.setattr("trading.data_stream.time.time", lambda: base_time)
    stream._register_rest_only(reason="dns")
    assert stream._rest_only_mode is True
    called = {"rest": 0, "refresh": 0, "ws": 0}

    async def fake_poll(duration: float) -> None:
        called["rest"] += 1
        stream._stop_event.set()

    async def fake_refresh() -> None:
        called["refresh"] += 1

    async def fake_ws() -> None:
        called["ws"] += 1
        stream._stop_event.set()

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(stream, "_poll_rest_data", fake_poll)
    monkeypatch.setattr(stream, "_refresh_reference_price", fake_refresh)
    monkeypatch.setattr(stream, "_consume_ws", fake_ws)
    monkeypatch.setattr("trading.data_stream.asyncio.sleep", fast_sleep)

    async def runner() -> None:
        try:
            await stream.start()
        finally:
            await stream.stop()

    asyncio.run(runner())
    assert called["rest"] >= 1
    assert called["refresh"] >= 1
    assert called["ws"] == 0


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
