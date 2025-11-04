from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
import statistics
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import aiohttp

from db import get_db


CallbackType = Union[Callable[[Dict[str, Any]], Awaitable[None]], Callable[[Dict[str, Any]], None]]


TOKEN_NORMALIZATION = {
    "WETH": "ETH",
    "WBTC": "BTC",
    "WBNB": "BNB",
    "MOG": "MOG",
    "USDC": "USDC",
    "USDT": "USDT",
    "DAI": "DAI",
    "SPX": "SPX",
}

COINGECKO_IDS = {
    "ETH": "ethereum",
    "BTC": "bitcoin",
    "BNB": "binancecoin",
    "USDC": "usd-coin",
    "USDT": "tether",
    "DAI": "dai",
    "SPX": "spx6900",
}


@dataclass
class Endpoint:
    name: str
    ws_template: Optional[str]
    subscribe_template: Optional[str]
    rest_template: Optional[str]
    headers: Optional[Dict[str, str]] = None



class MarketDataStream:
    """
    Handles live websocket subscriptions (when available) or falls back to a
    lightweight simulation so downstream systems always receive data.
    Each sample is persisted to the shared TradingDatabase for later training.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        symbol: str = "ETH-USD",
        chain: str = "ethereum",
        simulation_interval: float = 2.0,
        ws_template: Optional[str] = None,
        subscribe_template: Optional[str] = None,
    ) -> None:
        template = ws_template or os.getenv("MARKET_WS_TEMPLATE") or os.getenv("UNISWAP_WS_TEMPLATE")
        symbol_lower = symbol.lower().replace("/", "-")
        self._template = template
        self.url = url or (template.format(symbol=symbol_lower, SYMBOL=symbol.upper(), pair=symbol_lower) if template else None)
        self.subscribe_template = subscribe_template or os.getenv("MARKET_WS_SUBSCRIBE") or os.getenv(
            "UNISWAP_WS_SUBSCRIBE"
        )
        self.symbol = symbol
        self.chain = chain
        self.simulation_interval = simulation_interval
        self._callbacks: List[CallbackType] = []
        self._db = get_db()
        self._session: Optional[aiohttp.ClientSession] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._stop_event = asyncio.Event()
        self.reference_price: Optional[float] = None
        self.price_tolerance = float(os.getenv("PRICE_FEED_TOLERANCE", "0.05"))
        self.endpoints = self._build_endpoints()
        self._endpoint_scores: Dict[str, float] = {ep.name: 0.0 for ep in self.endpoints}
        self._endpoint_order = {ep.name: idx for idx, ep in enumerate(self.endpoints)}
        if not self.url:
            self._select_next_endpoint()
        self.rest_poll_interval = float(os.getenv("REST_POLL_INTERVAL", "5"))
        self._last_volume: float = 0.0
        self.consensus_sources = int(os.getenv("PRICE_CONSENSUS_SOURCES", "2"))
        self.consensus_window = float(os.getenv("PRICE_CONSENSUS_WINDOW", "60"))
        self.consensus_timeout = float(os.getenv("PRICE_CONSENSUS_TIMEOUT", "45"))
        self._recent_price_by_source: Dict[str, Tuple[float, float]] = {}
        self._last_consensus_ts = 0.0
        self._endpoint_failures: Dict[str, int] = {}

    def register(self, callback: CallbackType) -> None:
        self._callbacks.append(callback)

    async def start(self) -> None:
        self._stop_event.clear()
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        await self._refresh_reference_price()
        backoff = 5.0
        while not self._stop_event.is_set():
            if not self.url:
                print("[market-stream] websocket URL unavailable; waiting for configuration.")
                await asyncio.sleep(30.0)
                if self._template:
                    symbol_lower = self.symbol.lower().replace("/", "-")
                    self.url = self._template.format(symbol=symbol_lower, SYMBOL=self.symbol.upper(), pair=symbol_lower)
                if not self.url:
                    self._select_next_endpoint()
                continue
            try:
                await self._consume_ws()
                backoff = 5.0
            except aiohttp.WSServerHandshakeError as exc:
                if exc.status in (429, 503):
                    print(f"[market-stream] rate limited ({exc.status}); sleeping {backoff:.1f}s")
                else:
                    print(f"[market-stream] websocket handshake failed ({exc.status}); retrying in {backoff:.1f}s")
                await self._poll_rest_data(backoff)
                backoff = min(backoff * 1.5, 300.0)
                await self._refresh_reference_price()
                self._select_next_endpoint()
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"[market-stream] websocket error {exc}; retrying in {backoff:.1f}s")
                await self._poll_rest_data(backoff)
                backoff = min(backoff * 1.5, 300.0)
                await self._refresh_reference_price()
                self._select_next_endpoint()

    async def stop(self) -> None:
        self._stop_event.set()
        if self._session:
            await self._session.close()
            self._session = None
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    async def _consume_ws(self) -> None:  # pragma: no cover - requires network
        async with aiohttp.ClientSession() as session:
            self._session = session
            async with session.ws_connect(self.url) as ws:
                subscribe_msg = self._build_subscribe_payload(self.symbol)
                if subscribe_msg:
                    await ws.send_str(subscribe_msg)
                while not self._stop_event.is_set():
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        sample = self._normalize_payload(data)
                        if sample is None:
                            continue
                        await self._dispatch(sample)
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        if msg.data and isinstance(msg.data, Exception):
                            raise msg.data
                        break
                    elif msg.type == aiohttp.WSMsgType.TEXT and "rate" in (msg.data or "").lower():
                        print("[market-stream] rate-limit notice from server; backing off 15s")
                        await asyncio.sleep(15.0)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

    async def _poll_rest_data(self, duration: float) -> None:
        if self._http_session is None:
            return
        base, quote = _split_symbol(self.symbol)
        end_time = time.time() + max(duration, self.rest_poll_interval)
        while not self._stop_event.is_set() and time.time() < end_time:
            dispatched = False
            for endpoint in self._ranked_endpoints():
                price = await self._fetch_rest_price(endpoint, base, quote)
                if not price or price <= 0:
                    self._record_endpoint_failure(endpoint.name)
                    continue
                accepted, pending, consensus = self._confirm_consensus(endpoint.name, price)
                if pending:
                    continue
                if not accepted or consensus is None:
                    self._record_endpoint_failure(endpoint.name)
                    continue
                if not self._validate_price(consensus):
                    self._record_endpoint_failure(endpoint.name)
                    continue
                self._record_endpoint_success(endpoint.name)
                sample = {
                    "ts": time.time(),
                    "symbol": self.symbol,
                    "chain": self.chain,
                    "price": consensus,
                    "volume": self._last_volume,
                    "rest": endpoint.name,
                }
                await self._dispatch(sample)
                dispatched = True
                break
            if not dispatched:
                print("[market-stream] unable to obtain consensus price during REST poll cycle.")
            await asyncio.sleep(self.rest_poll_interval)

    async def _dispatch(self, sample: Dict[str, Any]) -> None:
        try:
            self._last_volume = float(sample.get("volume") or self._last_volume or 0.0)
        except Exception:
            pass
        self._db.insert_market_sample(
            chain=sample.get("chain", self.chain),
            symbol=sample.get("symbol", self.symbol),
            price=float(sample.get("price") or 0),
            volume=float(sample.get("volume") or 0),
            raw=sample,
        )
        for callback in list(self._callbacks):
            try:
                if asyncio.iscoroutinefunction(callback):  # type: ignore[arg-type]
                    await callback(sample)  # type: ignore[misc]
                else:
                    callback(sample)  # type: ignore[misc]
            except Exception as exc:
                print(f"[market-stream] callback error: {exc}")

    def _build_subscribe_payload(self, symbol: str) -> Optional[str]:
        if not self.subscribe_template:
            return None
        if self.subscribe_template == "COINBASE_DYNAMIC":
            base, quote = _split_symbol(symbol)
            base_cb = _to_coinbase(base)
            quote_cb = _to_coinbase_quote(quote)
            if not base_cb or not quote_cb:
                return None
            return json.dumps(
                {
                    "type": "subscribe",
                    "product_ids": [f"{base_cb}-{quote_cb}"],
                    "channels": ["ticker"],
                }
            )
        symbol_lower = symbol.lower().replace("/", "-")
        payload = self.subscribe_template.format(symbol=symbol_lower, SYMBOL=symbol.upper(), pair=symbol_lower)
        return payload

    def _normalize_payload(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Uniswap-style streaming payloads often have nested "data" nodes
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        price = float(payload.get("price") or payload.get("p") or 0)
        volume = float(payload.get("volume") or payload.get("v") or 0)
        if price == 0 and isinstance(data, dict):
            price = float(data.get("price") or data.get("priceUsd") or data.get("priceUSD") or 0)
        if volume == 0 and isinstance(data, dict):
            volume = float(data.get("volume") or data.get("amount") or data.get("amountUSD") or 0)
        symbol = (
            payload.get("symbol")
            or payload.get("s")
            or (data.get("symbol") if isinstance(data, dict) else None)
            or self.symbol
        )
        ts = float(payload.get("ts") or payload.get("time") or (data.get("timestamp") if isinstance(data, dict) else 0))
        if ts == 0:
            ts = time.time()
        if price <= 0:
            return None
        if not self._validate_price(price):
            return None
        return {
            "ts": ts,
            "symbol": symbol,
            "chain": self.chain,
            "price": price,
            "volume": volume,
            "raw": payload,
        }

    def _validate_price(self, price: float) -> bool:
        if self.reference_price is None:
            self.reference_price = price
            return True
        if price <= 0:
            return False
        diff = abs(price - self.reference_price) / max(self.reference_price, 1e-9)
        if diff <= self.price_tolerance:
            self.reference_price = 0.9 * self.reference_price + 0.1 * price
            return True
        print(
            f"[market-stream] price {price:.6f} deviates from reference {self.reference_price:.6f}; waiting for confirmation"
        )
        return False

    def _build_endpoints(self) -> List[Endpoint]:
        base, quote = _split_symbol(self.symbol)
        endpoints: List[Endpoint] = []
        binance_symbol = _binance_symbol(base, quote)
        if binance_symbol:
            endpoints.append(
                Endpoint(
                    name="binance",
                    ws_template="wss://stream.binance.com:9443/ws/{symbol}@trade",
                    subscribe_template=None,
                    rest_template="https://api.binance.com/api/v3/ticker/24hr?symbol={symbol_upper}",
                )
            )
        coinbase_base = _to_coinbase(base)
        coinbase_quote = _to_coinbase_quote(quote)
        if coinbase_base and coinbase_quote:
            endpoints.append(
                Endpoint(
                    name="coinbase",
                    ws_template="wss://ws-feed.exchange.coinbase.com",
                    subscribe_template="COINBASE_DYNAMIC",
                    rest_template=f"https://api.exchange.coinbase.com/products/{coinbase_base}-{coinbase_quote}/ticker",
                )
            )
        coingecko_id = COINGECKO_IDS.get(base)
        if coingecko_id:
            endpoints.append(
                Endpoint(
                    name="coingecko",
                    ws_template=None,
                    subscribe_template=None,
                    rest_template=f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies={quote.lower()}",
                )
            )
        bitstamp_symbol = f"{base.lower()}{quote.lower()}"
        if quote in {"USD", "USDT", "EUR"}:
            endpoints.append(
                Endpoint(
                    name="bitstamp",
                    ws_template=None,
                    subscribe_template=None,
                    rest_template=f"https://www.bitstamp.net/api/v2/ticker/{bitstamp_symbol}/",
                )
            )
        okx_quote = quote.upper()
        if okx_quote in {"USDT", "USDC", "USD"}:
            endpoints.append(
                Endpoint(
                    name="okx",
                    ws_template=None,
                    subscribe_template=None,
                    rest_template=f"https://www.okx.com/api/v5/market/ticker?instId={base.upper()}-{okx_quote}",
                )
            )
        kucoin_quote = quote.upper()
        if kucoin_quote in {"USDT", "USDC", "BTC", "ETH"}:
            endpoints.append(
                Endpoint(
                    name="kucoin",
                    ws_template=None,
                    subscribe_template=None,
                    rest_template=f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={base.upper()}-{kucoin_quote}",
                )
            )
        mexc_symbol = f"{base.upper()}{quote.upper()}"
        endpoints.append(
            Endpoint(
                name="mexc",
                ws_template=None,
                subscribe_template=None,
                rest_template=f"https://api.mexc.com/api/v3/ticker/price?symbol={mexc_symbol}",
            )
        )
        return endpoints

    def _ranked_endpoints(self) -> List[Endpoint]:
        if not self.endpoints:
            return []
        return sorted(
            self.endpoints,
            key=lambda ep: (
                -self._endpoint_scores.get(ep.name, 0.0),
                self._endpoint_order.get(ep.name, 0),
            ),
        )

    def _record_endpoint_success(self, name: str) -> None:
        self._endpoint_scores[name] = self._endpoint_scores.get(name, 0.0) + 1.0
        self._endpoint_failures[name] = 0

    def _record_endpoint_failure(self, name: str) -> None:
        self._endpoint_scores[name] = self._endpoint_scores.get(name, 0.0) - 2.0
        self._endpoint_failures[name] = self._endpoint_failures.get(name, 0) + 1
        if self._endpoint_failures[name] in (3, 10):
            print(f"[market-stream] endpoint {name} failing consensus check ({self._endpoint_failures[name]} strikes).")

    def _confirm_consensus(self, name: str, price: float) -> Tuple[bool, bool, Optional[float]]:
        now = time.time()
        self._recent_price_by_source[name] = (price, now)
        valid = [
            value
            for value, ts in self._recent_price_by_source.values()
            if now - ts <= self.consensus_window
        ]
        if not valid:
            return False, True, None
        min_required = min(self.consensus_sources, max(1, len(self.endpoints)))
        median_price = statistics.median(valid)
        diff = abs(price - median_price) / max(median_price, 1e-9)
        if len(valid) < min_required:
            return False, True, median_price
        if diff <= self.price_tolerance:
            self._last_consensus_ts = now
            return True, False, median_price
        if now - self._last_consensus_ts >= self.consensus_timeout and diff <= self.price_tolerance * 2:
            # Fallback: accept slightly wider band if consensus overdue
            self._last_consensus_ts = now
            return True, False, median_price
        return False, False, None

    async def _refresh_reference_price(self) -> None:
        if self._http_session is None:
            return
        base, quote = _split_symbol(self.symbol)
        for endpoint in self._ranked_endpoints():
            price = await self._fetch_rest_price(endpoint, base, quote)
            if not price or price <= 0:
                self._record_endpoint_failure(endpoint.name)
                continue
            accepted, pending, consensus = self._confirm_consensus(endpoint.name, price)
            if pending:
                continue
            if not accepted or consensus is None:
                self._record_endpoint_failure(endpoint.name)
                continue
            if not self._validate_price(consensus):
                self._record_endpoint_failure(endpoint.name)
                continue
            self._record_endpoint_success(endpoint.name)
            self.reference_price = consensus
            print(f"[market-stream] reference price from {endpoint.name}: {consensus:.6f}")
            return
        if self.reference_price is None:
            print("[market-stream] unable to establish reference price; will retry")
        else:
            self.reference_price = max(self.reference_price, 1e-9)

    async def _fetch_rest_price(self, endpoint: Endpoint, base: str, quote: str) -> Optional[float]:
        if self._http_session is None or not endpoint.rest_template:
            return None
        url = _render_rest(endpoint, base, quote)
        if not url:
            return None
        headers = endpoint.headers or {}
        try:
            async with self._http_session.get(url, timeout=10, headers=headers) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
        except Exception:
            return None
        return _extract_rest_price(endpoint.name, data, base, quote)

    def _select_next_endpoint(self) -> None:
        if not self.endpoints:
            return
        base, quote = _split_symbol(self.symbol)
        candidates = self._ranked_endpoints()
        for ep in candidates:
            ws_url = _render_ws(ep, base, quote)
            if ws_url:
                self.url = ws_url
                self.subscribe_template = ep.subscribe_template
                print(f"[market-stream] using {ep.name} endpoint -> {self.url}")
                return
        self.url = None


def _split_symbol(symbol: str) -> Tuple[str, str]:
    if "-" in symbol:
        base, quote = symbol.split("-", 1)
    elif "/" in symbol:
        base, quote = symbol.split("/", 1)
    else:
        base = symbol[:-3]
        quote = symbol[-3:]
    base = TOKEN_NORMALIZATION.get(base.upper(), base.upper())
    quote = TOKEN_NORMALIZATION.get(quote.upper(), quote.upper())
    return base, quote


def _binance_symbol(base: str, quote: str) -> Optional[str]:
    quote_map = {"USD": "USDT", "USDC": "USDC", "USDT": "USDT", "BUSD": "BUSD"}
    mapped_quote = quote_map.get(quote, quote)
    if mapped_quote not in {"USDT", "USDC", "BUSD", "BTC", "ETH", "BNB"}:
        return None
    token_map = {"ETH": "ETH", "BTC": "BTC", "BNB": "BNB", "MOG": "MOG"}
    mapped_base = token_map.get(base, base)
    symbol = f"{mapped_base}{mapped_quote}".lower()
    return symbol


def _to_coinbase(token: str) -> Optional[str]:
    mapping = {
        "ETH": "ETH",
        "BTC": "BTC",
        "USDT": "USDT",
        "USDC": "USDC",
        "DAI": "DAI",
    }
    return mapping.get(token)


def _to_coinbase_quote(token: str) -> Optional[str]:
    mapping = {
        "USD": "USD",
        "USDT": "USDT",
        "USDC": "USDC",
        "EUR": "EUR",
    }
    return mapping.get(token)


def _render_rest(endpoint: Endpoint, base: str, quote: str) -> Optional[str]:
    if endpoint.name == "binance":
        symbol = _binance_symbol(base, quote)
        if not symbol:
            return None
        return endpoint.rest_template.format(symbol_upper=symbol.upper(), symbol=symbol)
    if endpoint.name == "coinbase":
        base_cb = _to_coinbase(base)
        quote_cb = _to_coinbase_quote(quote)
        if not base_cb or not quote_cb:
            return None
        return endpoint.rest_template
    if endpoint.name == "coingecko":
        return endpoint.rest_template
    if endpoint.name in {"bitstamp", "mexc"}:
        return endpoint.rest_template
    if endpoint.name == "okx":
        return endpoint.rest_template
    if endpoint.name == "kucoin":
        return endpoint.rest_template
    return endpoint.rest_template


def _render_ws(endpoint: Endpoint, base: str, quote: str) -> Optional[str]:
    if not endpoint.ws_template:
        return None
    if endpoint.name == "binance":
        symbol = _binance_symbol(base, quote)
        return endpoint.ws_template.format(symbol=symbol) if symbol else None
    if endpoint.name == "coinbase":
        return endpoint.ws_template
    if endpoint.name == "coingecko":
        return None
    return endpoint.ws_template


def _extract_rest_price(name: str, payload: Dict[str, Any], base: str, quote: str) -> Optional[float]:
    try:
        if name == "binance":
            price = float(payload.get("lastPrice") or payload.get("weightedAvgPrice") or 0)
            return price if price > 0 else None
        if name == "coinbase":
            price = float(payload.get("price") or payload.get("ask") or 0)
            return price if price > 0 else None
        if name == "coingecko":
            cg_id = COINGECKO_IDS.get(base)
            if not cg_id:
                return None
            vs = quote.lower()
            return float(payload.get(cg_id, {}).get(vs) or 0)
        if name == "bitstamp":
            price = float(payload.get("last") or payload.get("ask") or 0)
            return price if price > 0 else None
        if name == "okx":
            items = payload.get("data") or []
            if items:
                price = float(items[0].get("last", 0) or items[0].get("lastSz", 0))
                return price if price > 0 else None
            return None
        if name == "kucoin":
            data = payload.get("data") or {}
            price = float(data.get("price") or data.get("bestAsk") or 0)
            return price if price > 0 else None
        if name == "mexc":
            price = float(payload.get("price") or 0)
            return price if price > 0 else None
    except Exception:
        return None
    return None
