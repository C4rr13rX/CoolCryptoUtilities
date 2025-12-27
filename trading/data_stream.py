from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
import statistics
from urllib.parse import quote_plus
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union, Set, Sequence
from collections import deque

import aiohttp

from services.adaptive_control import APIRateLimiter
from services.logging_utils import log_message
from services.offline_market import OfflinePriceStore, OfflineSnapshot

try:  # pragma: no cover - optional dependency may fail at import time
    from services.onchain_feed import OnChainPairFeed
except Exception:  # pragma: no cover - graceful fallback
    OnChainPairFeed = None  # type: ignore

from db import get_db
from trading.metrics import FeedbackSeverity, MetricStage, MetricsCollector
from trading.constants import PRIMARY_CHAIN, PRIMARY_SYMBOL


CallbackType = Union[Callable[[Dict[str, Any]], Awaitable[None]], Callable[[Dict[str, Any]], None]]


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0
    except ValueError:
        try:
            return float(str(value))
        except Exception:
            return 0.0


TOKEN_NORMALIZATION = {
    "WETH": "ETH",
    "WBTC": "BTC",
    "WBNB": "BNB",
    "MOG": "MOG",
    "USDC": "USDC",
    "USDT": "USDT",
    "DAI": "DAI",
    "SPX": "SPX",
    "USDBC": "USDC",
    "USDCE": "USDC",
    "USDC.E": "USDC",
}

_STABLE_TOKEN_PRICE = {
    "USD": 1.0,
    "USDC": 1.0,
    "USDC.E": 1.0,
    "USDBC": 1.0,
    "USDCE": 1.0,
    "USDT": 1.0,
    "DAI": 1.0,
    "BUSD": 1.0,
}

_HIGH_VALUE_TOKENS = {
    "BTC",
    "ETH",
    "WETH",
    "WBTC",
    "WSTETH",
    "STETH",
    "WBETH",
    "BNB",
}

COINGECKO_IDS = {
    "ETH": "ethereum",
    "BTC": "bitcoin",
    "BNB": "binancecoin",
    "WBTC": "wrapped-bitcoin",
    "USDC": "usd-coin",
    "USDT": "tether",
    "DAI": "dai",
    "SPX": "spx6900",
    "UNI": "uniswap",
    "PEPE": "pepe",
}

BINANCE_MARKETS: Set[Tuple[str, str]] = {
    ("ETH", "USDT"),
    ("ETH", "USDC"),
    ("BTC", "USDT"),  # allows BTC/USDT feeds when needed
    ("ETH", "BTC"),
}

# Endpoint controls (env override; defaults exclude noisiest WS providers)
_DEFAULT_ENDPOINT_EXCLUDE: Set[str] = {"mexc", "dexscreener"}
_ENV_ENDPOINT_INCLUDE: Set[str] = {
    name.strip().lower()
    for name in (os.getenv("MARKET_ENDPOINT_INCLUDE") or "").split(",")
    if name.strip()
}
_ENV_ENDPOINT_EXCLUDE: Set[str] = {
    name.strip().lower()
    for name in (os.getenv("MARKET_ENDPOINT_EXCLUDE") or "").split(",")
    if name.strip()
}


def _endpoint_allowed(name: str) -> bool:
    lname = name.lower()
    if _ENV_ENDPOINT_INCLUDE:
        return lname in _ENV_ENDPOINT_INCLUDE
    exclude = set(_DEFAULT_ENDPOINT_EXCLUDE)
    exclude.update(_ENV_ENDPOINT_EXCLUDE)
    return lname not in exclude


def _token_synonyms(token: str) -> set[str]:
    token_u = token.upper()
    synonyms = {token_u}
    normalized = TOKEN_NORMALIZATION.get(token_u)
    if normalized and normalized.upper() != token_u:
        synonyms.add(normalized.upper())
    for original, normalized in TOKEN_NORMALIZATION.items():
        if normalized.upper() == token_u:
            synonyms.add(original.upper())
    return synonyms


def _pair_symbol_variants(symbol: str) -> List[str]:
    symbol_upper = symbol.upper()
    symbol_dash = symbol_upper.replace("/", "-")
    parts = [part for part in symbol_dash.split("-") if part]
    if len(parts) < 2:
        return [symbol_upper] if symbol_upper else []
    base_raw, quote_raw = parts[0], parts[1]
    base_norm, quote_norm = _split_symbol(symbol)
    base_options = set(_token_synonyms(base_norm))
    quote_options = set(_token_synonyms(quote_norm))
    base_options.update({base_raw, base_norm})
    quote_options.update({quote_raw, quote_norm})
    variants: List[str] = []

    def _push(label: str) -> None:
        if label and label not in variants:
            variants.append(label)

    _push(symbol_upper)
    if symbol_dash != symbol_upper:
        _push(symbol_dash)
    for base in sorted(base_options):
        for quote in sorted(quote_options):
            if not base or not quote or base == quote:
                continue
            _push(f"{base}-{quote}")
            _push(f"{quote}-{base}")
    return variants


def _preferred_market_pair(base: str, quote: str) -> Tuple[str, str, bool]:
    base_u = base.upper()
    quote_u = quote.upper()
    base_stable = base_u in _STABLE_TOKEN_PRICE
    quote_stable = quote_u in _STABLE_TOKEN_PRICE
    base_high = base_u in _HIGH_VALUE_TOKENS
    quote_high = quote_u in _HIGH_VALUE_TOKENS
    if base_stable and not quote_stable:
        return quote_u, base_u, True
    if quote_stable and not base_stable:
        return base_u, quote_u, False
    if base_high and not quote_high and not quote_stable:
        return quote_u, base_u, True
    if quote_high and not base_high and not base_stable:
        return base_u, quote_u, False
    return base_u, quote_u, False


@dataclass
class Endpoint:
    name: str
    ws_template: Optional[str]
    subscribe_template: Optional[str]
    rest_template: Optional[str]
    headers: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class RestFetchResult:
    price: Optional[float]
    error: Optional[str] = None
    status: Optional[int] = None


@dataclass
class EndpointHealth:
    strikes: int = 0
    successes: int = 0
    cooldown_until: float = 0.0

    def register_failure(self) -> float:
        self.strikes += 1
        backoff = min(900.0, 2.0 * (2 ** min(6, self.strikes - 1)))
        self.cooldown_until = time.time() + backoff
        return backoff

    def register_success(self) -> None:
        self.successes += 1
        if self.strikes > 0:
            self.strikes -= 1
        if self.strikes == 0:
            self.cooldown_until = 0.0



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
        symbol: str = PRIMARY_SYMBOL,
        chain: str = PRIMARY_CHAIN,
        simulation_interval: float = 2.0,
        ws_template: Optional[str] = None,
        subscribe_template: Optional[str] = None,
    ) -> None:
        template = ws_template or os.getenv("MARKET_WS_TEMPLATE") or os.getenv("UNISWAP_WS_TEMPLATE")
        symbol_lower = symbol.lower().replace("/", "-")
        self._template = template
        self.symbol = symbol
        self.chain = chain
        self.url = url or (template.format(symbol=symbol_lower, SYMBOL=symbol.upper(), pair=symbol_lower) if template else None)
        subscribe_env = os.getenv("MARKET_WS_SUBSCRIBE") or os.getenv("UNISWAP_WS_SUBSCRIBE")
        env_subscribe_provided = bool(subscribe_env)
        self._subscribe_provided = subscribe_template is not None or env_subscribe_provided
        self.subscribe_template = subscribe_template if subscribe_template is not None else subscribe_env
        disable_subscribe = (os.getenv("MARKET_WS_DISABLE_SUBSCRIBE") or "").lower() in {"1", "true", "yes", "on"}
        allow_template_subscribe = (
            os.getenv("MARKET_WS_ALLOW_TEMPLATE_SUBSCRIBE")
            or os.getenv("ALLOW_TEMPLATE_SUBSCRIBE")
            or ("1" if env_subscribe_provided else "0")
        ).lower() in {"1", "true", "yes", "on"}
        self._template_subscribe_disabled = False
        if disable_subscribe:
            self.subscribe_template = None
            self._template_subscribe_disabled = True
        elif self._template and not self.subscribe_template:
            # Only disable auto-subscribe when no explicit subscribe payload is available.
            self._template_subscribe_disabled = True
        self._ws_disabled = bool(self._template and not self.subscribe_template)
        self.simulation_interval = simulation_interval
        self._callbacks: List[CallbackType] = []
        self._db = get_db()
        self._session: Optional[aiohttp.ClientSession] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._stop_event = asyncio.Event()
        self._rest_only_mode = False
        self._rest_only_last_check = 0.0
        self._rest_only_reason = "init"
        self._rest_only_retry = max(30.0, float(os.getenv("REST_ONLY_RETRY_SEC", "120")))
        self.reference_price: Optional[float] = None
        self.price_tolerance = float(os.getenv("PRICE_FEED_TOLERANCE", "0.05"))
        rest_base, rest_quote = _split_symbol(self.symbol)
        self._rest_base, self._rest_quote, self._rest_inverted = _preferred_market_pair(rest_base, rest_quote)
        self.endpoints = self._build_endpoints()
        self._endpoint_scores: Dict[str, float] = {ep.name: 0.0 for ep in self.endpoints}
        self._endpoint_order = {ep.name: idx for idx, ep in enumerate(self.endpoints)}
        self._endpoint_health: Dict[str, EndpointHealth] = {ep.name: EndpointHealth() for ep in self.endpoints}
        self._rest_metric_last = 0.0
        self._rest_metric_interval = max(5.0, float(os.getenv("REST_METRIC_LOG_INTERVAL", "30")))
        self._init_rest_intervals()
        self._rest_outage_base = max(5.0, float(os.getenv("REST_OUTAGE_BACKOFF_SEC", "15")))
        self._rest_outage_growth = max(1.1, float(os.getenv("REST_OUTAGE_GROWTH_FACTOR", "1.6")))
        self._rest_outage_max = max(self._rest_outage_base, float(os.getenv("REST_OUTAGE_MAX_SEC", "120")))
        self._rest_outage_until = 0.0
        self._rest_outage_failures = 0
        self._last_rest_outage_log = 0.0
        self._last_volume: float = 0.0
        self.consensus_sources = int(os.getenv("PRICE_CONSENSUS_SOURCES", "2"))

        self.consensus_window = float(os.getenv("PRICE_CONSENSUS_WINDOW", "60"))
        self.consensus_timeout = float(os.getenv("PRICE_CONSENSUS_TIMEOUT", "45"))
        self._recent_price_by_source: Dict[str, Tuple[float, float]] = {}
        self._last_consensus_ts = time.time()
        self._consensus_initialized = False
        self._endpoint_failures: Dict[str, int] = {}
        self._historical_reference = self._load_historical_reference()
        self._snapshot_seed_attempted = False
        self._endpoint_backoff_until: Dict[str, float] = {}
        self.current_endpoint: str = "bootstrap"
        self._sample_count = 0
        self._last_sample_log = 0.0
        self.metrics = MetricsCollector(self._db)
        self._last_emitted_by_source: Dict[str, Tuple[float, float]] = {}
        self._duplicate_drops = 0
        self._consensus_failures = 0
        self._consensus_relax_until = 0.0
        self._consensus_cooldown_threshold = max(4, int(os.getenv("CONSENSUS_FAILURE_COOLDOWN", "48")))
        self._consensus_cooldown_backoff = max(30.0, float(os.getenv("CONSENSUS_COOLDOWN_SEC", "180")))
        self._consensus_cooldown_until = 0.0
        self._last_cooldown_log = 0.0
        self._liquidity_alpha = min(1.0, max(1e-3, float(os.getenv("LIQUIDITY_ALPHA", "0.05"))))
        self._liquidity_score = 0.0
        self._liquidity_floor = max(0.0, float(os.getenv("LIQUIDITY_SCORE_FLOOR", "25.0")))
        self._low_liquidity_pause = max(15.0, float(os.getenv("LOW_LIQUIDITY_COOLDOWN_SEC", "90")))
        self._low_liquidity_pause_until = 0.0
        vol_window = max(1, int(os.getenv("STREAM_VOL_WINDOW", "360")))
        self._window_prices: deque = deque(maxlen=vol_window)
        self._ws_warning_logged = False
        if not self.url:
            self._select_next_endpoint()
        self.bias_alpha = float(os.getenv("PRICE_BIAS_ALPHA", "0.2"))
        self.vol_alpha = float(os.getenv("PRICE_VOL_ALPHA", "0.1"))
        self.vol_multiplier = float(os.getenv("PRICE_VOL_MULTIPLIER", "3.0"))
        self.max_bias = float(os.getenv("PRICE_MAX_BIAS", "0.05"))
        self._source_bias: Dict[str, float] = {}
        self._source_var: Dict[str, float] = {}
        self._global_price_ema: Optional[float] = None
        self._global_price_var: float = 0.0
        self._last_consensus_confidence: float = 0.0
        rejection_buffer = max(4, int(os.getenv("PRICE_REJECTION_BUFFER", "12")))
        default_confirmations = max(4, rejection_buffer // 2 + 1)
        rejection_confirmations = int(
            os.getenv("PRICE_REJECTION_CONFIRMATIONS", str(default_confirmations))
        )
        self._rejection_confirmations = max(3, min(rejection_confirmations, rejection_buffer))
        self._price_rejections: deque[Tuple[float, float]] = deque(maxlen=rejection_buffer)
        self._onchain_listener = None
        if OnChainPairFeed is not None:
            try:
                listener = OnChainPairFeed(chain=self.chain, symbol=self.symbol)
                if listener.available:
                    self._onchain_listener = listener
                    log_message("market-stream", f"on-chain listener enabled for {self.symbol} on {self.chain}.")
            except Exception as exc:
                log_message("market-stream", f"on-chain feed unavailable for {self.symbol}: {exc}", severity="warning")
        self._onchain_queue: Optional[asyncio.Queue] = None
        self._onchain_task: Optional[asyncio.Task] = None
        self._onchain_consumer: Optional[asyncio.Task] = None
        self._offline_enabled = os.getenv("MARKET_OFFLINE_FALLBACK", "1").lower() in {"1", "true", "yes", "on"}
        self._offline_store: Optional[OfflinePriceStore] = None
        if self._offline_enabled:
            try:
                self._offline_store = OfflinePriceStore()
            except Exception as exc:
                log_message("market-stream", f"offline store unavailable: {exc}", severity="warning")
                self._offline_enabled = False
        self.rate_limiter = APIRateLimiter(default_capacity=5.0, default_refill_rate=1.0)
        self._rest_parallel = max(1, int(os.getenv("REST_CONSENSUS_PARALLEL", "3")))
        self._rest_batch_retry = max(1, int(os.getenv("REST_CONSENSUS_BATCHES", "3")))
        self._rest_latency: Dict[str, deque] = {name: deque(maxlen=32) for name in self._endpoint_scores}
        self._last_rest_health: Dict[str, Any] = {}
        self._snapshot_refresh_interval = max(1, int(os.getenv("MARKET_SNAPSHOT_REFRESH_FAILS", "6")))
        self._last_snapshot_refresh = 0.0
        self._last_rest_unavailable_log = 0.0
        self._last_offline_log = 0.0
        self._debug(
            "init",
            extra={
                "url_set": bool(self.url),
                "template": bool(template),
                "subscribe": bool(self.subscribe_template),
                "ws_disabled": self._ws_disabled,
                "subscribe_disabled": self._template_subscribe_disabled,
            },
        )
        log_message(
            "market-stream",
            "stream configured",
            details={
                "symbol": self.symbol,
                "chain": self.chain,
                "url_set": bool(self.url),
                "template": bool(template),
                "subscribe_template": bool(self.subscribe_template),
            },
        )

    def register(self, callback: CallbackType) -> None:
        self._callbacks.append(callback)

    async def start(self) -> None:
        self._stop_event.clear()
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        self._debug("start", extra={"url_set": bool(self.url), "template": bool(self._template), "subscribe": bool(self.subscribe_template)})
        try:
            await self._refresh_reference_price()
            if self._onchain_listener and self._onchain_listener.available and not self._onchain_task:
                self._onchain_queue = asyncio.Queue()
                self._onchain_task = asyncio.create_task(self._onchain_listener.run(self._onchain_queue.put))
                self._onchain_consumer = asyncio.create_task(self._consume_onchain_queue())
            backoff = 5.0
            while not self._stop_event.is_set():
                self._debug(
                    "loop",
                    extra={
                        "url_set": bool(self.url),
                        "template": bool(self._template),
                        "subscribe": bool(self.subscribe_template),
                        "ws_disabled": self._ws_disabled,
                    },
                )
                if self._ws_disabled:
                    self.url = None
                    self._warn_websocket_unavailable()
                    await self._poll_rest_data(max(backoff, self.consensus_timeout))
                    await self._refresh_reference_price()
                    backoff = min(backoff * 1.2, 300.0)
                    continue
                if self._onchain_listener and self._onchain_listener.available and not self.url:
                    await asyncio.sleep(1.0)
                    continue
                if not self.url:
                    self._warn_websocket_unavailable()
                    await self._poll_rest_data(max(backoff, self.consensus_timeout))
                    if self._template:
                        symbol_lower = self.symbol.lower().replace("/", "-")
                        try:
                            self.url = self._template.format(symbol=symbol_lower, SYMBOL=self.symbol.upper(), pair=symbol_lower)
                        except Exception as exc:
                            self._debug("template_format_error", extra={"error": str(exc), "template": self._template})
                        # When a template is explicitly provided, avoid sending subscribe payloads by default.
                        self.subscribe_template = None
                    if not self.url:
                        self._debug("no_url_after_template", extra={"template": self._template})
                        await asyncio.sleep(1.0)
                        continue
                    self._debug("post-select", extra={"url_set": bool(self.url), "template": bool(self._template), "subscribe": bool(self.subscribe_template)})
                    await asyncio.sleep(1.0)
                    continue
                try:
                    await self._consume_ws()
                    backoff = 5.0
                except aiohttp.WSServerHandshakeError as exc:
                    if exc.status in (429, 503):
                        log_message("market-stream", f"rate limited ({exc.status}); sleeping {backoff:.1f}s", severity="warning")
                    elif exc.status == 451:
                        current = time.time()
                        self._endpoint_backoff_until[self.current_endpoint] = current + max(60.0, backoff * 2)
                        log_message(
                            "market-stream",
                            f"endpoint {self.current_endpoint} denied (451); backing off.",
                            severity="warning",
                            details={"resume_at": self._endpoint_backoff_until[self.current_endpoint]},
                        )
                    else:
                        log_message(
                            "market-stream",
                            f"websocket handshake failed ({exc.status}); retrying in {backoff:.1f}s",
                            severity="warning",
                        )
                    await self._poll_rest_data(backoff)
                    backoff = min(backoff * 1.5, 300.0)
                    await self._refresh_reference_price()
                    if not self._template:
                        self._select_next_endpoint()
                except Exception as exc:  # pragma: no cover - network dependent
                    self._debug("ws_error", extra={"error": str(exc), "url": self.url})
                    log_message(
                        "market-stream",
                        f"websocket error {exc}; retrying in {backoff:.1f}s",
                        severity="warning",
                        details={"exc_type": type(exc).__name__},
                    )
                    await self._poll_rest_data(backoff)
                    backoff = min(backoff * 1.5, 300.0)
                    await self._refresh_reference_price()
                    # Do not switch endpoints when template is in use; rely on REST/on-chain to recover.
                    if not self._template:
                        self._select_next_endpoint()
        except Exception as exc:
            self._debug("fatal_start_error", extra={"error": str(exc)})
            log_message("market-stream", f"fatal stream error: {exc}", severity="error")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._session:
            await self._session.close()
            self._session = None
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        if self._onchain_listener:
            await self._onchain_listener.stop()
        if self._onchain_task:
            self._onchain_task.cancel()
            try:
                await self._onchain_task
            except Exception:
                pass
            self._onchain_task = None
        if self._onchain_consumer:
            self._onchain_consumer.cancel()
            try:
                await self._onchain_consumer
            except Exception:
                pass
            self._onchain_consumer = None

    async def _consume_ws(self) -> None:  # pragma: no cover - requires network
        async with aiohttp.ClientSession() as session:
            self._session = session
            async with session.ws_connect(self.url) as ws:
                subscribe_msg = self._build_subscribe_payload(self.symbol)
                if subscribe_msg is not None:
                    try:
                        if isinstance(subscribe_msg, (dict, list)):
                            await ws.send_json(subscribe_msg)
                        else:
                            await ws.send_str(str(subscribe_msg))
                    except Exception as exc:
                        log_message("market-stream", f"failed to send subscribe payload: {exc}", severity="warning")
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
                        log_message("market-stream", "rate-limit notice from server; backing off 15s", severity="warning")
                        await asyncio.sleep(15.0)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

    async def _poll_rest_data(self, duration: float) -> None:
        if self._http_session is None:
            return
        base, quote = self._rest_base, self._rest_quote
        end_time = time.time() + max(duration, self.rest_poll_interval)
        while not self._stop_event.is_set() and time.time() < end_time:
            dispatched = False
            now = time.time()
            if self._rest_outage_active(now):
                if now - self._last_rest_outage_log >= 30.0:
                    log_message(
                        "market-stream",
                        "REST network outage cooling down; delaying polls",
                        severity="warning",
                        details={
                            "symbol": self.symbol,
                            "resume_in": round(max(0.0, self._rest_outage_until - now), 2),
                        },
                    )
                    self._last_rest_outage_log = now
                await self._dispatch_offline_snapshot()
                await self._handle_consensus_failure(cooldown_only=True)
                await asyncio.sleep(self._cooldown_sleep_interval(now))
                continue
            endpoints = self._eligible_rest_endpoints()
            if not endpoints:
                offline_ready = False
                if self._offline_enabled and self._offline_store:
                    try:
                        offline_ready = self._offline_snapshot() is not None
                    except Exception:
                        offline_ready = False
                if now - self._last_rest_unavailable_log >= 30.0:
                    log_message(
                        "market-stream",
                        "all REST endpoints cooling down.",
                        severity="info" if offline_ready else "warning",
                        details={"symbol": self.symbol, "offline_ready": offline_ready},
                    )
                    self._last_rest_unavailable_log = now
                await self._dispatch_offline_snapshot()
                await self._handle_consensus_failure(cooldown_only=True)
                await asyncio.sleep(self._cooldown_sleep_interval(now))
                continue
            attempts = 0
            total_attempted = 0
            total_network_errors = 0
            non_network_response = False
            outage_reason: Optional[str] = None
            outage_endpoint: Optional[str] = None
            for idx in range(0, len(endpoints), self._rest_parallel):
                if attempts >= self._rest_batch_retry:
                    break
                batch = endpoints[idx : idx + self._rest_parallel]
                if not batch:
                    break
                attempts += 1
                results = await asyncio.gather(
                    *(self._timed_rest_fetch(endpoint, base, quote) for endpoint in batch),
                    return_exceptions=False,
                )
                for endpoint, result, latency in results:
                    if result.error == "unavailable":
                        continue
                    self._record_rest_latency(endpoint.name, latency)
                    total_attempted += 1
                    if result.error in {"network", "timeout"}:
                        total_network_errors += 1
                        if outage_reason is None:
                            outage_reason = result.error
                            outage_endpoint = endpoint.name
                        continue
                    if result.error == "rate_limited_local":
                        continue
                    non_network_response = True
                    if result.error:
                        self._record_endpoint_failure(endpoint.name)
                        continue
                    price = result.price
                    if not price or price <= 0:
                        self._record_endpoint_failure(endpoint.name)
                        continue
                    accepted, pending, consensus, confidence = self._confirm_consensus(endpoint.name, price)
                    if pending:
                        continue
                    if not accepted or consensus is None:
                        self._record_endpoint_failure(endpoint.name)
                        continue
                    live_sources = self._live_source_count()
                    if not self._accept_consensus_price(
                        consensus,
                        confidence=confidence,
                        live_sources=live_sources,
                    ):
                        self._record_endpoint_failure(endpoint.name)
                        continue
                    self._record_endpoint_success(endpoint.name, price=price, consensus=consensus)
                    sample = {
                        "ts": time.time(),
                        "symbol": self.symbol,
                        "chain": self.chain,
                        "price": consensus,
                        "volume": self._last_volume,
                        "rest": endpoint.name,
                        "consensus_confidence": confidence,
                    }
                    await self._dispatch(sample)
                    dispatched = True
                    break
                if dispatched:
                    break
            outage_detected = total_attempted > 0 and total_network_errors == total_attempted
            if outage_detected:
                self._register_rest_outage(reason=outage_reason or "network", endpoint=outage_endpoint)
            if not dispatched:
                cooldown_only = outage_detected and not non_network_response
                fallback_used = await self._handle_consensus_failure(cooldown_only=cooldown_only)
                if not fallback_used and not outage_detected:
                    log_message(
                        "market-stream",
                        "unable to obtain consensus price during REST poll cycle.",
                        severity="warning",
                        details={"symbol": self.symbol, "active_endpoints": len(endpoints)},
                    )
            self._last_rest_health = self.rest_health()
            await asyncio.sleep(self.rest_poll_interval)

    async def _dispatch_offline_snapshot(self) -> None:
        """
        When all endpoints are cooling down, push an offline snapshot so
        downstream consumers (ghost trading) keep moving instead of stalling.
        """
        if not self._offline_store:
            return
        alias = self.symbol
        snapshot = self._offline_store.get_price(self.symbol)
        if snapshot:
            alias, snapshot = self._normalize_offline_snapshot(alias, snapshot)
        else:
            offline_hit = self._offline_snapshot()
            if not offline_hit:
                return
            alias, snapshot = offline_hit
        if not snapshot or snapshot.price <= 0:
            return
        now = time.time()
        sample = {
            "ts": snapshot.ts or now,
            "symbol": snapshot.symbol,
            "chain": self.chain,
            "price": float(snapshot.price),
            "volume": float(snapshot.volume or 0.0),
            "rest": "offline",
            "source": snapshot.source or "offline",
            "alias": alias,
        }
        try:
            await self._dispatch(sample)
            if now - self._last_offline_log >= 60.0:
                log_message(
                    "market-stream",
                    "using offline snapshot while endpoints recover",
                    severity="info",
                    details={
                        "symbol": snapshot.symbol,
                        "price": snapshot.price,
                        "source": sample["source"],
                        "alias": alias,
                    },
                )
                self._last_offline_log = now
        except Exception as exc:  # pragma: no cover - defensive
            log_message("market-stream", f"offline snapshot dispatch failed: {exc}", severity="warning")

    async def _consume_onchain_queue(self) -> None:
        if not self._onchain_queue:
            return
        while not self._stop_event.is_set():
            try:
                sample = await self._onchain_queue.get()
            except asyncio.CancelledError:
                break
            except Exception:
                continue
            if not isinstance(sample, dict):
                continue
            try:
                price_val = float(sample.get("price") or 0.0)
            except Exception:
                price_val = 0.0
            if price_val > 0:
                sample["price"] = self._normalize_live_price(price_val)
            sample.setdefault("rest", "onchain")
            sample.setdefault("source", "onchain")
            try:
                await self._dispatch(sample)
            except Exception as exc:
                log_message("market-stream", f"on-chain dispatch error: {exc}", severity="error")

    async def _dispatch(self, sample: Dict[str, Any]) -> None:
        try:
            self._last_volume = float(sample.get("volume") or self._last_volume or 0.0)
        except Exception:
            pass
        source = str(sample.get("rest") or self.current_endpoint or "stream")
        price_val = float(sample.get("price") or 0.0)
        ts_val = float(sample.get("ts") or time.time())
        consensus_conf = float(sample.get("consensus_confidence") or self._last_consensus_confidence or 0.0)
        if consensus_conf < 0.0:
            consensus_conf = 0.0
        sample["consensus_confidence"] = consensus_conf
        self._last_consensus_confidence = consensus_conf
        if self._should_skip_duplicate(source, price_val, ts_val):
            if self._duplicate_drops % 25 == 0:
                self.metrics.record(
                    MetricStage.DATA_STREAM,
                    {
                        "duplicate_rate": self._duplicate_drops / max(1, self._sample_count),
                    },
                    category="dedupe",
                    meta={"symbol": self.symbol, "source": source},
                )
            return

        self._sample_count += 1
        now = time.time()
        if now - self._last_sample_log >= 30.0:
            self._last_sample_log = now
            log_message(
                "market-stream",
                f"{self.symbol} flow samples",
                details={"count": self._sample_count, "price": price_val, "source": source},
            )
        self._db.insert_market_sample(
            chain=sample.get("chain", self.chain),
            symbol=sample.get("symbol", self.symbol),
            price=price_val,
            volume=float(sample.get("volume") or 0),
            raw=sample,
        )
        self._last_emitted_by_source[source] = (price_val, ts_val)
        self._window_prices.append(price_val)
        self._update_price_stats(price_val)
        if source and source not in {"fallback", "bootstrap"}:
            baseline = self.reference_price or price_val
            self._update_source_stats(source, price_val, baseline)
        volatility = statistics.pstdev(self._window_prices) if len(self._window_prices) > 1 else 0.0
        drift = 0.0 if self.reference_price is None else price_val - self.reference_price
        self._record_stream_metrics(
            sample,
            source,
            volatility=volatility,
            drift=drift,
            fallback=(sample.get("rest") == "fallback"),
        )
        for callback in list(self._callbacks):
            try:
                if asyncio.iscoroutinefunction(callback):  # type: ignore[arg-type]
                    await callback(sample)  # type: ignore[misc]
                else:
                    callback(sample)  # type: ignore[misc]
            except Exception as exc:
                log_message("market-stream", f"callback error: {exc}", severity="error")

    def _debug(self, label: str, extra: Optional[Dict[str, Any]] = None) -> None:
        try:
            path = Path("logs/stream_debug.log")
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ts": time.time(),
                "label": label,
                "symbol": getattr(self, "symbol", None),
                "chain": getattr(self, "chain", None),
                "url": getattr(self, "url", None),
                "template": bool(getattr(self, "_template", None)),
            }
            if extra:
                payload.update(extra)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    def _should_skip_duplicate(self, source: str, price: float, ts_val: float) -> bool:
        last = self._last_emitted_by_source.get(source)
        if not last:
            return False
        last_price, last_ts = last
        if ts_val <= last_ts:
            return False
        tolerance = max(1e-9, (self.reference_price or price) * 1e-5)
        if abs(price - last_price) <= tolerance and (ts_val - last_ts) <= 1.0:
            self._duplicate_drops += 1
            return True
        return False

    def _record_stream_metrics(
        self,
        sample: Dict[str, Any],
        source: str,
        *,
        volatility: float,
        drift: float,
        fallback: bool = False,
    ) -> None:
        price_val = float(sample.get("price") or 0.0)
        volume_val = float(sample.get("volume") or 0.0)
        tolerance = self._dynamic_tolerance(price_val if price_val > 0 else (self.reference_price or 1.0))
        metrics = {
            "price": price_val,
            "volume": volume_val,
            "rolling_volatility": volatility,
            "price_drift": drift,
            "consensus_age": max(0.0, time.time() - self._last_consensus_ts),
            "sample_count": self._sample_count,
            "duplicate_rate": self._duplicate_drops / max(1, self._sample_count),
            "consensus_failures": self._consensus_failures,
            "tolerance": tolerance,
            "consensus_confidence": float(self._last_consensus_confidence),
        }
        self.metrics.record(
            MetricStage.DATA_STREAM,
            metrics,
            category="stream_flow",
            meta={
                "symbol": self.symbol,
                "source": source,
                "fallback": fallback,
                "endpoint_score": self._endpoint_scores.get(source),
            },
        )

    def _update_price_stats(self, price: float) -> None:
        if price <= 0:
            return
        if self._global_price_ema is None:
            self._global_price_ema = price
            self._global_price_var = 0.0
            return
        delta = price - self._global_price_ema
        self._global_price_ema += self.vol_alpha * delta
        self._global_price_var = max(
            0.0,
            (1 - self.vol_alpha) * self._global_price_var + self.vol_alpha * (delta ** 2),
        )

    def _update_source_stats(self, name: str, price: float, consensus: float) -> None:
        if price <= 0 or consensus <= 0:
            return
        rel = (price - consensus) / max(consensus, 1e-9)
        prev_bias = self._source_bias.get(name, 0.0)
        new_bias = (1 - self.bias_alpha) * prev_bias + self.bias_alpha * rel
        new_bias = float(max(-self.max_bias, min(self.max_bias, new_bias)))
        self._source_bias[name] = new_bias
        prev_var = self._source_var.get(name, 0.0)
        self._source_var[name] = float((1 - self.vol_alpha) * prev_var + self.vol_alpha * (rel - new_bias) ** 2)

    def _dynamic_tolerance(self, base_price: float) -> float:
        tolerance = self.price_tolerance
        if self._global_price_ema is None or self._global_price_var <= 0:
            return tolerance
        rel_vol = math.sqrt(self._global_price_var) / max(self._global_price_ema, 1e-9)
        return max(tolerance, self.vol_multiplier * rel_vol)

    async def _handle_consensus_failure(self, *, cooldown_only: bool = False) -> bool:
        self._consensus_failures += 1
        if self._consensus_failures in {10, 20}:
            self._consensus_relax_until = time.time() + max(self.consensus_timeout, 120.0)
            self.metrics.feedback(
                "market_stream",
                severity=FeedbackSeverity.WARNING,
                label="consensus_relaxed",
                details={
                    "symbol": self.symbol,
                    "failures": self._consensus_failures,
                    "min_sources": 1,
                },
            )
        fallback_price = self._fallback_consensus_price()
        used_fallback = False
        if fallback_price:
            used_fallback = True
            sample = {
                "ts": time.time(),
                "symbol": self.symbol,
                "chain": self.chain,
                "price": fallback_price,
                "volume": self._last_volume,
                "rest": "fallback",
                "consensus_confidence": min(0.4, self._last_consensus_confidence or 0.25),
            }
            self.metrics.feedback(
                "market_stream",
                severity=FeedbackSeverity.WARNING,
                label="consensus_fallback",
                details={
                    "symbol": self.symbol,
                    "price": fallback_price,
                    "failures": self._consensus_failures,
                },
            )
            await self._dispatch(sample)
            self._seed_recent_reference(fallback_price)
        else:
            offline = await self._offline_failover_sample()
            if offline:
                used_fallback = True
                await self._dispatch(offline)
                self._seed_recent_reference(_safe_float(offline.get("price") or 0.0))
                self.metrics.feedback(
                    "market_stream",
                    severity=FeedbackSeverity.WARNING,
                    label="offline_price_injected",
                    details={
                        "symbol": self.symbol,
                        "price": offline.get("price"),
                        "source": offline.get("source", "offline"),
                    },
                )
                return True
            self.metrics.feedback(
                "market_stream",
                severity=FeedbackSeverity.CRITICAL,
                label="consensus_failure",
                details={
                    "symbol": self.symbol,
                    "failures": self._consensus_failures,
                },
            )
        if self._consensus_failures % self._snapshot_refresh_interval == 0:
            await self._refresh_market_snapshots()
        if not cooldown_only and self._consensus_failures % 4 == 0:
            self._grow_rest_interval(reason="consensus_failure")
            log_message(
                "market-stream",
                "consensus failure throttling REST polls",
                severity="warning",
                details={
                    "symbol": self.symbol,
                    "rest_interval": self.rest_poll_interval,
                    "failures": self._consensus_failures,
                },
            )
        return used_fallback

    def _fallback_consensus_price(self) -> Optional[float]:
        now = time.time()
        valid = [
            price
            for price, ts in self._recent_price_by_source.values()
            if now - ts <= self.consensus_window * 2 and price > 0
        ]
        if valid:
            return float(statistics.median(valid))
        if self.reference_price:
            return float(self.reference_price)
        offline_hit = self._offline_snapshot()
        if offline_hit:
            alias, snapshot = offline_hit
            self._recent_price_by_source[f"offline:{alias}"] = (snapshot.price, snapshot.ts or now)
            return float(snapshot.price)
        snapshot_price = self._load_snapshot_reference(allow_network=False)
        if snapshot_price:
            self._recent_price_by_source["snapshot:local"] = (snapshot_price, now)
            return float(snapshot_price)
        if self._historical_reference:
            return float(self._historical_reference)
        return None

    async def _offline_failover_sample(self) -> Optional[Dict[str, Any]]:
        offline_hit = self._offline_snapshot()
        if not offline_hit:
            return None
        alias, snapshot = offline_hit
        volume = snapshot.volume if snapshot.volume is not None else self._last_volume
        return {
            "ts": snapshot.ts or time.time(),
            "symbol": self.symbol,
            "chain": self.chain,
            "price": snapshot.price,
            "volume": volume,
            "rest": "offline",
            "source": snapshot.source or alias,
            "alias": alias,
        }

    def _offline_snapshot(self) -> Optional[Tuple[str, OfflineSnapshot]]:
        if not (self._offline_enabled and self._offline_store):
            return None
        pair_snapshot = self._offline_pair_snapshot()
        if pair_snapshot:
            return pair_snapshot
        seen = set()
        for label in self._offline_symbol_candidates():
            if label in seen:
                continue
            seen.add(label)
            snapshot = self._offline_store.get_price(label)
            if snapshot:
                return self._normalize_offline_snapshot(label, snapshot)
        return None

    def _normalize_offline_snapshot(
        self,
        alias: str,
        snapshot: OfflineSnapshot,
    ) -> Tuple[str, OfflineSnapshot]:
        price = _safe_float(getattr(snapshot, "price", 0.0))
        normalized = self._normalize_reference_value(price)
        if normalized <= 0 or abs(normalized - price) < 1e-12:
            return alias, snapshot
        symbol = str(getattr(snapshot, "symbol", alias or self.symbol)).upper()
        name = str(getattr(snapshot, "name", symbol))
        volume = getattr(snapshot, "volume", None)
        source = str(getattr(snapshot, "source", "offline"))
        ts_value = _safe_float(getattr(snapshot, "ts", 0.0)) or time.time()
        change = getattr(snapshot, "change_24h", None)
        return alias, OfflineSnapshot(
            symbol=symbol,
            name=name,
            price=float(normalized),
            volume=volume,
            source=source,
            ts=ts_value,
            change_24h=change,
        )

    def _offline_token_snapshot(self, token: str) -> Optional[OfflineSnapshot]:
        if not (self._offline_enabled and self._offline_store):
            return None
        token_upper = token.upper()
        for symbol in _token_synonyms(token_upper):
            snapshot = self._offline_store.get_price(symbol)
            if snapshot:
                return snapshot
        stable_price = _STABLE_TOKEN_PRICE.get(token_upper)
        if stable_price is not None:
            return OfflineSnapshot(
                symbol=token_upper,
                name=token_upper,
                price=stable_price,
                volume=None,
                source="stable_assumed",
                ts=time.time(),
                change_24h=None,
            )
        if token_upper in _HIGH_VALUE_TOKENS:
            derived = self._derived_high_value_snapshot(token_upper)
            if derived:
                return derived
        return None

    def _offline_pair_snapshot(self) -> Optional[Tuple[str, OfflineSnapshot]]:
        symbol = self.symbol
        if "-" not in symbol and "/" not in symbol:
            return None
        base, quote = _split_symbol(symbol)
        if not base or not quote or base == quote:
            return None
        base_snapshot = self._offline_token_snapshot(base)
        quote_snapshot = self._offline_token_snapshot(quote)
        if not base_snapshot or not quote_snapshot:
            return self._offline_pair_ohlcv_snapshot(base, quote)
        base_price = _safe_float(getattr(base_snapshot, "price", 0.0))
        quote_price = _safe_float(getattr(quote_snapshot, "price", 0.0))
        if base_price <= 0 or quote_price <= 0:
            return self._offline_pair_ohlcv_snapshot(base, quote)
        ratio = base_price / quote_price
        base_symbol = str(getattr(base_snapshot, "symbol", base)).upper()
        quote_symbol = str(getattr(quote_snapshot, "symbol", quote)).upper()
        base_name = str(getattr(base_snapshot, "name", base_symbol))
        quote_name = str(getattr(quote_snapshot, "name", quote_symbol))
        sources = [
            str(getattr(base_snapshot, "source", "")),
            str(getattr(quote_snapshot, "source", "")),
        ]
        sources = [source for source in sources if source]
        source_label = "+".join(sources) if sources else "offline_pair"
        ts_candidates = [
            _safe_float(getattr(base_snapshot, "ts", 0.0)),
            _safe_float(getattr(quote_snapshot, "ts", 0.0)),
        ]
        ts_value = max(ts_candidates) if ts_candidates else 0.0
        if ts_value <= 0:
            ts_value = time.time()
        snapshot = OfflineSnapshot(
            symbol=self.symbol.upper(),
            name=f"{base_name}/{quote_name}",
            price=float(ratio),
            volume=None,
            source=source_label,
            ts=ts_value,
            change_24h=None,
        )
        return f"{base_symbol}-{quote_symbol}", snapshot

    def _ohlcv_median_price(self, rows: Sequence[dict]) -> Optional[float]:
        if not rows:
            return None
        prices: List[float] = []
        sample = rows[-12:]
        for entry in sample:
            if not isinstance(entry, dict):
                continue
            price = _safe_float(
                entry.get("close")
                or entry.get("price")
                or entry.get("priceUsd")
                or entry.get("price_usd")
                or 0.0
            )
            if price > 0:
                prices.append(price)
        if not prices:
            return None
        return float(statistics.median(prices))

    def _offline_pair_ohlcv_snapshot(self, base: str, quote: str) -> Optional[Tuple[str, OfflineSnapshot]]:
        if not self._offline_store or not hasattr(self._offline_store, "get_ohlcv_tail"):
            return None
        candidates = _pair_symbol_variants(self.symbol)
        if not candidates:
            candidates = [f"{base}-{quote}", f"{quote}-{base}"]
        now = time.time()
        tail_bars = max(6, int(os.getenv("OFFLINE_OHLCV_BARS", "12")))
        base_label = base.upper()
        quote_label = quote.upper()
        for label in candidates:
            try:
                rows = self._offline_store.get_ohlcv_tail(label, bars=tail_bars)
            except Exception:
                continue
            if not rows:
                continue
            price = self._ohlcv_median_price(rows)
            if not price or price <= 0:
                continue
            normalized = self._normalize_pair_price(price, base_label, quote_label)
            if normalized <= 0:
                continue
            ts_value = now
            last_row = rows[-1] if rows else None
            if isinstance(last_row, dict):
                ts_value = _safe_float(
                    last_row.get("timestamp") or last_row.get("ts") or last_row.get("time") or 0.0
                ) or now
            snapshot = OfflineSnapshot(
                symbol=f"{base_label}-{quote_label}",
                name=f"{base_label}/{quote_label}",
                price=float(normalized),
                volume=None,
                source=f"ohlcv:{label}",
                ts=ts_value,
                change_24h=None,
            )
            return f"{base_label}-{quote_label}", snapshot
        return None

    def _normalize_reference_value(self, price: float) -> float:
        if price <= 0:
            return price
        if "-" not in self.symbol and "/" not in self.symbol:
            return price
        base, quote = _split_symbol(self.symbol)
        return self._normalize_pair_price(price, base, quote)

    def _normalize_live_price(self, price: float) -> float:
        return self._normalize_reference_value(price)

    def _normalize_pair_price(self, price: float, base: str, quote: str) -> float:
        if price <= 0:
            return price
        expected = self._expected_pair_ratio(base, quote)
        if expected:
            adjusted = float(price)
            inverse = 1.0 / adjusted
            error_direct = abs(math.log(adjusted) - math.log(expected))
            error_inverse = abs(math.log(inverse) - math.log(expected))
            if error_inverse + 0.05 < error_direct:
                return inverse
            return adjusted
        if self._should_invert_pair_price(price, base, quote):
            return 1.0 / price
        return price

    def _expected_pair_ratio(self, base: str, quote: str) -> Optional[float]:
        base_price = self._token_usd_hint(base)
        quote_price = self._token_usd_hint(quote)
        if not base_price or not quote_price:
            return None
        return base_price / quote_price

    def _should_invert_pair_price(self, price: float, base: str, quote: str) -> bool:
        if price <= 0:
            return False
        base_u = base.upper()
        quote_u = quote.upper()
        if base_u in _STABLE_TOKEN_PRICE and quote_u in _HIGH_VALUE_TOKENS:
            return price >= 1.0
        if quote_u in _STABLE_TOKEN_PRICE and base_u in _HIGH_VALUE_TOKENS:
            return price <= 1.0
        return False

    def _token_usd_hint(self, token: str) -> Optional[float]:
        token_upper = token.upper()
        stable_price = _STABLE_TOKEN_PRICE.get(token_upper)
        if stable_price is not None:
            return float(stable_price)
        if getattr(self, "_offline_enabled", False) and getattr(self, "_offline_store", None):
            snapshot = self._offline_token_snapshot(token_upper)
            if snapshot:
                price = _safe_float(getattr(snapshot, "price", 0.0))
                if price > 0:
                    return float(price)
        return None

    def _derived_high_value_snapshot(self, token_upper: str) -> Optional[OfflineSnapshot]:
        if not self._offline_store:
            return None
        try:
            snapshots = self._offline_store.snapshots(limit=48)
        except Exception:
            return None
        stable_tokens = set(_STABLE_TOKEN_PRICE)
        for snapshot in snapshots:
            symbol = str(getattr(snapshot, "symbol", "")).upper().replace("/", "-")
            if "-" not in symbol:
                continue
            base, quote = _split_symbol(symbol)
            if token_upper not in {base, quote}:
                continue
            other = quote if token_upper == base else base
            if other not in stable_tokens:
                continue
            price = _safe_float(getattr(snapshot, "price", 0.0))
            if price <= 0:
                continue
            derived = price if price >= 1.0 else 1.0 / price
            ts_value = _safe_float(getattr(snapshot, "ts", 0.0)) or time.time()
            source = str(getattr(snapshot, "source", "offline"))
            return OfflineSnapshot(
                symbol=token_upper,
                name=token_upper,
                price=float(derived),
                volume=None,
                source=f"{source}+pair_inferred",
                ts=ts_value,
                change_24h=None,
            )
        return None

    def _offline_symbol_candidates(self) -> List[str]:
        symbol_upper = self.symbol.upper()
        symbol_dash = symbol_upper.replace("/", "-")
        candidates: List[str] = _pair_symbol_variants(self.symbol)
        raw_parts: List[str] = []
        if "-" in symbol_dash:
            raw_parts = symbol_dash.split("-")
        base_norm, quote_norm = _split_symbol(self.symbol)
        augmented = raw_parts + [base_norm, quote_norm]
        for token in list(augmented):
            norm = TOKEN_NORMALIZATION.get(token.upper(), token.upper())
            if norm not in augmented:
                augmented.append(norm)
        for token in augmented:
            token_clean = token.upper().strip()
            if not token_clean:
                continue
            candidates.append(token_clean)
            if token_clean not in {"USD", "USDC", "USDT"}:
                candidates.append(f"{token_clean}-USDC")
                candidates.append(f"{token_clean}-USD")
        ordered: List[str] = []
        for label in candidates:
            key = label.upper()
            if key and key not in ordered:
                ordered.append(key)
            if len(ordered) >= 24:
                break
        return ordered

    def _build_subscribe_payload(self, symbol: str) -> Optional[Union[str, Dict[str, Any]]]:
        if not self.subscribe_template:
            return None
        if self.subscribe_template == "COINBASE_DYNAMIC":
            base, quote = _split_symbol(symbol)
            base, quote, _ = _preferred_market_pair(base, quote)
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
        rendered: Union[str, Dict[str, Any]] = self.subscribe_template
        try:
            rendered = self.subscribe_template.format(symbol=symbol_lower, SYMBOL=symbol.upper(), pair=symbol_lower)
        except KeyError:
            rendered = self.subscribe_template
        except Exception:
            rendered = self.subscribe_template
        if isinstance(rendered, str):
            trimmed = rendered.strip()
            if trimmed.startswith("{") or trimmed.startswith("["):
                try:
                    parsed = json.loads(trimmed)
                    return parsed
                except Exception:
                    return rendered
        return rendered

    def _normalize_payload(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Uniswap-style streaming payloads often have nested "data" nodes
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        price = _safe_float(payload.get("price") or payload.get("p"))
        volume = _safe_float(payload.get("volume") or payload.get("v"))
        if price == 0 and isinstance(data, dict):
            price = _safe_float(data.get("price") or data.get("priceUsd") or data.get("priceUSD"))
        if price == 0 and isinstance(data, dict):
            price = _safe_float(data.get("best_bid") or data.get("bestBid") or data.get("ask"))
        if volume == 0 and isinstance(data, dict):
            volume = _safe_float(data.get("volume") or data.get("amount") or data.get("amountUSD"))
        symbol = (
            payload.get("symbol")
            or payload.get("s")
            or (data.get("symbol") if isinstance(data, dict) else None)
            or self.symbol
        )
        ts = _safe_float(payload.get("ts") or payload.get("time") or (data.get("timestamp") if isinstance(data, dict) else 0))
        if ts == 0:
            ts = time.time()
        if not price or price <= 0:
            return None
        price = self._normalize_live_price(price)
        if not price or price <= 0:
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
        if price <= 0:
            return False
        if self.reference_price is None:
            self.reference_price = price
            self._update_price_stats(price)
            self._seed_recent_reference(price)
            return True

        tolerance = self._dynamic_tolerance(self.reference_price)
        diff = abs(price - self.reference_price) / max(self.reference_price, 1e-9)
        if diff <= tolerance:
            blend = self.vol_alpha
            self.reference_price = (1 - blend) * self.reference_price + blend * price
            self._update_price_stats(price)
            self._seed_recent_reference(self.reference_price)
            self._price_rejections.clear()
            return True

        if diff <= tolerance * 2:
            blend = self.vol_alpha * 0.25
            self.reference_price = (1 - blend) * self.reference_price + blend * price
            self._update_price_stats(price)
            self._seed_recent_reference(self.reference_price)
            self._price_rejections.clear()
            self.metrics.feedback(
                "market_stream",
                severity=FeedbackSeverity.WARNING,
                label="wide_acceptance",
                details={
                    "symbol": self.symbol,
                    "price": price,
                    "reference": self.reference_price,
                    "tolerance": tolerance,
                    "diff": diff,
                },
            )
            return True

        if self._handle_price_rejection(price, tolerance):
            log_message(
                "market-stream",
                "reference recalibrated after repeated deviations",
                severity="info",
                details={"price": price, "reference": self.reference_price, "symbol": self.symbol},
            )
            return True

        log_message(
            "market-stream",
            "price deviates from reference; waiting for confirmation",
            severity="warning",
            details={"price": price, "reference": self.reference_price, "symbol": self.symbol},
        )
        return False

    def _build_endpoints(self) -> List[Endpoint]:
        base, quote = self._rest_base, self._rest_quote
        endpoints: List[Endpoint] = []
        binance_symbol = _binance_symbol(base, quote)
        if binance_symbol and (base, quote) in BINANCE_MARKETS and _endpoint_allowed("binance"):
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
        if coinbase_base and coinbase_quote and _endpoint_allowed("coinbase"):
            endpoints.append(
                Endpoint(
                    name="coinbase",
                    ws_template="wss://ws-feed.exchange.coinbase.com",
                    subscribe_template="COINBASE_DYNAMIC",
                    rest_template=f"https://api.exchange.coinbase.com/products/{coinbase_base}-{coinbase_quote}/ticker",
                )
            )
        coingecko_id = COINGECKO_IDS.get(base)
        if coingecko_id and _endpoint_allowed("coingecko"):
            endpoints.append(
                Endpoint(
                    name="coingecko",
                    ws_template=None,
                    subscribe_template=None,
                    rest_template=f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies={quote.lower()}",
                )
            )
        bitstamp_symbol = f"{base.lower()}{quote.lower()}"
        if quote in {"USD", "USDT", "EUR"} and _endpoint_allowed("bitstamp"):
            endpoints.append(
                Endpoint(
                    name="bitstamp",
                    ws_template=None,
                    subscribe_template=None,
                    rest_template=f"https://www.bitstamp.net/api/v2/ticker/{bitstamp_symbol}/",
                )
            )
        okx_quote = quote.upper()
        if okx_quote in {"USDT", "USDC", "USD"} and _endpoint_allowed("okx"):
            endpoints.append(
                Endpoint(
                    name="okx",
                    ws_template=None,
                    subscribe_template=None,
                    rest_template=f"https://www.okx.com/api/v5/market/ticker?instId={base.upper()}-{okx_quote}",
                )
            )
        kucoin_quote = quote.upper()
        if kucoin_quote in {"USDT", "USDC", "BTC", "ETH"} and _endpoint_allowed("kucoin"):
            endpoints.append(
                Endpoint(
                    name="kucoin",
                    ws_template=None,
                    subscribe_template=None,
                    rest_template=f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={base.upper()}-{kucoin_quote}",
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

    def _eligible_rest_endpoints(self) -> List[Endpoint]:
        now = time.time()
        eligible: List[Endpoint] = []
        for endpoint in self._ranked_endpoints():
            resume = self._endpoint_backoff_until.get(endpoint.name, 0.0)
            if resume and resume > now:
                continue
            eligible.append(endpoint)
        return eligible

    def _rest_outage_active(self, now: Optional[float] = None) -> bool:
        now = now or time.time()
        return self._rest_outage_until > now

    def _clear_rest_outage(self) -> None:
        self._rest_outage_until = 0.0
        self._rest_outage_failures = 0

    def _register_rest_outage(self, *, reason: str, endpoint: Optional[str] = None) -> None:
        now = time.time()
        self._rest_outage_failures += 1
        exponent = min(6, max(0, self._rest_outage_failures - 1))
        backoff = min(self._rest_outage_max, self._rest_outage_base * (self._rest_outage_growth ** exponent))
        self._rest_outage_until = max(self._rest_outage_until, now + backoff)
        self.metrics.feedback(
            "market_stream",
            severity=FeedbackSeverity.WARNING,
            label="rest_outage",
            details={
                "symbol": self.symbol,
                "endpoint": endpoint,
                "reason": reason,
                "backoff_sec": round(max(0.0, self._rest_outage_until - now), 2),
            },
        )
        if now - self._last_rest_outage_log >= 30.0:
            log_message(
                "market-stream",
                "REST network outage detected; pausing polls",
                severity="warning",
                details={
                    "symbol": self.symbol,
                    "endpoint": endpoint,
                    "reason": reason,
                    "backoff_sec": round(max(0.0, self._rest_outage_until - now), 2),
                },
            )
            self._last_rest_outage_log = now

    def _cooldown_sleep_interval(self, now: Optional[float] = None) -> float:
        now = now or time.time()
        next_ready_values: List[float] = []
        if self._endpoint_backoff_until:
            next_ready_values.append(min(self._endpoint_backoff_until.values()))
        if self._rest_outage_until > now:
            next_ready_values.append(self._rest_outage_until)
        if not next_ready_values:
            return max(1.0, self.rest_poll_interval)
        next_ready = min(next_ready_values)
        delay = max(0.5, next_ready - now)
        base_delay = max(1.0, self.rest_poll_interval)
        return min(self._max_rest_interval, max(base_delay, delay))

    def _record_rest_latency(self, name: str, latency: float) -> None:
        bucket = self._rest_latency.setdefault(name, deque(maxlen=32))
        bucket.append(max(0.0, float(latency)))

    def rest_health(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {}
        now = time.time()
        for endpoint in self.endpoints:
            name = endpoint.name
            scores = {
                "score": round(self._endpoint_scores.get(name, 0.0), 3),
                "strikes": self._endpoint_failures.get(name, 0),
                "latency_ms": None,
                "backoff_until": self._endpoint_backoff_until.get(name, 0.0),
            }
            latencies = self._rest_latency.get(name)
            if latencies:
                scores["latency_ms"] = round(statistics.fmean(latencies) * 1000.0, 3)
            resume = scores["backoff_until"]
            if resume and resume <= now:
                scores["backoff_until"] = 0.0
            scores["cooldown_until"] = self._endpoint_health.get(name, EndpointHealth()).cooldown_until
            health[name] = scores
        return health

    def _init_rest_intervals(self) -> None:
        base = max(0.5, float(os.getenv("REST_POLL_INTERVAL", "3.0")))
        max_interval = max(base, float(os.getenv("REST_POLL_INTERVAL_MAX", "120.0")))
        growth_factor = max(1.1, float(os.getenv("REST_POLL_GROWTH_FACTOR", "1.6")))
        relax_factor = min(0.95, max(0.1, float(os.getenv("REST_POLL_RELAX_FACTOR", "0.7"))))
        relax_step = max(0.1, float(os.getenv("REST_POLL_RELAX_STEP", "1.0")))
        self._base_rest_interval = base
        self._max_rest_interval = max_interval
        self._rest_growth_factor = growth_factor
        self._rest_relax_factor = relax_factor
        self._rest_relax_step = relax_step
        self.rest_poll_interval = base
        self._rest_interval_last_change = time.time()
        self._rest_interval_reason = "init"

    def _grow_rest_interval(self, *, reason: str) -> None:
        previous = self.rest_poll_interval
        target = min(self._max_rest_interval, previous * self._rest_growth_factor)
        if target <= previous + 0.01:
            return
        self.rest_poll_interval = target
        self._rest_interval_last_change = time.time()
        self._rest_interval_reason = reason
        log_message(
            "market-stream",
            "REST polling slowed",
            severity="info",
            details={"interval": round(self.rest_poll_interval, 3), "reason": reason},
        )

    def _relax_rest_interval(self, *, reason: str) -> None:
        previous = self.rest_poll_interval
        if previous <= self._base_rest_interval:
            return
        reduced = max(self._base_rest_interval, previous * self._rest_relax_factor)
        if previous - reduced < 0.01:
            reduced = max(self._base_rest_interval, previous - self._rest_relax_step)
        if reduced >= previous:
            return
        self.rest_poll_interval = reduced
        self._rest_interval_last_change = time.time()
        self._rest_interval_reason = reason
        log_message(
            "market-stream",
            "REST polling relaxed",
            severity="debug",
            details={"interval": round(self.rest_poll_interval, 3), "reason": reason},
        )

    async def _timed_rest_fetch(self, endpoint: Endpoint, base: str, quote: str) -> Tuple[Endpoint, RestFetchResult, float]:
        start = time.time()
        result = await self._fetch_rest_price(endpoint, base, quote)
        latency = time.time() - start
        return endpoint, result, latency

    def _record_endpoint_success(self, name: str, *, price: Optional[float] = None, consensus: Optional[float] = None) -> None:
        self._endpoint_scores[name] = self._endpoint_scores.get(name, 0.0) + 1.0
        self._endpoint_failures[name] = 0
        health = self._endpoint_health.setdefault(name, EndpointHealth())
        health.register_success()
        if name in self._endpoint_backoff_until and health.cooldown_until == 0.0:
            self._endpoint_backoff_until.pop(name, None)
        if self._rest_outage_failures:
            self._clear_rest_outage()
        if self.rest_poll_interval > self._base_rest_interval:
            self._relax_rest_interval(reason="endpoint_success")
        if self._consensus_failures > 0:
            self._consensus_failures -= 1
        if price is not None and consensus is not None:
            self._update_source_stats(name, price, consensus)

    def _record_endpoint_failure(self, name: str) -> None:
        self._endpoint_scores[name] = self._endpoint_scores.get(name, 0.0) - 2.0
        self._endpoint_failures[name] = self._endpoint_failures.get(name, 0) + 1
        health = self._endpoint_health.setdefault(name, EndpointHealth())
        backoff = health.register_failure()
        previous = self._endpoint_backoff_until.get(name, 0.0)
        self._endpoint_backoff_until[name] = max(previous, health.cooldown_until)
        self.metrics.feedback(
            "market_stream",
            severity=FeedbackSeverity.WARNING,
            label="endpoint_failure",
            details={
                "endpoint": name,
                "strikes": health.strikes,
                "backoff_sec": round(max(0.0, health.cooldown_until - time.time()), 2),
            },
        )
        if self._endpoint_failures[name] in (3, 10):
            log_message(
                "market-stream",
                f"endpoint {name} failing consensus check",
                severity="warning",
                details={"strikes": self._endpoint_failures[name], "backoff_sec": round(backoff, 2)},
            )

    def _prune_recent_prices(self, now: Optional[float] = None) -> None:
        if not self._recent_price_by_source:
            return
        now = now or time.time()
        expiry = now - self.consensus_window * 3
        for source, (_, ts) in list(self._recent_price_by_source.items()):
            if ts < expiry:
                self._recent_price_by_source.pop(source, None)
        while len(self._recent_price_by_source) > 64:
            oldest = min(self._recent_price_by_source.items(), key=lambda item: item[1][1])
            self._recent_price_by_source.pop(oldest[0], None)

    def _seed_recent_reference(self, price: Optional[float] = None) -> None:
        value = price or self.reference_price
        if not value or value <= 0:
            return
        self._recent_price_by_source["reference"] = (value, time.time())

    def _handle_price_rejection(self, price: float, tolerance: float) -> bool:
        now = time.time()
        self._price_rejections.append((price, now))
        window = [val for val, ts in self._price_rejections if now - ts <= self.consensus_window]
        if len(window) < self._rejection_confirmations:
            return False
        median_price = statistics.median(window)
        span = max(window) - min(window)
        rel_span = span / max(median_price, 1e-9)
        if rel_span > tolerance:
            return False
        self.reference_price = median_price
        self._update_price_stats(median_price)
        self._seed_recent_reference(median_price)
        self.metrics.feedback(
            "market_stream",
            severity=FeedbackSeverity.INFO,
            label="reference_recenter",
            details={"symbol": self.symbol, "price": median_price, "span": rel_span},
        )
        self._price_rejections.clear()
        return True

    def _augment_consensus_sources(self, *, missing: int, now: float) -> List[Tuple[str, float]]:
        if missing <= 0:
            return []
        augmented: List[Tuple[str, float]] = []
        reference = self.reference_price
        if reference and reference > 0:
            self._recent_price_by_source.setdefault("reference", (reference, now))
            augmented.append(("reference", reference))
        offline_hit = self._offline_snapshot()
        if offline_hit:
            alias, snapshot = offline_hit
            ts = snapshot.ts or now
            key = f"offline:{alias}"
            self._recent_price_by_source[key] = (snapshot.price, ts)
            augmented.append((key, snapshot.price))
        return augmented

    def _consensus_confidence_score(
        self,
        values: Sequence[float],
        median_price: float,
        tolerance: float,
        min_sources: int,
    ) -> float:
        if not values or median_price <= 0:
            return 0.0
        spread = max(values) - min(values)
        rel_spread = spread / max(median_price, 1e-9)
        tightness = max(0.0, 1.0 - min(1.0, rel_spread / max(tolerance, 1e-9)))
        sample_bonus = min(1.0, len(values) / max(1, min_sources))
        score = max(0.05, min(1.0, tightness * sample_bonus))
        return float(round(score, 4))

    def _is_synthetic_source(self, source: str) -> bool:
        return source == "reference" or source.startswith(("offline:", "snapshot:"))

    def _live_source_count(self, now: Optional[float] = None) -> int:
        now = now or time.time()
        count = 0
        for source, (_, ts) in self._recent_price_by_source.items():
            if now - ts > self.consensus_window:
                continue
            if self._is_synthetic_source(source):
                continue
            count += 1
        return count

    def _mark_consensus_accepted(self, confidence: float) -> None:
        self._last_consensus_ts = time.time()
        self._consensus_relax_until = 0.0
        self._consensus_failures = 0
        self._last_consensus_confidence = max(0.0, float(confidence))
        self._consensus_initialized = True

    def _single_source_override_allowed(self, confidence: float) -> bool:
        if confidence <= 0:
            return False
        min_confidence = 0.25
        if not self._consensus_initialized:
            return confidence >= min_confidence
        now = time.time()
        if now < self._consensus_relax_until:
            return confidence >= min_confidence
        if now - self._last_consensus_ts >= self.consensus_timeout:
            return confidence >= min_confidence
        return False

    def _accept_consensus_price(self, consensus: float, *, confidence: float, live_sources: int) -> bool:
        if self._validate_price(consensus):
            self._mark_consensus_accepted(confidence)
            return True
        if live_sources < 2:
            if not self._single_source_override_allowed(confidence):
                return False
            self.reference_price = consensus
            self._update_price_stats(consensus)
            self._seed_recent_reference(consensus)
            self._price_rejections.clear()
            self._mark_consensus_accepted(confidence)
            self.metrics.feedback(
                "market_stream",
                severity=FeedbackSeverity.WARNING,
                label="reference_override_single",
                details={
                    "symbol": self.symbol,
                    "price": consensus,
                    "confidence": confidence,
                    "sources": live_sources,
                },
            )
            log_message(
                "market-stream",
                "reference updated from single-source consensus",
                severity="warning",
                details={"symbol": self.symbol, "price": consensus, "sources": live_sources},
            )
            return True
        self.reference_price = consensus
        self._update_price_stats(consensus)
        self._seed_recent_reference(consensus)
        self._price_rejections.clear()
        self._mark_consensus_accepted(confidence)
        self.metrics.feedback(
            "market_stream",
            severity=FeedbackSeverity.INFO,
            label="reference_override",
            details={
                "symbol": self.symbol,
                "price": consensus,
                "confidence": confidence,
                "sources": live_sources,
            },
        )
        log_message(
            "market-stream",
            "reference updated from multi-source consensus",
            severity="info",
            details={"symbol": self.symbol, "price": consensus, "sources": live_sources},
        )
        return True

    def _confirm_consensus(self, name: str, price: float) -> Tuple[bool, bool, Optional[float], float]:
        now = time.time()
        self._prune_recent_prices(now)
        if price <= 0:
            self._recent_price_by_source.pop(name, None)
            return False, False, None, 0.0
        self._recent_price_by_source[name] = (price, now)

        adjusted_values: List[Tuple[str, float]] = []
        for source, (value, ts) in list(self._recent_price_by_source.items()):
            if now - ts > self.consensus_window or value <= 0:
                continue
            bias_ratio = self._source_bias.get(source, 0.0)
            adjusted = value / (1.0 + bias_ratio)
            adjusted_values.append((source, adjusted))

        if not adjusted_values:
            return False, True, None, 0.0

        live_values = [val for source, val in adjusted_values if not self._is_synthetic_source(source)]
        min_required = min(self.consensus_sources, max(1, len(live_values)))
        missing = max(0, self.consensus_sources - len(live_values))
        if missing > 0:
            augmented = self._augment_consensus_sources(missing=missing, now=now)
            for source, value in augmented:
                bias_ratio = self._source_bias.get(source, 0.0)
                adjusted = value / (1.0 + bias_ratio)
                adjusted_values.append((source, adjusted))
        if time.time() < self._consensus_relax_until:
            min_required = 1
        values_only = [val for _, val in adjusted_values]
        consensus_values = live_values if len(live_values) == 1 else values_only
        median_price = statistics.median(consensus_values)
        tolerance = self._dynamic_tolerance(median_price)

        price_bias = self._source_bias.get(name, 0.0)
        price_adjusted = price / (1.0 + price_bias)
        diff = abs(price_adjusted - median_price) / max(median_price, 1e-9)

        if len(live_values) < min_required:
            return False, True, median_price, 0.0

        if diff <= tolerance:
            confidence = self._consensus_confidence_score(consensus_values, median_price, tolerance, min_required)
            for source, (value, ts) in list(self._recent_price_by_source.items()):
                if now - ts <= self.consensus_window and value > 0:
                    self._update_source_stats(source, value, median_price)
            return True, False, median_price, confidence

        overdue = now - self._last_consensus_ts >= self.consensus_timeout
        if overdue and diff <= tolerance * 2:
            confidence = max(
                0.2,
                0.8
                * self._consensus_confidence_score(
                    consensus_values,
                    median_price,
                    tolerance * 2,
                    max(1, min_required),
                ),
            )
            for source, (value, ts) in list(self._recent_price_by_source.items()):
                if now - ts <= self.consensus_window and value > 0:
                    self._update_source_stats(source, value, median_price)
            self._endpoint_scores[name] -= 0.5
            return True, False, median_price, confidence

        return False, False, median_price, 0.0

    async def _refresh_reference_price(self) -> None:
        if self._http_session is None:
            return
        if self._rest_outage_active():
            return
        base, quote = self._rest_base, self._rest_quote
        for endpoint in self._ranked_endpoints():
            if self._endpoint_backoff_until.get(endpoint.name, 0.0) > time.time():
                continue
            result = await self._fetch_rest_price(endpoint, base, quote)
            if result.error in {"network", "timeout"}:
                self._register_rest_outage(reason=result.error, endpoint=endpoint.name)
                continue
            if result.error in {"rate_limited_local", "unavailable"}:
                continue
            if result.error:
                self._record_endpoint_failure(endpoint.name)
                continue
            price = result.price
            if not price or price <= 0:
                self._record_endpoint_failure(endpoint.name)
                continue
            accepted, pending, consensus, confidence = self._confirm_consensus(endpoint.name, price)
            if pending:
                continue
            if not accepted or consensus is None:
                self._record_endpoint_failure(endpoint.name)
                continue
            live_sources = self._live_source_count()
            if not self._accept_consensus_price(
                consensus,
                confidence=confidence,
                live_sources=live_sources,
            ):
                self._record_endpoint_failure(endpoint.name)
                continue
            self._record_endpoint_success(endpoint.name, price=price, consensus=consensus)
            self._update_price_stats(consensus)
            self.reference_price = consensus
            self._seed_recent_reference(consensus)
            self._last_consensus_confidence = confidence
            log_message(
                "market-stream",
                f"reference price from {endpoint.name}",
                details={"price": consensus, "symbol": self.symbol},
            )
            return
        if self.reference_price is None:
            snapshot_price = self._load_snapshot_reference()
            if snapshot_price:
                self.reference_price = snapshot_price
                self._seed_recent_reference(snapshot_price)
                log_message(
                    "market-stream",
                    "using cached snapshot reference",
                    details={"symbol": self.symbol, "price": self.reference_price},
                )
            elif self._historical_reference:
                self.reference_price = self._historical_reference
                self._seed_recent_reference(self.reference_price)
                log_message(
                    "market-stream",
                    "using historical reference",
                    details={"symbol": self.symbol, "price": self.reference_price},
                )
            else:
                log_message("market-stream", "unable to establish reference price; will retry", severity="warning")
        else:
            self.reference_price = max(self.reference_price, 1e-9)

    async def _fetch_rest_price(self, endpoint: Endpoint, base: str, quote: str) -> RestFetchResult:
        if self._http_session is None or not endpoint.rest_template:
            return RestFetchResult(None, "unavailable")
        url = _render_rest(endpoint, base, quote)
        if not url:
            return RestFetchResult(None, "unavailable")
        headers = endpoint.headers or {}
        host = endpoint.name
        try:
            host = url.split("/")[2]
        except Exception:
            host = endpoint.name
        try:
            try:
                self.rate_limiter.acquire(host, tokens=1.0, timeout=5.0)
            except TimeoutError:
                log_message(
                    "market-stream",
                    f"rate limiter blocked REST call to {host}",
                    severity="warning",
                )
                return RestFetchResult(None, "rate_limited_local")
            async with self._http_session.get(url, timeout=10, headers=headers) as resp:
                if resp.status != 200:
                    error = "rate_limited" if resp.status in {418, 429, 451} else "http_error"
                    return RestFetchResult(None, error, status=resp.status)
                data = await resp.json()
        except (asyncio.TimeoutError, TimeoutError):
            return RestFetchResult(None, "timeout")
        except (aiohttp.ClientConnectorError, aiohttp.ClientConnectionError, aiohttp.ClientOSError, OSError):
            return RestFetchResult(None, "network")
        except Exception:
            return RestFetchResult(None, "exception")
        price = _extract_rest_price(endpoint.name, data, base, quote)
        if not price or price <= 0:
            return RestFetchResult(None, "invalid")
        normalized = self._normalize_live_price(price)
        if normalized <= 0:
            return RestFetchResult(None, "invalid")
        return RestFetchResult(normalized)

    def _select_next_endpoint(self) -> None:
        if not self.endpoints:
            return
        base, quote = self._rest_base, self._rest_quote
        candidates = self._ranked_endpoints()
        for ep in candidates:
            if self._endpoint_backoff_until.get(ep.name, 0.0) > time.time():
                continue
            ws_url = _render_ws(ep, base, quote)
            if ws_url:
                self.url = ws_url
                self.subscribe_template = ep.subscribe_template
                self.current_endpoint = ep.name
                log_message("market-stream", f"using {ep.name} endpoint", details={"url": self.url})
                self._ws_warning_logged = False
                return
        self.url = None
        self.current_endpoint = "unavailable"
        self._warn_websocket_unavailable()

    def _warn_websocket_unavailable(self) -> None:
        if self._ws_warning_logged:
            return
        self._debug(
            "ws_unavailable",
            extra={
                "template": bool(self._template),
                "subscribe": bool(self.subscribe_template),
                "url_set": bool(self.url),
                "wss_env": os.getenv("BASE_WSS_URL") or os.getenv("GLOBAL_WSS_URL"),
            },
        )
        log_message(
            "market-stream",
            "websocket URL unavailable; using REST consensus fallback",
            severity="warning",
            details={
                "symbol": self.symbol,
                "chain": self.chain,
                "template": bool(self._template),
                "subscribe_template": bool(self.subscribe_template),
                "url_set": bool(self.url),
                "wss_env": os.getenv("BASE_WSS_URL") or os.getenv("GLOBAL_WSS_URL"),
            },
        )
        self._ws_warning_logged = True

    def _load_historical_reference(self) -> Optional[float]:
        symbol_upper = self.symbol.upper()
        roots = [
            Path(os.getenv("HISTORICAL_DATA_DIR", "data/historical_ohlcv")),
            Path(os.getenv("HISTORICAL_DATA_ROOT", "data/historical_ohlcv")),
        ]
        for root in roots:
            if not root.exists():
                continue
            try:
                candidates = list(root.glob(f"*_{symbol_upper}.json"))
                if not candidates:
                    candidates = list(root.rglob(f"*_{symbol_upper}.json"))
            except Exception:
                candidates = []
            for json_file in candidates:
                try:
                    with json_file.open("r", encoding="utf-8") as handle:
                        rows = json.load(handle)
                except Exception:
                    continue
                if not isinstance(rows, list) or not rows:
                    continue
                tail = rows[-min(len(rows), 96) :]
                prices: List[float] = []
                for entry in tail[-16:]:
                    try:
                        price = float(entry.get("close") or entry.get("price") or 0.0)
                    except Exception:
                        price = 0.0
                    if price > 0:
                        prices.append(price)
                if prices:
                    return self._normalize_reference_value(float(statistics.fmean(prices)))
        return None

    def _load_snapshot_reference(self, *, allow_network: bool = True) -> Optional[float]:
        snapshot_path = Path(os.getenv("LOCAL_MARKET_CACHE", "data/market_snapshots.json")).expanduser()
        symbol_upper = self.symbol.upper()
        if "-" in symbol_upper or "/" in symbol_upper:
            candidates = {
                label.upper().replace("/", "-")
                for label in _pair_symbol_variants(self.symbol)
                if label
            }
        else:
            candidates = {symbol_upper}
        if not snapshot_path.exists():
            return self._prime_reference_from_local_market()
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        records = payload.get("data") or []
        token_prices: Dict[str, float] = {}

        def _snapshot_token_price(token: str) -> Optional[float]:
            token_u = token.upper()
            stable_price = _STABLE_TOKEN_PRICE.get(token_u)
            if stable_price is not None:
                return float(stable_price)
            for alias in _token_synonyms(token_u):
                value = token_prices.get(alias)
                if value:
                    return value
            return None
        for entry in records:
            try:
                entry_symbol = str(entry.get("symbol") or "").upper().replace("/", "-")
            except Exception:
                continue
            if entry_symbol not in candidates:
                price = entry.get("price_usd") or entry.get("priceUsd") or entry.get("price")
                try:
                    value = float(price)
                except (TypeError, ValueError):
                    continue
                if value <= 0:
                    continue
                if "-" not in entry_symbol:
                    token_prices.setdefault(entry_symbol, value)
                continue
            price = entry.get("price_usd") or entry.get("priceUsd") or entry.get("price")
            try:
                value = float(price)
            except (TypeError, ValueError):
                continue
            if value > 0:
                if "-" not in entry_symbol:
                    token_prices.setdefault(entry_symbol, value)
                return self._normalize_reference_value(value)
        if "-" in symbol_upper or "/" in symbol_upper:
            base, quote = _split_symbol(self.symbol)
            base_price = _snapshot_token_price(base)
            quote_price = _snapshot_token_price(quote)
            if base_price and quote_price:
                return self._normalize_reference_value(base_price / quote_price)
        if self._offline_enabled and self._offline_store:
            if "-" in symbol_upper or "/" in symbol_upper:
                offline_pair = self._offline_pair_snapshot()
                if offline_pair:
                    return self._normalize_reference_value(offline_pair[1].price)
            else:
                offline_token = self._offline_token_snapshot(symbol_upper)
                if offline_token:
                    token_price = _safe_float(getattr(offline_token, "price", 0.0))
                    if token_price > 0:
                        return self._normalize_reference_value(token_price)
        if allow_network:
            return self._prime_reference_from_local_market()
        return None

    def _prime_reference_from_local_market(self) -> Optional[float]:
        if getattr(self, "_snapshot_seed_attempted", False):
            return None
        self._snapshot_seed_attempted = True
        try:
            from services.public_api_clients import aggregate_market_data
        except Exception:
            return None
        try:
            snapshots = aggregate_market_data(symbols=[self.symbol], top_n=5)
        except Exception as exc:
            log_message(
                "market-stream",
                "snapshot seed failed",
                severity="warning",
                details={"symbol": self.symbol, "error": str(exc)},
            )
            return None
        symbol_upper = self.symbol.upper()
        for snap in snapshots:
            if getattr(snap, "symbol", "").upper() != symbol_upper:
                continue
            try:
                value = float(getattr(snap, "price_usd", 0.0))
            except Exception:
                value = 0.0
            if value <= 0:
                continue
            value = self._normalize_reference_value(value)
            log_message(
                "market-stream",
                "seeded reference from cached market data",
                severity="info",
                details={"symbol": self.symbol, "price": value},
            )
            return value
        return None

    async def _refresh_market_snapshots(self) -> None:
        if time.time() - self._last_snapshot_refresh < 60.0:
            return
        self._last_snapshot_refresh = time.time()
        try:
            from services.public_api_clients import aggregate_market_data
        except Exception:
            return
        symbol = self.symbol
        try:
            snapshots = await asyncio.to_thread(aggregate_market_data, symbols=[symbol], top_n=3)
        except Exception:
            return
        symbol_upper = symbol.upper()
        now = time.time()
        for snap in snapshots:
            if getattr(snap, "symbol", "").upper() != symbol_upper:
                continue
            try:
                value = float(getattr(snap, "price_usd", 0.0))
            except Exception:
                value = 0.0
            if value <= 0:
                continue
            value = self._normalize_reference_value(value)
            self._recent_price_by_source[f"snapshot:{snap.source}"] = (value, now)
            if not self.reference_price:
                self.reference_price = value
        return
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
    if endpoint.name in {"bitstamp", "mexc", "dexscreener"}:
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
        if name == "dexscreener":
            pairs = payload.get("pairs") or []
            best_ratio: Optional[float] = None
            best_liquidity = 0.0
            base_synonyms = _token_synonyms(base)
            quote_synonyms = _token_synonyms(quote)
            stable_quotes = {"USDT", "USDC", "BUSD", "USD"}
            for pair in pairs:
                base_info = pair.get("baseToken") or {}
                quote_info = pair.get("quoteToken") or {}
                base_symbol = str(base_info.get("symbol") or "").upper()
                quote_symbol = str(quote_info.get("symbol") or "").upper()
                if base_symbol not in base_synonyms or quote_symbol not in quote_synonyms:
                    continue
                ratio = pair.get("priceNative")
                if quote in stable_quotes:
                    usd_price = _safe_float(pair.get("priceUsd"))
                    if usd_price > 0:
                        ratio = usd_price
                if ratio is None:
                    base_usd = _safe_float(pair.get("priceUsd"))
                    quote_liq = _safe_float(pair.get("liquidity", {}).get("quote"))
                    base_liq = _safe_float(pair.get("liquidity", {}).get("base"))
                    if base_usd > 0 and quote_liq > 0 and base_liq > 0:
                        ratio = base_liq / max(quote_liq, 1e-12)
                ratio_val = _safe_float(ratio)
                liquidity_usd = _safe_float(pair.get("liquidity", {}).get("usd"))
                if ratio_val > 0 and liquidity_usd >= best_liquidity:
                    best_ratio = ratio_val
                    best_liquidity = liquidity_usd
            if best_ratio and best_ratio > 0:
                return float(best_ratio)
    except Exception:
        return None
    return None
