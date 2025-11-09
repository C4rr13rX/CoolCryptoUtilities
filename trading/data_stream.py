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


def _token_synonyms(token: str) -> set[str]:
    token_u = token.upper()
    synonyms = {token_u}
    for original, normalized in TOKEN_NORMALIZATION.items():
        if normalized.upper() == token_u:
            synonyms.add(original.upper())
    return synonyms


@dataclass
class Endpoint:
    name: str
    ws_template: Optional[str]
    subscribe_template: Optional[str]
    rest_template: Optional[str]
    headers: Optional[Dict[str, str]] = None


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
        self._endpoint_health: Dict[str, EndpointHealth] = {ep.name: EndpointHealth() for ep in self.endpoints}
        self.rest_poll_interval = float(os.getenv("REST_POLL_INTERVAL", "5"))
        self._base_rest_interval = self.rest_poll_interval
        self._last_volume: float = 0.0
        self.consensus_sources = int(os.getenv("PRICE_CONSENSUS_SOURCES", "2"))
        self.consensus_window = float(os.getenv("PRICE_CONSENSUS_WINDOW", "60"))
        self.consensus_timeout = float(os.getenv("PRICE_CONSENSUS_TIMEOUT", "45"))
        self._recent_price_by_source: Dict[str, Tuple[float, float]] = {}
        self._last_consensus_ts = time.time()
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

    def register(self, callback: CallbackType) -> None:
        self._callbacks.append(callback)

    async def start(self) -> None:
        self._stop_event.clear()
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        await self._refresh_reference_price()
        if self._onchain_listener and self._onchain_listener.available and not self._onchain_task:
            self._onchain_queue = asyncio.Queue()
            self._onchain_task = asyncio.create_task(self._onchain_listener.run(self._onchain_queue.put))
            self._onchain_consumer = asyncio.create_task(self._consume_onchain_queue())
        backoff = 5.0
        while not self._stop_event.is_set():
            if not self.url:
                self._warn_websocket_unavailable()
                await self._poll_rest_data(max(backoff, self.consensus_timeout))
                if self._template:
                    symbol_lower = self.symbol.lower().replace("/", "-")
                    self.url = self._template.format(symbol=symbol_lower, SYMBOL=self.symbol.upper(), pair=symbol_lower)
                if not self.url:
                    self._select_next_endpoint()
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
                self._select_next_endpoint()
            except Exception as exc:  # pragma: no cover - network dependent
                log_message("market-stream", f"websocket error {exc}; retrying in {backoff:.1f}s", severity="warning")
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
                        log_message("market-stream", "rate-limit notice from server; backing off 15s", severity="warning")
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
            endpoints = self._eligible_rest_endpoints()
            if not endpoints:
                now = time.time()
                if now - self._last_rest_unavailable_log >= 30.0:
                    log_message(
                        "market-stream",
                        "all REST endpoints cooling down.",
                        severity="warning",
                        details={"symbol": self.symbol},
                    )
                    self._last_rest_unavailable_log = now
                await self._handle_consensus_failure()
                await asyncio.sleep(min(self.rest_poll_interval, 2.0))
                continue
            attempts = 0
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
                for endpoint, price, latency in results:
                    self._record_rest_latency(endpoint.name, latency)
                    if not price or price <= 0:
                        self._record_endpoint_failure(endpoint.name)
                        continue
                    accepted, pending, consensus, confidence = self._confirm_consensus(endpoint.name, price)
                    if pending:
                        continue
                    if not accepted or consensus is None:
                        self._record_endpoint_failure(endpoint.name)
                        continue
                    if not self._validate_price(consensus):
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
            if not dispatched:
                log_message(
                    "market-stream",
                    "unable to obtain consensus price during REST poll cycle.",
                    severity="warning",
                    details={"symbol": self.symbol, "active_endpoints": len(endpoints)},
                )
                await self._handle_consensus_failure()
            self._last_rest_health = self.rest_health()
            await asyncio.sleep(self.rest_poll_interval)

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

    async def _handle_consensus_failure(self) -> None:
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
        if fallback_price:
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
                return
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
        if self._consensus_failures % 4 == 0:
            limit = self._base_rest_interval * 6
            self.rest_poll_interval = min(limit, self.rest_poll_interval * 1.25)
            log_message(
                "market-stream",
                "consensus failure throttling REST polls",
                severity="warning",
                details={"symbol": self.symbol, "rest_interval": self.rest_poll_interval, "failures": self._consensus_failures},
            )

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
        seen = set()
        for label in self._offline_symbol_candidates():
            if label in seen:
                continue
            seen.add(label)
            snapshot = self._offline_store.get_price(label)
            if snapshot:
                return label, snapshot
        return None

    def _offline_symbol_candidates(self) -> List[str]:
        symbol_upper = self.symbol.upper()
        candidates: List[str] = [symbol_upper]
        raw_parts: List[str] = []
        if "-" in symbol_upper:
            raw_parts = symbol_upper.split("-")
        elif "/" in symbol_upper:
            raw_parts = symbol_upper.split("/")
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
        symbol_clean = self.symbol.replace("/", "-")
        parts = [p.strip() for p in symbol_clean.split("-") if p.strip()]
        raw_base = parts[0].upper() if parts else "ETH"
        raw_quote = parts[1].upper() if len(parts) > 1 else "USD"
        base, quote = _split_symbol(self.symbol)
        endpoints: List[Endpoint] = []
        binance_symbol = _binance_symbol(base, quote)
        if binance_symbol and (base, quote) in BINANCE_MARKETS:
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
        dex_query = quote_plus(f"{raw_base} {raw_quote}")
        endpoints.append(
            Endpoint(
                name="dexscreener",
                ws_template=None,
                subscribe_template=None,
                rest_template=f"https://api.dexscreener.com/latest/dex/search?q={dex_query}",
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

    async def _timed_rest_fetch(self, endpoint: Endpoint, base: str, quote: str) -> Tuple[Endpoint, Optional[float], float]:
        start = time.time()
        price = await self._fetch_rest_price(endpoint, base, quote)
        latency = time.time() - start
        return endpoint, price, latency

    def _record_endpoint_success(self, name: str, *, price: Optional[float] = None, consensus: Optional[float] = None) -> None:
        self._endpoint_scores[name] = self._endpoint_scores.get(name, 0.0) + 1.0
        self._endpoint_failures[name] = 0
        health = self._endpoint_health.setdefault(name, EndpointHealth())
        health.register_success()
        if name in self._endpoint_backoff_until and health.cooldown_until == 0.0:
            self._endpoint_backoff_until.pop(name, None)
        if self.rest_poll_interval > self._base_rest_interval:
            self.rest_poll_interval = max(self._base_rest_interval, self.rest_poll_interval * 0.8)
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
        if len(window) < max(3, self._price_rejections.maxlen // 2):
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

        min_required = min(self.consensus_sources, max(1, len(adjusted_values)))
        missing = max(0, self.consensus_sources - len(adjusted_values))
        if missing > 0:
            augmented = self._augment_consensus_sources(missing=missing, now=now)
            for source, value in augmented:
                bias_ratio = self._source_bias.get(source, 0.0)
                adjusted = value / (1.0 + bias_ratio)
                adjusted_values.append((source, adjusted))
            min_required = min(self.consensus_sources, max(1, len(adjusted_values)))
        if time.time() < self._consensus_relax_until:
            min_required = 1
        values_only = [val for _, val in adjusted_values]
        median_price = statistics.median(values_only)
        tolerance = self._dynamic_tolerance(median_price)

        price_bias = self._source_bias.get(name, 0.0)
        price_adjusted = price / (1.0 + price_bias)
        diff = abs(price_adjusted - median_price) / max(median_price, 1e-9)

        if len(adjusted_values) < min_required:
            return False, True, median_price, 0.0

        if diff <= tolerance:
            confidence = self._consensus_confidence_score(values_only, median_price, tolerance, min_required)
            self._last_consensus_ts = now
            self._consensus_relax_until = 0.0
            self._consensus_failures = 0
            for source, (value, ts) in list(self._recent_price_by_source.items()):
                if now - ts <= self.consensus_window and value > 0:
                    self._update_source_stats(source, value, median_price)
            self._last_consensus_confidence = confidence
            return True, False, median_price, confidence

        overdue = now - self._last_consensus_ts >= self.consensus_timeout
        if overdue and diff <= tolerance * 2:
            confidence = max(
                0.2,
                0.8
                * self._consensus_confidence_score(
                    values_only,
                    median_price,
                    tolerance * 2,
                    max(1, min_required),
                ),
            )
            self._last_consensus_ts = now
            self._consensus_relax_until = 0.0
            self._consensus_failures = 0
            for source, (value, ts) in list(self._recent_price_by_source.items()):
                if now - ts <= self.consensus_window and value > 0:
                    self._update_source_stats(source, value, median_price)
            self._endpoint_scores[name] -= 0.5
            self._last_consensus_confidence = confidence
            return True, False, median_price, confidence

        return False, False, median_price, 0.0

    async def _refresh_reference_price(self) -> None:
        if self._http_session is None:
            return
        base, quote = _split_symbol(self.symbol)
        for endpoint in self._ranked_endpoints():
            if self._endpoint_backoff_until.get(endpoint.name, 0.0) > time.time():
                continue
            price = await self._fetch_rest_price(endpoint, base, quote)
            if not price or price <= 0:
                self._record_endpoint_failure(endpoint.name)
                continue
            accepted, pending, consensus, confidence = self._confirm_consensus(endpoint.name, price)
            if pending:
                continue
            if not accepted or consensus is None:
                self._record_endpoint_failure(endpoint.name)
                continue
            if not self._validate_price(consensus):
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

    async def _fetch_rest_price(self, endpoint: Endpoint, base: str, quote: str) -> Optional[float]:
        if self._http_session is None or not endpoint.rest_template:
            return None
        url = _render_rest(endpoint, base, quote)
        if not url:
            return None
        headers = endpoint.headers or {}
        try:
            host = url.split("/")[2]
            try:
                self.rate_limiter.acquire(host, tokens=1.0, timeout=5.0)
            except TimeoutError:
                log_message(
                    "market-stream",
                    f"rate limiter blocked REST call to {host}",
                    severity="warning",
                )
                return None
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
        log_message(
            "market-stream",
            "websocket URL unavailable; using REST consensus fallback",
            severity="warning",
            details={"symbol": self.symbol},
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
                    return float(statistics.fmean(prices))
        return None

    def _load_snapshot_reference(self) -> Optional[float]:
        snapshot_path = Path(os.getenv("LOCAL_MARKET_CACHE", "data/market_snapshots.json")).expanduser()
        symbol_upper = self.symbol.upper()
        if not snapshot_path.exists():
            return self._prime_reference_from_local_market()
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        records = payload.get("data") or []
        for entry in records:
            try:
                entry_symbol = str(entry.get("symbol") or "").upper()
            except Exception:
                continue
            if entry_symbol != symbol_upper:
                continue
            price = entry.get("price_usd") or entry.get("priceUsd")
            try:
                value = float(price)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return self._prime_reference_from_local_market()

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
