from __future__ import annotations

import asyncio
import functools
import json
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import math

from dataclasses import dataclass
from urllib.parse import quote_plus
import numpy as np
import requests

from router_wallet import CHAINS
from db import TradingDatabase, get_db
from trading.bot import TradingBot
from trading.data_stream import (
    MarketDataStream,
    _split_symbol,
    has_price_endpoints,
    TOKEN_NORMALIZATION,
)
from trading.pipeline import TrainingPipeline, ghost_reason_is_earned
from trading.portfolio import PortfolioState
from trading.constants import PRIMARY_CHAIN, PRIMARY_SYMBOL, top_pairs, pair_index_entries
from trading.constants import PRIMARY_CHAIN, PRIMARY_SYMBOL
from services.logging_utils import log_message
from trading.ghost_limits import resolve_pair_limit
from services.watchlists import load_watchlists
from services.background_workers import _ensure_assignment_template, _update_assignment, _run_download
from services.trading_accounting import is_usd_accounting_symbol

STABLE_TOKENS = {"USDC", "USDT", "DAI", "BUSD", "TUSD", "USDP", "USDD", "USDS", "GUSD"}

#: Refuse pairs whose P&L cannot be denominated in USD, at SELECTION.
#:
#: `trading/bot.py:6046` already refuses them -- `is_usd_accounting_pair` is
#: the first thing the decision path asks, and a pair that fails it can never
#: reach an entry however the market moves. Nothing upstream asked the same
#: question, so the feed spent its bandwidth streaming pairs whose only
#: possible outcome was `hold-price-domain`.
#:
#: MEASURED 2026-09-06 over 24h of `market_stream`: 13 of 42 streamed symbols
#: were non-USD-accounting and took 1793 of 4567 ticks -- 39.3% of the feed --
#: CBETH-WETH (175), CBETH-CBBTC (162), AERO-WETH (158), VVV-WETH (147),
#: USDT-USDC (147, stable base), VIRTUAL-WETH, EURC-WETH, MORPHO-WETH,
#: EURC-USDC, JITOSOL-CBBTC, SOL-CBBTC, TIBBIR-VIRTUAL, DAI-USDC. Over the
#: same window `organism_snapshots` shows 151 of 606 decisions (24.9%) ending
#: in `hold-price-domain` on exactly these symbols.
#:
#: That bandwidth is not free. Entries AND exits here are sample-driven -- a
#: position is only marked out when a tick for its symbol arrives -- so ticks
#: spent on an untradeable pair are ticks the tradeable ones did not get, and
#: the symbols that CAN trade were down at 2-6 ticks/hour (BASEMATE-USDC 2,
#: BST-USDC 4, TONY-USDC 6, CBZEC-USDC 6) against USDT-USDC's 51. Each of the
#: 13 also holds a slot against `select_pairs`'s limit.
#:
#: HELD POSITIONS ARE EXEMPT wherever this is applied. A position already open
#: in a non-USD pair needs its feed to be closed at all; dropping the stream
#: would strand it exactly as the dark-feed positions were stranded.
_REQUIRE_USD_ACCOUNTING = os.getenv(
    "SELECTOR_REQUIRE_USD_ACCOUNTING", "1"
).strip().lower() in {"1", "true", "yes", "on"}


def _can_denominate_pnl_in_usd(symbol: str) -> bool:
    """Selector-side gate; the env var is an escape hatch, not a default."""
    if not _REQUIRE_USD_ACCOUNTING:
        return True
    return is_usd_accounting_symbol(symbol)


#: Rank a symbol DOWN when no strategy may open a position in it.
#:
#: A bot slot is not a subscription, it is the decision cycle. Every entry
#: rule, every exit rule and every ghost round trip in this system hangs off
#: `TradingBot._handle_sample`, and only a bot calls it -- a data-only stream
#: publishes prices and decides nothing. So the pool's slot assignment IS the
#: allocation of decision cycles, and it was made on volume and volatility
#: alone: `select_pairs` ranks the tail of the candidate list by market
#: activity and never asks whether anything is allowed to trade the symbol.
#:
#: MEASURED 2026-09-07 over 6h, joining 596 `organism_snapshots` cycles to
#: 2968 `market_stream` ticks and to the four standing gates:
#:
#:     COMP-USDC       136 cycles   symbol_edge     -4.333% over 16 trips
#:     AERO-USDC       116          (atf_static only -- KEPT, see below)
#:     CBETH-USDC       90          symbol_edge + symbol_motion
#:     CLANKER-USDC     74          stop_survivability
#:     CBETH-CBBTC      45          stop_survivability + symbol_motion
#:     BASECAT/JITOSOL/CBBTC/VIRTUAL-WETH  34
#:                                  --
#:                                  379 of 596 (63.6%)
#:
#: Those 379 cycles produced zero entries. They cannot: the refusals above are
#: SYMBOL-level and unconditional, so every strategy dies on them. Meanwhile
#: the eight symbols `atf_static` -- the only executor with a live branch --
#: may actually enter carried 43.7% of the ticks and got 15.6% of the cycles:
#: COMP 0.37 cycles per tick against CBZEC-USDC's 0.018, a 20x skew toward
#: symbols nothing may buy. The whole distance to a live trade is atf_static's
#: fresh tradeable ghost round trips (1, against a re-arm bar of 20), and they
#: are produced by decision cycles.
#:
#: THIS IS A RANKING, NOT A GATE. A condemned symbol sinks below the eligible
#: ones and still takes a slot when there is nothing better to give it to, so
#: a pool larger than the candidate list behaves exactly as before and no
#: symbol is switched off. The three carve-outs are load-bearing:
#:
#:   * HELD POSITIONS AND ATF PRIORITIES ARE NEVER RANKED. They are promoted
#:     ahead of this list by the callers and never reach it. A held symbol
#:     without a bot is a position nothing can sell -- this repo has stranded
#:     one for 362.9h that way.
#:   * The POOLED symbol_edge verdict, never the per-strategy one. AERO-USDC
#:     is banned for atf_static (n=17, mean -0.992%) and ALLOWED pooled
#:     (n=46, mean +1.805%), so some strategy may still enter it and it keeps
#:     its rank. Asking the per-strategy question here would condemn a symbol
#:     on one strategy's record.
#:   * FAILS OPEN. A gate that cannot be imported, or that raises, leaves the
#:     symbol unranked -- the failure this exists to prevent is a wasted
#:     slot, never a narrower funnel.
_DEPRIORITISE_CONDEMNED = os.getenv(
    "SELECTOR_DEPRIORITISE_CONDEMNED", "1"
).strip().lower() in {"1", "true", "yes", "on"}


def _no_strategy_may_enter(symbol: str) -> Optional[str]:
    """The standing, symbol-level refusal that condemns ``symbol``, or None.

    Reads the same three functions ``trading/bot.py`` consults at the entry
    gate (:7707 pooled symbol edge, :7957 stop survivability, and the symbol
    motion gate beside them). Only refusals that take a SYMBOL and no strategy
    are asked, because only those condemn every strategy at once.
    """
    if not _DEPRIORITISE_CONDEMNED:
        return None
    pair = str(symbol or "").strip().upper()
    if not pair:
        return None
    try:
        from services.symbol_edge_gate import refusal_reason as _edge
    except Exception:  # noqa: BLE001 - unreadable verdict never condemns
        def _edge(_symbol: str, _strategy_id=None):  # type: ignore[misc]
            return None
    try:
        from services.stop_survivability_gate import refusal_reason as _stop
    except Exception:  # noqa: BLE001
        def _stop(_symbol: str):  # type: ignore[misc]
            return None
    try:
        from services.symbol_motion_gate import refusal_reason as _motion
    except Exception:  # noqa: BLE001
        def _motion(_symbol: str):  # type: ignore[misc]
            return None
    try:
        return _edge(pair, None) or _stop(pair) or _motion(pair) or None
    except Exception:  # noqa: BLE001
        return None


def _sink_condemned(
    candidates: List["PairCandidate"], *, protected: Optional[set] = None
) -> Tuple[List["PairCandidate"], List[str]]:
    """Move condemned candidates to the back, order preserved within groups.

    Returns ``(reordered, condemned_symbols)``. ``protected`` names symbols
    that keep their place however they are judged -- held positions, which
    the callers also promote to the front.
    """
    keep = {str(s or "").strip().upper() for s in (protected or set())}
    eligible: List["PairCandidate"] = []
    condemned: List["PairCandidate"] = []
    names: List[str] = []
    for candidate in candidates:
        symbol_u = str(getattr(candidate, "symbol", "") or "").upper()
        if symbol_u and symbol_u not in keep and _no_strategy_may_enter(symbol_u):
            condemned.append(candidate)
            names.append(symbol_u)
        else:
            eligible.append(candidate)
    if not condemned:
        return list(candidates), []
    return eligible + condemned, names

# Pair selection is blocking by nature -- live-price probes, CEX backfills,
# `proc.wait()` on download2000 -- and `reconcile_pairs` used to run it inline
# on the event loop that every market stream shares. That is the other half of
# the frozen feed: market_stream shows 20-45 minute holes all through
# 2026-09-01 while production was up the whole time.
#
# It gets its OWN single worker rather than asyncio's default executor.
# aiohttp resolves every hostname through loop.getaddrinfo() on that default
# pool (aiodns is not installed here), so parking a minutes-long selection
# there would starve DNS for the streams -- the exact failure documented in
# _fetch_rest_price, arriving by a different road. One worker also serialises
# concurrent reconciles, which is what we want: they duplicate each other's
# work.
_SELECTION_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pair-select")

DEFAULT_LIVE_PAIRS: List[str] = top_pairs(limit=6) or [PRIMARY_SYMBOL]
if PRIMARY_SYMBOL not in DEFAULT_LIVE_PAIRS:
    DEFAULT_LIVE_PAIRS.insert(0, PRIMARY_SYMBOL)

@dataclass
class PairCandidate:
    symbol: str
    tokens: List[str]
    avg_volume: float
    volatility: float
    score: float
    datapath: Path


_LIVE_PAIR_CACHE: Dict[str, bool] = {}
#: Suppressed pairs already reported this process, so the ~320 known-dead
#: pairs are announced once instead of on every selection pass.
_SUPPRESSION_LOGGED: set[str] = set()
_SUPPRESSION_TTL = float(os.getenv("PAIR_SUPPRESSION_TTL", str(6 * 3600)))

#: Disk-backed memory of pairs confirmed tradeable, so a restart does not
#: re-probe every one of them before any stream can start.
_LIVE_CACHE_PATH = Path(os.getenv("LIVE_PAIR_CACHE_PATH", "data/live_pairs.json"))
_LIVE_CACHE_TTL = float(os.getenv("LIVE_PAIR_CACHE_TTL", str(3600)))
_LIVE_CACHE_MEM: Optional[Dict[str, float]] = None


def _live_cache_load() -> Dict[str, float]:
    global _LIVE_CACHE_MEM
    if _LIVE_CACHE_MEM is None:
        try:
            _LIVE_CACHE_MEM = json.loads(_LIVE_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _LIVE_CACHE_MEM = {}
    return _LIVE_CACHE_MEM


def _live_cache_get(key: str) -> bool:
    """Was this pair confirmed live recently enough to trust without probing?"""
    try:
        seen = float(_live_cache_load().get(key, 0.0))
    except Exception:
        return False
    return bool(seen) and (time.time() - seen) < _LIVE_CACHE_TTL


def _live_cache_put(key: str) -> None:
    """Remember a confirmed-live pair. Best effort: never break selection."""
    try:
        cache = _live_cache_load()
        cache[key] = time.time()
        cutoff = time.time() - _LIVE_CACHE_TTL * 4
        for stale in [k for k, v in cache.items() if float(v or 0) < cutoff]:
            cache.pop(stale, None)
        _LIVE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LIVE_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass

_ALWAYS_LIVE_SYMBOLS = {
    "WETH-USDC",
    "USDC-WETH",
    "WETH-USDT",
    "USDT-WETH",
    "DAI-WETH",
    "WETH-DAI",
    "USDC-USDT",
}
_DB_BOOTSTRAPPED = False


class _LazyDB:
    def __getattr__(self, name: str):
        return getattr(get_db(), name)


_db: TradingDatabase = _LazyDB()  # type: ignore[assignment]


def _bootstrap_always_live_symbols() -> None:
    global _DB_BOOTSTRAPPED
    if _DB_BOOTSTRAPPED:
        return
    _DB_BOOTSTRAPPED = True
    for _core_symbol in list(_ALWAYS_LIVE_SYMBOLS):
        try:
            _db.clear_pair_suppression(_core_symbol)
        except Exception:
            pass

_PAIR_SCAN_VOLUME_LIMIT = int(os.getenv("PAIR_SCAN_VOLUME_LIMIT", "2000"))
_PAIR_CHAIN_LOCK_SECONDS = float(os.getenv("PAIR_CHAIN_LOCK_SECONDS", str(10 * 24 * 3600)))
_PAIR_MIN_VOL_SCORE = float(os.getenv("PAIR_MIN_VOL_SCORE", "0.05"))
_PAIR_MAX_SPREAD_SCORE = float(os.getenv("PAIR_MAX_SPREAD_SCORE", "0.03"))
_LOW_COST_CHAIN_ORDER = [
    "base",
    "arbitrum",
    "optimism",
    "polygon",
    "bsc",
    "ethereum",
]

def _load_top_symbols(limit: int = 100, *, chain: Optional[str] = None) -> List[str]:
    return top_pairs(limit=limit, chain=chain)




def _pair_key(symbol: str, chain: str) -> str:
    return f"{chain.lower()}::{symbol.upper()}"


def _token_synonyms(token: str) -> set[str]:
    token_u = token.upper()
    synonyms = {token_u}
    for original, normalized in TOKEN_NORMALIZATION.items():
        if normalized.upper() == token_u:
            synonyms.add(original.upper())
    return synonyms


def _probe_dexscreener(symbol: str) -> Optional[bool]:
    base, quote = _split_symbol(symbol)
    query = quote_plus(f"{base} {quote}")
    url = f"https://api.dexscreener.com/latest/dex/search?q={query}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None
        payload = resp.json()
    except Exception:
        return None
    pairs = payload.get("pairs") or []
    if not pairs:
        return False
    base_syn = _token_synonyms(base)
    quote_syn = _token_synonyms(quote)
    def _matches(pair_base: str, pair_quote: str) -> bool:
        return pair_base in base_syn and pair_quote in quote_syn
    for pair in pairs:
        base_info = pair.get("baseToken") or {}
        quote_info = pair.get("quoteToken") or {}
        base_symbol = str(base_info.get("symbol") or "").upper()
        quote_symbol = str(quote_info.get("symbol") or "").upper()
        if not (_matches(base_symbol, quote_symbol) or _matches(quote_symbol, base_symbol)):
            continue
        price = pair.get("priceNative") or pair.get("priceUsd")
        try:
            if price and float(price) > 0:
                return True
        except Exception:
            continue
    return False


def _has_historical_price(symbol: str) -> bool:
    data_dir = Path(os.getenv("HISTORICAL_DATA_ROOT", "data/historical_ohlcv")) / PRIMARY_CHAIN
    symbol_upper = symbol.upper()
    try:
        for json_file in data_dir.glob(f"*_{symbol_upper}.json"):
            with json_file.open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            if isinstance(rows, list) and rows:
                price = float(rows[-1].get("close") or rows[-1].get("price") or 0.0)
                if price > 0:
                    return True
    except Exception:
        return False
    return False



def _safe_print(message: str) -> None:
    """print() that cannot raise on a console that lacks the codepage.

    Arbitrary on-chain token symbols reach the logs, so every diagnostic here
    must survive a cp1252 stdout.
    """
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(message.encode(encoding, "replace").decode(encoding, "replace"))
    except Exception:
        pass


def _has_live_price(symbol: str, chain: str = PRIMARY_CHAIN) -> bool:
    _bootstrap_always_live_symbols()
    key = _pair_key(symbol, chain)
    symbol_u = symbol.upper()
    if symbol_u in _ALWAYS_LIVE_SYMBOLS:
        _LIVE_PAIR_CACHE[key] = True
        return True
    # One lookup, not two. `is_pair_suppressed` already fetches the record and
    # `_has_live_price` then fetched it again, so every one of the ~320
    # suppressed pairs cost two DB round trips on every selection pass. That is
    # most of the ~10 minutes production spends in pair enumeration before a
    # single market stream starts.
    record = _db.get_pair_suppression(key) or {}
    _suppressed = bool(record) and float(record.get("release_ts") or 0.0) > time.time()
    if record and not _suppressed:
        _db.clear_pair_suppression(key)
    if _suppressed:
        remaining = float(record.get("release_ts", 0.0)) - time.time()
        wait_minutes = max(0.0, remaining / 60.0)
        reason = record.get("reason") or "suppressed"
        # Windows consoles default to cp1252, and token symbols are arbitrary
        # on-chain strings: one pair with a non-ASCII name raised
        # UnicodeEncodeError here and took down pair selection -- and with it
        # the whole production manager, which then crash-looped every ~3
        # minutes under the supervisor. A diagnostic print must never be able
        # to stop trading, so degrade the characters rather than the process.
        # Log each suppressed pair once per process, not on every selection
        # pass. There are ~320 suppressed pairs and selection runs repeatedly,
        # so this printed 320 lines per pass forever -- it buried real startup
        # diagnostics and made a cheap cache lookup look like expensive work
        # while streams were still waiting to start.
        if key not in _SUPPRESSION_LOGGED:
            _SUPPRESSION_LOGGED.add(key)
            _safe_print(
                f"[pair-select] suppressed {symbol_u}: {reason}; "
                f"retry in ~{wait_minutes:.1f} min."
            )
        _LIVE_PAIR_CACHE[key] = False
        return False

    cached = _LIVE_PAIR_CACHE.get(key)
    if cached is not None:
        return cached

    # Confirmed-live pairs are remembered ACROSS restarts, not just in memory.
    #
    # Suppressions were already persisted, but positive results were not, so
    # every restart re-probed every tradeable pair from scratch: one live HTTP
    # call each, serially, on the main thread, before a single market stream
    # could start. Measured at 0.53s per probe across ~320 candidates -- about
    # 2.8 minutes of dead time per restart, paid again on the next one.
    #
    # A short TTL keeps this honest: a pair that stops trading is re-checked
    # within the hour rather than trusted forever.
    if _live_cache_get(key):
        _LIVE_PAIR_CACHE[key] = True
        return True

    probe_result = _probe_dexscreener(symbol_u)
    if probe_result:
        _LIVE_PAIR_CACHE[key] = True
        _live_cache_put(key)
        _db.clear_pair_suppression(key)
        return True
    if probe_result is None:
        _safe_print(f"[pair-select] probe deferred for {symbol_u}: dexscreener unreachable.")
        result = _has_historical_price(symbol_u)
        _LIVE_PAIR_CACHE[key] = result
        return result
    result = _has_historical_price(symbol_u)
    _LIVE_PAIR_CACHE[key] = result
    if not result:
        _db.record_pair_suppression(
            key,
            "no_live_market_data",
            ttl_seconds=_SUPPRESSION_TTL,
            metadata={"checked_at": time.time()},
        )
        _safe_print(f"[pair-select] skipping {symbol_u}: no live market data sources responded.")
    return result


def _has_streaming_feed(symbol: str, chain: str) -> bool:
    """
    Lightweight readiness check: ensure we have at least one endpoint or
    offline fallback to stream live price updates for the pair.
    """
    # Deliberately does NOT construct a MarketDataStream. This runs for every
    # candidate pair on every selection pass -- roughly 25 times a minute --
    # and a full construction is ~218x more expensive than the endpoint check
    # it is standing in for. Building and discarding streams here is what took
    # the production process to 77 threads and 871MB while starting almost no
    # real streams, starving the ones that mattered of samples.
    try:
        if has_price_endpoints(symbol):
            return True
    except Exception:
        return False
    # Parity with the previous implementation: an enabled offline store made
    # this return True regardless of symbol, since OfflinePriceStore is
    # symbol-independent. Kept so this change is a pure performance fix and
    # does not quietly tighten which pairs are eligible.
    return os.getenv("MARKET_OFFLINE_FALLBACK", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _ohlcv_exists(symbol: str, chain: str, data_root: Optional[Path] = None) -> bool:
    root = data_root or Path(os.getenv("HISTORICAL_DATA_ROOT", "data/historical_ohlcv"))
    data_dir = root / chain
    symbol_upper = symbol.upper()
    for json_file in data_dir.glob(f"*_{symbol_upper}.json"):
        try:
            with json_file.open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            if isinstance(rows, list) and rows:
                return True
        except Exception:
            continue
    return False


#: Wall-clock budget for OHLCV *backfill* inside one selection pass.
#:
#: Backfill is the last unbounded blocking call left on the bootstrap thread.
#: Two earlier passes fixed one venue at a time -- the news harvest inside
#: `_run_download` (2026-09-02) and Binance's HTTP 451 (2026-09-03) -- and the
#: stall simply moved to the next venue. Measured 2026-09-04 with py-spy against
#: production pid 416880, fourteen minutes after start with zero market_stream
#: rows written, the MainThread was parked in
#:
#:     select_pairs -> try_add_candidate -> _ensure_ohlcv -> download_pair
#:       -> download_pair_coinbase -> requests.get(api.exchange.coinbase.com)
#:
#: A direct timing of that call: MOONBASE-USDC (absent from Coinbase) costs
#: 0.9s, but CBETH-USDC costs **77.6s** -- 90 days of 5-minute candles is 13645
#: rows, paginated 300 at a time, 46 serial requests with a 0.35s delay between
#: them. `try_add_candidate` runs that once per candidate lacking candles, and
#: `_run_download` on the same path waits up to DOWNLOAD_SUBPROCESS_TIMEOUT_SEC
#: (300s) more. No market stream exists until `build()` returns, so every one of
#: those seconds is a hole in the feed.
#:
#: So the budget is set at the mechanism rather than per venue: past the
#: deadline `_ensure_ohlcv` performs NO network I/O at all and hands the symbol
#: to a background thread, which fills the candles for the next pass.
_OHLCV_BACKFILL_BUDGET_SEC = float(os.getenv("OHLCV_BACKFILL_BUDGET_SEC", "25"))

#: Symbols whose backfill was deferred past the budget, and the worker draining
#: them. Guarded by `_OHLCV_DEFER_LOCK`; entries are (chain, symbol, data_root).
_OHLCV_DEFERRED: "OrderedDict[Tuple[str, str], Optional[Path]]" = OrderedDict()
_OHLCV_DEFER_LOCK = threading.Lock()
_OHLCV_DEFER_THREAD: Optional[threading.Thread] = None


def _defer_ohlcv_backfill(chain: str, symbol: str, data_root: Optional[Path]) -> None:
    """Queue a missing backfill for a background thread and return immediately.

    The pair is not selected on this pass -- it has no candles, and that gate is
    unchanged. It joins on a later pass once the worker has filled them, which
    is strictly sooner than today, where the pass that would have picked it also
    holds every market stream dark while it downloads.
    """
    global _OHLCV_DEFER_THREAD
    key = (chain.lower(), symbol.upper())
    with _OHLCV_DEFER_LOCK:
        if key in _OHLCV_DEFERRED:
            return
        _OHLCV_DEFERRED[key] = data_root
        alive = _OHLCV_DEFER_THREAD is not None and _OHLCV_DEFER_THREAD.is_alive()
        if not alive:
            _OHLCV_DEFER_THREAD = threading.Thread(
                target=_drain_ohlcv_backfill,
                name="ohlcv-backfill",
                daemon=True,
            )
            _OHLCV_DEFER_THREAD.start()
    log_message(
        "pair-select",
        f"deferred OHLCV backfill for {symbol.upper()} (selection budget spent)",
        details={"chain": chain, "budget_sec": _OHLCV_BACKFILL_BUDGET_SEC},
    )


def _drain_ohlcv_backfill() -> None:
    """Backfill deferred symbols one at a time, off the selection thread."""
    while True:
        with _OHLCV_DEFER_LOCK:
            if not _OHLCV_DEFERRED:
                return
            (chain, symbol), data_root = _OHLCV_DEFERRED.popitem(last=False)
        try:
            # No deadline here: this thread owns nothing latency-critical.
            _ensure_ohlcv(chain, symbol, data_root=data_root, deadline=None)
        except Exception as exc:  # pragma: no cover - best effort
            log_message(
                "pair-select",
                f"background OHLCV backfill failed for {symbol}: {exc}",
                severity="warning",
            )


def _ensure_ohlcv(
    chain: str,
    symbol: str,
    data_root: Optional[Path] = None,
    *,
    deadline: Optional[float] = None,
) -> bool:
    """Does this pair have candles, downloading them if there is time to.

    ``deadline`` is a ``time.monotonic()`` reading, or None for "no budget".
    None is the default so the background worker and every existing caller keep
    the old unbounded behaviour; only the two call sites inside
    ``_select_for_chain`` -- the ones on the thread the feed waits on -- pass one.
    """
    if _ohlcv_exists(symbol, chain, data_root=data_root):
        try:
            _db.set_control_flag(f"ohlcv_ready::{chain.lower()}::{symbol.upper()}", "1")
        except Exception:
            pass
        return True
    # Past the budget nothing below this line may run: `_run_download` waits on a
    # subprocess for up to 300s and the CEX fallback measured 77.6s for a single
    # pair. Hand it to the background worker instead.
    if deadline is not None and time.monotonic() >= deadline:
        _defer_ohlcv_backfill(chain, symbol, data_root)
        return False
    # Avoid runaway downloads; only allow a short lookback window for new pairs
    os.environ.setdefault("HISTORICAL_WINDOW_DAYS", "30")
    os.environ.setdefault("HISTORICAL_TRIM", "1")
    assignment_path = Path("data") / f"{chain}_pair_provider_assignment.json"
    try:
        assignment = _ensure_assignment_template(chain, assignment_path)
    except FileNotFoundError:
        log_message("pair-select", f"pair index missing for {chain}; cannot backfill {symbol}", severity="warning")
        return False
    pairs = assignment.setdefault("pairs", {})
    symbol_u = symbol.upper()
    added = False
    for addr, meta in list(pairs.items()):
        if str(meta.get("symbol", "")).upper() == symbol_u:
            added = True
            break
    if not added:
        index_path = Path("data") / f"pair_index_{chain}.json"
        if index_path.exists():
            try:
                with index_path.open("r", encoding="utf-8") as fh:
                    index = json.load(fh)
                for addr, meta in index.items():
                    if str(meta.get("symbol", "")).upper() != symbol_u:
                        continue
                    pairs[addr] = {
                        "symbol": symbol_u,
                        "index": int(meta.get("index", len(pairs))),
                        "completed": False,
                    }
                    added = True
                    break
            except Exception as exc:
                log_message("pair-select", f"unable to update assignment for {symbol_u}: {exc}", severity="warning")
    if added:
        _update_assignment(assignment_path, assignment)
        # No news harvest on this path: this runs once per candidate that
        # lacks candles, on the thread the market streams share, and the
        # harvest it used to pull is a 142-source serial crawl. See the
        # docstring on _run_download for the measurement.
        _run_download(chain, assignment_path, collect_news=False)
        ready = _ohlcv_exists(symbol, chain, data_root=data_root)
        if ready:
            try:
                _db.set_control_flag(f"ohlcv_ready::{chain.lower()}::{symbol.upper()}", "1")
            except Exception:
                pass
            return ready
    # Fallback: try CEX download (Binance/CoinGecko) when on-chain data unavailable
    try:
        from services.cex_ohlcv_fallback import download_pair, save_ohlcv
        source, rows = download_pair(symbol_u, days_back=90)
        if rows:
            idx = int(next(
                (m.get("index", 0) for m in pairs.values() if str(m.get("symbol", "")).upper() == symbol_u),
                9999,
            ))
            root = data_root or Path(os.getenv("HISTORICAL_DATA_ROOT", "data/historical_ohlcv"))
            path = save_ohlcv(rows, symbol_u, idx, chain=chain, output_root=root)
            if path:
                log_message("pair-select", f"CEX fallback saved {symbol_u}: {len(rows)} candles from {source}")
                try:
                    _db.set_control_flag(f"ohlcv_ready::{chain.lower()}::{symbol_u}", "1")
                except Exception:
                    pass
                return True
    except Exception as exc:
        log_message("pair-select", f"CEX fallback failed for {symbol_u}: {exc}", severity="warning")
    return False


def _vol_spread_score(cand: PairCandidate) -> float:
    """
    Heuristic: higher score when volatility is moderate but not erratic.
    Penalize extreme volatility or missing volume.
    """
    vol = max(0.0, float(cand.volatility))
    volume = max(1e-6, float(cand.avg_volume))
    # Prefer moderate volatility with healthy volume
    return float((vol + 0.01) / (1.0 + vol * vol) * math.log1p(volume))


def _load_pair_metadata() -> Dict[str, str]:
    data = pair_index_entries()
    mapping: Dict[str, str] = {}
    for addr, info in data.items():
        if not isinstance(info, dict):
            continue
        symbol = str(info.get("symbol", "")).upper()
        if addr and symbol:
            mapping[symbol] = addr
    return mapping


def analyse_historical_pairs(
    *,
    data_dir: Path = Path(os.getenv("HISTORICAL_DATA_ROOT", "data/historical_ohlcv")) / PRIMARY_CHAIN,
    min_samples: int = 120,
) -> List[PairCandidate]:
    entries: List[PairCandidate] = []
    for json_file in sorted(data_dir.glob("*.json")):
        try:
            with json_file.open("r", encoding="utf-8") as fh:
                rows = json.load(fh)
        except Exception:
            continue
        if not isinstance(rows, list) or len(rows) < min_samples:
            continue

        closes = np.array([float(row.get("close", 0.0)) for row in rows], dtype=np.float64)
        net_volumes = np.array([float(row.get("net_volume", 0.0)) for row in rows], dtype=np.float64)
        buy_volumes = np.array([float(row.get("buy_volume", 0.0)) for row in rows], dtype=np.float64)
        sell_volumes = np.array([float(row.get("sell_volume", 0.0)) for row in rows], dtype=np.float64)

        if closes.size == 0:
            continue
        avg_volume = float(np.mean(np.abs(net_volumes)))
        liquidity = float(np.mean(np.abs(buy_volumes) + np.abs(sell_volumes)))
        price_mean = float(np.mean(closes))
        if price_mean <= 0:
            continue
        volatility = float(np.std(closes) / price_mean)
        score = avg_volume * (1.0 + volatility) + liquidity * 0.25

        pair_label = json_file.stem.split("_", 1)[-1].upper()
        tokens = [tok.strip() for tok in pair_label.split("-") if tok.strip()]

        # simple heuristics to avoid obvious scams
        if any("RUG" in tok or "SCAM" in tok for tok in tokens):
            continue

        entries.append(
            PairCandidate(
                symbol=pair_label,
                tokens=tokens,
                avg_volume=avg_volume,
                volatility=volatility,
                score=score,
                datapath=json_file,
            )
        )

    entries.sort(key=lambda c: c.score, reverse=True)
    return entries


def _chain_priority(start: str = PRIMARY_CHAIN) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    preferred = [start.lower()] + [ch for ch in _LOW_COST_CHAIN_ORDER if ch != start.lower()]
    for ch in preferred:
        if ch not in CHAINS or ch in seen:
            continue
        ordered.append(ch)
        seen.add(ch)
    for ch in CHAINS:
        if ch not in seen:
            ordered.append(ch)
            seen.add(ch)
    return ordered


def _chain_lock_state() -> Tuple[Optional[str], float]:
    try:
        raw = _db.get_control_flag("pair_chain_lock")
    except Exception:
        raw = None
    if not raw:
        return None, 0.0
    try:
        data = json.loads(raw)
        return str(data.get("chain") or "").lower() or None, float(data.get("started", 0.0))
    except Exception:
        return None, 0.0


def _persist_chain_lock(chain: str) -> None:
    payload = {"chain": chain.lower(), "started": time.time()}
    try:
        _db.set_control_flag("pair_chain_lock", json.dumps(payload))
    except Exception:
        pass


def select_pairs(
    *,
    limit: int = 6,
    min_volume: float = 25.0,
    data_dir: Path = Path("data/historical_ohlcv"),
) -> List[PairCandidate]:
    lock_chain, lock_started = _chain_lock_state()
    now = time.time()
    lock_active = lock_chain is not None and (now - lock_started) < _PAIR_CHAIN_LOCK_SECONDS
    chain_order = [lock_chain] if lock_active and lock_chain else []
    if not chain_order:
        chain_order = [PRIMARY_CHAIN]
    for ch in _chain_priority(chain_order[0]):
        if ch not in chain_order:
            chain_order.append(ch)

    # ONE backfill budget for the whole call, deliberately not per chain and not
    # per candidate. A chain that finds nothing falls through to the next, and
    # `_chain_priority` yields eight of them here (base, arbitrum, optimism,
    # polygon, bsc, ethereum, avalanche, zksync) -- a per-chain budget is an
    # eight-times-larger budget, and a per-candidate one is unbounded again.
    ohlcv_deadline = (
        time.monotonic() + _OHLCV_BACKFILL_BUDGET_SEC
        if _OHLCV_BACKFILL_BUDGET_SEC > 0
        else None
    )

    def _select_for_chain(chain: str) -> List[PairCandidate]:
        chain_dir = data_dir / chain
        candidates = analyse_historical_pairs(data_dir=chain_dir)
        candidate_map = {cand.symbol: cand for cand in candidates}
        try:
            watchlists = load_watchlists(_db)
        except Exception:
            watchlists = {}
        manual_symbols = list(
            dict.fromkeys((watchlists.get("stream") or []) + (watchlists.get("live") or []))
        )
        manual_candidates: List[PairCandidate] = []
        for symbol in manual_symbols:
            symbol_u = symbol.upper()
            if symbol_u not in candidate_map:
                tokens = [part.strip().upper() for part in symbol_u.split("-") if part.strip()]
                candidate_map[symbol_u] = PairCandidate(
                    symbol=symbol_u,
                    tokens=tokens or [symbol_u],
                    avg_volume=0.0,
                    volatility=0.0,
                    score=0.0,
                    datapath=Path("."),
                )
            manual_candidates.append(candidate_map[symbol_u])
        picked: List[PairCandidate] = []
        seen_tokens: set[str] = set()

        def _ensure_candidate(symbol: str) -> None:
            symbol_u = symbol.upper()
            if symbol_u not in candidate_map:
                tokens = [part.strip().upper() for part in symbol_u.split("-") if part.strip()]
                candidate_map[symbol_u] = PairCandidate(
                    symbol=symbol_u,
                    tokens=tokens or [symbol_u],
                    avg_volume=0.0,
                    volatility=0.0,
                    score=0.0,
                    datapath=Path("."),
                )

        def try_add_candidate(symbol: str, *, priority: bool = False) -> None:
            cand = candidate_map.get(symbol.upper())
            if not cand:
                return
            # Checked BEFORE the priority escape below, deliberately. The
            # manual watchlist and the wallet-holdings loop both enter here
            # with priority=True precisely so they skip the volume score, and
            # that loop builds the inverted `f"{stable}-{held}"` form
            # (USDC-AERO) alongside the real one -- a stable BASE, which the
            # accounting layer can never denominate. A priority flag says
            # "we have no volume history for this yet"; it does not say the
            # P&L formula will work.
            if not _can_denominate_pnl_in_usd(cand.symbol):
                return
            token_key = (tuple(sorted(cand.tokens)), chain.lower())
            if token_key in seen_tokens:
                return
            # Priority candidates (manual watchlist, wallet holdings) have no
            # historical volume yet, so the vol-score gate would reject every
            # one of them; safety still comes from the live-price probe,
            # streaming-feed check and OHLCV bootstrap below.
            if not priority and _vol_spread_score(cand) < _PAIR_MIN_VOL_SCORE:
                return
            if not _has_live_price(cand.symbol, chain=chain):
                return
            if not _has_streaming_feed(cand.symbol, chain=chain):
                return
            if not _ensure_ohlcv(chain, cand.symbol, data_root=data_dir, deadline=ohlcv_deadline):
                return
            picked.append(cand)
            seen_tokens.add(token_key)

        for cand in manual_candidates:
            if len(picked) >= limit:
                break
            try_add_candidate(cand.symbol, priority=True)

        # Wallet holdings (any nonzero balance, dust included) signal the
        # holder's interest — they stream ahead of generic market pairs, with
        # a couple of slots kept free so market discovery is never starved.
        held_symbols: set[str] = set()
        try:
            portfolio = PortfolioState()
            portfolio.refresh(force=True)
            held_symbols = {sym for (_, sym) in portfolio.holdings.keys()}
        except Exception:
            held_symbols = set()

        held_slot_cap = max(1, limit - 2) if limit > 2 else limit
        held_added_before = len(picked)
        for held in sorted(held_symbols):
            if len(picked) >= min(limit, held_added_before + held_slot_cap):
                break
            if held in STABLE_TOKENS:
                continue
            for stable in ("USDC", "USDT", "DAI"):
                if len(picked) >= min(limit, held_added_before + held_slot_cap):
                    break
                for pair_symbol in (f"{held}-{stable}", f"{stable}-{held}"):
                    _ensure_candidate(pair_symbol)
                    try_add_candidate(pair_symbol, priority=True)

        for symbol in DEFAULT_LIVE_PAIRS:
            if len(picked) >= limit:
                break
            try_add_candidate(symbol)

        for symbol in _load_top_symbols(_PAIR_SCAN_VOLUME_LIMIT, chain=chain):
            if len(picked) >= limit:
                break
            _ensure_candidate(symbol)
            try_add_candidate(symbol)

        scanned = 0
        max_scan = max(limit * 30, limit + 5)
        for cand in candidates:
            if len(picked) >= limit:
                break
            scanned += 1
            if scanned > max_scan and picked:
                break
            if cand.avg_volume < min_volume:
                continue
            # Repeated rather than shared with try_add_candidate: this loop is
            # the bulk scan over `analyse_historical_pairs()` and never calls
            # that helper. It is the source of the WETH- and CBBTC-quoted
            # pairs, whose symbols come straight from OHLCV filenames
            # (0011_CBETH-WETH.json), so a gate wired only into the helper
            # would leave the largest producer untouched.
            if not _can_denominate_pnl_in_usd(cand.symbol):
                continue
            token_key = (tuple(sorted(cand.tokens)), chain.lower())
            if token_key in seen_tokens:
                continue
            if _vol_spread_score(cand) < _PAIR_MIN_VOL_SCORE:
                continue
            if not _has_live_price(cand.symbol, chain=chain):
                continue
            if not _has_streaming_feed(cand.symbol, chain=chain):
                continue
            if not _ensure_ohlcv(chain, cand.symbol, data_root=data_dir, deadline=ohlcv_deadline):
                continue
            seen_tokens.add(token_key)
            picked.append(cand)
        return picked

    for chain in chain_order:
        if chain is None:
            continue
        best = _select_for_chain(chain)
        if best:
            if chain != lock_chain:
                _persist_chain_lock(chain)
            return best[:limit]
        # if we exhausted the locked chain, allow expansion after lock expiry
        if lock_chain and chain == lock_chain and (now - lock_started) >= _PAIR_CHAIN_LOCK_SECONDS:
            continue
    return []



def _genome_universe_symbols(limit: int) -> List[str]:
    """Quote-paired symbols for the current champion's tradeable universe.

    Returns [] when disabled or unavailable, so a missing GA checkout or an
    unscorable champion simply leaves pair selection exactly as it was.
    """
    if limit <= 0:
        return []
    try:
        from trading.genome.champion import champion_meets_objective, load_champion
        from trading.genome.feed import load_universe_bars
    except Exception:
        return []
    champion = load_champion()
    if champion is None or not champion_meets_objective(champion):
        return []
    try:
        assets = sorted(load_universe_bars())
    except Exception:
        return []
    quote = os.getenv("GENOME_PAIR_QUOTE", "USDC").upper()
    return [f"{asset.upper()}-{quote}" for asset in assets[:limit]]


def _held_position_symbols(db: Optional["TradingDatabase"]) -> List[str]:
    """Symbols this account currently holds an open ghost position in.

    You must be able to CLOSE what you hold. Every exit decision is made in
    ``TradingBot._interpret_predictions``, which is reached only from a market
    sample -- it reads ``symbol`` and ``sample_ts`` off the sample and then
    looks up ``self.positions.get(symbol)``. So the ``MAX_HOLD_SECONDS`` timeout,
    the stop loss and the confidence-drop exit are all consulted ONLY for a
    symbol that just ticked, on a bot that is actually running for it.

    ``build()`` composed the bot pool from ATF signals, focus assets, the genome
    seed and ``select_pairs()`` -- never from the position book. A symbol that
    dropped out of that selection kept its row in the persisted book and lost
    the only thing that could ever close it. Measured 2026-09-02 against the
    live book, 4 of 12 open positions had no ticking feed behind them:

        UNI-USDC      rsi_reversal@1w       held 362.9h   never ticked
        HIGH-USDC     (none)                held 237.1h   never ticked
        ARB-USDC      obv_accumulation@5h   held  11.5h   last tick 10.9h ago
        VIRTUAL-USDC  obv_accumulation@5d   held   0.6h   last tick  0.6h ago

    against ``MAX_HOLD_SECONDS`` of 3600. VIRTUAL-USDC is the live case rather
    than old damage: it was opened 34 minutes earlier and had not ticked once
    since entry.

    That is a graduation blocker, not just untidiness. Promotion is scored on
    CLOSED ghost trades, so a stranded position is a round trip that never
    reaches the ledger -- and ``atf_static``, the only executor that can spend
    real money, sits at 11 of the 20 trades it needs.

    Returns [] on any failure so a bad or missing state blob leaves pair
    selection exactly as it was.
    """
    if db is None:
        return []
    try:
        state = db.load_state()
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(state, dict):
        return []
    ghost = state.get("ghost_trading")
    if not isinstance(ghost, dict):
        return []
    positions = ghost.get("positions")
    if not isinstance(positions, dict):
        return []
    held: List[str] = []
    for symbol, position in positions.items():
        # An entry that is not a position row cannot be closed by giving it a
        # bot, and would only burn a slot.
        if not isinstance(position, dict):
            continue
        text = str(symbol or "").strip().upper()
        if text:
            held.append(text)
    return list(dict.fromkeys(held))


class GhostTradingSupervisor:
    def __init__(
        self,
        *,
        db: Optional[TradingDatabase] = None,
        pipeline: Optional[TrainingPipeline] = None,
        pair_limit: Optional[int] = None,
        stable_checkpoint_ratio: float = 0.15,
    ) -> None:
        self.db = db or get_db()
        self.pipeline = pipeline or TrainingPipeline(db=self.db)
        # Base number of concurrent stream+bot pairs. Was hard-coded to 6,
        # which capped the whole system at 6 streams no matter how big the
        # watchlist. Env-driven now (GHOST_PAIR_LIMIT, default 14) so a
        # capable box streams a real cross-section of the market + wallet;
        # resolve_pair_limit() still throttles it down under RAM pressure.
        if pair_limit is None:
            try:
                pair_limit = int(os.getenv("GHOST_PAIR_LIMIT", "14"))
            except (TypeError, ValueError):
                pair_limit = 14
        self.pair_limit = max(1, pair_limit)
        # Data-only stream coverage: the first pair_limit pairs get full
        # trading bots; additional pairs up to GHOST_STREAM_TOTAL get
        # lightweight data-only MarketDataStreams (WS -> market_stream, no
        # per-tick bot pipeline). This gives broad market coverage for the
        # dashboard + opportunity scanning at a fraction of the CPU of a full
        # bot, so the box can watch 100+ pairs without choking the event loop.
        try:
            self.stream_total = int(os.getenv("GHOST_STREAM_TOTAL", "0"))
        except (TypeError, ValueError):
            self.stream_total = 0
        self.data_streams: List["MarketDataStream"] = []
        self._effective_pair_limit = pair_limit
        self.stable_checkpoint_ratio = stable_checkpoint_ratio
        self.bots: List[TradingBot] = []
        self._tasks: List[asyncio.Task] = []
        self._require_ready_before_stream = os.getenv("REQUIRE_READY_BEFORE_STREAM", "1").lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _readiness_permits_live(readiness: Dict[str, Any]) -> bool:
        """May a bot be BUILT with live trading enabled?

        ``readiness["ready"]`` is the model-accuracy gate, which is degenerate
        on this deployment: precision 0.0 AND recall 0.0 across 639 samples,
        and 1.0 on a 71-sample run hours earlier. It is also starved by
        construction, since it is fed by the TradingBot path while atf_static
        runs its own ghost cycle.

        Forcing ``live_trading_enabled = False`` on that number meant every bot
        was built unable to trade live no matter what any strategy earned --
        _refresh_auto_execute returns early without the flag, so graduation
        could never reach execution. Observed 2026-08-27: all six live gates
        PASS with block_reason empty, and still zero live rows.

        A strategy that passed _ghost_validation on its own trade record is
        real evidence. The per-strategy gate in bot.py
        (``_strategy_live_approved``) still decides which directives may
        actually spend money; this only stops the aggregate metric from
        disabling the machinery wholesale. LIVE_REQUIRE_MODEL_READY=1 restores
        the strict coupling.
        """
        if bool(readiness.get("ready")):
            return True
        if (os.getenv("LIVE_REQUIRE_MODEL_READY", "0") or "0").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            return False
        if not bool(readiness.get("ghost_ready")):
            return False
        # The cold-start allowance exists to let collection BEGIN; it is not
        # evidence of anything. Everything else _ghost_validation reports next
        # to ready=True was earned -- including the empty reason, which is what
        # the STRICT path returns. See GHOST_EARNED_READY_REASONS.
        return ghost_reason_is_earned(readiness.get("ghost_reason"))

    def build(self) -> None:
        if self.bots:
            return
        focus_assets, _ = self.pipeline.ghost_focus_assets()
        atf_priority: List[str] = []
        try:
            from services.atf_static_strategy import latest_signals
            atf_priority = [
                str(sig.get("symbol") or "").upper()
                for sig in latest_signals(float(os.getenv("ATF_STATIC_SIGNAL_MAX_AGE_SEC", "1800")))
                if isinstance(sig, dict) and str(sig.get("symbol") or "").strip()
            ]
        except Exception:
            atf_priority = []
        readiness = self.pipeline.live_readiness_report()
        transition_plan = self.pipeline.ghost_live_transition_plan()
        horizon_bias = {}
        horizon_deficit = {}
        try:
            horizon_bias = self.pipeline.horizon_bias()
        except Exception:
            horizon_bias = {}
        dataset_meta = getattr(self.pipeline, "_last_dataset_meta", {})
        if isinstance(dataset_meta, dict):
            horizon_deficit = dataset_meta.get("horizon_deficit") or {}
            if not isinstance(horizon_deficit, dict):
                horizon_deficit = {}
        pair_limit, limit_meta = resolve_pair_limit(
            self.pair_limit,
            focus_assets=focus_assets,
            horizon_bias=horizon_bias,
            horizon_deficit=horizon_deficit,
            system_profile=getattr(self.pipeline, "system_profile", None),
        )
        # A held position must be given a bot before any new candidate, and the
        # limit must stretch to cover them all. Ordering alone is not enough:
        # measured 2026-09-02 the book held 12 open positions against a
        # resolved pair_limit of 8, so `all_ordered[:pair_limit]` would have
        # dropped four of them right back into the state they are being
        # rescued from. Bounded by max_limit so a pathological book cannot
        # spawn unlimited bots; anything past it keeps its place at the front
        # of the queue and is picked up by the next reconcile.
        held_symbols = _held_position_symbols(getattr(self, "db", None))
        if held_symbols:
            ceiling = int(limit_meta.get("max_limit") or pair_limit)
            boosted = min(max(pair_limit, len(held_symbols)), max(ceiling, pair_limit))
            if boosted != pair_limit:
                limit_meta["held_position_boost"] = {
                    "held": len(held_symbols), "from": pair_limit, "to": boosted,
                }
                limit_meta["adjusted"] = True
                pair_limit = boosted
        self._effective_pair_limit = pair_limit
        if limit_meta.get("adjusted"):
            limit_meta["focus_assets"] = focus_assets[:8]
            if horizon_deficit:
                limit_meta["horizon_deficit"] = horizon_deficit
            log_message("ghost-supervisor", "adjusted pair limit", severity="info", details=limit_meta)
        if readiness and readiness.get("reason") == "no_confusion_data":
            if self.pipeline.prime_confusion_windows():
                readiness = self.pipeline.live_readiness_report()
                transition_plan = self.pipeline.ghost_live_transition_plan()
        if readiness:
            log_message(
                "ghost-supervisor",
                "live readiness snapshot",
                severity="info" if readiness.get("ready") else "warning",
                details=readiness,
            )
        # Pull enough candidates for the full-bot tier AND the data-only tier.
        select_limit = max(pair_limit, self.stream_total)
        pairs = select_pairs(limit=select_limit)
        prioritized: List[PairCandidate] = []
        # The champion genome only scores the assets it was fitted on. When the
        # focus rotation holds none of them the strategy abstains on every bot,
        # which is why the ghost ledger stayed empty even with the feed
        # publishing 33 scorable signals: measured 2026-08-18 the rotation was
        # all Base memecoins (BASECAT, BSTONK, MEOW...) against a genome
        # universe of established DeFi names, for an overlap of exactly zero.
        #
        # 21 of the 33 genome assets ARE streamed, so seeding a few of them
        # gives the ladder something to record without displacing the neural
        # pipeline's own picks -- they are appended after atf_priority and
        # focus_assets, and the existing dedup keeps them from crowding.
        genome_seed = _genome_universe_symbols(int(os.getenv("GENOME_PAIR_SEED", "0")))
        # Held positions lead. Closing an open position is worth more than
        # opening a new one: it frees the slot, and it is the only way the
        # round trip ever reaches the ledger that gates graduation.
        for symbol in list(dict.fromkeys(
                held_symbols + atf_priority + list(focus_assets or []) + genome_seed)):
            tokens = [part.strip().upper() for part in symbol.split("-") if part.strip()]
            if not tokens:
                tokens = [symbol.upper()]
            prioritized.append(
                PairCandidate(
                    symbol=symbol.upper(),
                    tokens=tokens,
                    avg_volume=0.0,
                    volatility=0.0,
                    score=0.0,
                    datapath=Path("."),
                )
            )
        # Dedup the combined candidate list, preserving priority order.
        #
        # `prioritized` is held positions + ATF signals + focus assets + the
        # genome seed, and NONE of those passed through select_pairs -- so the
        # accounting gate wired in there has never seen them. Applied here or
        # they arrive with a bot attached.
        held_upper = {symbol.upper() for symbol in held_symbols}
        all_ordered: List[PairCandidate] = []
        seen: set[str] = set()
        refused_accounting: List[str] = []
        for candidate in prioritized + pairs:
            symbol_u = candidate.symbol.upper()
            if symbol_u in seen:
                continue
            # A HELD POSITION KEEPS ITS FEED whatever its quote token is. It is
            # the only thing that can close, and the exit path is sample-driven
            # -- taking the stream away is how positions became immortal.
            if symbol_u not in held_upper and not _can_denominate_pnl_in_usd(symbol_u):
                refused_accounting.append(symbol_u)
                seen.add(symbol_u)
                continue
            all_ordered.append(candidate)
            seen.add(symbol_u)
        if refused_accounting:
            log_message(
                "ghost-supervisor",
                f"refused {len(refused_accounting)} pair(s) that cannot denominate P&L in USD",
                severity="info",
                details={"symbols": refused_accounting[:16]},
            )
        if not all_ordered:
            all_ordered = list(pairs)
        # Slots go to symbols something may actually enter. `prioritized`
        # (held + ATF signals + focus + genome seed) leads the list and is
        # protected by `held_upper`; the rest is `select_pairs`' volume ranking,
        # which is where the condemned symbols come from. See
        # `_no_strategy_may_enter` for the 379-of-596 measurement.
        all_ordered, _condemned = _sink_condemned(all_ordered, protected=held_upper)
        if _condemned:
            log_message(
                "ghost-supervisor",
                f"ranked {len(_condemned)} pair(s) below the eligible ones: no "
                "strategy may open a position in them",
                severity="info",
                details={"symbols": _condemned[:16]},
            )
        if atf_priority:
            log_message(
                "ghost-supervisor",
                "prioritized ATF static pairs for build",
                details={"pairs": atf_priority[:8]},
            )
        # First pair_limit -> full trading bots; the remainder up to
        # stream_total -> data-only streams (coverage without bot CPU cost).
        ordered = all_ordered[:pair_limit]
        data_pairs = (
            all_ordered[pair_limit:self.stream_total]
            if self.stream_total > pair_limit else []
        )
        for pair in ordered:
            stream = MarketDataStream(symbol=pair.symbol, chain=PRIMARY_CHAIN)
            bot = TradingBot(db=self.db, stream=stream, pipeline=self.pipeline)
            bot.configure_route(pair.symbol, pair.tokens)
            bot.stable_checkpoint_ratio = self.stable_checkpoint_ratio
            bot.max_trade_share = 0.12
            if readiness and not self._readiness_permits_live(readiness):
                bot.live_trading_enabled = False
            if hasattr(bot, "apply_transition_plan"):
                bot.apply_transition_plan(transition_plan)
            self.bots.append(bot)

        # Data-only stream tier: lightweight WS->market_stream feeds for broad
        # coverage (dashboard + opportunity scanning) without a per-tick bot.
        for pair in data_pairs:
            try:
                self.data_streams.append(
                    MarketDataStream(symbol=pair.symbol, chain=PRIMARY_CHAIN))
            except Exception:
                continue
        if self.data_streams:
            log_message(
                "ghost-supervisor",
                f"data-only streams: {len(self.data_streams)} (+ {len(self.bots)} trading bots)",
                severity="info",
            )

        # Portfolio-level cross-token rotation: sell high on one pair, jump
        # into the freshest buy-low candidate on any other streamed pair.
        try:
            from trading.rotation import PortfolioRotator
            rotator = PortfolioRotator()
            for bot in self.bots:
                rotator.register_bot(bot)
            self.rotator = rotator
        except Exception as exc:
            log_message("ghost-supervisor", f"rotator init failed: {exc}", severity="warning")
            self.rotator = None

    async def start(self) -> None:
        if self._require_ready_before_stream:
            await self._await_readiness_gate()
        if not self.bots:
            self.build()
        if not self.bots:
            print("[ghost-supervisor] no eligible pairs found; using default stream")
            bot = TradingBot(db=self.db, pipeline=self.pipeline)
            bot.configure_route(bot.stream.symbol, bot.stream.symbol.split("-"))
            self.bots.append(bot)
        # Each bot runs under a supervisor wrapper so ONE crashing stream
        # can't take down the others (plain gather() propagates the first
        # exception and abandons the batch — that left the whole system with
        # a single surviving stream). Failed bots auto-restart with backoff.
        self._tasks = [asyncio.create_task(self._run_bot_forever(bot)) for bot in self.bots]
        # Data-only streams run their own resilient loop (WS -> market_stream).
        self._tasks.extend(
            asyncio.create_task(self._run_data_stream_forever(s)) for s in self.data_streams
        )
        self._tasks.append(asyncio.create_task(self._drain_trades()))
        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log_message("ghost-supervisor", f"unexpected supervisor error: {exc}", severity="error")

    async def reconcile_pairs(self) -> Dict[str, Any]:
        """Add newly prioritized watchlist pairs without restarting production."""
        existing_bots = {str(getattr(bot, "primary_symbol", "") or "").upper() for bot in self.bots}
        # Belt and braces: a bot's routes name the symbols it actually trades,
        # so read those too rather than trusting one attribute to be set.
        #
        # When `primary_symbol` was left at the module default this set held a
        # single element for the whole pool, reconcile could not see which
        # symbols were already covered, and it kept adding duplicate bots for
        # the same symbol -- 13 of them on BASECAT-USDC on 2026-08-31. Each
        # duplicate has its own `self.positions`, so nothing downstream could
        # notice, and one tick became 13 correlated positions in one symbol.
        # configure_route now sets the identity; this makes the dedupe hold
        # even if some future construction path forgets to.
        for bot in self.bots:
            for routed in (getattr(bot, "bus_routes", None) or {}):
                existing_bots.add(str(routed or "").upper())
        existing_bots.discard("")
        existing = set(existing_bots)
        existing.update(str(getattr(stream, "symbol", "") or "").upper() for stream in self.data_streams)
        atf_priority: List[str] = []
        try:
            from services.atf_static_strategy import latest_signals
            atf_priority = [
                str(sig.get("symbol") or "").upper()
                for sig in latest_signals(float(os.getenv("ATF_STATIC_SIGNAL_MAX_AGE_SEC", "1800")))
                if isinstance(sig, dict) and str(sig.get("symbol") or "").strip()
            ]
        except Exception:
            atf_priority = []
        atf_priority_set = set(atf_priority)
        pair_limit, _meta = resolve_pair_limit(
            self.pair_limit,
            focus_assets=[],
            system_profile=getattr(self.pipeline, "system_profile", None),
        )
        # Same rule as build(): a symbol we hold must have a bot that can close
        # it. Applied here as well as at startup because a position can be
        # stranded mid-session -- VIRTUAL-USDC was opened at 20:47 on
        # 2026-09-02 and had not ticked once in the 34 minutes since, inside a
        # process that had been up the whole time. A build()-only fix would
        # have left it stranded until the next restart.
        #
        # Read the book ONCE and keep both views. `held_symbols` is the
        # ADD list, so it drops anything already covered; `held_all` is the
        # PROTECT list and must not, because a symbol is only "covered" by
        # virtue of the very bot the eviction below is about to stop.
        held_book = _held_position_symbols(getattr(self, "db", None))
        held_all = set(held_book)
        # COVERAGE MEANS A BOT, NEVER A DATA-ONLY STREAM.
        #
        # `existing` deliberately includes `self.data_streams`, which is right
        # for pair SELECTION -- there is no point streaming a symbol twice. It
        # is wrong for the held book, because a data-only stream runs
        # `_run_data_stream_forever` (WS -> market_stream) and never calls
        # `_handle_sample`. Every exit rule there is -- stop, target, timed
        # exit, confidence drop, MAX_HOLD_FORCE_SECONDS -- hangs off
        # `_handle_sample`, so a data stream produces prices for a symbol that
        # still has nothing able to SELL it.
        #
        # Testing the held book against `existing` therefore marked exactly the
        # stranded positions as covered: the symbol ticks, so it looks healthy
        # from `market_stream`, while `_SYMBOL_LAST_TICK_TS` never sees it and
        # no bot holds it. Measured 2026-09-05, both live positions in the book
        # were in precisely that state --
        #
        #   CBBTC-USDC  held 19.8h  50 ticks/h in market_stream, no bot
        #   CBXRP-USDC  held 19.1h  49 ticks/h in market_stream, no bot
        #
        # -- against 100 `entry-refused-duplicate` rows on CBBTC-USDC and zero
        # live trades on the day, because atf_static is the only live-approved
        # strategy and both of its slots were held by positions it could not
        # reach. `_held_position_symbols` was written for this rule and reads
        # the book correctly; the answer was being thrown away one line later.
        held_symbols = [
            symbol for symbol in held_book if symbol not in existing_bots
        ]
        # The symbols that must end this reconcile owning a BOT, whatever else
        # is already streaming them.
        held_needs_bot = set(held_symbols)
        if held_symbols:
            ceiling = int(_meta.get("max_limit") or pair_limit)
            pair_limit = min(
                max(pair_limit, len(self.bots) + len(held_symbols)),
                max(ceiling, pair_limit),
            )
        full_slots = max(0, int(pair_limit) - len(self.bots))
        data_slots = max(0, int(self.stream_total) - len(self.bots) - len(self.data_streams))
        allow_replace = os.getenv("ATF_STATIC_REPLACE_BOTS", "1").lower() in {"1", "true", "yes", "on"}
        max_replacements = 0
        if allow_replace:
            try:
                max_replacements = max(0, min(int(os.getenv("ATF_STATIC_MAX_REPLACEMENTS", "2")), int(pair_limit)))
            except Exception:
                max_replacements = 2
        if (
            full_slots <= 0
            and data_slots <= 0
            and not held_symbols
            and not (allow_replace and atf_priority)
        ):
            return {"added_bots": [], "added_streams": [], "reason": "no_slots"}

        candidates = await asyncio.get_running_loop().run_in_executor(
            _SELECTION_EXECUTOR,
            functools.partial(
                select_pairs,
                limit=max(
                    int(pair_limit),
                    int(self.stream_total),
                    len(existing) + full_slots + data_slots + len(atf_priority),
                ),
            ),
        )
        # Held positions lead the queue, ahead of the ATF signals, for the same
        # reason as in build(): closing what we hold frees a slot and is the
        # only way the round trip reaches the graduation ledger.
        if held_symbols or atf_priority:
            by_symbol = {pair.symbol.upper(): pair for pair in candidates}
            promoted: List[PairCandidate] = []
            for sym in list(dict.fromkeys(held_symbols + atf_priority)):
                if sym in by_symbol:
                    promoted.append(by_symbol[sym])
                    continue
                tokens = [part.strip().upper() for part in sym.split("-") if part.strip()]
                promoted.append(
                    PairCandidate(
                        symbol=sym,
                        tokens=tokens or [sym],
                        avg_volume=0.0,
                        volatility=0.0,
                        score=1.0,
                        datapath=Path("."),
                    )
                )
            # Held symbols are promoted too, so they must be filtered out of
            # the remainder as well -- otherwise a held pair that select_pairs
            # also returned appears twice and burns two slots on one symbol.
            promoted_set = atf_priority_set | set(held_symbols)
            remainder = [pair for pair in candidates if pair.symbol.upper() not in promoted_set]
            candidates = promoted + remainder
        # Same ranking as build(), applied here as well because a reconcile is
        # what fills a freed slot mid-session and the pool that produced the
        # 379-of-596 skew was assembled by THIS path, not by build(). Held
        # symbols and ATF priorities are protected: they lead the list already,
        # and `held_all` is the set that must never lose its bot.
        candidates, _condemned = _sink_condemned(
            candidates, protected=held_all | atf_priority_set
        )
        if _condemned:
            log_message(
                "ghost-supervisor",
                f"ranked {len(_condemned)} pair(s) below the eligible ones: no "
                "strategy may open a position in them",
                severity="info",
                details={"symbols": _condemned[:16]},
            )
        readiness = self.pipeline.live_readiness_report()
        transition_plan = self.pipeline.ghost_live_transition_plan()
        added_bots: List[str] = []
        added_streams: List[str] = []
        replaced_bots: List[Dict[str, str]] = []

        async def _free_bot_slot_for(symbol: str) -> bool:
            nonlocal full_slots, max_replacements
            if full_slots > 0:
                return True
            if not allow_replace or max_replacements <= 0 or symbol not in atf_priority_set:
                return False
            # A BOT HOLDING A POSITION IS NOT SPARE CAPACITY.
            #
            # The victim used to be chosen on one test -- "its symbol is not an
            # ATF priority" -- which is blind to the only thing that makes a bot
            # irreplaceable: it is the single place an open position in its
            # symbol can ever be CLOSED. `_held_position_symbols` exists for
            # exactly that rule and is consulted seventy lines above when ADDING
            # bots; ignoring it when REMOVING them reintroduces, every reconcile,
            # the stranding it was written to end (see its docstring: four of
            # twelve open positions with no ticking feed, one held 362.9h).
            #
            # This is not theoretical capacity pressure. Measured 2026-09-04
            # 21:30-22:46 from the ghost-supervisor log, twelve bots were
            # evicted in 75 minutes -- CBETH-USDC among them, a symbol this
            # wallet has repeatedly been left holding unbooked tokens in -- at a
            # resolved pair_limit of 18 where `full_slots` is 0 and replacement
            # is the ONLY way a new candidate gets in. Nothing in that path
            # asked the book.
            #
            # Skip-and-keep-scanning rather than skip-the-replacement: a held
            # bot is passed over and an unheld one further down the pool is
            # taken instead. When every candidate is held we return False and
            # the ATF signal simply waits. That is the right trade -- a skipped
            # entry costs an opportunity, a position nothing can sell costs
            # capital.
            # SPEND THE CONDEMNED SLOT FIRST.
            #
            # The scan below picks the last replaceable bot, which is blind to
            # whether its symbol can ever produce an entry. With a full pool
            # that means an ATF signal evicts an ELIGIBLE symbol while COMP-USDC
            # -- banned pooled at -4.333% over 16 round trips -- keeps its slot
            # and its 136 decision cycles per 6h. Same predicate and same
            # carve-outs as `_sink_condemned`; a bot whose symbol nothing may
            # enter is the cheapest thing in the pool to stop.
            replace_idx = None
            for eligible_pass in (False, True):
                for idx in range(len(self.bots) - 1, -1, -1):
                    old_symbol = str(getattr(self.bots[idx], "primary_symbol", "") or "").upper()
                    if old_symbol in atf_priority_set:
                        continue
                    if old_symbol in held_all:
                        continue
                    if not eligible_pass and not _no_strategy_may_enter(old_symbol):
                        continue
                    replace_idx = idx
                    break
                if replace_idx is not None:
                    break
            if replace_idx is None:
                return False
            old_bot = self.bots.pop(replace_idx)
            old_symbol = str(getattr(old_bot, "primary_symbol", "") or "").upper()
            try:
                await old_bot.stop()
            except Exception:
                pass
            existing.discard(old_symbol)
            existing_bots.discard(old_symbol)
            full_slots += 1
            max_replacements -= 1
            replaced_bots.append({"old": old_symbol, "new": symbol})
            return True

        for pair in candidates:
            symbol = pair.symbol.upper()
            # A held symbol is NOT skipped for already being in `existing`: the
            # thing covering it may be the data-only stream that stranded it.
            # It still needs a bot, and this is the only loop that makes one.
            if symbol in existing and symbol not in held_needs_bot:
                continue
            # Same accounting gate as build(), and needed separately: the
            # `promoted` block above injects ATF-signal symbols that were never
            # in `candidates`, so they reach this loop without having passed
            # select_pairs. `held_all` is the book's open positions -- exempt,
            # because a position with no feed can never be closed.
            if symbol not in held_all and not _can_denominate_pnl_in_usd(symbol):
                continue
            if await _free_bot_slot_for(symbol):
                existing.add(symbol)
                existing_bots.add(symbol)
                held_needs_bot.discard(symbol)
                # Retire any data-only stream for this symbol first. Leaving it
                # would run two websockets for one pair and keep a data slot
                # spent on coverage that cannot close anything.
                for idx in range(len(self.data_streams) - 1, -1, -1):
                    if str(getattr(self.data_streams[idx], "symbol", "") or "").upper() != symbol:
                        continue
                    stale_stream = self.data_streams.pop(idx)
                    try:
                        if hasattr(stale_stream, "stop"):
                            await stale_stream.stop()
                    except Exception:
                        pass
                stream = MarketDataStream(symbol=symbol, chain=PRIMARY_CHAIN)
                bot = TradingBot(db=self.db, stream=stream, pipeline=self.pipeline)
                bot.configure_route(symbol, pair.tokens)
                bot.stable_checkpoint_ratio = self.stable_checkpoint_ratio
                bot.max_trade_share = 0.12
                # Same rule as build(): bots added by reconciliation must not
                # be disabled by the degenerate aggregate metric either.
                if readiness and not self._readiness_permits_live(readiness):
                    bot.live_trading_enabled = False
                if hasattr(bot, "apply_transition_plan"):
                    bot.apply_transition_plan(transition_plan)
                self.bots.append(bot)
                task = asyncio.create_task(self._run_bot_forever(bot))
                self._tasks.append(task)
                try:
                    if getattr(self, "rotator", None):
                        self.rotator.register_bot(bot)
                except Exception:
                    pass
                added_bots.append(symbol)
                full_slots -= 1
                continue
            if symbol in held_needs_bot:
                # No bot slot was free. Do NOT fall through to a data-only
                # stream: that is the exact coverage that made this position
                # unclosable, and adding it would re-mark the symbol as handled
                # on the next reconcile. Leave it uncovered so it stays at the
                # front of the queue until a real bot slot opens.
                continue
            if data_slots > 0:
                existing.add(symbol)
                stream = MarketDataStream(symbol=symbol, chain=PRIMARY_CHAIN)
                self.data_streams.append(stream)
                task = asyncio.create_task(self._run_data_stream_forever(stream))
                self._tasks.append(task)
                added_streams.append(symbol)
                data_slots -= 1
        if added_bots or added_streams:
            log_message(
                "ghost-supervisor",
                "reconciled new watchlist pairs",
                details={"added_bots": added_bots, "added_streams": added_streams, "replaced_bots": replaced_bots},
            )
        return {"added_bots": added_bots, "added_streams": added_streams, "replaced_bots": replaced_bots}

    async def _run_data_stream_forever(self, stream: "MarketDataStream") -> None:
        """Keep a data-only stream alive (WS -> market_stream), restart on crash."""
        sym = getattr(stream, "symbol", "?")
        backoff = 5.0
        while True:
            try:
                await stream.start()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_message(
                    "ghost-supervisor",
                    f"data stream {sym} crashed; restarting in {backoff:.0f}s",
                    severity="debug",
                    details={"error": str(exc)[:160]},
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 1.5, 120.0)

    async def _run_bot_forever(self, bot: "TradingBot") -> None:
        """Keep one bot's stream alive; restart it on crash with backoff."""
        sym = getattr(bot, "primary_symbol", "?")
        backoff = 5.0
        while True:
            try:
                await bot.start()
                # Clean return = intentional stop; don't respin.
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_message(
                    "ghost-supervisor",
                    f"bot {sym} crashed; restarting in {backoff:.0f}s",
                    severity="warning",
                    details={"error": str(exc)[:200]},
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 1.5, 120.0)

    async def stop(self) -> None:
        await asyncio.gather(
            *(bot.stop() for bot in self.bots),
            *(s.stop() for s in self.data_streams if hasattr(s, "stop")),
            return_exceptions=True,
        )
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _drain_trades(self) -> None:
        while True:
            await asyncio.sleep(5.0)
            for bot in self.bots:
                while True:
                    trade = bot.dequeue()
                    if not trade:
                        break
                    self._handle_trade(trade)

    def _handle_trade(self, trade: Dict[str, Any]) -> None:
        action = trade.get("action")
        symbol = trade.get("symbol")
        margin = float(trade.get("profit", trade.get("net_margin", 0.0)))
        checkpoint = float(trade.get("checkpoint", 0.0))
        try:
            for bot in self.bots:
                if hasattr(bot, "profit_equilibrium"):
                    bot.profit_equilibrium.record(margin, trade.get("ts") or time.time())
                if hasattr(bot, "swarm_selector"):
                    bot.swarm_selector.update("micro", margin, trade.get("ts") or time.time())
        except Exception:
            pass
        print(
            "[ghost-supervisor] %s %s margin=%.6f checkpoint=%.6f bank=%.6f"
            % (
                action,
                symbol,
                margin,
                checkpoint,
                float(trade.get("bank_balance", 0.0)),
            )
        )

    async def _await_readiness_gate(self) -> None:
        """
        Prevent the market stream from starting until we have at least one
        candidate that meets the ghost-readiness gate. This keeps the system
        from chewing resources on live feeds before a viable model exists.
        """
        poll = max(20.0, float(os.getenv("READINESS_POLL_INTERVAL", "45")))
        bootstrap_attempts = max(0, int(os.getenv("READINESS_BOOTSTRAP_ATTEMPTS", "3")))
        while True:
            readiness = self.pipeline.live_readiness_report() or {}
            if readiness.get("ready"):
                log_message("ghost-supervisor", "readiness gate satisfied; starting streams", details=readiness)
                return
            if readiness.get("mini_ready"):
                log_message("ghost-supervisor", "mini-readiness satisfied; starting streams in mini mode", details=readiness)
                return
            log_message(
                "ghost-supervisor",
                "waiting for model readiness before streaming",
                severity="warning",
                details={"ready": readiness.get("ready"), "reason": readiness.get("reason"), "samples": readiness.get("samples")},
            )
            if bootstrap_attempts > 0:
                bootstrap_attempts -= 1
                await asyncio.to_thread(self._bootstrap_candidate_training)
            await asyncio.sleep(poll)

    def _bootstrap_candidate_training(self) -> None:
        """
        Run a lightweight candidate training pass to shorten time-to-first-model
        on constrained machines.
        """
        focus_assets, _ = self.pipeline.ghost_focus_assets()
        try:
            self.pipeline.warm_dataset_cache(focus_assets=focus_assets or None, oversample=True)
        except Exception:
            pass
        prev_light = os.getenv("TRAIN_LIGHTWEIGHT")
        os.environ["TRAIN_LIGHTWEIGHT"] = "1"
        try:
            self.pipeline.train_candidate()
        finally:
            if prev_light is None:
                os.environ.pop("TRAIN_LIGHTWEIGHT", None)
            else:
                os.environ["TRAIN_LIGHTWEIGHT"] = prev_light
