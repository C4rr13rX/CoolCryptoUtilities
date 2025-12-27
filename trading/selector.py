from __future__ import annotations

import asyncio
import json
from pathlib import Path
import os
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
from trading.data_stream import MarketDataStream, _split_symbol, TOKEN_NORMALIZATION
from trading.pipeline import TrainingPipeline
from trading.portfolio import PortfolioState
from trading.constants import PRIMARY_CHAIN, PRIMARY_SYMBOL, top_pairs, pair_index_entries
from trading.constants import PRIMARY_CHAIN, PRIMARY_SYMBOL
from services.logging_utils import log_message
from trading.ghost_limits import resolve_pair_limit
from services.watchlists import load_watchlists
from services.background_workers import _ensure_assignment_template, _update_assignment, _run_download

STABLE_TOKENS = {"USDC", "USDT", "DAI", "BUSD", "TUSD", "USDP", "USDD", "USDS", "GUSD"}

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
_SUPPRESSION_TTL = float(os.getenv("PAIR_SUPPRESSION_TTL", str(6 * 3600)))
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


def _has_live_price(symbol: str, chain: str = PRIMARY_CHAIN) -> bool:
    _bootstrap_always_live_symbols()
    key = _pair_key(symbol, chain)
    symbol_u = symbol.upper()
    if symbol_u in _ALWAYS_LIVE_SYMBOLS:
        _LIVE_PAIR_CACHE[key] = True
        return True
    if _db.is_pair_suppressed(key):
        record = _db.get_pair_suppression(key) or {}
        remaining = float(record.get("release_ts", 0.0)) - time.time()
        wait_minutes = max(0.0, remaining / 60.0)
        reason = record.get("reason") or "suppressed"
        print(
            f"[pair-select] suppressed {symbol_u}: {reason}; retry in ~{wait_minutes:.1f} min."
        )
        _LIVE_PAIR_CACHE[key] = False
        return False

    cached = _LIVE_PAIR_CACHE.get(key)
    if cached is not None:
        return cached
    probe_result = _probe_dexscreener(symbol_u)
    if probe_result:
        _LIVE_PAIR_CACHE[key] = True
        _db.clear_pair_suppression(key)
        return True
    if probe_result is None:
        print(f"[pair-select] probe deferred for {symbol_u}: dexscreener unreachable.")
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
        print(f"[pair-select] skipping {symbol_u}: no live market data sources responded.")
    return result


def _has_streaming_feed(symbol: str, chain: str) -> bool:
    """
    Lightweight readiness check: ensure we have at least one endpoint or
    offline fallback to stream live price updates for the pair.
    """
    try:
        stream = MarketDataStream(symbol=symbol, chain=chain)
        if stream.url:
            return True
        if getattr(stream, "endpoints", None):
            return True
        if getattr(stream, "_offline_store", None) and stream._offline_store is not None:
            return True
    except Exception:
        return False
    return False


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


def _ensure_ohlcv(chain: str, symbol: str, data_root: Optional[Path] = None) -> bool:
    if _ohlcv_exists(symbol, chain, data_root=data_root):
        try:
            _db.set_control_flag(f"ohlcv_ready::{chain.lower()}::{symbol.upper()}", "1")
        except Exception:
            pass
        return True
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
        _run_download(chain, assignment_path)
        ready = _ohlcv_exists(symbol, chain, data_root=data_root)
        if ready:
            try:
                _db.set_control_flag(f"ohlcv_ready::{chain.lower()}::{symbol.upper()}", "1")
            except Exception:
                pass
        return ready
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

        def try_add_candidate(symbol: str) -> None:
            cand = candidate_map.get(symbol.upper())
            if not cand:
                return
            token_key = (tuple(sorted(cand.tokens)), chain.lower())
            if token_key in seen_tokens:
                return
            if _vol_spread_score(cand) < _PAIR_MIN_VOL_SCORE:
                return
            if not _has_live_price(cand.symbol, chain=chain):
                return
            if not _has_streaming_feed(cand.symbol, chain=chain):
                return
            if not _ensure_ohlcv(chain, cand.symbol, data_root=data_dir):
                return
            picked.append(cand)
            seen_tokens.add(token_key)

        for cand in manual_candidates:
            if len(picked) >= limit:
                break
            try_add_candidate(cand.symbol)

        for symbol in DEFAULT_LIVE_PAIRS:
            if len(picked) >= limit:
                break
            try_add_candidate(symbol)

        for symbol in _load_top_symbols(_PAIR_SCAN_VOLUME_LIMIT, chain=chain):
            if len(picked) >= limit:
                break
            if symbol.upper() not in candidate_map:
                tokens = [part.strip().upper() for part in symbol.split("-") if part.strip()]
                candidate_map[symbol.upper()] = PairCandidate(
                    symbol=symbol.upper(),
                    tokens=tokens or [symbol.upper()],
                    avg_volume=0.0,
                    volatility=0.0,
                    score=0.0,
                    datapath=Path("."),
                )
            try_add_candidate(symbol)

        held_symbols: set[str] = set()
        try:
            portfolio = PortfolioState()
            portfolio.refresh(force=True)
            held_symbols = {sym for (_, sym) in portfolio.holdings.keys()}
        except Exception:
            held_symbols = set()

        for held in held_symbols:
            if len(picked) >= limit:
                break
            if held in STABLE_TOKENS:
                continue
            for stable in ("USDC", "USDT", "DAI"):
                if len(picked) >= limit:
                    break
                try_add_candidate(f"{held}-{stable}")
                try_add_candidate(f"{stable}-{held}")

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
            token_key = (tuple(sorted(cand.tokens)), chain.lower())
            if token_key in seen_tokens:
                continue
            if _vol_spread_score(cand) < _PAIR_MIN_VOL_SCORE:
                continue
            if not _has_live_price(cand.symbol, chain=chain):
                continue
            if not _has_streaming_feed(cand.symbol, chain=chain):
                continue
            if not _ensure_ohlcv(chain, cand.symbol, data_root=data_dir):
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


class GhostTradingSupervisor:
    def __init__(
        self,
        *,
        db: Optional[TradingDatabase] = None,
        pipeline: Optional[TrainingPipeline] = None,
        pair_limit: int = 6,
        stable_checkpoint_ratio: float = 0.15,
    ) -> None:
        self.db = db or get_db()
        self.pipeline = pipeline or TrainingPipeline(db=self.db)
        self.pair_limit = pair_limit
        self._effective_pair_limit = pair_limit
        self.stable_checkpoint_ratio = stable_checkpoint_ratio
        self.bots: List[TradingBot] = []
        self._tasks: List[asyncio.Task] = []
        self._require_ready_before_stream = os.getenv("REQUIRE_READY_BEFORE_STREAM", "1").lower() in {"1", "true", "yes", "on"}

    def build(self) -> None:
        if self.bots:
            return
        focus_assets, _ = self.pipeline.ghost_focus_assets()
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
        pairs = select_pairs(limit=pair_limit)
        prioritized: List[PairCandidate] = []
        for symbol in focus_assets:
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
        ordered: List[PairCandidate] = []
        seen: set[str] = set()
        for candidate in prioritized + pairs:
            symbol_u = candidate.symbol.upper()
            if symbol_u in seen:
                continue
            ordered.append(candidate)
            seen.add(symbol_u)
            if len(ordered) >= pair_limit:
                break
        if not ordered:
            ordered = pairs[: pair_limit]
        for pair in ordered:
            stream = MarketDataStream(symbol=pair.symbol, chain=PRIMARY_CHAIN)
            bot = TradingBot(db=self.db, stream=stream, pipeline=self.pipeline)
            bot.configure_route(pair.symbol, pair.tokens)
            bot.stable_checkpoint_ratio = self.stable_checkpoint_ratio
            bot.max_trade_share = 0.12
            if readiness and not readiness.get("ready"):
                bot.live_trading_enabled = False
            if hasattr(bot, "apply_transition_plan"):
                bot.apply_transition_plan(transition_plan)
            self.bots.append(bot)

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
        self._tasks = [asyncio.create_task(bot.start()) for bot in self.bots]
        self._tasks.append(asyncio.create_task(self._drain_trades()))
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log_message("ghost-supervisor", f"unexpected supervisor error: {exc}", severity="error")

    async def stop(self) -> None:
        await asyncio.gather(*(bot.stop() for bot in self.bots), return_exceptions=True)
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
