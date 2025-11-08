from __future__ import annotations

import asyncio
import json
from pathlib import Path
import os
import time
from typing import Any, Dict, List, Optional

from dataclasses import dataclass
from urllib.parse import quote_plus
import numpy as np
import requests

from db import TradingDatabase, get_db
from trading.bot import TradingBot
from trading.data_stream import MarketDataStream, _split_symbol, TOKEN_NORMALIZATION
from trading.pipeline import TrainingPipeline
from trading.portfolio import PortfolioState
from trading.constants import PRIMARY_CHAIN, PRIMARY_SYMBOL, top_pairs, pair_index_entries
from trading.constants import PRIMARY_CHAIN, PRIMARY_SYMBOL
from services.logging_utils import log_message

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
_db: TradingDatabase = get_db()

for _core_symbol in list(_ALWAYS_LIVE_SYMBOLS):
    try:
        _db.clear_pair_suppression(_core_symbol)
    except Exception:
        pass


def _load_top_symbols(limit: int = 100) -> List[str]:
    return top_pairs(limit=limit)


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


def _has_live_price(symbol: str) -> bool:
    symbol_u = symbol.upper()
    if symbol_u in _ALWAYS_LIVE_SYMBOLS:
        _LIVE_PAIR_CACHE[symbol_u] = True
        return True
    if _db.is_pair_suppressed(symbol_u):
        record = _db.get_pair_suppression(symbol_u) or {}
        remaining = float(record.get("release_ts", 0.0)) - time.time()
        wait_minutes = max(0.0, remaining / 60.0)
        reason = record.get("reason") or "suppressed"
        print(
            f"[pair-select] suppressed {symbol_u}: {reason}; retry in ~{wait_minutes:.1f} min."
        )
        _LIVE_PAIR_CACHE[symbol_u] = False
        return False

    cached = _LIVE_PAIR_CACHE.get(symbol_u)
    if cached is not None:
        return cached
    probe_result = _probe_dexscreener(symbol_u)
    if probe_result:
        _LIVE_PAIR_CACHE[symbol_u] = True
        _db.clear_pair_suppression(symbol_u)
        return True
    if probe_result is None:
        print(f"[pair-select] probe deferred for {symbol_u}: dexscreener unreachable.")
        result = _has_historical_price(symbol_u)
        _LIVE_PAIR_CACHE[symbol_u] = result
        return result
    result = _has_historical_price(symbol_u)
    _LIVE_PAIR_CACHE[symbol_u] = result
    if not result:
        _db.record_pair_suppression(
            symbol_u,
            "no_live_market_data",
            ttl_seconds=_SUPPRESSION_TTL,
            metadata={"checked_at": time.time()},
        )
        print(f"[pair-select] skipping {symbol_u}: no live market data sources responded.")
    return result


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


def select_pairs(
    *,
    limit: int = 6,
    min_volume: float = 25.0,
    data_dir: Path = Path("data/historical_ohlcv"),
) -> List[PairCandidate]:
    candidates = analyse_historical_pairs(data_dir=data_dir)
    candidate_map = {cand.symbol: cand for cand in candidates}
    best: List[PairCandidate] = []
    seen_tokens: set[str] = set()

    def try_add_candidate(symbol: str) -> None:
        cand = candidate_map.get(symbol.upper())
        if not cand:
            return
        token_key = tuple(sorted(cand.tokens))
        if token_key in seen_tokens:
            return
        if not _has_live_price(cand.symbol):
            return
        best.append(cand)
        seen_tokens.add(token_key)

    for symbol in DEFAULT_LIVE_PAIRS:
        if len(best) >= limit:
            break
        try_add_candidate(symbol)

    for symbol in _load_top_symbols(limit * 5):
        if len(best) >= limit:
            break
        try_add_candidate(symbol)

    held_symbols: set[str] = set()
    try:
        portfolio = PortfolioState()
        portfolio.refresh(force=True)
        held_symbols = {sym for (_, sym) in portfolio.holdings.keys()}
    except Exception:
        held_symbols = set()

    for held in held_symbols:
        if len(best) >= limit:
            break
        if held in STABLE_TOKENS:
            continue
        for stable in ("USDC", "USDT", "DAI"):
            if len(best) >= limit:
                break
            try_add_candidate(f"{held}-{stable}")
            try_add_candidate(f"{stable}-{held}")

    scanned = 0
    max_scan = max(limit * 30, limit + 5)
    for cand in candidates:
        if len(best) >= limit:
            break
        scanned += 1
        if scanned > max_scan and best:
            break
        if cand.avg_volume < min_volume:
            continue
        token_key = tuple(sorted(cand.tokens))
        if token_key in seen_tokens:
            continue
        if not _has_live_price(cand.symbol):
            continue
        seen_tokens.add(token_key)
        best.append(cand)
    return best


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
        self.stable_checkpoint_ratio = stable_checkpoint_ratio
        self.bots: List[TradingBot] = []
        self._tasks: List[asyncio.Task] = []

    def build(self) -> None:
        if self.bots:
            return
        focus_assets, _ = self.pipeline.ghost_focus_assets()
        readiness = self.pipeline.live_readiness_report()
        if readiness and readiness.get("reason") == "no_confusion_data":
            if self.pipeline.prime_confusion_windows():
                readiness = self.pipeline.live_readiness_report()
        if readiness:
            log_message(
                "ghost-supervisor",
                "live readiness snapshot",
                severity="info" if readiness.get("ready") else "warning",
                details=readiness,
            )
        pairs = select_pairs(limit=self.pair_limit)
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
            if len(ordered) >= self.pair_limit:
                break
        if not ordered:
            ordered = pairs[: self.pair_limit]
        for pair in ordered:
            stream = MarketDataStream(symbol=pair.symbol, chain=PRIMARY_CHAIN)
            bot = TradingBot(db=self.db, stream=stream, pipeline=self.pipeline)
            bot.configure_route(pair.symbol, pair.tokens)
            bot.stable_checkpoint_ratio = self.stable_checkpoint_ratio
            bot.max_trade_share = 0.12
            if readiness and not readiness.get("ready"):
                bot.live_trading_enabled = False
            self.bots.append(bot)

    async def start(self) -> None:
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
