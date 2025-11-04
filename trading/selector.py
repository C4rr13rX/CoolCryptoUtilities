from __future__ import annotations

import asyncio
import json
from pathlib import Path
import os
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

STABLE_TOKENS = {"USDC", "USDT", "DAI", "BUSD", "TUSD", "USDP", "USDD", "USDS", "GUSD"}

DEFAULT_LIVE_PAIRS: List[str] = [
    "WETH-USDT",
    "USDC-WETH",
    "DAI-WETH",
    "WBTC-WETH",
    "UNI-WETH",
    "PEPE-WETH",
]

@dataclass
class PairCandidate:
    symbol: str
    tokens: List[str]
    avg_volume: float
    volatility: float
    score: float
    datapath: Path


_LIVE_PAIR_CACHE: Dict[str, bool] = {}


def _load_top_symbols(limit: int = 100) -> List[str]:
    path = Path(os.getenv("PAIR_INDEX_PATH", "data/pair_index_top2000.json"))
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
    except Exception:
        return []
    try:
        items = sorted(
            data.items(),
            key=lambda kv: int(kv[1].get("index", 0)),
        )
    except Exception:
        items = list(data.items())
    symbols: List[str] = []
    for _, info in items:
        symbol = str(info.get("symbol", "")).upper()
        if not symbol:
            continue
        symbols.append(symbol)
        if len(symbols) >= limit:
            break
    return symbols


def _token_synonyms(token: str) -> set[str]:
    token_u = token.upper()
    synonyms = {token_u}
    for original, normalized in TOKEN_NORMALIZATION.items():
        if normalized.upper() == token_u:
            synonyms.add(original.upper())
    return synonyms


def _probe_dexscreener(symbol: str) -> bool:
    base, quote = _split_symbol(symbol)
    query = quote_plus(f"{base} {quote}")
    url = f"https://api.dexscreener.com/latest/dex/search?q={query}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return False
        payload = resp.json()
    except Exception:
        return False
    pairs = payload.get("pairs") or []
    if not pairs:
        return False
    base_syn = _token_synonyms(base)
    quote_syn = _token_synonyms(quote)
    for pair in pairs:
        base_info = pair.get("baseToken") or {}
        quote_info = pair.get("quoteToken") or {}
        base_symbol = str(base_info.get("symbol") or "").upper()
        quote_symbol = str(quote_info.get("symbol") or "").upper()
        if base_symbol not in base_syn or quote_symbol not in quote_syn:
            continue
        price = pair.get("priceNative") or pair.get("priceUsd")
        try:
            if price and float(price) > 0:
                return True
        except Exception:
            continue
    return False


def _has_historical_price(symbol: str) -> bool:
    data_dir = Path("data/historical_ohlcv")
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
    cached = _LIVE_PAIR_CACHE.get(symbol)
    if cached is not None:
        return cached
    if _probe_dexscreener(symbol):
        _LIVE_PAIR_CACHE[symbol] = True
        return True
    result = _has_historical_price(symbol)
    _LIVE_PAIR_CACHE[symbol] = result
    if not result:
        print(f"[pair-select] skipping {symbol}: no live market data sources responded.")
    return result


def _load_pair_metadata() -> Dict[str, str]:
    index_path = Path("data/pair_index_top2000.json")
    if not index_path.exists():
        return {}
    try:
        with index_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    mapping: Dict[str, str] = {}
    for addr, info in data.items():
        symbol = str(info.get("symbol") or "").upper()
        if addr and symbol:
            mapping[symbol] = addr
    return mapping


def analyse_historical_pairs(
    *,
    data_dir: Path = Path("data/historical_ohlcv"),
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
        pairs = select_pairs(limit=self.pair_limit)
        for pair in pairs:
            stream = MarketDataStream(symbol=pair.symbol, chain="ethereum")
            bot = TradingBot(db=self.db, stream=stream, pipeline=self.pipeline)
            bot.configure_route(pair.symbol, pair.tokens)
            bot.stable_checkpoint_ratio = self.stable_checkpoint_ratio
            bot.max_trade_share = 0.12
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
