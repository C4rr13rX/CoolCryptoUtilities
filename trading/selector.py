from __future__ import annotations

import json
from dataclasses import dataclass
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from db import TradingDatabase, get_db
from trading.bot import TradingBot
from trading.data_stream import MarketDataStream
from trading.pipeline import TrainingPipeline

STABLE_TOKENS = {"USDC", "USDT", "DAI", "BUSD", "TUSD", "USDP", "USDD", "USDS", "GUSD"}


@dataclass
class PairCandidate:
    symbol: str
    tokens: List[str]
    avg_volume: float
    volatility: float
    score: float
    datapath: Path


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
    best: List[PairCandidate] = []
    seen_tokens: set[str] = set()
    for cand in candidates:
        if len(best) >= limit:
            break
        if cand.avg_volume < min_volume:
            continue
        token_key = tuple(sorted(cand.tokens))
        if token_key in seen_tokens:
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
