"""Multi-timescale horizon sweep.

The base strategies only look at the in-memory ``RouteState.samples`` deque
(``maxlen=720`` ticks — at most an hour or two of history), so every native
signal is intraday. This module runs the *full-window* strategies at a set of
longer timescales (5h → 1w) by feeding each one a resampled **HorizonView** of
the pair's stored history, so the same z-score / breakout / EMA-cross / RSI
logic becomes a genuine multi-day signal.

History source (backfill): the persisted ``market_stream`` tick table plus the
``data/historical_ohlcv/<chain>/*_<SYMBOL>.json`` bar archives — both resampled
to the horizon's bar interval and merged (live ticks win on overlap). This lets
the 1d/3d/5d/1w strategies find opportunities immediately instead of waiting
days for a live window to fill.

Each swept strategy gets its own ``strategy_id`` (``ema_cross@1d`` …) so it
flows into the per-strategy ghost ledger and graduates independently — exactly
the cross-horizon variability we want.

Politeness: history is cached per ``(symbol, bar_sec)`` with a bar-scaled TTL,
so a scheduler tick that fans 24 wrappers over 25 pairs hits the DB at most
once per symbol per horizon-tier per TTL, not once per strategy per tick.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from trading.strategies.base import Strategy, StrategyContext, env_flag

# label, bar_sec, window_bars — window_bars must exceed the widest base
# strategy min_samples (ema_cross/bollinger need 40).
HORIZON_SPECS: List[Tuple[str, int, int]] = [
    ("5h", 300, 72),      # 6.0h of 5m bars
    ("12h", 900, 64),     # 16.0h of 15m bars
    ("1d", 1800, 64),     # 32.0h of 30m bars
    ("3d", 3600, 72),     # 3.0d of 1h bars
    ("5d", 7200, 66),     # 5.5d of 2h bars
    ("1w", 14400, 60),    # 10.0d of 4h bars
]

# Only strategies that read the FULL sample window (no wall-clock lookback of
# their own) resample correctly onto a HorizonView. The trend rules (Donchian,
# SuperTrend, MACD) benefit most from longer timescales.
SWEEP_STRATEGY_IDS: Tuple[str, ...] = (
    "ema_cross",
    "rsi_reversal",
    "bollinger_squeeze",
    "volume_spike",
    "macd_momentum",
    "stochastic_reversal",
    "donchian_breakout",
    "supertrend_follow",
    "obv_accumulation",
)


def _db_path() -> str:
    return os.getenv("TRADING_CACHE_DB", os.path.join("storage", "trading_cache.db"))


class HorizonView:
    """Minimal RouteState stand-in exposing resampled bars as ``samples``."""

    __slots__ = ("symbol", "base_token", "quote_token", "samples")

    def __init__(self, symbol: str, base_token: str, quote_token: str,
                 samples: List[Tuple[float, float, float]]) -> None:
        self.symbol = symbol
        self.base_token = base_token
        self.quote_token = quote_token
        self.samples = samples


class HistoryProvider:
    """Resamples persisted history into (ts, price, volume) bars, cached."""

    def __init__(self, max_entries: int = 512) -> None:
        # (symbol, bar_sec) -> (built_at, bars)
        self._cache: Dict[Tuple[str, int], Tuple[float, List[Tuple[float, float, float]]]] = {}
        self._max_entries = max_entries

    # -- public -------------------------------------------------------------
    def get_bars(self, symbol: str, chain: str, bar_sec: int,
                 window_bars: int) -> List[Tuple[float, float, float]]:
        window_sec = float(bar_sec) * float(window_bars) * 1.5  # slack for gaps
        ttl = min(600.0, max(60.0, float(bar_sec)))
        now = time.time()
        key = (symbol, int(bar_sec))
        hit = self._cache.get(key)
        if hit is not None and (now - hit[0]) < ttl:
            return hit[1][-window_bars:]
        try:
            bars = self._build(symbol, chain, int(bar_sec), window_sec, now)
        except Exception:
            bars = hit[1] if hit else []
        if len(self._cache) >= self._max_entries:
            oldest = min(self._cache.items(), key=lambda kv: kv[1][0])[0]
            self._cache.pop(oldest, None)
        self._cache[key] = (now, bars)
        return bars[-window_bars:]

    # -- internals ----------------------------------------------------------
    def _build(self, symbol: str, chain: str, bar_sec: int, window_sec: float,
               now: float) -> List[Tuple[float, float, float]]:
        cutoff = now - window_sec
        buckets: Dict[int, Tuple[float, float, float]] = {}  # bucket_ts -> (last_ts, close, vol)
        # 1) archive JSON first (older, lower priority on overlap)
        self._merge_json(symbol, chain, bar_sec, cutoff, buckets)
        # 2) live market_stream ticks (win on overlap)
        self._merge_market_stream(symbol, bar_sec, cutoff, buckets)
        out = [(float(bts), c[1], c[2]) for bts, c in buckets.items()]
        out.sort(key=lambda r: r[0])
        return out

    def _merge_market_stream(self, symbol: str, bar_sec: int, cutoff: float,
                             buckets: Dict[int, Tuple[float, float, float]]) -> None:
        path = _db_path()
        if not os.path.exists(path):
            return
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
        try:
            rows = conn.execute(
                "SELECT ts, price, volume FROM market_stream "
                "WHERE symbol=? AND ts>=? ORDER BY ts",
                (symbol, cutoff),
            ).fetchall()
        finally:
            conn.close()
        for ts, price, vol in rows:
            if price is None or price <= 0:
                continue
            self._accumulate(buckets, bar_sec, float(ts), float(price), float(vol or 0.0), live=True)

    def _merge_json(self, symbol: str, chain: str, bar_sec: int, cutoff: float,
                    buckets: Dict[int, Tuple[float, float, float]]) -> None:
        pattern = os.path.join("data", "historical_ohlcv", "*", f"*_{symbol}.json")
        files = glob.glob(pattern)
        if not files:
            return
        # Prefer the archive with the most bars (deepest history).
        best = max(files, key=lambda f: os.path.getsize(f))
        try:
            data = json.load(open(best, "r", encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, list):
            return
        for bar in data:
            try:
                ts = float(bar.get("timestamp") or bar.get("ts") or 0.0)
                close = float(bar.get("close") or bar.get("price") or 0.0)
                vol = float(bar.get("net_volume") or bar.get("volume") or 0.0)
            except (AttributeError, TypeError, ValueError):
                continue
            if ts < cutoff or close <= 0:
                continue
            self._accumulate(buckets, bar_sec, ts, close, vol, live=False)

    @staticmethod
    def _accumulate(buckets: Dict[int, Tuple[float, float, float]], bar_sec: int,
                    ts: float, price: float, vol: float, *, live: bool) -> None:
        bts = int(ts // bar_sec) * bar_sec
        prev = buckets.get(bts)
        if prev is None:
            buckets[bts] = (ts, price, vol)
            return
        prev_ts, prev_close, prev_vol = prev
        # Live ticks override archive for the same bucket; within a source keep
        # the latest tick as the close and sum volume.
        if live or ts >= prev_ts:
            buckets[bts] = (ts, price, prev_vol + vol)
        else:
            buckets[bts] = (prev_ts, prev_close, prev_vol + vol)


_PROVIDER = HistoryProvider()


class MultiHorizonStrategy(Strategy):
    """Runs a full-window base strategy against a resampled HorizonView."""

    min_samples = 0  # self-gates on the view; evaluate_all must always call us

    def __init__(self, base: Strategy, label: str, bar_sec: int, window_bars: int) -> None:
        self._base = base
        self._label = label
        self._bar_sec = int(bar_sec)
        self._window_bars = int(window_bars)
        self.strategy_id = f"{base.strategy_id}@{label}"
        self.default_horizon = label
        self._min_bars = max(int(getattr(base, "min_samples", 24)), 24)

    def enabled(self) -> bool:
        return env_flag("STRATEGY_MULTIHORIZON_ENABLED", "1") and self._base.enabled()

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        bars = _PROVIDER.get_bars(state.symbol, ctx.chain, self._bar_sec, self._window_bars)
        if len(bars) < self._min_bars:
            return None
        view = HorizonView(state.symbol, state.base_token, state.quote_token, bars)
        cand = self._base.evaluate(view, ctx)
        if not cand:
            return None
        directive = cand.get("directive")
        if directive is not None:
            directive.strategy_id = self.strategy_id
            directive.horizon = self._label
            directive.reason = f"[{self._label}] {directive.reason}"
        meta = cand.setdefault("meta", {})
        meta["strategy"] = self.strategy_id
        meta["horizon"] = self._label
        return cand


def build_multihorizon_strategies(base_strategies: Sequence[Strategy]) -> List[MultiHorizonStrategy]:
    """Wrap eligible base strategies across every horizon tier."""
    by_id = {s.strategy_id: s for s in base_strategies}
    wrapped: List[MultiHorizonStrategy] = []
    for sid in SWEEP_STRATEGY_IDS:
        base = by_id.get(sid)
        if base is None:
            continue
        for label, bar_sec, window_bars in HORIZON_SPECS:
            wrapped.append(MultiHorizonStrategy(base, label, bar_sec, window_bars))
    return wrapped
