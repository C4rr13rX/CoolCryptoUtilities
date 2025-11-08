from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class OfflineSnapshot:
    symbol: str
    name: str
    price: float
    volume: Optional[float]
    source: str
    ts: float
    change_24h: Optional[float]

    def to_dict(self) -> Dict[str, float | str | None]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "price": self.price,
            "volume": self.volume,
            "source": self.source,
            "ts": self.ts,
            "change_24h": self.change_24h,
        }


class OfflinePriceStore:
    """
    Lightweight reader for locally persisted market snapshots and OHLCV dumps.
    Used when the live network feeds are unavailable so the trading stack
    continues operating with best-effort data rather than idling.
    """

    def __init__(
        self,
        *,
        snapshot_path: str | Path = "data/market_snapshots.json",
        history_dir: str | Path = "data/market_snapshots_history",
        ohlcv_root: str | Path = "data/historical_ohlcv",
        max_age: float = 120.0,
    ) -> None:
        self.snapshot_path = Path(snapshot_path).expanduser()
        self.history_dir = Path(history_dir).expanduser()
        self.ohlcv_root = Path(ohlcv_root).expanduser()
        self.max_age = max(15.0, float(max_age))
        self._lock = threading.Lock()
        self._cache: Dict[str, OfflineSnapshot] = {}
        self._cache_ts: float = 0.0
        self._cache_mtime: float = 0.0
        self._ohlcv_cache: Dict[str, List[dict]] = {}
        self._ohlcv_index: Dict[str, Path] = {}
        self._build_ohlcv_index()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_price(self, symbol: str) -> Optional[OfflineSnapshot]:
        self._refresh_cache_if_needed()
        return self._cache.get(symbol.upper())

    def snapshots(self, *, symbols: Optional[Sequence[str]] = None, limit: int = 25) -> List[OfflineSnapshot]:
        self._refresh_cache_if_needed()
        records = list(self._cache.values())
        if symbols:
            wanted = {sym.upper() for sym in symbols if sym}
            records = [snap for snap in records if snap.symbol in wanted]
        records.sort(key=lambda snap: snap.ts, reverse=True)
        return records[: limit or len(records)]

    def get_ohlcv_tail(self, symbol: str, bars: int = 180) -> List[dict]:
        symbol_u = symbol.upper()
        rows = self._load_ohlcv_rows(symbol_u)
        if not rows:
            return []
        if bars <= 0:
            return rows[:]
        return rows[-bars:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ohlcv_index(self) -> None:
        self._ohlcv_index.clear()
        if not self.ohlcv_root.exists():
            return
        try:
            for path in sorted(self.ohlcv_root.rglob("*.json")):
                stem = path.stem
                if "_" in stem:
                    _, label = stem.split("_", 1)
                else:
                    label = stem
                self._ohlcv_index.setdefault(label.upper(), path)
        except Exception:
            self._ohlcv_index = {}

    def _refresh_cache_if_needed(self) -> None:
        now = time.time()
        if self._cache and (now - self._cache_ts) < self.max_age:
            return
        snapshot_mtime = self._file_mtime(self.snapshot_path)
        history_mtime = self._latest_history_mtime()
        newest_source = max(snapshot_mtime, history_mtime)
        if self._cache and newest_source <= self._cache_mtime:
            self._cache_ts = now
            return
        with self._lock:
            if self._cache and (now - self._cache_ts) < self.max_age:
                return
            data: Dict[str, OfflineSnapshot] = {}
            records = self._load_snapshot_file(self.snapshot_path)
            if not records:
                records = []
            latest_history = self._load_snapshot_file(self._latest_history_file())
            if latest_history:
                records.extend(latest_history)
            timestamp = now
            for entry in records:
                try:
                    symbol = str(entry.get("symbol", "")).upper()
                    if not symbol:
                        continue
                    name = str(entry.get("name") or symbol)
                    price = _safe_float(entry.get("price_usd") or entry.get("price") or entry.get("priceUsd"))
                    if price <= 0:
                        continue
                    volume = entry.get("volume_24h") or entry.get("volumeUsd24Hr") or entry.get("volume")
                    change = entry.get("percent_change_24h") or entry.get("changePercent24Hr")
                    source = str(entry.get("source") or "offline")
                    ts = float(entry.get("ts") or entry.get("timestamp") or timestamp)
                    data[symbol] = OfflineSnapshot(
                        symbol=symbol,
                        name=name,
                        price=price,
                        volume=_safe_float(volume, default=0.0) if volume is not None else None,
                        source=source,
                        ts=ts,
                        change_24h=_safe_float(change) if change is not None else None,
                    )
                except Exception:
                    continue
            if data:
                self._cache = data
                self._cache_ts = now
                self._cache_mtime = newest_source or now

    def _load_snapshot_file(self, path: Optional[Path]) -> List[Dict[str, object]]:
        if not path or not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                return data
        if isinstance(payload, list):
            return payload
        return []

    def _file_mtime(self, path: Path) -> float:
        if not path or not path.exists():
            return 0.0
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _latest_history_file(self) -> Optional[Path]:
        if not self.history_dir.exists():
            return None
        try:
            files = sorted(self.history_dir.glob("*.json"))
            if not files:
                return None
            return files[-1]
        except Exception:
            return None

    def _latest_history_mtime(self) -> float:
        latest = self._latest_history_file()
        return self._file_mtime(latest) if latest else 0.0

    def _load_ohlcv_rows(self, symbol: str) -> List[dict]:
        cached = self._ohlcv_cache.get(symbol)
        if cached is not None:
            return cached
        path = self._ohlcv_index.get(symbol)
        if not path or not path.exists():
            return []
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(rows, list):
            self._ohlcv_cache[symbol] = rows
            return rows
        return []


__all__ = ["OfflinePriceStore", "OfflineSnapshot"]
