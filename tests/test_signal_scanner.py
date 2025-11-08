from __future__ import annotations

import json
import time
from pathlib import Path

from services import signal_scanner


class DummyDB:
    def __init__(self) -> None:
        self._kv: dict[str, object] = {}

    def list_market_pairs_since(self, *_args, **_kwargs):
        return []

    def get_market_price(self, *_args, **_kwargs):
        return None

    def average_volume(self, *_args, **_kwargs):
        return None

    def get_json(self, key: str):
        return self._kv.get(key)

    def set_json(self, key: str, value):
        self._kv[key] = value


def test_scan_price_signals_uses_historical_fallback(monkeypatch, tmp_path):
    dummy_db = DummyDB()
    monkeypatch.setattr(signal_scanner, "get_db", lambda: dummy_db)
    monkeypatch.setattr(signal_scanner, "FilterScamTokens", None)
    monkeypatch.setattr(signal_scanner, "DEFAULT_SCAN_CHAIN", "base")
    hist_root = tmp_path / "historical_ohlcv"
    hist_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(signal_scanner, "HIST_ROOT", hist_root)
    chain_dir = hist_root / "base"
    chain_dir.mkdir(parents=True, exist_ok=True)

    pair_index_template = tmp_path / "pair_index_{chain}.json"
    monkeypatch.setattr(signal_scanner, "PAIR_INDEX_TEMPLATE", str(pair_index_template))
    pair_index_path = Path(str(pair_index_template).format(chain="base"))
    pair_index_path.write_text(
        json.dumps({"0xabc": {"symbol": "TEST-USDC", "index": 0}}),
        encoding="utf-8",
    )

    now = int(time.time())
    candles = [
        {"timestamp": now - 1800, "open": 1.0, "close": 1.0, "net_volume": 150000},
        {"timestamp": now, "open": 1.0, "close": 2.0, "net_volume": 200000},
    ]
    (chain_dir / "0000_TEST-USDC.json").write_text(json.dumps(candles), encoding="utf-8")

    results, meta = signal_scanner.scan_price_signals(
        "24h",
        direction="bullish",
        limit=5,
        min_volume=10,
    )
    assert results, "expected at least one historical signal"
    entry = results[0]
    assert entry["symbol"] == "TEST-USDC"
    assert entry["source"] == "historical"
    assert meta["historical_hits"] >= 1
