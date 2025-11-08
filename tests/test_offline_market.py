from __future__ import annotations

import json
from pathlib import Path

from services.offline_market import OfflinePriceStore


def _write_snapshot(path: Path) -> None:
    payload = {
        "generated_at": 1700000000,
        "data": [
            {"symbol": "ETH", "name": "Ethereum", "price_usd": 2000.0, "volume_24h": 1234, "source": "mock"},
            {"symbol": "BTC", "name": "Bitcoin", "price_usd": 30000.0, "volume_24h": 555, "source": "mock"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_ohlcv(path: Path) -> None:
    rows = [
        {"timestamp": 1000, "open": 1.0, "high": 1.2, "low": 0.8, "close": 1.1, "net_volume": 10.0},
        {"timestamp": 2000, "open": 1.1, "high": 1.3, "low": 1.0, "close": 1.25, "net_volume": 12.0},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_offline_store_reads_snapshot_and_ohlcv(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.json"
    _write_snapshot(snapshot)
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    ohlcv_root = tmp_path / "ohlcv"
    ohlcv_root.mkdir()
    ohlcv_file = ohlcv_root / "0000_ETH-USDC.json"
    _write_ohlcv(ohlcv_file)

    store = OfflinePriceStore(snapshot_path=snapshot, history_dir=history_dir, ohlcv_root=ohlcv_root, max_age=0.01)

    eth = store.get_price("ETH")
    assert eth is not None
    assert eth.price == 2000.0

    tail = store.get_ohlcv_tail("ETH-USDC", bars=1)
    assert tail
    assert tail[-1]["close"] == 1.25


def test_offline_store_snapshots_limit(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.json"
    _write_snapshot(snapshot)
    store = OfflinePriceStore(snapshot_path=snapshot, history_dir=tmp_path / "history", ohlcv_root=tmp_path / "ohlcv")
    snaps = store.snapshots(limit=1)
    assert len(snaps) == 1
    assert snaps[0].symbol == "ETH"
