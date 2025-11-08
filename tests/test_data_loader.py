from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pytest
import time

from trading.data_loader import HistoricalDataLoader
from services.system_profile import SystemProfile


def _write_pair_index(path: Path, symbols: List[str]) -> None:
    payload: Dict[str, Dict[str, object]] = {}
    for idx, symbol in enumerate(symbols):
        payload[symbol] = {"symbol": symbol, "index": idx}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _synthetic_rows(count: int = 128) -> List[Dict[str, float]]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows: List[Dict[str, float]] = []
    for i in range(count):
        ts = int((start.timestamp()) + i * 60)
        price = 100.0 + i * 0.25
        rows.append(
            {
                "timestamp": ts,
                "open": price - 0.1,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price,
                "net_volume": 500 + i,
                "buy_volume": 260 + i * 0.5,
                "sell_volume": 240 + i * 0.5,
            }
        )
    return rows


@pytest.fixture
def historical_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pair_index = tmp_path / "pair_index_base.json"
    _write_pair_index(pair_index, ["ETH-USDC", "USDC-WETH"])
    monkeypatch.setenv("PAIR_INDEX_PATH", str(pair_index))
    monkeypatch.setenv("HISTORICAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CRYPTO_RSS_FEEDS", "")
    monkeypatch.setenv("TRAIN_POSITIVE_FLOOR", "0.1")

    # ensure loader does not hit network for news
    def fake_load_news(self: HistoricalDataLoader) -> List[Dict[str, object]]:
        ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
        item = {
            "timestamp": ts,
            "headline": "ETH rallies on Base network expansion",
            "article": "ETH price momentum continues as Base liquidity deepens.",
            "sentiment": "positive",
            "tokens": {"ETH", "USDC"},
        }
        self.news_index = {"ETH": [0], "USDC": [0]}
        return [item]

    monkeypatch.setattr(HistoricalDataLoader, "_load_news", fake_load_news, raising=True)
    return tmp_path


def test_build_dataset_shapes(historical_tmp: Path) -> None:
    data_path = historical_tmp / "history_ETH-USDC.json"
    data_path.write_text(json.dumps(_synthetic_rows(200)), encoding="utf-8")

    loader = HistoricalDataLoader(data_dir=historical_tmp, max_files=1, max_samples_per_file=64)
    inputs, targets = loader.build_dataset(window_size=16, sent_seq_len=12, tech_count=8)
    assert inputs is not None and targets is not None

    count = next(iter(inputs.values())).shape[0]
    assert count > 0
    assert inputs["price_vol_input"].shape == (count, 16, 2)
    assert inputs["sentiment_seq"].shape == (count, 12, 1)
    assert inputs["tech_input"].shape[1] == 8
    assert targets["price_mu"].shape == (count, 1)
    assert targets["price_dir"].shape == (count, 1)

    # Cached dataset should be identical on the second call.
    inputs_cached, targets_cached = loader.build_dataset(window_size=16, sent_seq_len=12, tech_count=8)
    assert np.array_equal(inputs["price_vol_input"], inputs_cached["price_vol_input"])
    assert np.array_equal(targets["price_mu"], targets_cached["price_mu"])


def test_sample_meta_includes_horizons(historical_tmp: Path) -> None:
    data_path = historical_tmp / "history_ETH-USDC.json"
    data_path.write_text(json.dumps(_synthetic_rows(240)), encoding="utf-8")

    loader = HistoricalDataLoader(data_dir=historical_tmp, max_files=1, max_samples_per_file=64)
    loader.build_dataset(window_size=20, sent_seq_len=12, tech_count=8)
    meta = loader.last_sample_meta()
    records = meta.get("records", [])
    assert records, "sample metadata records should be captured"
    record = records[0]
    assert record.get("lookahead_sec") is not None and record["lookahead_sec"] > 0
    horizons = record.get("horizons")
    assert isinstance(horizons, dict) and horizons, "per-horizon returns should be recorded"


def test_expand_limits_invalidate_cache(historical_tmp: Path) -> None:
    (historical_tmp / "history_ETH-USDC.json").write_text(json.dumps(_synthetic_rows(80)), encoding="utf-8")
    loader = HistoricalDataLoader(data_dir=historical_tmp, max_files=1, max_samples_per_file=16)

    loader.build_dataset(window_size=10, sent_seq_len=8, tech_count=6)
    assert loader._dataset_cache  # cache populated

    loader.expand_limits(factor=2.0, file_cap=4, sample_cap=128)
    assert loader.max_files == 2
    assert loader.max_samples_per_file >= 32
    assert not loader._dataset_cache  # expanding clears cache


def test_apply_system_profile_caps_limits(historical_tmp: Path) -> None:
    (historical_tmp / "history_ETH-USDC.json").write_text(json.dumps(_synthetic_rows(80)), encoding="utf-8")
    loader = HistoricalDataLoader(data_dir=historical_tmp, max_files=8, max_samples_per_file=1024)
    profile = SystemProfile(cpu_count=4, total_memory_gb=12.0, max_threads=4, is_low_power=True, memory_pressure=True)
    loader.apply_system_profile(profile)
    assert loader.max_files <= 6
    assert loader.max_samples_per_file <= 512


def test_dataset_builds_without_news(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pair_index = tmp_path / "pair_index_base.json"
    _write_pair_index(pair_index, ["ETH-USDC"])
    monkeypatch.setenv("PAIR_INDEX_PATH", str(pair_index))
    monkeypatch.setenv("HISTORICAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRAIN_POSITIVE_FLOOR", "0.1")

    (tmp_path / "history_ETH-USDC.json").write_text(json.dumps(_synthetic_rows(120)), encoding="utf-8")

    def fake_load_news(self: HistoricalDataLoader):
        self.news_index = {}
        return []

    monkeypatch.setattr(HistoricalDataLoader, "_load_news", fake_load_news, raising=True)

    loader = HistoricalDataLoader(data_dir=tmp_path, max_files=1, max_samples_per_file=32)
    inputs, targets = loader.build_dataset(window_size=16, sent_seq_len=12, tech_count=8)
    assert inputs is not None and targets is not None
    headlines = inputs["headline_text"].reshape(-1)
    assert any(str(headline).strip() for headline in headlines)


def test_request_news_backfill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CRYPTOPANIC_API_KEY", "test-token")

    (tmp_path / "history_ETH-USDC.json").write_text(json.dumps(_synthetic_rows(80)), encoding="utf-8")

    def no_news(self: HistoricalDataLoader):
        return []

    monkeypatch.setattr(HistoricalDataLoader, "_load_news", no_news, raising=True)

    loader = HistoricalDataLoader(data_dir=tmp_path, max_files=1, max_samples_per_file=16)

    def fake_fetch(self: HistoricalDataLoader, **kwargs):
        return [
            {
                "timestamp": kwargs["since_ts"] + 5,
                "headline": "ETH rally",
                "article": "ETH moved higher on network upgrades.",
                "sentiment": "positive",
                "tokens": ["ETH", "USDC"],
            }
        ]

    monkeypatch.setattr(HistoricalDataLoader, "_fetch_cryptopanic_posts", fake_fetch, raising=True)
    added_first = loader.request_news_backfill(symbols=["ETH-USDC"], lookback_sec=3600, center_ts=int(time.time()))
    assert added_first is True
    count_after_first = len(loader.news_items)
    added_second = loader.request_news_backfill(symbols=["ETH-USDC"], lookback_sec=3600, center_ts=int(time.time()))
    assert added_second is False
    assert len(loader.news_items) == count_after_first


def test_focus_alias_handles_usdbc(historical_tmp: Path) -> None:
    rows = _synthetic_rows(160)
    (historical_tmp / "history_USDC-WETH.json").write_text(json.dumps(rows), encoding="utf-8")
    loader = HistoricalDataLoader(data_dir=historical_tmp, max_files=1, max_samples_per_file=64)
    inputs, targets = loader.build_dataset(
        window_size=20,
        sent_seq_len=12,
        tech_count=8,
        focus_assets=["WETH-USDBC"],
    )
    assert inputs is not None and targets is not None


def test_rebalance_horizons_adjusts_stride(monkeypatch: pytest.MonkeyPatch, historical_tmp: Path) -> None:
    monkeypatch.setenv("DATASET_SAMPLING_STRIDE", "3")
    (historical_tmp / "history_ETH-USDC.json").write_text(json.dumps(_synthetic_rows(80)), encoding="utf-8")
    loader = HistoricalDataLoader(data_dir=historical_tmp, max_files=1, max_samples_per_file=32)
    assert loader.sampling_stride() == 3
    adjustments = loader.rebalance_horizons({"short": 16.0}, focus_assets=["ETH-USDC"])
    assert loader.sampling_stride() == 1
    assert adjustments.get("sampling_stride") == 1
    loader.rebalance_horizons({"short": 0.0, "mid": 0.0, "long": 0.0}, focus_assets=None)
    assert loader.sampling_stride() == 3
