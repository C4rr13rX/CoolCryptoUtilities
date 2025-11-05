from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pytest

from trading.data_loader import HistoricalDataLoader


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
    _write_pair_index(pair_index, ["ETH-USDC"])
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


def test_expand_limits_invalidate_cache(historical_tmp: Path) -> None:
    (historical_tmp / "history_ETH-USDC.json").write_text(json.dumps(_synthetic_rows(80)), encoding="utf-8")
    loader = HistoricalDataLoader(data_dir=historical_tmp, max_files=1, max_samples_per_file=16)

    loader.build_dataset(window_size=10, sent_seq_len=8, tech_count=6)
    assert loader._dataset_cache  # cache populated

    loader.expand_limits(factor=2.0, file_cap=4, sample_cap=128)
    assert loader.max_files == 2
    assert loader.max_samples_per_file >= 32
    assert not loader._dataset_cache  # expanding clears cache
