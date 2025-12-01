from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from trading.data_loader import HistoricalDataLoader
from trading.pipeline import TrainingPipeline


def _write_pair_index(path: Path, symbols: List[str]) -> None:
    payload: Dict[str, Dict[str, object]] = {}
    for idx, symbol in enumerate(symbols):
        payload[symbol] = {"symbol": symbol, "index": idx}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _synthetic_rows(count: int = 160) -> List[Dict[str, float]]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows: List[Dict[str, float]] = []
    for i in range(count):
        ts = int(start.timestamp() + i * 60)
        price = 100.0 + i * 0.5
        rows.append(
            {
                "timestamp": ts,
                "open": price - 0.3,
                "high": price + 0.4,
                "low": price - 0.5,
                "close": price,
                "net_volume": 3_000 + i * 2,
                "buy_volume": 1_600 + i,
                "sell_volume": 1_400 + i,
            }
        )
    return rows


def test_lab_preview_series(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pair_index = tmp_path / "pair_index_base.json"
    _write_pair_index(pair_index, ["ETH-USDC"])
    monkeypatch.setenv("PAIR_INDEX_PATH", str(pair_index))
    monkeypatch.setenv("HISTORICAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRAIN_POSITIVE_FLOOR", "0.1")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))

    data_path = tmp_path / "history_ETH-USDC.json"
    data_path.write_text(json.dumps(_synthetic_rows(180)), encoding="utf-8")

    def fake_load_news(self: HistoricalDataLoader):
        self.news_index = {}
        return []

    monkeypatch.setattr(HistoricalDataLoader, "_load_news", fake_load_news, raising=True)

    pipeline = TrainingPipeline()
    preview = pipeline.lab_preview_series(
        ["history_ETH-USDC.json"],
        batch_size=16,
        include_news=False,
    )

    assert preview["series"], "Expected preview series data"
    first = preview["series"][0]
    assert "timestamp" in first and "predicted_price" in first
    assert "metrics" in preview
    assert preview["metrics"]["samples"] >= len(preview["series"])


def test_pipeline_safe_float_handles_nested() -> None:
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    assert pipeline._safe_float({"value": "3.14"}) == pytest.approx(3.14)
    assert pipeline._safe_float([{"score": 2.5}]) == pytest.approx(2.5)
    assert pipeline._safe_float(np.array([1.23])) == pytest.approx(1.23)
