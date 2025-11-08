from __future__ import annotations

import time

from datetime import datetime, timedelta, timezone
from typing import List

from services.news_ingestor import EthicalNewsIngestor, NewsSource


def test_ethic_news_ingestor_writes_parquet(tmp_path):
    now = int(time.time())
    source = NewsSource(name="TestSource", url="http://example.com/feed")
    sample_entries = [
        {
            "title": "ETH surges",
            "summary": "Ethereum gains strength against USDC.",
            "published_parsed": now,
            "link": "http://example.com/post",
        }
    ]

    ingestor = EthicalNewsIngestor(
        sources=[source],
        output_path=tmp_path / "ethical.parquet",
        cache_dir=tmp_path / "cache",
        fetcher=lambda _src: sample_entries,
    )

    rows = ingestor.harvest(tokens={"ETH", "USDC"}, start_ts=now - 60, end_ts=now + 60)

    assert rows, "Rows should be harvested"
    assert (tmp_path / "ethical.parquet").exists()
    assert any("ETH" in row["tokens"] for row in rows)


def test_custom_catalog_and_harvest_window(monkeypatch, tmp_path):
    config_path = tmp_path / "sources.json"
    config_path.write_text(
        '[{"name": "ConfigSource", "url": "http://config.test/rss", "topics": ["eth", "l2"]}]',
        encoding="utf-8",
    )
    monkeypatch.setenv("ETHICAL_NEWS_SOURCES_PATH", str(config_path))
    now = datetime.now(timezone.utc)
    sample_entries = [
        {
            "title": "Layer2 growth",
            "summary": "ETH adoption on L2 surges.",
            "published_parsed": int(now.timestamp()),
            "link": "http://config.test/post",
        }
    ]
    ingestor = EthicalNewsIngestor(
        sources=[],
        output_path=tmp_path / "window.parquet",
        cache_dir=tmp_path / "cache",
        fetcher=lambda src: sample_entries if src.name == "ConfigSource" else [],
    )
    rows = ingestor.harvest_window(
        tokens=set(),
        start=now - timedelta(minutes=5),
        end=now + timedelta(minutes=5),
    )
    assert rows
    assert rows[0]["source"] == "ConfigSource"


def test_harvest_windows_batches_requests_multiple_ranges(monkeypatch, tmp_path):
    now = int(time.time())
    source = NewsSource(name="BatchSource", url="http://batch.test/feed", topics=("ETH",))
    ranges = [(now - 300, now - 200), (now - 100, now)]
    calls: List[int] = []
    empty_catalog = tmp_path / "sources.json"
    empty_catalog.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("ETHICAL_NEWS_SOURCES_PATH", str(empty_catalog))

    def fake_fetch(_src):
        idx = len(calls)
        calls.append(idx)
        ts = ranges[idx][0] + 10
        return [
            {
                "title": f"Batch {idx}",
                "summary": "ETH window coverage",
                "published_parsed": ts,
                "link": f"http://batch.test/{idx}",
            }
        ]

    ingestor = EthicalNewsIngestor(
        sources=[source],
        output_path=tmp_path / "batches.parquet",
        cache_dir=tmp_path / "cache",
        fetcher=fake_fetch,
    )
    rows = ingestor.harvest_windows(tokens={"ETH"}, ranges=ranges)
    assert len(rows) == 2
    assert calls == [0, 1]
