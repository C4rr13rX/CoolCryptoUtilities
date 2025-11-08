from __future__ import annotations

import time

from datetime import datetime, timedelta, timezone

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
