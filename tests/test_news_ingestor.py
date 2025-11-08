from __future__ import annotations

import time

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
