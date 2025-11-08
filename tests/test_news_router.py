from __future__ import annotations

import json
from pathlib import Path

from services.news_router import FreeNewsRouter


def test_free_news_router_window(tmp_path: Path) -> None:
    sample = tmp_path / "news.json"
    sample.write_text(
        json.dumps(
            [
                {
                    "timestamp": 1_700_000_000,
                    "headline": "Sample",
                    "article": "Sample ethical news snippet",
                    "sentiment": "neutral",
                    "tokens": ["ETH"],
                }
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"sources": [{"path": str(sample)}]}), encoding="utf-8")
    cache = tmp_path / "cache.parquet"
    router = FreeNewsRouter(config_path=config, cache_path=cache)
    window = router.window(start_ts=1_600_000_000, end_ts=1_800_000_000, tokens=["ETH"])
    assert not window.empty
    router.refresh_cache()
    assert cache.exists()
