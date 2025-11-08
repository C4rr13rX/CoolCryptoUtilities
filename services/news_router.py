from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

DEFAULT_CACHE = Path("data/news/free_news.parquet")


class FreeNewsRouter:
    """
    Loads additional free/ethical news snapshots from locally curated JSON or
    Parquet files so the training pipeline can request arbitrary windows.
    """

    def __init__(
        self,
        *,
        config_path: str | Path = "config/free_news_sources.json",
        cache_path: str | Path = DEFAULT_CACHE,
    ) -> None:
        self.config_path = Path(config_path)
        self.cache_path = Path(cache_path)
        self.sources = self._load_sources()

    def window(
        self,
        *,
        start_ts: int,
        end_ts: Optional[int] = None,
        tokens: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        end_ts = end_ts or int(time.time())
        rows = self._load_rows()
        if not rows:
            return pd.DataFrame(columns=["timestamp", "headline", "article", "sentiment", "tokens"])
        df = pd.DataFrame(rows)
        df = df[df["timestamp"].between(start_ts, end_ts)]
        if tokens:
            want = {tok.upper() for tok in tokens if tok}
            if want:
                df = df[df["tokens"].map(lambda values: bool(set(values) & want))]
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def refresh_cache(self) -> None:
        df = self.window(start_ts=0, end_ts=int(time.time()))
        if df.empty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.cache_path, index=False)

    def _load_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for source in self.sources:
            entries = self._read_source(source)
            rows.extend(entries)
        if self.cache_path.exists():
            try:
                cached = pd.read_parquet(self.cache_path)
                if not cached.empty:
                    rows.extend(cached.to_dict(orient="records"))
            except Exception:
                pass
        return rows

    def _load_sources(self) -> List[Dict[str, str]]:
        if not self.config_path.exists():
            default = Path("docs/free_news_samples.json")
            if default.exists():
                return [{"path": str(default)}]
            return []
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        sources = payload.get("sources") if isinstance(payload, dict) else None
        cleaned: List[Dict[str, str]] = []
        if isinstance(sources, list):
            for entry in sources:
                path = (entry or {}).get("path")
                if not path:
                    continue
                cleaned.append({"path": str(path)})
        return cleaned

    def _read_source(self, source: Dict[str, str]) -> List[Dict[str, object]]:
        path = Path(source.get("path", "")).expanduser()
        if not path.exists():
            return []
        if path.suffix.lower() == ".parquet":
            try:
                df = pd.read_parquet(path)
            except Exception:
                return []
            return df.to_dict(orient="records")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(payload, list):
            rows = []
            for entry in payload:
                rows.append(self._normalise_entry(entry))
            return [row for row in rows if row]
        if isinstance(payload, dict) and "articles" in payload:
            rows = []
            for entry in payload.get("articles", []):
                rows.append(self._normalise_entry(entry))
            return [row for row in rows if row]
        return []

    def _normalise_entry(self, entry: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not isinstance(entry, dict):
            return None
        timestamp = int(entry.get("timestamp") or entry.get("ts") or 0)
        headline = str(entry.get("headline") or "").strip()
        article = str(entry.get("article") or "").strip()
        sentiment = str(entry.get("sentiment") or "neutral").strip()
        tokens = entry.get("tokens") or entry.get("tickers")
        if isinstance(tokens, str):
            tokens = [tokens]
        if not tokens:
            tokens = headline.upper().split()
        token_list = sorted({tok.upper() for tok in tokens if isinstance(tok, str) and tok.strip()})
        if timestamp <= 0 or not article:
            return None
        return {
            "timestamp": timestamp,
            "headline": headline or article[:120],
            "article": article,
            "sentiment": sentiment or "neutral",
            "tokens": token_list,
        }


__all__ = ["FreeNewsRouter"]
