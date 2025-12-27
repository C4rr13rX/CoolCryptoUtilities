from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from datetime import datetime

import feedparser

try:  # feedparser 6.x exposes util.parse_date, older builds do not
    from feedparser.util import parse_date as _fp_parse_date  # type: ignore
except Exception:  # pragma: no cover - fallback path
    _fp_parse_date = None
import pandas as pd
from bs4 import BeautifulSoup


@dataclass
class NewsSource:
    name: str
    url: str
    topics: Sequence[str] = ()


DEFAULT_SOURCES: Sequence[NewsSource] = (
    NewsSource(name="CoinDesk", url="https://www.coindesk.com/arc/outboundfeeds/rss/", topics=("BTC", "ETH")),
    NewsSource(name="CoinTelegraph", url="https://cointelegraph.com/rss", topics=("L2", "DEFI")),
    NewsSource(name="Blockworks", url="https://blockworks.co/feed", topics=("MACRO", "MARKETS")),
    NewsSource(name="Ethereum Foundation", url="https://blog.ethereum.org/en/feed.xml", topics=("ETH", "STAKING", "L2")),
    NewsSource(name="US Treasury Press", url="https://home.treasury.gov/news/press-releases/rss", topics=("USD", "POLICY")),
    NewsSource(name="IMF Fintech", url="https://www.imf.org/external/pubs/ft/fandd/fintech/rss.xml", topics=("MACRO", "REGULATION")),
    NewsSource(name="GitHub Security Advisories", url="https://github.com/security-advisories.atom", topics=("SECURITY", "DEVOPS")),
    NewsSource(name="Decrypt", url="https://decrypt.co/feed", topics=("NFT", "GAMEFI", "DEFI")),
    NewsSource(name="The Defiant", url="https://thedefiant.io/feed", topics=("DEFI", "STABLECOIN")),
    NewsSource(name="ECB Press", url="https://www.ecb.europa.eu/rss/press.html", topics=("EURO", "MACRO")),
    NewsSource(name="BIS Press", url="https://www.bis.org/rss/press.xml", topics=("CBDC", "CENTRALBANK")),
    NewsSource(name="Federal Reserve Press", url="https://www.federalreserve.gov/feeds/press_all.xml", topics=("USD", "POLICY")),
    NewsSource(name="World Bank Blogs", url="https://blogs.worldbank.org/feed", topics=("MACRO", "DEVELOPMENT")),
    NewsSource(name="Financial Stability Board", url="https://www.fsb.org/feed/press/", topics=("REGULATION", "MACRO")),
    NewsSource(name="Chainalysis Insights", url="https://blog.chainalysis.com/feed", topics=("RISK", "ONCHAIN")),
)


class EthicalNewsIngestor:
    """Collects free/ethical crypto headlines and stores them locally for training."""

    def __init__(
        self,
        *,
        sources: Optional[Sequence[NewsSource]] = None,
        output_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        fetcher: Optional[Callable[[NewsSource], List[dict]]] = None,
        max_tokens: int = 12,
    ) -> None:
        base_sources = list(sources or DEFAULT_SOURCES)
        catalog_path = Path(os.getenv("ETHICAL_NEWS_SOURCES_PATH", "config/ethical_news_sources.json"))
        custom_sources = self._load_custom_sources(catalog_path)
        merged: Dict[str, NewsSource] = {source.name: source for source in base_sources}
        for source in custom_sources:
            merged[source.name] = source
        self.sources = tuple(merged.values())
        root_cache = cache_dir or Path(os.getenv("ETHICAL_NEWS_CACHE", "data/news/cache"))
        self.cache_dir = root_cache.expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = (output_path or Path(os.getenv("ETHICAL_NEWS_PATH", "data/news/ethical_news.parquet"))).expanduser()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_root = Path(os.getenv("ETHICAL_NEWS_ARCHIVE_DIR", "data/news/free_news")).expanduser()
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self._fetcher = fetcher or self._fetch_source
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def harvest(
        self,
        *,
        tokens: Iterable[str],
        start_ts: int,
        end_ts: int,
    ) -> List[dict]:
        keyword_set = {token.lower() for token in tokens if token}
        rows: List[dict] = []
        for source in self.sources:
            source_keywords = keyword_set | {topic.lower() for topic in getattr(source, "topics", []) if topic}
            entries = self._pull_entries(source)
            for entry in entries:
                ts = self._entry_timestamp(entry)
                if ts is None or ts < start_ts or ts > end_ts:
                    continue
                title = self._clean_text(entry.get("title"))
                summary = self._clean_text(entry.get("summary") or entry.get("description") or "")
                body = f"{title}\n{summary}".strip()
                if not body:
                    continue
                article_tokens = self._extract_tokens(body)
                if source_keywords and not (article_tokens & source_keywords):
                    continue
                selected_tokens = self._select_tokens(article_tokens, source_keywords)
                if not selected_tokens:
                    continue
                rows.append(
                    {
                        "timestamp": int(ts),
                        "headline": title[:256] or summary[:256],
                        "article": summary or title,
                        "sentiment": "neutral",
                        "tokens": sorted(selected_tokens),
                        "source": source.name,
                        "url": entry.get("link"),
                    }
                )
        if rows:
            self._write_parquet(rows)
            self._archive_rows(rows, start_ts, end_ts, keyword_set)
            return rows
        archived = self._load_archive_rows(start_ts, end_ts, keyword_set)
        if archived:
            self._write_parquet(archived)
        return archived

    def harvest_windows(
        self,
        *,
        tokens: Iterable[str],
        ranges: Sequence[Tuple[int, int]],
    ) -> List[dict]:
        rows: List[dict] = []
        for start_ts, end_ts in ranges:
            start_int = int(min(start_ts, end_ts))
            end_int = int(max(start_ts, end_ts))
            rows.extend(self.harvest(tokens=tokens, start_ts=start_int, end_ts=end_int))
        return rows

    def harvest_window(
        self,
        *,
        tokens: Iterable[str],
        start: datetime,
        end: datetime,
    ) -> List[dict]:
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        return self.harvest(tokens=tokens, start_ts=start_ts, end_ts=end_ts)

    def harvest_schedule_file(
        self,
        schedule_path: Path,
        *,
        fallback_tokens: Optional[Iterable[str]] = None,
    ) -> int:
        path = schedule_path.expanduser()
        if not path.exists():
            return 0
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        collected = 0
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            token_payload = entry.get("tokens") or fallback_tokens
            if not token_payload:
                continue
            start_ts = self._coerce_schedule_ts(entry.get("start_ts") or entry.get("start"))
            end_ts = self._coerce_schedule_ts(entry.get("end_ts") or entry.get("end"))
            if start_ts is None or end_ts is None:
                continue
            try:
                rows = self.harvest(tokens=token_payload, start_ts=start_ts, end_ts=end_ts)
                collected += len(rows)
            except Exception:
                continue
        return collected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pull_entries(self, source: NewsSource) -> List[dict]:
        try:
            entries = self._fetcher(source)
            if entries:
                self._persist_cache(source, entries)
                return entries
        except Exception:
            entries = None
        cached = self._load_cache(source)
        return cached or []

    def _fetch_source(self, source: NewsSource) -> List[dict]:
        parsed = feedparser.parse(source.url)
        if getattr(parsed, "bozo", False):
            raise RuntimeError(getattr(parsed, "bozo_exception", "invalid feed"))
        entries: List[dict] = []
        for entry in parsed.entries:
            entries.append(self._normalize_entry(entry))
        return entries

    def _normalize_entry(self, entry: Any) -> dict:
        summary = entry.get("summary") if isinstance(entry, dict) else getattr(entry, "summary", "")
        data = {
            "title": entry.get("title") if isinstance(entry, dict) else getattr(entry, "title", ""),
            "summary": summary,
            "description": entry.get("description") if isinstance(entry, dict) else getattr(entry, "description", ""),
            "link": entry.get("link") if isinstance(entry, dict) else getattr(entry, "link", ""),
            "published": entry.get("published") if isinstance(entry, dict) else getattr(entry, "published", ""),
        }
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            data["published_parsed"] = time.mktime(entry.published_parsed)
        elif isinstance(entry, dict) and isinstance(entry.get("published_parsed"), (tuple, list)):
            try:
                data["published_parsed"] = time.mktime(entry["published_parsed"])  # type: ignore[index]
            except Exception:
                pass
        return data

    def _entry_timestamp(self, entry: dict) -> Optional[int]:
        ts = entry.get("published_parsed")
        if isinstance(ts, (int, float)):
            return int(ts)
        published = entry.get("published")
        if isinstance(published, (int, float)):
            return int(published)
        if published:
            try:
                if _fp_parse_date:
                    parsed = _fp_parse_date(published)
                else:
                    parsed = feedparser._parse_date(published)  # type: ignore[attr-defined]
                if parsed:
                    return int(time.mktime(parsed))
            except Exception:
                return None
        return None

    def _coerce_schedule_ts(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except Exception:
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    return int(parsed.timestamp())
                except Exception:
                    return None
        return None

    def _clean_text(self, text: Any) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        soup = BeautifulSoup(raw, "html.parser")
        clean = soup.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", clean).strip()

    def _extract_tokens(self, text: str) -> Set[str]:
        tokens = set(re.findall(r"[A-Za-z0-9]{3,}", text.lower()))
        return tokens

    def _select_tokens(self, article_tokens: Set[str], keyword_set: Set[str]) -> Set[str]:
        if keyword_set:
            matched = article_tokens & keyword_set
        else:
            matched = article_tokens
        trimmed = sorted(matched)[: self.max_tokens]
        return {token.upper() for token in trimmed}

    def _write_parquet(self, rows: List[dict]) -> None:
        df_new = pd.DataFrame(rows)
        if self.output_path.exists():
            try:
                df_old = pd.read_parquet(self.output_path)
            except Exception:
                df_old = pd.DataFrame()
            df = pd.concat([df_old, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp", "headline", "source"])
        else:
            df = df_new
        try:
            df.to_parquet(self.output_path, index=False)
        except Exception:
            pass

    def _archive_rows(self, rows: List[dict], start_ts: int, end_ts: int, tokens: Iterable[str]) -> None:
        if not rows:
            return
        payload = {
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "tokens": sorted({str(token).upper() for token in tokens if token}),
            "count": len(rows),
            "articles": rows,
        }
        filename = f"{int(start_ts)}_{int(end_ts)}.json"
        path = self.archive_root / filename
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_archive_rows(self, start_ts: int, end_ts: int, tokens: Set[str]) -> List[dict]:
        if not self.archive_root.exists():
            return []
        wanted = {token.upper() for token in tokens if token}
        rows: List[dict] = []
        seen: Set[Tuple[int, str, str]] = set()
        for path in self.archive_root.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            try:
                payload_start = int(payload.get("start_ts", 0))
                payload_end = int(payload.get("end_ts", 0))
            except Exception:
                continue
            if payload_end < start_ts or payload_start > end_ts:
                continue
            payload_tokens = {str(token).upper() for token in payload.get("tokens", []) if token}
            if wanted and payload_tokens and not (wanted & payload_tokens):
                continue
            for article in payload.get("articles", []) or []:
                if not isinstance(article, dict):
                    continue
                try:
                    ts = int(article.get("timestamp", 0))
                except Exception:
                    continue
                if ts < start_ts or ts > end_ts:
                    continue
                article_tokens = {str(token).upper() for token in article.get("tokens", []) if token}
                if wanted and article_tokens and not (wanted & article_tokens):
                    continue
                key = (ts, str(article.get("headline", "")), str(article.get("source", "")))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(article)
        return rows

    def _load_custom_sources(self, path: Path) -> List[NewsSource]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        sources: List[NewsSource] = []
        if isinstance(payload, list):
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                url = str(entry.get("url") or "").strip()
                topics = entry.get("topics") or ()
                if not name or not url:
                    continue
                sources.append(NewsSource(name=name, url=url, topics=tuple(topics)))
        return sources

    def _cache_file(self, source: NewsSource) -> Path:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", source.name.lower()).strip("-") or "source"
        return self.cache_dir / f"{slug}.json"

    def _persist_cache(self, source: NewsSource, entries: Iterable[dict]) -> None:
        path = self._cache_file(source)
        try:
            path.write_text(json.dumps(list(entries)), encoding="utf-8")
        except Exception:
            path.unlink(missing_ok=True)

    def _load_cache(self, source: NewsSource) -> Optional[List[dict]]:
        path = self._cache_file(source)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            path.unlink(missing_ok=True)
            return None
        if not isinstance(payload, list):
            return None
        return [entry for entry in payload if isinstance(entry, dict)]
