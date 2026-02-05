"""
Lightweight DuckDuckGo-backed search and fetch helper for research mode.

Search is constrained to the allowed domain list; fetching is plain-text with
short timeouts. This module is intentionally simple so it can run in sandboxed
environments and be extended by delivery tasks.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urlencode, urlparse

import requests

from services.research_sources import allowed_domains, allowed_domains_for_query
from services import research_ranker
from services.web_search import is_safe_url


USER_AGENT = "BrandDozerResearchBot/1.0 (+https://example.invalid/research)"


@dataclass
class SearchResult:
    title: str
    url: str


@dataclass
class DocumentPayload:
    url: str
    title: str
    description: str
    content: str
    cached: bool = False
    score: float = 0.0
    metrics: Optional[dict] = None


class DuckDuckGoScraper:
    def __init__(
        self,
        session: Optional[requests.Session] = None,
        base_url: str = "https://duckduckgo.com/html/",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.session = session or requests.Session()
        self.base_url = base_url
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_delay = 0.1
        self._max_delay = 0.5

    def _polite_delay(self) -> None:
        time.sleep(random.uniform(self._min_delay, self._max_delay))

    def search(self, query: str, max_results: int = 10, domains: Optional[Iterable[str]] = None) -> List[SearchResult]:
        params = {"q": query, "kl": "us-en"}
        url = f"{self.base_url}?{urlencode(params)}"
        self._polite_delay()
        resp = self.session.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        resp.raise_for_status()
        html = resp.text
        results: List[SearchResult] = []
        anchor_pattern = r'<a[^>]*class="result__a"[^>]*href="(?P<href>https?://[^"]+)"[^>]*>(?P<title>.*?)</a>'
        for match in re.finditer(anchor_pattern, html, flags=re.I):
            href = match.group("href")
            title = _strip_tags(match.group("title"))
            if not _is_allowed(href, domains):
                continue
            results.append(SearchResult(title=title, url=href))
            if len(results) >= max_results:
                break
        return results

    def search_for_query(self, query: str, max_results: int = 10) -> List[SearchResult]:
        domains = allowed_domains_for_query(query, max_sources=12)
        return self.search(query, max_results=max_results, domains=domains)

    def fetch_text(self, url: str, timeout: int = 15, max_bytes: int = 200_000) -> str:
        if not is_safe_url(url):
            raise ValueError("unsafe url blocked")
        self._polite_delay()
        resp = self.session.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_length = resp.headers.get("Content-Length")
        if content_length and content_length.isdigit() and int(content_length) > max_bytes:
            raise ValueError("content too large")
        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= max_bytes:
                break
        payload = b"".join(chunks)
        return payload.decode(resp.encoding or "utf-8", errors="replace")

    def fetch_document(self, url: str, timeout: int = 15, max_chars: int = 200_000) -> DocumentPayload:
        """
        Fetch a document and return a normalized payload with basic metadata.
        Content is truncated to max_chars to keep artifacts manageable.
        """
        cached = False
        html = None
        cache_path = self._cache_path(url) if self.cache_dir else None
        if cache_path and cache_path.exists():
            try:
                html = cache_path.read_text(encoding="utf-8")
                cached = True
            except Exception:
                html = None
        if html is None:
            self._polite_delay()
            html = self.fetch_text(url, timeout=timeout)
            if cache_path:
                try:
                    cache_path.write_text(html, encoding="utf-8")
                except Exception:
                    pass
        meta = extract_metadata(html, url=url)
        content = _html_to_text(html)
        if max_chars and len(content) > max_chars:
            content = content[:max_chars]
        score, metrics = research_ranker.analyze_and_score(content, url=url)
        return DocumentPayload(
            url=url,
            title=meta.get("title") or "",
            description=meta.get("description") or "",
            content=content,
            cached=cached,
            score=score,
            metrics=metrics,
        )


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _html_to_text(html: str) -> str:
    # Remove scripts/styles and collapse whitespace
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html or "")
    text = re.sub(r"(?s)<[^>]+>", " ", html or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_metadata(html: str, url: str) -> dict:
    """
    Extract lightweight metadata (title/description) from HTML content.
    """
    if not html:
        return {"url": url, "title": "", "description": ""}
    title_match = re.search(r"<title[^>]*>(?P<title>.*?)</title>", html, flags=re.I | re.S)
    meta_desc = re.search(r'<meta[^>]+name=["\\\']description["\\\'][^>]+content=["\\\'](?P<desc>[^"\\\']+)["\\\']', html, flags=re.I)
    og_desc = re.search(r'<meta[^>]+property=["\\\']og:description["\\\'][^>]+content=["\\\'](?P<desc>[^"\\\']+)["\\\']', html, flags=re.I)
    title = _strip_tags(title_match.group("title")) if title_match else ""
    desc = ""
    for candidate in (meta_desc, og_desc):
        if candidate:
            desc = candidate.group("desc").strip()
            break
    return {"url": url, "title": title, "description": desc}


def _is_allowed(url: str, domains: Optional[Iterable[str]]) -> bool:
    if not url:
        return False
    allowed = set(domains) if domains else set(allowed_domains())
    parsed = urlparse(url)
    netloc = f"{parsed.scheme}://{parsed.netloc}"
    return any(netloc.startswith(domain.rstrip("/")) for domain in allowed)


def _hash_url(url: str) -> str:
    import hashlib

    return hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()


def _safe_filename(url: str) -> str:
    return _hash_url(url) + ".html"


def DuckDuckGoScraper__cache_path(self: DuckDuckGoScraper, url: str) -> Optional[Path]:
    if not self.cache_dir:
        return None
    return self.cache_dir / _safe_filename(url)


# Attach as private helper to keep typing intact without altering API surface
DuckDuckGoScraper._cache_path = DuckDuckGoScraper__cache_path
