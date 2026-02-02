from __future__ import annotations

import html
import json
import os
import random
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import List
from urllib.parse import quote, parse_qs, urlparse, unquote

import requests

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0"
DEFAULT_TIMEOUT = 12.0
MAX_FETCH_BYTES = 200_000
LITE_URL = "https://lite.duckduckgo.com/lite/?q={query}"
HTML_URL = "https://duckduckgo.com/html/?q={query}"


def _research_log_enabled() -> bool:
    return os.getenv("C0D3R_RESEARCH_VERBOSE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _research_log(message: str) -> None:
    if not _research_log_enabled():
        return
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] research: {message}"
        print(line, flush=True)
    except Exception:
        pass
    try:
        path = Path("runtime/c0d3r/research_live.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearch:
    """
    Lightweight web search + fetch helper intended for Codex sessions.
    Uses DuckDuckGo Lite (no API key) with a Firefox UA and trims HTML to text.
    """

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self._min_delay = 0.1
        self._max_delay = 0.5
        self._brave_key = (os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY") or "").strip()
        self._last_brave_request = 0.0
        self._brave_limit = int(os.getenv("BRAVE_MAX_SEARCHES_PER_MONTH", "2000") or "2000")
        self._brave_usage_path = Path("runtime/brave_usage.json")

    def _polite_delay(self) -> None:
        time.sleep(random.uniform(self._min_delay, self._max_delay))

    def _brave_delay(self) -> None:
        now = time.time()
        elapsed = now - self._last_brave_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_brave_request = time.time()

    def _brave_allowed(self) -> bool:
        if not self._brave_key:
            return False
        try:
            now = time.localtime()
            month_key = f"{now.tm_year}-{now.tm_mon:02d}"
            usage = self._read_brave_usage()
            count = int(usage.get(month_key, 0))
            return count < self._brave_limit
        except Exception:
            return True

    def _record_brave_usage(self) -> None:
        try:
            now = time.localtime()
            month_key = f"{now.tm_year}-{now.tm_mon:02d}"
            usage = self._read_brave_usage()
            usage[month_key] = int(usage.get(month_key, 0)) + 1
            self._brave_usage_path.parent.mkdir(parents=True, exist_ok=True)
            self._brave_usage_path.write_text(json.dumps(usage, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _read_brave_usage(self) -> dict:
        if not self._brave_usage_path.exists():
            return {}
        try:
            return json.loads(self._brave_usage_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        q = (query or "").strip()
        if not q:
            return []
        # Try lite endpoint first; fall back to html endpoint if no results.
        results: List[SearchResult] = []
        if self._brave_key and self._brave_allowed():
            _research_log(f"brave search: {q}")
            results = self._search_brave(q, limit=limit)
        if results:
            self._record_brave_usage()
            return self._dedupe_results(results, limit)
        for endpoint in (LITE_URL, HTML_URL):
            url = endpoint.format(query=quote(q))
            try:
                _research_log(f"duckduckgo search: {q}")
                self._polite_delay()
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
            except Exception:
                continue
            if "lite.duckduckgo" in url:
                results = self._parse_ddg_lite(resp.text)
            else:
                results = self._parse_ddg_html(resp.text)
            if results:
                break
        if not results:
            results = self._search_wikipedia(q, limit=limit)
        return self._dedupe_results(results, limit)

    def search_domains(
        self,
        query: str,
        domains: List[str],
        *,
        limit_per_domain: int = 2,
        total_limit: int = 8,
    ) -> List[SearchResult]:
        if not domains:
            return self.search(query, limit=total_limit)
        # Fast path: search once, then filter by allowed domains.
        if self._brave_key and self._brave_allowed():
            _research_log(f"brave search (filter domains): {query}")
            results = self.search(query, limit=total_limit * 2)
            filtered: List[SearchResult] = []
            for result in results:
                host = urlparse(result.url).netloc
                for domain in domains:
                    if domain.replace("https://", "").replace("http://", "") in host:
                        filtered.append(result)
                        break
                if len(filtered) >= total_limit:
                    break
            if filtered:
                return self._dedupe_results(filtered, total_limit)
        results: List[SearchResult] = []
        seen = set()
        for domain in domains:
            if len(results) >= total_limit:
                break
            parsed = urlparse(domain if "://" in domain else f"https://{domain}")
            host = parsed.netloc or parsed.path.split("/")[0]
            if not host:
                continue
            site_query = f"site:{host} {query}"
            _research_log(f"site search: {site_query}")
            for result in self.search(site_query, limit=limit_per_domain):
                if result.url in seen:
                    continue
                seen.add(result.url)
                results.append(result)
                if len(results) >= total_limit:
                    break
        if not results:
            return self.search(query, limit=total_limit)
        return results

    def fetch_text(self, url: str, *, max_bytes: int = MAX_FETCH_BYTES) -> str:
        _research_log(f"fetch: {url}")
        self._polite_delay()
        resp = self.session.get(url, timeout=self.timeout, stream=True)
        resp.raise_for_status()
        content = resp.content[:max_bytes]
        encoding = resp.encoding or resp.apparent_encoding or "utf-8"
        text = content.decode(encoding, errors="replace")
        return self._strip_html(text)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_url(url: str) -> str:
        """
        DuckDuckGo HTML results often wrap links as //duckduckgo.com/l/?uddg=<encoded>.
        Normalize to the target URL when possible.
        """
        if not url:
            return ""
        if url.startswith("//"):
            url = "https:" + url
        if "duckduckgo.com/l/?" in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            uddg = params.get("uddg") or []
            if uddg:
                try:
                    return unquote(uddg[0])
                except Exception:
                    return uddg[0]
        return url

    # ------------------------------------------------------------------ internal
    def _parse_ddg_lite(self, html_text: str) -> List[SearchResult]:
        """
        Parse DuckDuckGo Lite HTML results. This is intentionally simple and
        resilient to minor markup changes (no external parser dependency).
        """

        class LiteParser(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.results: list[SearchResult] = []
                self._current: dict[str, str] = {}
                self._in_link = False
                self._in_snippet = False

            def handle_starttag(self, tag, attrs) -> None:
                attr_map = dict(attrs)
                cls = attr_map.get("class", "")
                if tag == "a" and "result-link" in cls:
                    self._current = {"url": attr_map.get("href", ""), "title": "", "snippet": ""}
                    self._in_link = True
                elif tag == "td" and "result-snippet" in cls:
                    self._in_snippet = True

            def handle_endtag(self, tag) -> None:
                if tag == "a" and self._in_link:
                    self._in_link = False
                elif tag == "td" and self._in_snippet:
                    self._in_snippet = False
                    if self._current.get("url"):
                        title = html.unescape(self._current.get("title", "").strip() or "(untitled)")
                        snippet = html.unescape(self._current.get("snippet", "").strip())
                        self.results.append(SearchResult(title=title, url=self._current["url"], snippet=snippet))
                        self._current = {}

            def handle_data(self, data) -> None:
                if self._in_link:
                    self._current["title"] = (self._current.get("title", "") + data).strip()
                elif self._in_snippet:
                    self._current["snippet"] = (self._current.get("snippet", "") + data).strip()

        parser = LiteParser()
        parser.feed(html_text)
        return parser.results

    def _parse_ddg_html(self, html_text: str) -> List[SearchResult]:
        """
        Parse DuckDuckGo standard HTML results (html endpoint).
        """

        class HtmlParser(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.results: list[SearchResult] = []
                self._current: dict[str, str] = {}
                self._in_link = False
                self._in_snippet = False
                self._seen_links = 0

            def handle_starttag(self, tag, attrs) -> None:
                attr_map = dict(attrs)
                cls = attr_map.get("class", "")
                if tag == "a" and ("result__a" in cls or "result-link" in cls):
                    self._current = {"url": attr_map.get("href", ""), "title": "", "snippet": ""}
                    self._in_link = True
                elif tag in {"a", "div"} and ("result__snippet" in cls or "result-snippet" in cls):
                    self._in_snippet = True

            def handle_endtag(self, tag) -> None:
                if tag == "a" and self._in_link:
                    self._in_link = False
                elif tag in {"a", "div"} and self._in_snippet:
                    self._in_snippet = False
                    if self._current.get("url"):
                        title = html.unescape(self._current.get("title", "").strip() or "(untitled)")
                        snippet = html.unescape(self._current.get("snippet", "").strip())
                        self.results.append(SearchResult(title=title, url=self._current["url"], snippet=snippet))
                        self._current = {}

            def handle_data(self, data) -> None:
                if self._in_link:
                    self._current["title"] = (self._current.get("title", "") + data).strip()
                elif self._in_snippet:
                    self._current["snippet"] = (self._current.get("snippet", "") + data).strip()

        parser = HtmlParser()
        parser.feed(html_text)
        return parser.results

    def _search_wikipedia(self, query: str, *, limit: int = 5) -> List[SearchResult]:
        """
        Fallback search using Wikipedia OpenSearch API (no key required).
        """
        try:
            self._polite_delay()
            resp = self.session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": limit,
                    "namespace": 0,
                    "format": "json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            titles = payload[1] if len(payload) > 1 else []
            snippets = payload[2] if len(payload) > 2 else []
            urls = payload[3] if len(payload) > 3 else []
            results: List[SearchResult] = []
            for title, snippet, url in zip(titles, snippets, urls):
                results.append(SearchResult(title=title, url=url, snippet=snippet or ""))
            return results
        except Exception:
            return []

    def _search_brave(self, query: str, *, limit: int = 5) -> List[SearchResult]:
        if not self._brave_key:
            return []
        try:
            self._brave_delay()
            resp = self.session.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": self._brave_key},
                params={"q": query, "count": min(limit, 10)},
                timeout=self.timeout,
            )
            if resp.status_code in {401, 403, 429}:
                return []
            resp.raise_for_status()
            payload = resp.json()
            web = payload.get("web") or {}
            results: List[SearchResult] = []
            for item in (web.get("results") or [])[:limit]:
                title = str(item.get("title") or "").strip()
                url = str(item.get("url") or "").strip()
                snippet = str(item.get("description") or "").strip()
                if url:
                    results.append(SearchResult(title=title or "(untitled)", url=url, snippet=snippet))
            return results
        except Exception:
            return []

    def _dedupe_results(self, results: List[SearchResult], limit: int) -> List[SearchResult]:
        deduped: list[SearchResult] = []
        seen = set()
        for result in results:
            cleaned_url = self._clean_url(result.url)
            if not cleaned_url.startswith(("http://", "https://")):
                continue
            if cleaned_url in seen:
                continue
            seen.add(cleaned_url)
            deduped.append(SearchResult(title=result.title, url=cleaned_url, snippet=result.snippet))
            if len(deduped) >= limit:
                break
        return deduped

    @staticmethod
    def _strip_html(html_text: str) -> str:
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_text)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = html.unescape(text)
        return " ".join(text.split())


__all__ = ["WebSearch", "SearchResult"]
