from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import List
from urllib.parse import quote, parse_qs, urlparse, unquote

import requests

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0"
DEFAULT_TIMEOUT = 12.0
MAX_FETCH_BYTES = 200_000
LITE_URL = "https://lite.duckduckgo.com/lite/?q={query}"
HTML_URL = "https://duckduckgo.com/html/?q={query}"


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

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        q = (query or "").strip()
        if not q:
            return []
        # Try lite endpoint first; fall back to html endpoint if no results.
        results: List[SearchResult] = []
        for endpoint in (LITE_URL, HTML_URL):
            url = endpoint.format(query=quote(q))
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            if "lite.duckduckgo" in url:
                results = self._parse_ddg_lite(resp.text)
            else:
                results = self._parse_ddg_html(resp.text)
            if results:
                break
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

    def fetch_text(self, url: str, *, max_bytes: int = MAX_FETCH_BYTES) -> str:
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

    @staticmethod
    def _strip_html(html_text: str) -> str:
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_text)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = html.unescape(text)
        return " ".join(text.split())


__all__ = ["WebSearch", "SearchResult"]
