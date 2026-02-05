from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from services.research_sources import allowed_domains_for_query
from services.web_search import WebSearch, is_safe_url


DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/125.0"


@dataclass
class CrawlPolicy:
    max_depth: int = 2
    max_pages: int = 20
    allowed_domains: List[str] = field(default_factory=list)
    exclude_extensions: Tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".mp4",
        ".mp3",
        ".mov",
        ".avi",
        ".woff",
        ".woff2",
        ".ttf",
    )
    exclude_path_patterns: Tuple[str, ...] = (
        r"/wp-admin",
        r"/wp-content",
        r"/cdn-cgi",
        r"/static/",
        r"/assets/",
        r"/login",
        r"/signup",
    )


class WebResearcher:
    def __init__(self, *, search: Optional[WebSearch] = None, timeout: float = 12.0) -> None:
        self.search = search or WebSearch()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
        self._min_delay = 0.1
        self._max_delay = 0.5

    def _polite_delay(self) -> None:
        time.sleep(random.uniform(self._min_delay, self._max_delay))

    def search_and_fetch(
        self,
        query: str,
        *,
        limit: int = 5,
        max_bytes: int = 200_000,
        domains: Optional[List[str]] = None,
    ) -> List[str]:
        results = self.search.search_domains(query, domains or [], total_limit=limit)
        texts = []
        for result in results:
            try:
                texts.append(self.fetch_text(result.url, max_bytes=max_bytes))
            except Exception:
                continue
        return texts

    def fetch_text(self, url: str, *, max_bytes: int = 200_000) -> str:
        if not is_safe_url(url):
            raise ValueError("unsafe url blocked")
        self._polite_delay()
        content = self.search.fetch_bytes(url, max_bytes=max_bytes)
        html_text = content.decode("utf-8", errors="replace")
        return self._strip_html(html_text)

    def crawl(self, start_url: str, *, policy: Optional[CrawlPolicy] = None) -> List[Tuple[str, str]]:
        policy = policy or CrawlPolicy()
        start_url = self._normalize_url(start_url)
        domain = urlparse(start_url).netloc
        allowed = set(policy.allowed_domains or [domain])
        queue: List[Tuple[str, int]] = [(start_url, 0)]
        visited: Set[str] = set()
        results: List[Tuple[str, str]] = []

        while queue and len(results) < policy.max_pages:
            url, depth = queue.pop(0)
            if url in visited or depth > policy.max_depth:
                continue
            if not self._allowed(url, allowed, policy):
                continue
            visited.add(url)
            try:
                text, links = self._fetch_links(url)
            except Exception:
                continue
            results.append((url, text))
            if depth >= policy.max_depth:
                continue
            for link in links:
                if link not in visited and len(queue) < policy.max_pages:
                    queue.append((link, depth + 1))
        return results

    def _fetch_links(self, url: str) -> Tuple[str, List[str]]:
        if not is_safe_url(url):
            raise ValueError("unsafe url blocked")
        self._polite_delay()
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        encoding = resp.encoding or resp.apparent_encoding or "utf-8"
        html_text = resp.text if resp.encoding else resp.content.decode(encoding, errors="replace")
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        links = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if not href:
                continue
            links.append(self._normalize_url(urljoin(url, href)))
        text = " ".join(soup.get_text(" ").split())
        return text, links

    def _allowed(self, url: str, allowed: Set[str], policy: CrawlPolicy) -> bool:
        if not is_safe_url(url):
            return False
        parsed = urlparse(url)
        if parsed.netloc and parsed.netloc not in allowed:
            return False
        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in policy.exclude_extensions):
            return False
        for pattern in policy.exclude_path_patterns:
            if re.search(pattern, path):
                return False
        return True

    @staticmethod
    def _normalize_url(url: str) -> str:
        return url.split("#", 1)[0]

    @staticmethod
    def _strip_html(html_text: str) -> str:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        return " ".join(soup.get_text(" ").split())


class NCBIClient:
    def __init__(self, *, timeout: float = 10.0) -> None:
        self.base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.session = requests.Session()
        self.timeout = timeout

    def pubmed_summaries(self, query: str, *, retmax: int = 3) -> List[str]:
        term = (query or "").strip()
        if not term:
            return []
        search_params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax}
        search = self.session.get(f"{self.base}/esearch.fcgi", params=search_params, timeout=self.timeout)
        search.raise_for_status()
        search_payload = search.json()
        ids = search_payload.get("esearchresult", {}).get("idlist") or []
        if not ids:
            return []
        summary_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        summary = self.session.get(f"{self.base}/esummary.fcgi", params=summary_params, timeout=self.timeout)
        summary.raise_for_status()
        summary_payload = summary.json().get("result", {})
        summaries: List[str] = []
        for uid in ids:
            entry = summary_payload.get(uid) or {}
            title = str(entry.get("title") or "").strip()
            journal = str(entry.get("source") or "").strip()
            pubdate = str(entry.get("pubdate") or "").strip()
            if not title:
                continue
            line = f"{title} ({journal} {pubdate})"
            summaries.append(line.strip())
        return summaries


class ScholarlyAPIClient:
    def __init__(self, *, timeout: float = 10.0) -> None:
        self.session = requests.Session()
        self.timeout = timeout

    def openalex(self, query: str, *, limit: int = 3) -> List[str]:
        params = {"search": query, "per-page": limit}
        resp = self.session.get("https://api.openalex.org/works", params=params, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results") or []
        lines: List[str] = []
        for entry in results[:limit]:
            title = str(entry.get("display_name") or "").strip()
            year = entry.get("publication_year")
            url = entry.get("id") or ""
            if title:
                suffix = f" ({year})" if year else ""
                lines.append(f"{title}{suffix} {url}".strip())
        return lines

    def crossref(self, query: str, *, limit: int = 3) -> List[str]:
        params = {"query": query, "rows": limit}
        resp = self.session.get("https://api.crossref.org/works", params=params, timeout=self.timeout)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items") or []
        lines: List[str] = []
        for item in items[:limit]:
            title_list = item.get("title") or []
            title = title_list[0] if title_list else ""
            year = None
            issued = item.get("issued", {}).get("date-parts")
            if issued and issued[0]:
                year = issued[0][0]
            doi = item.get("DOI") or ""
            if title:
                suffix = f" ({year})" if year else ""
                url = f"https://doi.org/{doi}" if doi else ""
                lines.append(f"{title}{suffix} {url}".strip())
        return lines

    def datacite(self, query: str, *, limit: int = 3) -> List[str]:
        params = {"query": query, "page[size]": limit}
        resp = self.session.get("https://api.datacite.org/works", params=params, timeout=self.timeout)
        resp.raise_for_status()
        items = resp.json().get("data") or []
        lines: List[str] = []
        for item in items[:limit]:
            attrs = item.get("attributes") or {}
            title_list = attrs.get("titles") or []
            title = title_list[0].get("title") if title_list else ""
            year = attrs.get("publicationYear")
            doi = attrs.get("doi") or ""
            url = f"https://doi.org/{doi}" if doi else ""
            if title:
                suffix = f" ({year})" if year else ""
                lines.append(f"{title}{suffix} {url}".strip())
        return lines

    def semantic_scholar(self, query: str, *, limit: int = 3) -> List[str]:
        params = {"query": query, "limit": limit, "fields": "title,year,url"}
        resp = self.session.get("https://api.semanticscholar.org/graph/v1/paper/search", params=params, timeout=self.timeout)
        resp.raise_for_status()
        items = resp.json().get("data") or []
        lines: List[str] = []
        for item in items[:limit]:
            title = str(item.get("title") or "").strip()
            year = item.get("year")
            url = item.get("url") or ""
            if title:
                suffix = f" ({year})" if year else ""
                lines.append(f"{title}{suffix} {url}".strip())
        return lines


class ReferenceDataClient:
    def __init__(self, *, timeout: float = 10.0) -> None:
        self.session = requests.Session()
        self.timeout = timeout

    def pubchem_formula(self, name: str) -> str:
        if not name:
            return ""
        try:
            resp = self.session.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/MolecularFormula/JSON",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            props = payload.get("PropertyTable", {}).get("Properties") or []
            if not props:
                return ""
            formula = props[0].get("MolecularFormula") or ""
            return str(formula)
        except Exception:
            return ""

    def nist_constant(self, code: str) -> str:
        if not code:
            return ""
        try:
            resp = self.session.get(
                "https://physics.nist.gov/cgi-bin/cuu/Value",
                params={"name": code, "u": "MeV/c^2"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            text = resp.text
            match = re.search(r"Value:\s*([0-9.]+)", text)
            if not match:
                return ""
            return match.group(1)
        except Exception:
            return ""


__all__ = ["WebResearcher", "CrawlPolicy", "NCBIClient", "ScholarlyAPIClient", "ReferenceDataClient"]
