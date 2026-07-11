from __future__ import annotations

import json
import html as html_lib
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Any


class WebSearch:
    """
    Ethically scrapes search results at a human browsing pace and returns a
    JSON-ready summary for model consumption.

    The AI summarizes raw results; this class handles fetching + rate-limiting.

    Authoritative source prioritization:
      When the query appears scientific or engineering in nature, results
      from authoritative domains (NLM/NCBI, arxiv, IEEE, Nature, etc.)
      are ranked first.  This ensures the equation matrix and unbounded
      solver build on peer-reviewed research rather than pop-science.
    """

    DEFAULT_DELAY_S: float = 1.5   # Minimum seconds between requests.
    DEFAULT_MAX_RESULTS: int = 5

    # Authoritative domains ranked by trust for scientific queries.
    AUTHORITATIVE_DOMAINS: list[tuple[str, int]] = [
        ("ncbi.nlm.nih.gov", 10),   # PubMed, PMC — biology, medicine, chemistry
        ("pubmed.ncbi.nlm.nih.gov", 10),
        ("arxiv.org", 9),            # Physics, math, CS preprints
        ("nature.com", 9),           # Nature journals
        ("science.org", 9),          # Science / AAAS
        ("aps.org", 8),              # American Physical Society
        ("iop.org", 8),              # Institute of Physics
        ("ieee.org", 8),             # IEEE — engineering
        ("springer.com", 7),         # Springer journals
        ("wiley.com", 7),            # Wiley journals
        ("acs.org", 7),              # American Chemical Society
        ("royalsocietypublishing.org", 7),
        ("pnas.org", 7),             # PNAS
        ("sciencedirect.com", 6),    # Elsevier
        ("jstor.org", 6),            # JSTOR
        ("mathworld.wolfram.com", 6),  # Wolfram MathWorld
        ("nist.gov", 6),             # NIST — standards, constants
        ("wolframalpha.com", 5),     # Wolfram Alpha
        ("en.wikipedia.org", 3),     # Wikipedia — useful but lower trust
    ]

    # Keywords that indicate a scientific or engineering query.
    SCIENCE_KEYWORDS: set[str] = {
        "equation", "physics", "quantum", "relativity", "thermodynamic",
        "entropy", "energy", "force", "momentum", "wavelength",
        "frequency", "electromagnetic", "mechanics", "kinetic",
        "potential", "gravity", "gravitational", "acceleration",
        "velocity", "mass", "charge", "field", "wave", "particle",
        "photon", "electron", "proton", "neutron", "nucleus",
        "atom", "molecule", "chemical", "reaction", "catalyst",
        "enzyme", "protein", "dna", "rna", "cell", "biology",
        "organic", "inorganic", "polymer", "crystal",
        "semiconductor", "conductor", "insulator",
        "circuit", "voltage", "current", "resistance",
        "magnetic", "electric", "optical", "laser", "spectrum",
        "calculus", "differential", "integral", "matrix",
        "tensor", "vector", "scalar", "topology",
        "hypothesis", "experiment", "theory", "model",
        "paradox", "anomaly", "constraint", "variable",
        "sympy", "latex", "derivation", "proof",
        "engineering", "aerospace", "structural", "material",
        "fluid", "dynamics", "static", "thermal",
        "neuroscience", "cognitive", "neural", "brain",
        "pharmacology", "drug", "molecular", "genomic",
    }

    def __init__(
        self,
        session: Any,
        *,
        delay_s: float = DEFAULT_DELAY_S,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> None:
        self.session = session
        self.delay_s = delay_s
        self.max_results = max_results
        self._last_request: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, query: str) -> dict:
        """
        Fetch results for `query` and return a summarized dict:
          {query, results: [{title, url, snippet, authority_score}], summary: str}

        If the query is scientific, results are re-ranked to prioritize
        authoritative sources.
        """
        self._rate_limit()
        raw = self._fetch_results(query)

        # Prioritize authoritative sources for scientific queries.
        is_scientific = self._is_scientific_query(query)
        if is_scientific:
            raw = self._rank_by_authority(raw)

        summary = self._summarize(query, raw, scientific=is_scientific)
        return {
            "query": query,
            "results": raw,
            "summary": summary,
            "scientific": is_scientific,
        }

    def search_authoritative(self, query: str, domain_hint: str = "") -> dict:
        """
        Search with forced authoritative source prioritization.

        If `domain_hint` is provided (e.g. 'biology', 'physics'), the
        query is augmented with a site: filter for the most relevant
        authoritative domain.
        """
        augmented = query
        if domain_hint:
            site = self._domain_for_hint(domain_hint)
            if site:
                augmented = f"site:{site} {query}"

        self._rate_limit()
        raw = self._fetch_results(augmented)
        raw = self._rank_by_authority(raw)
        summary = self._summarize(query, raw, scientific=True)
        return {
            "query": query,
            "results": raw,
            "summary": summary,
            "scientific": True,
            "domain_hint": domain_hint,
        }

    # ------------------------------------------------------------------
    # Authority ranking
    # ------------------------------------------------------------------

    def _is_scientific_query(self, query: str) -> bool:
        """Detect if a query is scientific/engineering in nature."""
        tokens = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
        overlap = tokens & self.SCIENCE_KEYWORDS
        return len(overlap) >= 1

    def _rank_by_authority(self, results: list[dict]) -> list[dict]:
        """Re-rank results so authoritative domains appear first."""
        def _score(item: dict) -> int:
            url = item.get("url", "").lower()
            for domain, score in self.AUTHORITATIVE_DOMAINS:
                if domain in url:
                    item["authority_score"] = score
                    return -score  # Negative for descending sort.
            item["authority_score"] = 0
            return 0

        return sorted(results, key=_score)

    def _domain_for_hint(self, hint: str) -> str:
        """Map a domain hint to the best authoritative site."""
        hint = hint.lower().strip()
        mapping = {
            "biology": "ncbi.nlm.nih.gov",
            "medicine": "ncbi.nlm.nih.gov",
            "chemistry": "acs.org",
            "organic chemistry": "acs.org",
            "molecular biology": "ncbi.nlm.nih.gov",
            "pharmacology": "ncbi.nlm.nih.gov",
            "drug": "ncbi.nlm.nih.gov",
            "physics": "arxiv.org",
            "quantum": "arxiv.org",
            "relativity": "arxiv.org",
            "mathematics": "arxiv.org",
            "math": "mathworld.wolfram.com",
            "engineering": "ieee.org",
            "electrical": "ieee.org",
            "materials": "nature.com",
            "neuroscience": "ncbi.nlm.nih.gov",
            "astronomy": "arxiv.org",
            "cosmology": "arxiv.org",
        }
        return mapping.get(hint, "")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)
        self._last_request = time.time()

    def _fetch_results(self, query: str) -> list[dict]:
        """Fetch search results with no-key provider fallbacks.

        DuckDuckGo Lite is kept as the first low-noise source, but it is not
        reliable enough to be the only research dependency. This method now
        falls through to other freely accessible search/content APIs and, as a
        last resort, returns source-search URLs that the model can cite as
        unresolved search targets instead of consuming a blank DDG page.
        """
        if self._is_scientific_query(query):
            providers = [
                self._fetch_duckduckgo_lite,
                self._fetch_wikipedia,
                self._fetch_openalex,
                self._fetch_crossref,
                self._fetch_arxiv,
                self._fetch_pubmed,
                self._fetch_bing_html,
            ]
        else:
            providers = [
                self._fetch_duckduckgo_lite,
                self._fetch_wikipedia,
                self._fetch_openalex,
                self._fetch_crossref,
                self._fetch_bing_html,
            ]

        merged: list[dict] = []
        errors: list[str] = []
        for provider in providers:
            try:
                found = provider(query)
            except Exception as exc:
                errors.append(f"{provider.__name__}: {exc}")
                found = []
            for item in found:
                if self._usable_result(item):
                    merged.append(item)
            merged = self._dedupe_results(merged)
            if len(merged) >= self.max_results:
                return merged[:10]

        if merged:
            return merged[:10]

        fallback = self._fallback_source_queries(query)
        if errors:
            for item in fallback:
                item["search_errors"] = errors[-5:]
        return fallback

    def _fetch_duckduckgo_lite(self, query: str) -> list[dict]:
        """Fetch from DuckDuckGo Lite (no JS, no tracking)."""
        url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(
            url, headers={"User-Agent": "c0d3r/2.0 (ethical-search)"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        if self._looks_like_blank_search_page(html):
            return []
        return self._parse_html(html)

    def _fetch_bing_html(self, query: str) -> list[dict]:
        """Fetch from Bing's public HTML result page as a no-key fallback."""
        url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) c0d3r/2.0",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        if self._looks_like_blank_search_page(html):
            return []
        return self._parse_bing_html(html)

    def _fetch_wikipedia(self, query: str) -> list[dict]:
        """Fetch encyclopedia hits via Wikimedia's no-key search API."""
        search_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": str(min(5, self.max_results)),
            "utf8": "1",
        })
        data = self._json_get(search_url)
        hits = ((data.get("query") or {}).get("search") or []) if isinstance(data, dict) else []
        results: list[dict] = []
        for item in hits:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            snippet = html_lib.unescape(re.sub(r"<[^>]+>", " ", str(item.get("snippet") or ""))).strip()
            results.append({
                "title": f"Wikipedia: {title}",
                "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                "snippet": re.sub(r"\s+", " ", snippet),
                "authority_score": 3,
                "provider": "wikipedia",
            })
        return results

    def _fetch_openalex(self, query: str) -> list[dict]:
        """Fetch scholarly works via OpenAlex's no-key API."""
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode({
            "search": query,
            "per-page": str(min(5, self.max_results)),
        })
        data = self._json_get(url)
        hits = data.get("results") or [] if isinstance(data, dict) else []
        results: list[dict] = []
        for item in hits:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            source = ((item.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
            year = item.get("publication_year") or ""
            doi = item.get("doi") or ""
            landing = item.get("id") or doi or ""
            url_value = doi if str(doi).startswith("http") else str(landing)
            if not url_value:
                continue
            snippet = f"{source} {year}".strip()
            results.append({
                "title": title,
                "url": url_value,
                "snippet": snippet,
                "authority_score": 7,
                "provider": "openalex",
            })
        return results

    def _fetch_crossref(self, query: str) -> list[dict]:
        """Fetch scholarly metadata via Crossref's no-key API."""
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode({
            "query": query,
            "rows": str(min(5, self.max_results)),
        })
        data = self._json_get(url)
        hits = ((data.get("message") or {}).get("items") or []) if isinstance(data, dict) else []
        results: list[dict] = []
        for item in hits:
            title_list = item.get("title") or []
            title = str(title_list[0] if title_list else "").strip()
            doi = str(item.get("DOI") or "").strip()
            if not title or not doi:
                continue
            container = ", ".join(str(x) for x in (item.get("container-title") or [])[:1])
            year_parts = (((item.get("published-print") or item.get("published-online") or {}).get("date-parts") or [[None]])[0])
            year = year_parts[0] if year_parts else ""
            results.append({
                "title": title,
                "url": f"https://doi.org/{doi}",
                "snippet": f"{container} {year}".strip(),
                "authority_score": 7,
                "provider": "crossref",
            })
        return results

    def _fetch_arxiv(self, query: str) -> list[dict]:
        """Fetch scientific preprints via arXiv's no-key Atom API."""
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(min(5, self.max_results)),
        })
        req = urllib.request.Request(url, headers={"User-Agent": "c0d3r/2.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml = resp.read().decode("utf-8", errors="ignore")
        entries = re.findall(r"<entry>(.*?)</entry>", xml, flags=re.DOTALL | re.IGNORECASE)
        results: list[dict] = []
        for entry in entries:
            title = self._xml_text(entry, "title")
            summary = self._xml_text(entry, "summary")
            link_match = re.search(r'<link[^>]+href=["\']([^"\']+)["\'][^>]*>', entry, flags=re.IGNORECASE)
            url_value = html_lib.unescape(link_match.group(1)) if link_match else ""
            if title and url_value:
                results.append({
                    "title": f"arXiv preprint: {title}",
                    "url": url_value,
                    "snippet": re.sub(r"\s+", " ", summary)[:300],
                    "authority_score": 9,
                    "provider": "arxiv",
                })
        return results

    def _fetch_pubmed(self, query: str) -> list[dict]:
        """Fetch biomedical citations via NCBI E-utilities."""
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode({
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(min(5, self.max_results)),
        })
        data = self._json_get(search_url)
        ids = (((data.get("esearchresult") or {}).get("idlist") or []) if isinstance(data, dict) else [])
        if not ids:
            return []
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        })
        summary = self._json_get(summary_url)
        result_map = (summary.get("result") or {}) if isinstance(summary, dict) else {}
        results: list[dict] = []
        for pmid in ids:
            item = result_map.get(str(pmid)) or {}
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            journal = str(item.get("fulljournalname") or item.get("source") or "").strip()
            pubdate = str(item.get("pubdate") or "").strip()
            results.append({
                "title": f"PubMed: {title}",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "snippet": f"{journal} {pubdate}".strip(),
                "authority_score": 10,
                "provider": "pubmed",
            })
        return results

    @staticmethod
    def _parse_bing_html(html: str) -> list[dict]:
        results: list[dict] = []
        block_re = re.compile(r'<li[^>]+class=["\']b_algo["\'][^>]*>(.*?)</li>', re.I | re.S)
        for block in block_re.findall(html):
            link = re.search(r'<h2[^>]*>\s*<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', block, re.I | re.S)
            if not link:
                continue
            url_value = html_lib.unescape(link.group(1).strip())
            url_value = WebSearch._decode_bing_url(url_value)
            title = html_lib.unescape(re.sub(r"<[^>]+>", " ", link.group(2))).strip()
            snippet_match = re.search(r'<p[^>]*>(.*?)</p>', block, re.I | re.S)
            snippet = html_lib.unescape(re.sub(r"<[^>]+>", " ", snippet_match.group(1))).strip() if snippet_match else ""
            if url_value.startswith("http") and title:
                results.append({
                    "title": re.sub(r"\s+", " ", title),
                    "url": url_value,
                    "snippet": re.sub(r"\s+", " ", snippet),
                    "authority_score": 0,
                    "provider": "bing",
                })
            if len(results) >= 10:
                break
        return results

    @staticmethod
    def _json_get(url: str) -> Any:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "c0d3r/2.0 research fallback (contact: local)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    @staticmethod
    def _xml_text(xml: str, tag: str) -> str:
        match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, flags=re.I | re.S)
        if not match:
            return ""
        return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))).strip()

    @staticmethod
    def _usable_result(item: dict) -> bool:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        if not title or not url.startswith("http"):
            return False
        lowered = (title + " " + url).lower()
        bad_markers = ("403 forbidden", "captcha", "robot check", "enable javascript")
        if any(marker in lowered for marker in bad_markers):
            return False
        if title.lower() in {"here", "html", "duckduckgo", "bing"}:
            return False
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path in {"", "/"}:
            return False
        if parsed.netloc.endswith("bing.com") and parsed.path in {"", "/search"}:
            return False
        return True

    @staticmethod
    def _decode_bing_url(url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc.endswith("bing.com") and parsed.path.startswith("/ck/"):
            params = urllib.parse.parse_qs(parsed.query)
            encoded = params.get("u", [""])[0]
            if encoded.startswith("a1"):
                encoded = encoded[2:]
            if encoded:
                try:
                    import base64
                    padded = encoded + "=" * (-len(encoded) % 4)
                    decoded = base64.urlsafe_b64decode(padded).decode("utf-8", errors="ignore")
                    if decoded.startswith("http"):
                        return decoded
                except Exception:
                    pass
        return url

    @staticmethod
    def _looks_like_blank_search_page(html: str) -> bool:
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip().lower()
        if len(text) < 200:
            return True
        bad_markers = (
            "403 forbidden", "access denied", "unusual traffic", "captcha",
            "verify you are human", "no results found for",
        )
        return any(marker in text for marker in bad_markers)

    @staticmethod
    def _dedupe_results(results: list[dict]) -> list[dict]:
        seen: set[str] = set()
        deduped: list[dict] = []
        for item in results:
            url = re.sub(r"[?#].*$", "", str(item.get("url") or "").strip().lower())
            title = re.sub(r"\s+", " ", str(item.get("title") or "").strip().lower())
            key = url or title
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _fallback_source_queries(self, query: str) -> list[dict]:
        encoded = urllib.parse.quote_plus(query)
        targets = [
            ("Wikipedia search", f"https://en.wikipedia.org/w/index.php?search={encoded}", 2),
            ("OpenAlex works search", f"https://api.openalex.org/works?search={encoded}", 6),
            ("Crossref works search", f"https://api.crossref.org/works?query={encoded}", 6),
        ]
        if self._is_scientific_query(query):
            targets.extend([
                ("arXiv search", f"https://export.arxiv.org/api/query?search_query=all:{encoded}", 8),
                ("PubMed search", f"https://pubmed.ncbi.nlm.nih.gov/?term={encoded}", 9),
            ])
        targets.append(("Bing search", f"https://www.bing.com/search?q={encoded}", 1))
        return [
            {
                "title": f"{label}: {query}",
                "url": url,
                "snippet": "Fallback source-search URL generated because no parseable result rows were returned by no-key providers.",
                "authority_score": score,
                "provider": "fallback-source-query",
            }
            for label, url, score in targets[: self.max_results]
        ]

    @staticmethod
    def _parse_html(html: str) -> list[dict]:
        """Minimal regex extraction — no external dependencies."""
        results: list[dict] = []
        link_re = re.compile(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*class=["\']result-link["\'][^>]*>(.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippet_re = re.compile(
            r"<td[^>]+class=['\"]result-snippet['\"][^>]*>(.*?)</td>",
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippets = [
            html_lib.unescape(re.sub(r"<[^>]+>", " ", item)).strip()
            for item in snippet_re.findall(html)
        ]
        snippets = [re.sub(r"\s+", " ", item) for item in snippets]
        for index, m in enumerate(link_re.finditer(html)):
            raw_url, raw_title = m.group(1).strip(), m.group(2).strip()
            raw_url = html_lib.unescape(raw_url)
            title = html_lib.unescape(re.sub(r"<[^>]+>", " ", raw_title)).strip()
            title = re.sub(r"\s+", " ", title)
            url = raw_url
            if raw_url.startswith("//duckduckgo.com/l/") or raw_url.startswith("https://duckduckgo.com/l/"):
                parsed = urllib.parse.urlparse("https:" + raw_url if raw_url.startswith("//") else raw_url)
                query = urllib.parse.parse_qs(parsed.query)
                uddg = query.get("uddg", [""])[0]
                if uddg:
                    url = uddg
            elif raw_url.startswith("//"):
                url = "https:" + raw_url
            if url.startswith("http") and title:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippets[index] if index < len(snippets) else "",
                    "authority_score": 0,
                })
            if len(results) >= 10:
                break
        if results:
            return results

        # Fallback for less-specific result markup.
        for m in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, flags=re.IGNORECASE | re.DOTALL):
            raw_url, raw_title = html_lib.unescape(m.group(1).strip()), m.group(2).strip()
            title = html_lib.unescape(re.sub(r"<[^>]+>", " ", raw_title)).strip()
            title = re.sub(r"\s+", " ", title)
            if "uddg=" in raw_url:
                parsed = urllib.parse.urlparse("https:" + raw_url if raw_url.startswith("//") else raw_url)
                raw_url = urllib.parse.parse_qs(parsed.query).get("uddg", [raw_url])[0]
            if raw_url.startswith("//"):
                raw_url = "https:" + raw_url
            if raw_url.startswith("http") and title and "DuckDuckGo" not in title:
                results.append({"title": title, "url": raw_url, "snippet": "", "authority_score": 0})
            if len(results) >= 10:
                break
        return results

    def _summarize(
        self, query: str, results: list[dict], *, scientific: bool = False,
    ) -> str:
        """Ask the model to condense results into key points."""
        if not results:
            return ""
        if os.getenv("C0D3R_WEB_SEARCH_MODEL_SUMMARY", "").lower() not in {"1", "true", "yes", "on"}:
            bullets: list[str] = []
            for item in results[: self.max_results]:
                title = str(item.get("title") or "").strip()
                url = str(item.get("url") or "").strip()
                snippet = re.sub(r"\s+", " ", str(item.get("snippet") or "").strip())
                authority = item.get("authority_score")
                authority_text = f"; authority score {authority}" if authority else ""
                if snippet:
                    snippet = snippet[:260].rstrip()
                    bullets.append(f"- {title}: {snippet} ({url}{authority_text})")
                else:
                    bullets.append(f"- {title} ({url}{authority_text})")
            return "\n".join(bullets)
        if not self.session:
            return ""
        blob = json.dumps(results[: self.max_results], indent=2)
        if scientific:
            prompt = (
                f"Query: {query}\n\nSearch results (JSON):\n{blob}\n\n"
                "This is a scientific/engineering query.  Summarize the key "
                "findings in 3-5 bullet points.  Focus on:\n"
                "- Precise equations, constants, or mathematical relationships\n"
                "- Measurable quantities and their units\n"
                "- Connections between disciplines\n"
                "- Known paradoxes or anomalies mentioned\n"
                "Prioritize information from peer-reviewed or authoritative sources."
            )
        else:
            prompt = (
                f"Query: {query}\n\nSearch results (JSON):\n{blob}\n\n"
                "Summarize the key points in 3-5 bullet points for use as model context."
            )
        try:
            return self.session.send(prompt=prompt, stream=False) or ""
        except Exception:
            return ""
