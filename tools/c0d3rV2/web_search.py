from __future__ import annotations

import json
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
        """Fetch from DuckDuckGo Lite (no JS, no tracking)."""
        url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "c0d3r/2.0 (ethical-search)"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            return self._parse_html(html)
        except Exception:
            return []

    @staticmethod
    def _parse_html(html: str) -> list[dict]:
        """Minimal regex extraction — no external dependencies."""
        results: list[dict] = []
        for m in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)<', html):
            url, title = m.group(1).strip(), m.group(2).strip()
            if url.startswith("http") and title:
                results.append({
                    "title": title, "url": url, "snippet": "",
                    "authority_score": 0,
                })
            if len(results) >= 10:
                break
        return results

    def _summarize(
        self, query: str, results: list[dict], *, scientific: bool = False,
    ) -> str:
        """Ask the model to condense results into key points."""
        if not results or not self.session:
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
