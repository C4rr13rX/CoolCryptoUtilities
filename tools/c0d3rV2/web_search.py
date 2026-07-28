from __future__ import annotations

import json
import html as html_lib
import hashlib
import ipaddress
import os
import re
import socket
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
        ("api.openalex.org", 7),     # Scholarly identity + indexed abstracts
        ("openstax.org", 7),         # Openly licensed academic textbooks
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
        ("nasa.gov", 9),             # NASA technical and scientific material
        ("developer.mozilla.org", 8), # Web platform reference
        ("typescriptlang.org", 9),    # TypeScript language documentation
        ("threejs.org", 9),           # Three.js API/manual/source documentation
        ("docs.python.org", 9),       # Python language/library documentation
        ("docs.djangoproject.com", 9),# Django documentation
        ("doc.rust-lang.org", 9),     # Rust language documentation
        ("en.cppreference.com", 8),   # C/C++ language and library reference
        ("angular.dev", 9),           # Angular documentation
        ("react.dev", 9),             # React documentation
        ("w3.org", 9),                # Web standards
        ("rfc-editor.org", 9),        # Internet standards
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
            raw = self._rank_by_authority(raw, query)

        summary = self._summarize(query, raw, scientific=is_scientific)
        return {
            "query": query,
            "results": raw,
            "summary": summary,
            "scientific": is_scientific,
        }

    def discover(self, query: str) -> list[dict]:
        """Return multi-provider discovery metadata without a model summary."""
        self._rate_limit()
        raw = self._fetch_results(query)
        return self._rank_by_authority(raw, query) if self._is_scientific_query(query) else raw

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
        raw = self._rank_by_authority(raw, query)
        summary = self._summarize(query, raw, scientific=True)
        return {
            "query": query,
            "results": raw,
            "summary": summary,
            "scientific": True,
            "domain_hint": domain_hint,
        }

    def fetch_evidence(self, url: str, query: str, *, max_bytes: int = 524288) -> dict:
        """Fetch and relevance-check a discovered source without trusting metadata.

        Only public HTTP(S) hosts are allowed.  The response body is bounded,
        executable markup is removed, and a passage is selected around query
        terms.  Callers can therefore distinguish discovery metadata from
        content that was actually retrieved and inspected.
        """
        parsed = urllib.parse.urlparse(str(url))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return {"status": "rejected_url", "error": "Only public HTTP(S) URLs are allowed"}
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
            if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
                return {"status": "rejected_url", "error": "Host resolves to a non-public address"}
        except Exception as exc:
            return {"status": "fetch_failed", "error": f"DNS validation failed: {exc}"}
        request = urllib.request.Request(str(url), headers={
            "User-Agent": "c0d3r/2.0 archival-verifier",
            "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.2",
        })
        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                final_url = response.geturl()
                final_host = urllib.parse.urlparse(final_url).hostname or ""
                final_addresses = {item[4][0] for item in socket.getaddrinfo(final_host, 443)}
                if not final_addresses or any(not ipaddress.ip_address(address).is_global for address in final_addresses):
                    return {"status": "rejected_redirect", "error": "Redirect resolved to a non-public address"}
                content_type = str(response.headers.get("Content-Type") or "").lower()
                raw = response.read(max_bytes + 1)
                truncated = len(raw) > max_bytes
                raw = raw[:max_bytes]
        except Exception as exc:
            return {"status": "fetch_failed", "error": str(exc)}
        if not any(kind in content_type for kind in ("text/", "html", "json", "xml")):
            return {"status": "unsupported_content_type", "content_type": content_type, "final_url": final_url}
        decoded = raw.decode("utf-8", errors="ignore")
        evidence_level = "full_text_or_page"
        if "json" in content_type and "api.openalex.org" in (urllib.parse.urlparse(final_url).hostname or ""):
            try:
                payload = json.loads(decoded)
                inverted = payload.get("abstract_inverted_index") or {}
                positions = sorted((position, word) for word, indexes in inverted.items() for position in indexes)
                abstract = " ".join(word for _, word in positions)
                identity = " ".join(str(payload.get(key) or "") for key in ("title", "publication_year", "doi"))
                decoded = f"{identity} Abstract: {abstract}"
                evidence_level = "indexed_abstract"
            except Exception:
                pass
        decoded = re.sub(r"<(script|style|noscript)\b[^>]*>.*?</\1>", " ", decoded, flags=re.I | re.S)
        text = html_lib.unescape(re.sub(r"<[^>]+>", " ", decoded))
        text = re.sub(r"\s+", " ", text).strip()
        stop = {"the","and","for","with","from","that","this","what","when","where","using","result",
                "authoritative","source","archival","experiment","initially","does","into","near","how","far"}
        query_tokens = re.findall(r"[a-z0-9.]+", query.lower())
        terms = list(dict.fromkeys(token for token in query_tokens
                                   if len(token) >= 3 and token not in stop))
        significant = [token for token in query_tokens if len(token) >= 3 and token not in stop and not re.fullmatch(r"\d+(?:\.\d+)?", token)]
        phrases = list(dict.fromkeys(f"{a} {b}" for a, b in zip(significant, significant[1:])))
        lower = text.lower()
        matched = [term for term in terms if re.search(rf"\b{re.escape(term)}\b", lower)]
        matched_phrases = [phrase for phrase in phrases if phrase in lower]
        query_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", query))
        matched_numbers = sorted(number for number in query_numbers if re.search(rf"\b{re.escape(number)}\b", lower))
        coverage = len(matched) / max(1, len(terms))
        anchors = [lower.find(term) for term in matched if lower.find(term) >= 0]
        abstract_at = lower.find("abstract")
        center = abstract_at if abstract_at >= 0 and any(term in lower[abstract_at:abstract_at + 3000] for term in matched) else (min(anchors) if anchors else 0)
        passage = text[max(0, center - 400):center + 1600]
        # A source is claim-specific when its fetched body actually discusses
        # the request, not merely shares a keyword.  Distinctive multi-word
        # phrases are the strongest signal; failing that, matching several
        # significant terms together with either good coverage or the exact
        # numeric quantities named in the request (e.g. 20 C, 80 C, 3.3 kohm)
        # is strong evidence for quantitative technical questions, whose
        # word-problem phrasing otherwise inflates the term denominator and
        # depresses coverage below any single fixed cutoff.
        claim_specific = (len(matched_phrases) >= 2 or
                          (len(matched_phrases) >= 1 and bool(matched_numbers)) or
                          (coverage >= 0.45 and len(matched) >= 3) or
                          (len(matched) >= 3 and len(matched_numbers) >= 2))
        relevant = len(matched) >= 2 and coverage >= 0.12 and claim_specific and len(passage) >= 120
        return {
            "status": "verified_content" if relevant else "irrelevant_content",
            "final_url": final_url,
            "content_type": content_type,
            "content_sha256": hashlib.sha256(raw).hexdigest(),
            "content_bytes": len(raw),
            "content_truncated": truncated,
            "evidence_level": evidence_level,
            "matched_terms": matched,
            "matched_phrases": matched_phrases,
            "matched_numbers": matched_numbers,
            "relevance_coverage": round(coverage, 4),
            "passage": passage,
        }

    def authority_score(self, url: str) -> int:
        """Return authority from the actual URL domain, including redirects."""
        host = (urllib.parse.urlparse(str(url)).hostname or "").lower()
        for domain, score in self.AUTHORITATIVE_DOMAINS:
            if host == domain or host.endswith("." + domain):
                return score
        if host.endswith(".gov"):
            return 8
        if host.endswith(".edu") or host.endswith(".ac.uk"):
            return 6
        return 0

    # ------------------------------------------------------------------
    # Authority ranking
    # ------------------------------------------------------------------

    def _is_scientific_query(self, query: str) -> bool:
        """Detect if a query is scientific/engineering in nature."""
        tokens = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
        overlap = tokens & self.SCIENCE_KEYWORDS
        return len(overlap) >= 1

    def _rank_by_authority(self, results: list[dict], query: str = "") -> list[dict]:
        """Re-rank results so authoritative domains appear first."""
        query_terms = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 4}
        def _score(item: dict) -> tuple[int, int]:
            score = self.authority_score(item.get("url", ""))
            item["authority_score"] = score
            haystack = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
            overlap = len({term for term in query_terms if term in haystack})
            item["metadata_relevance"] = overlap
            return -overlap, -score

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

        merged: list[dict] = self._authoritative_seed_results(query)
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
            if len(merged) >= self.max_results and not self._is_scientific_query(query):
                return merged[:10]

        if merged:
            return merged[:30]

        fallback = self._fallback_source_queries(query)
        if errors:
            for item in fallback:
                item["search_errors"] = errors[-5:]
        return fallback

    def _authoritative_seed_results(self, query: str) -> list[dict]:
        """Return curated authority entry points; never pre-approve evidence."""
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        seeds: list[dict] = []
        if "fall" in tokens and ({"gravity", "gravitational", "acceleration"} & tokens):
            seeds.extend([
                {
                    "title": "NASA: Free Fall without Air Resistance",
                    "url": "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/free-fall-without-air-resistance/",
                    "snippet": "NASA educational treatment of free fall, weight, gravitational acceleration, and motion without air resistance.",
                    "provider": "authority-registry",
                    "authority_score": 9,
                },
                {
                    "title": "OpenStax University Physics: Free Fall",
                    "url": "https://openstax.org/books/university-physics-volume-1/pages/3-5-free-fall",
                    "snippet": "Open university-physics chapter deriving and applying constant-acceleration free-fall equations.",
                    "provider": "authority-registry",
                    "authority_score": 7,
                },
            ])
        seed_specs = [
            (({"insulated", "temperature", "heat"} & tokens), "OpenStax Physics: Heat, Specific Heat, and Heat Transfer", "https://openstax.org/books/physics/pages/11-2-heat-specific-heat-and-heat-transfer", "Calorimetry, conservation of energy, specific heat, and equilibrium temperature."),
            (({"circuit", "cutoff", "resistance", "capacitance"} & tokens), "OpenStax University Physics: RC Circuits", "https://openstax.org/books/university-physics-volume-2/pages/10-5-rc-circuits", "Open university-physics treatment of resistor-capacitor circuit equations and time constants."),
            (({"photon", "wavelength", "planck"} & tokens), "NIST CODATA Fundamental Physical Constants", "https://physics.nist.gov/cuu/Constants/index.html", "NIST reference values for Planck constant and speed of light used in photon-energy calculations."),
            (({"photon", "wavelength", "quantum"} & tokens), "OpenStax University Physics: Photon Energies", "https://openstax.org/books/university-physics-volume-3/pages/6-2-photoelectric-effect", "Photon energy, frequency, wavelength, and Planck relation."),
            (({"diffusion", "fickian", "transport"} & tokens), "NIST: Diffusion", "https://www.nist.gov/mml/materials-science-and-engineering-division/diffusion", "NIST materials-science reference material concerning diffusion and transport."),
            (({"orbit", "orbital", "satellite"} & tokens), "OpenStax University Physics: Satellite Orbits and Energy", "https://openstax.org/books/university-physics-volume-1/pages/13-4-satellite-orbits-and-energy", "Newtonian circular-orbit relations, orbital speed, period, and energy."),
            (({"angular", "momentum", "rigid"} & tokens), "OpenStax University Physics: Angular Momentum", "https://openstax.org/books/university-physics-volume-1/pages/11-2-angular-momentum", "Angular momentum and conservation for particles and rigid bodies."),
            (({"drag", "fluid", "wind", "aerodynamic", "navier", "reynolds"} & tokens), "OpenStax University Physics: Drag Force and Terminal Speed", "https://openstax.org/books/university-physics-volume-1/pages/6-4-drag-force-and-terminal-speed", "Drag force equation, drag coefficient, reference area, air density, and terminal-speed relations for subsonic continuum flow."),
            (({"drag", "fluid", "wind", "aerodynamic"} & tokens), "NASA Beginner's Guide: The Drag Equation", "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-equation/", "NASA derivation of the drag equation relating drag force to density, drag coefficient, reference area, and velocity squared."),
            (({"stress", "strain", "yield", "tensile", "strength", "axial", "fatigue", "materials", "wear", "elastic"} & tokens), "OpenStax University Physics: Stress, Strain, and Elastic Modulus", "https://openstax.org/books/university-physics-volume-1/pages/12-3-stress-strain-and-elastic-modulus", "Normal stress sigma = F/A, strain, elastic modulus, yield, and axial-load failure relations for engineering-scale components."),
            (({"planet", "planetary", "atmosphere"} & tokens), "NASA Planetary Fact Sheet", "https://nssdc.gsfc.nasa.gov/planetary/factsheet/", "NASA planetary physical parameters used to calibrate bounded planet models."),
            (({"galaxy", "galactic", "cosmological", "cosmology"} & tokens), "NASA LAMBDA Cosmology Resources", "https://lambda.gsfc.nasa.gov/education/graphic_history/univ_evol.html", "NASA cosmology reference covering expansion history and bounded cosmological models."),
        ]
        for matched, title, url, snippet in seed_specs:
            if matched:
                seeds.append({"title": title, "url": url, "snippet": snippet,
                              "provider": "authority-registry", "authority_score": self.authority_score(url)})
        return seeds

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
            landing = str(item.get("id") or "")
            work_id = landing.rstrip("/").split("/")[-1]
            url_value = f"https://api.openalex.org/works/{work_id}" if work_id else str(doi)
            if not url_value:
                continue
            inverted = item.get("abstract_inverted_index") or {}
            positions = sorted((position, word) for word, indexes in inverted.items() for position in indexes)
            abstract = " ".join(word for _, word in positions)
            snippet = f"{source} {year} {abstract[:800]}".strip()
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
