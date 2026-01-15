"""
Allowed open-access research sources for BrandDozer research runs.

These sources are intentionally limited to openly accessible content that can be
scraped for scientific and engineering research. Always respect the target
site's robots.txt and terms of use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ResearchSource:
    name: str
    domain: str
    category: str
    notes: str
    license: str


ALLOWED_RESEARCH_SOURCES: List[ResearchSource] = [
    ResearchSource(
        name="PubMed / NLM",
        domain="https://pubmed.ncbi.nlm.nih.gov",
        category="biomedical",
        notes="Abstracts and metadata only; full text via PMC where open-access.",
        license="Public metadata; respect NLM usage policies.",
    ),
    ResearchSource(
        name="PMC (PubMed Central)",
        domain="https://www.ncbi.nlm.nih.gov/pmc",
        category="biomedical",
        notes="Open-access full text for qualifying articles.",
        license="Open-access; license varies per article (CC-BY/CC0).",
    ),
    ResearchSource(
        name="arXiv",
        domain="https://arxiv.org",
        category="physics/math/CS",
        notes="Preprints across physics, math, CS; bulk access allowed via HTML/PDF.",
        license="Open access; citation requested.",
    ),
    ResearchSource(
        name="NASA ADS",
        domain="https://ui.adsabs.harvard.edu",
        category="astrophysics",
        notes="Abstracts and links; PDFs often open via arXiv or publisher OA.",
        license="Public metadata.",
    ),
    ResearchSource(
        name="DOAJ",
        domain="https://doaj.org",
        category="multidisciplinary",
        notes="Directory of open-access journals and articles.",
        license="Public metadata; articles vary (mostly CC-BY).",
    ),
    ResearchSource(
        name="OpenAlex",
        domain="https://openalex.org",
        category="metadata",
        notes="Open bibliographic metadata with links to OA content.",
        license="CC0.",
    ),
    ResearchSource(
        name="bioRxiv",
        domain="https://www.biorxiv.org",
        category="biological-systems",
        notes="Open-access biology preprints with rich abstracts and PDFs.",
        license="Open-access; license varies per article (often CC-BY).",
    ),
    ResearchSource(
        name="PLOS (Public Library of Science)",
        domain="https://journals.plos.org",
        category="biological-systems",
        notes="Open-access journals across biology and medicine with high-density abstracts.",
        license="CC-BY.",
    ),
    ResearchSource(
        name="W3C / IETF RFC",
        domain="https://datatracker.ietf.org",
        category="engineering-standards",
        notes="Open internet standards (RFCs); mirror available at rfc-editor.org.",
        license="Open/public domain text.",
    ),
    ResearchSource(
        name="NIST (Reference/Electronics/Metrology)",
        domain="https://www.nist.gov",
        category="engineering-standards",
        notes="Open reference materials, handbooks, and datasets.",
        license="Public domain (US Gov); verify per document.",
    ),
    ResearchSource(
        name="USPTO Bulk Data",
        domain="https://bulkdata.uspto.gov",
        category="patents",
        notes="Bulk patent data for research; observe rate and usage guidance.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="IEEE Standards Open",
        domain="https://opensource.ieee.org",
        category="engineering-standards",
        notes="Open-access standards and supporting material.",
        license="Open; check per standard.",
    ),
    ResearchSource(
        name="Open-source hardware docs (OSHWA directory)",
        domain="https://certification.oshwa.org",
        category="electronics",
        notes="Certified open hardware projects with schematics/BoMs.",
        license="Open hardware licenses; check per project.",
    ),
    ResearchSource(
        name="KiCad Library (schematics/footprints)",
        domain="https://gitlab.com/kicad",
        category="electronics",
        notes="Open EDA footprints/schematics; useful for reference designs.",
        license="Various OSHW/CC licenses; check per repo.",
    ),
    ResearchSource(
        name="ChaosBook",
        domain="https://chaosbook.org",
        category="complex-systems",
        notes="Open chaos theory textbook with high-density exposition.",
        license="CC BY-NC-SA 4.0.",
    ),
    ResearchSource(
        name="Santa Fe Institute",
        domain="https://www.santafe.edu",
        category="complex-systems",
        notes="Complex systems research articles, news, and working papers.",
        license="Site content publicly readable; respect usage guidance.",
    ),
    ResearchSource(
        name="Complexity Explorer",
        domain="https://www.complexityexplorer.org",
        category="complex-systems",
        notes="Open courses and resources on complexity science.",
        license="Content licenses vary; educational use permitted.",
    ),
    ResearchSource(
        name="StackOverflow / StackExchange (CC-BY-SA)",
        domain="https://stackoverflow.com",
        category="code",
        notes="Programming Q&A; respect CC-BY-SA attribution.",
        license="CC-BY-SA 4.0.",
    ),
]


def allowed_research_sources() -> List[ResearchSource]:
    """Return allowed research sources."""
    return list(ALLOWED_RESEARCH_SOURCES)


def allowed_domains() -> List[str]:
    """Return domains that research scrapers should allow."""
    return [src.domain for src in ALLOWED_RESEARCH_SOURCES]
