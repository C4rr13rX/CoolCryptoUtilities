"""
Allowed open-access research sources for BrandDozer research runs.

These sources are intentionally limited to openly accessible content that can be
scraped for scientific and engineering research. Always respect the target
site's robots.txt and terms of use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


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
    # Biomedical & life sciences
    ResearchSource(
        name="NCBI",
        domain="https://www.ncbi.nlm.nih.gov",
        category="biomedical",
        notes="NCBI landing for structured biomedical datasets and publications.",
        license="Public metadata; content licenses vary by resource.",
    ),
    ResearchSource(
        name="NCBI E-utilities",
        domain="https://eutils.ncbi.nlm.nih.gov",
        category="biomedical",
        notes="NCBI E-utilities API (no key required, rate-limited).",
        license="Public metadata; respect NCBI usage policies.",
    ),
    ResearchSource(
        name="PubChem",
        domain="https://pubchem.ncbi.nlm.nih.gov",
        category="chemistry",
        notes="Chemical structures, properties, and bioactivity via PUG-REST.",
        license="Public domain where applicable; check per dataset.",
    ),
    ResearchSource(
        name="RCSB PDB",
        domain="https://www.rcsb.org",
        category="biomedical",
        notes="Protein structure database with API access.",
        license="Open data; see RCSB usage guidance.",
    ),
    ResearchSource(
        name="UniProt",
        domain="https://www.uniprot.org",
        category="biomedical",
        notes="Protein sequence and annotation datasets with REST access.",
        license="Open data; cite UniProt where appropriate.",
    ),
    ResearchSource(
        name="Ensembl",
        domain="https://www.ensembl.org",
        category="biomedical",
        notes="Genomics data with REST API for genes, variants, and sequences.",
        license="Open access; cite Ensembl.",
    ),
    ResearchSource(
        name="UCSC Genome Browser",
        domain="https://genome.ucsc.edu",
        category="biomedical",
        notes="Genome data with track hubs and query interfaces.",
        license="Open access; cite UCSC.",
    ),
    ResearchSource(
        name="Europe PMC",
        domain="https://europepmc.org",
        category="biomedical",
        notes="Open-access biomedical literature and metadata API.",
        license="Public metadata; OA content varies.",
    ),
    ResearchSource(
        name="EMBL-EBI",
        domain="https://www.ebi.ac.uk",
        category="biomedical",
        notes="Life science data resources (ENA, BioModels, ChEMBL).",
        license="Open access; resource-specific licenses apply.",
    ),
    ResearchSource(
        name="OpenNeuro",
        domain="https://openneuro.org",
        category="biomedical",
        notes="Open neuroimaging datasets with structured metadata.",
        license="Open data; dataset-specific licenses apply.",
    ),
    ResearchSource(
        name="PhysioNet",
        domain="https://physionet.org",
        category="biomedical",
        notes="Physiological signal datasets and open challenges.",
        license="Open access; follow PhysioNet usage policies.",
    ),
    ResearchSource(
        name="Human Protein Atlas",
        domain="https://www.proteinatlas.org",
        category="biomedical",
        notes="Tissue/cell protein expression data.",
        license="Open access; cite Human Protein Atlas.",
    ),
    ResearchSource(
        name="GTEx Portal",
        domain="https://gtexportal.org",
        category="biomedical",
        notes="Genotype-Tissue Expression data portal.",
        license="Open access; follow GTEx data use.",
    ),
    ResearchSource(
        name="ClinicalTrials.gov",
        domain="https://clinicaltrials.gov",
        category="biomedical",
        notes="Clinical trials registry with JSON/CSV access.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="OpenFDA",
        domain="https://api.fda.gov",
        category="biomedical",
        notes="OpenFDA endpoints for drug/device adverse events.",
        license="Public domain (US Gov).",
    ),
    # Scholarly metadata & repositories
    ResearchSource(
        name="Crossref API",
        domain="https://api.crossref.org",
        category="metadata",
        notes="Scholarly metadata with DOI resolution.",
        license="CC0 metadata.",
    ),
    ResearchSource(
        name="DataCite API",
        domain="https://api.datacite.org",
        category="metadata",
        notes="DOI metadata for datasets and research outputs.",
        license="CC0 metadata.",
    ),
    ResearchSource(
        name="OpenCitations",
        domain="https://opencitations.net",
        category="metadata",
        notes="Open citation indexes with APIs.",
        license="CC0 metadata.",
    ),
    ResearchSource(
        name="Semantic Scholar API",
        domain="https://api.semanticscholar.org",
        category="metadata",
        notes="Open scholarly graph API (rate-limited, no key required).",
        license="Non-commercial use; see Semantic Scholar terms.",
    ),
    ResearchSource(
        name="Zenodo",
        domain="https://zenodo.org",
        category="datasets",
        notes="Open data repository with REST API.",
        license="Open access; dataset licenses vary.",
    ),
    ResearchSource(
        name="Figshare API",
        domain="https://api.figshare.com",
        category="datasets",
        notes="Open research datasets with REST API.",
        license="Open access; dataset licenses vary.",
    ),
    ResearchSource(
        name="Dryad",
        domain="https://datadryad.org",
        category="datasets",
        notes="Open data repository with structured metadata.",
        license="Open access; dataset licenses vary.",
    ),
    ResearchSource(
        name="Harvard Dataverse",
        domain="https://dataverse.harvard.edu",
        category="datasets",
        notes="Open datasets with API access.",
        license="Open access; dataset licenses vary.",
    ),
    ResearchSource(
        name="OpenML",
        domain="https://www.openml.org",
        category="datasets",
        notes="Machine learning datasets and tasks with API.",
        license="Open access; dataset licenses vary.",
    ),
    # Government & global statistics
    ResearchSource(
        name="Data.gov",
        domain="https://data.gov",
        category="datasets",
        notes="US government open data catalog.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="USGS Water Services",
        domain="https://waterservices.usgs.gov",
        category="earth_science",
        notes="Hydrology data services with simple query APIs.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="USGS Earthquake",
        domain="https://earthquake.usgs.gov",
        category="earth_science",
        notes="Earthquake feeds and catalogs with JSON endpoints.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="USGS ScienceBase",
        domain="https://www.sciencebase.gov",
        category="earth_science",
        notes="Geoscience datasets and metadata catalog.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="NOAA Weather API",
        domain="https://api.weather.gov",
        category="earth_science",
        notes="National Weather Service API (no key required).",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="NOAA CO-OPS Tides",
        domain="https://api.tidesandcurrents.noaa.gov",
        category="earth_science",
        notes="Tides and currents API (no key required).",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="EPA Envirofacts",
        domain="https://envirofacts.epa.gov",
        category="earth_science",
        notes="Environmental data warehouse with query endpoints.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="EPA ECHO",
        domain="https://echo.epa.gov",
        category="earth_science",
        notes="Environmental compliance data and APIs.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="WHO GHO API",
        domain="https://ghoapi.azureedge.net",
        category="biomedical",
        notes="World Health Organization Global Health Observatory API.",
        license="Open data; cite WHO.",
    ),
    ResearchSource(
        name="CDC Open Data",
        domain="https://data.cdc.gov",
        category="biomedical",
        notes="CDC datasets on health, surveillance, and outcomes.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="World Bank API",
        domain="https://api.worldbank.org",
        category="statistics",
        notes="World Bank data API for indicators and country stats.",
        license="Open data; cite World Bank.",
    ),
    ResearchSource(
        name="UN Data",
        domain="https://data.un.org",
        category="statistics",
        notes="United Nations data catalog for statistics.",
        license="Open data; cite UN.",
    ),
    ResearchSource(
        name="Eurostat",
        domain="https://ec.europa.eu/eurostat",
        category="statistics",
        notes="European statistics with SDMX/JSON endpoints.",
        license="Open data; cite Eurostat.",
    ),
    # Geospatial & climate
    ResearchSource(
        name="OpenStreetMap Overpass API",
        domain="https://overpass-api.de",
        category="earth_science",
        notes="Overpass API for OSM geospatial queries.",
        license="ODbL.",
    ),
    ResearchSource(
        name="OpenTopoData",
        domain="https://api.opentopodata.org",
        category="earth_science",
        notes="Open elevation API (no key required).",
        license="Open data; dataset licenses vary.",
    ),
    ResearchSource(
        name="Open-Meteo",
        domain="https://api.open-meteo.com",
        category="earth_science",
        notes="Weather and climate API (no key required).",
        license="CC BY 4.0 for data; cite Open-Meteo.",
    ),
    ResearchSource(
        name="HydroShare",
        domain="https://www.hydroshare.org",
        category="earth_science",
        notes="Hydrology data repository with open access.",
        license="Open data; dataset licenses vary.",
    ),
    # Physics & astronomy
    ResearchSource(
        name="CERN Open Data",
        domain="https://opendata.cern.ch",
        category="physics",
        notes="High-energy physics datasets and documentation.",
        license="Open data; cite CERN.",
    ),
    ResearchSource(
        name="Particle Data Group",
        domain="https://pdg.lbl.gov",
        category="physics",
        notes="Authoritative particle physics data tables.",
        license="Open access; cite PDG.",
    ),
    ResearchSource(
        name="INSPIRE HEP",
        domain="https://inspirehep.net",
        category="physics",
        notes="High-energy physics literature and metadata.",
        license="Open access; metadata usage allowed.",
    ),
    ResearchSource(
        name="VizieR Catalog Service",
        domain="https://vizier.cds.unistra.fr",
        category="astronomy",
        notes="Astronomical catalogs with machine-readable tables.",
        license="Open data; cite CDS/VizieR.",
    ),
    ResearchSource(
        name="SIMBAD Astronomical Database",
        domain="https://simbad.u-strasbg.fr",
        category="astronomy",
        notes="Astronomical object database with query endpoints.",
        license="Open data; cite SIMBAD.",
    ),
    ResearchSource(
        name="NASA Technical Reports (NTRS)",
        domain="https://ntrs.nasa.gov",
        category="engineering",
        notes="NASA technical reports and datasets.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="DOE OSTI",
        domain="https://osti.gov",
        category="engineering",
        notes="US DOE technical reports and datasets.",
        license="Public domain (US Gov).",
    ),
    # Math & engineering references
    ResearchSource(
        name="NIST DLMF",
        domain="https://dlmf.nist.gov",
        category="math",
        notes="Digital Library of Mathematical Functions.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="OEIS",
        domain="https://oeis.org",
        category="math",
        notes="Integer sequence database with formulas and references.",
        license="Open access; cite OEIS.",
    ),
    ResearchSource(
        name="RFC Editor",
        domain="https://rfc-editor.org",
        category="standards",
        notes="Internet standards text (RFCs).",
        license="Public domain.",
    ),
    ResearchSource(
        name="W3C Standards",
        domain="https://www.w3.org",
        category="standards",
        notes="Open web standards with technical specs.",
        license="W3C Document License.",
    ),
    ResearchSource(
        name="NIST Chemistry WebBook",
        domain="https://webbook.nist.gov",
        category="chemistry",
        notes="Thermochemical, spectra, and chemical property data.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="NIST Data",
        domain="https://data.nist.gov",
        category="engineering",
        notes="NIST datasets catalog and APIs.",
        license="Public domain (US Gov).",
    ),
    ResearchSource(
        name="Materials Cloud",
        domain="https://www.materialscloud.org",
        category="materials",
        notes="Materials science datasets and workflows.",
        license="Open access; dataset licenses vary.",
    ),
    ResearchSource(
        name="Open Energy Info (OpenEI)",
        domain="https://openei.org",
        category="engineering",
        notes="Energy datasets and APIs (no key required for basic access).",
        license="Open data; cite OpenEI.",
    ),
    ResearchSource(
        name="medRxiv",
        domain="https://www.medrxiv.org",
        category="biomedical",
        notes="Medical preprints with open access PDFs.",
        license="Open access; license varies per article.",
    ),
]


KEYWORD_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "biomedical": (
        "medicine",
        "clinical",
        "disease",
        "drug",
        "pharma",
        "patient",
        "trial",
        "rna",
        "dna",
        "protein",
        "genome",
        "neuro",
        "cancer",
        "virus",
        "biomedical",
        "pubmed",
        "ncbi",
        "pmc",
    ),
    "biology": ("biology", "cell", "enzyme", "organism", "ecology", "evolution", "microbio"),
    "chemistry": ("chemistry", "chemical", "compound", "molecule", "reaction", "spectra", "pubchem", "caffeine"),
    "physics": ("physics", "quantum", "particle", "relativity", "optics", "astrophysics", "electron", "mass"),
    "astronomy": ("astronomy", "cosmology", "telescope", "stellar", "galaxy"),
    "math": ("math", "algebra", "calculus", "topology", "statistics", "probability", "matrix", "dlmf", "nist"),
    "computer_science": ("algorithm", "complexity", "data structure", "compiler", "database", "network", "programming", "software", "ai", "ml"),
    "engineering": ("engineering", "circuit", "electronics", "mechanical", "materials", "aerospace", "control", "signal", "rf", "power"),
    "materials": ("materials", "alloy", "crystal", "semiconductor"),
    "earth_science": ("geology", "earthquake", "hydrology", "climate", "weather", "ocean", "atmosphere", "geospatial", "gis", "noaa"),
    "standards": ("rfc", "ietf", "w3c", "standard", "specification", "protocol"),
    "patents": ("patent", "intellectual property"),
    "datasets": ("dataset", "open data", "catalog", "repository"),
    "statistics": ("statistics", "economics", "indicator", "population", "health stats"),
}


def _score_categories(query: str) -> Dict[str, int]:
    text = (query or "").lower()
    scores: Dict[str, int] = {}
    for category, keywords in KEYWORD_CATEGORIES.items():
        count = sum(1 for kw in keywords if kw in text)
        if count:
            scores[category] = count
    return scores


def select_sources_for_query(query: str, *, max_sources: int = 12) -> List[ResearchSource]:
    scores = _score_categories(query)
    if scores:
        ranked_categories = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        preferred = [cat for cat, _ in ranked_categories]
    else:
        preferred = []
    selected: List[ResearchSource] = []
    if preferred:
        for cat in preferred:
            for src in ALLOWED_RESEARCH_SOURCES:
                if src.category == cat and src not in selected:
                    selected.append(src)
                if len(selected) >= max_sources:
                    return selected
    for src in ALLOWED_RESEARCH_SOURCES:
        if src.category in {"metadata", "datasets", "standards"} and src not in selected:
            selected.append(src)
        if len(selected) >= max_sources:
            return selected
    for src in ALLOWED_RESEARCH_SOURCES:
        if src not in selected:
            selected.append(src)
        if len(selected) >= max_sources:
            break
    return selected


def allowed_domains_for_query(query: str, *, max_sources: int = 12) -> List[str]:
    return [src.domain for src in select_sources_for_query(query, max_sources=max_sources)]


def allowed_research_sources() -> List[ResearchSource]:
    """Return allowed research sources."""
    return list(ALLOWED_RESEARCH_SOURCES)


def allowed_domains() -> List[str]:
    """Return domains that research scrapers should allow."""
    return [src.domain for src in ALLOWED_RESEARCH_SOURCES]
