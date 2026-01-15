"""
Lightweight research content scorer for research mode.

Heuristics target information density and relevance to complex systems,
chaos theory, and biological systems to prioritize high-value documents
without invoking external LLMs.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, Tuple
from urllib.parse import urlparse

from services.research_sources import allowed_research_sources


CHAOS_TERMS = {
    "chaos",
    "chaotic",
    "bifurcation",
    "fractal",
    "attractor",
    "lyapunov",
    "turbulence",
    "nonlinear",
    "sensitive dependence",
    "stochastic",
    "percolation",
    "criticality",
    "entropy",
}

BIO_TERMS = {
    "cellular",
    "protein",
    "gene",
    "genomic",
    "metabolic",
    "immunology",
    "neuron",
    "synapse",
    "signal transduction",
    "pathway",
    "microbiome",
    "biome",
    "epigenetic",
}


def analyze_and_score(text: str, *, url: str = "") -> Tuple[float, Dict]:
    metrics = analyze_text(text or "", url=url)
    # Normalize metrics into [0,1]
    lex_density = metrics.get("lexical_density", 0.0)
    long_ratio = metrics.get("long_word_ratio", 0.0)
    topical = metrics.get("topical_density", 0.0)
    entropy = metrics.get("entropy_norm", 0.0)
    score = (
        0.35 * lex_density
        + 0.25 * long_ratio
        + 0.25 * topical
        + 0.15 * entropy
    )
    metrics["score"] = score
    return score, metrics


def analyze_text(text: str, *, url: str = "") -> Dict:
    tokens = _tokenize(text)
    token_count = max(1, len(tokens))
    unique_tokens = len(set(tokens))
    long_words = sum(1 for t in tokens if len(t) >= 10)
    lex_density = unique_tokens / token_count
    long_word_ratio = long_words / token_count
    topical_hits = _topical_hits(tokens)
    topical_density = topical_hits / token_count
    entropy_norm = _entropy_norm(tokens)
    category = _category_from_url(url)
    return {
        "token_count": token_count,
        "unique_tokens": unique_tokens,
        "lexical_density": lex_density,
        "long_word_ratio": long_word_ratio,
        "topical_hits": topical_hits,
        "topical_density": topical_density,
        "entropy_norm": entropy_norm,
        "category": category,
    }


def _category_from_url(url: str) -> str:
    if not url:
        return ""
    host = urlparse(url).netloc.lower()
    for src in allowed_research_sources():
        domain = urlparse(src.domain).netloc.lower()
        if host.endswith(domain):
            return src.category
    return ""


def _tokenize(text: str) -> Iterable[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())


def _topical_hits(tokens: Iterable[str]) -> int:
    chaos = sum(1 for t in tokens if t in CHAOS_TERMS)
    bio = sum(1 for t in tokens if t in BIO_TERMS)
    return chaos + bio


def _entropy_norm(tokens: Iterable[str]) -> float:
    freq = Counter(tokens)
    total = sum(freq.values()) or 1
    probs = [v / total for v in freq.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(total) if total > 1 else 1.0
    return min(1.0, entropy / max_entropy if max_entropy else 0.0)
