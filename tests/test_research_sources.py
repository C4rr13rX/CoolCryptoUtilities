from services import research_sources


def test_allowed_research_sources_non_empty():
    sources = research_sources.allowed_research_sources()
    assert sources, "Expected at least one allowed research source"
    domains = research_sources.allowed_domains()
    assert domains, "Expected allowed domains list"
    # Basic category coverage
    categories = {src.category for src in sources}
    assert {"biomedical", "engineering-standards", "electronics", "complex-systems", "biological-systems"}.intersection(categories)


def test_allowed_domains_align_with_sources():
    sources = research_sources.allowed_research_sources()
    domains = set(research_sources.allowed_domains())
    assert all(src.domain in domains for src in sources)
