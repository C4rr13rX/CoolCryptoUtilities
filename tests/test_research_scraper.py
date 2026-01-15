from unittest import mock

import pytest

from pathlib import Path
from tempfile import TemporaryDirectory
from services.research_scraper import (
    DuckDuckGoScraper,
    _is_allowed,
    _strip_tags,
    _html_to_text,
    extract_metadata,
)


def test_strip_tags():
    assert _strip_tags("<b>Title</b>") == "Title"
    assert _strip_tags("plain text") == "plain text"


def test_is_allowed_filters_domains():
    domains = ["https://example.com", "https://foo.org"]
    assert _is_allowed("https://example.com/page", domains)
    assert not _is_allowed("https://bar.com/page", domains)


def test_html_to_text_strips_scripts_and_collapses_space():
    html = "<html><head><script>var x=1;</script></head><body><h1>Title</h1><p>Body text</p></body></html>"
    assert _html_to_text(html) == "Title Body text"


def test_extract_metadata_prefers_meta_description():
    html = """
    <html><head>
    <title>My Page</title>
    <meta name="description" content="Meta description">
    </head><body></body></html>
    """
    meta = extract_metadata(html, url="https://example.com")
    assert meta["title"] == "My Page"
    assert meta["description"] == "Meta description"


def test_search_parses_results_and_filters_domains():
    html = """
    <a class="result__a" href="https://allowed.com/doc1">Doc 1</a>
    <a class="result__a" href="https://blocked.com/doc2">Doc 2</a>
    """
    scraper = DuckDuckGoScraper()
    with mock.patch.object(scraper.session, "get") as get_mock:
        response = mock.Mock()
        response.status_code = 200
        response.text = html
        get_mock.return_value = response
        results = scraper.search("query", max_results=5, domains=["https://allowed.com"])
    assert len(results) == 1
    assert results[0].url == "https://allowed.com/doc1"
    assert results[0].title == "Doc 1"


def test_fetch_text_uses_session_get():
    scraper = DuckDuckGoScraper()
    with mock.patch.object(scraper.session, "get") as get_mock:
        response = mock.Mock()
        response.status_code = 200
        response.text = "content"
        response.encoding = "utf-8"
        get_mock.return_value = response
        content = scraper.fetch_text("https://allowed.com/doc")
    assert content == "content"


def test_fetch_document_truncates_and_returns_metadata():
    scraper = DuckDuckGoScraper()
    with mock.patch.object(scraper, "fetch_text") as fetch_text:
        fetch_text.return_value = "<html><head><title>Doc</title></head><body>" + ("X" * 10_000) + "</body></html>"
        doc = scraper.fetch_document("https://allowed.com/doc", max_chars=100)
    assert doc.title == "Doc"
    assert len(doc.content) <= 100
    assert doc.cached is False
    assert doc.score >= 0
    assert doc.metrics is not None


def test_fetch_document_uses_cache_when_available():
    with TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        scraper = DuckDuckGoScraper(cache_dir=cache_dir)
        url = "https://allowed.com/doc"
        html = "<html><head><title>Cached</title></head><body>Cached body</body></html>"
        # Pre-populate cache
        (cache_dir / f"{scraper._cache_path(url).name}").write_text(html, encoding="utf-8")
        with mock.patch.object(scraper, "fetch_text") as fetch_text:
            doc = scraper.fetch_document(url)
        fetch_text.assert_not_called()
        assert doc.cached is True
        assert doc.title == "Cached"
