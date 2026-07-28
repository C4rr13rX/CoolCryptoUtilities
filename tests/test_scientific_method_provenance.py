from unittest.mock import MagicMock

from tools.c0d3rV2.tool_registry import ScientificMethodTool


def test_generic_claim_is_inconclusive_without_executed_test(tmp_path):
    search = MagicMock()
    search.search_authoritative.return_value = {
        "summary": "A search-engine-generated summary.",
        "results": [{
            "title": "A relevant paper",
            "url": "https://doi.org/10.1000/example",
            "snippet": "Metadata snippet",
            "provider": "crossref",
            "authority_score": 7,
        }],
        "scientific": True,
    }
    search.fetch_evidence.return_value = {
        "status": "verified_content", "content_sha256": "a" * 64,
        "passage": "Treatment X and outcome Y were evaluated in a controlled empirical study.",
    }

    result = ScientificMethodTool(search, runtime_dir=tmp_path).execute({
        "question": "Does treatment X improve outcome Y?",
        "domain": "medicine",
    })

    assert result["conclusion"]["status"] == "inconclusive"
    assert result["conclusion"]["confidence"] == 0.0
    assert result["conclusion"]["supported_hypothesis"] is None
    assert result["conclusion"]["falsification_criteria"]


def test_research_records_provenance_and_rejects_fallback_queries(tmp_path):
    search = MagicMock()
    search.search_authoritative.return_value = {
        "summary": "",
        "results": [
            {
                "title": "PubMed search: claim",
                "url": "https://pubmed.ncbi.nlm.nih.gov/?term=claim",
                "snippet": "Fallback source-search URL generated",
                "provider": "fallback-source-query",
                "authority_score": 9,
            },
            {
                "title": "Measured result",
                "url": "https://example.edu/paper",
                "provider": "publisher",
                "authority_score": 8,
            },
        ],
        "scientific": True,
    }
    search.fetch_evidence.return_value = {
        "status": "verified_content", "content_sha256": "b" * 64,
        "passage": "The measured result directly addresses the generic empirical claim.",
    }

    result = ScientificMethodTool(search, runtime_dir=tmp_path).execute({"question": "A generic empirical claim"})
    fallback, paper = result["research"]["results"]

    assert result["research"]["usable_evidence_count"] == 1
    assert fallback["evidence_status"] == "discovery_only"
    assert fallback["evidence_usable"] is False
    assert paper["evidence_usable"] is True
    assert paper["evidence_status"] == "verified_content"
    assert len(paper["provenance_sha256"]) == 64
    assert paper["retrieved_at"].endswith("Z")


def test_metadata_only_or_irrelevant_content_cannot_be_evidence(tmp_path):
    search = MagicMock()
    search.search_authoritative.return_value = {
        "summary": "Misleading metadata",
        "results": [{"title": "Apple leadership", "url": "https://doi.org/example",
                     "provider": "crossref", "authority_score": 7}],
        "scientific": True,
    }
    search.fetch_evidence.return_value = {
        "status": "irrelevant_content", "matched_terms": ["fall"], "relevance_coverage": 0.05,
    }
    result = ScientificMethodTool(search, runtime_dir=tmp_path).execute({
        "question": "How far does an object fall under gravity?", "domain": "physics",
    })
    source = result["research"]["results"][0]
    assert source["evidence_usable"] is False
    assert source["evidence_status"] == "irrelevant_content"
    assert result["research"]["usable_evidence_count"] == 0


def test_known_executable_baseline_still_supports_monty_hall(tmp_path):
    search = MagicMock()
    search.search_authoritative.return_value = {"results": [], "summary": "", "scientific": True}
    result = ScientificMethodTool(search, runtime_dir=tmp_path).execute({
        "question": "In the Monty Hall problem, should the player switch?",
        "domain": "probability",
    })
    assert result["conclusion"]["status"] == "supported"
    assert result["conclusion"]["supported_hypothesis"] == "switch"
    assert result["validation"]["switch_probability"] == 2 / 3
