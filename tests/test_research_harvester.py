from io import BytesIO

from reportlab.pdfgen import canvas

from tools.c0d3rV2.plugins.research_harvester import (
    HarvestConfig,
    ResearchHarvester,
)
from tools.c0d3rV2.plugins.research_harvester.harvester import extract_pdf_text
from tools.c0d3rV2.tool_registry import ResearchHarvesterTool


def test_harvester_indexes_and_retrieves_provenance(tmp_path):
    harvester = ResearchHarvester(tmp_path)
    indexed = harvester.ingest(
        url="https://threejs.org/docs/renderer-adapter",
        title="Renderer dependency injection",
        query="WebGL renderer dependency injection",
        content=(
            "Define a renderer port and inject it into domain scene construction. "
            "Create the real WebGLRenderer in the browser composition root. "
            "Tests provide a renderer test double implementing the port."
        ),
    )
    result = harvester.search("WebGLRenderer renderer port dependency injection", limit=3)
    assert indexed["status"] == "indexed"
    assert result["results"][0]["url"].startswith("https://threejs.org/")
    assert result["results"][0]["content_sha256"]
    assert result["coverage"] > 0.4


def test_harvest_config_enforces_hard_bounds():
    bounded = HarvestConfig(max_depth=99, max_pages=9999, delay_seconds=0).bounded()
    assert bounded.max_depth == 4
    assert bounded.max_pages == 200
    assert bounded.delay_seconds == 0.1


def test_pdf_archival_evidence_is_text_extractable():
    output = BytesIO()
    document = canvas.Canvas(output)
    document.setTitle("Verified Engineering Study")
    document.drawString(
        72, 720, "The controller remained stable in the declared operating regime."
    )
    document.save()
    content, title = extract_pdf_text(output.getvalue(), "fallback")
    assert "controller remained stable" in content
    assert title == "Verified Engineering Study"


def test_tool_retrieves_local_before_web_expansion(tmp_path):
    harvester = ResearchHarvester(tmp_path)
    harvester.ingest(
        url="https://docs.python.org/3/reference/datamodel.html",
        title="Python data model",
        content="Python classes define typed behavior through methods and explicit object invariants. " * 3,
        query="Python classes typed methods object invariants",
    )
    harvester.ingest(
        url="https://peps.python.org/pep-0484/",
        title="Python type hints",
        content="Python typed classes use methods, explicit object invariants, and type annotations. " * 3,
        query="Python classes typed methods object invariants",
    )

    class FailIfCalled:
        def discover(self, query):
            raise AssertionError("web discovery should not run for sufficient local evidence")

    tool = ResearchHarvesterTool(harvester, FailIfCalled())
    result = tool.execute({
        "action": "research",
        "query": "Python classes typed methods object invariants",
    })
    assert result["status"] == "local_hit"


def test_retrieval_prioritizes_official_documentation(tmp_path):
    harvester = ResearchHarvester(tmp_path)
    for url, title in (
        ("https://stackoverflow.com/questions/example", "Community answer"),
        ("https://threejs.org/docs/pages/WebGLRenderer.html", "Official WebGLRenderer"),
    ):
        harvester.ingest(
            url=url, title=title,
            content="WebGLRenderer renderer dependency injection adapter testing context " * 3,
            query="renderer adapter testing",
        )
    result = harvester.search("WebGLRenderer dependency injection adapter testing")
    assert result["results"][0]["url"].startswith("https://threejs.org/")


def test_project_policy_persists_bounded_feedback_controls(tmp_path):
    harvester = ResearchHarvester(tmp_path)
    policy = harvester.configure_project(
        "robot-world", query="renderer dependency injection",
        seeds=["https://threejs.org/docs/pages/WebGLRenderer.html"],
        config=HarvestConfig(max_depth=99, max_pages=999),
        coverage_target=4.0, refresh_seconds=1, max_rounds=99,
    )
    assert policy["project_key"] == "robot-world"
    assert policy["config"]["max_depth"] == 4
    assert policy["config"]["max_pages"] == 200
    assert policy["coverage_target"] == 1.0
    assert policy["refresh_seconds"] == 300
    assert policy["max_rounds"] == 4
    assert policy["due"] is True


def test_project_refresh_stops_when_local_coverage_is_sufficient(tmp_path):
    harvester = ResearchHarvester(tmp_path)
    for index in range(2):
        harvester.ingest(
            url=f"https://threejs.org/docs/example-{index}", title=f"Evidence {index}",
            content="renderer dependency injection adapter contract factory browser composition root " * 3,
            query="renderer dependency injection adapter contract",
        )
    tool = ResearchHarvesterTool(harvester, None)
    tool.execute({
        "action": "project_configure", "project_key": "robot-world",
        "query": "renderer dependency injection adapter contract", "coverage_target": 0.6,
    })
    result = tool.execute({"action": "project_refresh", "project_key": "robot-world", "force": True})
    assert result["status"] == "sufficient"
    assert result["rounds"] == []
    assert result["reason"] == "coverage target met"
    unchanged = harvester.configure_project(
        "robot-world", query="renderer dependency injection adapter contract",
        seeds=[], config=HarvestConfig(), coverage_target=0.6,
    )
    assert unchanged["due"] is False
    changed = harvester.configure_project(
        "robot-world", query="a different renderer research question",
        seeds=[], config=HarvestConfig(), coverage_target=0.6,
    )
    assert changed["due"] is True
