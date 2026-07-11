from __future__ import annotations

from tools.c0d3rV2.plugins.agent_the_freeloader.benchmarks import CASES, case_prompt, run_benchmarks
from scripts.validate_atf_benchmark import project_root


def test_benchmark_catalog_has_requested_frameworks() -> None:
    frameworks = " ".join(case.framework for case in CASES).lower()
    assert "django" in frameworks
    assert "dearpygui" in frameworks
    assert "ionic 8" in frameworks
    assert "qt 6" in frameworks
    assert len(CASES) >= 5
    assert all(case.constraints and case.acceptance for case in CASES)


def test_benchmark_prompt_requires_evidence_and_web_search() -> None:
    prompt = case_prompt(CASES[0])
    assert "Implement the complete project" in prompt
    assert "WebSearch" in prompt
    assert "do not fabricate" in prompt


def test_enqueue_only_creates_isolated_jobs(tmp_path) -> None:
    report = run_benchmarks(
        tmp_path, ["tkinter-pid-instrument"], max_attempts=2,
        hours=0.01, enqueue_only=True,
    )
    job = report["jobs"]["tkinter-pid-instrument"]
    assert job["status"] == "queued"
    assert job["max_attempts"] == 2
    assert "validate_atf_benchmark.py" in job["validation_command"]


def test_validator_accepts_one_nested_framework_root(tmp_path) -> None:
    nested = tmp_path / "generated-app"
    nested.mkdir()
    (nested / "angular.json").write_text("{}", encoding="utf-8")
    assert project_root(tmp_path, "angular.json") == nested
