"""branddozer/modes.py — What each BrandDozer project mode actually runs.

Single source of truth for the UI, so forms describe the real execution
model instead of showing one generic prompt loop for every project.

Background
----------
A BrandDozer project executes in one of two fundamentally different ways,
decided by ``BrandProject.workflow_kind``:

* **empty ``workflow_kind``** — the generic loop in
  ``services/branddozer_runner.py::_run_cycle``.  It sends
  ``default_prompt`` and then each entry of ``interjections`` on *every*
  scheduled interval.  Here "runs every cycle" is literally true.

* **non-empty ``workflow_kind``** — ``run_project_workflow`` short-circuits
  that loop entirely (``branddozer_runner.py:172``), so ``default_prompt``
  and ``interjections`` are **never read**.  Showing them as "runs every
  cycle" is actively wrong for these projects.

Delivery runs are a third shape: a one-shot goal, not a repeating loop.
A research delivery run feeds its prompt to the *planner* role once
(``research.py::_plan``), which decomposes it into work packages that
drive six distinct role sub-cycles.  Calling that prompt "runs every
cycle" misdescribes it — hence ``prompt_role`` below.
"""
from __future__ import annotations

from typing import Any


# Phases of a research delivery run, in execution order.  Mirrors
# ResearchWorkflow's call sequence in web/branddozer/research.py.
RESEARCH_PHASES = (
    {
        "role": "research_planner",
        "label": "Planner",
        "detail": "Turns your goal into a protocol and 3-8 work packages.",
        "consumes_run_prompt": True,
    },
    {
        "role": "literature_reviewer",
        "label": "Literature review",
        "detail": "Runs per work package; gathers findings, sources, claims.",
        "consumes_run_prompt": False,
    },
    {
        "role": "methods_reviewer",
        "label": "Methods review",
        "detail": "Audits method, bias risks, validity limits.",
        "consumes_run_prompt": False,
    },
    {
        "role": "research_writer",
        "label": "Writer",
        "detail": "Drafts the paper from verified evidence.",
        "consumes_run_prompt": False,
    },
    {
        "role": "citation_auditor",
        "label": "Citation audit",
        "detail": "Verifies every claim maps to a real, checked source.",
        "consumes_run_prompt": False,
    },
    {
        "role": "peer_reviewer",
        "label": "Peer review",
        "detail": "Accepts or sends back for revision (bounded rounds).",
        "consumes_run_prompt": False,
    },
)

SOFTWARE_PHASES = (
    {
        "role": "planner",
        "label": "Planner",
        "detail": "Turns your goal into a backlog of testable items.",
        "consumes_run_prompt": True,
    },
    {
        "role": "worker",
        "label": "Build",
        "detail": "Implements backlog items in the workspace.",
        "consumes_run_prompt": False,
    },
    {
        "role": "qa",
        "label": "QA / gates",
        "detail": "Runs smoke tests and quality gates.",
        "consumes_run_prompt": False,
    },
    {
        "role": "auditor",
        "label": "Audit",
        "detail": "Reviews the result before acceptance.",
        "consumes_run_prompt": False,
    },
)


# Continuous-project modes, keyed by workflow_kind ("" = generic loop).
PROJECT_MODES: tuple[dict[str, Any], ...] = (
    {
        "id": "",
        "label": "Continuous prompt loop",
        "summary": "Sends your prompts on a repeating schedule.",
        "uses_prompt_loop": True,
        "prompt_label": "Default prompt (runs every cycle)",
        "prompt_help": (
            "Sent at the start of every scheduled cycle, then each "
            "interjection below runs in order."
        ),
        "supports_interjections": True,
        "uses_interval": True,
    },
    {
        "id": "digital_product_business",
        "label": "Digital product loop",
        "summary": (
            "Runs the built-in product-refinement workflow. Your prompts are "
            "not used; the workflow defines each cycle's work."
        ),
        "uses_prompt_loop": False,
        "prompt_label": "Mission (context only)",
        "prompt_help": (
            "Optional context shown on the project card. This workflow does "
            "not send it as a prompt."
        ),
        "supports_interjections": False,
        "uses_interval": True,
    },
)

PROJECT_MODE_IDS = {mode["id"] for mode in PROJECT_MODES}


# One-shot delivery runs, keyed by project_type.
DELIVERY_MODES: tuple[dict[str, Any], ...] = (
    {
        "id": "software",
        "label": "Software / technology",
        "summary": "Builds working software in the project workspace.",
        "prompt_label": "Build goal",
        "prompt_help": (
            "Describe what to build. The planner reads this once and "
            "decomposes it into backlog items — it is not re-sent each cycle."
        ),
        "prompt_placeholder": (
            "Describe the software to build: what it does, who uses it, and "
            "what done looks like."
        ),
        "prompt_role": "planner",
        "phases": SOFTWARE_PHASES,
        "fields": ["team_mode", "smoke_test_cmd"],
    },
    {
        "id": "research",
        "label": "Archival research paper",
        "summary": "Produces a cited research paper from archival sources.",
        "prompt_label": "Research goal",
        "prompt_help": (
            "State the question, scope, and audience. The planner reads this "
            "once and derives work packages — each later phase writes its own "
            "prompts, so this is not re-sent every cycle."
        ),
        "prompt_placeholder": (
            "State the research goal, question, scope, and intended "
            "scientific or engineering audience."
        ),
        "prompt_role": "research_planner",
        "phases": RESEARCH_PHASES,
        # Research always runs the full role team; team_mode is not offered.
        "fields": [
            "target_journal", "citation_style", "min_words",
            "min_verified_sources",
        ],
    },
)

DELIVERY_MODE_IDS = {mode["id"] for mode in DELIVERY_MODES}


def project_mode(workflow_kind: str | None) -> dict[str, Any]:
    """Descriptor for a continuous project's execution mode."""
    key = str(workflow_kind or "").strip().lower()
    for mode in PROJECT_MODES:
        if mode["id"] == key:
            return mode
    # Unknown workflow_kind: it still bypasses the prompt loop, so describe
    # it honestly rather than falling back to the generic loop's wording.
    return {
        "id": key,
        "label": key.replace("_", " ").title() or "Custom workflow",
        "summary": "Custom workflow; your prompts are not used as cycle input.",
        "uses_prompt_loop": False,
        "prompt_label": "Mission (context only)",
        "prompt_help": "This workflow does not send the prompt each cycle.",
        "supports_interjections": False,
        "uses_interval": True,
    }


def delivery_mode(project_type: str | None) -> dict[str, Any]:
    """Descriptor for a one-shot delivery run's output type."""
    key = str(project_type or "software").strip().lower()
    for mode in DELIVERY_MODES:
        if mode["id"] == key:
            return mode
    return DELIVERY_MODES[0]


def catalog() -> dict[str, Any]:
    """Full mode catalog for the UI."""
    return {
        "project_modes": list(PROJECT_MODES),
        "delivery_modes": list(DELIVERY_MODES),
    }


__all__ = [
    "PROJECT_MODES",
    "DELIVERY_MODES",
    "PROJECT_MODE_IDS",
    "DELIVERY_MODE_IDS",
    "RESEARCH_PHASES",
    "SOFTWARE_PHASES",
    "project_mode",
    "delivery_mode",
    "catalog",
]
