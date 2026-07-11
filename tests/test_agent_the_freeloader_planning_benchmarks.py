from __future__ import annotations

from tools.c0d3rV2.plugins.agent_the_freeloader.planning_benchmarks import CASES, score_plan


def test_planning_catalog_covers_requested_domains() -> None:
    text = " ".join(f"{case.id} {case.domain}" for case in CASES).lower()
    assert "blockchain" in text
    assert "intranet" in text
    assert "radio" in text
    assert "scientific" in text


def test_plan_score_rewards_integrated_constraint_rich_plan() -> None:
    case = CASES[0]
    branches = [
        {
            "id": "requirements", "description": "measure consensus partition behavior and data sovereignty",
            "rationale": "establish constraints", "dependencies": [],
            "constraints": ["key rotation", "scheduler cost"],
            "acceptance_criteria": ["validation test passes with zero raw regulated data on-chain"],
            "recovery_policy": "rollback divergent research and return to acceptance criteria",
        },
        {
            "id": "recovery", "description": "implement recovery",
            "rationale": "resilience", "dependencies": ["requirements"],
            "constraints": ["partition recovery"],
            "acceptance_criteria": ["recovery test passes"],
            "recovery_policy": "reconverge on requirements after evidence review",
        },
    ]
    score = score_plan(case, branches)
    assert score["score"] >= 80
    assert not score["invalid_dependencies"]


def test_plan_score_rejects_forward_dependency() -> None:
    case = CASES[0]
    score = score_plan(case, [{
        "id": "first", "description": "x", "rationale": "x", "dependencies": ["later"],
        "constraints": ["x"], "acceptance_criteria": ["test passes"],
        "recovery_policy": "return to plan",
    }, {
        "id": "later", "description": "y", "rationale": "y", "dependencies": [],
        "constraints": ["y"], "acceptance_criteria": ["test passes"],
        "recovery_policy": "return to plan",
    }])
    assert score["invalid_dependencies"] == ["first->later"]
