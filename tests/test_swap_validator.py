from __future__ import annotations

from trading.swap_validator import SwapValidator


def test_plan_transition_scales_with_confidence() -> None:
    validator = SwapValidator()
    readiness = {"precision": 0.75, "recall": 0.72, "samples": 180}
    plan = validator.plan_transition(
        positions={},
        exposure={"WETH-USDC": 0.2},
        readiness=readiness,
        risk_budget=1.0,
    )
    assert plan["allowed"] is True
    assert plan["adjusted_risk_budget"] <= plan["risk_budget"]
    assert plan["confidence_margin"] > 0


def test_plan_transition_blocks_when_exposure_exceeds_budget() -> None:
    validator = SwapValidator()
    readiness = {"precision": 0.5, "recall": 0.52, "samples": 40}
    plan = validator.plan_transition(
        positions={"WETH-USDC": {"entry": 1}},
        exposure={"WETH-USDC": 0.5},
        readiness=readiness,
        risk_budget=0.25,
    )
    assert plan["allowed"] is False
    assert plan["confidence_margin"] < 0
