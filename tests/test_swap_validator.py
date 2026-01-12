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


def test_plan_transition_blocks_on_ghost_and_sparse_wallet() -> None:
    validator = SwapValidator()
    readiness = {
        "precision": 0.72,
        "recall": 0.71,
        "samples": 140,
        "ghost_ready": False,
        "ghost_reason": "tail_risk",
        "wallet_state": {"sparse": True, "stable_usd": 10.0, "min_capital_usd": 50.0},
    }
    plan = validator.plan_transition(
        positions={},
        exposure={},
        readiness=readiness,
        risk_budget=1.0,
    )
    assert plan["allowed"] is False
    assert "ghost_not_ready" in plan["block_reasons"]
    assert plan["bus_swap_plan"] is not None


def test_plan_transition_requires_ghost_sample_depth() -> None:
    validator = SwapValidator()
    readiness = {
        "precision": 0.7,
        "recall": 0.69,
        "ghost_validation": {"ready": True, "samples": 5, "min_trades": 10},
    }
    plan = validator.plan_transition(
        positions={},
        exposure={},
        readiness=readiness,
        risk_budget=1.0,
    )
    assert plan["allowed"] is False
    assert "ghost_sample_gap" in plan["block_reasons"]
    assert plan["bus_swap_plan"]["reason"].startswith("ghost") or plan["bus_swap_plan"]["reason"] == "fragmented_wallet"


def test_plan_transition_consolidates_fragmented_wallet() -> None:
    validator = SwapValidator()
    readiness = {"precision": 0.72, "recall": 0.7, "samples": 120}
    wallet_state = {"fragmented": True, "fragment_ratio": 0.6, "dust_tokens": ["DUST", "TINY"], "stable_usd": 200.0}
    plan = validator.plan_transition(
        positions={},
        exposure={},
        readiness=readiness,
        risk_budget=1.0,
        wallet_state=wallet_state,
    )
    assert plan["allowed"] is False
    assert "fragmented_wallet" in plan["block_reasons"]
    assert plan["bus_swap_plan"] is not None
    assert plan["bus_swap_plan"]["action"] == "consolidate_fragments"
