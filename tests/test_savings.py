from __future__ import annotations

from trading.savings import StableSavingsPlanner


def test_savings_planner_batches_contributions() -> None:
    planner = StableSavingsPlanner(min_batch=100.0)
    event1 = planner.record_allocation(
        amount=60.0,
        token="USDC",
        mode="ghost",
        equilibrium_score=0.8,
        trade_id="t1",
    )
    assert event1.amount == 60.0
    assert not planner.drain_ready_transfers()

    planner.record_allocation(
        amount=50.0,
        token="USDC",
        mode="ghost",
        equilibrium_score=0.82,
        trade_id="t2",
    )
    transfers = planner.drain_ready_transfers()
    assert len(transfers) == 1
    assert transfers[0].amount == 110.0
    assert transfers[0].token == "USDC"
