"""A strategy's first-ever outcome must reach the LEDGER, not only the registry.

The ledger is what gates graduation; the registry is only the lifetime record.
A strategy present in one and absent from the other can never graduate no
matter how well it trades, and nothing pinned that the two stay together.

That is not hypothetical. Measured 2026-09-02, money_button had exactly ONE
round trip in the whole database -- TOAD-USDC, entered 08-31 16:21, exited
16:26, +0.005278. The registry holds it. data/strategy_ledger.json held only
atf_static, obv_accumulation@1w, rsi_reversal@5h and obv_accumulation@3d --
money_button was absent entirely. The lost write is explained by the
last-write-wins race fixed in 86ae43f/3b49f93 (09-02 01:00, after that trade),
and ``record()`` writes the registry BEFORE it takes the ledger's file lock, so
the registry surviving alone is exactly the shape that failure leaves behind.

These pin the two writes as one unit, from the first outcome onward.
"""
from __future__ import annotations

import pytest

import services.strategy_registry as registry
from trading.strategies.ledger import StrategyLedger


@pytest.fixture()
def paired(tmp_path, monkeypatch):
    """A ledger whose registry side-effect is redirected to a temp file.

    ``record()`` only feeds the registry when it is the PRODUCTION ledger, so
    DEFAULT_PATH is pointed at the temp ledger to exercise the real both-files
    path without writing to data/strategy_registry.json.
    """
    led_path = tmp_path / "ledger.json"
    monkeypatch.setattr(StrategyLedger, "DEFAULT_PATH", led_path)
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_TRADES", "5")
    return StrategyLedger(path=led_path)


def test_a_first_ever_outcome_lands_in_the_ledger(paired):
    """money_button's real trade, at its real size."""
    paired.record("money_button", profit=0.005277519134108606, mode="ghost",
                  confidence=0.5, symbol="TOAD-USDC")

    stats = paired.stats("money_button")["ghost"]
    assert stats["trades"] == 1, "the ledger must hold the strategy's first trade"
    assert stats["wins"] == 1
    assert stats["total_profit"] == pytest.approx(0.005277519134108606)


def test_one_record_call_writes_both_files(paired):
    """Neither file may gain a trade the other one missed."""
    paired.record("money_button", profit=0.005277519134108606, mode="ghost",
                  symbol="TOAD-USDC")

    led = paired.stats("money_button")["ghost"]
    reg = registry.get_strategy("money_button")["lifetime"]["ghost"]
    assert led["trades"] == reg["trades"] == 1
    assert led["wins"] == reg["wins"] == 1


def test_a_small_first_outcome_is_not_rejected_as_implausible(paired):
    """The plausibility guard has no history to scale against on trade #1.

    A strategy whose very first outcome is discarded is one that can never
    accumulate the history the guard needs -- it would be permanently stuck at
    zero trades, which reads identically to "it never traded".
    """
    for profit in (0.005277519134108606, -0.0031, 0.0):
        paired.record("money_button", profit=profit, mode="ghost", symbol="TOAD-USDC")

    assert paired.stats("money_button")["ghost"]["trades"] == 3


def test_the_symbol_reaches_the_lifetime_record(paired):
    """Without this the registry holds an empty symbols map and per-symbol
    behaviour cannot be analysed at all."""
    paired.record("money_button", profit=0.0052, mode="ghost", symbol="TOAD-USDC")
    paired.record("money_button", profit=-0.0011, mode="ghost", symbol="BSTONK-USDC")

    symbols = registry.get_strategy("money_button")["lifetime"]["ghost"]["symbols"]
    assert symbols == {"TOAD-USDC": 1, "BSTONK-USDC": 1}


def test_a_strategy_in_the_registry_is_never_missing_from_the_ledger(paired):
    """The divergence itself, stated as an invariant."""
    for i, profit in enumerate([0.004, -0.002, 0.006, -0.001, 0.003]):
        paired.record("money_button", profit=profit, mode="ghost",
                      symbol=f"S{i}-USDC")

    def _ghost_trades(entry):
        return int(((entry.get("lifetime") or {}).get("ghost") or {}).get("trades", 0) or 0)

    in_registry = {
        str(ent.get("strategy_id") or "")
        for ent in registry.list_strategies()
        if _ghost_trades(ent) > 0
    }
    in_ledger = {
        sid for sid, ent in paired.snapshot().items()
        if int((ent.get("ghost") or {}).get("trades", 0) or 0) > 0
    }
    assert in_registry <= in_ledger, (
        f"strategies with a lifetime record but no ledger entry: "
        f"{sorted(in_registry - in_ledger)} -- these can never graduate"
    )
