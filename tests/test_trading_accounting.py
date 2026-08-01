from __future__ import annotations

import pytest

from db import TradingDatabase
from services.trading_accounting import (
    ACCOUNTING_VERSION,
    is_usd_accounting_pair,
    trading_accounting_snapshot,
    validate_outcome_math,
)


def _record(db: TradingDatabase, *, outcome_id: str = "trade-1:exit:1", profit: float = 1.4) -> bool:
    return db.record_trade_outcome(
        outcome_id=outcome_id,
        trade_id="trade-1",
        wallet="ghost",
        chain="base",
        symbol="ETH-USDC",
        session_id=2,
        base_token="ETH",
        quote_token="USDC",
        pnl_currency="USD",
        entry_price=100.0,
        exit_price=101.5,
        quantity=1.0,
        gross_profit=1.5,
        fee_cost=0.1,
        checkpoint=0.4,
        net_profit=profit,
        status="closed",
    )


def test_usd_accounting_pair_rejects_cross_crypto_and_stable_stable() -> None:
    assert is_usd_accounting_pair("ETH", "USDC") is True
    assert is_usd_accounting_pair("WBTC", "WETH") is False
    assert is_usd_accounting_pair("USDT", "USDC") is False


def test_outcome_math_requires_matching_units_and_equation() -> None:
    valid, reason = validate_outcome_math(
        entry_price=100.0,
        exit_price=101.5,
        quantity=1.0,
        gross_profit=1.5,
        fee_cost=0.1,
        net_profit=1.4,
        base_token="ETH",
        quote_token="USDC",
    )
    assert (valid, reason) == (True, "valid")

    valid, reason = validate_outcome_math(
        entry_price=35.0,
        exit_price=68_000.0,
        quantity=0.01,
        gross_profit=679.65,
        fee_cost=0.1,
        net_profit=679.55,
        base_token="WBTC",
        quote_token="WETH",
    )
    assert valid is False
    assert reason == "pnl_currency_not_usd"


def test_outcome_commit_is_idempotent_and_summary_uses_same_population(tmp_path) -> None:
    db = TradingDatabase(str(tmp_path / "accounting.db"))
    assert _record(db) is True
    assert _record(db) is False

    summary = db.trade_outcome_summary("ghost")
    assert summary["closed"] == 1
    assert summary["profitable"] == 1
    assert summary["unprofitable"] == 0
    assert summary["net_profit"] == pytest.approx(1.4)
    assert summary["checkpoint"] == pytest.approx(0.4)
    assert summary["win_rate"] == pytest.approx(1.0)


def test_dashboard_accounting_excludes_legacy_cache(tmp_path) -> None:
    db = TradingDatabase(str(tmp_path / "accounting.db"))
    db.save_state({
        "ghost_trading": {
            "accounting_version": 0,
            "total_profit": 189.61,
            "stable_bank": 27.67,
            "positions": {"ETH-USDC": {}},
        }
    })

    snapshot = trading_accounting_snapshot(db)

    assert snapshot["version"] == ACCOUNTING_VERSION
    assert snapshot["ghost"]["net_profit"] == 0.0
    assert snapshot["ghost"]["closed"] == 0
    assert snapshot["ghost"]["open"] == 1
    assert snapshot["integrity"]["valid"] is False
    assert snapshot["integrity"]["legacy_cached_profit"] == pytest.approx(189.61)
