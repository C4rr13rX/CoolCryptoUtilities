from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.fund_tracker import FundTracker
from trading.portfolio import PortfolioState, TokenHolding
from trading.scheduler import BusScheduler, HorizonSignal, RouteState, TradeDirective


@pytest.fixture
def temp_tracker(tmp_path: Path):
    tracker_file = tmp_path / "test_fund_reservations.json"
    tracker = FundTracker(path=tracker_file)
    tracker.clear_all()
    return tracker


def test_fund_tracker_reservations_and_transfers(temp_tracker: FundTracker):
    # 1. Initially zero locked
    assert temp_tracker.get_locked_amount("base", "USDC") == 0.0

    # 2. Reserve 15.5 USDC for a trade
    res = temp_tracker.reserve_funds(
        chain="base",
        token="USDC",
        amount=15.5,
        reason="test_directive",
        ttl_seconds=60.0,
    )
    assert res["id"].startswith("res_")
    assert temp_tracker.get_locked_amount("base", "USDC") == 15.5

    # 3. Add outbound transfer of 20.0 USDC to an external wallet
    xfer = temp_tracker.record_outbound_transfer(
        chain="base",
        token="USDC",
        amount=20.0,
        destination="0x1234567890123456789012345678901234567890",
        reason="manual_send",
        tx_hash="0xabcdef123456",
        source="send_service",
    )
    assert xfer["id"].startswith("xfer_")
    assert temp_tracker.get_locked_amount("base", "USDC") == 35.5

    # 4. Release trade reservation
    assert temp_tracker.release_reservation(res["id"]) is True
    assert temp_tracker.get_locked_amount("base", "USDC") == 20.0

    # 5. Check cache reconciliation:
    # If cache refreshes at T > transfer_ts + 5s, transfer is reconciled
    future_cache_ts = time.time() + 10.0
    reconciled = temp_tracker.reconcile_with_cache("base", cache_ts=future_cache_ts)
    assert reconciled == 1
    # After reconciliation, locked amount drops to 0 because on-chain balance now reflects it
    assert temp_tracker.get_locked_amount("base", "USDC") == 0.0


def test_fund_tracker_token_matching(temp_tracker: FundTracker):
    assert temp_tracker.normalize_token("eth", "base") == "ETH"
    assert temp_tracker.normalize_token("WETH", "base") == "ETH"
    assert temp_tracker.normalize_token("native", "base") == "ETH"
    assert temp_tracker.tokens_match("ETH", "WETH", "base") is True
    assert temp_tracker.tokens_match("usdc", "USDC", "base") is True


def test_portfolio_deducts_locked_funds(temp_tracker: FundTracker, monkeypatch):
    monkeypatch.setattr("trading.portfolio.get_fund_tracker", lambda: temp_tracker)

    # Construct portfolio with mocked DB and cache
    mock_db = MagicMock()
    portfolio = PortfolioState(
        wallet="0x291c0000000000000000000000000000000968ad",
        chains=("base",),
        db=mock_db,
    )
    portfolio.fund_tracker = temp_tracker

    # Populate synthetic holdings: 50.0 USDC, 0.05 ETH
    portfolio.holdings = {
        ("base", "USDC"): TokenHolding(
            token="0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
            symbol="USDC",
            quantity=50.0,
            usd=50.0,
            chain="base",
            raw={},
        )
    }
    portfolio.native_balances = {"base": 0.05}
    portfolio.native_usd = {"base": 150.0}

    # Verify initial raw and available quantities match
    assert portfolio.get_raw_quantity("USDC", chain="base") == 50.0
    assert portfolio.get_quantity("USDC", chain="base") == 50.0
    assert portfolio.get_raw_native_balance("base") == 0.05
    assert portfolio.get_native_balance("base") == 0.05

    # Now simulate a transfer out of 30.0 USDC to another wallet
    temp_tracker.record_outbound_transfer(
        chain="base",
        token="USDC",
        amount=30.0,
        destination="0xexternal_wallet",
        reason="transfer_out",
    )

    # Also simulate 0.02 ETH locked for gas / native send
    temp_tracker.reserve_funds(
        chain="base",
        token="ETH",
        amount=0.02,
        reason="native_send",
    )

    # Available balances should immediately reflect deductions!
    assert portfolio.get_raw_quantity("USDC", chain="base") == 50.0
    assert portfolio.get_quantity("USDC", chain="base") == 20.0  # 50 - 30

    assert portfolio.get_raw_native_balance("base") == 0.05
    assert round(portfolio.get_native_balance("base"), 4) == 0.03  # 0.05 - 0.02

    # Check stable liquidity and summary
    assert portfolio.stable_liquidity("base") == 20.0
    summary = portfolio.summary()
    assert summary["locked_quote"] == 30.0
    assert summary["locked_native"] == 0.02


def test_bus_scheduler_aware_of_locked_funds(temp_tracker: FundTracker, monkeypatch):
    monkeypatch.setattr("trading.portfolio.get_fund_tracker", lambda: temp_tracker)
    monkeypatch.setattr("trading.scheduler.get_fund_tracker", lambda: temp_tracker)

    mock_db = MagicMock()
    scheduler = BusScheduler(db=mock_db)
    scheduler.fund_tracker = temp_tracker

    portfolio = PortfolioState(
        wallet="0x291c0000000000000000000000000000000968ad",
        chains=("base",),
        db=mock_db,
    )
    portfolio.fund_tracker = temp_tracker
    portfolio.holdings = {
        ("base", "USDC"): TokenHolding(
            token="0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
            symbol="USDC",
            quantity=10.0,
            usd=10.0,
            chain="base",
            raw={},
        )
    }
    portfolio.native_balances = {"base": 0.05}

    # Transfer out entire USDC balance to vault
    temp_tracker.record_outbound_transfer(
        chain="base",
        token="USDC",
        amount=10.0,
        destination="0xvault",
        reason="profit_vault_sweep",
    )

    # Mock signals so scheduler reaches quote balance check
    scheduler._forecast = lambda st: [
        HorizonSignal(
            label="1m",
            seconds=60,
            predicted_price=3050.0,
            expected_return=0.03,
            zscore=2.0,
        )
    ]

    # BusScheduler evaluate sample
    sample = {
        "symbol": "CBETH-USDC",
        "price": 3000.0,
        "volume": 50000.0,
        "chain": "base",
        "ts": time.time(),
    }
    pred_summary = {
        "direction_prob": 0.85,
        "exit_conf": 0.90,
        "net_margin": 0.02,
    }

    directive = scheduler.evaluate(
        sample=sample,
        pred_summary=pred_summary,
        portfolio=portfolio,
        live_trading=True,
    )

    # Must be None because quote balance was transferred out!
    assert directive is None

    # Check route state filter reason
    route_state = scheduler.routes.get("CBETH-USDC")
    assert route_state is not None
    assert "funds_locked_or_transferred" in route_state.last_filter_reason
