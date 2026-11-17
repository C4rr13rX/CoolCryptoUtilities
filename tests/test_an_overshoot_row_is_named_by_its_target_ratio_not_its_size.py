"""Overshoot take-profit rows must be named by exit/target ratio, never gain size."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.annul_overshoot_take_profits import (
    DEFAULT_FEE_RATE,
    apply_corrections,
    judge_overshoot_row,
    scan_overshoot_rows,
)


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test_outcomes.db"
    con = sqlite3.connect(str(db_path))
    con.execute(
        "CREATE TABLE trading_ops ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, wallet TEXT, chain TEXT, "
        "symbol TEXT, action TEXT, status TEXT, details TEXT)"
    )
    con.execute(
        "CREATE TABLE trade_outcomes ("
        "outcome_id TEXT PRIMARY KEY, trade_id TEXT, wallet TEXT, chain TEXT, "
        "symbol TEXT, entry_price REAL, exit_price REAL, quantity REAL, "
        "gross_profit REAL, fee_cost REAL, net_profit REAL, status TEXT, "
        "ts REAL, details TEXT)"
    )
    con.commit()
    con.close()
    return db_path


def _insert_round_trip(
    db_path: Path,
    *,
    hash_id: str,
    symbol: str,
    strategy_id: str,
    entry_price: float,
    target_price: float | None,
    exit_price: float,
    quantity: float = 10.0,
    fee_cost: float = 0.01,
    reason: str = "take_profit_limit",
    ts: float = 1_790_000_000.0,
) -> None:
    con = sqlite3.connect(str(db_path))
    trade_id = f"2:{symbol}:{hash_id}"
    if target_price is not None:
        con.execute(
            "INSERT INTO trading_ops (ts, wallet, chain, symbol, action, status, details) "
            "VALUES (?, 'ghost', 'base', ?, 'enter', 'ghost-entry', ?)",
            (
                ts - 60.0,
                symbol,
                json.dumps(
                    {
                        "trade_id": trade_id,
                        "strategy_id": strategy_id,
                        "target_price": target_price,
                    }
                ),
            ),
        )
    gross = (exit_price - entry_price) * quantity
    net = gross - fee_cost
    con.execute(
        "INSERT INTO trade_outcomes VALUES (?, ?, 'ghost', 'base', ?, ?, ?, ?, ?, ?, ?, 'closed', ?, ?)",
        (
            f"{trade_id}:exit:1",
            trade_id,
            symbol,
            entry_price,
            exit_price,
            quantity,
            gross,
            fee_cost,
            net,
            ts,
            json.dumps({"mode": "ghost", "reason": reason, "strategy_id": strategy_id}),
        ),
    )
    con.commit()
    con.close()


def test_large_legitimate_winner_at_its_own_target_is_kept_while_small_overshoot_is_caught(
    tmp_path: Path,
) -> None:
    """A size threshold gets BOTH of these wrong:
    - BIG_WIN (+30% return, target +30%, exit/target = 1.000) is a real fill at its limit -> KEEP.
    - SMALL_OVERSHOOT (+4% return, target +0.8%, exit/target = 1.0317 > 1+fee) is a gap fill -> CATCH.
    """
    db_path = _make_db(tmp_path)
    _insert_round_trip(
        db_path,
        hash_id="aaa111",
        symbol="WETH-USDC",
        strategy_id="money_button",
        entry_price=1.00,
        target_price=1.30,
        exit_price=1.3005,  # +30.05% gain, but only 0.038% past its 1.30 target (< 0.3187% fee)
    )
    _insert_round_trip(
        db_path,
        hash_id="bbb222",
        symbol="BSTONK-USDC",
        strategy_id="atf_static",
        entry_price=1.00,
        target_price=1.008,
        exit_price=1.040,  # only +4.0% gain, but 3.17% past its 1.008 target (> 0.3187% fee)
    )

    con = sqlite3.connect(str(db_path))
    rows = {r["symbol"]: r for r in scan_overshoot_rows(con, fee_rate=DEFAULT_FEE_RATE)}
    con.close()

    assert rows["WETH-USDC"]["is_overshoot"] is False
    assert rows["WETH-USDC"]["verdict"] == "at_or_within_limit"

    assert rows["BSTONK-USDC"]["is_overshoot"] is True
    assert rows["BSTONK-USDC"]["verdict"] == "overshoot"
    assert rows["BSTONK-USDC"]["target_ratio"] == pytest.approx(1.040 / 1.008)


def test_missing_entry_target_is_marked_unknown_and_never_guessed(tmp_path: Path) -> None:
    """A row with no recoverable target_price must not be annulled or repriced on a guessed limit."""
    db_path = _make_db(tmp_path)
    _insert_round_trip(
        db_path,
        hash_id="ccc333",
        symbol="UNI-USDC",
        strategy_id="rsi_reversal@1w",
        entry_price=2.859,
        target_price=None,
        exit_price=6.3723,
    )

    res = apply_corrections(db_path, action="reprice", backup=False)
    assert res["modified"] == 0

    con = sqlite3.connect(str(db_path))
    status, exit_px = con.execute(
        "SELECT status, exit_price FROM trade_outcomes WHERE symbol='UNI-USDC'"
    ).fetchone()
    con.close()
    assert status == "closed"
    assert exit_px == pytest.approx(6.3723)


def test_reprice_clamps_overshoot_to_its_own_limit_and_leaves_non_limit_exits_untouched(
    tmp_path: Path,
) -> None:
    db_path = _make_db(tmp_path)
    _insert_round_trip(
        db_path,
        hash_id="ddd444",
        symbol="AERO-USDC",
        strategy_id="atf_static",
        entry_price=1.00,
        target_price=1.05,
        exit_price=1.25,
        quantity=2.0,
        fee_cost=0.01,
        reason="take_profit_limit",
    )
    _insert_round_trip(
        db_path,
        hash_id="eee555",
        symbol="CBBTC-USDC",
        strategy_id="atf_static",
        entry_price=1.00,
        target_price=1.05,
        exit_price=0.95,
        quantity=2.0,
        fee_cost=0.01,
        reason="stop_loss",
    )

    res = apply_corrections(db_path, action="reprice", fee_rate=DEFAULT_FEE_RATE, backup=False)
    assert res["modified"] == 1

    expected_fill = 1.05 * (1.0 + DEFAULT_FEE_RATE)
    expected_gross = (expected_fill - 1.00) * 2.0
    expected_net = expected_gross - 0.01

    con = sqlite3.connect(str(db_path))
    exit_px, gross, net = con.execute(
        "SELECT exit_price, gross_profit, net_profit FROM trade_outcomes WHERE symbol='AERO-USDC'"
    ).fetchone()
    con.close()
    assert exit_px == pytest.approx(expected_fill)
    assert gross == pytest.approx(expected_gross)
    assert net == pytest.approx(expected_net)


def test_judge_overshoot_row_ignores_gain_size_completely() -> None:
    huge_real_win = judge_overshoot_row(
        entry_price=10.0,
        exit_price=18.0,  # +80% gain!
        quantity=1.0,
        gross_profit=8.0,
        net_profit=7.95,
        fee_cost=0.05,
        reason="target_hit",
        target_price=18.0,  # target was 18.0 -> ratio 1.000
    )
    assert huge_real_win["is_overshoot"] is False
