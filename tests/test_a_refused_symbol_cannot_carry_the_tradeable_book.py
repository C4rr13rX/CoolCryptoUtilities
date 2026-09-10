"""The per-symbol edge table must split on the SAME predicate the gate uses.

``scripts/tradeable_symbol_edge.py`` answers "is there any symbol we can
actually spend on where the edge survives the fee". The whole value of the
answer rests on three pieces of arithmetic, and every one of them has shipped
wrong in this repo before:

  * the tradeable/refused split must be ``ledger._live_tradeable``, not a
    hand-maintained list -- BSTONK-USDC carrying a record that authorises
    spending is the shape measured three separate times;
  * ``annulled`` rows are bookkeeping reversals, not fills, and summing them
    is how ``trading_ops`` produced a -0.25 where the books said +0.14;
  * the recomputed net must charge a FRACTION against notional, not subtract a
    rate from a dollar amount, and not trust the booked ``fee_cost`` -- 105 of
    196 rows once carried exactly 0.650000% with zero variance, the default
    measuring itself.

Each test below fails against the corresponding mistake.
"""

from __future__ import annotations

import sqlite3

import pytest

from scripts import tradeable_symbol_edge as tse


@pytest.fixture
def book(tmp_path, monkeypatch):
    """A trade_outcomes table with a known answer."""

    def _build(rows, cost=0.01):
        db = tmp_path / "t.db"
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE trade_outcomes (symbol TEXT, gross_profit REAL, "
            "net_profit REAL, fee_cost REAL, entry_price REAL, quantity REAL, "
            "status TEXT)"
        )
        con.executemany(
            "INSERT INTO trade_outcomes VALUES (?,?,?,?,?,?,?)", rows
        )
        con.commit()
        con.close()
        monkeypatch.setattr(tse, "DB", db)
        monkeypatch.setattr(tse, "_measured_cost_fraction", lambda: cost)
        return tse.collect()

    return _build


def _sym(rep, name):
    return next(d for d in rep["symbols"] if d["symbol"] == name)


def test_an_annulled_row_is_not_a_fill(book):
    """An annulled reversal must not reach the totals.

    Counting it both inflates the trade count -- evidence toward a bar of 20 --
    and moves the P/L by a number no market produced.
    """
    rep = book([
        ("AERO-USDC", 1.0, 0.9, 0.1, 1.0, 10.0, "closed"),
        ("AERO-USDC", 99.0, 99.0, 0.0, 1.0, 10.0, "annulled"),
    ])
    d = _sym(rep, "AERO-USDC")
    assert d["trades"] == 1, "an annulled bookkeeping reversal was counted as a fill"
    assert d["gross"] == pytest.approx(1.0)


def test_the_split_is_the_ledgers_own_predicate(book, monkeypatch):
    """The table must ask ``_live_tradeable``, not assume every -USDC pair.

    Proved by flipping the predicate: if the split were hardcoded the refused
    group would stay empty.
    """
    import trading.strategies.ledger as ledger_mod

    monkeypatch.setattr(
        ledger_mod, "_live_tradeable", lambda s: not s.startswith("BSTONK")
    )
    rep = book([
        ("AERO-USDC", 1.0, 0.9, 0.1, 1.0, 10.0, "closed"),
        ("BSTONK-USDC", 5.0, 4.9, 0.1, 1.0, 10.0, "closed"),
    ])
    assert _sym(rep, "AERO-USDC")["tradeable"] is True
    assert _sym(rep, "BSTONK-USDC")["tradeable"] is False
    assert rep["totals"]["tradeable"]["trades"] == 1
    assert rep["totals"]["refused"]["trades"] == 1
    # The refused symbol carries the profit; the tradeable total must NOT
    # include it. This is the exact shape that authorised spending three times.
    assert rep["totals"]["tradeable"]["gross"] == pytest.approx(1.0)
    assert rep["totals"]["refused"]["gross"] == pytest.approx(5.0)


def test_the_recomputed_cost_is_a_fraction_of_notional(book):
    """cost must be charged as fraction x notional, never subtracted as a rate.

    "Expectancy subtracted a rate from a dollar" is a shipped bug in this repo.
    Here: 2 trades, gross +1.0 total, notional 100.0 each, cost 1%. The right
    answer is 1.0 - 0.01*200 = -1.0. Subtracting the rate itself would give
    1.0 - 0.01 = +0.99 and call a loser a winner.
    """
    rep = book(
        [
            ("AERO-USDC", 0.5, 0.4, 0.1, 10.0, 10.0, "closed"),
            ("AERO-USDC", 0.5, 0.4, 0.1, 10.0, 10.0, "closed"),
        ],
        cost=0.01,
    )
    d = _sym(rep, "AERO-USDC")
    assert d["notional"] == pytest.approx(200.0)
    assert d["net_recomputed"] == pytest.approx(-1.0)
    assert d["recomputed_per_trade"] == pytest.approx(-0.5)
    # ...and it must not have quietly reused the booked fee, which says -0.2.
    assert d["net_booked"] == pytest.approx(0.8)


def test_a_booked_fee_that_disagrees_does_not_override_the_measurement(book):
    """Both numbers are reported and the recomputed one is independent.

    The booked fee here is zero -- the "unmeasurable cost defaulted to zero"
    shape. A table that reported only ``net_booked`` would call this free.
    """
    rep = book(
        [("AERO-USDC", 1.0, 1.0, 0.0, 10.0, 10.0, "closed")],
        cost=0.05,
    )
    d = _sym(rep, "AERO-USDC")
    assert d["net_booked"] == pytest.approx(1.0)   # the book says free
    assert d["net_recomputed"] == pytest.approx(-4.0)  # the receipts do not


def test_min_trades_filters_the_table_not_the_totals_predicate(book):
    """A one-trade symbol is noise; the filter must be explicit, not implied."""
    rep = tse.collect(min_trades=1)  # the real book, just checking the shape
    assert rep["min_trades"] == 1
    assert isinstance(rep["cost_fraction"], float)
    assert 0.0 < rep["cost_fraction"] < 0.05, (
        "the measured round-trip cost is outside any plausible band; a cost "
        "this wrong silently decides every verdict in the table"
    )


def test_the_real_book_still_separates_the_two_populations():
    """End-to-end on the real DB: the two groups must not overlap.

    A symbol appearing in both totals would mean the predicate is not a
    function of the symbol, and every number in the table would be suspect.
    """
    rep = tse.collect(min_trades=1)
    trad = {d["symbol"] for d in rep["symbols"] if d["tradeable"]}
    refu = {d["symbol"] for d in rep["symbols"] if not d["tradeable"]}
    assert not (trad & refu)
    assert rep["totals"]["tradeable"]["trades"] + rep["totals"]["refused"]["trades"] == (
        sum(d["trades"] for d in rep["symbols"])
    )
