"""The AERO pair from item [d763940a], built from the real booked numbers.

    AERO-USDC  entry 0.436805 -> exit 1.140000   +160.99%  net +3.2066
    AERO-USDC  entry 1.140000 -> exit 0.513839    -54.93%  net -1.1115

The second trade's ENTRY IS THE FIRST TRADE'S EXIT. The feed had not reached
AERO yet at that moment -- its first AERO tick is roughly seven hours later --
so the FEED half of the corroboration gate returns unjudgeable and the ghost
lane, which is the lane that booked these, lets it through.

These tests fail against the feed-only gate: ``entry_price_is_corroborated``
returned True for 1.140000 with no feed coverage, because unjudgeable defaults
to allowed for ghost. They pass once the book's own prior entries are consulted.
"""

from __future__ import annotations

import sqlite3

import pytest

from services.entry_price_corroboration import (
    BOOK_MAX_RATIO,
    book_disagreement,
    entry_price_is_corroborated,
)

# The real rows, and the real ordering: the honest entry is booked first.
HONEST_ENTRY = 0.436805
CONTAMINATED_ENTRY = 1.140000
T0 = 1_756_200_000.0


@pytest.fixture()
def book(tmp_path):
    """A trading_cache with the AERO pair and NO feed coverage, as it was."""
    db = tmp_path / "trading_cache.db"
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE trade_outcomes (outcome_id INTEGER PRIMARY KEY, ts REAL, "
        "symbol TEXT, entry_price REAL, exit_price REAL, net_profit REAL)"
    )
    con.execute("CREATE TABLE market_stream (symbol TEXT, ts REAL, price REAL)")
    con.execute(
        "INSERT INTO trade_outcomes (ts, symbol, entry_price, exit_price, net_profit) "
        "VALUES (?, 'AERO-USDC', ?, ?, 3.2066)",
        (T0, HONEST_ENTRY, CONTAMINATED_ENTRY),
    )
    con.commit()
    con.close()
    return db


def test_the_contaminated_entry_is_refused_with_no_feed_at_all(book):
    """1.140000 is 2.61x the only AERO entry the book has ever opened."""
    ok = entry_price_is_corroborated(
        "AERO-USDC", CONTAMINATED_ENTRY, at_ts=T0 + 600.0, db_path=book)
    assert ok is False, (
        "the second trade opened at the first trade's contaminated exit; the "
        "gate must refuse it even though the feed had not reached AERO")


def test_the_honest_entry_that_came_first_is_not_refused(book):
    """0.436805 has no prior AERO entry behind it, so it is unjudgeable.

    This is the asymmetry that makes the gate safe: it refuses the row that
    INHERITED the fiction, not the row that preceded it.
    """
    r = book_disagreement("AERO-USDC", HONEST_ENTRY, at_ts=T0 - 1.0, db_path=book)
    assert r["ratio"] is None
    assert r["support"] == 0
    assert entry_price_is_corroborated(
        "AERO-USDC", HONEST_ENTRY, at_ts=T0 - 1.0, db_path=book) is True


def test_the_ratio_is_the_measured_one(book):
    r = book_disagreement("AERO-USDC", CONTAMINATED_ENTRY, at_ts=T0 + 600.0,
                          db_path=book)
    assert r["median"] == pytest.approx(HONEST_ENTRY)
    assert r["ratio"] == pytest.approx(CONTAMINATED_ENTRY / HONEST_ENTRY, rel=1e-6)
    assert r["ratio"] > BOOK_MAX_RATIO
    assert r["disagrees"] is True


def test_a_normal_move_in_the_same_symbol_is_not_refused(book):
    """The gate must not refuse a real move, or it is switched off, not safe.

    0.53 is 1.21x the prior entry -- inside the p99 of 1.52 measured over the
    175 judgeable rows by scripts/entry_basis_census.py.
    """
    assert entry_price_is_corroborated(
        "AERO-USDC", 0.53, at_ts=T0 + 600.0, db_path=book) is True


def test_a_prior_entry_older_than_the_window_does_not_judge(book):
    """A basis from a day-plus ago is not evidence about today's price."""
    r = book_disagreement("AERO-USDC", CONTAMINATED_ENTRY,
                          at_ts=T0 + 86400.0 + 60.0, db_path=book)
    assert r["ratio"] is None, "the prior entry has aged out of the window"


def test_an_exit_price_cannot_corroborate_itself(book):
    """The contaminated tick IS the first trade's EXIT.

    Measured in scripts/entry_basis_census.py: including exits drags AERO's
    median from 0.436805 to 0.788402 and the contaminated row reads 1.45x --
    under BSTONK-USDC's legitimate 1.81x, so it is MISSED. The gate therefore
    reads entries only, and this test is what holds that line.
    """
    r = book_disagreement("AERO-USDC", CONTAMINATED_ENTRY, at_ts=T0 + 600.0,
                          db_path=book)
    assert r["support"] == 1, "one prior ENTRY, not two booked prices"
    assert r["median"] == pytest.approx(HONEST_ENTRY)
