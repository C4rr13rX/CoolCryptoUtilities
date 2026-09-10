"""A scout ghost-exit row that is a FRACTION must never be summed as DOLLARS.

THE BUG THIS PREVENTS, AND WHY IT IS WORTH A FILE
-------------------------------------------------
``services/atf_static_strategy.py`` used to write the bare return FRACTION into
the ``profit`` key of its ghost-exit rows, and never charged them a fee. That
writer was corrected on 2026-09-06 to write USD net of the round trip, and it
stamps ``profit_unit: "usd"`` so a reader can tell a corrected row from a legacy
one BY FACT rather than by guessing from its magnitude.

Both populations are still in ``trading_ops`` and will be for as long as the
table is append-only, which is forever. Measured 2026-09-10 the scout's 109
ghost-exit rows are 105 legacy fractions (sum +2.0705, ages 3.4-7.8 days) and 4
corrected dollar rows (sum -0.1097). A reader that ignores ``profit_unit`` adds
+2.0705 "dollars" that are actually a sum of percentages, and it does so with
the SIGN INVERTED relative to the only rows that were ever charged a cost.

That is not a hypothetical: it is the same class of defect as the fee in the
wrong currency and the t-test over dollars that should have been over returns,
and the ledger's +6.4498 for this strategy is built on it.

So the loader in ``scripts/scout_book_audit.py`` refuses any row that does not
say ``usd``. These tests pin that refusal. They fail against a loader that
treats an absent ``profit_unit`` as dollars -- which is exactly what every
naive reader of this table does.
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from scripts.scout_book_audit import (
    SCOUT_STRATEGY_ID,
    load_scout_rows,
    verify_units,
)


def _make_db(tmp_path, rows):
    path = tmp_path / "ops.db"
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE trading_ops (id INTEGER PRIMARY KEY, ts REAL, wallet TEXT, "
        "chain TEXT, symbol TEXT, action TEXT, status TEXT, details TEXT)"
    )
    for ts, symbol, details in rows:
        con.execute(
            "INSERT INTO trading_ops (ts, wallet, chain, symbol, action, status, details) "
            "VALUES (?, 'ghost', 'base', ?, 'exit', 'ghost-exit', ?)",
            (ts, symbol, json.dumps(details)),
        )
    con.commit()
    con.close()
    return path


def _legacy_row(symbol="AERO-USDC", fraction=0.0421):
    """A row as the writer produced it BEFORE the 2026-09-06 correction.

    Note what is absent: no ``profit_unit``, no ``clip_usd``, no
    ``roundtrip_cost_usd``. ``profit`` is the bare fraction and no fee was ever
    charged against it. These are the real keys, taken from the live table.
    """
    return {
        "source": "c0d3rv2_atf_static",
        "strategy_id": SCOUT_STRATEGY_ID,
        "symbol": symbol,
        "chain": "base",
        "entry_price": 1.0,
        "exit_price": 1.0 + fraction,
        "profit": fraction,
        "reason": "target",
        "exit_reason": "target",
        "entry_ts": 1000.0,
        "exit_ts": 2000.0,
    }


def _corrected_row(symbol="AERO-USDC", fraction=0.0421, clip=6.0):
    cost = 0.004047 + 0.003187 * clip
    return {
        "source": "c0d3rv2_atf_static",
        "strategy_id": SCOUT_STRATEGY_ID,
        "symbol": symbol,
        "chain": "base",
        "entry_price": 1.0,
        "exit_price": 1.0 + fraction,
        "profit": fraction * clip - cost,
        "profit_unit": "usd",
        "return_pct": fraction,
        "clip_usd": clip,
        "roundtrip_cost_usd": cost,
        "reason": "target",
        "exit_reason": "target",
        "entry_ts": 1000.0,
        "exit_ts": 2000.0,
    }


def test_a_legacy_fraction_row_is_not_loaded_as_a_dollar_row(tmp_path):
    """The core guard: an unstamped row is REFUSED, not silently believed.

    A loader that reads ``profit`` without checking ``profit_unit`` returns
    this row with net=0.0421 "dollars", which is a percentage wearing a dollar
    sign. Refusing it is the only safe reading, because the row also carries no
    ``clip_usd`` -- there is nothing to multiply the fraction BY, so the dollar
    value is not merely wrong, it is unrecoverable.
    """
    now = time.time()
    db = _make_db(tmp_path, [(now - 100.0, "AERO-USDC", _legacy_row())])

    rows = load_scout_rows(db, now - 86400.0)

    assert rows == [], (
        "a legacy ghost-exit row with no profit_unit was loaded as USD; "
        "its 'profit' is a bare fraction and summing it into a dollar book "
        "is the defect that put 104 fraction rows into the ghost book"
    )


def test_a_corrected_usd_row_is_loaded_and_keeps_its_sign(tmp_path):
    """The other half: the guard must not refuse everything.

    A guard that drops every row would 'pass' the test above while destroying
    the book, so this pins that a stamped row survives AND that it arrives
    net-of-cost -- negative here, because a 0.0421 fraction on a $6 clip does
    clear its cost and a losing one must stay losing.
    """
    now = time.time()
    db = _make_db(tmp_path, [(now - 100.0, "AERO-USDC", _corrected_row(fraction=0.0421))])

    rows = load_scout_rows(db, now - 86400.0)

    assert len(rows) == 1
    row = rows[0]
    assert row["strategy_id"] == SCOUT_STRATEGY_ID
    assert row["notional"] == pytest.approx(6.0)
    # net is USD net of the round trip; gross is net + fees, pre-cost.
    assert row["net"] == pytest.approx(0.0421 * 6.0 - (0.004047 + 0.003187 * 6.0))
    assert row["gross"] == pytest.approx(0.0421 * 6.0)
    assert row["gross"] > row["net"], "gross must sit above net by the fee leg"


def test_a_losing_corrected_row_stays_negative(tmp_path):
    """A trip that does not clear its cost must arrive NEGATIVE.

    This is the row shape the whole audit turns on: measured 2026-09-10 all 4
    corrected scout rows sum to -0.1097 while the 105 unstamped ones sum to
    +2.0705. If a losing row could arrive positive, that sign inversion would
    be invisible.
    """
    now = time.time()
    db = _make_db(tmp_path, [(now - 100.0, "AERO-USDC", _corrected_row(fraction=0.0001))])

    rows = load_scout_rows(db, now - 86400.0)

    assert len(rows) == 1
    assert rows[0]["net"] < 0.0, "a trip below its cost must book a loss"


def test_rows_from_another_strategy_are_not_counted_as_the_scouts(tmp_path):
    """Attribution is on details.strategy_id, never on a LIKE over the blob.

    A substring match over the JSON counted 120 rows where the scout has 109:
    other writers name the scout inside their own payloads (a refusal row says
    which strategy it refused), and counting a refusal as a round trip inflates
    the exact book this audit exists to deflate.
    """
    now = time.time()
    other = _corrected_row()
    other["strategy_id"] = "atf_static"
    # A refusal-shaped payload that MENTIONS the scout without being its trip.
    mentions = _corrected_row()
    mentions["strategy_id"] = "rsi_reversal"
    mentions["refused_strategy"] = SCOUT_STRATEGY_ID

    db = _make_db(tmp_path, [
        (now - 300.0, "AERO-USDC", other),
        (now - 200.0, "AERO-USDC", mentions),
        (now - 100.0, "AERO-USDC", _corrected_row()),
    ])

    rows = load_scout_rows(db, now - 86400.0)

    assert len(rows) == 1, (
        "attribution matched a strategy id mentioned inside another writer's "
        "payload; only details.strategy_id names the row's own author"
    )


def test_only_ghost_exit_rows_are_round_trips(tmp_path):
    """An entry is not a round trip. Counting one books a close that never happened."""
    now = time.time()
    path = tmp_path / "ops.db"
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE trading_ops (id INTEGER PRIMARY KEY, ts REAL, wallet TEXT, "
        "chain TEXT, symbol TEXT, action TEXT, status TEXT, details TEXT)"
    )
    for status in ("ghost-entry", "entry-predropped-edge-ban", "ghost-exit"):
        con.execute(
            "INSERT INTO trading_ops (ts, wallet, chain, symbol, action, status, details) "
            "VALUES (?, 'ghost', 'base', 'AERO-USDC', 'exit', ?, ?)",
            (now - 100.0, status, json.dumps(_corrected_row())),
        )
    con.commit()
    con.close()

    rows = load_scout_rows(path, now - 86400.0)

    assert len(rows) == 1, "only ghost-exit rows close a round trip"


def test_the_units_check_catches_a_gross_that_disagrees_with_its_legs(tmp_path):
    """``verify_units`` must actually detect a broken mapping, not always pass.

    ``gross`` is reconstructed as ``net + fees`` and must equal
    ``return_pct * clip_usd``. If a future edit to the mapping breaks that,
    every dollar figure downstream is wrong, so the check has to have teeth.
    """
    now = time.time()
    good = _corrected_row()
    bad = _corrected_row(symbol="CBBTC-USDC")
    # A row whose stated return does not match the dollars it booked.
    bad["return_pct"] = 0.9

    db = _make_db(tmp_path, [(now - 200.0, "AERO-USDC", good),
                             (now - 100.0, "CBBTC-USDC", bad)])
    rows = load_scout_rows(db, now - 86400.0)
    report = verify_units(rows)

    assert report["rows"] == 2
    assert report["worst_abs_gap_usd"] > 1.0
    assert report["worst_symbol"] == "CBBTC-USDC"

    clean = verify_units([r for r in rows if r["symbol"] == "AERO-USDC"])
    assert clean["worst_abs_gap_usd"] == pytest.approx(0.0, abs=1e-9)
