"""The plausibility censor must run where production DECIDES, not only in reports.

Item [af06afd6]. ``services.outcome_plausibility`` was in 23 scripts and ZERO
files on the trading path (Sage p716), so every census this crew wrote measured
a different book from the one ``symbol_edge_gate`` refuses symbols on. Measured
p720 over the 308 rows ``_load_book``'s own SELECT returns, the censor drops 6
rows as ``repricing_return`` -- four spurious wins and two spurious losses --
and that moves the pooled verdict on 2 of 60 symbols, both STRICTER.

These tests pin the four things that can silently go wrong with a read-time
censor, in the order they would hurt:

  1. it does not fire at all (the no-op that reports a de-contaminated number
     which is the raw one),
  2. it fires on everything (this function decides every symbol refusal in
     production, so a defect here bans the universe),
  3. it needs a column an older schema lacks and raises instead of degrading,
  4. it "censors" by deleting rows from the tape.
"""

from __future__ import annotations

import importlib
import sqlite3

import pytest


SCHEMA = """
CREATE TABLE trade_outcomes (
    outcome_id INTEGER PRIMARY KEY,
    ts REAL,
    symbol TEXT,
    entry_price REAL,
    exit_price REAL,
    quantity REAL,
    gross_profit REAL,
    net_profit REAL,
    status TEXT,
    details TEXT
)
"""

# No exit_price column at all -- the schema the fallback ladder exists to survive.
SCHEMA_OLD = """
CREATE TABLE trade_outcomes (
    outcome_id INTEGER PRIMARY KEY,
    ts REAL,
    symbol TEXT,
    entry_price REAL,
    quantity REAL,
    gross_profit REAL,
    net_profit REAL,
    status TEXT,
    details TEXT
)
"""

DETAILS = '{"strategy_id": "unit_test"}'


def _ordinary(idx, symbol="FOO-USDC", ret=0.01):
    """A plausible round trip: a 1% move on a $10 notional."""
    entry, qty = 100.0, 0.1
    return (idx, 1_756_200_000.0 + idx, symbol, entry, entry * (1.0 + ret),
            qty, entry * qty * ret, entry * qty * ret, "closed", DETAILS)


def _repricing(idx, symbol="FOO-USDC", ret=1.5):
    """A repricing artifact: exit is 2.5x entry, past IMPLAUSIBLE_RET=0.50."""
    return _ordinary(idx, symbol=symbol, ret=ret)


def _gate(tmp_path, rows, schema=SCHEMA):
    """A fresh gate module pointed at a book we control."""
    db = tmp_path / "book.db"
    conn = sqlite3.connect(db)
    conn.execute(schema)
    cols = 10 if schema is SCHEMA else 9
    for row in rows:
        if cols == 9:  # drop exit_price from the tuple
            row = row[:4] + row[5:]
        conn.execute(
            "INSERT INTO trade_outcomes VALUES (%s)" % ",".join(["?"] * cols),
            row,
        )
    conn.commit()
    conn.close()

    gate = importlib.reload(importlib.import_module("services.symbol_edge_gate"))
    gate.DB_PATH = db
    return gate, db


def test_a_repricing_row_is_not_counted_by_the_gate(tmp_path):
    """THE POINT OF THE ITEM: the censor fires on the trading path."""
    rows = [_ordinary(i) for i in range(5)] + [_repricing(99)]
    gate, _ = _gate(tmp_path, rows)

    book, _pairs = gate._load_book()

    assert gate._censored_rows == 1, "the censor did not fire -- this is the no-op"
    assert len(book["FOO-USDC"]) == 5, "the repricing row reached the verdict"
    # And it is the artifact that went, not an ordinary row: a +150% return
    # would be the largest trip in the book by 150x.
    assert max(t.ret for t in book["FOO-USDC"]) == pytest.approx(0.01)


def test_the_censor_does_not_ban_the_universe(tmp_path):
    """A book of ordinary rows must survive intact. A gate that drops every row
    stops refusing anything, which reads as a pass and is a silent failure."""
    rows = [_ordinary(i) for i in range(12)]
    gate, _ = _gate(tmp_path, rows)

    book, pairs = gate._load_book()

    assert gate._censored_rows == 0
    assert len(book["FOO-USDC"]) == 12
    assert len(pairs[("unit_test", "FOO-USDC")]) == 12


def test_a_schema_without_exit_price_degrades_instead_of_raising(tmp_path):
    """Criterion the item flagged as the real work: older schemas lacking the
    column are precisely what the three-tier ladder exists to survive. Without
    exit_price the ratio arm is UNJUDGEABLE, so rows are KEPT -- the same
    direction every other rung of that ladder takes."""
    rows = [_ordinary(i) for i in range(5)] + [_repricing(99)]
    gate, _ = _gate(tmp_path, rows, schema=SCHEMA_OLD)

    book, _pairs = gate._load_book()

    assert gate._censored_rows == 0, "nothing can be judged without exit_price"
    assert len(book["FOO-USDC"]) == 6, "the ladder must not lose the book"


def test_the_censor_does_not_touch_the_stored_tape(tmp_path):
    """READ TIME ONLY. The raw tape stays auditable; the graduation report
    censors at read the same way and the two now agree by construction."""
    rows = [_ordinary(i) for i in range(5)] + [_repricing(99)]
    gate, db = _gate(tmp_path, rows)

    gate._load_book()

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0] == 6
        # including the row it refused to count
        assert conn.execute(
            "SELECT COUNT(*) FROM trade_outcomes WHERE outcome_id = 99"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_a_raising_censor_falls_back_to_the_uncensored_book(tmp_path, monkeypatch):
    """QA p721 (Sage), reopening this item against its own commit c465d1b.

    The shipped fail-open wrapped the IMPORT of ``implausible_reason`` and not
    the per-row CALL, which sat inside the ``try`` whose handler is
    ``return {}, {}``. One raise on one row therefore discarded the whole book
    -- and an empty book is not a closed gate. ``_rebuild`` iterates
    ``book.items()``, so zero rows produce zero verdicts and the cache then
    serves allow-everything. Measured against the production store before the
    guard: ``banned_symbols()`` went 8 -> 0 and AERO-USDC, BASECAT-USDC and
    COMP-USDC all read ALLOWED.

    So this pins the DIRECTION the code failed, which is the opposite of the
    one its comment guarded against.
    """
    rows = [_ordinary(i) for i in range(12)] + [_repricing(99)]
    gate, _ = _gate(tmp_path, rows)

    import services.outcome_plausibility as plausibility

    def boom(row, scales=None):
        raise ValueError("synthetic failure inside the censor")

    monkeypatch.setattr(plausibility, "implausible_reason", boom)

    book, pairs = gate._load_book()

    assert book, "a raising censor emptied the book -- that UNBANS every symbol"
    # The pre-change book is 13 rows: uncensored, which is the safe answer.
    assert len(book["FOO-USDC"]) == 13
    assert len(pairs[("unit_test", "FOO-USDC")]) == 13
    # And the operator-facing counter must not keep a stale value: the log line
    # it feeds said "dropped 6 row(s)" about a book holding zero.
    assert gate._censored_rows == 0


def test_a_raising_scales_call_also_falls_back(tmp_path, monkeypatch):
    """``strategy_scales`` is called once, outside the row loop, and was inside
    the same unguarded ``try``. Same blast radius, different line."""
    rows = [_ordinary(i) for i in range(12)]
    gate, _ = _gate(tmp_path, rows)

    import services.outcome_plausibility as plausibility

    def boom(judged):
        raise ValueError("synthetic failure building the scales")

    monkeypatch.setattr(plausibility, "strategy_scales", boom)

    book, _pairs = gate._load_book()

    assert len(book["FOO-USDC"]) == 12
    assert gate._censored_rows == 0
