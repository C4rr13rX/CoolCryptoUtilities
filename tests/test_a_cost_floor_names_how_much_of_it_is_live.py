"""A charged round-trip cost must say how many of its samples were LIVE.

``services.round_trip_cost`` reads ``trade_outcomes``, drops the echo rows and
the sub-``MIN_NOTIONAL`` rows, and charges p75 of the last ``WINDOW`` survivors
to every profitability gate in the system. It never said where those survivors
came from, and measured 2026-09-17 against the production book that mattered::

    real fees in the book : 123  (18 live / 105 ghost)
    the window that sets the cost: 0 live / 20 ghost
    live  median 0.5048%  p75 0.8647%  n=18
    ghost median 0.4893%  p75 1.2259%  n=105
    round trip charged    : 0.4613%

Eighteen live fee measurements sat in the book and NONE of them were in the
window that set the price, because ghost rows are written about ten times more
often than live ones and the window is chosen by recency alone.

THE FIRST VERSION OF THIS DOCSTRING SAID THAT WAS COSTING MONEY. It is not,
and the claim is withdrawn. Measured under each candidate selection rule,
every live-inclusive one charges the SAME number::

    last 20 by recency (today)      0.4613%
    last 10 live + last 10 ghost    0.4613%    identical
    last 40 by recency              0.4652%    +0.83%
    all 123 rows                    1.2259%    +166%, the stale gas regime

and the two populations are statistically indistinguishable -- Mann-Whitney
U=719.0, p=0.1065 over n=18 live against 105 ghost. A p75 over 18 rows is one
or two rows, not a regime.

The count is therefore here for the DIVERGENCE case: the exclusion is
structural, so on the day live fees part company with ghost fees the charged
cost will not notice and nobody will be told.

The item [297f47fd] claimed all 122 real fees were ghost. That is false: 18
are live. The defect is narrower and worse -- the live rows exist, carry real
fees, and are structurally excluded from the only window anybody charges.

These tests fail against a body whose cost carries no provenance.
"""

from __future__ import annotations

import importlib
import json
import sqlite3

import pytest


@pytest.fixture()
def cost():
    mod = importlib.import_module("services.round_trip_cost")
    mod.reset_cache()
    yield mod
    mod.reset_cache()


def _book(tmp_path, rows):
    """Write a ``trade_outcomes`` book. Each row: (notional, ratio, mode)."""
    path = tmp_path / "book.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE trade_outcomes (ts REAL, entry_price REAL, "
        "quantity REAL, fee_cost REAL, details TEXT)"
    )
    for index, (notional, ratio, mode) in enumerate(rows):
        details = None if mode is None else json.dumps({"mode": mode})
        conn.execute(
            "INSERT INTO trade_outcomes VALUES (?,?,?,?,?)",
            (float(index), float(notional), 1.0, notional * ratio, details),
        )
    conn.commit()
    conn.close()
    return path


def test_a_sample_carries_whether_it_was_live_or_ghost(cost, tmp_path):
    path = _book(tmp_path, [
        (1.0, 0.004, "live_exit"),
        (1.0, 0.005, "ghost_exit"),
        (1.0, 0.006, "live"),
    ])
    samples = cost.fee_ratio_samples(db_path=path)
    assert [kind for _, kind in samples] == ["live", "ghost", "live"]
    assert cost.provenance_counts(samples) == (2, 1)


def test_a_row_that_cannot_prove_it_was_live_is_counted_ghost(cost, tmp_path):
    """Unreadable, absent and non-dict details must never inflate LIVE.

    The count exists to say how much of the cost floor rests on fills that
    actually paid gas. A row that cannot prove it paid gas is not evidence
    that it did.
    """
    path = _book(tmp_path, [
        (1.0, 0.004, None),          # no details at all
        (1.0, 0.005, "ghost_exit"),
        (1.0, 0.006, "live_entry"),
    ])
    conn = sqlite3.connect(str(path))
    conn.execute("INSERT INTO trade_outcomes VALUES (9,1.0,1.0,0.007,'not json')")
    conn.execute("INSERT INTO trade_outcomes VALUES (10,1.0,1.0,0.008,'[1,2]')")
    conn.commit()
    conn.close()

    live, ghost = cost.provenance_counts(cost.fee_ratio_samples(db_path=path))
    assert (live, ghost) == (1, 4)


def test_real_fee_ratios_still_returns_the_bare_floats(cost, tmp_path):
    """The provenance must not break the callers that only want numbers."""
    path = _book(tmp_path, [
        (1.0, 0.004, "live_exit"),
        (1.0, 0.005, "ghost_exit"),
    ])
    ratios = cost.real_fee_ratios(db_path=path)
    assert ratios == [pytest.approx(0.004), pytest.approx(0.005)]
    assert all(isinstance(value, float) for value in ratios)


def test_the_charged_cost_names_its_live_and_ghost_count(cost, tmp_path):
    """The count must be written beside the cost, not merely be available."""
    rows = [(1.0, 0.004 + index * 1e-5, "ghost_exit") for index in range(20)]
    rows.append((1.0, 0.009, "live_exit"))
    path = _book(tmp_path, rows)

    _, reason = cost.measure(db_path=path)
    assert "1 live / 19 ghost" in reason, reason


def test_a_window_with_no_live_evidence_says_zero_live(cost, tmp_path):
    """The production case: 18 live fees in the book, 0 in the window.

    This is the one that matters. A window of pure ghost rows must announce
    itself, because the charged number looks identical either way.
    """
    rows = [(1.0, 0.0086, "live_exit") for _ in range(18)]
    rows += [(1.0, 0.0046, "ghost_exit") for _ in range(20)]
    path = _book(tmp_path, rows)

    samples = cost.fee_ratio_samples(db_path=path)
    assert cost.provenance_counts(samples) == (18, 20)

    charged, reason = cost.measure(db_path=path)
    assert "0 live / 20 ghost" in reason, reason
    # and the cost it charged is the ghost number, not the live one
    assert charged == pytest.approx(0.0046, abs=1e-6)


def test_the_fallback_paths_carry_the_count_too(cost, tmp_path):
    """A cost with no provenance is how the window went unnoticed."""
    path = _book(tmp_path, [(1.0, 0.004, "live_exit")])
    charged, reason = cost.measure(db_path=path)
    assert charged == pytest.approx(cost.fallback_cost())
    assert "1 live / 0 ghost" in reason, reason


def test_a_book_with_no_details_column_still_prices(cost, tmp_path):
    """The provenance must not be able to break the COST.

    Adding ``details`` to the SELECT broke ten existing tests at once: books
    written without that column raised, the except returned [], and [] does
    not read as "no provenance" -- it reads as "no fees in this book" and
    drops every gate silently onto the 0.65% fallback constant. The cost is
    the old guarantee and the provenance is the new one; the old one must
    not depend on the new one being available.
    """
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE trade_outcomes (ts REAL, entry_price REAL, "
        "quantity REAL, fee_cost REAL)"
    )
    for index in range(20):
        conn.execute("INSERT INTO trade_outcomes VALUES (?,1.0,1.0,?)",
                     (float(index), 0.004 + index * 1e-5))
    conn.commit()
    conn.close()

    samples = cost.fee_ratio_samples(db_path=path)
    assert len(samples) == 20
    # unknown provenance is ghost -- it must never inflate the live count
    assert cost.provenance_counts(samples) == (0, 20)

    charged, reason = cost.measure(db_path=path)
    assert charged != pytest.approx(cost.fallback_cost()), (
        "a legacy book must still be MEASURED, not dropped to the fallback")
    assert "0 live / 20 ghost" in reason, reason


def test_a_missing_book_still_does_not_raise(cost, tmp_path):
    """A cost read must never take the money path down with it."""
    assert cost.fee_ratio_samples(db_path=tmp_path / "nope.db") == []
    assert cost.real_fee_ratios(db_path=tmp_path / "nope.db") == []
