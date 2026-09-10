"""A dropped implausible row must be NAMED, not counted.

Item [d763940a] asks ``scripts/tradeable_book.py --symbols`` to report 0
implausible rows after the entry-basis fix, "or name the ones it still cannot
judge". The first half is not achievable by an entry guard: the guard runs when
a position opens and the report reads history, so a row already in the book
stays in it whatever the guard now says.

So the honest close is the second half. Before this change the report said only
"ALSO dropping 1 implausible row(s)" -- a number nobody can act on, which hides
whether the remaining row is one the guard now catches or one no price source
can reach. These tests hold the line that each row is named and carries a
verdict.
"""

from __future__ import annotations

import scripts.tradeable_book as tb


def _row(symbol, entry, exit_price, net, ts=1_756_200_000.0):
    return {
        "symbol": symbol, "strategy_id": "unit_test", "mode": "ghost",
        "ts": ts, "entry_price": entry, "exit_price": exit_price,
        "net": net, "gross": net, "fees": 0.0, "notional": 1.0,
        "reason": "take_profit_limit",
    }


def _report(rows):
    return tb.symbol_edge(
        rows=rows, is_tradeable=lambda s: True, now=1_756_300_000.0, days=30.0)


def test_the_implausible_row_is_named_with_its_prices():
    r = _report([_row("AERO-USDC", 0.436805, 1.140000, 3.2066)])
    assert r["implausible_rows"] == 1
    named = r["named_implausible"]
    assert len(named) == 1, "one dropped row, one named row"
    assert named[0]["symbol"] == "AERO-USDC"
    assert named[0]["entry"] == 0.436805
    assert named[0]["exit"] == 1.140000


def test_a_named_row_is_counted_once_not_once_per_bucket():
    """The analysis walks two buckets per row (per-symbol and the grid).

    Appending inside that loop without a guard names every row twice, which
    would double the list against the count it sits beside.
    """
    r = _report([_row("AERO-USDC", 0.436805, 1.140000, 3.2066)])
    assert len(r["named_implausible"]) == r["implausible_rows"]


def test_a_plausible_row_is_not_named():
    r = _report([_row("AERO-USDC", 0.50, 0.51, 0.01)])
    assert r["implausible_rows"] == 0
    assert r["named_implausible"] == []


def test_the_rendered_report_carries_a_verdict_for_each_named_row():
    r = _report([_row("AERO-USDC", 0.436805, 1.140000, 3.2066)])
    text = tb.render_symbol_edge(r)
    assert "THE IMPLAUSIBLE ROWS, NAMED" in text
    assert "AERO-USDC" in text
    assert ("WOULD BE REFUSED" in text or "CANNOT JUDGE" in text), (
        "naming a row without saying whether the guard reaches it leaves the "
        "reader exactly where the bare count did")
