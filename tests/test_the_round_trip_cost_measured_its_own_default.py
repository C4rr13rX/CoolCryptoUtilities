"""The round-trip cost was a constant that had been written back into the book.

``symbol_edge_gate`` documented 0.0065 as "the measured median of
``fee_cost / notional`` over the 143 closed round trips on 2026-09-04". It
was not a measurement. Measured 2026-09-07 over the 196 rows then in
``trade_outcomes``, 105 carried a ratio of EXACTLY 0.650000%, min == max,
zero variance -- the constant echoed back. A median over a population that is
half constant still reads like a statistic, which is why nobody caught it.

The 82 rows that are real evidence had moved: real median 1.2259% on
2026-09-03/04, and p75 of the last 20 real fees 0.4738% on 2026-09-07. So the
literal was 47% too LOW during the era it claimed to measure and 37% too HIGH
afterwards. Too low passes losing symbols. Too high bans symbols that
genuinely pay -- a graduation blocker wearing a safety margin's clothes.

The load-bearing test here is ``test_a_book_of_pure_echo_is_not_evidence``:
against the old behaviour -- averaging the book without excluding the echo --
it returns the constant while claiming to have measured it. Every assertion
about the echo goes red against that.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from services import round_trip_cost as rtc


def _book(tmp_path: Path, ratios, *, notional=100.0, name="book.db") -> Path:
    """A trade_outcomes book whose rows have exactly these fee ratios."""
    path = tmp_path / name
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE trade_outcomes ("
        " outcome_id INTEGER PRIMARY KEY, ts INTEGER, entry_price REAL,"
        " quantity REAL, fee_cost REAL)")
    for index, ratio in enumerate(ratios):
        conn.execute(
            "INSERT INTO trade_outcomes (ts, entry_price, quantity, fee_cost)"
            " VALUES (?,?,?,?)",
            (1_700_000_000 + index, notional, 1.0, ratio * notional))
    conn.commit()
    conn.close()
    return path


@pytest.fixture(autouse=True)
def _clear_cache():
    rtc.reset_cache()
    yield
    rtc.reset_cache()


class TestTheEchoIsNotEvidence:
    def test_a_book_of_pure_echo_is_not_evidence(self, tmp_path):
        # 105 rows all at exactly the fallback -- the real shape of the book
        # on 2026-09-07. The old behaviour averages these and reports 0.65%
        # as a measurement. It must report the fallback AND say it is one.
        book = _book(tmp_path, [rtc.FALLBACK_COST] * 105)
        assert rtc.real_fee_ratios(db_path=book) == [], (
            "rows equal to the constant are the constant, not fees paid")
        cost, reason = rtc.measure(db_path=book)
        assert cost == pytest.approx(rtc.FALLBACK_COST)
        assert "fallback" in reason, (
            "a fallback that does not announce itself is indistinguishable "
            f"from a measurement: {reason}")

    def test_the_echo_is_dropped_but_real_fees_at_other_values_are_kept(self, tmp_path):
        book = _book(tmp_path, [rtc.FALLBACK_COST] * 20 + [0.0047] * 10)
        ratios = rtc.real_fee_ratios(db_path=book)
        assert len(ratios) == 10
        assert all(r == pytest.approx(0.0047) for r in ratios)

    def test_a_real_fee_that_merely_rounds_near_the_constant_is_kept(self, tmp_path):
        # The filter is exact to 1e-9, not a tolerance band -- a 0.6501% fee
        # is a fee somebody paid.
        book = _book(tmp_path, [0.006501] * 12)
        assert len(rtc.real_fee_ratios(db_path=book)) == 12

    def test_a_mixed_book_measures_the_real_rows_not_the_blend(self, tmp_path):
        # The bug in one line: blending 100 echoes with 20 real 0.0047 fees
        # gives ~0.0062 and calls the cost nearly unchanged. The truth is
        # 0.0047.
        book = _book(tmp_path, [rtc.FALLBACK_COST] * 100 + [0.0047] * 20)
        cost, reason = rtc.measure(db_path=book)
        assert cost == pytest.approx(0.0047)
        assert cost < 0.006, f"the echo leaked into the measurement: {reason}"


class TestItStaysConservative:
    def test_it_charges_the_upper_quantile_not_the_middle(self, tmp_path):
        # Understating cost is the direction that loses money.
        book = _book(tmp_path, [0.001] * 10 + [0.009] * 10)
        cost, _ = rtc.measure(db_path=book)
        assert cost == pytest.approx(0.009), (
            "p75 of a half-and-half window is the upper value")

    def test_thin_evidence_falls_back_rather_than_inventing_a_number(self, tmp_path):
        book = _book(tmp_path, [0.0001] * (rtc.MIN_SAMPLES - 1))
        cost, reason = rtc.measure(db_path=book)
        assert cost == pytest.approx(rtc.FALLBACK_COST)
        assert "need" in reason and "fallback" in reason

    def test_enough_evidence_is_believed(self, tmp_path):
        book = _book(tmp_path, [0.0047] * rtc.MIN_SAMPLES)
        cost, reason = rtc.measure(db_path=book)
        assert cost == pytest.approx(0.0047), reason

    def test_a_dust_notional_cannot_set_the_bar(self, tmp_path):
        # The book holds a round trip whose 1.2e-14 notional reads as -54%.
        # One of those must not be able to ban the entire universe.
        book = _book(tmp_path, [0.0047] * 12, notional=100.0)
        conn = sqlite3.connect(book)
        conn.execute(
            "INSERT INTO trade_outcomes (ts, entry_price, quantity, fee_cost)"
            " VALUES (?,?,?,?)", (1_800_000_000, 1e-14, 1.0, 5e-15))
        conn.commit()
        conn.close()
        ratios = rtc.real_fee_ratios(db_path=book)
        assert len(ratios) == 12, "the dust row was divided through anyway"

    def test_an_absurd_ratio_is_a_broken_row_not_an_expensive_trade(self, tmp_path):
        # The book holds a row at 54237%.
        book = _book(tmp_path, [0.0047] * 12 + [542.37])
        assert len(rtc.real_fee_ratios(db_path=book)) == 12

    def test_a_measurement_outside_the_hard_bounds_is_refused(self, tmp_path):
        # A malformed book must not be able to switch the profitability bar
        # off by measuring a cost of nearly zero.
        book = _book(tmp_path, [1e-9] * 20)
        cost, reason = rtc.measure(db_path=book)
        assert cost == pytest.approx(rtc.FALLBACK_COST)
        assert "malformed" in reason

    def test_a_missing_book_falls_back_instead_of_raising(self, tmp_path):
        cost, reason = rtc.measure(db_path=tmp_path / "nope.db")
        assert cost == pytest.approx(rtc.FALLBACK_COST)
        assert "fallback" in reason

    def test_the_window_is_recent_not_lifetime(self, tmp_path):
        # Gas from four days ago must not set today's bar. Old rows at 1.2%
        # followed by a full window at 0.0047 must measure 0.0047.
        book = _book(tmp_path, [0.0123] * 60 + [0.0047] * rtc.WINDOW)
        cost, _ = rtc.measure(db_path=book)
        assert cost == pytest.approx(0.0047)


class TestTheGateAsksTheMeasuredCost:
    def test_the_verdict_moves_when_the_measured_cost_moves(self, monkeypatch):
        # Behavioural, not a source scan: a book of steady +0.55% returns is
        # BELOW the stale 0.650% literal and ABOVE the measured 0.4738%. The
        # old code bans it for paying its way; the new code does not.
        from services import symbol_edge_gate as gate

        # _verdict takes Trips, not bare returns: it needs the notional to
        # weigh a book, and every production caller builds them the same way
        # (symbol_edge_gate:546/:581, scripts/tradeable_book.py:544). Passing
        # floats here raised AttributeError and read as a source bug for
        # several passes -- it was only ever this fixture lagging the signature.
        returns = [gate.Trip(ret=0.0055, gross=0.55, notional=100.0)] * 12
        monkeypatch.setattr(gate, "round_trip_cost",
                            lambda **_: 0.0065, raising=True)
        stale = gate._verdict(returns)
        monkeypatch.setattr(gate, "round_trip_cost",
                            lambda **_: 0.004738, raising=True)
        honest = gate._verdict(returns)

        assert stale is not None, (
            "premise: +0.550% per round trip is judged against a 0.650% cost")
        assert honest is None, (
            "a symbol returning +0.550% clears the measured 0.4738% round "
            f"trip and must not be judged; got {honest}")

    def test_the_verdict_does_not_read_the_stale_literal(self):
        # If the accessor is bypassed, the test above cannot fail, so the
        # source property still has to be asserted.
        import inspect

        from services import symbol_edge_gate as gate

        parts = inspect.getsource(gate).split("def _verdict", 1)
        assert len(parts) == 2, "the verdict function was renamed"
        body = parts[1].split("\ndef ", 1)[0]
        assert "round_trip_cost(" in body, (
            "the verdict must measure the cost, not read the fallback literal")
        assert "ROUND_TRIP_COST" not in body, (
            "a literal cost read survives inside the verdict")

    def test_the_cost_is_bound_once_per_verdict(self):
        import inspect

        from services import symbol_edge_gate as gate

        body = inspect.getsource(gate).split("def _verdict", 1)[1]
        body = body.split("\ndef ", 1)[0]
        assert body.count("round_trip_cost(") == 1, (
            "calling the accessor per clause lets a cache expiry compare the "
            "mean against one cost and the sign test against another")

    def test_the_cost_follows_the_book_the_gate_was_pointed_at(self, tmp_path):
        # A caller that redirects DB_PATH to a throwaway book must not be
        # charged the production book's fee. This is the coupling that broke
        # test_a_return_exactly_at_cost_is_not_a_win.
        import inspect

        from services import symbol_edge_gate as gate

        body = inspect.getsource(gate).split("def _verdict", 1)[1]
        body = body.split("\ndef ", 1)[0]
        assert "db_path=DB_PATH" in body, (
            "the cost must be read from the same book the verdict is reading")

    def test_the_fallback_is_read_at_call_time_not_at_import(self, monkeypatch):
        monkeypatch.setenv("SYMBOL_EDGE_ROUND_TRIP_COST", "0.0123")
        assert rtc.fallback_cost() == pytest.approx(0.0123), (
            "an env-configurable constant frozen at import ignores every "
            "caller that sets it afterwards")

    def test_two_books_do_not_share_one_cached_cost(self, tmp_path):
        cheap = _book(tmp_path, [0.0020] * 20, name="cheap.db")
        dear = _book(tmp_path, [0.0090] * 20, name="dear.db")
        assert rtc.round_trip_cost(db_path=cheap) == pytest.approx(0.0020)
        assert rtc.round_trip_cost(db_path=dear) == pytest.approx(0.0090), (
            "the second book was served the first book's cached answer")

    def test_the_fallback_constant_still_exists_for_a_thin_book(self):
        from services import symbol_edge_gate as gate

        assert gate.ROUND_TRIP_COST > 0


class TestUnitsAtTheBoundary:
    def test_the_cost_is_a_fraction_not_a_percentage(self, tmp_path):
        book = _book(tmp_path, [0.0047] * 20)
        cost, _ = rtc.measure(db_path=book)
        assert 0.0 < cost < 0.1, (
            f"{cost} is not a fraction of notional; 0.47% is 0.0047, not 0.47")

    def test_the_real_book_measures_below_the_stale_literal(self):
        # The live claim of this pass, asserted against the real book so it
        # cannot silently stop being true.
        ratios = rtc.real_fee_ratios()
        if len(ratios) < rtc.MIN_SAMPLES:
            pytest.skip("book has too few real fees on this machine")
        cost, reason = rtc.measure()
        assert cost < rtc.FALLBACK_COST, (
            f"the measured cost {cost:.4%} is no longer below the stale "
            f"{rtc.FALLBACK_COST:.4%} literal -- {reason}")
