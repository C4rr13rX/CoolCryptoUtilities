"""The symbol edge gate must charge each round trip at ITS OWN clip.

``services/symbol_edge_gate.py`` bans a symbol whose book does not pay for its
own round trips, and it gates every live entry site
(``trading/bot.py``, ``trading/swap_schedule.py``,
``services/atf_static_strategy.py``) plus the ledger's
``_live_tradeable`` -- so what it charges decides both which symbols may be
entered and which ghost trades count as evidence toward graduation.

THE DEFECT THESE PIN. ``_verdict`` ran three legs. The third costed each trip
with ``modelled_cost(notional)`` -- fixed charge plus rate -- while the mean
screen, the t-test and the sign test subtracted ONE flat
``round_trip_cost()`` fraction from every return regardless of size. That is
the collapse ``modelled_cost``'s own docstring names:

    "collapsing the two into one rate prices a $0.50 trip and a $20 trip
     identically and is how a book that 'clears the variable floor' still
     loses money."

At a $0.20 clip the FIXED charge alone is 2.02% of notional. A symbol
returning +1.0% per round trip there loses money on every single trip, and
the flat 0.65% rate read it as clearing its cost.

DIRECTION. This is not a loosening. Measured 2026-09-23 over the 263 trips
the live book yields (median notional $2.00), the per-trip cost is 0.521% at
the median against a 0.465% flat rate: 61.2% of real trips are charged MORE
after this change than before it.

These tests build their own database rather than reading the live one, so
they keep meaning something after the book moves on.
"""

from __future__ import annotations

import importlib
import sqlite3

import pytest


_book_seq = [0]


def _make_book(tmp_path, rows):
    """rows: (symbol, net_profit, entry_price, quantity)."""
    _book_seq[0] += 1
    db = tmp_path / f"clip_book_{_book_seq[0]}.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE trade_outcomes ("
        " outcome_id TEXT, ts REAL, symbol TEXT, entry_price REAL,"
        " quantity REAL, net_profit REAL, status TEXT)"
    )
    for i, (symbol, net, entry_price, quantity) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_outcomes VALUES (?,?,?,?,?,?,'closed')",
            (f"o{i}", 1788500000.0 + i, symbol, entry_price, quantity, net),
        )
    conn.commit()
    conn.close()
    return db


@pytest.fixture()
def gate(tmp_path, monkeypatch):
    """Fresh module bound to a throwaway book, with the cache defeated."""

    def _build(rows):
        db = _make_book(tmp_path, rows)
        import services.symbol_edge_gate as mod

        mod = importlib.reload(mod)
        monkeypatch.setattr(mod, "DB_PATH", db)
        mod._cache.clear()
        mod._pair_cache.clear()
        mod._cache_built_at = 0.0
        return mod

    return _build


def _book(symbol, notional, n=10, ret=0.010, spread=0.004):
    """``n`` round trips on ``symbol`` returning ``ret`` of ``notional``.

    The returns must genuinely vary -- ``_t_statistic`` returns 0.0 for a run
    of identical values -- so ``spread`` alternates either side of the mean,
    keeping the mean exactly ``ret`` for even ``n``.
    """
    price = 2.0
    quantity = notional / price
    return [
        (symbol, (ret + (spread if i % 2 else -spread)) * notional, price, quantity)
        for i in range(n)
    ]


class TestTheFixedChargeIsPaidAtTheClipItWasPaidAt:
    def test_a_dust_clip_earning_above_the_flat_rate_is_still_refused(self, gate):
        """+1.0% per trip at a $0.20 clip is a loss on every trip.

        The fixed charge is $0.004047, which is 2.02% of a $0.20 notional, so
        the true cost there is 2.34% -- the symbol loses 1.34% per round trip.
        Against ONE flat 0.65% rate the mean read as clearing its cost and the
        gate returned before running a single test. This assertion is what the
        flat rate could not make.
        """
        mod = gate(_book("DUST-USDC", 0.20, n=10, ret=+0.010))
        reason = mod.refusal_reason("DUST-USDC")
        assert reason, (
            "a symbol whose fixed charge alone is 2.02% of its notional must "
            "not read as clearing a 0.65% cost"
        )
        assert "cost" in reason

    def test_the_same_return_at_a_fundable_clip_is_allowed(self, gate):
        """The counterpart, so this is a per-clip judgement and not a blanket ban.

        Identical returns at a $20 clip cost 0.339% -- the fixed charge is
        0.02% of notional there -- so +1.0% genuinely clears and must pass.
        """
        mod = gate(_book("FUND-USDC", 20.0, n=10, ret=+0.010))
        assert mod.refusal_reason("FUND-USDC") is None

    def test_the_reason_quotes_the_clip_it_charged(self, gate):
        """A refusal naming a cost the trips did not pay cannot be acted on.

        The old message quoted the flat rate whatever the book's clips were.
        """
        mod = gate(_book("DUST-USDC", 0.20, n=10, ret=+0.010))
        reason = mod.refusal_reason("DUST-USDC")
        assert "at their own clips" in reason


class TestTheExistingGuaranteesSurvive:
    """The properties test_symbol_edge_is_judged_on_return_not_size pins.

    Re-asserted here against the per-clip cost, because a cost that varies
    with size is the one change that could plausibly have broken them.
    """

    def test_a_losing_return_bans_at_any_clip(self, gate):
        for notional in (0.10, 10.0):
            mod = gate(_book("BAD-USDC", notional, n=8, ret=-0.05))
            assert mod.refusal_reason("BAD-USDC"), notional

    def test_a_thin_record_is_never_judged(self, gate):
        mod = gate(_book("NEW-USDC", 0.20, n=2, ret=+0.010))
        assert mod.refusal_reason("NEW-USDC") is None

    def test_stable_legs_are_never_banned(self, gate):
        mod = gate(_book("USDC", 0.20, n=10, ret=+0.010))
        assert mod.refusal_reason("USDC") is None
