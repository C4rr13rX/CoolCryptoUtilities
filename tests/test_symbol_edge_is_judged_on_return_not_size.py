"""The symbol edge gate must measure EDGE, not position SIZE.

``services/symbol_edge_gate.py`` decides which symbols may be entered at all
-- it gates ``trading/bot.py``, ``trading/swap_schedule.py`` and
``services/atf_static_strategy.py`` alike -- and it used to run its t-test on
``net_profit`` in quote units.

The ghost book is sized as a share of the stable leg, so that denominator
moves with the wallet. Measured 2026-09-04 over the 143 closed round trips in
storage/trading_cache.db, the notional WITHIN one symbol ranged
$0.0211..$3.0439 for BASECAT (144x) and $0.0119..$2.7908 for CBXRP (234x). A
mean over dollars drawn from a 144x size range is mostly a statement about
sizing policy, so a symbol traded big while flat can out-rank a symbol traded
small while genuinely bleeding.

These tests pin the three properties that fix depends on:

  1. the verdict is invariant to position size,
  2. the null hypothesis is "clears its own round-trip cost", not "above
     zero",
  3. dust closes are dropped rather than divided through.

They build their own database rather than reading the live one, so they keep
meaning something after the book moves on.
"""

from __future__ import annotations

import importlib
import sqlite3

import pytest


_book_seq = [0]


def _make_book(tmp_path, rows):
    """rows: (symbol, net_profit, entry_price, quantity).

    A fresh file per call so one test can build two books (the size-
    invariance test needs a small-notional book and a large one).
    """
    _book_seq[0] += 1
    db = tmp_path / f"trading_cache_{_book_seq[0]}.db"
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

    def _build(rows, **env):
        db = _make_book(tmp_path, rows)
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        import services.symbol_edge_gate as mod

        mod = importlib.reload(mod)
        monkeypatch.setattr(mod, "DB_PATH", db)
        mod._cache.clear()
        mod._cache_built_at = 0.0
        return mod

    return _build


def _losing(symbol, notional, n=8, ret=-0.05, spread=0.004):
    """n round trips on `symbol` averaging `ret` of `notional`.

    The returns must genuinely vary: ``_t_statistic`` returns 0.0 for a run
    of identical values (no dispersion is no evidence), so a helper emitting
    one repeated number would test nothing. ``spread`` alternates either side
    of the mean, which keeps the mean exactly `ret` for even `n` while giving
    the t-test a real standard deviation.
    """
    price = 2.0
    quantity = notional / price
    rows = []
    for i in range(n):
        this = ret + (spread if i % 2 else -spread)
        rows.append((symbol, this * notional, price, quantity))
    return rows


class TestVerdictIsSizeInvariant:
    def test_the_same_returns_ban_at_any_position_size(self, gate):
        """A 100x change in size must not change the verdict.

        This is the property the dollar test did not have: scaling every
        notional by 100 scales the dollar mean and its standard deviation
        together, but the old rule's ``mean >= 0`` screen and the reported
        total both moved, and mixing the two scales in one book -- which is
        what the real book does -- is what made the statistic meaningless.
        """
        small = gate(_losing("BAD-USDC", 0.10))
        assert small.refusal_reason("BAD-USDC")

        big = gate(_losing("BAD-USDC", 10.0))
        assert big.refusal_reason("BAD-USDC")

    def test_size_alone_cannot_create_a_ban(self, gate):
        """Big trades at a cost-clearing return stay allowed.

        Under a dollar test, a symbol traded at 100x size accumulates 100x
        the dollar swing; only the return says whether that swing was edge.
        """
        mod = gate(_losing("FINE-USDC", 10.0, n=8, ret=+0.02))
        assert mod.refusal_reason("FINE-USDC") is None


class TestCostIsTheNullHypothesis:
    def test_a_symbol_that_does_not_clear_its_cost_is_refused(self, gate):
        """+0.1% per round trip against a 0.65% cost is a loser.

        The old rule compared the mean to zero, so this symbol read as a
        winner while every trip on it lost 0.55% net.
        """
        mod = gate(_losing("THIN-USDC", 1.0, n=10, ret=+0.001))
        reason = mod.refusal_reason("THIN-USDC")
        assert reason, "a symbol below its round-trip cost must be refused"
        assert "cost" in reason

    def test_a_symbol_that_clears_its_cost_is_allowed(self, gate):
        mod = gate(_losing("GOOD-USDC", 1.0, n=10, ret=+0.05))
        assert mod.refusal_reason("GOOD-USDC") is None


class TestDustIsDroppedNotDivided:
    def test_a_dust_close_cannot_dominate_the_mean(self, gate):
        """The real book holds a 1.2e-14 notional whose -6.5e-12 is -54%.

        Dividing through it would let one rounding artifact ban a symbol
        that eight real round trips say is fine.
        """
        rows = _losing("REAL-USDC", 1.0, n=8, ret=+0.05)
        rows += [("REAL-USDC", -6.5e-12, 4.273e-07, 2.804653661e-08)] * 4
        mod = gate(rows)
        assert mod.refusal_reason("REAL-USDC") is None


class TestContractIsUnchanged:
    """Three production callers depend on this exact shape.

    trading/bot.py:6916, trading/swap_schedule.py:253 and
    services/atf_static_strategy.py:658 all use the result as a truthy
    Optional[str]; a bare bool or a dict would silently change all three.
    """

    def test_refusal_reason_returns_str_or_none(self, gate):
        mod = gate(_losing("BAD-USDC", 1.0) + _losing("OK-USDC", 1.0, ret=+0.05))
        assert isinstance(mod.refusal_reason("BAD-USDC"), str)
        assert mod.refusal_reason("OK-USDC") is None
        assert mod.refusal_reason("NEVER-TRADED-USDC") is None
        assert mod.refusal_reason("") is None

    def test_a_thin_record_is_never_judged(self, gate):
        """Below MIN_SAMPLES a losing run is indistinguishable from variance."""
        mod = gate(_losing("NEW-USDC", 1.0, n=2))
        assert mod.refusal_reason("NEW-USDC") is None

    def test_stable_legs_are_never_banned(self, gate):
        """Banning a routing leg would not avoid a bad trade, only trading."""
        mod = gate(_losing("USDC", 1.0, n=10))
        assert mod.refusal_reason("USDC") is None

    def test_an_unreadable_book_fails_open(self, gate, monkeypatch):
        """A gate that cannot read its evidence has nothing to refuse on."""
        mod = gate(_losing("BAD-USDC", 1.0))
        monkeypatch.setattr(mod, "DB_PATH", "/nonexistent/nope.db")
        mod._cache.clear()
        mod._cache_built_at = 0.0
        assert mod.refusal_reason("BAD-USDC") is None
