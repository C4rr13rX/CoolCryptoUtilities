"""A symbol that loses erratically must not outrun the test that judges it.

``symbol_edge_gate`` bans on ``t < -1.7`` of (return - cost). That statistic
puts the symbol's own dispersion in the denominator, so a symbol that loses
steadily AND erratically inflates the number that is supposed to convict it.

MEASURED 2026-09-05 over 163 closed round trips in storage/trading_cache.db:

    COMP-USDC   16 trades   mean return -4.333%   t=-1.44   NOT banned
                            1 of 16 round trips cleared the 0.650% cost
                            -1.2368 realised, second only to BASECAT

Sixteen samples and a five-point-per-trade shortfall, walking through a
threshold built to catch precisely that. The repair is a second reading of
the same question with no dispersion in it: of n round trips, how many
cleared the cost? Under "this symbol pays for its own trading" that count is
Binomial(n, 0.5), and 1-of-16 is p=0.000259.

THE ORDERING IS THE SAFETY ARGUMENT, AND IT IS WHAT THESE TESTS EXIST FOR.
A sign test applied on its own bans the best symbol in the book. AERO-USDC
clears cost on 3 of its 38 round trips -- p=0.0000, a more extreme count
than COMP's -- while returning +2.350% per trade and +1.97 in total, because
its payoff is carried by rare large wins. CBBTC-USDC is 1 of 10 and positive
the same way. Ranking those as losers would invert the book.

So the sign test is only ever reached for a symbol whose MEAN is already
below cost. It decides how sure we are that a loser is a loser; it can never
overturn a positive mean. ``test_a_rare_win_payoff_is_never_banned`` is the
test that fails if that ordering is ever swapped.

Validated out of sample on the module's own 60/40 split, scoring the
untouched holdout:

    t-test only          bans BASECAT, CBXRP        +1.2030 -> +1.3038
    t-test + sign test   bans BASECAT, CBXRP, COMP  +1.2030 -> +1.3983

These build their own book rather than reading the live one, so they keep
meaning something after the real book moves on.
"""

from __future__ import annotations

import importlib
import sqlite3

import pytest


COST = 0.0065

_book_seq = [0]


def _make_book(tmp_path, rows):
    """rows: (symbol, net_profit, entry_price, quantity)."""
    _book_seq[0] += 1
    db = tmp_path / f"edge_book_{_book_seq[0]}.db"
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
    """Fresh module bound to a throwaway book, with the verdict cache defeated."""

    def _build(rows, **env):
        db = _make_book(tmp_path, rows)
        env.setdefault("SYMBOL_EDGE_ROUND_TRIP_COST", COST)
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        import services.symbol_edge_gate as mod

        mod = importlib.reload(mod)
        monkeypatch.setattr(mod, "DB_PATH", db)
        mod._cache.clear()
        mod._cache_built_at = 0.0
        return mod

    return _build


def _rows(symbol, returns, price=2.0, notional=1.0):
    """One row per return, all at the same notional so return == net/notional."""
    quantity = notional / price
    return [(symbol, r * notional, price, quantity) for r in returns]


# The measured COMP-USDC shape: one large win carrying the dispersion, a long
# tail of losses, mean well below cost, t inside -1.7.
_COMP = [0.40] + [-0.07 + (0.001 if i % 2 else -0.001) for i in range(15)]

# The measured AERO-USDC shape: three large wins, thirty-five small losses,
# mean comfortably above cost, sign count MORE extreme than COMP's.
_AERO = [0.30, 0.29, 0.31] + [-0.005 + (0.001 if i % 2 else -0.001) for i in range(35)]


def test_the_t_test_alone_misses_this_shape(gate) -> None:
    """Guards the premise: without the sign test, this symbol walks free."""
    mod = gate(_rows("COMP-USDC", _COMP), SYMBOL_EDGE_SIGN_MAX_P="0.0")
    assert mod.refusal_reason("COMP-USDC") is None


def test_a_steady_loser_is_banned_by_the_sign_test(gate) -> None:
    mod = gate(_rows("COMP-USDC", _COMP))
    reason = mod.refusal_reason("COMP-USDC")
    assert reason is not None
    assert "sign test" in reason
    assert "only 1 cleared it" in reason


def test_a_rare_win_payoff_is_never_banned(gate) -> None:
    """The safety property. AERO's sign count is more extreme than COMP's."""
    mod = gate(_rows("AERO-USDC", _AERO))
    assert mod._sign_test_p(_AERO, COST) < mod._sign_test_p(_COMP, COST)
    assert mod.refusal_reason("AERO-USDC") is None


def test_the_mean_check_runs_before_either_test(gate) -> None:
    """Both shapes in one book: only the one losing on average is refused."""
    mod = gate(_rows("COMP-USDC", _COMP) + _rows("AERO-USDC", _AERO))
    assert set(mod.banned_symbols()) == {"COMP-USDC"}


#: Four round trips landing exactly ON the cost and four losing, one of them
#: badly enough to keep the t-statistic at -1.15. Whether this symbol is
#: refused depends entirely on which side of the boundary a tie falls: counted
#: as wins it is 4-of-8 (p=0.637, allowed), counted as ties it is 0-of-8
#: (p=0.0039, refused). Nothing else in the gate separates the two readings.
_TIED = [COST, COST, COST, COST, -0.60, -0.02, -0.02, -0.02]


def test_a_return_exactly_at_cost_is_not_a_win(gate) -> None:
    """A round trip that exactly repays its cost made nothing, so it is not
    evidence that the symbol pays. Ties fall on the not-clearing side."""
    mod = gate(_rows("TIED-USDC", _TIED))
    # The premise: the t-test is quiet here, so the verdict is the sign test's.
    assert mod._t_statistic([r - COST for r in _TIED]) > mod.MAX_T
    assert mod._sign_test_p(_TIED, COST) == pytest.approx(0.5 ** 8)
    assert mod.refusal_reason("TIED-USDC") is not None
    # And had ties counted as wins it would have read 4-of-8 and walked.
    assert mod._sign_test_p([r + 1e-9 for r in _TIED], COST) > 0.5


def test_the_statistic_is_the_exact_binomial(gate) -> None:
    mod = gate(_rows("X-USDC", [-0.05] * 6))
    assert mod._sign_test_p([-1.0] * 6, COST) == pytest.approx(0.015625)
    assert mod._sign_test_p([1.0] + [-1.0] * 15, COST) == pytest.approx(0.000259, rel=1e-2)
    assert mod._sign_test_p([1.0] * 8 + [-1.0] * 8, COST) == pytest.approx(0.598, rel=1e-2)


def test_one_trade_is_never_evidence(gate) -> None:
    """No dispersion and no count -- both tests must decline to convict."""
    mod = gate(_rows("ONCE-USDC", [-0.90]))
    assert mod._sign_test_p([-0.90], COST) == 1.0
    assert mod.refusal_reason("ONCE-USDC") is None


def test_too_few_samples_is_still_too_few(gate) -> None:
    """The sign test does not smuggle a verdict past MIN_SAMPLES: four straight
    losses are p=0.0625, but four trades is not a judgement either way."""
    mod = gate(_rows("THIN-USDC", [-0.30, -0.28, -0.31, -0.29]))
    assert mod.refusal_reason("THIN-USDC") is None


def test_an_unreadable_book_bans_nothing(gate, tmp_path) -> None:
    """Fails OPEN, unchanged: a gate that cannot read its evidence has none."""
    mod = gate(_rows("COMP-USDC", _COMP))
    mod.DB_PATH = tmp_path / "does_not_exist.db"
    mod._cache.clear()
    mod._cache_built_at = 0.0
    assert mod.refusal_reason("COMP-USDC") is None
