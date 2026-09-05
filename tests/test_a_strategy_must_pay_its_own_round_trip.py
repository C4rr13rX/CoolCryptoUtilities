"""``services/strategy_edge_gate.py`` decides which STRATEGIES may open a trade.

It can refuse real entries, so these tests carry two burdens in equal measure:
that it fires on a strategy the book has condemned, and that it stays SILENT
otherwise. A gate that refuses everything is the same as being switched off,
and this repo has shipped that failure more than once.

The specific things that must never regress:

  * It bans on RETURN AGAINST COST, never against zero. A strategy returning
    +0.1% per round trip against a 0.65% cost is a loser, and this repo has
    shipped the compare-to-zero shape often enough that
    services/profit_logic_audit.py now lints for it.
  * It NEVER promotes. A strongly positive t is not acted on, because the
    sample is thin enough that acting on it would be fitting noise.
  * It never bans ``atf_static``, the only strategy that has ever placed a
    profitable live trade.
  * It fails OPEN. A gate that cannot read its book has no evidence to refuse
    on.
  * It judges on RETURN, not on dollars -- the ghost book's notional varies by
    more than 100x, so a t-statistic over dollars measures sizing policy.
"""
from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_book(path: Path, rows) -> None:
    """A trade_outcomes table shaped like the real one.

    ``rows`` are ``(strategy_id, net_profit, entry_price, quantity)``.
    """
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE trade_outcomes ("
        " outcome_id INTEGER PRIMARY KEY, ts REAL, status TEXT,"
        " net_profit REAL, entry_price REAL, quantity REAL, details TEXT)"
    )
    for index, (strategy, net, entry_price, quantity) in enumerate(rows):
        conn.execute(
            "INSERT INTO trade_outcomes"
            " (ts, status, net_profit, entry_price, quantity, details)"
            " VALUES (?, 'closed', ?, ?, ?, ?)",
            (1000.0 + index, net, entry_price, quantity,
             json.dumps({"strategy_id": strategy})),
        )
    conn.commit()
    conn.close()


@pytest.fixture()
def gate(tmp_path, monkeypatch):
    """The gate module pointed at a book we control, cache cleared."""
    import services.strategy_edge_gate as module
    importlib.reload(module)

    db_path = tmp_path / "book.db"
    monkeypatch.setattr(module, "DB_PATH", db_path)

    def _load(rows):
        if db_path.exists():
            db_path.unlink()
        _make_book(db_path, rows)
        module._cache.clear()
        module._cache_built_at = 0.0
        return module

    return _load


def _losing(strategy: str, n: int = 6):
    """n round trips at roughly -1.5% return, well under the 0.65% cost."""
    # entry_price * quantity = 1.0, so net_profit IS the return.
    return [(strategy, -0.015 - (i % 3) * 0.001, 1.0, 1.0) for i in range(n)]


def _winning(strategy: str, n: int = 6):
    """n round trips at roughly +4% return, comfortably above cost."""
    return [(strategy, 0.04 + (i % 3) * 0.001, 1.0, 1.0) for i in range(n)]


def test_it_refuses_a_strategy_that_cannot_pay_its_round_trip(gate):
    module = gate(_losing("obv_accumulation@1w"))
    reason = module.refusal_reason("obv_accumulation@1w")
    assert reason, "a strategy losing 1.5% per round trip must be refused"
    assert "cost" in reason, f"the reason must name the cost it failed: {reason}"


def test_it_allows_a_strategy_that_clears_its_cost(gate):
    module = gate(_winning("donchian_breakout@5d"))
    assert module.refusal_reason("donchian_breakout@5d") is None


def test_a_thin_positive_edge_is_still_refused(gate):
    """The whole point: +0.1% against a 0.65% cost is a LOSS.

    Comparing profit to zero instead of to cost is the single shape that has
    cost this repo the most money, and it is what this gate exists to catch on
    the strategy axis. A strategy that is reliably, boringly profitable in
    gross terms and reliably below its own fees must be refused.
    """
    module = gate([("thin_edge", 0.001, 1.0, 1.0) for _ in range(8)])
    reason = module.refusal_reason("thin_edge")
    assert reason, (
        "a strategy returning +0.1% against a 0.65% round trip is losing "
        "money; refusing it is the entire purpose of testing against cost "
        "rather than against zero")


def test_a_perfectly_consistent_loser_is_refused(gate):
    """Zero variance is the STRONGEST evidence, not the absence of evidence.

    A real bug, found by test 2026-09-05. Student's t divides by the standard
    deviation, so a strategy losing EXACTLY the same fraction on every round
    trip has stdev 0 and an undefined t. Returning 0.0 there -- "no evidence"
    -- waved through the most reliable loser it is possible to construct,
    while refusing a noisier strategy losing the same amount on average.

    That is precisely backwards, and it is the kind of hole that survives
    because the arithmetic looks careful.
    """
    module = gate([("clockwork_loser", -0.02, 1.0, 1.0) for _ in range(8)])
    reason = module.refusal_reason("clockwork_loser")
    assert reason, (
        "a strategy that loses 2% on every single round trip must be "
        "refused; perfect consistency is the strongest possible evidence of "
        "a negative edge, not a reason to abstain")


def test_a_consistent_break_even_is_not_evidence(gate):
    """Exactly zero excess return stays zero, even with no dispersion.

    The zero-variance branch must not turn "reliably breaks even against
    cost" into an infinite verdict in either direction.
    """
    # net return == ROUND_TRIP_COST exactly: excess is 0.0 every time.
    module = gate([("break_even", 0.0065, 1.0, 1.0) for _ in range(8)])
    assert module.refusal_reason("break_even") is None


def test_it_never_promotes_however_good_the_evidence(gate):
    """A strongly positive t must produce no verdict at all.

    The asymmetry is deliberate. Refusing a strategy we have evidence against
    costs an opportunity; acting on a positive result of the same strength
    costs money whenever that evidence was luck, and with n=3 minimum it
    often is.
    """
    module = gate(_winning("stellar", n=12))
    assert module.refusal_reason("stellar") is None
    assert "stellar" not in module.banned_strategies(), (
        "banned_strategies must contain only refusals -- a promotion list "
        "would invite a caller to trade on a thin positive sample")


def test_the_only_live_earner_is_never_banned(gate):
    """``atf_static`` is protected even when its sample turns against it.

    It is the only strategy that has ever placed a profitable live trade.
    Banning it on a ghost sample would take live trading to zero, which is
    the exact failure this module exists to prevent.
    """
    module = gate(_losing("atf_static", n=20))
    assert module.refusal_reason("atf_static") is None, (
        "atf_static is in NEVER_BAN: banning the sole live earner would stop "
        "live trading entirely")


def test_too_few_trades_is_not_a_verdict(gate):
    """Below MIN_SAMPLES the gate must stay silent, not guess."""
    module = gate(_losing("barely_traded", n=2))
    assert module.refusal_reason("barely_traded") is None


def test_it_fails_open_when_the_book_cannot_be_read(tmp_path, monkeypatch):
    """No book, no evidence, no refusal.

    Blocking every strategy because a database was locked would be far worse
    than letting one bad trade through.
    """
    import services.strategy_edge_gate as module
    importlib.reload(module)
    monkeypatch.setattr(module, "DB_PATH", tmp_path / "does-not-exist.db")
    module._cache.clear()
    module._cache_built_at = 0.0
    assert module.refusal_reason("anything") is None
    assert module.banned_strategies() == {}


def test_an_unknown_or_empty_strategy_is_allowed(gate):
    module = gate(_losing("something_bad"))
    assert module.refusal_reason("never_seen_before") is None
    assert module.refusal_reason("") is None
    assert module.refusal_reason(None) is None


def test_the_verdict_is_size_invariant(gate):
    """A 100x range of position sizes must not change the verdict.

    The ghost book is sized as a share of the stable leg, so notional within
    one strategy has ranged more than 100x. Judging on dollars would measure
    sizing policy rather than edge -- the "wrong units across a boundary"
    failure this repo keeps shipping.
    """
    # Same -1.5% return every time, but notionals spanning 100x.
    rows = []
    for index in range(8):
        notional = 0.05 * (10 ** (index % 3))   # 0.05 .. 5.0
        rows.append(("sized_wildly", -0.015 * notional, notional, 1.0))
    module = gate(rows)
    reason = module.refusal_reason("sized_wildly")
    assert reason, "a consistent -1.5% return must be caught at any size"
    assert "-1.500%" in reason, (
        f"the mean must be reported as a RETURN, not a dollar total: {reason}")


def test_a_strategy_is_judged_only_on_its_own_trades(gate):
    """One strategy's losses must not condemn another's.

    Both traded; only the loser is refused. If attribution leaked, the winner
    would be banned by association and the evidence budget would drain into
    exactly the strategies this gate is meant to protect.
    """
    module = gate(_losing("the_loser") + _winning("the_winner"))
    assert module.refusal_reason("the_loser")
    assert module.refusal_reason("the_winner") is None


def test_it_reads_every_strategy_id_key_the_writers_use(tmp_path):
    """Three executors write the id under three different keys.

    Missing one silently drops that executor's entire history, and a strategy
    with no history is one this gate cannot judge -- it would fail open on
    precisely the trades it most needs to see.
    """
    import services.strategy_edge_gate as module
    importlib.reload(module)
    db_path = tmp_path / "keys.db"
    id_keys = [
        {"strategy_id": "keyed"},
        {"strategy": "keyed"},
        {"meta": {"strategy": "keyed"}},
    ]
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE trade_outcomes ("
        " outcome_id INTEGER PRIMARY KEY, ts REAL, status TEXT,"
        " net_profit REAL, entry_price REAL, quantity REAL, details TEXT)"
    )
    for index in range(9):
        conn.execute(
            "INSERT INTO trade_outcomes"
            " (ts, status, net_profit, entry_price, quantity, details)"
            " VALUES (?, 'closed', ?, 1.0, 1.0, ?)",
            (1000.0 + index, -0.015 - (index % 3) * 0.001,
             json.dumps(id_keys[index % 3])),
        )
    conn.commit()
    conn.close()

    module.DB_PATH = db_path
    module._cache.clear()
    module._cache_built_at = 0.0
    reason = module.refusal_reason("keyed")
    assert reason, "all three id keys must be read"
    assert "9 closed" in reason, (
        f"every row should have been attributed, got: {reason}")


def test_both_entry_paths_consult_the_gate():
    """One rule, two entry paths -- and only one of them holding the line.

    trading/bot.py and services/atf_static_strategy.py open positions
    INDEPENDENTLY. When symbol_edge_gate shipped into the bot alone, the scout
    kept entering the very symbol the bot was refusing, minutes later. This
    test exists so that cannot silently happen again.
    """
    bot = (ROOT / "trading" / "bot.py").read_text(encoding="utf-8")
    scout = (ROOT / "services" / "atf_static_strategy.py").read_text(encoding="utf-8")
    assert "strategy_edge_refusal" in bot, (
        "trading/bot.py must consult the strategy edge gate")
    assert "_strategy_edge_refusal" in scout, (
        "the ATF scout opens positions on its own path and must consult the "
        "same gate, or a strategy refused in the bot keeps trading here")


def test_the_scout_can_still_exit_when_it_is_banned():
    """A condemned strategy must still close what it holds.

    Gating exits would strand every open position -- the disarming bug that
    left a demoted bot unable to sell what it had bought. The check must sit
    after the exit pass and before the entry loop.
    """
    source = (ROOT / "services" / "atf_static_strategy.py").read_text(encoding="utf-8")
    gate_at = source.index("scout_refusal = _strategy_edge_refusal")
    exit_at = source.index('status="ghost-exit"')
    entry_loop_at = source.index("for sig in signals:", gate_at - 4000)
    assert exit_at < gate_at, (
        "the exit pass must run BEFORE the strategy gate, or a banned "
        "strategy could never sell what it already bought")
    assert gate_at < entry_loop_at, (
        "the gate must run BEFORE the entry loop, or it refuses nothing")
