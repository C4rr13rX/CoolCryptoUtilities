"""``symbol_edge_gate`` pooled every executor into one verdict per symbol.

Measured 2026-09-07 over the 191-row ``trade_outcomes`` book:

    AERO-USDC   pooled       n=46  mean +1.805%   ALLOWED (clears the cost)
    AERO-USDC   atf_static   n=17  mean -0.992%   t=-6.24
    CBBTC-USDC  atf_static   n= 6  mean -1.077%   t=-4.73

The pooled mean is carried by 18 rows from a different executor at +5.736%.
``atf_static`` is the ONLY strategy in the registry with a live branch, and
AERO-USDC was 6 of the 9 round trips in its post-demotion re-arm window
(2 wins, -0.2232). So the gate whose entire purpose is "do not re-trade a
symbol the book has proven we lose on" was, every cycle, waving the live lane
back into the one symbol it had the most decisive evidence against -- because
a *different* executor was good at it.

A directive is always a ``(strategy, symbol)`` pair. The pooled book answers
for the symbol alone, which is a question no entry site ever asks.

The tests below fail against the pooled-only gate: `_pooled_verdict_allows`
pins the premise, and the pair verdict it must now produce did not exist.

The rule BANS ONLY, and the pooled verdict is checked first and never
overturned -- a good record on one slice is exactly the noise-fitting the
module docstring refuses to act on.

Validated out of sample on the 92 attributed rows, fitted on the first 55 and
applied to the untouched 37: holdout net +0.2177 -> +0.2400 (+0.0223), and it
removed no winning round trip.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

COST = 0.0065

#: A payoff shape that clears cost on average: three large wins carrying a
#: tail of small losses. This is the AERO-USDC shape the gate deliberately
#: does not ban, and it is what one executor contributes to the pool.
_WINNER = [0.30, 0.29, 0.31] + [-0.004 + (0.001 if i % 2 else -0.001) for i in range(9)]

#: A steady loser at the same symbol: below cost every time, with enough
#: dispersion for the t-statistic to be real rather than a divide-by-zero.
_LOSER = [-0.010 + (0.002 if i % 2 else -0.002) for i in range(12)]


def _make_book(tmp_path: Path, rows) -> Path:
    """A trade_outcomes book WITH a details column, as production has."""
    db = tmp_path / "book.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE trade_outcomes ("
        " outcome_id TEXT, ts REAL, symbol TEXT, entry_price REAL,"
        " quantity REAL, net_profit REAL, status TEXT, details TEXT)"
    )
    for i, (symbol, strategy, net, entry_price, quantity) in enumerate(rows):
        details = json.dumps({"strategy_id": strategy}) if strategy else None
        conn.execute(
            "INSERT INTO trade_outcomes VALUES (?,?,?,?,?,?,'closed',?)",
            (f"o{i}", 1788500000.0 + i, symbol, entry_price, quantity, net, details),
        )
    conn.commit()
    conn.close()
    return db


def _rows(symbol, strategy, returns, price=2.0, notional=1.0):
    """One row per return, all at the same notional so return == net/notional."""
    quantity = notional / price
    return [(symbol, strategy, r * notional, price, quantity) for r in returns]


@pytest.fixture()
def gate(tmp_path, monkeypatch):
    """Fresh module bound to a throwaway book, with the verdict cache defeated."""

    def _build(rows, **env):
        db = _make_book(tmp_path, rows)
        env.setdefault("SYMBOL_EDGE_ROUND_TRIP_COST", str(COST))
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        import services.symbol_edge_gate as mod

        mod = importlib.reload(mod)
        monkeypatch.setattr(mod, "DB_PATH", db)
        mod._cache.clear()
        mod._pair_cache.clear()
        mod._pair_seen.clear()
        mod._cache_built_at = 0.0
        return mod

    return _build


#: The measured shape, in miniature: one symbol, two executors, opposite records.
_MIXED = (
    _rows("AERO-USDC", "scout", _WINNER)
    + _rows("AERO-USDC", "atf_static", _LOSER)
)


def test_the_pooled_verdict_allows_it(gate) -> None:
    """The premise. Without this the rest proves nothing.

    Pooled, the winner's three large wins drag the mean above cost, so the
    gate's very first check short-circuits and NEITHER test runs. This is the
    old behaviour, and it is the whole bug.
    """
    mod = gate(_MIXED)
    assert mod.refusal_reason("AERO-USDC") is None
    assert "AERO-USDC" not in mod.banned_symbols()


def test_the_executor_that_loses_on_it_is_refused(gate) -> None:
    mod = gate(_MIXED)
    reason = mod.refusal_reason("AERO-USDC", "atf_static")
    assert reason is not None, (
        "the executor with a measured negative edge on this symbol must be "
        "refused even though the pooled book clears cost"
    )
    assert "atf_static" in reason
    assert ("atf_static", "AERO-USDC") in mod.banned_pairs()


def test_the_executor_that_pays_on_it_is_not(gate) -> None:
    """BANS ONLY. The good record is not punished by the bad one either."""
    mod = gate(_MIXED)
    assert mod.refusal_reason("AERO-USDC", "scout") is None
    assert ("scout", "AERO-USDC") not in mod.banned_pairs()


def test_omitting_the_strategy_id_is_the_old_answer(gate) -> None:
    """Every existing call site keeps working, and keeps its old verdict.

    The parameter can only ever tighten: a caller that cannot name an
    executor gets exactly the pooled result it got before.
    """
    mod = gate(_MIXED)
    assert mod.refusal_reason("AERO-USDC") is None
    assert mod.refusal_reason("AERO-USDC", None) is None
    assert mod.refusal_reason("AERO-USDC", "") is None


def test_a_pooled_ban_is_never_overturned_by_a_good_slice(gate) -> None:
    """The safety property, and the reason the pooled check runs first.

    A symbol the whole book condemns stays condemned however well one
    executor happens to have done on it. Acting on a positive record over a
    slice is precisely the noise-fitting this module refuses to do -- and a
    slice is smaller than the pool, so it is noisier, not less.
    """
    # `lucky` clears cost on its own (mean +0.800% against 0.650%) and would
    # be ALLOWED if it were judged alone; the pool is -0.690% at t=-10.06.
    rows = (
        _rows("BASECAT-USDC", "loser_a", _LOSER)
        + _rows("BASECAT-USDC", "loser_b", _LOSER)
        + _rows("BASECAT-USDC", "lucky", [0.008, 0.009, 0.007, 0.0085, 0.0075])
    )
    mod = gate(rows)
    assert mod.refusal_reason("BASECAT-USDC") is not None
    assert mod.refusal_reason("BASECAT-USDC", "lucky") is not None
    assert mod.banned_pairs() == {}, (
        "a symbol already banned outright must not also be listed per executor"
    )


def test_an_unattributed_row_is_charged_to_nobody(gate) -> None:
    """The book predates the scout/bot ledger split.

    Rows with no ``strategy_id`` could belong to either executor, and those
    two have opposite records on the same symbol. Guessing would manufacture
    the exact verdict this rule exists to get right, so they contribute to
    the pooled book only.
    """
    mod = gate(_rows("AERO-USDC", "scout", _WINNER) + _rows("AERO-USDC", None, _LOSER))
    assert mod.refusal_reason("AERO-USDC") is None
    assert mod.banned_pairs() == {}


def test_a_book_with_no_details_column_still_gives_pooled_verdicts(gate,
                                                                   tmp_path,
                                                                   monkeypatch) -> None:
    """Fails open on an older schema rather than refusing everything."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE trade_outcomes ("
        " outcome_id TEXT, ts REAL, symbol TEXT, entry_price REAL,"
        " quantity REAL, net_profit REAL, status TEXT)"
    )
    for i, (symbol, _s, net, price, qty) in enumerate(_rows("COMP-USDC", None, _LOSER)):
        conn.execute(
            "INSERT INTO trade_outcomes VALUES (?,?,?,?,?,?,'closed')",
            (f"o{i}", 1788500000.0 + i, symbol, price, qty, net),
        )
    conn.commit()
    conn.close()

    monkeypatch.setenv("SYMBOL_EDGE_ROUND_TRIP_COST", str(COST))
    import services.symbol_edge_gate as mod

    mod = importlib.reload(mod)
    monkeypatch.setattr(mod, "DB_PATH", db)
    mod._cache.clear()
    mod._pair_cache.clear()
    mod._cache_built_at = 0.0
    assert mod.refusal_reason("COMP-USDC") is not None
    assert mod.banned_pairs() == {}


def test_the_stable_legs_are_never_banned_per_executor(gate) -> None:
    """NEVER_BAN has to hold at both granularities.

    Banning the quote leg for one executor would not avoid a bad trade, it
    would make that executor unable to trade at all.
    """
    mod = gate(_rows("USDC-USDT", "atf_static", _LOSER))
    assert mod.refusal_reason("USDC-USDT", "atf_static") is None
    assert mod.banned_pairs() == {}


# --- wiring -----------------------------------------------------------------
#
# Structural, and deliberately so: reaching the bot's call site behaviourally
# means driving `_interpret_predictions`, a 4000-line method behind a dozen
# upstream gates. These assert what the CALL is, not what a comment claims --
# a regression that drops the argument fails them.

def _call_arg_count(path: Path, callee: str) -> list:
    """Positional-argument counts of every call to `callee` in `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    counts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == callee:
            counts.append(len(node.args))
    return counts


def test_both_entry_paths_name_the_executor() -> None:
    """One rule, two entry paths.

    trading/bot.py and the ATF scout open positions independently. When
    symbol_edge_gate first shipped into the bot alone, the scout kept
    entering the very symbol the bot was refusing, minutes later. A
    per-executor verdict wired into only one of them has the same hole.
    """
    bot = _call_arg_count(ROOT / "trading" / "bot.py", "symbol_edge_refusal")
    scout = _call_arg_count(
        ROOT / "services" / "atf_static_strategy.py", "_symbol_edge_refusal")
    assert bot and max(bot) == 2, (
        "trading/bot.py must pass the directive's strategy_id to the gate")
    assert scout and max(scout) == 2, (
        "the ATF scout writes ghost-entry rows on its own path and must pass "
        "SCOUT_STRATEGY_ID to the same gate")


def test_the_import_fallbacks_accept_the_executor() -> None:
    """A missing gate must not stop trading -- and must not TypeError either.

    Both call sites wrap the import in try/except and define a stub. A stub
    still taking one argument turns an unimportable gate into a crash at
    every entry decision, which is worse than the gate being absent.
    """
    import services.symbol_edge_gate as mod

    assert len(inspect.signature(mod.refusal_reason).parameters) == 2

    for path, stub in (
        (ROOT / "trading" / "bot.py", "symbol_edge_refusal"),
        (ROOT / "services" / "atf_static_strategy.py", "_symbol_edge_refusal"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stubs = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == stub
        ]
        assert stubs, f"{path.name} defines no {stub} fallback"
        for node in stubs:
            assert len(node.args.args) == 2, (
                f"{path.name}: the {stub} fallback must accept the strategy id"
            )
