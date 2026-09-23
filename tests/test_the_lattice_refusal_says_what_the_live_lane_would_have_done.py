"""The row that actually stops the entry path must say what the live lane would have done.

Measured 2026-09-23 (pass 679, Cove) on ``storage/trading_cache.db``:

    trading_ops, last 24h, by status
        entry-arbitration            2228
        entry-predropped-edge-ban    1899
        entry-refused-lattice        1756   <- trading/bot.py:8781
        ghost_candidate               248
        entry-refused-no-candidates   211
        ghost_candidate_quote_ok      179
        published                     102
        ghost-entry                     6

The entry path dies at the lattice ~180 lines ABOVE the swap guard (bot.py:8860)
and ~190 above the live/ghost fork (bot.py:8968). Everything the loop built to
explain "the live lane is silent" sits downstream of a gate that refuses 1756
times a day, so none of those instruments could ever have fired:

    feedback_events label='entry_downgraded_to_ghost'
        20d:3  19d:62  18d:59  17d:6  16d:6  then ZERO for 16 days

The refusal row itself exists the whole time. It was invisible to every live-lane
query for one reason: it carries neither a ``live-`` status nor (since the arm
went off) a ``wallet='live'``, and those are the two filters anybody writes.

    entry-refused-lattice by day-ago and wallet
        17d live:3 | 16d live:3 ghost:4 | 15d live:18 | 14d live:36 ghost:37
        13d..0d ghost only

That wallet flip is the arm going off at ~2026-09-09 -- FOUR DAYS AFTER the
2026-09-05 live cliff, so they are two events, not one. Inferring the arm from
the ``wallet`` column is exactly the ambiguity this test closes: ``wallet='ghost'``
conflates "the operator has not armed the lane" with "this is a ghost bot".

So the refusal records both facts outright:

    live_trading_enabled  -- the arm as the bot itself sees it
    live_approved         -- whether THIS directive's strategy would have been
                             allowed to spend money had the lattice passed

    enabled=False                  -> the lane is not armed
    enabled=True, approved=False   -> a live entry refused by graduation
    enabled=True, approved=True    -> a REAL live trade the lattice turned back,
                                      which has never been observed here

This is instrumentation. Neither field is read back, and the refusal is unchanged.

WHAT THIS TEST PROVES AND WHAT IT DOES NOT. It asserts structurally, over the
parsed AST rather than over the file as text, that the ``entry-refused-lattice``
``log_trade`` call carries both keys -- a grep would pass on a mention in a
comment. It does NOT drive a live decision cycle end to end; the entry function
is thousands of lines and needs a chain, a wallet, a feed and a directive. The
behaviour underneath the second field is covered separately below by calling
``_strategy_live_approved`` directly under both arm states.
"""

from __future__ import annotations

import ast
import pathlib

BOT = pathlib.Path(__file__).resolve().parents[1] / "trading" / "bot.py"

REQUIRED_KEYS = {"live_trading_enabled", "live_approved"}


def _lattice_log_trade_calls() -> list[ast.Call]:
    """Every ``log_trade(..., status="entry-refused-lattice", ...)`` call."""
    tree = ast.parse(BOT.read_text(encoding="utf-8"))
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "log_trade":
            continue
        for kw in node.keywords:
            if (
                kw.arg == "status"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "entry-refused-lattice"
            ):
                found.append(node)
    return found


def test_the_lattice_refusal_is_still_logged_at_all() -> None:
    """If this call ever disappears the entry path dies with no row at all."""
    calls = _lattice_log_trade_calls()
    assert calls, (
        "no log_trade(status='entry-refused-lattice') call found in trading/bot.py. "
        "This is the row that records the gate refusing 1756 entries a day; without "
        "it the entry path stops silently and the 17-day 'the live lane wrote no "
        "refusal' reading becomes true for real."
    )


def test_the_lattice_refusal_names_the_arm_and_the_approval() -> None:
    """Both live-lane facts must be on the row, not inferred from ``wallet``."""
    for call in _lattice_log_trade_calls():
        details = next(
            (kw.value for kw in call.keywords if kw.arg == "details"), None
        )
        assert isinstance(details, ast.Dict), (
            "the entry-refused-lattice log_trade call must pass a literal details "
            "dict so its contents are checkable"
        )
        keys = {
            k.value
            for k in details.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        missing = REQUIRED_KEYS - keys
        assert not missing, (
            "entry-refused-lattice details is missing %s. Without these the only "
            "readout of the live arm is the row's wallet column, which cannot tell "
            "'the operator has not armed the lane' from 'this is a ghost bot' -- the "
            "ambiguity that made the 2026-09-09 arm change look like the 2026-09-05 "
            "live cliff." % sorted(missing)
        )


def test_the_approval_field_is_not_a_constant() -> None:
    """``live_approved`` must be computed, not written True/False in place.

    A literal would pass the key check above while telling the reader nothing,
    which is the failure mode of every label this loop has had to re-derive.
    """
    for call in _lattice_log_trade_calls():
        details = next(
            (kw.value for kw in call.keywords if kw.arg == "details"), None
        )
        assert isinstance(details, ast.Dict)
        for key, value in zip(details.keys, details.values):
            if isinstance(key, ast.Constant) and key.value in REQUIRED_KEYS:
                assert not isinstance(value, ast.Constant), (
                    "%s is a hard-coded constant on the entry-refused-lattice row. "
                    "It must be read from the bot's own state, or it is a label with "
                    "no measurement behind it." % key.value
                )


def test_strategy_live_approval_fails_closed_under_the_arm() -> None:
    """The second field means something: unidentified strategies are refused.

    ``live_approved`` is ``live_trading_enabled and _strategy_live_approved(d)``.
    This exercises the underlying gate directly -- a directive with no
    strategy_id is an unidentified strategy asking to spend money, and under an
    explicit operator arm it must be refused rather than waved through as a
    legacy path.
    """
    import types

    from trading.bot import TradingBot

    bot = types.SimpleNamespace(strategy_ledger=None)

    class _Armed:
        @staticmethod
        def _operator_armed_live() -> bool:
            return True

    import trading.selector as selector

    original = selector.GhostTradingSupervisor._operator_armed_live
    selector.GhostTradingSupervisor._operator_armed_live = _Armed._operator_armed_live
    try:
        # No strategy_id -> no ledger row can approve it -> refused under the arm.
        assert TradingBot._strategy_live_approved(bot, None) is False
    finally:
        selector.GhostTradingSupervisor._operator_armed_live = original
