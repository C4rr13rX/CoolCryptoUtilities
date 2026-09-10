"""The ghost exit booking branch must not read an unbound bracket price.

``trading/bot.py`` binds ``entry_price_held`` / ``target_price_held`` at
7320-7321, inside the ``else:`` at 7307 -- the HELD-POSITION exit logic. Two
exit paths set ``should_exit = True`` without ever passing through it
(bot.py:7131 and bot.py:7216), so reaching the booking site at
``limit_exit_fill_price`` by either of them raised::

    UnboundLocalError: cannot access local variable 'target_price_held'
    where it is not associated with a value

That is a raise at the BOOKING site. The round trip does not close, no
``trade_outcomes`` row is written, and no reason is logged -- the silent shape
the status command reports as "GHOST: no ghost activity in 1h". Measured
2026-09-10, 226 trading_ops over 2h contained 1 ghost-entry and 0 closes, and
graduation counts closed round trips.

This asserts on the CODE rather than driving a tick, because the two paths that
skip the binding are reached through ~1900 lines of branch state that a unit
fixture cannot honestly reproduce -- and an assertion that "some tick closed"
would pass against the bug whenever the fixture happened to take the held-
position branch, which is exactly how this survived.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

BOT = Path(__file__).resolve().parents[1] / "trading" / "bot.py"

#: The names the booking site reads out of the position's bracket.
BRACKET_NAMES = ("target_price_held", "entry_price_held")


@pytest.fixture(scope="module")
def tree():
    return ast.parse(BOT.read_text(encoding="utf-8"))


def _exit_branch(tree):
    """The ``if should_exit and pos is not None:`` block, as an AST node."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        src = ast.dump(node.test)
        if "should_exit" in src and "pos" in src and isinstance(node.test, ast.BoolOp):
            return node
    raise AssertionError("could not find the `should_exit and pos is not None` branch")


def test_the_booking_branch_binds_the_bracket_prices_it_reads(tree):
    """Every bracket name the branch READS must be BOUND inside the branch.

    Binding them inside is what makes the branch independent of which of the
    ~7 `should_exit = True` sites got there. A fix that only handled today's
    two known paths would pass a test that named those two paths, and break on
    the eighth.
    """
    branch = _exit_branch(tree)

    read = {
        n.id
        for n in ast.walk(branch)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in BRACKET_NAMES
    }
    bound = {
        n.id
        for n in ast.walk(branch)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in BRACKET_NAMES
    }

    unbound = sorted(read - bound)
    assert not unbound, (
        "trading/bot.py's ghost exit booking branch READS %s without binding "
        "it; any should_exit path that skips the held-position branch raises "
        "UnboundLocalError at the booking site and closes no round trip"
        % ", ".join(unbound)
    )


def test_the_binding_happens_before_the_first_read(tree):
    """Order matters: a bind after the read is still an UnboundLocalError."""
    branch = _exit_branch(tree)

    for name in BRACKET_NAMES:
        stores = [
            n.lineno
            for n in ast.walk(branch)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == name
        ]
        loads = [
            n.lineno
            for n in ast.walk(branch)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == name
        ]
        if not loads:
            continue
        assert stores, "%s is read in the booking branch and never bound" % name
        assert min(stores) < min(loads), (
            "%s is bound at line %d but first read at line %d -- the bind is "
            "below the read and does not protect it" % (name, min(stores), min(loads))
        )


def test_the_bracket_prices_are_read_off_the_position(tree):
    """They must come from ``pos``, not be defaulted to zero.

    A ``target_price_held = 0.0`` placeholder would also stop the raise -- and
    would silently switch the overshoot clamp OFF, because
    ``limit_exit_fill_price`` returns the tick unchanged unless
    ``target > entry > 0``. That trades a loud crash for the exact fabricated
    take-profit fills the clamp exists to stop.
    """
    branch = _exit_branch(tree)
    source = ast.unparse(branch)

    for name in BRACKET_NAMES:
        key = name.replace("_held", "")
        assert ("%s = float(pos.get('%s'" % (name, key)) in source, (
            "%s must be read off pos['%s'] inside the booking branch; a "
            "constant default would disarm the overshoot clamp" % (name, key)
        )
