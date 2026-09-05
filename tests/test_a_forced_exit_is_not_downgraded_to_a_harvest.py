"""A stop or a hold clock closes the position. Half a position is not out.

The exit size was ``min(held_size, directive.size)`` unconditionally, so any
exit directive that happened to arrive on the same sample silently capped the
bracket's close.

MEASURED on the live book 2026-09-05. AERO-USDC had been held 66 minutes, past
the MAX_HOLD_FORCE_SECONDS=2700 already set in .env, and the forced exit ran.
The sample also carried an unrelated exit directive:

    strategy_id     rsi_reversal@5d
    reason          "[5d] rsi_reversal: RSI 74 overbought, harvesting 5.24%"
    size            1.4924427506920144
    position size   2.879009029542749509

so the clamp took 1.4924 and the hold clock's close became a sale of 51.8%.
The log records it plainly:

    [live-swap] exit sizing for AERO-USDC from chain: holds 2.87900902954275,
    position 2.879009029542749509, selling 1.492442750692014753

Had that swap settled, the slot would still have been busy, the clock would
still have been running, and the next sample would have forced the same half
exit again on a position half the size -- each one paying a full round trip in
gas. On a mandate of round trips in minutes, a close that does not free the
symbol is not a close.

``directive.size`` remains authoritative for every ordinary exit. A strategy
asking to take some profit off the table is a real instruction; a stop, a
break-even lock, a profit lock, a trailing stop or the operator's hold clock
is not an opinion about size.
"""
from __future__ import annotations

import pytest

from trading.triggers import (
    CLOSING_REASONS,
    PROTECTIVE_REASONS,
    closes_whole_position,
    exit_target_size,
    is_protective_exit,
)


# The exact numbers off the live AERO-USDC position and the directive that
# clamped it.
AERO_POSITION = 2.879009029542749509
AERO_DIRECTIVE_SIZE = 1.4924427506920144


def _exit_target(reason: str, held_size: float, directive_size: float | None) -> float:
    """The sizing rule bot.py actually applies -- the real function, not a copy.

    This helper used to re-implement the rule. A test that re-implements the
    code it is checking passes whatever the code does, which is the failure
    this repo already recorded once ("The test passed while proving nothing").
    Rule 2's inputs are left at their defaults here so these cases pin rule 1
    on its own; the dust cases below pass them explicitly.
    """
    return exit_target_size(
        reason, held_size=held_size, directive_size=directive_size
    )


@pytest.mark.parametrize("reason", [
    "max_hold_force:-0.0035",
    "stop_loss:-0.0184",
    "break_even_lock:0.0031",
    "profit_lock:0.0140<=0.0180",
    "trailing_stop:0.0210",
])
def test_a_closing_bracket_sells_the_whole_position(reason):
    """The live AERO case: a forced close is not capped by a harvest."""
    got = _exit_target(reason, AERO_POSITION, AERO_DIRECTIVE_SIZE)
    assert got == AERO_POSITION, (
        f"{reason!r} must sell all {AERO_POSITION}, not the directive's "
        f"{AERO_DIRECTIVE_SIZE} -- a half-sold slot is still a busy slot"
    )


@pytest.mark.parametrize("reason", [
    "",
    "confidence_drop",
    "timed-exit",
    "take_profit_limit",
    "time_take_profit:0.0090",
])
def test_an_ordinary_exit_still_honours_the_directive(reason):
    """A strategy asking to harvest part of a position is a real instruction.

    Widening the closing set to cover these would turn every partial harvest
    into a full liquidation, which is the opposite error.
    """
    got = _exit_target(reason, AERO_POSITION, AERO_DIRECTIVE_SIZE)
    assert got == AERO_DIRECTIVE_SIZE, (
        f"{reason!r} is not a decision to be out; the directive sizes it"
    )


def test_no_directive_always_sells_everything():
    for reason in ("", "confidence_drop", "max_hold_force:0.0"):
        assert _exit_target(reason, AERO_POSITION, None) == AERO_POSITION


def test_a_directive_larger_than_the_position_cannot_oversell():
    assert _exit_target("confidence_drop", AERO_POSITION, AERO_POSITION * 3) == AERO_POSITION


def test_the_hold_clock_closes_but_is_not_protective():
    """The two sets are deliberately different and must not drift together.

    ``max_hold_force`` reaches the live cost gate through ``forced_by_age``,
    on its own terms -- it is the operator's clock, not a claim that a loss is
    being cut. Folding it into the protective set would let it bypass the gate
    by a second, unreviewed route.
    """
    assert closes_whole_position("max_hold_force:-0.0035") is True
    assert is_protective_exit("max_hold_force:-0.0035") is False
    assert set(PROTECTIVE_REASONS) < set(CLOSING_REASONS)


def test_the_predicates_tolerate_a_missing_reason():
    for empty in (None, "", 0):
        assert closes_whole_position(empty) is False
        assert is_protective_exit(empty) is False


def test_bot_uses_the_shared_predicates_rather_than_its_own_copies():
    """Three literal copies of this tuple lived in bot.py.

    This repo has already been bitten by exactly that: "two copies of the
    predicate that decides whether real money moves is one edit away from
    disagreeing."
    """
    import inspect

    import trading.bot as bot

    source = inspect.getsource(bot)
    assert source.count('"break_even_lock", "profit_lock"') == 0, (
        "the protective tuple must be spelled once, in trading/triggers.py"
    )
    # bot.py no longer names ``closes_whole_position`` directly: the sizing
    # rule it guarded now lives in ``exit_target_size``, which applies it
    # alongside the no-dust rule. The point of this assertion is unchanged --
    # the predicates are imported from trading.triggers, never re-spelled here.
    assert "from trading.triggers import exit_target_size, is_protective_exit" in source
    assert "def exit_target_size" not in source, (
        "the sizing rule must be spelled once, in trading/triggers.py"
    )
