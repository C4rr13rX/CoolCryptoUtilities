import time

from trading.triggers import evaluate_long_triggers


def _pos(entry=100.0, target=105.0, high=None, age=600.0):
    now = time.time()
    return {
        "entry_price": entry,
        "target_price": target,
        "entry_ts": now - age,
        "trigger_state": {"high_watermark": high or entry},
    }, now


def test_take_profit_target_exits():
    pos, now = _pos()
    result = evaluate_long_triggers(pos, price=105.1, fee_rate=0.0075, now_ts=now, live=False)
    assert result.should_exit
    assert result.reason == "take_profit_limit"


def test_stop_loss_exits():
    pos, now = _pos(target=0)
    result = evaluate_long_triggers(pos, price=97.0, fee_rate=0.0075, now_ts=now, live=False)
    assert result.should_exit
    assert result.reason.startswith("stop_loss")


def test_break_even_lock_after_winner_round_trips():
    pos, now = _pos(target=0, high=103.0)
    result = evaluate_long_triggers(pos, price=100.7, fee_rate=0.0075, now_ts=now, live=False)
    assert result.should_exit
    assert result.reason.startswith("break_even_lock")


def test_trailing_stop_locks_profit():
    pos, now = _pos(target=0, high=106.0)
    result = evaluate_long_triggers(pos, price=104.5, fee_rate=0.0075, now_ts=now, live=False)
    assert result.should_exit
    assert result.reason.startswith(("profit_lock", "trailing_stop"))


def test_take_profit_requires_clearing_the_cost_basis():
    """A fill above the plan's target must not read as an instant win.

    Live BSTONK-USDC, 2026-09-03: the plan was built on a seen price of
    0.00191429 with a +5% target of 0.00201000, but the swap filled at
    0.00208181 -- above the target. `price >= target` was true from the
    instant the position opened, so the take-profit would have closed a
    position that was 3.57% underwater and recorded it as profit-taking.
    """
    pos, now = _pos(entry=0.00208181, target=0.00201000)
    result = evaluate_long_triggers(
        pos, price=0.00208181, fee_rate=0.0065, now_ts=now, live=True
    )
    assert not result.should_exit, (
        "take-profit fired below the cost basis: %s" % result.reason
    )

    # The same position genuinely above its basis still takes profit.
    pos, now = _pos(entry=0.00208181, target=0.00201000)
    result = evaluate_long_triggers(
        pos, price=0.00220000, fee_rate=0.0065, now_ts=now, live=True
    )
    assert result.should_exit
    assert result.reason == "take_profit_limit"


def test_live_stop_is_tighter_than_ghost():
    """The live default (1.5%) is what the BSTONK position should have hit."""
    pos, now = _pos(entry=0.0020818052696136586, target=0.0)
    # First feed sample after that entry landed at -13.08%.
    result = evaluate_long_triggers(
        pos, price=0.0018095238, fee_rate=0.0065, now_ts=now, live=True
    )
    assert result.should_exit
    assert result.reason.startswith("stop_loss")
