import time

from trading.triggers import evaluate_long_triggers


def _pos(entry=100.0, target=105.0, high=None, age=600.0, mode="ghost"):
    now = time.time()
    return {
        "mode": mode,
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


def test_a_flat_live_position_has_no_exit_without_the_hold_clock(monkeypatch):
    """The hole the hold clock fills, in the numbers that found it.

    CBXRP-USDC, live, opened by atf_static at 1788543478 and still open 20.6h
    later. Entry 1.41056722, target 1.46966765 (+4.19%), live stop -1.50%. Over
    554 ticks the price never left -1.46%..-0.39%, so the target was
    unreachable, the stop was missed by 4 basis points, and high_pnl_pct never
    went positive -- break-even, profit-lock and trailing never armed.
    """
    monkeypatch.delenv("MAX_HOLD_FORCE_SECONDS", raising=False)
    entry = 1.4105672172894166
    pos, now = _pos(entry=entry, target=1.4696676505689195, age=20.6 * 3600.0,
                    mode="live")

    # The worst tick it ever printed: still not a stop.
    result = evaluate_long_triggers(
        pos, price=1.39, fee_rate=0.0086, now_ts=now, live=True
    )
    assert not result.should_exit, "the -1.46% low was a stop after all: %s" % result.reason


def test_the_hold_clock_closes_a_position_that_goes_nowhere(monkeypatch):
    """With the operator's clock set, the same tick finally proposes an exit."""
    monkeypatch.setenv("MAX_HOLD_FORCE_SECONDS", "2700")
    entry = 1.4105672172894166
    pos, now = _pos(entry=entry, target=1.4696676505689195, age=20.6 * 3600.0,
                    mode="live")

    result = evaluate_long_triggers(
        pos, price=1.40, fee_rate=0.0086, now_ts=now, live=True
    )
    assert result.should_exit
    assert result.reason.startswith("max_hold_force")
    # Not a protective reason: bot.py bypasses the cost gate for this one via
    # forced_by_age, not by claiming the close pays for itself.
    assert not result.reason.startswith(
        ("stop_loss", "break_even_lock", "profit_lock", "trailing_stop")
    )


def test_the_hold_clock_does_not_fire_before_its_window(monkeypatch):
    monkeypatch.setenv("MAX_HOLD_FORCE_SECONDS", "2700")
    pos, now = _pos(entry=1.4105672172894166, target=1.4696676505689195, age=600.0,
                    mode="live")

    result = evaluate_long_triggers(
        pos, price=1.40, fee_rate=0.0086, now_ts=now, live=True
    )
    assert not result.should_exit


def test_the_hold_clock_leaves_the_ghost_lane_alone(monkeypatch):
    """Ghost has its own clock, and this book feeds graduation.

    Passed live=True deliberately: bot.py hands this function
    ``live=self.live_trading_enabled``, so every ghost position on a
    live-enabled bot arrives with live=True. The position's own mode is what
    must decide, or the clock force-closes the ghost book.
    """
    monkeypatch.setenv("MAX_HOLD_FORCE_SECONDS", "2700")
    pos, now = _pos(entry=1.4105672172894166, target=1.4696676505689195,
                    age=20.6 * 3600.0, mode="ghost")

    result = evaluate_long_triggers(
        pos, price=1.40, fee_rate=0.0086, now_ts=now, live=True
    )
    assert not result.should_exit


def test_an_unset_hold_clock_changes_nothing(monkeypatch):
    monkeypatch.delenv("MAX_HOLD_FORCE_SECONDS", raising=False)
    pos, now = _pos(entry=1.4105672172894166, target=1.4696676505689195, age=100 * 3600.0,
                    mode="live")

    result = evaluate_long_triggers(
        pos, price=1.40, fee_rate=0.0086, now_ts=now, live=True
    )
    assert not result.should_exit
