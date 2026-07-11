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
