"""``stale_exit_secs`` is a wall clock and must not need a tick to be reached.

THE BUG. ``stale_exit_secs`` promises a close at 900 seconds. Every exit rule in
this bot is reachable only from ``_handle_sample``, which is the only caller of
``_interpret_predictions`` and passes ONE sample for ONE symbol -- and only after
two further gates that have nothing to do with exits:

  * the model-window gate (``len(self._buffer) < self.window_size``), which a bot
    added by ``reconcile_pairs`` for a HELD symbol -- added precisely so the
    position can be closed -- must fill before any exit rule runs at all;
  * the duplicate-signature return, which drops a repeated ``(symbol, ts)``, so a
    symbol whose publisher restamps one timestamp never evaluates an exit.

MEASURED over the 14 days to 2026-09-11 by ``scripts/hold_time_edge.py`` on the
78 round trips the live lane could have placed: 86% outlive the 900s clock, and
ticks/min falls monotonically with hold time (0.71 under 5 min, 0.27 at 1-4
hours, 0.14 past four hours). The trips that overstay are the ones the feed
stopped watching, so the positions that most need rule 4 are the least able to
reach it. Separately, 20 positions in the same window were ABANDONED by the
dark-feed sweep -- median 180.6 minutes held, 17 of them tradeable -- booking no
outcome at all, which is 20% of every position that ended.

WHAT IS PINNED HERE.

  * A ghost position past the clock closes on the WALL CLOCK with no further
    tick ever arriving on its own symbol (the test that fails without the sweep).
  * A position INSIDE the clock is not touched: this adds no exit rule and
    shortens no clock.
  * A mark older than ``_stale_sweep_price_max_age_sec`` does NOT close the
    position. Marking out against an hour-old price fabricates an outcome --
    AERO-USDC once booked +161% that way -- so past the bound the dark-feed
    sweep owns the position and books nothing.
  * A LIVE position is never closed here. The live lane has its own wall-clock
    treatment (``_exit_dark_live_positions``) which asks the CHAIN for a price.
  * Two bots sharing the merged book close one position ONCE. A double-booked
    outcome is worse than a missed one: graduation counts it twice.
  * The sweep and the exit chain read ONE clock. When they were two env reads the
    sweep could offer a position the chain then refused, on every tick, forever.

SELECTION EFFECT, stated because the number that motivates this fix carries it:
the hold-time split is over REALISED trips, so a trip may have closed fast
BECAUSE it hit its target. Closing more trips on time moves the population so
the question can be asked at n>=30; it does not prove the fast ones were good.
"""

from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest

import trading.bot as bot_module
from trading.bot import TradingBot

from tests.test_a_stale_loser_is_measured_not_forecast import _bot as _bare_bot


def _bot():
    """The shared harness plus the outbound decision queue the sweep writes to.

    ``queue`` is a real ``TradingBot`` attribute set in ``__init__`` -- the
    shared fixture omits it because ``_interpret_predictions`` alone never
    appends. The sweep does, exactly as ``_handle_sample`` does.
    """
    bot = _bare_bot()
    bot.queue = []
    return bot

SYMBOL = "CP-USDC"
TOKEN = "0xcccccccccccccccccccccccccccccccccccccccc"
ENTRY = 1.0
# A nothing-move: inside the 2% stop, below the round-trip cost, so the ONLY
# rule that can close it is the clock. -0.2% is the median shape of the
# 1-4 hour bucket that carries the book's whole loss.
MARK = 0.998
SIZE = 6.0
STALE_SEC = 900.0


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "1")
    monkeypatch.setenv("GHOST_NEG_EXIT_SECONDS", str(int(STALE_SEC)))
    monkeypatch.setenv("GHOST_STOP_LOSS_PCT", "0.02")
    monkeypatch.setenv("MIN_HOLD_SECONDS", "300")
    monkeypatch.setenv("MAX_HOLD_FORCE_SECONDS", "0")
    monkeypatch.setenv("GHOST_STALE_SWEEP_PRICE_MAX_AGE_SEC", "900")
    TradingBot.reset_symbol_tick_registry()
    bot_module._STALE_SWEEP_IN_FLIGHT.clear()
    yield
    TradingBot.reset_symbol_tick_registry()
    bot_module._STALE_SWEEP_IN_FLIGHT.clear()


def _position(*, mode: str, held_sec: float, trade_id: str = "2:ghost:stale-clock") -> dict:
    opened = time.time() - held_sec
    return {
        "mode": mode,
        "strategy_id": "rsi_reversal",
        "size": SIZE,
        "entry_price": ENTRY,
        # No target, so rule 1 cannot decide anything here.
        "target_price": 0.0,
        "entry_ts": opened,
        "ts": opened,
        "trade_id": trade_id,
        "route": [TOKEN, "USDC"],
        "quote_spent": ENTRY * SIZE,
        "base_token_address": TOKEN,
        "chain": "base",
    }


def _seen(symbol: str, *, price: float, age_sec: float) -> None:
    """Record a pool-wide tick for ``symbol`` observed ``age_sec`` ago."""
    recorder = TradingBot.__new__(TradingBot)
    TradingBot._note_symbol_tick(recorder, symbol, time.time() - age_sec, price=price)


def _sweep(bot: TradingBot, *, now: float | None = None) -> int:
    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot,
        "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (sym, explicit or TOKEN),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync):
        return asyncio.run(
            bot._close_stale_positions_on_the_clock(
                time.time() if now is None else now
            )
        )


# --------------------------------------------------------------------------
# The failure this file exists to prevent.
# --------------------------------------------------------------------------


def test_a_stale_ghost_position_closes_without_a_tick_on_its_own_symbol():
    """2x the clock, a fresh pool-wide mark, and no tick on its own symbol.

    This is the 1-4 hour bucket. Before the sweep existed nothing in this bot
    could close it: the position's own symbol is not what the holding bot is
    streaming, or its window is unfilled, or its timestamp repeats -- and
    ``_interpret_predictions`` is only ever called from a tick.
    """
    bot = _bot()
    bot.positions[SYMBOL] = _position(mode="ghost", held_sec=2 * STALE_SEC)
    _seen(SYMBOL, price=MARK, age_sec=30.0)

    closed = _sweep(bot)

    assert closed == 1, "a position at 2x the stale clock was not closed"
    exits = [d for d in bot.queue if d.get("action") == "exit"]
    assert exits, f"no exit decision queued; queue={list(bot.queue)}"
    reason = str(exits[-1].get("exit_reason") or exits[-1].get("reason") or "")
    assert reason.startswith("timed-exit"), (
        "the wall clock must close it under rule 4's own name -- the ghost exit "
        f"gate admits `timed-exit` by name and refuses other reasons; got {reason!r}"
    )
    assert SYMBOL not in bot.positions, "the slot was not freed"
    assert bot.db.outcomes, "the round trip was closed but no outcome was booked"


def test_a_position_inside_the_clock_is_not_swept():
    """No rule is added and no clock is shortened: inside 900s, nothing happens."""
    bot = _bot()
    bot.positions[SYMBOL] = _position(mode="ghost", held_sec=STALE_SEC - 60.0)
    _seen(SYMBOL, price=MARK, age_sec=30.0)

    assert _sweep(bot) == 0
    assert SYMBOL in bot.positions, "a position inside its clock was closed"
    assert not bot.db.outcomes, "an outcome was booked inside the clock"


def test_a_stale_mark_does_not_book_a_fabricated_outcome():
    """Past the price bound the dark-feed sweep owns it, and books nothing.

    This is the guard that separates "close it honestly" from the +161%
    stale-repricing artifact ``StrategyLedger._is_implausible`` exists to reject.
    """
    bot = _bot()
    bot.positions[SYMBOL] = _position(mode="ghost", held_sec=4 * STALE_SEC)
    _seen(SYMBOL, price=MARK, age_sec=1200.0)   # older than the 900s bound

    assert _sweep(bot) == 0
    assert SYMBOL in bot.positions
    assert not bot.db.outcomes, "a position was marked out against a 20-minute-old price"


def test_a_live_position_is_not_closed_by_the_ghost_wall_clock():
    """A live exit must go through the chain, not a marked-out book entry."""
    bot = _bot()
    bot.positions[SYMBOL] = _position(mode="live", held_sec=4 * STALE_SEC)
    _seen(SYMBOL, price=MARK, age_sec=30.0)

    assert _sweep(bot) == 0
    assert SYMBOL in bot.positions, "a LIVE position was closed by the ghost sweep"


def test_two_bots_on_one_merged_book_close_the_position_once():
    """Every bot runs this sweep over the SAME merged book.

    Without the in-flight claim both book an outcome for one round trip, and
    graduation counts the evidence twice.
    """
    position = _position(mode="ghost", held_sec=2 * STALE_SEC)
    first = _bot()
    second = _bot()
    first.positions[SYMBOL] = position
    # The SAME row, still in the second bot's view of the merged book -- which is
    # what a concurrent sweep sees, and is also exactly what a bot whose book
    # snapshot predates the first bot's `_save_state` sees. Removing it here
    # instead would test `dict.pop`, not the claim.
    second.positions[SYMBOL] = dict(position)
    _seen(SYMBOL, price=MARK, age_sec=30.0)

    closed = _sweep(first) + _sweep(second)

    assert closed == 1, f"one round trip closed {closed} times"
    assert len(first.db.outcomes) + len(second.db.outcomes) == 1, (
        "one round trip booked two outcomes -- graduation would count the "
        "evidence twice"
    )


def test_the_sweep_and_the_exit_chain_read_one_clock(monkeypatch):
    """A sweep on a shorter clock than the chain proposes what the chain refuses."""
    bot = _bot()
    monkeypatch.setenv("GHOST_NEG_EXIT_SECONDS", "1800")
    assert bot._stale_exit_secs() == 1800.0

    bot.positions[SYMBOL] = _position(mode="ghost", held_sec=1200.0, trade_id="inside")
    _seen(SYMBOL, price=MARK, age_sec=30.0)
    assert _sweep(bot) == 0, "swept a position inside the clock the chain is using"

    # A SECOND BOT, not a second sweep on the first: the sweep is throttled to
    # once per 30s on a path that runs per tick, so re-calling it on the same
    # instance would measure the throttle rather than the clock.
    other = _bot()
    other.positions[SYMBOL] = _position(mode="ghost", held_sec=2400.0, trade_id="past")
    assert _sweep(other) == 1, "did not sweep a position past the clock the chain is using"


def test_a_recorded_price_always_belongs_to_the_recorded_timestamp():
    """The freshness bound cannot refuse a mismatch it cannot see.

    ``_note_symbol_tick`` only accepts a price when the timestamp advances, so
    the two maps always describe the same tick. A later, cheaper tick arriving
    with an older timestamp must not repaint the price the sweep will mark out
    against.
    """
    bot = TradingBot.__new__(TradingBot)
    now = time.time()
    TradingBot._note_symbol_tick(bot, SYMBOL, now, price=1.0)
    TradingBot._note_symbol_tick(bot, SYMBOL, now - 600.0, price=0.5)

    assert bot_module._SYMBOL_LAST_TICK_TS[SYMBOL] == pytest.approx(now)
    assert bot_module._SYMBOL_LAST_TICK_PX[SYMBOL] == pytest.approx(1.0), (
        "an older tick's price was paired with a newer tick's timestamp"
    )


def test_a_position_with_no_entry_time_has_no_clock_to_outlive():
    """An unknown entry time is not evidence of a stale position.

    The same choice rule 4 makes for an unknown cost basis: the max-hold
    eviction owns it rather than an invented hold time.
    """
    bot = _bot()
    pos = _position(mode="ghost", held_sec=4 * STALE_SEC)
    pos["entry_ts"] = 0.0
    pos["ts"] = 0.0
    bot.positions[SYMBOL] = pos
    _seen(SYMBOL, price=MARK, age_sec=30.0)

    assert _sweep(bot) == 0
    assert SYMBOL in bot.positions
