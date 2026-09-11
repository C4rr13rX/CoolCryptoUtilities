"""A live position with no feed must become sellable, not immortal.

``_abandon_dark_feed_positions`` correctly refuses to un-book a LIVE position:
it is the only record of tokens the wallet holds. But refusing to ABANDON is
not the same as being able to EXIT, and every exit rule here is sample-driven
(``_handle_sample`` is the only caller of ``_interpret_predictions`` and it
carries one sample for one symbol). So a live position whose symbol stops
ticking is unreachable by the stop, the target, the timed exit and even
``MAX_HOLD_FORCE_SECONDS`` -- and it holds its symbol against every further
entry. ``_exit_dark_live_positions`` exists to sell it at the chain price.

It had never once fired. Measured 2026-09-05 from ``trading_ops``:

    live-exit-forced-dark-feed rows, all time : 0
    entry-refused-duplicate on CBBTC-USDC     : 100
    CBBTC-USDC held by atf_static             : 19.8h
    CBXRP-USDC held by atf_static             : 19.1h
    live trades on the day                    : 0

The cause was the restart guard. The tick map starts empty on a fresh process,
so nothing may be convicted until the bot has watched the stream for a while --
correct, but the wait was ``dark_after`` (3600s) for every position, and this
pipeline's longest unbroken run in 24h was 36.6 minutes with a median gap under
10. A guard needing 60 consecutive minutes inside a process recycled every ~10
is not conservative, it is unreachable, and atf_static -- the ONLY live-approved
strategy -- was left holding two tokens it had no path to sell.

The rules pinned here:

  * a live position OLDER than the darkness window is convicted on the settle
    window, not the full one. Its ``entry_ts`` is persisted and survives the
    restart, so predating this process by hours is real evidence rather than an
    artifact of the empty tick map;
  * a YOUNGER live position still waits out the full window -- it can only be
    convicted by silence this process actually observed;
  * a symbol that IS ticking is never force-sold, however old the position. Age
    only shortens how long we wait to believe silence; it never sells anything
    on its own;
  * nothing at all fires inside the settle window, so a fresh process cannot
    dump the book on its first sample;
  * the exit is queued with no price attached, because a price from a dark feed
    is the stale number this sweep exists to distrust.

KNOWN GAP, measured 2026-09-05 and NOT yet closed. ``_queue_forced_live_exit``
documents itself as going "through the ordinary execution path with every guard
it carries". It does not. It appends to ``self.queue``, and the only consumer of
that queue is ``GhostTradingSupervisor._drain_trades`` (selector.py), which
calls ``_handle_trade`` -- a function that prints a line and updates
``profit_equilibrium``. It never swaps. The real live exit executes INSIDE
``_interpret_predictions`` via ``asyncio.to_thread(swapper.swap, ...)``, which
is reachable only from a sample, which is the very thing a dark feed denies.

So this sweep currently produces a ``live-exit-forced-dark-feed`` row and sells
nothing. The tests below pin the decision it emits, not a settled swap. The
structural fix shipped alongside them is in ``reconcile_pairs``: a held symbol
now always gets a BOT rather than a data-only stream, so the ordinary,
fully-guarded exit path can reach the position at all. Wiring this sweep to a
real sell is the remaining work.
"""

from __future__ import annotations

import time
import types

import pytest

from trading.bot import TradingBot

DARK = 900.0      # DARK_LIVE_EXIT_SEC default
SETTLE = 600.0    # DARK_LIVE_RESTART_SETTLE_SEC default


@pytest.fixture(autouse=True)
def _clean_tick_registry():
    """The tick map is process-wide, so it must not leak between tests."""
    TradingBot.reset_symbol_tick_registry()
    yield
    TradingBot.reset_symbol_tick_registry()


class _Stub:
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class _DB(_Stub):
    def __init__(self):
        self.logged: list = []

    def log_trade(self, **kwargs):
        self.logged.append(kwargs)
        return True


def _bot(positions):
    bot = TradingBot.__new__(TradingBot)
    bot.positions = positions
    bot.db = _DB()
    bot.primary_chain = "base"
    bot.queue = []
    bot.swarm = types.SimpleNamespace(to_dict=lambda: {})
    return bot


def _live(age_sec, now, symbol="CBBTC-USDC"):
    return {
        "mode": "live",
        "strategy_id": "atf_static",
        "trade_id": f"2:{symbol}:deadbeef",
        "entry_ts": now - age_sec,
        "ts": now - age_sec,
        "entry_price": 79910.09593421653,
        "size": 2.189e-05,
    }


def _statuses(bot):
    return [row["status"] for row in bot.db.logged]


def test_the_19_hour_position_is_sold_without_an_hour_of_uptime():
    """The exact 2026-09-05 deadlock, at the uptime the process actually gets."""
    now = time.time()
    bot = _bot({"CBBTC-USDC": _live(age_sec=19.8 * 3600, now=now)})
    # A realistic run: 11 minutes up, past the settle window, nowhere near the
    # hour the old guard demanded.
    bot.__dict__["_dark_live_watch_since"] = now - (SETTLE + 60.0)

    assert bot._exit_dark_live_positions(now) == 1
    assert "live-exit-forced-dark-feed" in _statuses(bot)
    assert len(bot.queue) == 1, "the sell must reach the execution queue"


def test_the_forced_exit_decision_carries_no_stale_price():
    now = time.time()
    bot = _bot({"CBBTC-USDC": _live(age_sec=19.8 * 3600, now=now)})
    bot.__dict__["_dark_live_watch_since"] = now - (SETTLE + 60.0)
    bot._exit_dark_live_positions(now)

    decision = bot.queue[0]
    assert decision["action"] == "exit"
    assert decision["wallet"] == "live"
    assert decision["forced"] is True
    assert decision["symbol"] == "CBBTC-USDC"
    # The executor reads the chain. A quoted price here would be exactly the
    # stale number the dark feed cannot be trusted for.
    assert "price" not in decision


def test_a_ticking_symbol_is_never_force_sold_however_old():
    """Age only shortens the wait for silence. It never sells on its own."""
    now = time.time()
    bot = _bot({"CBBTC-USDC": _live(age_sec=19.8 * 3600, now=now)})
    bot.__dict__["_dark_live_watch_since"] = now - (SETTLE + 60.0)
    # A bot in the pool is running exit rules on it -- the ordinary stop,
    # target and timed exit all still reach this position.
    bot._note_symbol_tick("CBBTC-USDC", now - 30.0)

    assert bot._exit_dark_live_positions(now) == 0
    assert bot.queue == []
    assert "CBBTC-USDC" in bot.positions


def test_a_young_live_position_still_waits_the_full_window():
    """Silence we did not observe is not evidence, for anything recent."""
    now = time.time()
    bot = _bot({"CBBTC-USDC": _live(age_sec=DARK - 120.0, now=now)})
    bot.__dict__["_dark_live_watch_since"] = now - (SETTLE + 60.0)

    assert bot._exit_dark_live_positions(now) == 0
    assert bot.queue == []


def test_nothing_is_sold_inside_the_settle_window():
    """A fresh process must not dump the book before any symbol can tick."""
    now = time.time()
    bot = _bot({"CBBTC-USDC": _live(age_sec=19.8 * 3600, now=now)})
    bot.__dict__["_dark_live_watch_since"] = now - (SETTLE - 60.0)

    assert bot._exit_dark_live_positions(now) == 0
    assert bot.queue == []


def test_the_first_call_only_starts_the_clock():
    now = time.time()
    bot = _bot({"CBBTC-USDC": _live(age_sec=19.8 * 3600, now=now)})

    assert bot._exit_dark_live_positions(now) == 0, "no watch history yet"
    assert bot.queue == []
    # ...and once the settle window has passed, it acts.
    assert bot._exit_dark_live_positions(now + SETTLE + 1.0) == 1


def test_a_ghost_position_is_not_touched_by_the_live_sweep():
    """Ghost positions are abandoned by the sibling sweep, never sold."""
    now = time.time()
    pos = _live(age_sec=19.8 * 3600, now=now)
    pos["mode"] = "ghost"
    bot = _bot({"CBBTC-USDC": pos})
    bot.__dict__["_dark_live_watch_since"] = now - (SETTLE + 60.0)

    assert bot._exit_dark_live_positions(now) == 0
    assert bot.queue == []


def test_both_stuck_symbols_are_freed_together():
    """CBBTC and CBXRP both blocked atf_static; one sweep must clear both."""
    now = time.time()
    bot = _bot({
        "CBBTC-USDC": _live(age_sec=19.8 * 3600, now=now, symbol="CBBTC-USDC"),
        "CBXRP-USDC": _live(age_sec=19.1 * 3600, now=now, symbol="CBXRP-USDC"),
    })
    bot.__dict__["_dark_live_watch_since"] = now - (SETTLE + 60.0)

    assert bot._exit_dark_live_positions(now) == 2
    assert {d["symbol"] for d in bot.queue} == {"CBBTC-USDC", "CBXRP-USDC"}


def test_the_settle_window_is_not_the_darkness_window():
    """Tying them together is what made the sweep unreachable. Keep them apart."""
    bot = _bot({})
    assert bot._dark_live_restart_settle_sec() < bot._dark_live_exit_sec(), (
        "the restart guard only needs long enough to see a tick; reusing the "
        "3600s exit threshold is the bug that kept this sweep from ever firing"
    )


def test_the_darkness_window_is_short_enough_for_the_stop_to_mean_anything():
    """3600s let BPAD-USDC lose 19% inside the window and never be looked at.

    Live entry 2026-09-05 17:48:02, last price 17:48:49, next price 18:16:47 --
    1678s of silence, comfortably inside a 3600s window, so this sweep never
    considered the position. The stop is 1.5%; the first tick after the hole
    fired it at -19.21% and the trade closed -0.2549 on a $1.50 clip, more than
    the entire live book (-0.1864 over 18 round trips, +0.0686 without it).

    Measured over 7d of market_stream on the symbols we actually trade (24,053
    inter-tick gaps), the p90 absolute move across a gap of at least T, and the
    expected loss avoided by selling at T net of the measured 0.555% round trip:

        450s   4.15%   -0.01%      <- force-selling starts paying for itself
        900s   6.16%   +0.26%
        1800s  8.47%   +0.85%
        3600s 15.55%   +1.88%      <- the old window

    Below ~450s the fee exceeds the loss avoided and dumping on every slow
    patch would be its own bug, so the window wants to sit just above that and
    above the 600s settle guard. Anything approaching 3600s means a 1.5% stop
    is enforced against a p90 move ten times its size.
    """
    bot = _bot({})
    assert bot._dark_live_exit_sec() <= 900.0, (
        "a live position may not go unpriced for longer than the stop can "
        "survive; 1678s of silence realised -19.21% against a 1.5% stop"
    )


# ------------------------------------------------- aliveness is not a forecast


def test_a_tick_marks_the_symbol_alive_before_the_model_window_fills():
    """A newly added bot must not report its own symbol dark for a full window.

    ``reconcile_pairs`` adds a bot for a held symbol precisely so the position
    can be closed. That bot starts with an empty buffer, and the tick map was
    only written AFTER the window gate -- so for one whole window (over an hour
    at CBBTC-USDC's measured 50 ticks/h against 60 steps) every bot in the pool
    read the symbol it had just added as dark. A tick arriving is the symbol
    being alive; whether this bot can yet form a prediction is a separate
    question, answered by the window gate itself.
    """
    import asyncio
    import collections

    TradingBot.reset_symbol_tick_registry()
    try:
        bot = TradingBot.__new__(TradingBot)
        bot._processing_sample = False
        bot._pending_queue = collections.deque(maxlen=8)
        bot._equilibrium_last_adjust = 1e18   # skip the equilibrium branch
        bot._buffer = []
        bot.window_size = 60                  # far from full -> early return
        bot._latency_window = collections.deque(maxlen=8)
        bot._latency_samples = 0
        bot.metrics = _Stub()

        now = time.time()
        asyncio.run(bot._handle_sample({"symbol": "CBBTC-USDC", "ts": now}))

        assert len(bot._buffer) == 1, "the window gate must still short-circuit"
        seen = TradingBot.symbol_tick_snapshot() if hasattr(
            TradingBot, "symbol_tick_snapshot") else None
        if seen is None:
            from trading.bot import _SYMBOL_LAST_TICK_TS, _SYMBOL_LAST_TICK_LOCK
            with _SYMBOL_LAST_TICK_LOCK:
                seen = dict(_SYMBOL_LAST_TICK_TS)
        assert seen.get("CBBTC-USDC") == pytest.approx(now), (
            "a tick that arrived must mark the symbol alive even though the "
            "model buffer is not full yet"
        )
    finally:
        TradingBot.reset_symbol_tick_registry()
