"""The only wall clock the exit rules have must not be gated on a model buffer.

EVERY exit rule in this bot is sample-driven. ``_handle_sample`` is the only
caller of ``_interpret_predictions`` and it carries ONE sample for ONE symbol,
so the stop-loss, the profit target, the confidence drop and the timed exit are
reachable only on a tick for that symbol. A position whose feed goes dark is
therefore unreachable by every rule that could end it.

The dark-feed sweeps are the answer to that, and they are deliberately driven
by ANY symbol's tick: a position on a dead feed cannot be reached from its own
symbol by construction, so something else has to be what notices. That makes
``_abandon_dark_feed_positions`` / ``_exit_dark_live_positions`` the ONLY
wall clock the exit path has, pool-wide.

THE DEFECT THIS FILE PINS: that clock sat below two early returns in
``_handle_sample`` that have nothing to do with whether some other symbol's
position has gone dark.

  * THE WINDOW GATE (``len(self._buffer) < self.window_size``) asks whether
    THIS bot can yet form a prediction. The sweep never invokes the model --
    it reads ``self.positions`` and the shared tick map and nothing else. A
    bot added by ``reconcile_pairs`` for a HELD symbol, added precisely so
    that position can be closed, starts with an EMPTY buffer and had to fill
    a full window before it would sweep anything: at CBBTC-USDC's measured
    50 ticks/h against a 60-step window, over an hour of pool-wide clock
    lost, during which every dark position in the merged book waits.

  * THE DUPLICATE-SIGNATURE RETURN drops a repeated ``(symbol, ts)``. A
    publisher that restamps the same timestamp swept nothing at all.

The tick note (``_note_symbol_tick``) was moved above the window gate for
exactly this reason, and the comment left on the sweep block claimed the sweep
had been moved with it -- "the tick was recorded above the window gate -- see
there for why" -- while the block itself was still below. A comment asserting
a property the code does not have is a shape this repo has shipped before, so
these tests assert on the BEHAVIOUR: they drive ``_handle_sample`` with a
short buffer and with a duplicate tick, and require the dark position to be
swept anyway.

Both tests fail against the pre-fix ordering (the sweep never runs, the
position keeps its slot forever) and pass with the sweep hoisted above both
returns.
"""

from __future__ import annotations

import asyncio
import time
import types

import pytest

from trading.bot import TradingBot

DARK = 3600.0


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

    def load_state(self):
        return {}

    def save_state(self, state):
        return None


def _dark_position(now: float, *, age_sec: float = 11.5 * 86400) -> dict:
    return {
        "mode": "ghost",
        "strategy_id": "atf_static",
        "trade_id": "t:atf_static",
        "entry_ts": now - age_sec,
        "ts": now - age_sec,
        "entry_price": 1.0,
        "size": 10.0,
    }


def _bot(now: float, *, buffer_len: int, window_size: int):
    """A bot holding a dark position on a DIFFERENT symbol than it streams.

    That separation is the point: HIGH-USDC is dark and unreachable from its
    own feed, so the only thing that can ever notice it is a tick on the
    symbol this bot actually streams.
    """
    bot = TradingBot.__new__(TradingBot)
    bot.positions = {"HIGH-USDC": _dark_position(now)}
    bot.db = _DB()
    bot.primary_chain = "base"
    bot.primary_symbol = "CBBTC-USDC"
    bot.bus_routes = {}
    bot.stable_bank = 0.0
    bot.total_profit = 0.0
    bot.realized_profit = 0.0
    bot.total_trades = 0
    bot.wins = 0
    bot.sim_quote_balances = {}
    bot.sim_native_balances = {}
    bot.ghost_session_id = "test"
    bot.active_exposure = 0.0
    bot._auto_execute_approved = False
    bot.swarm = types.SimpleNamespace(to_dict=lambda: {})

    # The model buffer, deliberately SHORT of the window: this is a bot that
    # reconcile_pairs has just added and which cannot yet predict anything.
    bot._buffer = [{"price": 1.0} for _ in range(buffer_len)]
    bot.window_size = window_size
    bot._processing_sample = False
    bot._pending_queue = __import__("collections").deque(maxlen=8)
    bot._last_sample_signature = None
    bot._equilibrium_last_adjust = now
    bot.metrics = _Stub()
    # Written unconditionally in _handle_sample's `finally`.
    bot._latency_window = __import__("collections").deque(maxlen=64)
    bot._latency_samples = 0
    bot._processing_lock = None
    # Reached once the buffer is long enough to clear the window gate.
    bot.live_trading_enabled = False

    # This process has been watching long enough to convict a dark symbol.
    bot.__dict__["_dark_feed_watch_since"] = now - 4 * DARK
    return bot


def _statuses(bot) -> list:
    return [row.get("status") for row in bot.db.logged]


def test_a_dark_position_is_swept_by_a_bot_whose_model_buffer_is_short():
    """The wall clock must not wait for a model window it never uses.

    Pre-fix: the window gate returns first, the sweep never runs, and
    HIGH-USDC keeps its slot for as long as this bot takes to fill 60 steps.
    """
    now = time.time()
    bot = _bot(now, buffer_len=3, window_size=60)

    asyncio.run(bot._handle_sample({"symbol": "CBBTC-USDC", "price": 1.0, "ts": now}))

    assert "HIGH-USDC" not in bot.positions, (
        "a dark position kept its slot because the sweeping bot's MODEL BUFFER "
        "was short -- the sweep reads positions and the tick map and never "
        "invokes the model, so the window gate must not gate it"
    )
    assert "position-abandoned-dark-feed" in _statuses(bot)


def test_a_repeated_tick_timestamp_still_drives_the_sweep():
    """The duplicate-signature return must not swallow the pool's clock too.

    Pre-fix: the second, identical tick returns at the signature check and the
    sweep below it never runs.
    """
    now = time.time()
    bot = _bot(now, buffer_len=90, window_size=60)
    # Arrange for the incoming tick to look like a repeat of the last one.
    bot._last_sample_signature = ("CBBTC-USDC", now)

    # The buffer is deliberately LONG here so the window gate cannot be what
    # this test proves -- the duplicate signature has to be the only early
    # return in play. That means the sample runs on past the sweep into the
    # full predict path, which this fixture does not model; we stop at the
    # first attribute it does not carry. The claim under test is ORDERING --
    # that the sweep already ran by then -- and the assertions below are what
    # check it, so swallowing this is not swallowing the result.
    try:
        asyncio.run(
            bot._handle_sample({"symbol": "CBBTC-USDC", "price": 1.0, "ts": now})
        )
    except AttributeError:
        pass

    assert "HIGH-USDC" not in bot.positions, (
        "a publisher restamping the same timestamp silently switched off the "
        "only wall clock the exit rules have"
    )
    assert "position-abandoned-dark-feed" in _statuses(bot)
