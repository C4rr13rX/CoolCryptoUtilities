"""A position whose symbol stops ticking must not hold its slot forever.

EVERY exit rule in this bot is sample-driven. ``_handle_sample`` is the only
caller of ``_interpret_predictions`` and it carries ONE sample for ONE symbol,
so the stop-loss, the profit target, the confidence drop, the timed exit and
even the ``MAX_HOLD_FORCE_SECONDS`` escape hatch are all reachable only on a
tick for that symbol. A position whose feed goes dark is therefore not "held" --
it is unreachable by every rule that could end it.

Measured 2026-09-04 on the persisted book, cross-referenced against
``market_stream``: 4 of 13 open ghost positions sat on symbols with no tick at
all, while only 5 of the 14 symbols carrying a feed were free to enter.

    HIGH-USDC     11.5 DAYS held, strategy_id empty
    VIRTUAL-USDC  39.0 hours   obv_accumulation@5d
    ARB-USDC      14.4 hours   obv_accumulation@1d
    PEPE-USDC      2.9 hours   money_button   <- the priority lane, stranded

``entry-refused-duplicate`` was the single most common thing the pipeline did
(468 in 24h) because the slots were held by positions that could never close.
That is the drought underneath link 5: graduation is scored on 20 COMPLETED
ghost trades per strategy, and the book was full of trades that structurally
could not complete. Ghost evidence yield measured 82 exits against 761 entries
in 24h (10.8%).

The rules pinned here:

  * a ghost position on a symbol that has not been priced for
    GHOST_DARK_FEED_ABANDON_SEC is ABANDONED -- dropped from the book with no
    outcome recorded, logged as ``position-abandoned-dark-feed``;
  * it is abandoned, never CLOSED. Closing means marking out against a price
    from an hour or eleven days ago, and that stale-entry repricing is the
    artifact ``StrategyLedger._is_implausible`` exists to reject (AERO-USDC once
    booked +161% that way). A lost observation is honest; a fabricated outcome
    in the book that gates real money is not;
  * a LIVE position is NEVER abandoned for a dark feed, whatever its age. It is
    the only record of tokens the wallet holds, and a missing price is not
    evidence they are gone -- that question belongs to
    ``_drop_phantom_live_position``, which asks the chain. Fail closed;
  * a position on a symbol that IS still ticking is untouched -- including when
    the tick was seen by a DIFFERENT bot in the pool. Darkness is a fact about
    the symbol, not about whoever is asking;
  * nothing is reaped until the bot has watched the stream for a full darkness
    window. The tick map starts empty on a fresh process, so a naive sweep would
    abandon the entire book on the first sample after every restart.

The cross-bot rule was added 2026-09-04 after measuring the last 25
``position-abandoned-dark-feed`` rows against ``market_stream``: 18 of 25 were
abandoned while the feed was LIVE, because the tick map was per-instance while
the position book is merged across the pool.

    22:52:29  AERO-USDC  claimed 82.8 min silent, last tick  3.2 min ago
    22:52:29  MOG-USDC   claimed 82.8 min silent, last tick  1.4 min ago
    22:29:20  COMP-USDC  claimed 65.3 min silent, last tick  0.9 min ago

The identical claimed silence across unrelated symbols is the signature: with
nothing in the map, ``last_seen`` fell back to the sweeping bot's own
watch-start for every one of them at once.
"""

from __future__ import annotations

import time
import types

import pytest

from trading.bot import TradingBot

DARK = 3600.0


@pytest.fixture(autouse=True)
def _clean_tick_registry():
    """The tick map is process-wide now, so it must not leak between tests."""
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
        self.saved: list = []

    def log_trade(self, **kwargs):
        self.logged.append(kwargs)
        return True

    def load_state(self):
        return {}

    def save_state(self, state):
        self.saved.append(state)


def _bot(positions):
    """A bot with exactly the attributes the sweep and _save_state touch."""
    bot = TradingBot.__new__(TradingBot)
    bot.positions = positions
    bot.db = _DB()
    bot.primary_chain = "base"
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
    return bot


def _position(mode="ghost", strategy="atf_static", age_sec=7200.0, now=None):
    now = now if now is not None else time.time()
    return {
        "mode": mode,
        "strategy_id": strategy,
        "trade_id": f"t:{strategy}",
        "entry_ts": now - age_sec,
        "ts": now - age_sec,
        "entry_price": 1.0,
        "size": 10.0,
    }


def _statuses(bot):
    return [row["status"] for row in bot.db.logged]


def test_a_stale_ghost_position_is_abandoned_and_frees_its_slot():
    now = time.time()
    bot = _bot({"HIGH-USDC": _position(age_sec=11.5 * 86400, now=now)})
    # This process has been watching long enough to convict.
    bot.__dict__["_dark_feed_watch_since"] = now - 4 * DARK

    assert bot._abandon_dark_feed_positions(now) == 1
    assert "HIGH-USDC" not in bot.positions, "the slot must be freed"
    assert "position-abandoned-dark-feed" in _statuses(bot)


def test_the_abandoned_position_records_no_outcome():
    """Abandoned, never closed -- no P&L may be invented from a stale price."""
    now = time.time()
    bot = _bot({"HIGH-USDC": _position(age_sec=11.5 * 86400, now=now)})
    bot.__dict__["_dark_feed_watch_since"] = now - 4 * DARK
    bot._abandon_dark_feed_positions(now)

    for row in bot.db.logged:
        assert row["action"] == "hold", "an abandonment is not an exit"
        details = row.get("details") or {}
        for forbidden in ("net_profit", "profit", "exit_price", "gross_profit"):
            assert forbidden not in details, (
                f"{forbidden} in an abandonment would fabricate an outcome "
                "against a stale price"
            )


def test_a_ticking_symbol_is_never_abandoned():
    now = time.time()
    bot = _bot({"CBBTC-USDC": _position(age_sec=11.5 * 86400, now=now)})
    bot.__dict__["_dark_feed_watch_since"] = now - 4 * DARK
    # Old position, but the feed is alive -- the exit rules can still reach it.
    bot._note_symbol_tick("CBBTC-USDC", now - 30.0)

    assert bot._abandon_dark_feed_positions(now) == 0
    assert "CBBTC-USDC" in bot.positions


def test_a_tick_seen_by_ANOTHER_BOT_still_saves_the_position():
    """The exact 2026-09-04 failure: AERO reaped 3.2 min after it last ticked.

    GhostSupervisor gives each bot ONE ``MarketDataStream(symbol=...)`` but
    seeds every bot from the MERGED position book, so the bot sweeping a
    position is usually not the bot streaming that symbol. A per-instance tick
    map therefore convicted every symbol but the sweeper's own.
    """
    now = time.time()
    aero_streamer = _bot({})                                   # streams AERO
    sweeper = _bot({"AERO-USDC": _position(age_sec=7200.0, now=now)})
    sweeper.__dict__["_dark_feed_watch_since"] = now - 4 * DARK

    # The tick lands on the OTHER bot, exactly as the stream delivers it.
    aero_streamer._note_symbol_tick("AERO-USDC", now - 192.0)

    assert sweeper._abandon_dark_feed_positions(now) == 0
    assert "AERO-USDC" in sweeper.positions, (
        "a symbol another bot in the pool is still pricing is reachable by "
        "that bot's exit rules and must not be abandoned"
    )
    assert "position-abandoned-dark-feed" not in _statuses(sweeper)


def test_a_symbol_no_bot_streams_is_still_reaped():
    """The rule must not become 'never reap'.

    A data-only stream (selector.py 1383) writes ``market_stream`` but never
    calls ``_handle_sample``, so its symbol stays absent from the shared map --
    correctly, because no bot is running exit rules on it and nothing can ever
    close a position there.
    """
    now = time.time()
    other = _bot({})
    sweeper = _bot({"HIGH-USDC": _position(age_sec=11.5 * 86400, now=now)})
    sweeper.__dict__["_dark_feed_watch_since"] = now - 4 * DARK
    other._note_symbol_tick("AERO-USDC", now - 10.0)   # a different symbol

    assert sweeper._abandon_dark_feed_positions(now) == 1
    assert "HIGH-USDC" not in sweeper.positions
    assert "position-abandoned-dark-feed" in _statuses(sweeper)


def test_the_tick_map_is_one_map_for_the_whole_pool():
    """Type, shape and units at the boundary, asserted rather than assumed."""
    now = time.time()
    a, b = _bot({}), _bot({})
    a._note_symbol_tick("AERO-USDC", now - 192.0)

    seen = b._last_tick_ts
    assert seen is a._last_tick_ts, "one map, shared by every bot in the pool"
    value = seen["AERO-USDC"]
    assert isinstance(value, float)               # not str, not int, not None
    # Epoch SECONDS on the same clock as ``sample["ts"]`` -- a millisecond
    # value here would read as the year 58681 and nothing would ever be dark.
    assert 0.0 < now - value < DARK

    # Only ever moves forward: an out-of-order tick must not resurrect silence.
    a._note_symbol_tick("AERO-USDC", now - 9999.0)
    assert seen["AERO-USDC"] == value

    # Junk at the boundary is dropped, never stored.
    for junk in ("", "   ", None):
        a._note_symbol_tick(junk, now)
    a._note_symbol_tick("NAN-USDC", float("nan"))
    a._note_symbol_tick("NEG-USDC", -1.0)
    a._note_symbol_tick("STR-USDC", "not-a-number")
    assert set(seen) == {"AERO-USDC"}


def test_a_live_position_is_kept_however_dark_its_feed():
    """Real tokens are never un-booked because a price stopped arriving."""
    now = time.time()
    bot = _bot({"BSTONK-USDC": _position(mode="live", age_sec=9 * 86400, now=now)})
    bot.__dict__["_dark_feed_watch_since"] = now - 4 * DARK

    assert bot._abandon_dark_feed_positions(now) == 0
    assert "BSTONK-USDC" in bot.positions, (
        "a live position is the only record of tokens the wallet holds"
    )
    assert "live-position-dark-feed" in _statuses(bot)
    assert "position-abandoned-dark-feed" not in _statuses(bot)


def test_nothing_is_reaped_until_a_full_window_has_been_watched():
    """The tick map is empty on a fresh process; every symbol looks dark."""
    now = time.time()
    bot = _bot({
        "HIGH-USDC": _position(age_sec=11.5 * 86400, now=now),
        "ARB-USDC": _position(age_sec=14.4 * 3600, now=now),
    })

    # First call on a brand-new bot: the watch window opens now.
    assert bot._abandon_dark_feed_positions(now) == 0
    assert len(bot.positions) == 2, "a restart must not empty the book"

    # Still inside the window.
    bot.__dict__["_dark_feed_next_sweep"] = 0.0
    assert bot._abandon_dark_feed_positions(now + DARK - 1.0) == 0
    assert len(bot.positions) == 2

    # Past it, the same positions are convicted.
    bot.__dict__["_dark_feed_next_sweep"] = 0.0
    assert bot._abandon_dark_feed_positions(now + DARK + 1.0) == 2
    assert bot.positions == {}


def test_a_reaped_symbol_does_not_stay_owned():
    """Borrowed ownership must not outlive the save that used it.

    ``_save_state`` expresses a removal as "owned AND absent from my payload",
    which stays true forever once claimed. A permanent claim would make this bot
    delete that symbol on EVERY later save -- including one where another bot in
    the pool had legitimately reopened the position. That is the shared-book
    clobber the ownership rule exists to prevent.
    """
    now = time.time()
    bot = _bot({"HIGH-USDC": _position(age_sec=11.5 * 86400, now=now)})
    bot.__dict__["_dark_feed_watch_since"] = now - 4 * DARK
    assert "HIGH-USDC" not in bot._owned_symbols

    assert bot._abandon_dark_feed_positions(now) == 1
    assert "HIGH-USDC" not in bot._owned_symbols, (
        "ownership was borrowed for the save and must be handed back"
    )
    # The removal really was persisted while the claim was held.
    assert bot.db.saved, "_save_state must run so the drop survives a restart"
    assert "HIGH-USDC" not in bot.db.saved[-1]["ghost_trading"]["positions"]


def test_a_symbol_this_bot_already_owned_stays_owned():
    now = time.time()
    bot = _bot({"HIGH-USDC": _position(age_sec=11.5 * 86400, now=now)})
    bot.__dict__["_dark_feed_watch_since"] = now - 4 * DARK
    bot._claim_position_symbol("HIGH-USDC")

    assert bot._abandon_dark_feed_positions(now) == 1
    assert "HIGH-USDC" in bot._owned_symbols, (
        "a symbol this bot genuinely held must keep its ownership"
    )


def test_the_sweep_can_be_disabled():
    now = time.time()
    bot = _bot({"HIGH-USDC": _position(age_sec=11.5 * 86400, now=now)})
    bot.__dict__["_dark_feed_watch_since"] = now - 4 * DARK
    import os
    from unittest import mock

    with mock.patch.dict(os.environ, {"GHOST_DARK_FEED_ABANDON_SEC": "0"}):
        assert bot._dark_feed_abandon_sec() == 0.0
        assert bot._abandon_dark_feed_positions(now) == 0
    assert "HIGH-USDC" in bot.positions


def test_the_threshold_sits_at_the_measured_p99_inter_tick_gap():
    """3600s was chosen from the gap distribution, not picked.

    Over 6h of ``market_stream`` (n=1472): p50 42s, p90 406s, p99 3604s. A
    default below the p99 would reap symbols merely in a slow patch.
    """
    bot = TradingBot.__new__(TradingBot)
    assert bot._dark_feed_abandon_sec() == 3600.0
