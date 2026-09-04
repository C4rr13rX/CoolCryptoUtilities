"""One strategy's entry must not destroy another strategy's working position.

The same-strategy case was fixed in c6dfa38 and it worked: measured over the
24h to 2026-09-04 11:00, ZERO same-strategy evictions occurred in the last 8h,
against 616 in the 24h window as a whole.

What is left is the cross-strategy case, and it is now the whole of the leak::

    792  ghost entries          (24h)
     83  ghost exits            (24h)
    683  slot evictions         (24h)  -- 616 same-strategy, 67 CROSS-strategy

    last 8h:  60 entries, 22 exits, 0 same-strategy evictions, 24 cross

So in the current window the surviving path destroys slightly more positions
than the exit path completes. ``StrategyLedger.record()`` is only ever called
from the exit path, and both graduation and re-arming need 20 ghost trades from
ONE strategy, so every eviction is an outcome that credits nobody. atf_static --
the only strategy on this account that has ever spent real money -- is demoted
and needs 20 FRESH ghost trades to re-arm; it had booked none since 06:15 while
making 5 ghost entries in the preceding 2h. That is link 5.

The rule pinned here, with both carve-outs, each sized against the same 24h of
``trading_ops`` rather than assumed:

  * a ghost entry for a symbol another strategy is holding is REFUSED, logged
    as ``entry-refused-slot-busy``, and the sample falls through to the
    held-position branch so the held position's bracket, target and timed exit
    are still evaluated and it can actually finish;

  * EXCEPT a LIVE entry, which still displaces a ghost position. Refusing it
    would let a simulated position lock a graduated strategy out of a symbol,
    which is link 6 and was 7 of the 9 live-capable symbols on 2026-09-02. It
    is cheap to keep: of the 53 young cross-strategy evictions in 24h, 50 were
    ghost-over-ghost and only 3 were live-over-ghost;

  * EXCEPT a position past ``MAX_HOLD_SECONDS``, which stays evictable. Exits
    here are sample-driven, so a symbol whose feed goes quiet is never closed
    and eviction is the ONLY thing that frees its slot. The book on 2026-09-04
    held HIGH-USDC for 989,066s (11.4 days) with no feed at all, and 5 of its
    13 slots were past the 3600s max hold. Protecting those would have traded
    an evidence leak for a permanently blocked symbol.

  * a position with no strategy_id is NOT protected: its outcome books as
    "unclassified" and counts toward no strategy's graduation.
"""

from __future__ import annotations

import time

from tests.test_a_strategy_does_not_clobber_its_own_position import (  # noqa: E402
    SYMBOL,
    _bot,
    _directive,
    _enter,
    _GhostOnlyLedger,
    _live_execution,  # noqa: F401  -- autouse fixture
    _position,
    _settling_bot,
)


def _ghost_bot():
    bot = _bot()
    bot.strategy_ledger = _GhostOnlyLedger()
    return bot


# --------------------------------------------------------------------------
# The 50 ghost-over-ghost evictions.
# --------------------------------------------------------------------------

def test_a_ghost_entry_does_not_evict_another_strategys_position() -> None:
    """The position survives intact, so it can still reach an exit."""
    bot = _ghost_bot()
    original = _position("rsi_reversal@12h", mode="ghost")
    bot.positions[SYMBOL] = original

    _enter(bot, _directive("atf_static"))

    held = bot.positions[SYMBOL]
    assert held["trade_id"] == "ghost-original-trade-id"
    assert held["strategy_id"] == "rsi_reversal@12h"
    assert held["size"] == 100.0
    assert held["entry_price"] == 0.9
    assert held["entry_ts"] == original["entry_ts"]
    assert not [r for r in bot.db.logged if r.get("status") == "position-released"]
    # The refusal row is the environment-independent half of this assertion.
    # "No release happened" is also true when the entry failed downstream for
    # its own reasons -- the address book refusing a token, say -- so on its own
    # it would pass against the unfixed code and pin nothing.
    assert [r for r in bot.db.logged if r.get("status") == "entry-refused-slot-busy"]


def test_the_slot_busy_refusal_is_recorded() -> None:
    """An unlogged refusal is indistinguishable from a lane that wanted nothing.

    That silence is what hid the same-strategy case for weeks.
    """
    bot = _ghost_bot()
    bot.positions[SYMBOL] = _position("rsi_reversal@12h", mode="ghost")

    _enter(bot, _directive("atf_static"))

    rows = [r for r in bot.db.logged if r.get("status") == "entry-refused-slot-busy"]
    assert len(rows) == 1
    details = rows[0]["details"]
    assert details["reason"] == "symbol_held_by_another_strategy"
    assert details["strategy_id"] == "atf_static"
    assert details["held_strategy_id"] == "rsi_reversal@12h"
    assert details["held_mode"] == "ghost"
    # Units: seconds, not ms, and measured from the sample clock.
    assert 0.0 <= details["held_secs"] < 120.0
    assert details["max_hold_sec"] == 3600.0


def test_the_refusal_does_not_masquerade_as_the_duplicate_one() -> None:
    """Two different bugs, two different rows -- forensics needs them apart."""
    bot = _ghost_bot()
    bot.positions[SYMBOL] = _position("rsi_reversal@12h", mode="ghost")

    _enter(bot, _directive("atf_static"))

    assert not [
        r for r in bot.db.logged if r.get("status") == "entry-refused-duplicate"
    ]


# --------------------------------------------------------------------------
# Carve-out 1: real money still displaces a simulation (link 6).
# --------------------------------------------------------------------------

def test_a_live_entry_still_displaces_another_strategys_ghost_position() -> None:
    """A simulated position must never lock a graduated strategy out of a symbol.

    Asserted as "this refusal did not fire", not as "the entry completed". What
    happens after the predicate lets the entry through belongs to the entry
    path, and the release itself is already pinned by
    ``test_a_different_strategy_still_releases`` in the sibling module. Pinning
    it again here would only couple this test to a swap that reaches the
    address book and the chain.
    """
    bot, swapper = _settling_bot()
    bot.positions[SYMBOL] = _position("donchian_breakout@5d", mode="ghost")

    _enter(bot, _directive("atf_static"), swapper=swapper)

    assert not [
        r for r in bot.db.logged if r.get("status") == "entry-refused-slot-busy"
    ]


# --------------------------------------------------------------------------
# Carve-out 2: a slot that can no longer exit must not be locked forever.
# --------------------------------------------------------------------------

def test_a_position_past_max_hold_is_still_evictable() -> None:
    """Exits are sample-driven; a quiet feed would otherwise block the symbol.

    HIGH-USDC sat in the live book for 989,066s with no feed and no strategy.
    Eviction is the only thing that clears one of those.
    """
    bot = _ghost_bot()
    bot.positions[SYMBOL] = _position(
        "rsi_reversal@12h", mode="ghost", held_secs=4000.0
    )

    _enter(bot, _directive("atf_static"))

    assert not [
        r for r in bot.db.logged if r.get("status") == "entry-refused-slot-busy"
    ]
    assert not [
        r for r in bot.db.logged if r.get("status") == "entry-refused-duplicate"
    ]


def test_the_max_hold_boundary_protects_a_position_just_under_it() -> None:
    """3599s is still a working position; the cutoff is not off by an hour."""
    bot = _ghost_bot()
    bot.positions[SYMBOL] = _position(
        "rsi_reversal@12h", mode="ghost", held_secs=3500.0
    )

    _enter(bot, _directive("atf_static"))

    assert not [r for r in bot.db.logged if r.get("status") == "position-released"]
    assert [r for r in bot.db.logged if r.get("status") == "entry-refused-slot-busy"]


# --------------------------------------------------------------------------
# Unattributed and free slots keep their existing behaviour.
# --------------------------------------------------------------------------

def test_an_unattributed_position_is_still_evictable() -> None:
    """Its outcome books as "unclassified" and credits no strategy's graduation."""
    bot = _ghost_bot()
    bot.positions[SYMBOL] = _position("", mode="ghost")

    _enter(bot, _directive("atf_static"))

    assert not [
        r for r in bot.db.logged if r.get("status") == "entry-refused-slot-busy"
    ]


def test_an_entry_on_a_free_symbol_is_untouched() -> None:
    bot = _ghost_bot()
    assert SYMBOL not in bot.positions

    _enter(bot, _directive("atf_static"))

    assert not [
        r for r in bot.db.logged if r.get("status") == "entry-refused-slot-busy"
    ]
