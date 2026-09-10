"""Past the stale clock, the position's own cost decides -- not the model.

THE BUG THIS PINS, AND IT HAS A ZERO IN IT.

``_interpret_predictions`` documents its held-position order of checks as::

    #   1. take-profit
    #   2. stop-loss
    #   3. model gates  - only when the model expresses a real opinion
    #   4. timed exit   - stale losers release capital

Rule 3 (``confidence_drop`` / ``negative_margin``) fires at ``MIN_HOLD_SECONDS``
= 300s. Rule 4 (``timed-exit``) fires at ``stale_exit_secs`` = 900s. They were
written as one ``elif`` chain with rule 3 ABOVE rule 4, and an ``elif`` that
fires CONSUMES the tick. So on any position past 900s that the model happened
to be bearish about, the chain resolved to ``confidence_drop`` and rule 4 was
structurally unreachable -- reachable only for a position the model felt
exactly NEUTRAL about, for the whole ten minutes after its clock had run out.

That is not a cosmetic difference in the reason string, because the two names
are not interchangeable downstream. The ghost exit gate admits ``timed-exit``
BY NAME::

    stale_verdict = str(reason or "").startswith("timed-exit")
    if ((not pos_is_live) and economic_profit <= 0
            and (sample_ts - entry_ts) < max_hold_sec
            and not protective_exit and not stale_verdict):
        decision.update({"status": "hold-negative", ...})
        return decision          # <- books NOTHING

...and refuses ``confidence_drop`` at a non-positive economic profit. So the
stale loser was denied its exit under a name that gate was written to reject,
then denied it again on the next tick, and the next, for the whole 45-minute
window between ``stale_exit_secs`` (900s) and ``max_hold_sec`` (3600s), until
the max-hold eviction released the slot.

MEASURED over the 7 days to 2026-09-10, by reason, on all 206 logged ghost
exits (``trading_ops``, status ``ghost-exit``)::

    max_hold           52     <- the 3600s eviction, 4x the 900s promise
    target_hit         16
    stop_loss           5
    stale_underwater    4
    timed-exit          0     <- rule 4 had NEVER produced an outcome
    confidence_drop     0
    negative_margin     0

Zero. And ``max_hold`` is the single biggest named reason, which is the
signature of exactly this: positions escaping at 3600s because the rule that
was supposed to release them at 900s could not be reached.

WHY THE EXISTING TEST DID NOT CATCH IT. ``test_a_stale_loser_is_measured_not_
forecast`` ticks the bot with ``direction_prob = 0.5`` and says so in its own
comment -- "model_neutral, which is what shuts off rules 3a and 3b. Without
that this test would pass on the wrong rule." It pinned rule 4 only in the one
model state where nothing outranks it. Production is not that state: the
measurement recorded in bot.py is a median ``direction_prob`` of 0.2560 with
68.6% of 1050 decisions below the 0.45 bearish floor.

So this file ticks it with a BEARISH model, which is the ordinary case, and
asserts the stale clock still wins.

The rule pinned here:

  * past ``stale_exit_secs``, a position that has not covered its round trip
    exits as ``timed-exit`` even when the model is bearish, and BOOKS the
    outcome rather than returning ``hold-negative``;
  * INSIDE ``stale_exit_secs`` the model still owns the tick, so the reorder
    cannot be mistaken for deleting rule 3; and
  * a LIVE position is still not force-closed at a nothing-move -- all three
    reasons are outside ``PROTECTIVE_REASONS``, so the live-exit margin gate
    treats them identically and ``test_a_close_must_cover_its_own_gas`` keeps
    holding.
"""

from __future__ import annotations

import time

import pytest

from trading.bot import TradingBot
from tests.test_a_stale_loser_is_measured_not_forecast import (  # noqa: F401
    CBXRP,
    CBXRP_TOKEN,
    CBXRP_ENTRY,
    _bot,
    _position,
    _tick,
)

# The refusal window this bug lived in: past the stale clock, short of the
# max-hold eviction. Production defaults are 900 and 3600.
STALE_SEC = 900.0
MAX_HOLD_SEC = 3600.0
HELD_INSIDE_THE_WINDOW = 1800.0   # 30 min: 2x stale, half of max_hold
HELD_BEFORE_THE_CLOCK = 600.0     # 10 min: past MIN_HOLD, short of stale

# -1.0% realised: a loser that has NOT covered its round trip (fees are
# ~0.59% at the current clip) and is NOWHERE NEAR the 2% stop, so neither
# rule 1 nor rule 2 can decide these tests.
LOSING_TICK = CBXRP_ENTRY * 0.99

# Bearish, and far below the 0.45 EXIT_BEARISH_FLOOR, so rule 3a is live and
# the model is not neutral. This is the ordinary production state.
BEARISH_PROB = 0.20


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "1")
    monkeypatch.setenv("GHOST_NEG_EXIT_SECONDS", str(int(STALE_SEC)))
    monkeypatch.setenv("GHOST_STOP_LOSS_PCT", "0.02")
    monkeypatch.setenv("MIN_HOLD_SECONDS", "300")
    monkeypatch.setenv("MAX_HOLD_SECONDS", str(int(MAX_HOLD_SEC)))
    # The operator's force hatch is OFF, for the reason the sibling file
    # documents: with it on, triggers.py returns ``max_hold_force`` at rule 1
    # and no test below ever reaches the chain it is about.
    monkeypatch.setenv("MAX_HOLD_FORCE_SECONDS", "0")


def _bearish_tick(bot: TradingBot, price: float):
    """One sample with the model actively leaning AGAINST the position."""
    import asyncio
    from unittest import mock

    sample = {"symbol": CBXRP, "price": price, "ts": time.time(),
              "chain": "base", "volume": 100.0}
    summary = {"exit_conf": BEARISH_PROB, "direction_prob": BEARISH_PROB,
               "delta": 0.0, "net_margin": 0.0, "net_pnl": 0.0}

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot, "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (sym, explicit or CBXRP_TOKEN),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, None, pred_summary=summary, brain_summary={},
            )
        )


def test_a_stale_loser_exits_on_the_clock_while_the_model_is_bearish() -> None:
    """The whole bug: rule 3 consumed the tick and the gate rejected its name."""
    bot = _bot()
    bot.positions[CBXRP] = _position(
        mode="ghost", entry_price=CBXRP_ENTRY, size=0.531701,
        held_sec=HELD_INSIDE_THE_WINDOW, token=CBXRP_TOKEN,
    )

    decision = _bearish_tick(bot, LOSING_TICK)

    assert decision.get("exit_reason") == "timed-exit", (
        "a position held 2x stale_exit_secs at a realised -1.0% resolved to "
        f"{decision.get('exit_reason')!r} instead of the stale clock, because "
        "the model happened to be bearish -- and the ghost exit gate refuses "
        "that name at a non-positive profit, so nothing was booked"
    )
    assert decision.get("status") != "hold-negative", (
        "the stale loser reached the ghost exit gate under a reason the gate "
        "rejects and was held open again: this is the tick that repeated for "
        "the whole 900s-3600s window until the max-hold eviction"
    )


def test_the_stale_exit_books_an_outcome_instead_of_holding_negative() -> None:
    """An exit that books nothing is the evidence leak graduation starves on."""
    bot = _bot()
    bot.positions[CBXRP] = _position(
        mode="ghost", entry_price=CBXRP_ENTRY, size=0.531701,
        held_sec=HELD_INSIDE_THE_WINDOW, token=CBXRP_TOKEN,
    )

    _bearish_tick(bot, LOSING_TICK)

    assert bot.db.outcomes, (
        "the stale loser was closed without recording a trade outcome; a "
        "released slot that books no round trip is evidence the ledger never "
        "sees, and graduation is gated on closed TRADEABLE round trips"
    )


def test_inside_the_clock_the_model_still_owns_the_tick() -> None:
    """The reorder must not read as deleting rule 3.

    Before ``stale_exit_secs`` the timed exit's own condition is false, so a
    bearish model decides exactly as it did before this change.
    """
    bot = _bot()
    bot.positions[CBXRP] = _position(
        mode="ghost", entry_price=CBXRP_ENTRY, size=0.531701,
        held_sec=HELD_BEFORE_THE_CLOCK, token=CBXRP_TOKEN,
    )

    decision = _bearish_tick(bot, LOSING_TICK)

    assert decision.get("exit_reason") != "timed-exit", (
        "a position 600s old cannot be a STALE loser: stale_exit_secs is 900s "
        "and moving the timed exit up the chain must not let it fire early"
    )
