"""Half the book closed on a market move smaller than the fee it paid.

The held-position branch of ``trading/bot.py`` has four exits. Three of them
price the trade: ``target_hit`` requires ``price > entry * (1 + fees)``,
``timed-exit`` requires ``pnl_pct_held < fees``, and the stop is a price. The
fourth pair -- ``confidence_drop`` and ``negative_margin`` -- close a position
because the MODEL changed its mind, and asked nothing at all about whether the
position had moved far enough to pay for the closing.

That would be a rounding error if the two rules ran at the same time. They do
not. ``timed-exit`` waits for ``GHOST_STALE_EXIT_SECONDS`` (900s); the model
rules fire at ``MIN_HOLD_SECONDS`` (300s). So the cost-blind rule pre-empted
the cost-aware one by ten minutes on every held position.

MEASURED 2026-09-07 over the 117 closed round trips of the last 7 days on
symbols the live lane could actually have traded -- ``stop_is_unenforceable``
False, the same population graduation scores:

    whole tradeable book                 n=117   net -0.234002
    closed on |gross| < the fee paid     n= 58   net -0.749246
    the rest                             n= 59   net +0.515244

Those 58 moved -0.003928 of gross BETWEEN THEM. The market did nothing; they
paid 0.745317 in fees to find out. They are not losing trades, they are the fee
booked 58 times, and they are the whole of the book's loss and more. By exit
reason: confidence_drop 29, timed 17, negative_margin 5.

``direction_prob < bearish_floor`` was not an opinion either. Over 1050
decisions in 24h the median ``direction_prob`` was 0.2560 and 68.6% sat below
the 0.45 floor, so for any position past 300s the condition was very nearly a
constant.

THE DEFERRAL IS WHAT PAYS, measured over 879 ghost entries of the last 7 days,
each walked forward on its own ``market_stream`` prices:

    still inside +/-fees at 300s     650 of 879   73.9%
    ...of those, by 900s:
        escaped UP past +fees        102          15.7%   decidable winner
        escaped DOWN past -fees       36           5.5%   real loss to cut
        still inside the band        512          78.8%   -> timed-exit

102 winners to 36 losers, 2.8:1, out of trades that were all being closed flat
at 300s for a certain -0.386%.

The band is symmetric because what it measures is "has this position moved
enough for its closing to mean anything", not "is it winning" -- a move DOWN
through the cost is a real loss and these rules SHOULD cut it.

What must not change, and is asserted below: the stop still outranks
everything, the stale clock still releases an in-band position so nothing
becomes immortal, an unknown cost basis is still not deferred, and a position
that HAS moved is still closed on the model's word.
"""

from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest

from trading.bot import TradingBot

# Same fixture as the sibling exit tests: a neutral pipeline threshold, a
# ledger that approves live, a portfolio that answers balance queries and a DB
# whose record_trade_outcome returns truthy so an exit actually books.
from tests.test_a_stale_loser_is_measured_not_forecast import _bot

COMP = "COMP-USDC"
COMP_TOKEN = "0x9e1028F5F1D5eDE59748FFceE5532509976840E0"
COMP_ENTRY = 19.87
COMP_SIZE = 0.07548842476094615

STALE_SEC = 900.0
MIN_HOLD_SEC = 300.0
# Past MIN_HOLD_SECONDS, so the model rules are live; short of the stale clock,
# so `timed-exit` cannot be what answers. This is the window the whole finding
# lives in.
HELD_SEC = 480.0
STOP_PCT = 0.02
# trading/triggers.py picks its stop from the BOT's live flag, not the
# position's, so this is the one that actually binds on the shared fixture.
LIVE_STOP_PCT = 0.015


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "1")
    monkeypatch.setenv("GHOST_STALE_EXIT_SECONDS", str(int(STALE_SEC)))
    # The old name overrides the new one, so it must be cleared or these tests
    # would be timed by whatever an operator left in .env.
    monkeypatch.delenv("GHOST_NEG_EXIT_SECONDS", raising=False)
    monkeypatch.setenv("GHOST_STOP_LOSS_PCT", str(STOP_PCT))
    monkeypatch.setenv("LIVE_STOP_LOSS_PCT", str(LIVE_STOP_PCT))
    monkeypatch.setenv("MIN_HOLD_SECONDS", str(int(MIN_HOLD_SEC)))
    monkeypatch.setenv("EXIT_BEARISH_FLOOR", "0.45")
    # The operator's force hatch is a separate path with its own tests.
    monkeypatch.setenv("MAX_HOLD_FORCE_SECONDS", "0")


def _position(*, mode: str = "ghost", entry_price: float = COMP_ENTRY,
              held_sec: float = HELD_SEC) -> dict:
    opened = time.time() - held_sec
    return {
        "mode": mode,
        "strategy_id": "atf_static",
        "size": COMP_SIZE,
        "entry_price": entry_price,
        # 0.0 keeps `target_hit` out of the way; it has its own tests.
        "target_price": 0.0,
        "entry_ts": opened,
        "ts": opened,
        "trade_id": f"2:{mode}:inside-the-cost-band",
        "route": [COMP_TOKEN, "USDC"],
        "quote_spent": entry_price * COMP_SIZE,
        "base_token_address": COMP_TOKEN,
    }


def _tick(bot: TradingBot, price: float, *, direction_prob: float,
          net_margin: float = 0.0):
    """One sample carrying a model opinion.

    ``exit_conf`` is held at 0.5 and ``direction_prob`` is what moves, so
    ``model_neutral`` (which needs BOTH within 0.02 of 0.5) is False and the
    rules under test are actually reachable.
    """
    sample = {"symbol": COMP, "price": price, "ts": time.time(),
              "chain": "base", "volume": 100.0}
    summary = {"exit_conf": 0.5, "direction_prob": direction_prob,
               "delta": 0.0, "net_margin": net_margin, "net_pnl": net_margin}

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot, "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (sym, explicit or COMP_TOKEN),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, None, pred_summary=summary, brain_summary={},
            )
        )


def _fees(bot: TradingBot) -> float:
    """The size-aware round-trip rate the exit branch measures against."""
    return float(bot._roundtrip_fee_rate(notional_hint=None))


# --------------------------------------------------------------------------
# The bug: an opinion spending a round trip the move never earned.
# --------------------------------------------------------------------------

def test_a_bearish_model_cannot_close_a_position_that_never_moved() -> None:
    """Dead flat at 8 minutes. Closing it books the fee and nothing else."""
    bot = _bot()
    bot.positions[COMP] = _position()

    decision = _tick(bot, COMP_ENTRY, direction_prob=0.05)

    assert decision.get("exit_reason") != "confidence_drop", (
        "the model closed a position that had not moved at all, paying a full "
        "round trip for a gross return of zero -- 29 of the 58 fee-burn exits "
        "measured on the tradeable book"
    )
    assert decision["action"] != "exit", decision
    deferred = decision.get("exit_deferred_inside_cost")
    assert deferred and deferred["would_have_been"] == "confidence_drop", decision


def test_a_predicted_loss_cannot_close_a_position_that_never_moved() -> None:
    """The same rule for `negative_margin`: a forecast is not a move either."""
    bot = _bot()
    bot.positions[COMP] = _position()

    # Above the bearish floor, so `confidence_drop` cannot be what answers,
    # and a predicted margin well below -fees so `negative_margin` is live.
    decision = _tick(bot, COMP_ENTRY, direction_prob=0.60, net_margin=-0.50)

    assert decision.get("exit_reason") != "negative_margin", decision
    assert decision["action"] != "exit", decision
    deferred = decision.get("exit_deferred_inside_cost")
    assert deferred and deferred["would_have_been"] == "negative_margin", decision


# --------------------------------------------------------------------------
# ...and everything the deferral must not break.
# --------------------------------------------------------------------------

def test_a_move_up_through_the_cost_is_still_closed_on_the_models_word() -> None:
    """Deferring is not switching the rule off. Past +fees it must still fire."""
    bot = _bot()
    bot.positions[COMP] = _position()
    up = COMP_ENTRY * (1.0 + _fees(bot) * 2.0)

    decision = _tick(bot, up, direction_prob=0.05)

    assert decision.get("exit_reason") == "confidence_drop", (
        "a position that has cleared its round trip is a decidable trade and "
        "the model is still allowed to bank it"
    )


def test_a_move_down_through_the_cost_is_still_cut() -> None:
    """The band is symmetric: a real loss inside the stop is still cut."""
    bot = _bot()
    bot.positions[COMP] = _position()
    # Past -fees but inside the protective stop, so `trading/triggers.py`
    # cannot be what answers and the assertion is about the model rule. The
    # trigger module picks its threshold from the BOT's live flag rather than
    # the position's, so the tighter LIVE_STOP_LOSS_PCT is the one that binds
    # on this fixture -- hence the margin against that value, not STOP_PCT.
    drop = _fees(bot) * 1.3
    down = COMP_ENTRY * (1.0 - drop)
    assert drop < LIVE_STOP_PCT < STOP_PCT, drop

    decision = _tick(bot, down, direction_prob=0.05)

    # `reason`, not `exit_reason`: this fixture is live-approved, so the exit
    # the rule proposes is then judged by the live margin gate and refused as
    # `hold-negative` -- a 1.12% move does not cover gas at this clip. That
    # refusal is the gate doing its job and has its own tests; what belongs to
    # THIS rule is that it proposed the exit rather than deferring it.
    assert decision.get("reason") == "confidence_drop", (
        "a move down through the round trip is a real loss and the model rule "
        "is what cuts it"
    )
    assert "exit_deferred_inside_cost" not in decision, decision


def test_the_stop_still_outranks_the_deferral() -> None:
    """A deferral must never be able to swallow the protective bracket."""
    bot = _bot()
    bot.positions[COMP] = _position()

    decision = _tick(bot, COMP_ENTRY * (1.0 - STOP_PCT * 1.5), direction_prob=0.05)

    assert str(decision.get("exit_reason") or "").startswith("stop_loss"), decision
    assert "exit_deferred_inside_cost" not in decision, decision


def test_a_deferred_position_is_still_released_by_its_clock() -> None:
    """Nothing becomes immortal: the stale clock still takes an in-band position.

    This is the whole safety argument for deferring, and it is the bug this
    test caught: an `elif` that fires CONSUMES the tick, so the first draft of
    the deferral sat above `timed-exit` with no upper bound and swallowed it on
    every sample. A bearish model would have pinned the position open forever.

    Past `stale_exit_secs` the deferral stops applying and the chain resolves.

    WHICH RULE ANSWERS CHANGED 2026-09-10, AND THE TEST WAS THE STALE HALF.
    This used to assert `confidence_drop`, and noted that the model rule
    "outranks the clock". That ordering was itself the defect: rule 3 fires at
    MIN_HOLD (300s) and rule 4 at stale_exit_secs (900s), so on every position
    past the clock that the model was bearish about, `timed-exit` was
    unreachable. Measured over 7 days on all 206 logged ghost exits, it had
    fired ZERO times, while `max_hold` -- the 3600s eviction -- was the single
    biggest named reason. `timed-exit` now sits above the opinion rules and
    this fixture resolves to it.

    The assertion below is therefore STRONGER than the one it replaces, not
    weaker, and note what else moved with it. This fixture is flat, so its
    economic profit is non-positive and the ghost exit gate refused the old
    `confidence_drop` as `hold-negative` and returned WITHOUT booking -- which
    is why `action` used to read "hold" here. The gate admits `timed-exit` by
    name (`stale_verdict`), so the same fixture now books a real outcome. A
    released slot that records no round trip is the evidence leak graduation
    starves on.

    The claim this test exists for is untouched: the deferral does not outlive
    the clock that bounds it.
    """
    bot = _bot()
    bot.positions[COMP] = _position(held_sec=STALE_SEC + 120.0)

    decision = _tick(bot, COMP_ENTRY, direction_prob=0.05)

    assert "exit_deferred_inside_cost" not in decision, (
        "the deferral outlived the clock that is supposed to bound it, so a "
        "bearish model would pin the position open for as long as it stayed "
        "bearish -- the immortal-position failure this repo has paid for twice"
    )
    # An exit IS proposed past the clock, and it now carries the cost-aware
    # verdict rather than the model's opinion, so the ghost exit gate admits it
    # and the position books its outcome instead of holding negative.
    assert decision.get("exit_reason") == "timed-exit", (
        "past stale_exit_secs the position's own failure to cover its round "
        "trip must decide, not whether the model happens to be bearish: while "
        "rule 3 outranked rule 4, `timed-exit` produced ZERO of 206 ghost "
        f"exits in 7 days -- got {decision.get('exit_reason')!r}"
    )
    assert decision.get("action") == "exit", (
        "a flat position past its clock was released without booking a round "
        "trip, because the gate rejects the reason rule 3 gave it"
    )


def test_an_unknown_cost_basis_is_not_deferred() -> None:
    """entry_price 0 reads pnl_pct_held 0.0, which is not a measurement.

    Deferring on it would pin the position inside the band -- `timed-exit`
    also requires a price, so the only thing left to release it would be the
    max-hold eviction.

    On this path the accounting guard answers first (entry_price 0 against an
    observed 19.87 is an unusable ratio) and the exit chain is never reached at
    all, which is the pre-existing behaviour. What this test pins is the one
    thing that is mine to get right: the deferral does not fire, so it cannot
    be what holds an unpriced position.
    """
    bot = _bot()
    pos = _position()
    pos["entry_price"] = 0.0
    bot.positions[COMP] = pos

    decision = _tick(bot, COMP_ENTRY, direction_prob=0.05)

    assert "exit_deferred_inside_cost" not in decision, decision
