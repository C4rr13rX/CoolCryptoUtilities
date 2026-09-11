"""A position between zero and its own cost had no exit on any clock.

Rule 4 of the held-position branch is the stale clock, and it read::

    elif pnl_pct_held < 0 and held_secs > GHOST_NEG_EXIT_SECONDS:

Strictly negative. Every rule above it needs the position to have MOVED --
to its target (rule 1), to the stop (rule 2), or far enough up to arm
break-even, profit-lock or trailing in triggers.py. So a position that drifted
to +0.08% and stopped satisfied nothing at all, on any clock, forever.

MEASURED on the live book 2026-09-05 16:10. All three open slots were in
exactly that state:

    COMP-USDC   ghost  held 35.6m  realised +0.000%  target +1.97%
    CBBTC-USDC  ghost  held 22.0m  realised +0.077%  target +9.99%
    AERO-USDC   live   held 22.1m  realised -0.418%  target +4.57%

CBBTC's +9.99% target is not reachable in the tens of minutes this pipeline
trades on, so its only surviving exit was the 2% stop -- and while it waited,
its slot refused every further entry on the symbol. The hour to 16:10 recorded
7 entries and ZERO exits, 37 of its 64 refusals being entry-refused-duplicate
and entry-refused-slot-busy against these three symbols; six earlier hours the
same day logged no entry and no exit at all.

The rule now tests the position's own COST rather than the sign of its P/L: a
position that has not cleared the round trip it would have to pay is not
working, whether it is down 1.5% or flat. That is the same fraction rule 1
already requires price to clear, so both ends of the bracket measure against
one cost basis.

What must NOT change, and is asserted below:

  * a winner is still left alone -- 2% clears the ~0.86% round trip at the
    ghost clip and belongs to its target, trailing stop or profit lock;
  * an unknown cost basis (entry_price 0) is still not evidence of anything,
    and is released by the max-hold eviction rather than sold on a number
    nobody can compute;
  * the clock still has to run out;
  * and a LIVE position is still not force-closed at a nothing-move --
    "timed-exit" is not in the protective set, so the live margin gate refuses
    it as `hold-negative`.
"""

from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest

from trading.bot import TradingBot

# The sibling file already builds a TradingBot wired for exactly this branch:
# a neutral pipeline threshold, a ledger that approves live, a portfolio that
# answers balance queries, and a DB whose record_trade_outcome returns truthy
# so an exit actually books. Rebuilding those hundred lines here would give two
# harnesses to keep in step, and the one that drifted would fail on the fixture
# rather than on the rule.
from tests.test_a_stale_loser_is_measured_not_forecast import _bot

# COMP-USDC and CBBTC-USDC as the book actually held them at 16:10.
COMP = "COMP-USDC"
COMP_TOKEN = "0x9e1028F5F1D5eDE59748FFceE5532509976840E0"
COMP_ENTRY = 19.87
COMP_FLAT = 19.87                 # realised +0.000% after 35.6 minutes
COMP_SIZE = 0.07548842476094615

CBBTC = "CBBTC-USDC"
CBBTC_TOKEN = "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf"
CBBTC_ENTRY = 79746.12
CBBTC_DRIFT = 79807.7             # realised +0.077%, against a +9.99% target
CBBTC_SIZE = 2.1921850361462255e-05

STALE_SEC = 900.0
HELD_SEC = 2136.0                 # 35.6 minutes, COMP's actual age


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "1")
    monkeypatch.setenv("GHOST_STALE_EXIT_SECONDS", str(int(STALE_SEC)))
    # The old name overrides the new one, so it must be cleared or the tests
    # below would be timed by whatever an operator left in .env.
    monkeypatch.delenv("GHOST_NEG_EXIT_SECONDS", raising=False)
    monkeypatch.setenv("GHOST_STOP_LOSS_PCT", "0.02")
    monkeypatch.setenv("MIN_HOLD_SECONDS", "300")
    # The operator's force hatch is a separate path with its own tests; it is
    # off here so the rule under test is the only thing that can decide.
    monkeypatch.setenv("MAX_HOLD_FORCE_SECONDS", "0")


def _position(
    *,
    mode: str,
    entry_price: float,
    size: float,
    held_sec: float,
    token: str,
    target_price: float = 0.0,
) -> dict:
    opened = time.time() - held_sec
    return {
        "mode": mode,
        "strategy_id": "atf_static",
        "size": size,
        "entry_price": entry_price,
        "target_price": target_price,
        "entry_ts": opened,
        "ts": opened,
        "trade_id": f"2:{mode}:went-nowhere",
        "route": [token, "USDC"],
        "quote_spent": entry_price * size,
        "base_token_address": token,
    }


def _tick(bot: TradingBot, symbol: str, token: str, price: float):
    """One sample with the model expressing no opinion at all.

    direction_prob / exit_conf at 0.5 is `model_neutral`, which shuts off
    rules 3a and 3b -- without it these tests could pass on the wrong rule.
    """
    sample = {"symbol": symbol, "price": price, "ts": time.time(),
              "chain": "base", "volume": 100.0}
    summary = {"exit_conf": 0.5, "direction_prob": 0.5, "delta": 0.0,
               "net_margin": 0.0, "net_pnl": 0.0}

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot, "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (sym, explicit or token),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, None, pred_summary=summary, brain_summary={},
            )
        )


# --------------------------------------------------------------------------
# The bug: flat is not negative, so nothing ever fired.
# --------------------------------------------------------------------------

def test_a_dead_flat_position_exits_once_its_clock_runs_out() -> None:
    """COMP-USDC: +0.000% after 35.6 minutes, and no rule could touch it."""
    bot = _bot()
    bot.positions[COMP] = _position(
        mode="ghost", entry_price=COMP_ENTRY, size=COMP_SIZE,
        held_sec=HELD_SEC, token=COMP_TOKEN, target_price=20.26128179368896,
    )

    decision = _tick(bot, COMP, COMP_TOKEN, COMP_FLAT)

    assert decision["action"] == "exit", (
        "a position at exactly break-even for 35.6 minutes held its symbol "
        "slot against every further entry and had no exit on any clock"
    )
    assert decision.get("exit_reason") == "timed-exit", decision


def test_a_drift_that_never_covers_the_round_trip_exits() -> None:
    """CBBTC-USDC: +0.077% against a +9.99% target it cannot reach in minutes."""
    bot = _bot()
    bot.positions[CBBTC] = _position(
        mode="ghost", entry_price=CBBTC_ENTRY, size=CBBTC_SIZE,
        held_sec=HELD_SEC, token=CBBTC_TOKEN, target_price=87708.77008199999,
    )

    decision = _tick(bot, CBBTC, CBBTC_TOKEN, CBBTC_DRIFT)

    assert decision["action"] == "exit", (
        "+0.077% does not cover the round trip and never reaches a +9.99% "
        "target inside the horizon this pipeline trades on"
    )
    assert decision.get("exit_reason") == "timed-exit", decision


# --------------------------------------------------------------------------
# ...and the three things widening it must not break.
# --------------------------------------------------------------------------

def test_a_position_that_cleared_its_cost_is_left_to_its_target() -> None:
    """Up 2% is a working position, not a stale one. It keeps its slot."""
    bot = _bot()
    bot.positions[COMP] = _position(
        mode="ghost", entry_price=COMP_ENTRY, size=COMP_SIZE,
        held_sec=HELD_SEC, token=COMP_TOKEN,
    )

    decision = _tick(bot, COMP, COMP_TOKEN, COMP_ENTRY * 1.02)

    assert decision.get("exit_reason") != "timed-exit", decision


def test_an_unknown_cost_basis_is_still_not_evidence() -> None:
    """entry_price 0 reads pnl_pct_held 0.0 -- which is not a measurement."""
    bot = _bot()
    pos = _position(
        mode="ghost", entry_price=COMP_ENTRY, size=COMP_SIZE,
        held_sec=HELD_SEC, token=COMP_TOKEN,
    )
    pos["entry_price"] = 0.0
    bot.positions[COMP] = pos

    decision = _tick(bot, COMP, COMP_TOKEN, COMP_FLAT)

    assert decision.get("exit_reason") != "timed-exit", decision


def test_the_clock_still_has_to_run_out() -> None:
    """Same flat price, fresh position. Minutes is the horizon, not seconds."""
    bot = _bot()
    bot.positions[COMP] = _position(
        mode="ghost", entry_price=COMP_ENTRY, size=COMP_SIZE,
        held_sec=STALE_SEC - 300.0, token=COMP_TOKEN,
    )

    decision = _tick(bot, COMP, COMP_TOKEN, COMP_FLAT)

    assert decision.get("exit_reason") != "timed-exit", decision


def test_a_live_nothing_move_is_still_held_not_sold() -> None:
    """The widened rule proposes; the live margin gate still disposes.

    Closing a live position costs gas whatever the reason, and "timed-exit" is
    deliberately absent from the protective set, so the cost gate refuses it.
    Widening rule 4 must not smuggle the forced closes back in.
    """
    bot = _bot()
    bot.positions[CBBTC] = _position(
        mode="live", entry_price=CBBTC_ENTRY, size=CBBTC_SIZE,
        held_sec=HELD_SEC, token=CBBTC_TOKEN,
    )

    decision = _tick(bot, CBBTC, CBBTC_TOKEN, CBBTC_DRIFT)

    assert decision["status"] == "hold-negative", decision
    assert CBBTC in bot.positions, "the live position must survive the refusal"
