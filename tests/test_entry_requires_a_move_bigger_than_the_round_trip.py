"""Entry tested DIRECTION and never MOVE SIZE, so it bought moves too small to pay.

The model-long entry conjunction in ``trading/bot.py`` reads four things:
``direction_prob``, ``exit_conf``, ``net_margin`` and ``delta``. The first two
are confidences, the third is the net_margin head's own arithmetic, and
``delta`` -- which IS the model's forward expected return (``price_mu``, a
dimensionless fraction; see ``_summarise_predictions``) -- was read for its
SIGN alone:

    and delta >= 0.0

So a move that was correctly predicted and too small to pay for itself was
admitted. On this feed that is most ticks. Measured 2026-09-10, median absolute
15-minute return 0.2233% against a round trip of 0.3187% + $0.004047/notional:
only 37.5% of ticks move FURTHER than cost, so on 62.5% of them a perfectly
correct direction call still loses money. No level fix to the direction head
touches that -- it is an arithmetic property of the cost floor.

THE THRESHOLD IS THE MEASURED COST FLOOR AND IS SIZE-DEPENDENT.
``services/roundtrip_cost.py`` measures this account's settled receipts as
``cost_usd = 0.004047 + 0.003187 * notional``, so as a fraction of notional

    c(N) = 0.003187 + 0.004047 / N

which is 0.3862% at a $6.00 clip and 0.8583% at the $0.75 ghost floor. A flat
percentage is wrong at both ends, and this repo has already shipped a flat
0.65%. ``entry_fees`` in that branch is already exactly c(N) for the notional
the entry is about to spend, so the conjunct compares two fractions of the same
notional with no conversion and introduces no new constant.

These tests fail against ``delta >= 0.0`` and pass against
``delta >= min_expected_move``. The third one would also catch the flat-percentage
regression, and the second exists because a gate that refuses everything is a
bug and not safety.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.roundtrip_cost import roundtrip_cost_rate  # noqa: E402
from trading.bot import TradingBot  # noqa: E402

# Same fixture as the sibling entry/exit decision tests.
from tests.test_a_stale_loser_is_measured_not_forecast import _bot  # noqa: E402

SYMBOL = "AERO-USDC"
TOKEN = "0x940181a94A35A4569E4529A3CDfB74e38FD98631"
PRICE = 0.5460656031729441

# 0.99 clears ``enter_threshold``, which is
# ``max(pipeline.decision_threshold, MIN_CONFIDENCE)`` clamped to 0.99. This
# test is about the SIZE conjunct, so the confidence conjuncts are held wide
# open rather than lowered -- no threshold is touched anywhere here.
CONFIDENT = 0.99
# Comfortably past ``min_margin_gate`` (entry_fees * 1.5) and the $0.02
# SMALL_PROFIT_FLOOR, so the margin conjuncts cannot be what answers.
FAT_MARGIN = 0.50


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LIVE_TRADES_DRY_RUN", "1")
    # The ghost exploration fallback is a SEPARATE entry path with its own
    # momentum floor. Leaving it on would let a refused model entry come back
    # as an explore entry and the assertions below would be measuring it.
    monkeypatch.setenv("GHOST_EXPLORE_ENABLED", "0")
    monkeypatch.delenv("ENTRY_MIN_MOVE_COST_MULT", raising=False)


def _decide(bot: TradingBot, *, delta: float) -> dict:
    """One entry cycle on an empty book carrying a predicted move of `delta`."""
    sample = {
        "symbol": SYMBOL,
        "price": PRICE,
        "ts": __import__("time").time(),
        "chain": "base",
        "volume": 5000.0,
    }
    summary = {
        "exit_conf": CONFIDENT,
        "direction_prob": CONFIDENT,
        "delta": delta,
        "price_mu": delta,
        "net_margin": FAT_MARGIN,
        "net_pnl": FAT_MARGIN,
    }

    async def _no_sync(self, *args, **kwargs):
        return None

    with mock.patch.object(
        TradingBot,
        "_resolve_live_trade_asset",
        lambda self, chain, sym, explicit=None: (sym, explicit or TOKEN),
    ), mock.patch.object(TradingBot, "_run_wallet_sync", _no_sync):
        return asyncio.run(
            bot._interpret_predictions(
                None, sample, None, pred_summary=summary, brain_summary={},
            )
        )


def _ghost_bot() -> TradingBot:
    bot = _bot()
    # Ghost, so a passing decision books a simulated position rather than
    # reaching the live executor. The conjunction under test is the same one.
    bot.live_trading_enabled = False
    bot.positions = {}
    return bot


# --------------------------------------------------------------------------
# The bug.
# --------------------------------------------------------------------------

def test_an_entry_is_refused_when_the_predicted_move_is_under_the_round_trip() -> None:
    """A real, correctly-signed, too-small move. Buying it books the fee."""
    bot = _ghost_bot()
    decision = _decide(bot, delta=0.0)

    assert decision["action"] != "enter", (
        "entry admitted a cycle whose own model predicted a forward return of "
        "zero: direction was tested, move size was not"
    )
    assert decision.get("expected_move_clears_cost") is False, decision
    assert SYMBOL not in bot.positions, bot.positions


def test_a_predicted_move_at_half_the_round_trip_is_still_refused() -> None:
    """Not a sign test and not a zero test: half of cost is still a loss."""
    bot = _ghost_bot()
    floor = float(decision_floor(bot))
    decision = _decide(bot, delta=floor * 0.5)

    assert decision["action"] != "enter", decision
    assert decision.get("expected_move_clears_cost") is False, decision


# --------------------------------------------------------------------------
# ...and what the conjunct must NOT do.
# --------------------------------------------------------------------------

def test_a_predicted_move_that_clears_the_round_trip_still_enters() -> None:
    """A gate that refuses everything is a bug, not safety."""
    bot = _ghost_bot()
    floor = float(decision_floor(bot))
    decision = _decide(bot, delta=floor * 2.0)

    assert decision.get("expected_move_clears_cost") is True, decision
    assert decision["action"] == "enter", decision


def test_the_move_floor_is_the_size_aware_cost_and_not_a_flat_percentage() -> None:
    """0.65% flat has already shipped here once. The floor is c(N), exactly."""
    bot = _ghost_bot()
    decision = _decide(bot, delta=0.0)

    floor = float(decision["min_expected_move"])
    fee_rate = float(decision["entry_fee_rate"])
    assert floor == pytest.approx(fee_rate, rel=1e-12), decision
    notional = float(decision.get("trade_notional_usd") or 0.0)
    if notional > 0:
        assert floor == pytest.approx(roundtrip_cost_rate(notional), rel=1e-9), (
            "the floor must be the measured round trip for THIS trade's size, "
            f"not a constant: notional ${notional:.4f}"
        )


def decision_floor(bot: TradingBot) -> float:
    """The cost floor the branch will measure against, read from a dry cycle."""
    probe = _decide(bot, delta=0.0)
    bot.positions.pop(SYMBOL, None)
    return float(probe["min_expected_move"])
