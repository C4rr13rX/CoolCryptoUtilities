"""A clamped limit exit must book the CLAMPED gross, or it books nothing at all.

THIS IS THE TEST THAT THE UNIT TEST NEXT DOOR COULD NOT BE.
``tests/test_a_ghost_take_profit_cannot_fill_past_its_own_limit.py`` calls
``limit_exit_fill_price`` directly and passes. It passed while the seam it
feeds was broken, because a function verified in isolation proves the function
and NOT the wiring. The wiring is what this file tests.

5504769 added the clamp and bound it to ``exit_price_effective``. The ghost
booking branch of ``trading/bot.py`` then computed:

    gross_profit = (price - entry_price) * exit_size      # <-- RAW price

while the SAME row reported ``exit_price=exit_price_effective`` and handed both
to ``services.trading_accounting.validate_outcome_math``. Two defects in one
line:

  (1) the clamp never reached the P/L, so the limit-discipline fix was a no-op
      for the only number graduation reads; and

  (2) ``validate_outcome_math`` cross-checks ``(exit_price - entry_price) *
      quantity`` against ``gross_profit`` to 1e-8. The two disagree BY
      CONSTRUCTION on every overshoot, so it returned ``gross_profit_mismatch``
      and the exit fell to the ``hold-accounting-invalid`` return: THE POSITION
      NEVER CLOSES.

Historically 12 of 14 take-profit exits overshot their 1.05 target, so (2)
would have refused nearly every profitable ghost exit the moment production
reloaded -- the same signature as "the ghost lane is not closing", arriving
from a commit whose message says it fixed the book.

Caught at 0 rows damaged; the incidence query over ``trading_ops`` since the
commit returned 0.

Each test below asserts the composition, and each one FAILS against the old
`(price - entry_price)` arithmetic. That is the point: a test that passes both
ways proves nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.trading_accounting import validate_outcome_math  # noqa: E402
from trading.bot import limit_exit_fill_price  # noqa: E402

FEE = 0.003187          # one leg, the receipts-derived variable rate
QTY = 3.0
GAS = 0.004047          # the fixed leg of the round-trip cost


def _book_ghost_exit(*, entry: float, target: float, tick: float,
                     qty: float = QTY, reason: str = "take_profit_limit"):
    """Reproduce the ghost booking branch of ``trading/bot.py``.

    Mirrors the real sequence: clamp first, then take gross FROM THE CLAMPED
    PRICE, then validate. The single line under test is the ``gross_profit``
    one -- everything else here is the surrounding code's actual shape.
    """
    exit_price_effective = limit_exit_fill_price(
        price=tick, target=target, entry=entry, fee_rate=FEE,
        reason=reason, is_live=False)
    notional = max(qty * entry, 1e-9)
    gross_profit = (exit_price_effective - entry) * qty
    fee_cost = max(notional * FEE, 0.0) + GAS
    net_profit = gross_profit - fee_cost
    valid, why = validate_outcome_math(
        entry_price=entry, exit_price=exit_price_effective, quantity=qty,
        gross_profit=gross_profit, fee_cost=fee_cost, net_profit=net_profit,
        base_token="TEST", quote_token="USDC")
    return exit_price_effective, gross_profit, valid, why


def test_a_clamped_limit_exit_still_books_a_valid_outcome():
    """The acceptance case. Old arithmetic -> gross_profit_mismatch."""
    entry, target, tick = 100.0, 105.0, 220.0
    eff, gross, valid, why = _book_ghost_exit(entry=entry, target=target, tick=tick)

    # The clamp applied: we did NOT book the 220 tick.
    assert eff < tick
    assert eff == pytest.approx(target * (1.0 + FEE))

    # ...and the gross agrees with the price we said we filled at.
    assert gross == pytest.approx((eff - entry) * QTY)
    assert valid, why
    assert why == "valid"

    # The old line booked the tick. Prove that it is exactly what the
    # validator rejects -- this is the regression, reproduced.
    stale_gross = (tick - entry) * QTY
    stale_valid, stale_why = validate_outcome_math(
        entry_price=entry, exit_price=eff, quantity=QTY,
        gross_profit=stale_gross, fee_cost=1.0, net_profit=stale_gross - 1.0,
        base_token="TEST", quote_token="USDC")
    assert not stale_valid
    assert stale_why == "gross_profit_mismatch"


def test_the_clamp_reaches_the_pl_and_not_just_the_recorded_price():
    """The no-op half of the defect: the book's NUMBER has to change."""
    entry, target, tick = 2.859, 2.859 * 1.05, 6.3723   # the real UNI-USDC row
    eff, gross, valid, why = _book_ghost_exit(entry=entry, target=target, tick=tick)
    assert valid, why

    # The tick was a +122.89% round trip. The limit is +5%.
    assert (tick / entry - 1.0) > 1.2
    # The tolerance scales the TARGET, not the entry: the booked return is
    # 1.05*(1+FEE) - 1, which is 0.05 + 1.05*FEE. Checking units at the
    # boundary -- the first draft of this line said 0.05 + FEE and was wrong.
    assert gross / (entry * QTY) == pytest.approx(0.05 + 1.05 * FEE, abs=1e-9)
    # The booked gross must be a small fraction of what the tick would have paid.
    assert gross < 0.05 * ((tick - entry) * QTY)


@pytest.mark.parametrize("reason", ["take_profit_limit", "target_hit"])
def test_both_limit_reasons_book_a_consistent_outcome(reason):
    entry, target, tick = 10.0, 10.5, 25.0
    eff, gross, valid, why = _book_ghost_exit(
        entry=entry, target=target, tick=tick, reason=reason)
    assert eff == pytest.approx(target * (1.0 + FEE))
    assert valid, why


@pytest.mark.parametrize("tick", [104.0, 105.0, 105.2])
def test_an_exit_that_did_not_overshoot_is_booked_unchanged(tick):
    """No clamp, no change: the fix must be identical for ordinary exits.

    ``exit_price_effective`` IS ``price`` whenever the clamp does not apply, so
    the new line and the old one agree here. Guards against 'fixing' the
    overshoot by quietly repricing every other exit.
    """
    entry, target = 100.0, 105.0
    eff, gross, valid, why = _book_ghost_exit(entry=entry, target=target, tick=tick)
    assert eff == pytest.approx(tick)
    assert gross == pytest.approx((tick - entry) * QTY)
    assert valid, why


def test_a_stop_loss_is_not_clamped_and_still_books_consistently():
    """A stop is a MARKET order; it genuinely fills through its level.

    Retracted advice, kept as a test so nobody re-adds the clamp here: 1135a79
    established that clamping the stop side would invent a price the market
    never offered. The booked gross must follow the tick, and still validate.
    """
    entry, target, tick = 100.0, 105.0, 88.0
    eff, gross, valid, why = _book_ghost_exit(
        entry=entry, target=target, tick=tick, reason="stop_loss")
    assert eff == pytest.approx(tick)
    assert gross == pytest.approx((tick - entry) * QTY)
    assert gross < 0.0
    assert valid, why


def test_a_live_exit_is_never_clamped_and_still_books_consistently():
    """Live books a real receipt; clamping it would invent a number."""
    entry, target, tick = 100.0, 105.0, 220.0
    eff = limit_exit_fill_price(price=tick, target=target, entry=entry,
                                fee_rate=FEE, reason="take_profit_limit",
                                is_live=True)
    assert eff == pytest.approx(tick)
    gross = (eff - entry) * QTY
    valid, why = validate_outcome_math(
        entry_price=entry, exit_price=eff, quantity=QTY, gross_profit=gross,
        fee_cost=1.0, net_profit=gross - 1.0,
        base_token="TEST", quote_token="USDC")
    assert valid, why
