"""A limit exit may not book the overshoot that tripped it.

``take_profit_limit`` (trading/triggers.py:188) and ``target_hit``
(trading/bot.py) both fire on ``price >= target_price``, and the ghost exit
booked ``exit_price_effective = price`` -- the tick that CROSSED the target,
not the target. That credits the position with the entire distance between two
samples. A real limit order fills at the limit.

Measured 2026-09-10 over the 124 closed ghost round trips of the last 7 days,
7 of the 14 take-profit exits booked above 1.10x their target:

    BSTONK   +17.28%   BSTONK   +17.83%   BSTONK   +23.68%   BSTONK  +25.35%
    BASECAT  +17.31%   BASELINE +57.94%   UNI-USDC +122.89%  (2.859 -> 6.3723)

Those SEVEN ROWS are +2.2905 of the book's +2.3461 of gross. The other 117
trips carry +0.0556, which is zero. The live-tradeable book without them is
106 trips at -0.3225 of gross -- NEGATIVE. Graduation reads this book, so
every symbol-admission and cost-model conclusion drawn from it was drawn from
seven sampling gaps.

THE ASYMMETRY IS THE TELL. The LIVE exit path already refuses an implausible
fill (``_fill_price_disagrees_with_feed``) and books the feed price instead.
The ghost path had no such check, so the ghost book could record fills the
live lane would reject on sight -- the worst direction for the difference to
run in, because the ghost book is the evidence that earns a live licence.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.bot import LIMIT_EXIT_REASONS, limit_exit_fill_price  # noqa: E402

ENTRY = 2.859
TARGET = ENTRY * 1.05
FEE = 0.003187      # one leg, the receipts-derived variable rate


def test_a_ghost_take_profit_cannot_fill_past_its_own_limit():
    """The acceptance case: target = entry*1.05, tick at entry*2.2."""
    tick = ENTRY * 2.2
    booked = limit_exit_fill_price(price=tick, target=TARGET, entry=ENTRY,
                                   fee_rate=FEE, reason="take_profit_limit",
                                   is_live=False)
    # The old behaviour booked the tick, which is a +120% round trip on a
    # position that asked for +5%.
    assert booked < tick
    assert booked <= TARGET * (1.0 + FEE)
    # ...and it is not clamped below the limit either: a take-profit that fills
    # AT its target is the ordinary, correct outcome and must keep its gain.
    assert booked >= TARGET
    realised = booked / ENTRY - 1.0
    assert 0.05 <= realised <= 0.06, (
        "a +5%% target must book about +5%%, not %.2f%%" % (realised * 100.0)
    )


def test_the_real_uni_row_is_the_case_this_prevents():
    """UNI-USDC entry 2.859 -> exit 6.3723, +122.89%, 103% of the book's edge."""
    booked = limit_exit_fill_price(price=6.3723, target=TARGET, entry=ENTRY,
                                   fee_rate=FEE, reason="target_hit",
                                   is_live=False)
    assert booked / ENTRY - 1.0 < 0.06


def test_a_fill_inside_the_fee_is_ordinary_slippage_and_is_kept():
    """The tolerance has to admit a real fill, or it is just a haircut."""
    tick = TARGET * (1.0 + FEE * 0.5)
    assert limit_exit_fill_price(price=tick, target=TARGET, entry=ENTRY,
                                 fee_rate=FEE, reason="take_profit_limit",
                                 is_live=False) == tick


def test_a_live_exit_books_what_the_chain_actually_paid():
    """Clamping a receipt would invent a number the wallet did not receive."""
    tick = ENTRY * 2.2
    assert limit_exit_fill_price(price=tick, target=TARGET, entry=ENTRY,
                                 fee_rate=FEE, reason="take_profit_limit",
                                 is_live=True) == tick


def test_only_limit_exits_are_clamped():
    """A stop, a timed exit or a model exit is not a limit against a target.

    Those close at the price the market is at; clamping them to a take-profit
    target that was never reached would BOOK A PROFIT THAT DID NOT HAPPEN --
    the opposite error, and a worse one.
    """
    tick = ENTRY * 2.2
    for reason in ("stop_loss:-0.0200", "timed-exit", "model_exit", "forced_by_age"):
        assert reason not in LIMIT_EXIT_REASONS
        assert limit_exit_fill_price(price=tick, target=TARGET, entry=ENTRY,
                                     fee_rate=FEE, reason=reason,
                                     is_live=False) == tick


def test_a_target_at_or_below_the_entry_is_left_alone():
    """Both triggers compare upward, so this is not the shape -- do not guess."""
    tick = ENTRY * 2.2
    for target in (0.0, -1.0, ENTRY, ENTRY * 0.9):
        assert limit_exit_fill_price(price=tick, target=target, entry=ENTRY,
                                     fee_rate=FEE, reason="take_profit_limit",
                                     is_live=False) == tick


def test_unreadable_inputs_book_the_tick_rather_than_raising():
    """The tokens are already sold; refusing to book leaves a phantom position."""
    assert limit_exit_fill_price(price=1.0, target=None, entry=ENTRY,
                                 fee_rate=FEE, reason="take_profit_limit",
                                 is_live=False) == 1.0
    assert limit_exit_fill_price(price=1.0, target=TARGET, entry=ENTRY,
                                 fee_rate="not a number",
                                 reason="take_profit_limit",
                                 is_live=False) == 1.0
