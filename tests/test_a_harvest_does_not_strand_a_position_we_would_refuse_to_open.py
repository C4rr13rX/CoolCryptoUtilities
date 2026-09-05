"""A partial exit must not leave behind a position the entry gate would refuse.

The entry path already refuses to CREATE a live position below
``MIN_DIRECTIVE_NOTIONAL_USD`` -- it raises the size to that floor instead,
because "a trade too small to clear its own costs is not worth placing".
Nothing said the same about what an EXIT leaves behind, so a harvest directive
could manufacture exactly the position the entry gate exists to prevent, and
then hold the symbol slot with it.

MEASURED on the live AERO-USDC book 2026-09-05, twice from the same rule:

    entry 11:46:40  2.8790090295427495 @ 0.521012606979643
      13:18:28  rsi_reversal harvest sold 1.5540563151090325 (54%),
                stranding 1.3249527144337170
      13:26:59  only max_hold_force finished it, 8 minutes later

    entry 13:28:11  2.848195683623783 @ 0.5266492076455708
      14:18:16  rsi_reversal harvest sold 1.6332641838399833 (57%),
                stranding 1.2149314997837997 = $0.6476
      no second exit: still held 66 minutes later, with the wallet's
      balanceOf returning exactly 1.2149314997837999 AERO

Splitting a minimum-size clip raises the breakeven hurdle on what remains,
because the fixed leg of the cost does not shrink with the position. At the
receipt-fitted round trip of ``$0.004047 + 0.3187% of notional``:

    whole      $1.5000 -> costs $0.008828 = 0.5885% of itself
    remainder  $0.6476 -> costs $0.006110 = 0.9436% of itself

The leftover must therefore move 1.6x as far as the position it was cut from
just to break even, and it spends far more of its life on the wrong side of
the live cost gate. Measured at 14:40 that particular remainder was NOT
gate-refused -- the price had risen enough to clear it -- so the stall is the
hurdle plus the missing second exit, not the gate alone. The outcome is not in
doubt either way: 66 minutes held, no second exit, and
``entry-refused-duplicate`` on the most-traded symbol in the book.

The rule is deliberately one-directional: it only ever rounds a sale UP to the
whole position, and only for live positions. A harvest that leaves a viable
remainder is untouched.
"""
from __future__ import annotations

import pytest

from trading.triggers import exit_target_size


# The live position, the harvest that cut it, and the dust that was left.
AERO_HELD = 2.848195683623783
AERO_HARVEST = 1.6332641838399833
AERO_STRANDED = AERO_HELD - AERO_HARVEST          # 1.2149314997837997

# TWO PRICES, AND THEY ARE NOT INTERCHANGEABLE. The entry price is what the
# clip COST ($1.50 exactly); the market price is what the remainder is WORTH
# now, and worth-now is what decides whether it can pay to close. Valuing the
# remainder at entry gives $0.6398 instead of $0.6476 -- both under the floor
# here, so the verdict is unchanged, but the hurdle is a ratio and the wrong
# leg silently moves it.
AERO_ENTRY_PRICE = 0.5266492076455708
AERO_MARKET_PRICE = 0.5330143494511826

# .env: MIN_DIRECTIVE_NOTIONAL_USD=0.75
FLOOR_USD = 0.75

# The receipt-fitted round trip, from tests/test_the_entry_gate_charges_the_size_it_trades.py
FIXED_USD = 0.004047
RATE = 0.003187


def _cost_to_close(notional_usd: float) -> float:
    return FIXED_USD + RATE * notional_usd


def test_the_stranded_leg_really_was_below_the_floor():
    """The premise, in dollars, before any assertion about behaviour."""
    stranded_usd = AERO_STRANDED * AERO_MARKET_PRICE
    assert stranded_usd == pytest.approx(0.647576, abs=1e-5)
    assert stranded_usd < FLOOR_USD, (
        "if the remainder cleared the floor there would be nothing to fix"
    )
    # Under the floor at either price, so the verdict does not rest on which.
    assert AERO_STRANDED * AERO_ENTRY_PRICE == pytest.approx(0.639843, abs=1e-5)
    assert AERO_STRANDED * AERO_ENTRY_PRICE < FLOOR_USD


def test_the_clip_was_exactly_the_one_the_ramp_approved():
    """Anchors the whole case to the $1.50 named in .env, not a re-derivation."""
    assert AERO_HELD * AERO_ENTRY_PRICE == pytest.approx(1.50, abs=1e-9)


def test_the_stranded_leg_costs_more_to_close_than_the_whole_position_did():
    """Why a dust remainder stalls: splitting raises the breakeven hurdle."""
    whole_usd = AERO_HELD * AERO_ENTRY_PRICE
    stranded_usd = AERO_STRANDED * AERO_MARKET_PRICE

    whole_hurdle = _cost_to_close(whole_usd) / whole_usd
    stranded_hurdle = _cost_to_close(stranded_usd) / stranded_usd

    # 0.5885% is the entry_fee_rate recorded verbatim on the live 13:28 and
    # 13:48 entries, and the number .env quotes for a $1.50 clip.
    assert whole_hurdle == pytest.approx(0.005885, abs=1e-5)
    assert stranded_hurdle == pytest.approx(0.009436, abs=1e-5)
    assert stranded_hurdle > whole_hurdle, (
        "the remainder must move further than the position it was cut from "
        "just to break even -- that is the stall, stated as a number"
    )


def test_a_harvest_that_would_strand_dust_sells_everything():
    """The live case: 57% harvest on a $1.50 clip becomes a full close."""
    got = exit_target_size(
        "[5d] rsi_reversal: RSI 85 overbought, harvesting 7.66%",
        held_size=AERO_HELD,
        directive_size=AERO_HARVEST,
        price=AERO_MARKET_PRICE,
        live=True,
        dust_floor_usd=FLOOR_USD,
    )
    assert got == AERO_HELD, (
        f"selling {AERO_HARVEST} of {AERO_HELD} strands "
        f"${AERO_STRANDED * AERO_MARKET_PRICE:.4f}, under the ${FLOOR_USD} the entry "
        f"gate enforces -- close it instead of holding the slot with dust"
    )


def test_a_harvest_leaving_a_viable_remainder_is_untouched():
    """The rule must not turn every partial harvest into a full liquidation."""
    held = 20.0                       # $10.53 at the AERO price
    harvest = 5.0                     # leaves 15.0 = $7.90, well over the floor
    got = exit_target_size(
        "[5d] rsi_reversal: RSI 85 overbought, harvesting 7.66%",
        held_size=held,
        directive_size=harvest,
        price=AERO_MARKET_PRICE,
        live=True,
        dust_floor_usd=FLOOR_USD,
    )
    assert got == harvest, "a remainder that can trade on its own is left alone"


def test_the_rule_never_makes_an_exit_smaller():
    """One-directional by construction: it rounds up to the whole position."""
    for harvest in (0.01, 0.5, 1.0, AERO_HARVEST, AERO_HELD * 0.999):
        got = exit_target_size(
            "confidence_drop",
            held_size=AERO_HELD,
            directive_size=harvest,
            price=AERO_MARKET_PRICE,
            live=True,
            dust_floor_usd=FLOOR_USD,
        )
        assert got in (harvest, AERO_HELD)
        assert got >= harvest, "the sale may grow to a full close, never shrink"


def test_the_ghost_lane_is_untouched():
    """Ghost spends a simulated purse with its own floor."""
    got = exit_target_size(
        "[5d] rsi_reversal: RSI 85 overbought, harvesting 7.66%",
        held_size=AERO_HELD,
        directive_size=AERO_HARVEST,
        price=AERO_MARKET_PRICE,
        live=False,
        dust_floor_usd=FLOOR_USD,
    )
    assert got == AERO_HARVEST


def test_the_rule_is_off_until_its_inputs_are_supplied():
    """A caller that passes neither price nor floor gets the old behaviour."""
    assert exit_target_size(
        "confidence_drop", held_size=AERO_HELD, directive_size=AERO_HARVEST
    ) == AERO_HARVEST


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_an_unusable_price_cannot_force_a_full_close(price):
    """Units at the boundary: the rule needs a price to value the remainder.

    Without one it must decline to act rather than guess -- forcing a full
    liquidation off a zero or negative print is how a bad tick sells a book.
    """
    assert exit_target_size(
        "confidence_drop",
        held_size=AERO_HELD,
        directive_size=AERO_HARVEST,
        price=price,
        live=True,
        dust_floor_usd=FLOOR_USD,
    ) == AERO_HARVEST


def test_an_empty_position_sells_nothing():
    assert exit_target_size(
        "stop_loss:-0.01", held_size=0.0, directive_size=1.0, price=AERO_MARKET_PRICE,
        live=True, dust_floor_usd=FLOOR_USD,
    ) == 0.0


def test_a_closing_reason_still_outranks_the_directive_and_the_floor():
    """Rule 1 is unchanged and reaches the same answer without needing rule 2."""
    assert exit_target_size(
        "max_hold_force:0.0067",
        held_size=AERO_HELD,
        directive_size=AERO_HARVEST,
        price=AERO_MARKET_PRICE,
        live=True,
        dust_floor_usd=FLOOR_USD,
    ) == AERO_HELD
