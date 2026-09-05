"""An exit must not strand chain balance that no other position claims.

MEASURED 2026-09-04 on AERO-USDC, wallet 0x291c854811e92906a658Fb94Aa511bF919f968ad.

The five settled AERO transactions on this wallet net to exactly the booked
size. Each exit really did sell its whole position -- the receipts close each
round trip to zero:

    0xfce0036c5c9e2e73a5985a5cfb849ed745e55a7b07c2c27bca00cca85d0888b5  +1.494620938001
    0x121349488665874b2259c283d268cda7220663d11b146d2651983a0566316bb5  -1.494620938001
    0x1ba066c1f1a4d87349077e0f720c6abd4a4b2e70a1aab5d874536b6139ba5c25  +1.473004744878
    0x50f3ee13c2a4ca257752e36e1761d52bf8cc98cb515f9f1dedf1550a2a36e422  -1.473004744878
    0xfc928fab33be2987921545dda457fef84295f4797870fabd2b123916070b2661  +1.498535132128

    net from these five        1.498535132128053115 raw = the booked position
    balanceOf                  3.044816584088252614
    unexplained by any of them 1.546281451960

That 1.546281451960 AERO (~$0.77 at 0.4992 -- five times the entire live P/L
of +0.1423) predates every AERO row in trading_ops, so no unmatched-buy row
explains it. It is invisible to the reconciler, which rebuilds unmatched buys
from the ops log and STOPS AT THE NEWEST SETTLED SELL on the assumption that
the sell closed everything older. That assumption is false precisely because
the exit clamped to ``min(book, chain)``: a clamped sell leaves residue, and
the stop-at-sell rule then makes the residue permanently unreachable.

$0.77 is also far above EXIT_DUST_SWEEP_USD, so the dust sweep never reached
it either.

The rule these tests pin: sell the balance nothing else claims, and never a
token another position -- or the kept half of a partial exit -- is holding.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.cli_utils import to_base_units
from trading.bot import TradingBot


AERO = "0x940181a94A35A4569E4529A3CDfB74e38FD98631"
AERO_DECIMALS = 18

BOOKED_RAW = 1_498_535_132_128_053_116     # the position the book records
CHAIN_RAW = 3_044_816_584_088_252_614      # what balanceOf actually reports
ORPHAN_RAW = CHAIN_RAW - BOOKED_RAW        # 1.546281451960 AERO, unreachable
AERO_PRICE = 0.4991747346                  # last tick 2026-09-04 20:30:21


def _bot(positions=None) -> TradingBot:
    """No I/O: the sizer reads only its arguments, os.environ and the swapper."""
    bot = TradingBot.__new__(TradingBot)
    bot.positions = positions if positions is not None else {}
    return bot


class _Swapper:
    """SwapService.token_balance_raw's measured contract: (raw, decimals)."""

    def __init__(self, reading):
        self.reading = reading

    def token_balance_raw(self, chain, token, owner=None):
        return self.reading


def _aero_position(size_human: float, *, mode: str = "live") -> dict:
    return {
        "mode": mode,
        "size": size_human,
        "base_symbol": "AERO",
        "base_token_address": AERO,
    }


def _size(bot, *, position_size, held_size=None, sweep=True, price=AERO_PRICE):
    return bot._size_live_exit(
        _Swapper((CHAIN_RAW, AERO_DECIMALS)),
        chain="base",
        token=AERO,
        symbol="AERO-USDC",
        position_size=position_size,
        price=price,
        held_size=held_size,
        sweep_unclaimed=sweep,
    )


def test_the_orphan_is_worth_far_more_than_the_dust_floor(monkeypatch):
    """Why the existing sweep could never have recovered it."""
    monkeypatch.setenv("EXIT_DUST_SWEEP_USD", "0.50")
    orphan_usd = (ORPHAN_RAW / 10 ** AERO_DECIMALS) * AERO_PRICE
    assert orphan_usd == pytest.approx(0.7718, abs=1e-3)
    assert orphan_usd > _bot()._exit_dust_sweep_usd()


def test_exit_sells_the_orphan_no_position_claims():
    """The measured case: one AERO position, 1.546 AERO nothing accounts for."""
    bot = _bot({"AERO-USDC": _aero_position(1.498535132128053)})
    sized = _size(bot, position_size=1.498535132128053)

    assert sized["onchain_raw"] == CHAIN_RAW
    # Within a rounding step of the measured orphan: the book carries a float,
    # and `str(1.498535132128053)` round-trips 116 wei short of the integer the
    # chain reported. The orphan is ~1.5e18 raw, so this is 1e-16 of it -- and
    # `exit_raw` below is exact regardless, because it comes from balanceOf.
    assert sized["unclaimed_raw"] == pytest.approx(ORPHAN_RAW, abs=1_000)
    assert sized["exit_raw"] == CHAIN_RAW, "the whole balance, or it strands again"
    assert to_base_units(sized["amount"], sized["decimals"]) == CHAIN_RAW


def test_another_live_position_in_the_same_token_is_never_spent():
    """Two bots, one AERO balance. Closing one must not sell the other's.

    Selling too little strands dust a later sweep can still reach; selling too
    much spends a position that is still open, and that is unrecoverable.
    """
    other = 1.4
    bot = _bot({
        "AERO-USDC": _aero_position(1.498535132128053),
        "AERO-USDT": _aero_position(other),
    })
    sized = _size(bot, position_size=1.498535132128053)

    reserved = to_base_units(str(other), AERO_DECIMALS)
    assert sized["exit_raw"] == CHAIN_RAW - reserved
    assert sized["exit_raw"] < CHAIN_RAW
    assert CHAIN_RAW - sized["exit_raw"] == reserved


def test_a_partial_exit_keeps_the_half_it_meant_to_keep():
    """`exit_target = min(held_size, directive.size)` -- the rest stays held.

    The orphan is still sold; only the part of THIS position the directive
    chose not to sell is reserved.
    """
    held = 1.498535132128053
    bot = _bot({"AERO-USDC": _aero_position(held)})
    sized = _size(bot, position_size=held / 2.0, held_size=held)

    kept = to_base_units(str(held - held / 2.0), AERO_DECIMALS)
    assert CHAIN_RAW - sized["exit_raw"] == kept
    assert sized["unclaimed_raw"] == pytest.approx(ORPHAN_RAW, abs=1_000), (
        "the orphan is sold; the kept half is not"
    )
    # The kept half is $0.374, under the $0.50 dust floor. Nothing downstream
    # may sweep it just because it is small -- it is an open position.
    assert kept > 0 and sized["swept"] is False


def test_a_ghost_position_reserves_nothing():
    """A ghost position holds no tokens, so it cannot claim any."""
    bot = _bot({
        "AERO-USDC": _aero_position(1.498535132128053),
        "AERO-USDT": _aero_position(1.4, mode="ghost"),
    })
    assert _size(bot, position_size=1.498535132128053)["exit_raw"] == CHAIN_RAW


def test_sweep_is_off_by_default_so_other_callers_are_unchanged():
    """The dust sweeper and the post-exit probe must keep the old contract."""
    bot = _bot({"AERO-USDC": _aero_position(1.498535132128053)})
    sized = _size(bot, position_size=1.498535132128053, sweep=False)

    assert sized["exit_raw"] == to_base_units("1.498535132128053", AERO_DECIMALS)
    assert sized["unclaimed_raw"] == 0

    # position_size=0.0 is how the post-exit probe re-reads the balance; it
    # must still report the holding without selling anything.
    probe = _size(bot, position_size=0.0, sweep=False)
    assert probe["exit_raw"] == 0
    assert probe["onchain_raw"] == CHAIN_RAW


def test_never_sells_more_than_the_chain_holds_even_when_sweeping():
    """The overshoot guard survives the new path: asking for more reverts."""
    bot = _bot({"AERO-USDC": _aero_position(99.0)})
    sized = _size(bot, position_size=99.0)
    assert sized["exit_raw"] == CHAIN_RAW
    assert sized["exit_raw"] <= sized["onchain_raw"]


def test_a_book_the_bot_never_built_does_not_crash_the_exit():
    """Not every construction path runs __init__; an exit must still size."""
    bot = TradingBot.__new__(TradingBot)          # no `positions` attribute
    assert _size(bot, position_size=1.498535132128053)["exit_raw"] == CHAIN_RAW
