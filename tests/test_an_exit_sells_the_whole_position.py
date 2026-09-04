"""An exit must sell the WHOLE position, sized from the chain.

Measured on base 2026-09-03, wallet 0x291c854811e92906a658Fb94Aa511bF919f968ad.
Across 15 settled live swaps the wallet sent 11 buys of exactly 0.750000 USDC
(8.250000 out) and got 4 sells back (2.268969 in), net -5.981031. Two separate
defects produced that, and both are pinned here.

1. THE EXIT WAS SIZED FROM A CACHE THAT DOES NOT HOLD EVERY TOKEN.
   ``available_base`` came from ``portfolio.get_quantity()``, which reads the
   ``balances`` table. Read side by side against the chain that day:

       SYMBOL    balances.quantity       balanceOf()
       USDC              15.196303         15.196303
       AERO      1.5462814519601995   1.5462814519601995
       CBETH      3.73253011411e-07    3.73253011411e-07
       CBBTC              (no row)         3.709e-05
       BSTONK             (no row)   360.264243225393
       BASECAT            (no row)   38.09724680310889

   ``get_quantity`` answers 0.0 for a missing row, so ``exit_size = min(held,
   0.0)`` was 0 and every exit of those three was refused
   ``insufficient_base``. Seven positions and 5.25 USDC could not be closed --
   while entries, sized in USDC (which IS cached), kept firing. That is the
   whole of "buys outrun sells three to one".

2. THE AMOUNT WAS TRUNCATED TO SIX DECIMAL PLACES.
   The call passed ``f"{exit_size:.6f}"``. Exit
   0x9ffdd1cfe3f17fbeddb2b3091b49f8f6524b5a0538ae9a2f3aadb56610264e6c held
   0.000111373 cbETH at block 50832309 and sold 0.000111000 -- the format
   string kept 0.000000373 back, and that residue is in the wallet today.
   cbETH has 18 decimals; flooring at six is a leak, not a rounding choice.
"""

from __future__ import annotations

import os

import pytest

from services.cli_utils import from_base_units, to_base_units
from trading.bot import TradingBot


CBETH_DECIMALS = 18
CBBTC_DECIMALS = 8


def _bot() -> TradingBot:
    """A bot instance with no I/O: the methods under test read only their
    arguments, os.environ and the swapper handed to them."""
    return TradingBot.__new__(TradingBot)


class _Swapper:
    """Stands in for SwapService.token_balance_raw, whose real contract was
    measured against base: a 2-tuple of ints ``(raw, decimals)``, or None."""

    def __init__(self, reading):
        self.reading = reading
        self.calls = []

    def token_balance_raw(self, chain, token, owner=None):
        self.calls.append((chain, token, owner))
        if isinstance(self.reading, Exception):
            raise self.reading
        return self.reading


# --------------------------------------------------------------------------
# 2. The truncation, at the boundary where it happened
# --------------------------------------------------------------------------


def test_six_decimal_format_is_what_left_cbeth_behind():
    """A regression fence on the exact numbers from exit
    0x9ffdd1cfe3f17fbeddb2b3091b49f8f6524b5a0538ae9a2f3aadb56610264e6c."""
    held_raw = 111_373_000_000_000  # 0.000111373 cbETH at block 50832309
    held_human = float(from_base_units(held_raw, CBETH_DECIMALS))

    truncated = f"{held_human:.6f}"
    assert truncated == "0.000111"
    left_behind = held_raw - to_base_units(truncated, CBETH_DECIMALS)
    assert left_behind == 373_000_000_000  # 0.000000373 cbETH, stuck forever
    assert from_base_units(left_behind, CBETH_DECIMALS) == "0.000000373000000000"

    # The replacement loses nothing at all.
    assert to_base_units(from_base_units(held_raw, CBETH_DECIMALS), CBETH_DECIMALS) == held_raw


@pytest.mark.parametrize(
    "raw,decimals",
    [
        (373_253_011_411, 18),            # the cbETH dust in the wallet now
        (1_546_281_451_960_199_499, 18),  # AERO: float(str()) would round UP and revert
        (3_709, 8),                       # cbBTC
        (15_196_303, 6),                  # USDC
        (360_264_243_225_392_976_659, 18),  # BSTONK
        (1, 18),                          # one wei
        (0, 18),
    ],
)
def test_amount_string_round_trips_exactly(raw, decimals):
    """Whatever swap() parses must be the integer the chain reported.

    Never exponent notation either -- ``str(Decimal('3.7e-7'))`` is "3.7E-7"
    and to_base_units would have to parse that back.
    """
    text = from_base_units(raw, decimals)
    assert isinstance(text, str)
    assert "E" not in text and "e" not in text
    assert to_base_units(text, decimals) == raw


# --------------------------------------------------------------------------
# 1. The sizing, which is the part that stranded real money
# --------------------------------------------------------------------------


def test_exit_sells_a_token_the_balances_cache_has_no_row_for():
    """cbBTC: 4 entries, 3.00 USDC, no cache row, 3709 raw held on chain.

    The old path read 0.0 from the cache and refused. The chain is the
    authority, so the exit is sized and the position can close.
    """
    bot = _bot()
    swapper = _Swapper((3709, CBBTC_DECIMALS))
    sized = bot._size_live_exit(
        swapper,
        chain="base",
        token="0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",
        symbol="CBBTC-USDC",
        position_size=3.709e-05,
        price=110_000.0,
    )
    assert sized is not None
    assert sized["onchain_raw"] == 3709
    assert sized["exit_raw"] == 3709, "the whole balance, or it strands again"
    assert sized["amount"] == "0.00003709"
    assert to_base_units(sized["amount"], sized["decimals"]) == 3709


def test_exit_never_sells_more_than_the_chain_holds():
    """A position record claiming more than the wallet has must not overshoot.

    Asking for more than ``balanceOf`` reverts, which costs gas and leaves the
    position open -- the same outcome as refusing, but paid for.
    """
    bot = _bot()
    swapper = _Swapper((100_000_000_000_000, CBETH_DECIMALS))  # 0.0001 held
    sized = bot._size_live_exit(
        swapper,
        chain="base",
        token="0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22",
        symbol="CBETH-USDC",
        position_size=0.000264,  # the book thinks it holds more
        price=2840.0,
    )
    assert sized["exit_raw"] == 100_000_000_000_000
    assert sized["exit_raw"] <= sized["onchain_raw"]


def test_a_residual_too_small_to_trade_is_swept_not_stranded(monkeypatch):
    """Closing one of two positions must not leave an unsellable remnant.

    Held 0.000375373 cbETH against a 0.000264 position at $2840: the 0.000111373
    left over is worth $0.32, under the $0.50 floor, so it goes with this exit.
    That remnant is exactly what exit
    0x927834717d12395c1eb3d9148609a2b8142a59403205d1caa4ffd04e68e0e005
    left behind: it held 0.000273750 and sold 0.000162000.
    """
    monkeypatch.setenv("EXIT_DUST_SWEEP_USD", "0.50")
    bot = _bot()
    swapper = _Swapper((375_373_000_000_000, CBETH_DECIMALS))
    sized = bot._size_live_exit(
        swapper,
        chain="base",
        token="0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22",
        symbol="CBETH-USDC",
        position_size=0.000264,
        price=2840.0,
    )
    assert sized["swept"] is True
    assert sized["exit_raw"] == 375_373_000_000_000
    assert sized["onchain_raw"] - sized["exit_raw"] == 0


def test_a_residual_worth_trading_is_left_for_its_own_position(monkeypatch):
    """The sweep must not steal a second position that can still be sold.

    Same token, but the remnant is worth $1.42 -- above the floor and above the
    0.75 clip this book trades in, so it stays for its own exit.
    """
    monkeypatch.setenv("EXIT_DUST_SWEEP_USD", "0.50")
    bot = _bot()
    swapper = _Swapper((764_000_000_000_000, CBETH_DECIMALS))  # 0.000764 held
    sized = bot._size_live_exit(
        swapper,
        chain="base",
        token="0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22",
        symbol="CBETH-USDC",
        position_size=0.000264,
        price=2840.0,
    )
    assert sized["swept"] is False
    assert sized["exit_raw"] == to_base_units("0.000264", CBETH_DECIMALS)
    assert sized["onchain_raw"] - sized["exit_raw"] == 500_000_000_000_000


def test_an_unreadable_balance_is_none_and_never_zero():
    """"The RPC is down" and "the position is gone" are opposite facts.

    Returning 0.0 for an unreadable balance is how the truncation happened in
    the first place. None makes the caller refuse and retry next sample.
    """
    bot = _bot()
    assert (
        bot._size_live_exit(
            _Swapper(None), chain="base", token="0xdead", symbol="X-USDC",
            position_size=1.0, price=1.0,
        )
        is None
    )
    assert (
        bot._size_live_exit(
            _Swapper(RuntimeError("all endpoints refused")), chain="base",
            token="0xdead", symbol="X-USDC", position_size=1.0, price=1.0,
        )
        is None
    ), "a raising balance read must not crash the exit either"


def test_a_measured_zero_is_distinct_from_unreadable():
    """A wallet that genuinely holds nothing reports 0, not None: the position
    is already closed and the caller should say so rather than retry forever."""
    bot = _bot()
    sized = bot._size_live_exit(
        _Swapper((0, CBETH_DECIMALS)), chain="base", token="0xdead",
        symbol="CBETH-USDC", position_size=0.000264, price=2840.0,
    )
    assert sized is not None
    assert sized["exit_raw"] == 0
    assert sized["onchain_human"] == 0.0


def test_a_zero_or_unusable_price_does_not_sweep():
    """The sweep decides in USD. With no usable price there is no judgement to
    make, so the position size stands and nothing extra is sold."""
    bot = _bot()
    for price in (0.0, -1.0, float("nan"), float("inf")):
        sized = bot._size_live_exit(
            _Swapper((375_373_000_000_000, CBETH_DECIMALS)), chain="base",
            token="0xdead", symbol="CBETH-USDC", position_size=0.000264,
            price=price,
        )
        assert sized["swept"] is False, price
        assert sized["exit_raw"] == to_base_units("0.000264", CBETH_DECIMALS)


def test_dust_floor_falls_back_through_env(monkeypatch):
    bot = _bot()
    monkeypatch.delenv("EXIT_DUST_SWEEP_USD", raising=False)
    monkeypatch.delenv("WALLET_DUST_USD", raising=False)
    assert bot._exit_dust_sweep_usd() == 0.50
    monkeypatch.setenv("WALLET_DUST_USD", "0.25")
    assert bot._exit_dust_sweep_usd() == 0.25
    monkeypatch.setenv("EXIT_DUST_SWEEP_USD", "1.5")
    assert bot._exit_dust_sweep_usd() == 1.5
    monkeypatch.setenv("EXIT_DUST_SWEEP_USD", "not-a-number")
    assert bot._exit_dust_sweep_usd() == 0.50, "a bad value must not sweep everything"
