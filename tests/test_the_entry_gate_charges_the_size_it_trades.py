"""The fee the entry gate charges must be the fee that trade actually pays.

``_roundtrip_fee_rate`` has been size-aware since the flat 0.65% constant was
replaced by the receipt-fitted ``$0.004047 + 0.3187% of notional``. But the
binding the entry gate read was computed once, near the top of
``_interpret_predictions``, with ``notional_hint=None`` -- which prices the
round trip at the LIVE CLIP. Real trades are not clip-sized.

MEASURED 2026-09-05 against the 13 live round trips settled on chain. The clip
rate was 0.589%; the rate each trade actually owed ranged 0.454%..1.600%:

    CBETH-USDC   $0.3158 notional   owed 1.600%   charged 0.589%   lost
    CBETH-USDC   $0.4438 notional   owed 1.231%   charged 0.589%
    AERO-USDC    $0.7500 notional   owed 0.858%   charged 0.589%
    CBBTC-USDC   $1.7492 notional   owed 0.550%   charged 0.589%
    CBBTC-USDC   $3.0000 notional   owed 0.454%   charged 0.589%

Eleven of the thirteen were under-charged, and the error is largest exactly
where the account trades most. This is the same wrong-in-both-directions shape
the flat constant had: too cheap for the small trades that were bleeding, too
expensive for the large ones, which were refused for edge they did not owe.

So the property is not "the gate is stricter" or "the gate is looser". It is
that the gate bills each trade its OWN cost -- and that the position it writes
records the cost its entry was judged against, because the exit path reads that
field back as ``predicted_margin`` to decide whether the trade is on plan.
"""

from __future__ import annotations

import pytest

from trading.bot import TradingBot


FIXED_USD = 0.004047
RATE = 0.003187


def _bot(clip_usd: float) -> TradingBot:
    """A bot with no __init__ -- only the fee arithmetic is under test.

    Built through ``__new__`` deliberately: the same construction the live
    refusal tests use, and the reason ``_owned_symbols`` and
    ``_phantom_checked_at`` are lazy properties rather than __init__ fields.
    """
    bot = TradingBot.__new__(TradingBot)
    bot._live_clip_usd = lambda: clip_usd  # type: ignore[method-assign]
    return bot


def _owed(notional: float) -> float:
    """The round-trip rate a trade of this size actually pays."""
    return max((FIXED_USD + RATE * notional) / notional, RATE)


@pytest.mark.parametrize(
    "notional",
    [0.3158, 0.4438, 0.75, 0.8097, 1.7492, 3.0, 19.94],
)
def test_the_rate_follows_the_notional(notional):
    """Every settled size is billed the cost its receipt showed."""
    bot = _bot(1.50)
    assert bot._roundtrip_fee_rate(notional_hint=notional) == pytest.approx(
        _owed(notional), rel=1e-9
    )


def test_a_smaller_trade_pays_a_higher_rate_than_the_clip():
    """The fixed component does not shrink, so the rate rises as size falls.

    This is the direction that was costing money: a $0.32 trade billed at the
    $1.50 clip rate is charged 0.589% against 1.600% owed, so the gate let it
    through on edge that could never cover it.
    """
    bot = _bot(1.50)
    at_clip = bot._roundtrip_fee_rate(notional_hint=None)
    small = bot._roundtrip_fee_rate(notional_hint=0.3158)
    assert small > at_clip
    assert small == pytest.approx(0.016, abs=5e-4)
    assert at_clip == pytest.approx(0.00589, abs=5e-5)


def test_a_larger_trade_pays_a_lower_rate_than_the_clip():
    """The other direction, which was refusing trades that could pay.

    Guards the gate against being turned into a pure tightening: a $3.00 trade
    owes 0.454%, and charging it the 0.589% clip rate demands a third more edge
    than the trade actually costs.
    """
    bot = _bot(1.50)
    at_clip = bot._roundtrip_fee_rate(notional_hint=None)
    large = bot._roundtrip_fee_rate(notional_hint=3.0)
    assert large < at_clip
    assert large == pytest.approx(0.00454, abs=5e-5)


def test_the_rate_never_falls_below_the_pure_rate():
    """No size amortises the fixed part to zero -- no DEX is free."""
    bot = _bot(1.50)
    assert bot._roundtrip_fee_rate(notional_hint=1e9) == pytest.approx(RATE)


def test_an_unknown_size_still_prices_at_the_clip():
    """The ``notional_hint=None`` default is correct where size is not yet known.

    The fix is that the ENTRY gate no longer uses it, not that the fallback is
    wrong: at line 5909 there is genuinely no size to price against yet.
    """
    bot = _bot(1.50)
    assert bot._roundtrip_fee_rate(notional_hint=None) == pytest.approx(_owed(1.50))
    assert bot._roundtrip_fee_rate(notional_hint=0.0) == pytest.approx(_owed(1.50))
    assert bot._roundtrip_fee_rate(notional_hint=-5.0) == pytest.approx(_owed(1.50))


def test_the_gate_reads_a_size_aware_binding_not_the_clip_one():
    """The entry gate's arithmetic must name ``entry_fees``, not ``fees``.

    Pinned against the source because the defect was not a wrong formula -- the
    formula was already right -- but the gate reading the wrong BINDING of it.
    A future edit that reverts these four call sites to ``fees`` restores the
    bug while every numeric test above still passes.
    """
    import inspect

    src = inspect.getsource(TradingBot._interpret_predictions)

    assert "entry_fees = self._roundtrip_fee_rate(notional_hint=trade_notional_usd)" in src
    # The four readers that decide whether an entry happens.
    assert "min_margin_required = max(entry_fees * 1.5, MIN_NET_MARGIN)" in src
    assert "expected_profit_units = max(0.0, margin - entry_fees) * trade_notional_usd" in src
    assert "min_margin_gate = max(min_margin_required, entry_fees)" in src
    assert "net_margin_after_fees = margin - entry_fees" in src
    # And both position records the exit path reads back as predicted_margin.
    # Scoped to that one key: the ``net_margin_after_fees`` field on the
    # decision dict near the top IS still clip-priced, correctly -- it is
    # written before any size exists, and it is telemetry, not a gate.
    assert '"expected_margin_after_fees": margin - fees,' not in src
    assert src.count('"expected_margin_after_fees": margin - entry_fees,') == 2


def test_the_notional_is_computed_before_the_fee_that_prices_it():
    """Ordering is the whole fix: the size must exist before it is billed.

    ``trade_notional_usd`` was assigned two lines AFTER ``min_margin_required``
    consumed the fee. Anything that moves it back below makes ``entry_fees``
    price a notional of zero and silently fall back to the clip.
    """
    import inspect

    lines = inspect.getsource(TradingBot._interpret_predictions).splitlines()

    def _line_of(needle: str) -> int:
        return next(i for i, ln in enumerate(lines) if needle in ln)

    notional_at = _line_of("trade_notional_usd = max(trade_size, 0.0)")
    fee_at = _line_of("entry_fees = self._roundtrip_fee_rate")
    gate_at = _line_of("min_margin_required = max(entry_fees * 1.5")

    assert notional_at < fee_at < gate_at
