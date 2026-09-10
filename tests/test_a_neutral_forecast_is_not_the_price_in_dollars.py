"""The "no model" summary must be dimensionless in every field.

``_neutral_pred_summary`` is what ``trading/bot.py`` hands downstream when TF
is unavailable, and its contract is a non-committal tick: 50/50 direction,
zero expected return. Every field held to that except one -- ``price_mu`` was
``float(current_price or 0.0)``, the price in dollars, in a dict whose other
members are 0.5 and 0.0.

``price_mu`` is read as a RETURN. ``_summarise_predictions`` sets
``delta = price_mu`` outright, the ``net_margin`` head tracks it to a constant
fee (0.006-0.007 in every 2h bucket over 24h, measured 2026-09-10), and
``pipeline.horizon_forecast`` takes it as its first positional argument with
``current_price`` passed separately beside it.

Measured 2026-09-10 over 5634 ``organism_snapshots`` cycles: 14 carried
``abs(price_mu) > 10`` and every one was this summary -- exit_conf 0.5,
direction_prob 0.5, net_margin 0.0 -- topping out at WBTC-USDC 78143.700 and
CBBTC-USDC 77970.870 against a target scale of ~0.01. On a $0.00003 token a
price and a return are the same order of magnitude, which is why it survived.

Same family as the four units bugs in the standing instructions: a fee in the
wrong currency, a t-test over dollars that should have been over returns, gas
priced in the traded pair.
"""

from __future__ import annotations

import pytest

from trading.bot import TradingBot


#: The real WBTC-USDC price from the worst recorded row.
CONTAMINATING_PRICE = 78143.7


def _neutral(price):
    return TradingBot._neutral_pred_summary(  # type: ignore[misc]
        object.__new__(TradingBot), current_price=price
    )


def test_a_neutral_price_mu_is_a_return_not_seventy_eight_thousand_dollars() -> None:
    summary = _neutral(CONTAMINATING_PRICE)

    # Pre-fix this was 78143.7 -- read downstream as a +7,814,370% return.
    assert abs(float(summary["price_mu"])) < 1.0, (
        "price_mu is consumed as a return; a magnitude above 1.0 is not a "
        "return, it is a price"
    )
    assert float(summary["price_mu"]) == pytest.approx(0.0)


def test_the_price_is_not_lost_by_neutralising_the_forecast() -> None:
    """The fix must not delete information, only put it in the right field."""
    summary = _neutral(CONTAMINATING_PRICE)

    assert float(summary["current_price"]) == pytest.approx(CONTAMINATING_PRICE)


def test_every_neutral_field_is_dimensionless_together() -> None:
    """The whole point: one field disagreeing with its siblings is the bug."""
    summary = _neutral(CONTAMINATING_PRICE)

    assert float(summary["exit_conf"]) == pytest.approx(0.5)
    assert float(summary["direction_prob"]) == pytest.approx(0.5)
    assert float(summary["net_margin"]) == pytest.approx(0.0)
    assert float(summary["net_pnl"]) == pytest.approx(0.0)
    assert float(summary["expected_return"]) == pytest.approx(0.0)
    assert float(summary["price_log_var"]) == pytest.approx(0.0)
    assert summary["model_available"] is False
    # A cheap invariant that would have caught this at any price: no forecast
    # field in a NEUTRAL summary may exceed 1.0 in magnitude.
    forecast_fields = (
        "exit_conf", "direction_prob", "net_margin", "net_pnl",
        "expected_return", "price_mu", "price_log_var",
    )
    for field in forecast_fields:
        assert abs(float(summary[field])) <= 1.0, f"{field} is not dimensionless"


def test_a_cheap_token_would_have_hidden_this_and_still_reads_zero() -> None:
    """On a $0.00003 token a price and a return look alike -- assert anyway."""
    summary = _neutral(0.00003413)

    assert float(summary["price_mu"]) == pytest.approx(0.0)
    assert float(summary["current_price"]) == pytest.approx(0.00003413)
