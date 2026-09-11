"""A 1.25-billion-percent forecast entered a trade, and it asked for the biggest clip.

`trading/strategies/base.py`'s `make_candidate` is the ONE place every strategy's
directive is built. Several strategies price their edge as a ratio over the
recent window -- `obv_accumulation` uses `(recent_high - last_price) /
last_price` -- so one bar from a foreign price regime under the same ticker
does not produce a slightly wrong forecast, it produces an unbounded one.

Measured over the 7 days to 2026-09-11 on the 123 directive-path entries (the
path that placed every live entry in that window, see
data/directive_path_gate_census.md):

    expected_return       entries
      < 5%                     60
      5% .. 20%                54
      20% .. 50%                7
      50% .. 100%               0
      100% .. 1000%             1   VIRTUAL-USDC,  obv_accumulation@1w  1.1095
      >= 1000%                  1   CLANKER-USDC,  obv_accumulation@3d  12551318.65

The CLANKER row entered. Two things make an absurd forecast worse than a wrong
one rather than merely equal to it:

  * it clears the only cost test on that path BY CONSTRUCTION.
    `_lattice_refusal`'s probability layer compares the forecast against the
    round-trip cost, so the larger the garbage the safer it looks.
  * `_size_enter(ctx, expected_return - fee_rate)` scales the clip with the
    forecast, so the contaminated row also asks for the LARGEST position the
    sizer will grant.

The bound sits far above anything this feed has legitimately produced (110.95%)
and far below the contaminated row, and refuses 1 of 123 entries on that window.
"""

from __future__ import annotations

import pytest

from trading.strategies.base import Strategy, StrategyContext


class _Probe(Strategy):
    strategy_id = "probe"
    default_horizon = "45m"

    def evaluate(self, state, ctx):  # pragma: no cover - not exercised here
        return None


class _State:
    symbol = "CLANKER-USDC"
    base_token = "CLANKER"
    quote_token = "USDC"


def _ctx(last_price: float = 1.023e-06) -> StrategyContext:
    return StrategyContext(
        chain="base",
        last_price=last_price,
        last_volume=1000.0,
        fee_rate=0.0065,
        available_quote=100.0,
        available_base=0.0,
    )


def _enter(expected_return: float, *, last_price: float = 1.023e-06):
    return _Probe().make_candidate(
        _State(),
        _ctx(last_price),
        action="enter",
        expected_return=expected_return,
        target_price=last_price * (1.0 + expected_return),
        confidence=0.7,
        reason="probe",
    )


class TestTheContaminatedForecastIsRefused:
    def test_the_clanker_row_that_entered_cannot_enter(self):
        """expected_return = 12551318.65, the row measured in the 7-day census."""
        assert _enter(12551318.648094) is None

    def test_a_forecast_just_over_the_bound_is_refused(self):
        assert _enter(2.5) is None


class TestEveryPlausibleForecastStILLTrades:
    """A guard that refuses everything is the same as being switched off."""

    @pytest.mark.parametrize(
        "expected_return",
        [0.0100, 0.0500, 0.1000, 0.2000, 0.4900, 1.1095, 2.0000],
    )
    def test_the_whole_measured_population_survives(self, expected_return):
        """1.1095 is the largest plausible row in the census and must pass."""
        candidate = _enter(expected_return)
        assert candidate is not None, f"{expected_return} was refused"
        assert candidate["directive"].expected_return == pytest.approx(expected_return)

    def test_the_clip_still_scales_with_a_plausible_forecast(self):
        """The sizer is untouched below the bound."""
        small = _enter(0.02)
        large = _enter(0.40)
        assert small is not None and large is not None
        assert large["directive"].size > small["directive"].size


class TestTheBoundIsTunableAndBounded:
    def test_lowering_the_bound_refuses_more(self, monkeypatch):
        monkeypatch.setenv("STRATEGY_MAX_EXPECTED_RETURN", "0.10")
        assert _enter(0.20) is None
        assert _enter(0.05) is not None

    def test_a_nonsense_setting_cannot_disable_the_guard(self, monkeypatch):
        """env_float clamps to [0.05, 100.0]; 1e9 must not readmit the row."""
        monkeypatch.setenv("STRATEGY_MAX_EXPECTED_RETURN", "1e9")
        assert _enter(12551318.648094) is None
