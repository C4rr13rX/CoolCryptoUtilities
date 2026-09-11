"""The only cost test on the directive path could not read most horizons.

`trading/bot.py`'s decision chain reaches `should_enter = True` from a
strategy-emitted directive in an `elif` ABOVE the model conjunction, so
direction_prob, net_margin and the move-size conjunct never bind on that path.
`_lattice_refusal` is the only cost test left on it -- and it fails open, by
design, on any horizon label it cannot read.

Measured from `trading_ops` over the 7 days to 2026-09-11, deduped to entries:

    entries on bot.py's chain         128
      via the directive path          126   (all 10 live entries)
      via the model conjunction         2

    of the 126, the lattice exit taken
      unreadable horizon               53   (42%)  <- this file
        'atf'                          35
        '45m'                          13
        '20m'                           5
      reached the window test          73

"45m" and "20m" are not exotic labels. They are the `default_horizon` class
attribute of seven strategies in the population: rsi_reversal,
obv_accumulation, bollinger_squeeze, supertrend_follow and mean_reversion at
45m, stochastic_reversal and volume_spike at 20m. A fixed lookup table that
listed 5m/10m/15m/30m and jumped to 1h could not price a single entry any of
them emitted at its default.

The fix parses the label rather than looking it up. A label that names no
duration at all still fails open -- this layer must never become the reason
nothing trades -- but the exit taken is now recorded, so the residual is a
count instead of a reconstruction.
"""

from __future__ import annotations

import random
import time

import pytest

from trading.bot import TradingBot, horizon_seconds


def _persistent_series(n: int = 300, bar_sec: float = 300.0) -> list:
    """A series with autocorrelated returns, so a horizon is measurable."""
    rng = random.Random(20260911)
    now = time.time()
    out = []
    price, previous = 100.0, 0.0
    for i in range(n):
        step = 0.6 * previous + rng.gauss(0, 0.003)
        price *= 1 + step
        previous = step
        out.append({"ts": now - (n - i) * bar_sec, "price": price})
    return out


class _Directive:
    def __init__(self, horizon: str, expected_return: float = 0.03) -> None:
        self.action = "enter"
        self.expected_return = expected_return
        self.horizon = horizon
        self.size = 1.0
        self.strategy_id = "test"


def _bot(buffer):
    bot = TradingBot.__new__(TradingBot)
    bot._buffer = buffer
    bot._live_clip_usd = lambda: 0.75
    bot._roundtrip_fee_rate = lambda notional_hint=None: 0.0065
    return bot


class TestTheLabelIsParsedNotLookedUp:
    def test_the_default_horizons_of_seven_strategies_are_readable(self):
        """The two labels that disarmed the gate on 18 of 126 entries.

        Under the old fixed table both of these returned 0.0 and
        `_lattice_refusal` returned None before doing any arithmetic.
        """
        assert horizon_seconds("45m") == pytest.approx(2700.0)
        assert horizon_seconds("20m") == pytest.approx(1200.0)

    def test_the_labels_that_already_worked_still_do(self):
        """Parsing must not move a number the table already got right."""
        assert horizon_seconds("15m") == pytest.approx(900.0)
        assert horizon_seconds("1h") == pytest.approx(3600.0)
        assert horizon_seconds("12h") == pytest.approx(43200.0)
        assert horizon_seconds("1d") == pytest.approx(86400.0)
        assert horizon_seconds("1w") == pytest.approx(604800.0)

    def test_a_label_naming_no_duration_stays_zero(self):
        """'atf' is 35 of the 53. It is not a duration and must not be guessed.

        Inventing a horizon for a label that names none would make the chaos
        layer judge a forecast against a window nobody stated -- a fabricated
        number is worse than a counted fail-open.
        """
        assert horizon_seconds("atf") == 0.0
        assert horizon_seconds("") == 0.0
        assert horizon_seconds(None) == 0.0
        assert horizon_seconds("soon") == 0.0
        assert horizon_seconds("-5m") == 0.0
        assert horizon_seconds("0m") == 0.0


class TestTheGateNowPricesThoseEntries:
    def test_a_forty_five_minute_directive_is_judged_against_its_round_trip(self):
        """The behaviour the label fix exists for.

        expected_return 0.0005 against a 0.0065 round trip cannot pay. Under
        the old table this returned None -- no cost test at all -- because the
        horizon lookup missed before the probability layer was reached.
        """
        bot = _bot(_persistent_series(bar_sec=300.0))
        refusal = bot._lattice_refusal(
            "TEST-USDC", _Directive("45m", expected_return=0.0005), {"price": 100.0}
        )
        assert refusal is not None, "a sub-cost 45m forecast was not priced"
        assert bot._lattice_last_exit == "answered_refuse"

    def test_the_gate_still_gets_out_of_the_way_when_it_cannot_answer(self):
        """It only ever ADDS a reason to refuse. 'atf' must still pass through."""
        bot = _bot(_persistent_series(bar_sec=300.0))
        assert bot._lattice_refusal(
            "TEST-USDC", _Directive("atf", expected_return=0.0005), {"price": 100.0}
        ) is None
        assert bot._lattice_last_exit == "unreadable_horizon:atf"


class TestTheExitTakenIsRecorded:
    """A fail-open that is not counted reads exactly like a pass in the log."""

    def test_a_short_window_names_itself_and_its_length(self):
        bot = _bot(_persistent_series(n=10, bar_sec=300.0))
        assert bot._lattice_refusal(
            "TEST-USDC", _Directive("45m"), {"price": 100.0}
        ) is None
        assert bot._lattice_last_exit == "short_window:10"

    def test_a_non_positive_expected_return_names_itself(self):
        bot = _bot(_persistent_series(bar_sec=300.0))
        assert bot._lattice_refusal(
            "TEST-USDC", _Directive("45m", expected_return=0.0), {"price": 100.0}
        ) is None
        assert bot._lattice_last_exit == "expected_return_not_positive"

    def test_an_exit_directive_names_itself(self):
        bot = _bot(_persistent_series(bar_sec=300.0))
        directive = _Directive("45m")
        directive.action = "exit"
        assert bot._lattice_refusal("TEST-USDC", directive, {"price": 100.0}) is None
        assert bot._lattice_last_exit == "not_an_entry"

    def test_a_passing_forecast_is_distinguishable_from_a_fail_open(self):
        """The whole point: 'allowed' and 'never looked' must not be one value."""
        bot = _bot(_persistent_series(bar_sec=300.0))
        bot._lattice_refusal("TEST-USDC", _Directive("15m"), {"price": 100.0})
        assert bot._lattice_last_exit in {"answered_pass", "answered_refuse"}
