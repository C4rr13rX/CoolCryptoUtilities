"""A usable forecast horizon longer than the observed window is not permission.

``lattice.evaluate_signal``'s first layer refuses an entry when
``proposed_horizon_sec > usable_horizon_sec``. ``lyapunov_horizon_sec``
produces that bound as ``1 / exponent``, where the exponent is a mean of log
divergence ratios. On a near-flat series -- a stablecoin pair, a frozen feed --
those ratios cancel to a value indistinguishable from zero and the reciprocal
is astronomical. Measured on the live market stream 2026-09-07:

    EURC-USDC   129 ticks over a 17980 s window  ->  usable horizon 5.963e16 s

That is 3.3e12 times the span it was estimated from, about 1.9 billion years,
and it does not merely report a wrong number: it SWITCHES THE GATE OFF. Any
proposed horizon compares as smaller, so the layer whose whole job is refusing
forecasts the data cannot support waves every one of them through -- on
precisely the flattest symbols, where a forecast is least supportable.

Same lesson as the swap guard's frozen-feed clause: flat is unmeasurable, not
calm. And the same losing shape ``services.profit_logic_audit`` exists to
catch: an unmeasurable quantity defaulting to the permissive value.

These tests fail against the unbounded version.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = str(ROOT / "web")
if WEB not in sys.path:
    sys.path.insert(0, WEB)

from tradingagent.chaos import chaos_profile, lyapunov_horizon_sec  # noqa: E402


#: The real series that produced the bug, captured from the live market
#: stream on 2026-09-07: 129 EURC-USDC ticks holding THREE distinct prices
#: (1.16 .. 1.1600000000000004) with 9 nonzero returns in 128. A synthetic
#: gaussian "flat" series does NOT reproduce it -- the first version of this
#: test used one and passed against the unbounded code, proving nothing. The
#: degeneracy needs the real shape: a near-constant series whose few moves are
#: float-noise-sized repeats.
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "eurc_usdc_flat_series.json"


def _eurc_usdc():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["prices"], float(data["bar_sec"])


def _almost_flat(count: int = 400, seed: int = 5, scale: float = 1e-9):
    """A stablecoin-shaped series: real ticks, moves at the noise floor."""
    rng = random.Random(seed)
    price, out = 1.0, []
    for _ in range(count):
        price = max(1e-9, price * (1.0 + rng.gauss(0.0, scale)))
        out.append(price)
    return out


def _volatile(count: int = 400, seed: int = 5):
    rng = random.Random(seed)
    price, out = 100.0, []
    for _ in range(count):
        price = max(1e-6, price * (1.0 + rng.gauss(0.0, 0.02)))
        out.append(price)
    return out


def test_the_real_eurc_usdc_series_does_not_earn_an_unbounded_horizon():
    """The exact series that produced the bug.

    Against the unbounded code this returns 5.9634e+16 s from a 194 s window
    -- 3.082e+14 times its own span. Against the bounded code it returns None,
    which the lattice reads as "unmeasurable is not permission".
    """
    prices, bar_sec = _eurc_usdc()
    horizon = lyapunov_horizon_sec(prices, bar_sec)
    window = len(prices) * bar_sec
    assert horizon is None or horizon <= window, (
        f"a {window:.0f}s window produced a {horizon:.4e}s usable horizon -- "
        f"{horizon / window:.3e}x the data it was estimated from")


def test_the_gate_refuses_the_real_flat_symbol():
    """The consequence at the layer that spends money, on the real series.

    The chaos layer refuses when ``proposed_horizon_sec > usable``. With an
    unbounded estimate no proposal is ever larger, so the guard is off.
    """
    prices, bar_sec = _eurc_usdc()
    usable = chaos_profile("EURC-USDC", prices, bar_sec).get("usable_horizon_sec")
    a_five_hour_forecast = 300 * 60.0
    assert usable is None or usable < a_five_hour_forecast, (
        f"EURC-USDC reported a {usable:.4e}s usable horizon, so a "
        f"{a_five_hour_forecast:.0f}s forecast passes the chaos layer")


def test_a_flat_series_does_not_earn_an_unbounded_horizon():
    """The same property on synthetic near-flat series, across cadences."""
    for scale in (1e-9, 1e-7, 1e-5):
        prices = _almost_flat(scale=scale)
        bar_sec = 2.0
        horizon = lyapunov_horizon_sec(prices, bar_sec)
        window = len(prices) * bar_sec
        assert horizon is None or horizon <= window, (
            f"scale {scale}: a {window:.0f}s window produced a {horizon:.3e}s "
            f"usable horizon")


def test_no_series_claims_a_horizon_longer_than_its_own_window():
    """Across shapes and cadences, the bound holds. A decay timescale cannot
    be measured beyond the span observed, whatever the series looks like."""
    for seed in range(12):
        for prices, bar_sec in (
            (_almost_flat(seed=seed), 2.0),
            (_almost_flat(seed=seed, scale=1e-6), 60.0),
            (_volatile(seed=seed), 3600.0),
        ):
            horizon = lyapunov_horizon_sec(prices, bar_sec)
            if horizon is None:
                continue
            window = len(prices) * bar_sec
            assert horizon <= window, (
                f"seed {seed}, bar {bar_sec}s: horizon {horizon:.3e}s exceeds "
                f"the {window:.0f}s window")
            assert math.isfinite(horizon) and horizon > 0


def test_a_measurable_series_still_gets_a_horizon():
    """The bound must not be a blanket refusal.

    A guard that refuses everything is the same as being switched off. A
    genuinely chaotic series has to keep producing a finite, usable number
    inside its own window.
    """
    measured = 0
    for seed in range(20):
        horizon = lyapunov_horizon_sec(_volatile(seed=seed), 60.0)
        if horizon is not None:
            measured += 1
            assert 0 < horizon <= len(_volatile(seed=seed)) * 60.0
    assert measured >= 10, (
        f"only {measured}/20 volatile series kept a measurable horizon -- the "
        f"bound has become a blanket refusal")
