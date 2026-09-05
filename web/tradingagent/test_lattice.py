"""Tests for the chaos layer and the model lattice.

Written against series whose character is KNOWN by construction, because a
chaos statistic that cannot separate a trending series from a random walk is
worse than no statistic -- it produces confident numbers about nothing.
"""

from __future__ import annotations

import random

from django.test import SimpleTestCase

from .chaos import chaos_profile, hurst_exponent, lyapunov_horizon_sec
from .lattice import Calibration, evaluate_signal, update_calibration


def _random_walk(n: int = 300, seed: int = 20260905) -> list:
    rng = random.Random(seed)
    out = [100.0]
    for _ in range(n):
        out.append(out[-1] * (1 + rng.gauss(0, 0.004)))
    return out


def _trending(n: int = 300, seed: int = 20260905) -> list:
    """A PERSISTENT series: each return partly repeats the last one.

    Not a drift. A constant drift with independent noise is a random walk
    with drift -- its returns are independent, so H is correctly ~0.5, and an
    earlier version of this test asserted otherwise and was simply wrong.
    Persistence means the RETURNS are autocorrelated, which is what makes a
    move tend to continue.
    """
    rng = random.Random(seed)
    out = [100.0]
    previous = 0.0
    for _ in range(n):
        # 60% of the last return carries forward: that IS persistence.
        step = 0.6 * previous + rng.gauss(0, 0.003)
        out.append(out[-1] * (1 + step))
        previous = step
    return out


def _mean_reverting(n: int = 300, seed: int = 20260905) -> list:
    rng = random.Random(seed)
    out = [100.0]
    for _ in range(n):
        out.append(out[-1] + (100.0 - out[-1]) * 0.15 + rng.gauss(0, 0.3))
    return out


class ChaosMeasurementTests(SimpleTestCase):
    def test_a_short_series_is_unmeasurable_not_random(self):
        """None means "no evidence", never a confident 0.5.

        Reporting a three-point series as a random walk is a claim we cannot
        support, and a caller acting on it would be trading on the absence of
        data.
        """
        self.assertIsNone(hurst_exponent([100.0, 101.0, 102.0]))
        profile = chaos_profile("X", [100.0, 101.0, 102.0], 300.0)
        self.assertEqual(profile["character"], "unmeasurable")

    def test_a_trending_series_reads_as_persistent(self):
        profile = chaos_profile("TREND", _trending(), 300.0)
        self.assertEqual(profile["character"], "persistent")

    def test_a_mean_reverting_series_is_not_called_a_random_walk(self):
        """R/S alone got this wrong, which is why there are two witnesses.

        Measured: the rescaled-range estimator reported H=0.501 for a
        genuinely mean-reverting series -- indistinguishable from a random
        walk. Return autocorrelation measures reversion directly and caught
        it at -0.087.
        """
        profile = chaos_profile("REVERT", _mean_reverting(), 300.0)
        self.assertEqual(profile["character"], "mean-reverting")

    def test_a_random_walk_is_not_called_tradeable(self):
        profile = chaos_profile("WALK", _random_walk(), 300.0)
        self.assertEqual(profile["character"], "random-walk")

    def test_the_horizon_is_reported_in_seconds_not_bars(self):
        horizon = lyapunov_horizon_sec(_trending(), 300.0)
        self.assertIsNotNone(horizon)
        # A 5-minute bar series cannot have a sub-second usable horizon; if
        # it does, the units were lost somewhere.
        self.assertGreater(horizon, 1.0)


class LatticeTests(SimpleTestCase):
    def _signal(self, horizon_sec: float, **kw):
        params = dict(
            symbol="TREND", prices=_trending(), bar_sec=300.0,
            proposed_horizon_sec=horizon_sec, expected_return=0.03,
            round_trip_cost=0.0065, notional_usd=0.75,
        )
        params.update(kw)
        return evaluate_signal(**params)

    def test_a_signal_inside_the_usable_horizon_survives(self):
        result = self._signal(600)
        self.assertTrue(result["passed"], result["reason"])

    def test_the_same_signal_projected_too_far_is_refused_by_chaos(self):
        """The check that would have stopped the @1w variants.

        Same series, same expected return, same cost -- only the horizon
        changes. A forecast past the Lyapunov time is not a bolder prediction,
        it is arithmetic on noise.
        """
        result = self._signal(259200)          # 3 days
        self.assertFalse(result["passed"])
        self.assertEqual(result["stopped_at"], "chaos")

    def test_an_edge_that_cannot_clear_its_cost_is_refused(self):
        result = self._signal(600, expected_return=0.001)
        self.assertFalse(result["passed"])
        self.assertEqual(result["stopped_at"], "probability")

    def test_every_layer_reports_why(self):
        """A lattice that only says no is no better than the flat list."""
        result = self._signal(259200)
        self.assertTrue(result["reason"])
        for layer in result["layers"]:
            self.assertTrue(layer["reason"], f"{layer['layer']} gave no reason")

    def test_an_unmeasurable_series_does_not_pass_by_default(self):
        """Silence is not permission."""
        result = self._signal(600, prices=[100.0, 101.0, 102.0])
        self.assertFalse(result["passed"])
        self.assertEqual(result["stopped_at"], "chaos")


class CalibrationFeedbackTests(SimpleTestCase):
    def test_outcomes_past_the_predicted_horizon_widen_it(self):
        cal = Calibration()
        before = cal.horizon_scale
        update_calibration(cal, predicted_horizon_sec=600,
                           realised_hold_sec=1800)
        self.assertGreater(cal.horizon_scale, before)

    def test_fills_better_than_expected_relax_the_discount(self):
        cal = Calibration()
        before = cal.adverse_scale
        update_calibration(cal, expected_edge=0.01, realised_edge=0.02)
        self.assertLess(cal.adverse_scale, before)

    def test_calibration_cannot_run_away(self):
        """A layer that has doubled its own permissiveness is broken.

        Feedback that compounds without a bound eventually licences anything,
        which is the failure mode of every self-tuning system that has no
        stop.
        """
        cal = Calibration()
        for _ in range(200):
            update_calibration(cal, predicted_horizon_sec=600,
                               realised_hold_sec=100000)
        self.assertLessEqual(cal.horizon_scale, 2.0)
        self.assertGreaterEqual(cal.horizon_scale, 0.5)

    def test_a_model_cannot_calibrate_itself_on_nothing(self):
        cal = Calibration()
        before = cal.as_dict()
        update_calibration(cal)
        self.assertEqual(cal.as_dict(), before)
