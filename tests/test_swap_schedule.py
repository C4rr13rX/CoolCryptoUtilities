"""Tests for the forward swap schedule.

Written around the constraints that actually protect money: the same dollar
cannot fund two overlapping legs, a leg must clear its own round-trip cost,
and a forecast the market has already refuted must be dropped rather than
waited out.
"""

from __future__ import annotations

import time

import pytest

from trading.swap_schedule import (
    ScheduledLeg,
    build_schedule,
    predictions_to_candidates,
    recalculate,
)

NOW = 1_788_540_000.0


def _pending(label, seconds_ahead, predicted, start_price=100.0, hit_rate=0.70):
    """A prediction row.

    hit_rate defaults to a PROVEN horizon so the existing tests exercise what
    they were written for; the gate itself is tested explicitly below.
    """
    return {
        "label": label,
        "resolve_ts": NOW + seconds_ahead,
        "predicted_return": predicted,
        "start_price": start_price,
        "hit_rate": hit_rate,
    }


@pytest.fixture(autouse=True)
def _clear_executed_legs():
    """Executed-leg memory is module state and must not leak between tests.

    It is deliberately process-wide in production -- a replan must not be able
    to re-offer a leg the bot already traded -- so the tests have to clear it
    rather than the module avoiding it.
    """
    import trading.swap_schedule as mod

    mod._EXECUTED_LEG_IDS.clear()
    yield
    mod._EXECUTED_LEG_IDS.clear()


class TestPredictionsToCandidates:
    def test_unresolved_predictions_become_candidates(self):
        out = predictions_to_candidates(
            "AERO-USDC", [_pending("1h", 3600, 0.03)], now=NOW)
        assert len(out) == 1
        assert out[0]["symbol"] == "AERO-USDC"
        assert out[0]["target_price"] == pytest.approx(103.0)

    def test_already_resolved_predictions_are_history_not_plan(self):
        """A forecast whose time has passed belongs to the accuracy tracker."""
        out = predictions_to_candidates(
            "AERO-USDC", [_pending("1h", -3600, 0.03)], now=NOW)
        assert out == []

    def test_negative_forecasts_are_not_scheduled(self):
        out = predictions_to_candidates(
            "AERO-USDC", [_pending("1h", 3600, -0.02)], now=NOW)
        assert out == []

    def test_a_large_forecast_is_not_refused_for_being_large(self):
        """A 300% opportunity is the thing worth acting on, not a bug.

        An earlier version capped forecasts at 200% because nothing bigger
        had been observed, which would have refused exactly the trades most
        worth taking.
        """
        for predicted in (1.74, 3.0, 4.99):
            out = predictions_to_candidates(
                "AERO-USDC",
                [{**_pending("1d", 86400, predicted), "fit_window_sec": 7200.0}],
                now=NOW)
            assert len(out) == 1, f"{predicted:+.2%} should schedule"

    def test_the_clamp_constant_is_refused(self):
        """+/-5.0 exactly is np.clip's output, not a prediction.

        The forecast is exp(intercept + slope*minutes) on log price, so it
        compounds with the horizon; a 0.05%/min drift reaches +7000% over
        three days and is truncated to exactly 5.0.
        """
        for predicted in (5.0, -5.0):
            assert predictions_to_candidates(
                "AERO-USDC",
                [{**_pending("3d", 259200, predicted), "fit_window_sec": 7200.0}],
                now=NOW) == []

    def test_extrapolation_far_beyond_the_fit_window_is_refused(self):
        """Three days projected from two hours is 36x -- the window's noise."""
        assert predictions_to_candidates(
            "AERO-USDC",
            [{**_pending("3d", 259200, 0.05), "fit_window_sec": 7200.0}],
            now=NOW) == []

    def test_a_proportionate_projection_is_kept(self):
        """Three days from a twelve-hour window is 6x, which the fit supports."""
        out = predictions_to_candidates(
            "AERO-USDC",
            [{**_pending("3d", 259200, 0.05), "fit_window_sec": 43200.0}],
            now=NOW)
        assert len(out) == 1

    def test_a_row_without_a_fit_window_is_not_judged_on_one(self):
        """Absence of the field is not evidence of over-extrapolation."""
        out = predictions_to_candidates(
            "AERO-USDC", [_pending("3d", 259200, 0.05)], now=NOW)
        assert len(out) == 1

    def test_banned_symbols_are_not_planned_around(self):
        """A leg on a symbol the entry gate refuses holds capital and buys
        nothing."""
        import trading.swap_schedule as mod

        original = mod._symbol_edge_refusal
        mod._symbol_edge_refusal = lambda sym: "measured negative edge" if sym == "BAD-USDC" else None
        try:
            assert predictions_to_candidates(
                "BAD-USDC", [_pending("1h", 3600, 0.05)], now=NOW) == []
            assert len(predictions_to_candidates(
                "GOOD-USDC", [_pending("1h", 3600, 0.05)], now=NOW)) == 1
        finally:
            mod._symbol_edge_refusal = original

    def test_an_unproven_horizon_may_not_spend_money(self):
        """The four trades that lost money came from horizons with no record.

        "We have not established this works" is not a licence to trade.
        """
        row = _pending("15m", 900, 0.015)
        row.pop("hit_rate")
        assert predictions_to_candidates("CRUX-USDC", [row], now=NOW) == []

    def test_a_coin_flip_horizon_is_refused(self):
        """50% directional accuracy loses the round-trip cost every time."""
        assert predictions_to_candidates(
            "CRUX-USDC", [_pending("15m", 900, 0.015, hit_rate=0.50)], now=NOW) == []

    def test_a_proven_horizon_is_allowed(self):
        assert len(predictions_to_candidates(
            "CRUX-USDC", [_pending("15m", 900, 0.015, hit_rate=0.72)], now=NOW)) == 1

    def test_the_same_forecast_is_traded_at_most_once(self):
        """A leg removed from the plan came straight back on the next replan.

        Measured: 60 executions from 18 distinct legs, one firing eight
        times, each repeat paying a full round trip.
        """
        import trading.swap_schedule as mod

        cands = predictions_to_candidates(
            "CRUX-USDC", [_pending("15m", 900, 0.015)], now=NOW)
        first = build_schedule(cands, capital_usd=18.0, clip_usd=0.75,
                               roundtrip_cost_rate=0.0065, now=NOW)
        assert len(first.legs) == 1

        mod.mark_leg_executed(first.legs[0].leg_id, now=NOW)
        again = build_schedule(cands, capital_usd=18.0, clip_usd=0.75,
                               roundtrip_cost_rate=0.0065, now=NOW)
        assert again.legs == []
        assert any("already been traded" in r["reason"] for r in again.rejected)

    def test_a_new_forecast_on_the_same_symbol_still_schedules(self):
        """Execute-once must not become never-again for that symbol."""
        import trading.swap_schedule as mod

        first = build_schedule(
            predictions_to_candidates("CRUX-USDC", [_pending("15m", 900, 0.015)], now=NOW),
            capital_usd=18.0, clip_usd=0.75, roundtrip_cost_rate=0.0065, now=NOW)
        mod.mark_leg_executed(first.legs[0].leg_id, now=NOW)

        later = build_schedule(
            predictions_to_candidates("CRUX-USDC", [_pending("15m", 1800, 0.015)], now=NOW),
            capital_usd=18.0, clip_usd=0.75, roundtrip_cost_rate=0.0065, now=NOW)
        assert len(later.legs) == 1

    def test_unusable_rows_are_skipped_not_crashed(self):
        rows = [None, {}, {"resolve_ts": "x"}, _pending("1h", 3600, 0.03, 0.0)]
        assert predictions_to_candidates("A-USDC", rows, now=NOW) == []


class TestBuildSchedule:
    def _cands(self, *specs):
        out = []
        for symbol, label, ahead, predicted in specs:
            out.extend(predictions_to_candidates(
                symbol, [_pending(label, ahead, predicted)], now=NOW))
        return out

    def test_a_leg_must_clear_its_own_round_trip(self):
        """+0.1% at a 0.75% cost is a loss the model happens to be sure of."""
        schedule = build_schedule(
            self._cands(("AERO-USDC", "1h", 3600, 0.001)),
            capital_usd=20.0, clip_usd=0.75,
            roundtrip_cost_rate=0.0075, now=NOW)
        assert schedule.legs == []
        assert "does not clear" in schedule.rejected[0]["reason"]

    def test_a_leg_that_clears_the_cost_is_scheduled(self):
        schedule = build_schedule(
            self._cands(("AERO-USDC", "1h", 3600, 0.03)),
            capital_usd=20.0, clip_usd=0.75,
            roundtrip_cost_rate=0.0075, now=NOW)
        assert len(schedule.legs) == 1
        assert schedule.legs[0].symbol == "AERO-USDC"

    def test_the_same_dollar_cannot_fund_two_legs(self):
        """The constraint the per-tick path never had to think about."""
        schedule = build_schedule(
            self._cands(
                ("A-USDC", "1h", 3600, 0.05),
                ("B-USDC", "1h", 3600, 0.04),
                ("C-USDC", "1h", 3600, 0.03),
            ),
            capital_usd=1.0,          # room for ONE $0.75 leg
            clip_usd=0.75,
            roundtrip_cost_rate=0.0075, now=NOW)
        assert len(schedule.legs) == 1
        assert schedule.committed_usd(NOW) <= 1.0
        assert any("uncommitted" in r["reason"] for r in schedule.rejected)

    def test_capital_is_never_oversubscribed(self):
        schedule = build_schedule(
            self._cands(*[(f"S{i}-USDC", "1h", 3600, 0.05) for i in range(10)]),
            capital_usd=3.0, clip_usd=0.75,
            roundtrip_cost_rate=0.0075, now=NOW)
        assert schedule.committed_usd(NOW) <= 3.0 + 1e-9

    def test_one_leg_per_symbol(self):
        """Two horizons on one token are the same bet, not two."""
        cands = self._cands(
            ("AERO-USDC", "1h", 3600, 0.05),
            ("AERO-USDC", "6h", 21600, 0.09),
        )
        schedule = build_schedule(
            cands, capital_usd=20.0, clip_usd=0.75,
            roundtrip_cost_rate=0.0075, now=NOW)
        assert len(schedule.legs) == 1
        assert any("another horizon" in r["reason"] for r in schedule.rejected)

    def test_denser_returns_are_scheduled_first(self):
        """+3% in an hour beats +4% in a day for the same dollar."""
        cands = self._cands(
            ("SLOW-USDC", "1d", 86400, 0.04),
            ("FAST-USDC", "1h", 3600, 0.03),
        )
        schedule = build_schedule(
            cands, capital_usd=0.9, clip_usd=0.75,
            roundtrip_cost_rate=0.0075, now=NOW)
        assert [leg.symbol for leg in schedule.legs] == ["FAST-USDC"]

    def test_no_capital_schedules_nothing_and_says_so(self):
        schedule = build_schedule(
            self._cands(("A-USDC", "1h", 3600, 0.05)),
            capital_usd=0.0, clip_usd=0.75,
            roundtrip_cost_rate=0.0075, now=NOW)
        assert schedule.legs == []
        assert schedule.rejected

    def test_every_rejection_carries_a_reason(self):
        schedule = build_schedule(
            self._cands(
                ("A-USDC", "1h", 3600, 0.05),
                ("B-USDC", "1h", 3600, 0.0001),
            ),
            capital_usd=0.9, clip_usd=0.75,
            roundtrip_cost_rate=0.0075, now=NOW)
        assert all(r.get("reason") for r in schedule.rejected)


class TestGuards:
    def _leg(self, **kw):
        base = dict(
            symbol="AERO-USDC", action="enter", horizon="1h",
            execute_after_ts=NOW, expires_ts=NOW + 3600,
            expected_return=0.03, entry_price=100.0, target_price=103.0,
            notional_usd=0.75, leg_id="L1",
        )
        base.update(kw)
        return ScheduledLeg(**base)

    def test_a_refuted_forecast_is_dropped(self):
        """Half the predicted move against us means the forecast was wrong."""
        leg = self._leg(invalidate_below=98.5)
        schedule = build_schedule([], capital_usd=1.0, clip_usd=0.75,
                                  roundtrip_cost_rate=0.0075, now=NOW)
        schedule.legs = [leg]
        kept, dropped = recalculate(schedule, {"AERO-USDC": 98.0}, now=NOW)
        assert kept.legs == []
        assert "refuted" in dropped[0]["reason"]

    def test_a_leg_still_on_track_survives(self):
        leg = self._leg(invalidate_below=98.5)
        schedule = build_schedule([], capital_usd=1.0, clip_usd=0.75,
                                  roundtrip_cost_rate=0.0075, now=NOW)
        schedule.legs = [leg]
        kept, dropped = recalculate(schedule, {"AERO-USDC": 101.0}, now=NOW)
        assert len(kept.legs) == 1
        assert dropped == []

    def test_an_expired_forecast_is_dropped(self):
        leg = self._leg(expires_ts=NOW - 1)
        schedule = build_schedule([], capital_usd=1.0, clip_usd=0.75,
                                  roundtrip_cost_rate=0.0075, now=NOW)
        schedule.legs = [leg]
        kept, dropped = recalculate(schedule, {"AERO-USDC": 101.0}, now=NOW)
        assert kept.legs == []
        assert "elapsed" in dropped[0]["reason"]

    def test_a_dependent_leg_goes_with_its_dependency(self):
        first = self._leg(leg_id="hop-1", invalidate_below=98.5)
        second = self._leg(symbol="B-USDC", leg_id="hop-2", depends_on="hop-1")
        schedule = build_schedule([], capital_usd=2.0, clip_usd=0.75,
                                  roundtrip_cost_rate=0.0075, now=NOW)
        schedule.legs = [first, second]
        kept, dropped = recalculate(
            schedule, {"AERO-USDC": 98.0, "B-USDC": 100.0}, now=NOW)
        assert kept.legs == []
        assert any("depends on hop-1" in d["reason"] for d in dropped)

    def test_a_missing_price_does_not_invalidate(self):
        """No quote is not evidence the forecast was wrong."""
        leg = self._leg(invalidate_below=98.5)
        schedule = build_schedule([], capital_usd=1.0, clip_usd=0.75,
                                  roundtrip_cost_rate=0.0075, now=NOW)
        schedule.legs = [leg]
        kept, _ = recalculate(schedule, {}, now=NOW)
        assert len(kept.legs) == 1

    def test_ripe_legs_are_the_ones_due_now(self):
        soon = self._leg(leg_id="now", execute_after_ts=NOW - 10)
        later = self._leg(symbol="B-USDC", leg_id="later",
                          execute_after_ts=NOW + 600)
        schedule = build_schedule([], capital_usd=2.0, clip_usd=0.75,
                                  roundtrip_cost_rate=0.0075, now=NOW)
        schedule.legs = [soon, later]
        assert [leg.leg_id for leg in schedule.ripe(NOW)] == ["now"]
