"""``strategy_edge_gate`` must judge at the MEASURED cost, not a 0.650% constant.

``services/symbol_edge_gate`` moved to ``services.round_trip_cost`` (0.4653%
measured 2026-09-10) and ``services/strategy_edge_gate`` did not -- it read its
``ROUND_TRIP_COST = 0.0065`` literal directly, under a comment claiming it was
"the same figure symbol_edge_gate tests against". Two gates answered one
question at two different bars, and the second one charged every strategy a
0.185%-of-notional surcharge it never pays.

That surcharge is not a rounding difference; it decides bans. Re-priced against
the measured cost, 2 of the 7 standing verdicts are entirely its artifact::

    rsi_reversal          10 trips  mean -0.625%  t=-1.85 -> -1.44  LIFTS
    stochastic_reversal    4 trips  mean -0.168%  t=-1.86 -> -1.29  LIFTS
    bus_schedule           4 trips  mean -1.366%  t=-3.70 -> -3.36  stands
    obv_accumulation@1w    9 trips  mean -1.668%  t=-8.39 -> -7.72  stands

rsi_reversal is the strategy closest to graduation on tradeable evidence, and
it was being refused entries for failing to clear a cost it is not charged.

These tests fail against the pre-2026-09-10 body, which used the literal.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def gate():
    mod = importlib.import_module("services.strategy_edge_gate")
    mod._cache.clear()
    mod._cache_built_at = 0.0
    yield mod
    mod._cache.clear()
    mod._cache_built_at = 0.0


def _book(monkeypatch, gate, book):
    monkeypatch.setattr(gate, "_load_book", lambda *a, **k: dict(book))


def _measured(monkeypatch, value):
    """Point the measurement at ``value``, wherever the gate imports it from."""
    rtc = importlib.import_module("services.round_trip_cost")
    monkeypatch.setattr(rtc, "round_trip_cost", lambda *a, **k: value)


def test_the_verdict_is_taken_at_the_measured_cost_not_the_constant(gate, monkeypatch):
    """A book that clears 0.4653% but not 0.650% must NOT be banned.

    This is rsi_reversal's shape: a mean below the constant, above the truth.
    """
    _measured(monkeypatch, 0.004653005333880809)
    # Ten trips whose mean (+0.55%) sits between the measured cost and the
    # constant, with dispersion tight enough that the t fires at 0.650%.
    _book(monkeypatch, gate, {"rsi_reversal": [0.0055] * 10})

    assert gate.refusal_reason("rsi_reversal") is None

    # And the SAME book at the old constant is refused -- which is what makes
    # this a measurement of the bug rather than of an always-allowed fixture.
    gate._cache.clear()
    gate._cache_built_at = 0.0
    _measured(monkeypatch, 0.0065)
    assert gate.refusal_reason("rsi_reversal") is not None


def test_a_genuinely_losing_strategy_is_still_refused(gate, monkeypatch):
    """Re-pricing must not switch the gate off. obv_accumulation@1w stands.

    Its mean is -1.668%, four times the measured cost in the wrong direction;
    no honest cost lifts it, and a change that did would be loosening the bar.
    """
    _measured(monkeypatch, 0.004653005333880809)
    _book(monkeypatch, gate, {"obv_accumulation@1w": [-0.0167, -0.0165, -0.0170] * 3})

    reason = gate.refusal_reason("obv_accumulation@1w")
    assert reason is not None
    assert "0.465% cost" in reason, reason


def test_the_reason_string_quotes_the_cost_it_actually_used(gate, monkeypatch):
    """A reason that cites a cost the verdict did not use is a false record.

    Every stored ban in this system is read back by a human deciding whether
    it was correct; a reason citing 0.650% while the test ran at 0.4653% sends
    that reader to re-derive a number the code never computed.
    """
    _measured(monkeypatch, 0.004653005333880809)
    _book(monkeypatch, gate, {"bus_schedule": [-0.0140, -0.0130, -0.0135, -0.0145]})

    reason = gate.refusal_reason("bus_schedule")
    assert reason is not None
    assert "0.465% cost" in reason, reason
    assert "0.650%" not in reason, reason


def test_an_unreadable_measurement_falls_back_rather_than_charging_zero(gate, monkeypatch):
    """A broken read is not a free round trip.

    Returning 0.0 would make every non-negative strategy clear cost and ban
    nothing -- a gate switched off by an exception, which is how this repo has
    lost money before.
    """
    rtc = importlib.import_module("services.round_trip_cost")

    def _boom(*a, **k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(rtc, "round_trip_cost", _boom)
    assert gate._cost() == gate.ROUND_TRIP_COST

    monkeypatch.setattr(rtc, "round_trip_cost", lambda *a, **k: 0.0)
    assert gate._cost() == gate.ROUND_TRIP_COST

    monkeypatch.setattr(rtc, "round_trip_cost", lambda *a, **k: float("nan"))
    assert gate._cost() == gate.ROUND_TRIP_COST


def test_both_edge_gates_charge_the_same_cost(gate):
    """The invariant the old comment asserted and the code stopped honouring.

    Asserts on what the two modules COMPUTE, not on what either says about
    itself -- the comment claiming they agreed outlived the agreement by
    however long it took to measure this.
    """
    symbol_gate = importlib.import_module("services.symbol_edge_gate")
    from services.round_trip_cost import round_trip_cost

    assert gate._cost() == pytest.approx(round_trip_cost(db_path=symbol_gate.DB_PATH))
