"""A lookback label must not take the tick off a real forecast horizon.

One tick produces many candidates and spends exactly ONE. The chaos layer
refuses a candidate whose proposed horizon exceeds the measured decay limit,
and for a suffixed strategy id that "proposed horizon" is the id's LOOKBACK
label, not a forecast. Measured over 24h of ``entry-arbitration`` rows: a tick
spent on a suffixed candidate clears the lattice 12.4% of the time and one
spent on a suffixless candidate clears it 82.1% of the time, and on 246 ticks
a day the arbitrator picked the suffixed one while the suffixless one was
standing on the same tick.

The second test is the one that matters most: this deferral must never reduce
the candidate count to zero. Buying a lower refusal rate by admitting fewer
candidates is the failure mode the item that commissioned this work forbids
explicitly, and it is the failure mode a naive "drop all long horizons" has.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from trading.scheduler import (
    _MAX_USABLE_HORIZON_SEC,
    _HORIZON_SUFFIX,
    _HORIZON_UNIT_SEC,
)
from trading.scheduler import BusScheduler


def _cand(strategy_id: str, score: float = 1.0, action: str = "enter"):
    return {
        "directive": SimpleNamespace(action=action, strategy_id=strategy_id),
        "score": score,
    }


def _ids(candidates):
    return [c["directive"].strategy_id for c in candidates]


@pytest.fixture
def defer():
    """Bind the method without constructing a scheduler (it needs a db)."""
    return lambda cands: BusScheduler._defer_label_horizon_candidates(
        SimpleNamespace(), cands
    )


def test_the_suffix_parser_agrees_with_the_ids_actually_offered():
    """The labels are not hypothetical -- these are ids seen at arbitration."""
    seen = {
        "rsi_reversal@12h": 12 * 3600,
        "rsi_reversal@5h": 5 * 3600,
        "stochastic_reversal@3d": 3 * 86400,
        "supertrend_follow@1w": 604800,
        "obv_accumulation@1d": 86400,
    }
    for sid, expected in seen.items():
        match = _HORIZON_SUFFIX.search(sid)
        assert match is not None, sid
        got = int(match.group(1)) * _HORIZON_UNIT_SEC[match.group(2)]
        assert got == expected, sid
    for sid in ("rsi_reversal", "atf_static", "donchian_breakout", "tf_forecast"):
        assert _HORIZON_SUFFIX.search(sid) is None, sid


def test_a_labelled_candidate_is_deferred_when_a_real_one_stands_beside_it(defer):
    candidates = [_cand("rsi_reversal@12h", 9.0), _cand("rsi_reversal", 0.1)]
    kept, deferred = defer(candidates)
    assert _ids(kept) == ["rsi_reversal"]
    assert deferred == ["rsi_reversal@12h"]


def test_the_deferral_cannot_empty_the_candidate_set(defer):
    """79.7% of real ticks offer ONLY labelled candidates. Keep them all."""
    candidates = [
        _cand("rsi_reversal@12h", 9.0),
        _cand("stochastic_reversal@3d", 8.0),
        _cand("supertrend_follow@1w", 7.0),
    ]
    kept, deferred = defer(candidates)
    assert _ids(kept) == _ids(candidates)
    assert deferred == []


def test_a_single_candidate_tick_is_never_touched(defer):
    candidates = [_cand("donchian_breakout@5d", 3.0)]
    kept, deferred = defer(candidates)
    assert _ids(kept) == ["donchian_breakout@5d"]
    assert deferred == []


def test_non_enter_candidates_are_left_alone(defer):
    """The deferral reasons about ENTRY horizons; an exit is not its business."""
    candidates = [
        _cand("rsi_reversal@12h", 9.0, action="exit"),
        _cand("rsi_reversal@5h", 8.0, action="exit"),
    ]
    kept, deferred = defer(candidates)
    assert _ids(kept) == _ids(candidates)
    assert deferred == []


def test_an_exit_candidate_does_not_count_as_the_real_horizon_survivor(defer):
    """An exit standing beside two labelled enters must not license deferring
    both of them -- that would leave the tick with no enter candidate at all,
    which is the empty-set failure in disguise."""
    candidates = [
        _cand("rsi_reversal@12h", 9.0),
        _cand("stochastic_reversal@3d", 8.0),
        _cand("atf_static", 1.0, action="exit"),
    ]
    kept, deferred = defer(candidates)
    enters = [c for c in kept if c["directive"].action == "enter"]
    assert enters, "the tick must keep at least one enter candidate"
    assert deferred == []


def test_a_short_label_inside_the_decay_limit_is_kept(defer):
    """This is a decay-limit test, not an anti-suffix test. A label BELOW the
    measured usable horizon describes a forecast the chaos layer can accept,
    so it must survive beside a suffixless candidate."""
    assert _MAX_USABLE_HORIZON_SEC >= 60.0
    candidates = [_cand("rsi_reversal@1m", 9.0), _cand("rsi_reversal", 0.1)]
    kept, deferred = defer(candidates)
    assert _ids(kept) == ["rsi_reversal@1m", "rsi_reversal"]
    assert deferred == []
