"""A brain that cannot answer must not read as a brain with nothing to say.

``trading.brain_bridge.predict_outcome`` returns ``(None, 0.0)`` for a refused
connection, a 30-second timeout, a bad body AND for a genuine abstention, and
its callers in the money path treat None as "no opinion". On 2026-09-07 the
node on :8090 -- the ``BRAIN_ENDPOINT`` default that ``trading/bot.py:5014``
and ``services/ga_service.py:296`` both query -- served ``/health`` and
``/brain/stats`` normally for 32 hours while every ``/brain/predict`` timed
out. Every brain query in the live lane returned nothing for that entire
window and nothing logged, counted or reported it.

These tests pin the distinction ``services.brain_health`` draws. They are
written against the two broken nodes as MEASURED that morning, not against
invented ones:

    :8090  stats answered (521224 concepts / 10725783 terminals),
           /brain/predict timed out          -> BLOCKED
    :8091  /brain/predict answered in 0.00s,
           known_atom_count 0, no terminals  -> EMPTY

Both must report ``usable is False``. The old behaviour cannot pass these:
``predict_outcome`` has no state, no reason and no ``usable`` -- it collapses
all four cases onto one value, which is the defect.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.brain_health import (  # noqa: E402
    BLOCKED, EMPTY, READY, UNREACHABLE, BrainHealth, probe,
)

ENDPOINT = "http://127.0.0.1:8090"

#: The fabric numbers :8090 was still reporting while it could not answer.
LOADED_STATS = {
    "total_concepts": 521224,
    "total_terminals": 10725783,
    "resident_terminals": 1351667,
}

#: What an empty node replies with -- :8091's actual body, verbatim.
EMPTY_ANSWER = {
    "answer": None,
    "integrated_confidence": 0.0,
    "known_atom_count": 0,
    "learning": False,
    "outside_grounding": True,
}


def _probe(*, alive=True, stats=None, answer=None, delay=0.0, deadline=8.0):
    """Drive probe() with injected transports -- no node, no network."""

    def _listening(_endpoint, _timeout):
        return alive

    def _stats(_endpoint, _timeout):
        return stats

    def _predict(_endpoint, _frame, timeout):
        if delay:
            # Model a call that blows the deadline. Sleep the smaller of the
            # delay and the transport timeout so the test stays fast; the
            # returned value is what matters.
            time.sleep(min(delay, 0.05))
        if delay and delay >= timeout:
            return None  # the real transport returns None on timeout
        return answer

    return probe(ENDPOINT, deadline=deadline, listening=_listening,
                 stats=_stats, predict=_predict)


def test_a_node_serving_stats_while_predict_times_out_is_blocked_not_ready():
    """The exact :8090 shape: healthy gauges, dead fabric.

    Reading stats as evidence of readiness is what let this run 32 hours.
    """
    health = _probe(stats=LOADED_STATS, answer=None)

    assert health.state == BLOCKED
    assert health.usable is False
    assert health.silence_is_an_outage is True
    # The reason must name the trap, so the next reader does not repeat it.
    assert "stats answered" in health.reason
    # The fabric numbers are still reported -- they are diagnostic, and a
    # blocked node that LOOKS big is the confusing case worth showing.
    assert health.total_terminals == 10725783


def test_a_fast_empty_fabric_is_not_healthiest_just_because_it_is_fastest():
    """The :8091 shape: instant replies from a brain that knows nothing.

    An empty fabric answers faster than a loaded one, so any check that grades
    on latency alone ranks the unloaded brain best. Emptiness has to be read
    off fabric size.
    """
    health = _probe(stats={"total_terminals": 0, "total_concepts": 0},
                    answer=EMPTY_ANSWER)

    assert health.state == EMPTY
    assert health.usable is False
    assert health.known_atom_count == 0
    assert health.latency_seconds < 1.0  # it WAS fast, and still not usable


def test_no_listener_is_unreachable_and_distinguishable_from_blocked():
    """'Not running' and 'running but stuck' are different repairs."""
    health = _probe(alive=False)

    assert health.state == UNREACHABLE
    assert health.usable is False
    assert health.state != BLOCKED


def test_only_a_loaded_fabric_that_answers_in_time_is_ready():
    health = _probe(stats=LOADED_STATS,
                    answer={"answer": None, "known_atom_count": 4})

    assert health.state == READY
    assert health.usable is True
    # THE POINT OF THE MODULE: under READY -- and only under READY -- a null
    # answer from the bridge is an abstention rather than an outage.
    assert health.silence_is_an_outage is False


def test_a_null_answer_means_opposite_things_under_ready_and_blocked():
    """One value, two meanings -- which is the defect being fixed.

    The bridge returns (None, 0.0) in both of these. Nothing downstream could
    tell them apart before this module existed.
    """
    blocked = _probe(stats=LOADED_STATS, answer=None)
    ready = _probe(stats=LOADED_STATS, answer={"answer": None, "known_atom_count": 0})

    # Identical bridge-level observation: no answer.
    assert blocked.usable != ready.usable
    assert {blocked.state, ready.state} == {BLOCKED, READY}


def test_a_slow_answer_past_the_decision_deadline_counts_as_blocked():
    """A reply that arrives after the decision is made is not a reply.

    The bridge's 30s transport timeout exists so a slow call still lands; a
    trade cannot wait that long. These are different deadlines and the health
    probe uses the trading one.
    """
    health = probe(
        ENDPOINT, deadline=0.01,
        listening=lambda _e, _t: True,
        stats=lambda _e, _t: LOADED_STATS,
        predict=lambda _e, _f, _t: (time.sleep(0.05) or {"known_atom_count": 3}),
    )

    assert health.state == BLOCKED
    assert health.usable is False
    assert "past the" in health.reason


def test_the_probe_never_raises_even_when_every_transport_explodes():
    """A health check that can throw is one more silent failure."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("transport exploded")

    health = probe(ENDPOINT, listening=_boom, stats=_boom, predict=_boom)

    assert isinstance(health, BrainHealth)
    assert health.usable is False


def test_the_probe_never_writes_to_the_fabric(monkeypatch):
    """It must not train the brain it is measuring.

    /brain/observe and /brain/consolidate both mutate; only /brain/predict is
    read-only. This asserts on the URL the REAL transport requests -- grepping
    the source would pass on a docstring, which is the mistake this repo has
    shipped before (a test asserting on a word that appeared only in a comment).
    """
    import services.brain_health as module

    requested: list[str] = []

    class _Response:
        def read(self):
            return b'{"known_atom_count": 1}'

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def _urlopen(request, timeout=None):
        requested.append(request.full_url if hasattr(request, "full_url") else str(request))
        return _Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", _urlopen)

    answer = module._default_predict(ENDPOINT, "probe frame", 5.0)

    assert answer == {"known_atom_count": 1}
    assert requested == [f"{ENDPOINT}/brain/predict"]
    # The mutating routes must never be reached by a health probe.
    assert not any(path in url for url in requested
                   for path in ("/brain/consolidate", "/brain/observe", "/brain/tick"))


def test_states_are_byte_disjoint_so_one_cannot_be_read_as_another():
    """The brain substrate has no tokenizer and neither do log greps.

    'loss' inside 'loss_big' cost this repo a whole experiment round
    (OUTCOME_TOKENS in trading/brain_bridge.py documents it). State names get
    the same treatment.
    """
    for state in (UNREACHABLE, BLOCKED, EMPTY, READY):
        others = [s for s in (UNREACHABLE, BLOCKED, EMPTY, READY) if s != state]
        for other in others:
            assert state not in other, f"{state!r} is a substring of {other!r}"


@pytest.mark.parametrize("stats_value", [None, {}, {"total_terminals": None}])
def test_a_node_whose_stats_are_missing_still_grades_on_the_fabric_call(stats_value):
    """Stats are colour, not evidence -- their absence must not crash or pass."""
    blocked = _probe(stats=stats_value, answer=None)
    assert blocked.state == BLOCKED
    assert blocked.usable is False

    empty = _probe(stats=stats_value, answer=EMPTY_ANSWER)
    assert empty.state == EMPTY
    assert empty.usable is False
