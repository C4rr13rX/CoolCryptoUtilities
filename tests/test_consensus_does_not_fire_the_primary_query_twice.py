"""Consensus must not count the primary query set as one of its own witnesses.

``discriminating_collections`` returns the measured query in DISTINCTNESS
order, so on the AERO-USDC corpus the primary arrives as
``('geometry','temporal','cross')`` while ``CONSENSUS_QUERIES[0]`` is
``('temporal','geometry','cross')`` -- the same query, a different tuple.
``predict`` deduped its members by TUPLE, so the duplicate survived as a fifth
member, and it cost three things at once:

  * a fifth round trip on a path documented as costing four;
  * an inflated unanimity rate, because a member that IS the primary cannot
    disagree with it on the merits -- every "99.4% unanimous" figure quoted in
    this repo was measured over 3 distinct query sets plus a copy;
  * spurious abstentions. The node is not perfectly deterministic -- an A-vs-A
    control moved 4 of 100 held-out predictions in pass 111 -- so the copy
    could "disagree" with the primary on node noise alone and turn an
    undisputed answer into a ``split`` hold.

This test drives ``predict`` against a stub transport that records every query
it is asked to fire, so it asserts on what the code DOES rather than on what
the comment says.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.omen_brain import (  # noqa: E402
    COLLECTIONS, CONSENSUS_QUERIES, OMEN_POOL, OMEN_TROUGH, OmenBrain,
)


class _RecordingBrain(OmenBrain):
    """Answers every query with the same label and remembers the pool sets.

    Only the STAGE 2 fires are recorded. Stage 1 asks the regime pool a
    separate question and is not a consensus member; counting it was the first
    thing this test got wrong.
    """

    def __init__(self):
        super().__init__(endpoint="127.0.0.1:9")
        self.fired = []

    def supports_multi(self) -> bool:
        return True

    def _predict(self, streams, target_pool):
        if target_pool == OMEN_POOL:
            self.fired.append(tuple(sorted(s["pool_id"] for s in streams)))
        return f"omen {OMEN_TROUGH}", 0.9

    def _degenerate(self) -> bool:
        return False


def _frames():
    """A frame per collection, so every query set has something to fire."""
    return {name: f"{name} x=u1 y=r2" for name in
            ("temporal", "geometry", "cross", "flow", "volatility",
             "horizon", "instrument")}


def _run(query):
    brain = _RecordingBrain()
    brain.predict(_frames(), symbol="AERO-USDC", chain="base", as_of_ts=1,
                  price=1.0, horizon_bars=12, bar_seconds=3600,
                  regime="calm", query_collections=list(query),
                  consensus=True)
    return brain.fired


def test_a_reordered_primary_is_not_fired_as_a_second_witness():
    # The ordering discriminating_collections actually produced on AERO-USDC.
    primary = ("geometry", "temporal", "cross")
    assert tuple(CONSENSUS_QUERIES[0]) != primary, (
        "this test is only meaningful while a consensus query set is a "
        "REORDERING of the measured set")
    assert frozenset(CONSENSUS_QUERIES[0]) == frozenset(primary), (
        "and is the same set")

    fired = _run(primary)

    # Four distinct query sets, not five: the reordered primary is dropped.
    assert len(fired) == len(CONSENSUS_QUERIES), (
        f"consensus fired {len(fired)} queries; CONSENSUS_QUERIES names "
        f"{len(CONSENSUS_QUERIES)}")
    assert len(set(fired)) == len(fired), (
        "no query set may be fired twice -- a duplicate agrees with the "
        "primary by construction and abstains on node noise")


def test_the_primary_is_still_fired_first_and_unchanged():
    """The answer must stay the primary set's answer, deduping or not."""
    primary = ("geometry", "temporal", "cross")
    fired = _run(primary)
    expected = tuple(sorted(
        c.pool_id for c in COLLECTIONS if c.name in primary))
    assert fired[0] == expected, (
        "the first fire must be the primary query set: a consensus read and a "
        "plain read differ only on whether the answer is ADMITTED")


def test_an_already_canonical_primary_still_fires_four():
    """The dedupe must not drop a member when nothing was reordered."""
    fired = _run(tuple(CONSENSUS_QUERIES[0]))
    assert len(fired) == len(CONSENSUS_QUERIES)
    assert len(set(fired)) == len(fired)
