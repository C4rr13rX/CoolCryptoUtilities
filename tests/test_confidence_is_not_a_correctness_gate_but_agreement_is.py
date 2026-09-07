"""Confidence cannot tell a caller the answer is wrong. Agreement can.

Measured 2026-09-07 against the fabric trained on 2725 AERO-USDC pairs,
firing four query sets per ask (``CONSENSUS_QUERIES``):

                   answers agree            answers split
    train recall   99.4%  (169/170, 85.0%)  73.3%  (22/30, 15.0%)
    held-out       33.3%  ( 32/ 96, 19.2%)  29.7%  (120/404, 80.8%)

    mean confidence when RIGHT vs when WRONG
    train recall   0.999 vs 0.969   gap +0.030
    held-out       0.965 vs 0.967   gap -0.002

So the brain's own confidence separates a correct answer from an incorrect
one by essentially nothing, while unanimity separates them by 26 points on
reproduction. ``OMEN_CONFIDENCE_FLOOR`` remains a NOISE filter -- it rejects
frames built from garbage, which score 0.61-0.68 -- and it must never be
read as a correctness filter. That distinction is what these tests pin.

Read the held-out row honestly: unanimity is a REPRODUCTION gate, not an
edge gate. 33.3% against a 31.2% majority class is not an edge, and the buys
unanimity admitted still lost 0.2516% per trade. Nothing here says the omen
strategy should be switched on.

None of this needs a running node.
"""
from __future__ import annotations

import pytest

from trading.omen_brain import (
    CONSENSUS_QUERIES, COLLECTIONS_BY_NAME, OMEN_POOL, OMEN_TROUGH, Omen,
    OmenBrain, REGIME_POOL, omen_frame,
)

ALL_NAMES = ("geometry", "temporal", "flow", "volatility", "cross",
             "horizon", "instrument")


def _frames() -> dict:
    return {name: f"{COLLECTIONS_BY_NAME[name].prefix} k=1" for name in ALL_NAMES}


class _Scripted(OmenBrain):
    """Answers a scripted sequence of omen labels, one per query fired."""

    def __init__(self, answers) -> None:
        super().__init__(endpoint="http://127.0.0.1:1")
        self._answers = list(answers)
        self._multi_supported = True
        self.omen_queries = 0

    def _predict(self, streams, target_pool):
        if target_pool == REGIME_POOL:
            return "regime chop", 0.9
        index = min(self.omen_queries, len(self._answers) - 1)
        self.omen_queries += 1
        return omen_frame(self._answers[index]), 0.99


def _predict(brain, **kwargs):
    return brain.predict(
        _frames(), symbol="AERO-USDC", chain="base", as_of_ts=1_700_000_000,
        price=1.0, horizon_bars=12, bar_seconds=3600, regime="chop", **kwargs)


# --- the gate -------------------------------------------------------------

def test_a_split_answer_is_refused_not_traded():
    """73.3% right on a split. It is an abstention, not a weak signal."""
    brain = _Scripted([OMEN_TROUGH, "crest", OMEN_TROUGH, OMEN_TROUGH])
    omen = _predict(brain, consensus=True)
    assert omen.verdict == "split"
    assert omen.action == "hold"
    assert not omen.is_actionable, (
        "a split omen reached the money path; reproduction on a split is "
        "73.3% against 99.4% on a unanimous one")


def test_a_unanimous_answer_is_admitted():
    brain = _Scripted([OMEN_TROUGH] * 4)
    omen = _predict(brain, consensus=True)
    assert omen.verdict == "admitted"
    assert omen.omen == OMEN_TROUGH
    assert omen.support["unanimous"] is True


def test_consensus_fires_every_member_query():
    brain = _Scripted([OMEN_TROUGH] * 4)
    _predict(brain, consensus=True)
    assert brain.omen_queries == len(CONSENSUS_QUERIES), (
        "a consensus read that fires fewer queries than it claims is not a "
        "consensus")


def test_a_single_query_read_costs_exactly_one_round_trip():
    """Consensus is opt-in. The default path must not quadruple latency."""
    brain = _Scripted([OMEN_TROUGH])
    _predict(brain)
    assert brain.omen_queries == 1


def test_consensus_and_a_plain_read_agree_on_what_was_predicted():
    """The primary member is the same query either way, so the two reads can
    only differ on whether the answer is ADMITTED -- never on the label."""
    plain = _predict(_Scripted([OMEN_TROUGH, "crest", "crest", "crest"]))
    voted = _predict(_Scripted([OMEN_TROUGH, "crest", "crest", "crest"]),
                     consensus=True)
    assert plain.omen == OMEN_TROUGH
    assert voted.verdict == "split"
    assert voted.support["member_answers"][0] == OMEN_TROUGH


def test_a_split_reports_which_members_disagreed():
    brain = _Scripted([OMEN_TROUGH, "murk", OMEN_TROUGH, "crest"])
    omen = _predict(brain, consensus=True)
    assert omen.support["member_answers"] == [OMEN_TROUGH, "murk",
                                              OMEN_TROUGH, "crest"]


def test_high_confidence_does_not_rescue_a_split():
    """The whole point. Every scripted answer here carries 0.99."""
    brain = _Scripted([OMEN_TROUGH, "crest", OMEN_TROUGH, OMEN_TROUGH])
    omen = _predict(brain, consensus=True, confidence_floor=0.0)
    assert omen.verdict == "split", (
        "confidence separated right from wrong by -0.002 held-out; it "
        "cannot be allowed to override the gate that separated them by 26 "
        "points")


# --- the schema -----------------------------------------------------------

def test_split_is_a_named_verdict_carrying_hold():
    assert "split" in Omen.VERDICTS
    with pytest.raises(ValueError):
        Omen(schema_version="omen.v1", symbol="X", chain="base",
             as_of_ts=1, price=1.0, horizon_bars=12, bar_seconds=3600,
             omen=OMEN_TROUGH, action="buy", confidence=0.99,
             cost_fraction=0.0065, threshold_fraction=0.00975,
             expected_move_fraction=0.00975, verdict="split")


def test_the_member_queries_all_contain_the_discriminating_collections():
    """A member built only from diluting streams would vote noise into the
    gate and make agreement mean less, not more."""
    for member in CONSENSUS_QUERIES:
        for required in ("temporal", "geometry", "cross"):
            assert required in member, (
                f"member {member} omits {required}, which the dilution law "
                f"measured as load-bearing")


def test_there_are_enough_members_for_agreement_to_mean_anything():
    assert len(CONSENSUS_QUERIES) >= 3
    assert len({tuple(m) for m in CONSENSUS_QUERIES}) == len(CONSENSUS_QUERIES), (
        "duplicate members agree with themselves for free")
