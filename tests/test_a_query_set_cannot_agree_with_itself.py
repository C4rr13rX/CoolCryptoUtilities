"""The agreement census must not count a query set agreeing with ITSELF.

Two defects this pins, both of which were live when the census was written.

1. ``discriminating_collections`` returns the measured query set in
   DISTINCTNESS order, so on the AERO corpus the primary measured as
   ``('geometry','temporal','cross')`` while ``CONSENSUS_QUERIES[0]`` is
   ``('temporal','geometry','cross')``. Those are the same SET and fire the
   same query, but they are different TUPLES. ``omen_brain``'s consensus
   construction dedupes by tuple, so it keeps the duplicate as a fifth member
   which agrees with the primary BY CONSTRUCTION -- inflating every
   "unanimous" rate this repo has quoted and costing an extra round trip on
   the live path. The census dedupes by ``frozenset`` instead.

2. A run in which the query sets NEVER disagree measured nothing about
   agreement: either the extra collections are redundant or the query path is
   not firing them, which is exactly the pass-108 failure where two arms came
   back byte-for-byte identical over 180 held-out predictions. The census must
   FAIL that run rather than report "agreement pays".
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from trading.omen_brain import CONSENSUS_QUERIES, OMEN_CREST, OMEN_TROUGH  # noqa: E402


def _members(measured):
    """The census's member construction, deduped by SET rather than by tuple."""
    seen = {frozenset(measured)}
    members = [tuple(measured)]
    for candidate in CONSENSUS_QUERIES:
        if frozenset(candidate) not in seen:
            seen.add(frozenset(candidate))
            members.append(tuple(candidate))
    return members


def test_a_reordered_primary_is_not_a_second_member():
    # The exact ordering measured on 0004_AERO-USDC.json, pass 111.
    measured = ("geometry", "temporal", "cross")
    members = _members(measured)

    # Dedupe by tuple -- what omen_brain does -- keeps the reordered duplicate.
    by_tuple = [measured] + [m for m in CONSENSUS_QUERIES
                             if tuple(m) != tuple(measured)]
    assert len(by_tuple) == 5, "the tuple dedupe is expected to keep the duplicate"
    assert frozenset(by_tuple[1]) == frozenset(measured), (
        "member 1 fires the same query as the primary")

    # Dedupe by set does not.
    assert len(members) == 4
    sets = [frozenset(m) for m in members]
    assert len(sets) == len(set(sets)), "no member may repeat another as a set"
    assert members[0] == measured, "the primary must stay first: it is the answer"


def test_zero_disagreement_is_a_failed_run_not_a_result():
    """A census where every set answers the same thing has measured nothing."""
    source = (ROOT / "scripts" / "omen_agreement_census.py").read_text(
        encoding="utf-8")
    assert "if disagreements == 0:" in source, (
        "the census must detect a run in which the query sets never disagreed")
    tail = source.split("if disagreements == 0:", 1)[1]
    assert "return 3" in tail.split("return 0", 1)[0], (
        "a zero-disagreement run must exit NONZERO, not report an agreement "
        "number -- that is the pass-108 failure")


def test_an_arm_is_scored_on_money_before_accuracy():
    """Per-trade net and n, never a bare accuracy on a shrunken subset."""
    from omen_agreement_census import score

    rows = [
        # a trough that paid, a trough that did not, a crest that fell
        {"verdict": "admitted", "label": OMEN_TROUGH, "truth": OMEN_TROUGH,
         "forward": 0.05, "actionable": True, "action": "buy", "confidence": 0.9},
        {"verdict": "admitted", "label": OMEN_TROUGH, "truth": OMEN_CREST,
         "forward": -0.02, "actionable": True, "action": "buy", "confidence": 0.9},
        {"verdict": "admitted", "label": OMEN_CREST, "truth": OMEN_CREST,
         "forward": -0.03, "actionable": False, "action": "hold", "confidence": 0.9},
        {"verdict": "hold", "label": "hold", "truth": OMEN_TROUGH,
         "forward": 0.01, "actionable": False, "action": "hold", "confidence": 0.1},
    ]
    result = score(rows, "ALL")

    # n on BOTH sides -- a subset result without its n is not a result.
    assert result["n_samples"] == 4
    assert result["n_admitted"] == 3

    # Money: two buys, each charged the round trip.
    assert result["buy_omens"] == 2
    assert result["buy_net_per_trade"] < 0.05, "the round trip must be charged"

    # Trough precision is "did the trade PAY", not "was the label right":
    # one of the two troughs cleared the round trip.
    assert result["trough_called"] == 2
    assert result["trough_precision_paid"] == 0.5

    # The crest is scored as an EXIT against the forward return, never shorted.
    assert result["crest_called"] == 1
    assert result["crest_precision_fell"] == 1.0

    # Exact accuracy is over the ANSWERED asks, and is the demoted control.
    assert result["exact_accuracy"] == 2 / 3


def test_the_census_refuses_productions_fabric():
    source = (ROOT / "scripts" / "omen_agreement_census.py").read_text(
        encoding="utf-8")
    assert '"8090" in endpoint' in source, (
        "the census must refuse to train or measure against production's node")
