"""Hypotheses generated from refusals.

The closed-trade sample is selected by the very gates being evaluated, so it
can say how good our accepted trades were and nothing about whether the gates
are right. The refusals are the other half of that record, and these tests are
about turning them into claims that can actually be checked.
"""

from __future__ import annotations

from django.test import SimpleTestCase

from .hypotheses import MIN_COMPARISON, MIN_REFUSALS, as_theorems, generate


def _census(count: int, symbols: dict, key: str = "entry-refused-test"):
    return {key: {"reason": key, "layer": "", "count": count,
                  "symbols": symbols, "examples": ["because"]}}


def _rows(symbol_returns: dict) -> list:
    out = []
    for symbol, values in symbol_returns.items():
        for value in values:
            out.append({"symbol": symbol, "return": value})
    return out


class GenerationTests(SimpleTestCase):
    def test_a_thin_reason_is_not_turned_into_a_claim(self):
        """A rule that fired four times is an anecdote, not a belief.

        Generating a hypothesis for it fills the report with untestable noise
        that crowds out the ones worth reading.
        """
        proposals = generate(_rows({"A-USDC": [0.01] * 40}),
                             _census(MIN_REFUSALS - 1, {"A-USDC": 3}))
        assert proposals == []

    def test_a_reason_with_too_few_closed_trades_is_reported_as_untestable(self):
        """"We cannot judge this yet" is itself worth saying."""
        proposals = generate(_rows({"A-USDC": [0.01] * 3,
                                    "B-USDC": [0.02] * 40}),
                             _census(MIN_REFUSALS + 5, {"A-USDC": 20}))
        assert len(proposals) == 1
        assert proposals[0]["testable"] is False
        assert "too few" in proposals[0]["statement"]

    def test_a_guard_on_underperforming_symbols_looks_justified(self):
        proposals = generate(
            _rows({"BAD-USDC": [-0.05] * (MIN_COMPARISON + 4),
                   "GOOD-USDC": [0.03] * (MIN_COMPARISON + 4)}),
            _census(MIN_REFUSALS + 10, {"BAD-USDC": 30}))
        assert proposals[0]["testable"] is True
        assert "justified" in proposals[0]["suspicion"]
        evidence = proposals[0]["evidence"]
        assert evidence["mean_return_on_guarded"] < evidence["mean_return_elsewhere"]

    def test_a_guard_on_outperforming_symbols_is_flagged(self):
        """The finding worth surfacing loudly: a guard costing money.

        A rule that fires constantly on symbols which then outperform is
        refusing profitable trades, and nothing else in this package would
        notice -- the refused trades are absent from every other sample.
        """
        proposals = generate(
            _rows({"GOOD-USDC": [0.05] * (MIN_COMPARISON + 4),
                   "MEH-USDC": [-0.01] * (MIN_COMPARISON + 4)}),
            _census(MIN_REFUSALS + 10, {"GOOD-USDC": 30}))
        assert proposals[0]["testable"] is True
        # Assert on the BEHAVIOUR, not on the wording. An earlier version of
        # this test looked for the word "costing", which appears only in the
        # source comment -- so it failed while the code was right.
        assert "refusing profitable" in proposals[0]["suspicion"]
        evidence = proposals[0]["evidence"]
        assert evidence["mean_return_on_guarded"] > evidence["mean_return_elsewhere"]

    def test_the_evidence_that_provoked_the_question_travels_with_it(self):
        """A reader must be able to see WHY the question was asked."""
        proposals = generate(
            _rows({"A-USDC": [-0.05] * (MIN_COMPARISON + 4),
                   "B-USDC": [0.03] * (MIN_COMPARISON + 4)}),
            _census(MIN_REFUSALS + 10, {"A-USDC": 30}))
        evidence = proposals[0]["evidence"]
        assert evidence["count"] > 0
        assert evidence["n_guarded"] > 0
        assert evidence["n_elsewhere"] > 0


class ConfoundingTests(SimpleTestCase):
    """Some guards fire BECAUSE a symbol is doing well.

    The correlation is real and the causal reading is backwards, and a
    generator that cannot tell those apart will eventually talk someone into
    deleting a guard that was working.
    """

    def test_a_selection_confounded_guard_gets_no_verdict(self):
        """entry-refused-live-held nearly cost us a working guard.

        It was flagged as "guarding symbols that perform BETTER" (+0.03817
        against +0.02149) and was one step from being acted on. But it fires
        only when a LIVE position is already open, and live positions exist on
        our better symbols because those are the ones that graduated. Removing
        it would let the bot double-buy a token it already owns.
        """
        proposals = generate(
            _rows({"GOOD-USDC": [0.05] * (MIN_COMPARISON + 4),
                   "MEH-USDC": [-0.01] * (MIN_COMPARISON + 4)}),
            _census(MIN_REFUSALS + 10, {"GOOD-USDC": 30},
                    key="entry-refused-live-held"))
        assert len(proposals) == 1
        assert proposals[0]["confounded"] is True
        assert proposals[0]["testable"] is False
        # Case-insensitive: the statement capitalises ALREADY for emphasis,
        # and a test that pins exact casing is testing prose rather than
        # behaviour. That mistake has now been made twice in this file.
        assert "already being traded" in proposals[0]["statement"].lower()

    def test_the_numbers_are_still_reported(self):
        """Refuse the verdict, not the evidence.

        A reader should still see what provoked the question -- the finding
        is real even though the conclusion does not follow.
        """
        proposals = generate(
            _rows({"GOOD-USDC": [0.05] * (MIN_COMPARISON + 4),
                   "MEH-USDC": [-0.01] * (MIN_COMPARISON + 4)}),
            _census(MIN_REFUSALS + 10, {"GOOD-USDC": 30},
                    key="entry-refused-duplicate"))
        evidence = proposals[0]["evidence"]
        assert evidence["mean_return_on_guarded"] > evidence["mean_return_elsewhere"]

    def test_an_unconfounded_guard_is_still_judged(self):
        """The exemption must not become a blanket amnesty."""
        proposals = generate(
            _rows({"GOOD-USDC": [0.05] * (MIN_COMPARISON + 4),
                   "MEH-USDC": [-0.01] * (MIN_COMPARISON + 4)}),
            _census(MIN_REFUSALS + 10, {"GOOD-USDC": 30},
                    key="guard-blocked-live"))
        assert proposals[0]["testable"] is True
        assert "refusing profitable" in proposals[0]["suspicion"]


class TheoremWrappingTests(SimpleTestCase):
    def test_only_testable_proposals_become_theorems(self):
        """Untestable ones are worth REPORTING and not worth testing."""
        proposals = [
            {"id": "a", "statement": "x", "testable": True,
             "symbols": ["A-USDC"], "suspicion": ""},
            {"id": "b", "statement": "y", "testable": False},
        ]
        theorems = as_theorems(proposals)
        assert len(theorems) == 1
        assert theorems[0].name == "a"

    def test_the_predicate_selects_the_guarded_symbols(self):
        theorems = as_theorems([{"id": "a", "statement": "x", "testable": True,
                                 "symbols": ["A-USDC"], "suspicion": ""}])
        predicate = theorems[0].predicate
        assert predicate({"symbol": "A-USDC"}) is True
        assert predicate({"symbol": "B-USDC"}) is False

    def test_a_proposal_with_no_symbols_is_dropped(self):
        assert as_theorems([{"id": "a", "statement": "x", "testable": True,
                             "symbols": []}]) == []
