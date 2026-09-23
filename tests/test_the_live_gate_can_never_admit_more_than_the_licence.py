"""The order path may refuse what the licence allows. It may NEVER do the reverse.

``trading/bot.py:_strategy_live_approved`` is the single seam behind every
``bot.py`` path that spends real money -- ``entry_will_be_simulated``,
``entry_spends_real_money``, ``entry_is_live`` and the ``live_approved``
conjunction all read it. Commit ``b367603`` changed which question it asks:
``StrategyLedger.is_live_approved`` (does this rule HOLD a licence) became
``StrategyLedger.may_place_live_order`` (licence AND ``live_order_refusal``,
which refuses a sid whose own RECORDED book lost money).

THE INVARIANT THIS PINS, and it is the one the lead asked for by name: does
that change make the set of strategies allowed to spend real money LARGER,
SMALLER, or IDENTICAL? It must never be larger, for any ledger state whatever
-- not "it is not larger on today's file". Measured over the live ledger on
2026-09-23 the two sets were identical (``['money_button']`` both ways), which
is a fact about one file and would stop meaning anything the moment the file
moved. The property below is a fact about the code.

A one-line change to this function and a one-line HOLE in the live-money gate
look identical in a diff. This is the test that tells them apart, and it is
written so that it keeps telling them apart after the function is edited again.
"""

from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace

from trading.strategies.ledger import StrategyLedger


class _FakeLedger:
    """Answers the two ledger questions from a script, not from a file.

    Every combination is exercised, including the one that cannot arise from
    the real conjunction, because the guarantee has to hold against a FUTURE
    edit to ``may_place_live_order`` -- not merely against today's.
    """

    def __init__(self, licensed: bool, refusal):
        self._licensed = licensed
        self._refusal = refusal

    def is_live_approved(self, sid: str) -> bool:
        return self._licensed

    def live_order_refusal(self, sid: str):
        return self._refusal

    def may_place_live_order(self, sid: str) -> bool:
        return StrategyLedger.may_place_live_order(self, sid)


def _order_path_admits(ledger, sid: str) -> bool:
    """Drive the real function, not a copy of its logic.

    A helper that re-implemented the conjunction would pass this test while the
    production seam did something else -- which is the failure mode the
    function's own six regression tests exist for.
    """
    from trading.bot import TradingBot

    return TradingBot._strategy_live_approved(
        SimpleNamespace(strategy_ledger=ledger),
        SimpleNamespace(strategy_id=sid),
    )


class TheOrderPathIsASubsetOfTheLicence(unittest.TestCase):
    def setUp(self):
        # STRATEGY_GRADUATION_ENFORCED defaults to "1"; the enforced branch is
        # the only one that consults the ledger at all, and it is the branch
        # that spends money. The unenforced and unarmed legacy branches are
        # pinned by the function's own existing tests and are not this
        # invariant's subject.
        import os

        self._restore = os.environ.get("STRATEGY_GRADUATION_ENFORCED")
        os.environ["STRATEGY_GRADUATION_ENFORCED"] = "1"

    def tearDown(self):
        import os

        if self._restore is None:
            os.environ.pop("STRATEGY_GRADUATION_ENFORCED", None)
        else:
            os.environ["STRATEGY_GRADUATION_ENFORCED"] = self._restore

    def test_no_ledger_state_lets_the_order_path_outrun_the_licence(self):
        """Across every (licence, book) combination: admitted implies licensed."""
        for licensed, refusal in itertools.product(
            (True, False),
            (None, "live book lost -0.0161", "ghost book lost -2.0534"),
        ):
            ledger = _FakeLedger(licensed, refusal)
            admitted = _order_path_admits(ledger, "some_strategy")
            self.assertFalse(
                admitted and not licensed,
                "the order path admitted a strategy the licence check refuses "
                f"(licensed={licensed!r}, refusal={refusal!r}) -- that is a "
                "hole in the live-money gate, not a tightening",
            )

    def test_a_losing_book_is_refused_even_with_a_licence(self):
        """The half of the question that was not being asked before b367603."""
        self.assertFalse(
            _order_path_admits(_FakeLedger(True, "live book lost -0.0161"), "s"),
            "a licence is necessary, not sufficient -- a rule whose own "
            "recorded book lost money must not place a live order",
        )

    def test_a_licensed_and_paying_strategy_is_still_admitted(self):
        """The gate must not have become unsatisfiable, which is the other
        way a safety change goes wrong in this repo."""
        self.assertTrue(_order_path_admits(_FakeLedger(True, None), "s"))


class TheLedgerConjunctionIsStillAConjunction(unittest.TestCase):
    """``may_place_live_order`` is what makes the subset property structural.

    If a later edit made it answer the book question FIRST, or answer it
    instead of the licence question, the class above would still pass on the
    fake while production widened. This asserts the shape that makes the
    guarantee hold.
    """

    def test_the_licence_is_the_first_term_and_short_circuits(self):
        seen = []

        class _Recording(_FakeLedger):
            def is_live_approved(self, sid):
                seen.append("licence")
                return False

            def live_order_refusal(self, sid):
                seen.append("book")
                return None

        self.assertFalse(_Recording(False, None).may_place_live_order("s"))
        self.assertEqual(
            seen, ["licence"],
            "an unlicensed strategy must be refused WITHOUT consulting its "
            "book -- otherwise the book could overturn the licence",
        )


if __name__ == "__main__":
    unittest.main()
