"""Step 6 must not refuse a strategy on the bar step 5 just licensed it on.

Step 5 grants the live licence in ``trading/strategies/ledger.py``'s
``_evaluate_graduation_locked``::

    min_trades = _env_int("STRATEGY_GRADUATION_MIN_TRADES", 20)

Step 6 judges EXACTLY the strategies step 5 licensed -- its candidate list is
``StrategyLedger().approved_ids()`` at ``trading/pipeline.py`` -- and its
sample-size bar was an independent ``MIN_GHOST_TRADES_FOR_PROMOTION``,
default 25.

20 through 24 tradeable ghost trades is therefore a window in which a strategy
is LICENSED by step 5 and GUARANTEED to be refused by step 6, for the same
reason step 5 just cleared it on. A licence void on arrival.

The drift is visible in what did not drift: the win rate is 0.55 on both sides
(``STRATEGY_GRADUATION_MIN_WINRATE`` / ``MIN_GHOST_WIN_RATE``), deliberately
identical. Only the count disagreed.

THIS DOES NOT OPEN STEP 6. The live ladder refuses ``money_button`` at 0 ghost
trades against either bar, so no strategy clears anything because of this
change. What it removes is a window that can only ever produce a dead licence.
``atf_static``, the nearest strategy to the bar, lands on exactly 20.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import trading.pipeline as pipeline_module  # noqa: E402
from trading.strategies.ledger import _env_int  # noqa: E402

#: What step 6's bar fell back to before it was made to read step 5's. Named
#: here rather than imported so this file asserts against the OLD behaviour
#: instead of failing to import under it -- a collection error proves nothing
#: about a threshold.
PRE_FIX_STEP_SIX_DEFAULT = "25"

#: The env names step 6 reads, cleared so a developer's shell cannot decide
#: what this test measures.
STEP_SIX_ENV = ("MIN_GHOST_TRADES_FOR_PROMOTION", "MIN_GHOST_TRADES_OVERRIDE")


def _step_six_min_trades() -> int:
    """What ``TrainingPipeline.__init__`` would set ``min_ghost_trades`` to.

    Reproduces the one expression rather than constructing a pipeline, which
    needs a database, a feed and a model on disk.
    """
    derive = getattr(pipeline_module, "_graduation_min_trades", None)
    fallback = str(derive()) if derive is not None else PRE_FIX_STEP_SIX_DEFAULT
    return int(
        os.getenv(
            "MIN_GHOST_TRADES_FOR_PROMOTION",
            os.getenv("MIN_GHOST_TRADES_OVERRIDE", fallback),
        )
    )


def _step_five_min_trades() -> int:
    return int(_env_int("STRATEGY_GRADUATION_MIN_TRADES", 20))


class TheTwoStagesReadOneBarTest(unittest.TestCase):
    def _clear(self, **extra: str):
        env = dict(os.environ)
        for name in STEP_SIX_ENV:
            env.pop(name, None)
        env.pop("STRATEGY_GRADUATION_MIN_TRADES", None)
        env.update(extra)
        return mock.patch.dict(os.environ, env, clear=True)

    def test_the_default_bars_are_the_same_number(self):
        """BEFORE: step 5 licensed at 20 and step 6 refused under 25."""
        with self._clear():
            self.assertEqual(_step_six_min_trades(), _step_five_min_trades())

    def test_no_count_is_licensed_by_one_stage_and_refused_by_the_other(self):
        """The 20..24 window produced a licence that was void on arrival."""
        with self._clear():
            five = _step_five_min_trades()
            six = _step_six_min_trades()
        void = [n for n in range(five, max(five, six)) if n >= five and n < six]
        self.assertEqual(
            void,
            [],
            "these tradeable-trade counts graduate at step 5 and are refused "
            "by step 6 on the same criterion: %r" % (void,),
        )

    def test_raising_the_graduation_bar_raises_both_stages(self):
        """One knob, or they drift apart again the first time it is turned."""
        with self._clear(STRATEGY_GRADUATION_MIN_TRADES="40"):
            self.assertEqual(_step_five_min_trades(), 40)
            self.assertEqual(_step_six_min_trades(), 40)

    def test_an_explicit_step_six_override_still_wins(self):
        """The operator keeps the ability to make step 6 stricter on purpose."""
        with self._clear(MIN_GHOST_TRADES_FOR_PROMOTION="60"):
            self.assertEqual(_step_six_min_trades(), 60)
        with self._clear(MIN_GHOST_TRADES_OVERRIDE="55"):
            self.assertEqual(_step_six_min_trades(), 55)

    def test_the_bar_is_not_loosened_below_the_licence(self):
        """Unifying must never let step 6 pass a book step 5 would refuse.

        The direction matters: a step-6 bar BELOW step 5's would admit
        strategies that never earned a licence. This asserts the inequality,
        not the equality, so it stays true if a later pass decides step 6
        should be stricter again.
        """
        with self._clear():
            self.assertGreaterEqual(_step_six_min_trades(), _step_five_min_trades())


class TheBarIsReadFromTheSharedHelperTest(unittest.TestCase):
    """Close the weakness in the tests above: they REPRODUCE the expression.

    ``_step_six_min_trades`` rebuilds ``TrainingPipeline.__init__``'s getenv
    chain rather than calling it, because constructing a pipeline needs a
    database, a feed and a model on disk. That makes every assertion above
    true of the copy and silent about the original: revert pipeline.py's
    fallback to a hardcoded "25" and they all still pass.

    So read the real ``__init__`` and assert on what it does. Comment lines
    are stripped first -- a source scan that cannot tell a call from a
    sentence is worth less than no scan at all, and this repo spent a gate
    rejection on exactly that in the same pass this file was written.
    """

    def _init_source(self) -> str:
        import inspect

        from trading.pipeline import TrainingPipeline

        body = inspect.getsource(TrainingPipeline.__init__)
        return "\n".join(
            line for line in body.splitlines()
            if not line.lstrip().startswith("#")
        )

    def test_step_six_reads_the_shared_helper(self):
        src = self._init_source()
        self.assertIn(
            "_graduation_min_trades()", src,
            "min_ghost_trades must derive step 5's bar, not carry its own")

    def test_step_six_carries_no_hardcoded_fallback(self):
        """A literal here is a second answer to step 5's question.

        Matched against the whole statement rather than the file, so an
        unrelated numeric default elsewhere in __init__ cannot mask it.
        """
        import re

        src = self._init_source()
        match = re.search(
            r"self\.min_ghost_trades\s*=\s*(.*?)(?=\n\s*self\.)", src, re.S)
        self.assertIsNotNone(match, "min_ghost_trades is no longer assigned in __init__")
        statement = match.group(1)
        self.assertNotRegex(
            statement, r"[\"']\d+[\"']",
            "a hardcoded trade-count default here is step 5's question answered "
            "twice -- the 20..24 window this file exists to close: %r" % statement)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
