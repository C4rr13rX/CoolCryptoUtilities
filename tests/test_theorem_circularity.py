"""A theorem may not split trades by the outcome it is judged on.

`loss_capped` did exactly that. Its condition was `return > -0.013` while the
quantity being compared IS the return, so it read "trades that did not lose
more than 1.3% outperform trades that did" -- true by construction. It
reported SUPPORTED at p=0.0000 with Cohen's d=1.07 on 289 held-out trades:
the strongest result this module has ever produced, ten times larger than
anything else in the report, and worth nothing. It was also unactionable,
since the condition can only be evaluated after the trade is over.

That is the worst shape a bug can take here. It does not fail, it does not
look wrong, and it outranks every honest finding in the same table.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

from tradingagent.theorems import (  # noqa: E402
    Theorem, _is_circular, _OUTCOME_FIELDS, candidate_theorems,
)


class CircularityGuard(unittest.TestCase):

    def test_the_exact_defect_that_shipped_is_caught(self):
        t = Theorem("loss_capped", "...",
                    lambda tr: float(tr.get("return") or 0.0) > -0.013)
        self.assertTrue(_is_circular(t))

    def test_a_condition_on_a_pre_trade_fact_is_allowed(self):
        """hold_sec and volatility_pct are known before the outcome, so
        splitting on them is a real claim, not a tautology."""
        for field in ("hold_sec", "volatility_pct", "clip_usd"):
            t = Theorem("ok", "...",
                        lambda tr, f=field: float(tr.get(f) or 0.0) <= 1.0)
            self.assertFalse(_is_circular(t), field)

    def test_every_outcome_alias_is_caught_not_just_return(self):
        """A renamed field is the same defect. Caught by reading the
        closure's bytecode rather than the theorem's name."""
        for field in _OUTCOME_FIELDS:
            t = Theorem("x", "...",
                        lambda tr, f=field: float(tr.get(f) or 0.0) > 0)
            self.assertTrue(_is_circular(t), field)

    def test_a_predicate_with_no_code_object_is_not_a_crash(self):
        class Weird:
            pass
        t = Theorem("x", "...", Weird())
        self.assertFalse(_is_circular(t))

    def test_the_guard_actually_reads_the_predicate_field(self):
        """The first version read `theorem.condition`, which does not exist,
        so getattr returned None and the guard returned False for
        EVERYTHING. It would have shipped looking like it worked. This
        fails if the attribute name drifts again."""
        t = Theorem("loss_capped", "...",
                    lambda tr: float(tr.get("return") or 0.0) > -0.013)
        self.assertTrue(hasattr(t, "predicate"))
        self.assertTrue(_is_circular(t))


class GeneratedCandidates(unittest.TestCase):

    def _trades(self, n=200):
        return [{"return": 0.01 * (i % 7 - 3), "hold_sec": 100 + i,
                 "clip_usd": 1.0 + (i % 5)} for i in range(n)]

    def test_no_generated_candidate_is_circular(self):
        for t in candidate_theorems(self._trades()):
            self.assertFalse(_is_circular(t), t.name)

    def test_loss_capped_is_gone(self):
        names = [t.name for t in candidate_theorems(self._trades())]
        self.assertNotIn("loss_capped", names)

    def test_the_honest_candidates_survive(self):
        """The guard must not empty the list -- a filter that rejects
        everything is indistinguishable from a broken generator."""
        names = [t.name for t in candidate_theorems(self._trades())]
        self.assertIn("hold_below_median", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
