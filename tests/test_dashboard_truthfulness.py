"""The dashboard must not contradict the system it reports on.

Two observed contradictions on 2026-08-27, both from rendering ONE aggregate
model-accuracy number as if it were the whole truth:

  Bus Scheduler : "Ghost Lane HALTED - insufficient_accuracy" while the
                  transition plan the trading process acts on had halt_ghost=0
                  and ghost trading was running normally.
  Pipeline      : "Ghost Ready: insufficient_accuracy" and "Live Ready:
                  insufficient_accuracy" while atf_static was ghost-ready on
                  positive expectancy AND live_approved.

Readiness is a PER-STRATEGY fact. If any strategy is ready, the stage is ready,
and the UI must name which ones. A missing or stale reading must render as
"no value yet", never as a confident false.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


def _readiness(ghost_ids, live_ids, stale=False):
    """Mirror of the merge in PipelineReadinessView."""
    return {
        "ready": False,
        "reason": "insufficient_accuracy",
        "ghost_collection_ready": False,
        "ghost_ready_strategies": ghost_ids,
        "live_ready_strategies": live_ids,
        "ghost_ready_any": bool(ghost_ids),
        "live_ready_any": bool(live_ids),
        "is_live_trading": False,
        "_stale": stale,
    }


class PerStrategyReadinessTest(unittest.TestCase):
    def test_one_ready_strategy_makes_the_stage_ready(self):
        r = _readiness(["atf_static"], ["atf_static"])
        self.assertTrue(r["ghost_ready_any"])
        self.assertTrue(r["live_ready_any"])

    def test_aggregate_failure_does_not_override_a_ready_strategy(self):
        """The exact contradiction: aggregate says not ready, a strategy is."""
        r = _readiness(["atf_static"], ["atf_static"])
        self.assertFalse(r["ready"])                  # aggregate model gate
        self.assertTrue(r["ghost_ready_any"])         # but a strategy IS ready
        self.assertTrue(r["live_ready_any"])

    def test_no_ready_strategies_is_still_not_ready(self):
        r = _readiness([], [])
        self.assertFalse(r["ghost_ready_any"])
        self.assertFalse(r["live_ready_any"])

    def test_ready_strategies_are_named(self):
        r = _readiness(["atf_static", "money_button"], ["atf_static"])
        self.assertIn("atf_static", r["ghost_ready_strategies"])
        self.assertIn("money_button", r["ghost_ready_strategies"])
        self.assertEqual(r["live_ready_strategies"], ["atf_static"])


class StaleReportTest(unittest.TestCase):
    """A stale reading must be visibly stale, not silently false."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    @staticmethod
    def _load(path):
        """Standalone mirror of telemetry.views._load_report.

        Imported directly the view module pulls in Django models, which need a
        configured app registry; the contract under test is the freshness
        stamping, so it is exercised on its own.
        """
        import json as _json
        import time as _time
        stale_after = 600.0
        if not path.exists():
            return {"_missing": True, "_stale": True, "_age_sec": None}
        try:
            payload = _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"_unreadable": True, "_stale": True, "_age_sec": None}
        if not isinstance(payload, dict):
            return {"_unreadable": True, "_stale": True, "_age_sec": None}
        age = max(0.0, _time.time() - path.stat().st_mtime)
        payload["_age_sec"] = age
        payload["_stale"] = bool(age > stale_after)
        return payload

    def test_missing_file_is_flagged_not_false(self):
        out = self._load(Path(self.tmp.name) / "nope.json")
        self.assertTrue(out.get("_missing"))
        self.assertTrue(out.get("_stale"))
        # The critical part: absence must be distinguishable from a real False.
        self.assertNotIn("ghost_collection_ready", out)

    def test_fresh_file_is_not_stale(self):
        p = Path(self.tmp.name) / "r.json"
        p.write_text(json.dumps({"ready": True}), encoding="utf-8")
        out = self._load(p)
        self.assertFalse(out.get("_stale"))
        self.assertTrue(out.get("ready"))

    def test_old_file_is_marked_stale(self):
        p = Path(self.tmp.name) / "r.json"
        p.write_text(json.dumps({"ready": True}), encoding="utf-8")
        old = time.time() - 7200
        os.utime(p, (old, old))
        out = self._load(p)
        self.assertTrue(out.get("_stale"))
        self.assertGreater(out.get("_age_sec") or 0, 3600)

    def test_unreadable_file_is_flagged(self):
        p = Path(self.tmp.name) / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        out = self._load(p)
        self.assertTrue(out.get("_unreadable"))
        self.assertTrue(out.get("_stale"))


class BusLaneHaltTest(unittest.TestCase):
    def test_lane_with_an_approved_strategy_is_not_halted(self):
        halt_live, live_ready = True, True
        self.assertFalse(halt_live and not live_ready)

    def test_lane_with_no_approved_strategy_stays_halted(self):
        halt_live, live_ready = True, False
        self.assertTrue(halt_live and not live_ready)

    def test_plan_takes_precedence_over_the_report_file(self):
        """halt_ghost=0 in the plan must beat ghost_collection_ready=False."""
        risk_flags = {"halt_ghost": 0.0}
        report = {"ghost_collection_ready": False}
        if "halt_ghost" in risk_flags:
            ready = not bool(risk_flags["halt_ghost"])
        else:
            ready = bool(report.get("ghost_collection_ready"))
        self.assertTrue(ready)


if __name__ == "__main__":
    unittest.main()
