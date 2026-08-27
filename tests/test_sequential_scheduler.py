"""Production work runs one task at a time, paced by real machine headroom.

Replaces a loop that fired its whole task list per cycle: one slow task
(news_enrichment at its 120s timeout) pushed the backlog past the governor cap,
and the orchestrator SKIPPED the entire cycle -- trading included. Observed
2026-08-27: no ghost entry landed for 64 minutes while ticks kept flowing.

So: sequential ACROSS categories (a stalled feed delays the feed, not trading),
prioritised by what the money path needs, and polite enough that the machine
stays usable.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

from services.sequential_scheduler import (
    CATEGORY_ORDER,
    CATEGORY_PRIORITY,
    Pressure,
    SequentialScheduler,
    Task,
)


def _rec(log, name, delay=0.0):
    def _fn():
        if delay:
            time.sleep(delay)
        log.append(name)
    return _fn


class PriorityTest(unittest.TestCase):
    def test_trade_outranks_everything(self):
        self.assertLess(CATEGORY_PRIORITY["trade"], CATEGORY_PRIORITY["feed"])
        self.assertLess(CATEGORY_PRIORITY["feed"], CATEGORY_PRIORITY["model"])
        self.assertLess(CATEGORY_PRIORITY["model"], CATEGORY_PRIORITY["housekeeping"])

    def test_first_pass_runs_trade_before_housekeeping(self):
        log = []
        s = SequentialScheduler()
        s.add(Task("telemetry", _rec(log, "telemetry"), category="housekeeping"))
        s.add(Task("trade", _rec(log, "trade"), category="trade"))
        with mock.patch("services.sequential_scheduler.read_pressure",
                        return_value=Pressure(cpu=10.0, free_mb=8000.0, mem_pct=40.0)):
            s.run_once()
        self.assertEqual(log[0], "trade")


class SequentialAcrossAreasTest(unittest.TestCase):
    def test_every_category_runs_in_one_pass(self):
        """The point of 'sequential': breadth across areas, not depth into one."""
        log = []
        s = SequentialScheduler()
        for cat in CATEGORY_ORDER:
            s.add(Task("%s_a" % cat, _rec(log, "%s_a" % cat), category=cat))
            s.add(Task("%s_b" % cat, _rec(log, "%s_b" % cat), category=cat))
        with mock.patch("services.sequential_scheduler.read_pressure",
                        return_value=Pressure(cpu=10.0, free_mb=8000.0, mem_pct=40.0)):
            s.run_once()
        for cat in CATEGORY_ORDER:
            self.assertIn("%s_a" % cat, log, "%s never ran" % cat)

    def test_category_start_rotates_between_cycles(self):
        s = SequentialScheduler()
        for cat in CATEGORY_ORDER:
            s.add(Task(cat, lambda: None, category=cat))
        first = [t.category for t in s._ordered(time.time())]
        s.cycles += 1
        second = [t.category for t in s._ordered(time.time())]
        self.assertNotEqual(first, second)


class PressureTest(unittest.TestCase):
    def test_high_memory_pauses_new_work(self):
        s = SequentialScheduler(mem_pause_pct=90.0)
        self.assertTrue(s.should_pause(Pressure(cpu=10.0, free_mb=200.0, mem_pct=95.0)))

    def test_high_cpu_pauses_new_work(self):
        s = SequentialScheduler(cpu_pause_pct=85.0)
        self.assertTrue(s.should_pause(Pressure(cpu=95.0, free_mb=8000.0, mem_pct=40.0)))

    def test_healthy_machine_does_not_pause(self):
        s = SequentialScheduler()
        self.assertFalse(s.should_pause(Pressure(cpu=20.0, free_mb=8000.0, mem_pct=45.0)))

    def test_unknown_pressure_does_not_pause(self):
        """Unreadable metrics must never silently stop trading."""
        s = SequentialScheduler()
        self.assertFalse(s.should_pause(Pressure()))

    def test_critical_task_runs_under_pressure(self):
        """Trading must not be deferred because the machine is busy."""
        log = []
        s = SequentialScheduler(mem_pause_pct=50.0)
        s.add(Task("trade", _rec(log, "trade"), category="trade", critical=True))
        s.add(Task("train", _rec(log, "train"), category="model"))
        with mock.patch("services.sequential_scheduler.read_pressure",
                        return_value=Pressure(cpu=99.0, free_mb=100.0, mem_pct=99.0)):
            summary = s.run_once()
        self.assertIn("trade", log)
        self.assertNotIn("train", log)
        self.assertIn("train", summary["deferred"])

    def test_heavy_task_defers_on_low_memory(self):
        log = []
        s = SequentialScheduler()
        s.add(Task("heavy", _rec(log, "heavy"), category="model", min_free_mb=4000.0))
        with mock.patch("services.sequential_scheduler.read_pressure",
                        return_value=Pressure(cpu=10.0, free_mb=500.0, mem_pct=60.0)):
            s.run_once()
        self.assertEqual(log, [])


class IsolationTest(unittest.TestCase):
    def test_one_slow_task_does_not_stop_the_rest(self):
        """The original failure: a stalled task must not skip the cycle."""
        log = []
        s = SequentialScheduler()
        s.add(Task("news", _rec(log, "news", delay=2.0), category="feed", timeout_sec=1.0))
        s.add(Task("trade", _rec(log, "trade"), category="trade"))
        with mock.patch("services.sequential_scheduler.read_pressure",
                        return_value=Pressure(cpu=10.0, free_mb=8000.0, mem_pct=40.0)):
            s.run_once()
        self.assertIn("trade", log)

    def test_failing_task_is_recorded_not_raised(self):
        def boom():
            raise ValueError("nope")
        s = SequentialScheduler()
        s.add(Task("bad", boom, category="model"))
        with mock.patch("services.sequential_scheduler.read_pressure",
                        return_value=Pressure(cpu=10.0, free_mb=8000.0, mem_pct=40.0)):
            s.run_once()
        task = s._tasks[0]
        self.assertFalse(task.last_ok)
        self.assertIn("ValueError", task.last_error)

    def test_interval_prevents_rerun(self):
        log = []
        s = SequentialScheduler()
        s.add(Task("x", _rec(log, "x"), category="feed", interval_sec=3600.0))
        with mock.patch("services.sequential_scheduler.read_pressure",
                        return_value=Pressure(cpu=10.0, free_mb=8000.0, mem_pct=40.0)):
            s.run_once()
            s.run_once()
        self.assertEqual(len(log), 1)


class StatusTest(unittest.TestCase):
    def test_never_run_task_reports_none_not_zero(self):
        """'--' in the UI: absent is not the same as zero."""
        s = SequentialScheduler()
        s.add(Task("x", lambda: None, category="feed"))
        row = s.status()["tasks"][0]
        self.assertIsNone(row["last_run"])
        self.assertIsNone(row["avg_sec"])
        self.assertIsNone(row["last_ok"])

    def test_status_reports_pressure_and_thresholds(self):
        s = SequentialScheduler()
        st = s.status()
        self.assertIn("pressure", st)
        self.assertIn("cpu_pause_pct", st["thresholds"])


if __name__ == "__main__":
    unittest.main()
