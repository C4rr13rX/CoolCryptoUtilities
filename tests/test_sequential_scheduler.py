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

import threading
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


class OverrunTest(unittest.TestCase):
    """A timed-out task is abandoned, not killed. It must not be re-spawned.

    Observed 2026-09-02: data_ingest kept exceeding its 90s timeout, and every
    following cycle started ANOTHER copy on top of the one still running. 77
    live "seq-data_ingest" threads accumulated, all queued on the single
    database lock, and the trading cycle starved behind them -- 2 ticks and 1
    cycle in ten minutes, no ghost trade for three hours.
    """

    def _blocked_task(self, release):
        def _fn():
            release.wait(timeout=30)
        # critical=True keeps the test independent of the ambient pressure on
        # whatever machine runs it, and pins that the guard covers the trade
        # path too -- a critical task must never be double-started either.
        return Task("slow", _fn, category="trade", timeout_sec=1.0, critical=True)

    def test_timed_out_task_is_not_started_again_while_still_running(self):
        release = threading.Event()
        task = self._blocked_task(release)
        s = SequentialScheduler()
        s.add(task)
        try:
            first = s.run_once()
            self.assertEqual(first["ran"], ["slow"])
            self.assertTrue(task.is_running(), "worker should have outlived the join")

            # Several more passes while the abandoned worker is still alive.
            for _ in range(5):
                summary = s.run_once()
                self.assertEqual(summary["ran"], [], "must not stack a second copy")
                self.assertEqual(summary["deferred"], ["slow"])

            self.assertEqual(task.runs, 1, "exactly one worker was ever started")
            self.assertEqual(task.overruns, 5)
            self.assertEqual(s.overrun_total, 5)
        finally:
            release.set()

    def test_task_runs_again_once_the_previous_worker_finishes(self):
        release = threading.Event()
        task = self._blocked_task(release)
        s = SequentialScheduler()
        s.add(task)
        try:
            s.run_once()
            self.assertEqual(s.run_once()["ran"], [], "still running -> skipped")
        finally:
            release.set()
        task.thread.join(timeout=5)
        self.assertFalse(task.is_running())
        # The guard is a skip, not a permanent disable.
        self.assertEqual(s.run_once()["ran"], ["slow"])
        self.assertEqual(task.runs, 2)

    def test_overrun_is_reported_so_the_stall_is_visible(self):
        events = []
        release = threading.Event()
        task = self._blocked_task(release)
        s = SequentialScheduler(on_event=lambda name, payload: events.append((name, payload)))
        s.add(task)
        try:
            s.run_once()
            s.run_once()
        finally:
            release.set()
        names = [name for name, _ in events]
        self.assertIn("task_timeout", names)
        self.assertIn("task_overrun", names)
        payload = dict(events[names.index("task_overrun")][1])
        self.assertEqual(payload["task"], "slow")
        self.assertIn("running_for", payload)

    def test_a_fast_task_is_never_treated_as_an_overrun(self):
        log = []
        s = SequentialScheduler()
        s.add(Task("quick", _rec(log, "quick"), category="trade", critical=True))
        for _ in range(4):
            self.assertEqual(s.run_once()["ran"], ["quick"])
        self.assertEqual(len(log), 4)
        self.assertEqual(s.overrun_total, 0)


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


class ModelTaskMustNotHoldTheQueueTest(unittest.TestCase):
    """A model task is not allowed to make the FEED wait for it.

    Measured 2026-09-11 over six hours of logs/system.log with
    scripts/seq_queue_budget.py: dataset_warmup timed out 28 times and held the
    scheduler thread 93.36 seconds per ten minutes -- 15.6% of the whole queue
    -- because `_execute` joined it on the scheduler's own thread. The join
    bought nothing: the worker is a daemon thread that is ABANDONED on timeout,
    never killed, so waiting 120 seconds only decided when the scheduler would
    admit that. Every second of it was a second data_ingest, sitting behind it
    in the same SequentialScheduler, did not get.

    These fail against the pre-detach scheduler: run_once() blocked for the
    slow model task's full duration and the feed task ran only after it.
    """

    def test_a_slow_model_task_does_not_delay_the_feed_task_behind_it(self):
        order = []
        started = threading.Event()

        def slow_model():
            started.set()
            order.append("model_started")
            time.sleep(3.0)
            order.append("model_finished")

        def feed():
            order.append("feed_ran")

        s = SequentialScheduler()
        # category order puts model AFTER feed, so force model first by cycling
        # the rotation to the pass where it leads.
        s.add(Task("m", slow_model, category="model", timeout_sec=3.0, detach=True))
        s.add(Task("f", feed, category="feed"))

        began = time.time()
        summary = s.run_once()
        elapsed = time.time() - began

        self.assertTrue(started.wait(2.0), "the detached model worker never started")
        # The scheduler must NOT have waited for the 3s worker.
        self.assertLess(
            elapsed, 1.5,
            "run_once took %.2fs -- the scheduler is still joining the model task, "
            "which is exactly the 93.4s/10min the feed was paying" % elapsed,
        )
        self.assertIn("f", summary["ran"])
        self.assertIn("feed_ran", order)
        self.assertNotIn(
            "model_finished", order,
            "the cycle only ended after the model worker finished",
        )

    def test_a_detached_task_is_still_never_started_twice(self):
        """Detaching must not become a way to stack copies of a slow task.

        The overrun guard is the whole reason detaching is safe: an abandoned
        worker already blocks its own restart, so removing the join changes
        WHEN it is abandoned, not WHETHER a second copy can start.
        """
        runs = []

        def slow():
            runs.append(1)
            time.sleep(2.0)

        s = SequentialScheduler()
        s.add(Task("m", slow, category="model", timeout_sec=2.0, detach=True))
        s.run_once()
        s.run_once()
        s.run_once()
        self.assertEqual(len(runs), 1, "a second copy was started on top of the first")
        self.assertGreaterEqual(s.overrun_total, 1)

    def test_a_detached_failure_is_recorded_and_not_silent(self):
        """Nobody is joining, so the worker has to record its own outcome."""
        def boom():
            raise RuntimeError("model blew up")

        s = SequentialScheduler()
        t = Task("m", boom, category="model", detach=True)
        s.add(t)
        s.run_once()
        for _ in range(50):
            if t.last_ok is False:
                break
            time.sleep(0.05)
        self.assertIs(t.last_ok, False)
        self.assertIn("model blew up", t.last_error)
        self.assertEqual(t.failures, 1)

    def test_held_seconds_are_attributed_to_the_category_that_blocked(self):
        """The measurement the fix is judged by, live rather than from a log.

        The log can only ever show a FLOOR -- a task that finishes inside its
        timeout logs nothing -- so the scheduler counts the seconds itself.
        """
        s = SequentialScheduler()
        s.add(Task("joined", lambda: time.sleep(0.4), category="housekeeping"))
        s.add(Task("detached", lambda: time.sleep(0.4), category="model", detach=True))
        s.run_once()
        held = s.status()["held_sec_per_10min_by_category"]
        self.assertGreater(held["housekeeping"], held["model"],
                           "the detached task is still charging the queue")

    def test_production_registers_both_model_tasks_detached(self):
        """The fix has to be wired, not just available.

        Asserted against the registration itself rather than a comment: this
        repo has shipped a test that passed on a word appearing only in a
        comment.
        """
        import inspect
        import production

        src = inspect.getsource(production.ProductionManager._build_sequential_scheduler)
        for name in ("dataset_warmup", "candidate_training"):
            idx = src.index('Task("%s"' % name)
            # the Task(...) call ends at the next Task( or the closing bracket
            nxt = src.find('Task("', idx + 6)
            chunk = src[idx: nxt if nxt != -1 else len(src)]
            self.assertIn('category="model"', chunk)
            self.assertIn("detach=True", chunk,
                          "%s is a model task still joined on the scheduler thread" % name)


if __name__ == "__main__":
    unittest.main()
