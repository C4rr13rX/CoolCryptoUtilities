"""A wedged download worker must not strand the feed task behind it.

Measured 2026-09-11 (pass 120, scripts/seq_queue_budget.py over logs/system.log)
and re-measured 2026-09-16 (pass 431, scripts/data_ingest_starvation_census.py):
`data_ingest` -- the sequential scheduler's FEED task -- produced 346
task_overrun events in six hours and its longest abandoned run was 6879 seconds.
An overrun means the PREVIOUS run is still alive and the scheduler declined to
start a second copy, so for those 114 minutes the task was not running at all.
The census named the culprits: the static:<chain> OHLCV download workers, 78
over-budget runs in 0.8 days totalling 17.9h, worst single call static:optimism
at 6443.9s.

Pass 120 made that stall ATTRIBUTABLE (`in_flight`, `wedged()`). It did not make
it STOP: run_cycle still called each worker synchronously and waited however long
the call took. These tests pin the deadline that makes the cycle return.

They fail against the unbounded run_cycle, which had no way to stop waiting.

What is NOT claimed here: that the wedged call is killed. It cannot be -- it is a
synchronous socket read inside someone else's library, and Python has no way to
interrupt one. The worker is run on a daemon thread and left behind. The thing
that makes that safe rather than a thread leak is the second test: while the
abandoned thread is alive its label is SKIPPED, so at most one thread per worker
is ever outstanding no matter how many cycles run.
"""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock


class _FakeWorker:
    def __init__(self, chain="base", fn=None):
        self.chain = chain
        self._fn = fn or (lambda: None)
        self.calls = 0

    def run_once(self):
        self.calls += 1
        self._fn()


def _bare_supervisor(static=(), dynamic=None, market=None, discovery=None):
    """A supervisor with its __init__ bypassed.

    The real __init__ reads pair-index files off disk and builds a discovery
    coordinator; none of that is what is under test here.
    """
    from services.background_workers import TokenDownloadSupervisor

    sup = TokenDownloadSupervisor.__new__(TokenDownloadSupervisor)
    sup.in_flight = None
    sup.last_worker_sec = {}
    sup.last_cycle_sec = 0.0
    sup.static_workers = list(static)
    sup.dynamic_worker = dynamic or _FakeWorker()
    sup.market_worker = market or _FakeWorker()
    sup.discovery_worker = discovery or _FakeWorker()
    return sup


class WedgedWorkerDoesNotStrandTheCycleTest(unittest.TestCase):
    def setUp(self):
        self._release = threading.Event()
        self.addCleanup(self._release.set)

    def _hang(self):
        """Blocks until the test releases it, like a socket read that never returns."""
        self._release.wait(30.0)

    def test_run_cycle_returns_on_its_deadline_instead_of_waiting_forever(self):
        """The 6879-second stall, bounded this time.

        Against the old run_cycle this blocks for the whole hang: there was no
        deadline, so the cycle waited exactly as long as the wedged call did.
        """
        from services import background_workers as bw

        sup = _bare_supervisor(static=[_FakeWorker(chain="optimism", fn=self._hang)])
        with mock.patch.object(sup, "WORKER_DEADLINE_SEC", 0.2), \
                mock.patch.object(bw, "log_message"):
            started = time.time()
            sup.run_cycle()
            elapsed = time.time() - started

        self.assertLess(
            elapsed, 5.0,
            "run_cycle waited on the wedged worker instead of returning on its deadline",
        )

    def test_the_workers_after_the_wedged_one_still_run(self):
        """This is the whole point: a wedged chain cost the market worker its turn.

        static:<chain> runs FIRST in run_cycle and `market` -- the worker that
        writes the prices table -- runs after it.
        """
        from services import background_workers as bw

        ran = []
        sup = _bare_supervisor(
            static=[_FakeWorker(chain="optimism", fn=self._hang)],
            dynamic=_FakeWorker(fn=lambda: ran.append("dynamic")),
            market=_FakeWorker(fn=lambda: ran.append("market")),
            discovery=_FakeWorker(fn=lambda: ran.append("discovery")),
        )
        with mock.patch.object(sup, "WORKER_DEADLINE_SEC", 0.2), \
                mock.patch.object(bw, "log_message"):
            sup.run_cycle()

        self.assertEqual(ran, ["dynamic", "market", "discovery"])

    def test_a_wedged_worker_is_not_started_a_second_time_while_it_is_alive(self):
        """The bound on the leak.

        Walking away from a thread is only safe if we do not start another one
        next cycle. A worker wedged against a source that is not answering would
        otherwise accumulate one live thread per cycle, forever.
        """
        from services import background_workers as bw

        worker = _FakeWorker(chain="optimism", fn=self._hang)
        sup = _bare_supervisor(static=[worker])
        with mock.patch.object(sup, "WORKER_DEADLINE_SEC", 0.2), \
                mock.patch.object(bw, "log_message"):
            sup.run_cycle()
            sup.run_cycle()
            sup.run_cycle()

        self.assertEqual(
            worker.calls, 1,
            "a wedged worker was re-entered while the previous call was still alive",
        )
        alive = [t for t in threading.enumerate() if t.name == "dl-static:optimism"]
        self.assertEqual(len(alive), 1, "more than one thread outstanding for one worker")

    def test_a_worker_past_its_deadline_is_still_named_after_the_cycle_moves_on(self):
        """Walking away must not lose the name.

        A worker the cycle gave up on is MORE interesting than one it is still
        inside, so wedged() keeps reporting it.
        """
        from services import background_workers as bw

        sup = _bare_supervisor(static=[_FakeWorker(chain="optimism", fn=self._hang)])
        with mock.patch.object(sup, "WORKER_DEADLINE_SEC", 0.2), \
                mock.patch.object(bw, "log_message"):
            sup.run_cycle()

        flight = sup.wedged()
        self.assertIsNotNone(flight, "the abandoned worker is not reported anywhere")
        self.assertEqual(flight["worker"], "static:optimism")
        self.assertTrue(flight["abandoned"])
        self.assertGreater(flight["running_for_sec"], 0.0)

    def test_the_worker_resumes_once_the_wedged_call_finally_returns(self):
        """A deadline is not a demotion: the skip lasts exactly as long as the hang.

        Without this, one slow cycle would retire a chain permanently.
        """
        from services import background_workers as bw

        worker = _FakeWorker(chain="optimism", fn=self._hang)
        sup = _bare_supervisor(static=[worker])
        with mock.patch.object(sup, "WORKER_DEADLINE_SEC", 0.2), \
                mock.patch.object(bw, "log_message"):
            sup.run_cycle()
            self.assertEqual(worker.calls, 1)

            self._release.set()
            for _ in range(50):
                if not sup.wedged():
                    break
                time.sleep(0.1)

            sup.run_cycle()

        self.assertEqual(worker.calls, 2, "the worker never resumed after recovering")
        self.assertIsNone(sup.wedged())

    def test_a_healthy_cycle_is_unchanged(self):
        """Every worker still runs, is still timed, and nothing is abandoned."""
        sup = _bare_supervisor(static=[_FakeWorker(chain="base")])
        sup.run_cycle()
        self.assertEqual(
            sorted(sup.last_worker_sec),
            ["discovery", "dynamic", "market", "static:base"],
        )
        self.assertIsNone(sup.wedged())


if __name__ == "__main__":
    unittest.main()
