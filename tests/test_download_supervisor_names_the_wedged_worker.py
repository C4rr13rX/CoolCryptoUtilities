"""A stalled download worker has to NAME itself while it is still stalled.

Measured 2026-09-11 (pass 120, scripts/seq_queue_budget.py over logs/system.log):
`data_ingest` -- the sequential scheduler's FEED task -- overran 346 times in
six hours and its longest abandoned run was 6879 seconds, 114 minutes with the
feed task not running at all. The longest single market-stream event-loop block
over the same span is 7393.1s and matches it.

That stall could not be attributed to a worker, because
`TokenDownloadSupervisor.run_cycle` called seven workers one after another and
recorded nothing about which one it was inside. Silence in the log is not
evidence either way: `discovery` has a 1800s interval, so a quiet discovery
worker is indistinguishable from a wedged one.

These tests pin the record that makes the next stall attributable. They are not
a cure for the stall -- nothing here can interrupt a synchronous call -- they
are the difference between "something is stuck" and naming it.
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


class WedgedWorkerIsNamedTest(unittest.TestCase):
    def test_a_wedged_worker_names_itself_while_it_is_still_wedged(self):
        """The 6879-second stall, attributable this time.

        Fails against the old run_cycle: there was no `in_flight` and no
        `wedged()`, so a cycle stuck inside the market worker looked exactly
        like a cycle stuck inside any of the other six.
        """
        release = threading.Event()
        entered = threading.Event()

        def hang():
            entered.set()
            release.wait(10.0)

        sup = _bare_supervisor(market=_FakeWorker(fn=hang))
        cycle = threading.Thread(target=sup.run_cycle, daemon=True)
        cycle.start()
        self.assertTrue(entered.wait(5.0), "the market worker never started")
        time.sleep(0.2)

        flight = sup.wedged()
        self.assertIsNotNone(flight, "run_cycle is stalled and reports nothing")
        self.assertEqual(flight["worker"], "market")
        self.assertGreater(flight["running_for_sec"], 0.0)

        release.set()
        cycle.join(timeout=5.0)
        self.assertIsNone(sup.wedged(), "in_flight outlived the cycle")

    def test_no_cycle_running_reports_none_not_a_stale_name(self):
        """Absent is not 'stuck in whatever ran last'."""
        sup = _bare_supervisor()
        self.assertIsNone(sup.wedged())
        sup.run_cycle()
        self.assertIsNone(sup.wedged())

    def test_a_slow_worker_is_logged_by_name_with_its_seconds(self):
        """One log line that says which of the seven calls is the slow one."""
        from services import background_workers as bw

        sup = _bare_supervisor(discovery=_FakeWorker(fn=lambda: time.sleep(0.05)))
        with mock.patch.object(sup, "SLOW_WORKER_SEC", 0.01), \
                mock.patch.object(bw, "log_message") as logger:
            sup.run_cycle()
        slow = [c for c in logger.call_args_list if "slow worker" in str(c)]
        self.assertTrue(slow, "a worker over budget was not named in the log")
        self.assertIn("discovery", str(slow[-1]))

    def test_one_failing_worker_does_not_stop_the_others(self):
        """The pre-existing guarantee, kept.

        run_cycle wrapped every worker in its own try/except precisely so one
        broken chain could not cost the rest of the cycle. Refactoring the
        seven call sites into one helper must not quietly drop that.
        """
        from services import background_workers as bw

        order = []

        def boom():
            order.append("dynamic")
            raise RuntimeError("chain down")

        sup = _bare_supervisor(
            dynamic=_FakeWorker(fn=boom),
            market=_FakeWorker(fn=lambda: order.append("market")),
            discovery=_FakeWorker(fn=lambda: order.append("discovery")),
        )
        with mock.patch.object(bw, "log_message"):
            sup.run_cycle()
        self.assertEqual(order, ["dynamic", "market", "discovery"])
        self.assertIsNone(sup.wedged())

    def test_every_worker_is_timed_so_the_slow_one_is_rankable(self):
        sup = _bare_supervisor(static=[_FakeWorker(chain="base")])
        sup.run_cycle()
        self.assertEqual(
            sorted(sup.last_worker_sec),
            ["discovery", "dynamic", "market", "static:base"],
        )
        self.assertGreaterEqual(sup.last_cycle_sec, 0.0)


if __name__ == "__main__":
    unittest.main()
