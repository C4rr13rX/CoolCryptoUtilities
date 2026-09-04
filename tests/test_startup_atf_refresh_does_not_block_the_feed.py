"""The startup ATF refresh must not hold the thread the feed waits on.

No market stream exists until `ProductionManager.start()` reaches
`supervisor.build()`, so anything called ahead of it on that thread is a hole
in `market_stream`. `_task_atf_static_strategy` runs one

    main.py --action swap_quote --payload {...}

SUBPROCESS per candidate, serially, each with a 45s timeout
(`services/atf_static_strategy._quote_probe`). Bounded per probe, unbounded in
aggregate.

Measured 2026-09-04 with py-spy against production pid 511836, 1.9 minutes
after start with zero ticks and zero trading_ops written:

    start (production.py:180)
      -> _task_atf_static_strategy -> build_static_strategy_signals
      -> _quote_probe -> subprocess.run(main.py --action swap_quote)

The previous boot spent 00:39:39-00:44:10 in the same phase.

Nothing in `start()` needs its result: it is a registered scheduler task
(interval 600s), it runs again on the cycle loop, and `build()` reads
`latest_signals` over an ATF_STATIC_SIGNAL_MAX_AGE_SEC=1800 window, so the
previous run's signals still prioritise pairs on this pass.

Two properties:

  1. the startup dispatch returns immediately, on a different thread
  2. the refresh has two concurrent callers now (that thread and the
     scheduler/cycle loop) and its throttle was a check-then-set, so a second
     caller arriving mid-refresh must SKIP rather than queue another serial run
"""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import production


def _manager() -> production.ProductionManager:
    mgr = production.ProductionManager.__new__(production.ProductionManager)
    mgr._last_atf_static_refresh_ts = 0.0
    mgr._atf_static_refresh_lock = threading.Lock()
    return mgr


class StartupDispatchIsOffThread(unittest.TestCase):
    BLOCK_SEC = 1.0

    def test_dispatch_returns_before_the_refresh_finishes(self):
        mgr = _manager()
        ran_on: list = []
        released = threading.Event()

        def slow_refresh(_self):
            ran_on.append(threading.current_thread())
            released.wait(self.BLOCK_SEC * 3)

        with mock.patch.object(
            production.ProductionManager, "_task_atf_static_strategy", slow_refresh
        ):
            started = time.monotonic()
            thread = mgr._start_atf_static_refresh_async()
            elapsed = time.monotonic() - started
            try:
                # Inline, this call would not return for BLOCK_SEC*3.
                self.assertLess(
                    elapsed, self.BLOCK_SEC,
                    f"startup ATF refresh held the caller for {elapsed:.2f}s",
                )
                self.assertTrue(thread.is_alive() or ran_on)
            finally:
                released.set()
                thread.join(timeout=5.0)

        self.assertTrue(ran_on, "the refresh never ran at all")
        self.assertIsNot(
            ran_on[0], threading.current_thread(),
            "the refresh ran on the bootstrap thread",
        )

    def test_a_failing_refresh_does_not_escape_the_thread(self):
        """start() must survive a refresh that raises, as it did inline."""
        mgr = _manager()

        def boom(_self):
            raise RuntimeError("quote probe exploded")

        with mock.patch.object(
            production.ProductionManager, "_task_atf_static_strategy", boom
        ), mock.patch.object(production, "log_message") as logged:
            thread = mgr._start_atf_static_refresh_async()
            thread.join(timeout=5.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(
            any("startup ATF static refresh failed" in str(c) for c in logged.call_args_list),
            "a failing startup refresh was swallowed silently",
        )


class ConcurrentCallersDoNotBothProbe(unittest.TestCase):
    """The throttle is a check-then-set and now has two callers."""

    def test_a_second_caller_skips_while_a_refresh_is_running(self):
        mgr = _manager()
        entered = threading.Event()
        release = threading.Event()
        runs = []

        def slow_run():
            runs.append(1)
            entered.set()
            release.wait(5.0)

        with mock.patch.object(
            production.ProductionManager, "_run_atf_static_refresh", lambda _self: slow_run()
        ):
            first = threading.Thread(target=mgr._task_atf_static_strategy)
            first.start()
            self.assertTrue(entered.wait(5.0), "the first refresh never started")
            # The scheduler firing while the startup thread is mid-probe.
            mgr._task_atf_static_strategy()
            release.set()
            first.join(timeout=5.0)

        self.assertEqual(len(runs), 1, "two concurrent refreshes both probed")

    def test_the_throttle_still_bounds_repeat_calls(self):
        """The interval gate must survive the lock being added around it."""
        mgr = _manager()
        runs = []

        with mock.patch.object(
            production.ProductionManager,
            "_run_atf_static_refresh",
            lambda _self: runs.append(1),
        ):
            mgr._task_atf_static_strategy()
            mgr._task_atf_static_strategy()

        self.assertEqual(len(runs), 1, "the refresh interval stopped being enforced")

    def test_the_lock_is_released_when_the_refresh_raises(self):
        """A raising refresh must not wedge every later one."""
        mgr = _manager()
        calls = []

        def boom(_self):
            calls.append(1)
            raise RuntimeError("probe failed")

        with mock.patch.object(
            production.ProductionManager, "_run_atf_static_refresh", boom
        ):
            with self.assertRaises(RuntimeError):
                mgr._task_atf_static_strategy()

        self.assertFalse(mgr._atf_static_refresh_lock.locked())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
