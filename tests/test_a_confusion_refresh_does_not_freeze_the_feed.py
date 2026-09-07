"""The live-readiness check must not stop the price feed.

Measured 2026-09-06 19:53. The feed reported 0 ticks in 10 minutes across 32
symbols, newest sample 1323 seconds old, while market-stream.log filled with
"REST network outage detected; pausing polls". The network was fine --
dexscreener answered in 0.50s from the same box. A py-spy dump of the
production process named the real holder:

    Thread 46444 (active+gil): "Thread-23 (_run_supervisor_loop)"
        _hurst (trading/advanced_algorithms.py:170)
        _compute_tech_features (trading/data_loader.py:1232)
        build_dataset (trading/data_loader.py:867)
        prime_confusion_windows (trading/pipeline.py:5693)
        ensure_confusion_fresh (trading/pipeline.py:3846)
        live_readiness_report (trading/pipeline.py:4815)
        ghost_live_transition_plan (trading/pipeline.py:5009)
        _maybe_transition_to_live (trading/bot.py:4258)
        _handle_sample (trading/bot.py:5765)
        _run (asyncio/events.py:89)          <-- on the event loop
        run_forever (asyncio/base_events.py:683)

_handle_sample is a coroutine on the same asyncio loop that polls every price
endpoint, and it called a full dataset rebuild plus a TF evaluation
synchronously. CONFUSION_REFRESH_MAX_AGE defaults to 900, so the feed died for
minutes at a time, every 15 minutes, and the stream blamed the network for the
timeouts it had caused itself.

The gate that decides whether we may trade live was the reason no price could
arrive to trade on.
"""

from __future__ import annotations

import asyncio
import threading
import time

from trading.pipeline import TrainingPipeline


class _Recorder:
    """The two attributes the offload touches, and a blocking body to stand in
    for the dataset rebuild."""

    def __init__(self, hold: float, min_gap: float = 0.0) -> None:
        self._confusion_refresh_lock = threading.Lock()
        self._confusion_refresh_thread = None
        self._confusion_refresh_last_dispatch = 0.0
        self._confusion_refresh_min_gap = min_gap
        self._hold = hold
        self.started = threading.Event()
        self.finished = threading.Event()
        self.calls = 0

    def _prime_confusion_windows_blocking(self, *, min_samples: int = 128, force: bool = False) -> bool:
        self.calls += 1
        self.started.set()
        time.sleep(self._hold)
        self.finished.set()
        return True

    # The real methods under test, bound to this stand-in so the assertions
    # exercise shipped code rather than a copy of it.
    prime_confusion_windows = TrainingPipeline.prime_confusion_windows
    _offload_confusion_refresh = TrainingPipeline._offload_confusion_refresh
    _last_confusion_report: dict = {}


def test_a_refresh_on_the_event_loop_returns_without_blocking_it():
    recorder = _Recorder(hold=2.0)

    async def scenario():
        loop_free_at = None
        started = time.perf_counter()
        result = recorder.prime_confusion_windows(force=True)
        elapsed = time.perf_counter() - started

        # The loop must be able to run its next callback -- a price poll --
        # immediately, not after the rebuild finishes.
        ticks = 0

        async def poll_prices():
            nonlocal ticks
            while not recorder.finished.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        loop_free_at = elapsed
        await asyncio.wait_for(poll_prices(), timeout=10.0)
        return result, loop_free_at, ticks

    result, elapsed, ticks = asyncio.run(scenario())

    assert result is False, (
        "an offloaded refresh reports 'not refreshed right now' so the caller "
        "judges on the cached report"
    )
    assert elapsed < 0.5, (
        f"prime_confusion_windows held the event loop for {elapsed:.2f}s; the "
        f"price feed cannot poll while it does"
    )
    # Ticks kept arriving for the whole 2s the rebuild was running. Before the
    # fix this was 0: the loop was inside the rebuild.
    assert ticks > 20, f"only {ticks} feed polls ran during the refresh"
    assert recorder.calls == 1, "the work still has to happen, just off the loop"


def test_the_refresh_still_blocks_when_there_is_no_loop_to_starve():
    """Off the event loop -- the training thread, selector, a CLI -- the caller
    wants the answer, and there is no feed to protect. Behaviour is unchanged."""
    recorder = _Recorder(hold=0.05)

    result = recorder.prime_confusion_windows(force=True)

    assert result is True
    assert recorder.finished.is_set(), "the refresh ran inline and completed"
    assert recorder.calls == 1


def test_a_refresh_already_running_is_not_started_again():
    """ensure_confusion_fresh re-asks on every sample until a new report is
    stamped. One worker per tick would be a fork bomb against _train_lock."""
    recorder = _Recorder(hold=1.0)

    async def scenario():
        first = recorder.prime_confusion_windows(force=True)
        recorder.started.wait(timeout=5.0)
        # Twenty more samples arrive while the first refresh is still going.
        repeats = [recorder.prime_confusion_windows(force=True) for _ in range(20)]
        await asyncio.sleep(0)
        return first, repeats

    first, repeats = asyncio.run(scenario())

    assert first is False
    assert all(r is False for r in repeats)
    recorder.finished.wait(timeout=5.0)
    assert recorder.calls == 1, (
        f"{recorder.calls} refreshes were started; single-flight means one"
    )


def test_a_refresh_that_fails_instantly_does_not_spawn_a_thread_per_sample():
    """Single-flight alone is not enough.

    _prime_confusion_windows_blocking takes _train_lock non-blocking and returns
    in microseconds when training holds it -- without stamping a new refresh
    timestamp. So the next sample finds no live thread and starts another, and
    _handle_sample runs per sample across 32 symbols. Without a dispatch floor
    that is hundreds of threads a minute, all failing at the same lock.
    """
    recorder = _Recorder(hold=0.0, min_gap=30.0)

    async def scenario():
        results = []
        for _ in range(200):
            results.append(recorder.prime_confusion_windows(force=True))
            await asyncio.sleep(0)
        return results

    results = asyncio.run(scenario())

    assert all(r is False for r in results)
    assert recorder.calls == 1, (
        f"200 samples spawned {recorder.calls} refreshes; the dispatch floor "
        f"(CONFUSION_REFRESH_MIN_GAP) should have allowed one"
    )
