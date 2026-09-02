"""A download subprocess must never be able to block production startup.

`_ensure_ohlcv` reaches `_run_download` from `selector.build`, which runs on the
MAIN thread inside `production.start`. The `proc.wait()` there had no timeout,
so one slow download2000 child stopped the whole system from coming up.

Observed 2026-09-02: production restarted at 15:42, its MainThread parked in
`subprocess.wait` via `_run_download`, and 9 minutes later the trading loop had
still not started and the prices table had taken zero rows.

These tests pin the bound and the cleanup. The tree matters as much as the
timeout: download2000 spawns its own workers, and killing only the direct child
leaves the orphans that later saturate the per-IP connection budget.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from services import background_workers as bw


class _FakeProc:
    """Stands in for Popen: records how it was waited on and killed."""

    def __init__(self, *, finishes_after: float = 0.0, pid: int = 4321):
        self.pid = pid
        self._finishes_after = finishes_after
        self._started = time.time()
        self.waits: list = []
        self.killed = False

    def wait(self, timeout=None):
        self.waits.append(timeout)
        elapsed = time.time() - self._started
        if self._finishes_after <= elapsed:
            return 0
        if timeout is None:
            raise AssertionError("unbounded wait: this is the bug under test")
        if timeout >= self._finishes_after - elapsed:
            return 0
        raise subprocess.TimeoutExpired(cmd="download2000.py", timeout=timeout)


@pytest.fixture()
def killed(monkeypatch):
    seen = []
    monkeypatch.setattr(bw, "_kill_tree", lambda pid: seen.append(pid))
    return seen


class TestWaitIsBounded:
    def test_a_hung_download_is_killed_instead_of_waited_on_forever(self, monkeypatch, killed):
        monkeypatch.setenv("DOWNLOAD_SUBPROCESS_TIMEOUT_SEC", "0.2")
        proc = _FakeProc(finishes_after=999.0)

        bw._wait_or_kill(proc, "base")

        assert killed == [proc.pid], "the hung tree must be killed"
        assert proc.waits[0] == pytest.approx(0.2), "the first wait must be bounded"

    def test_a_quick_download_is_never_killed(self, monkeypatch, killed):
        monkeypatch.setenv("DOWNLOAD_SUBPROCESS_TIMEOUT_SEC", "30")
        proc = _FakeProc(finishes_after=0.0)

        bw._wait_or_kill(proc, "base")

        assert killed == [], "a download that finished must not be killed"

    def test_the_call_returns_rather_than_raising_on_timeout(self, monkeypatch, killed):
        """Startup must continue past a dead download, not crash on it."""
        monkeypatch.setenv("DOWNLOAD_SUBPROCESS_TIMEOUT_SEC", "0.1")
        proc = _FakeProc(finishes_after=999.0)

        bw._wait_or_kill(proc, "base")  # must not raise

    def test_timeout_is_configurable(self, monkeypatch, killed):
        monkeypatch.setenv("DOWNLOAD_SUBPROCESS_TIMEOUT_SEC", "7")
        proc = _FakeProc(finishes_after=999.0)

        bw._wait_or_kill(proc, "base")

        assert proc.waits[0] == pytest.approx(7.0)

    def test_zero_timeout_is_an_explicit_opt_out(self, monkeypatch, killed):
        """0 means "wait forever" on purpose, for offline backfills."""
        monkeypatch.setenv("DOWNLOAD_SUBPROCESS_TIMEOUT_SEC", "0")
        proc = _FakeProc(finishes_after=0.0)

        bw._wait_or_kill(proc, "base")

        assert proc.waits == [None], "opt-out should use the unbounded wait"
        assert killed == []


class TestKillTree:
    @pytest.mark.skipif(sys.platform != "win32", reason="spawn helper is shell-specific")
    def test_kill_tree_kills_the_grandchild_too(self):
        """The orphan problem is grandchildren, so pin that they die."""
        psutil = pytest.importorskip("psutil")

        # A parent python that spawns a long-lived child python.
        code = (
            "import subprocess,sys,time;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);"
            "print(p.pid,flush=True);"
            "time.sleep(120)"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True
        )
        try:
            child_pid = int(parent.stdout.readline().strip())
            assert psutil.pid_exists(child_pid)

            bw._kill_tree(parent.pid)

            deadline = time.time() + 20
            while time.time() < deadline and psutil.pid_exists(child_pid):
                time.sleep(0.2)
            assert not psutil.pid_exists(child_pid), "grandchild survived: this is the orphan leak"
        finally:
            try:
                parent.kill()
            except Exception:
                pass
