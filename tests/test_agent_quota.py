"""The agent must wait out a quota wall, not hammer it.

A spent session is not a crash. Treated as one, the worker retries on its
normal cadence -- spawning a process per cycle that produces nothing until
the window reopens, and filling the run history with failures where the truth
is "waiting".
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))

from tradingagent.engine import _parse_reset_epoch, _QUOTA_MARKERS  # noqa: E402


class TestDetection:
    @pytest.mark.parametrize("message", [
        "Claude usage limit reached. Your limit will reset at 1788600000",
        "Error: rate limit exceeded, try again later",
        "HTTP 429 Too Many Requests",
        "quota exceeded for this billing period",
        "insufficient_quota: please upgrade",
    ])
    def test_the_wall_is_recognised_however_it_is_worded(self, message):
        """Neither CLI guarantees the wording or the stream it lands on."""
        assert any(m in message.lower() for m in _QUOTA_MARKERS), message

    @pytest.mark.parametrize("message", [
        "unparsable JSON: expecting value",
        "claude not found on PATH",
        "connection reset by peer",
    ])
    def test_ordinary_failures_are_not_mistaken_for_a_wall(self, message):
        """Backing off an hour on a JSON parse error would be its own outage."""
        assert not any(m in message.lower() for m in _QUOTA_MARKERS), message


class TestResetParsing:
    def test_a_stated_reset_is_used(self):
        future = int(time.time()) + 1800
        parsed = _parse_reset_epoch(f'{{"resets_at": {future}}}')
        assert parsed == pytest.approx(float(future), abs=1.0)

    def test_milliseconds_are_understood(self):
        future_ms = (int(time.time()) + 1800) * 1000
        parsed = _parse_reset_epoch(f'"resetsAt":{future_ms}')
        assert parsed is not None
        assert parsed == pytest.approx(future_ms / 1000.0, abs=1.0)

    def test_a_reset_in_the_past_is_refused(self):
        """A stale timestamp would mean "wait zero seconds", i.e. hammer it."""
        assert _parse_reset_epoch(f'"resets_at": {int(time.time()) - 5000}') is None

    def test_an_absurd_reset_is_refused(self):
        """More than a day out is a parse error, not a wait."""
        assert _parse_reset_epoch(f'"resets_at": {int(time.time()) + 999999}') is None

    def test_no_timestamp_is_none_not_a_guess(self):
        """None means "we do not know" -- the caller backs off on a schedule.

        A guessed reset either wastes the window or retries into a closed
        door, and both are worse than a fixed backoff.
        """
        assert _parse_reset_epoch("usage limit reached, try later") is None


class TestBackoff:
    def _command(self):
        from tradingagent.management.commands.tradingagent_worker import Command

        command = Command()
        command._quota_streak = 0
        return command

    class _Run:
        def __init__(self, report="", status="failed", resets_at=None):
            self.report = report
            self.status = status
            self.resets_at = resets_at

    def test_a_normal_run_waits_the_ordinary_interval(self):
        command = self._command()
        assert command._quota_wait_seconds(self._Run("opened 1 position",
                                                     "completed")) == 0.0

    def test_a_quota_run_waits(self):
        command = self._command()
        assert command._quota_wait_seconds(self._Run("quota exhausted")) > 0

    def test_a_stated_reset_is_preferred_over_the_backoff(self):
        command = self._command()
        wait = command._quota_wait_seconds(
            self._Run("quota exhausted", resets_at=time.time() + 900))
        # 15 minutes plus the safety minute, not the 5-minute first backoff.
        assert 900 < wait < 1100

    def test_the_backoff_grows_with_the_streak(self):
        command = self._command()
        waits = [command._quota_wait_seconds(self._Run("quota exhausted"))
                 for _ in range(4)]
        assert waits == sorted(waits), waits
        assert waits[-1] > waits[0]

    def test_the_backoff_is_capped(self):
        command = self._command()
        waits = [command._quota_wait_seconds(self._Run("quota exhausted"))
                 for _ in range(20)]
        assert max(waits) <= 3600.0
