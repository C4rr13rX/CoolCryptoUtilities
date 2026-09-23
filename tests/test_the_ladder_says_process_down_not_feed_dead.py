"""Ladder step 1 must not report a feed verdict when nothing is running.

WHY THIS TEST EXISTS. ``scripts/graduation_status.py`` printed
``[FAIL] 1 FEED no ticks in 10min`` identically whether the feed was starved
or the host had rebooted, and those have opposite fixes. Measured over the 528
hours ending 2026-09-23 06:00 (pass 680): 60 hours had zero production log
lines AND zero market_stream rows -- production was dead. A FURTHER 54 hours,
including an unbroken 60-hour run from 2026-09-13 11:00, had production alive
and logging with zero market_stream rows -- the feed genuinely starved. One
verdict string covered all 114, and it cost three wrong readings in pass 679.

These tests pin the three states apart WITHOUT stopping production: the
process probe is injected. Stopping the real thing would take the whole crew's
tree down and is not something a test may do.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import live_path_check as lpc  # noqa: E402


def test_no_production_process_reports_the_process_and_refuses_a_feed_verdict(monkeypatch):
    monkeypatch.setattr(lpc, "_production_procs", lambda: [])
    link = lpc.check_feed(time.time())

    # UNKNOWN, not FAIL. With no process there is no feed measurement to make.
    assert link.ok is None, "a dead host must not be reported as a feed failure"
    assert "PRODUCTION IS NOT RUNNING" in link.detail
    assert "ticks" not in link.detail, (
        "step 1 printed a feed verdict with no process running: %r" % link.detail)


def test_a_live_process_with_no_ticks_is_a_real_feed_failure(monkeypatch):
    monkeypatch.setattr(lpc, "_production_procs", lambda: [time.time() - 3600.0])
    monkeypatch.setattr(lpc, "_db", lambda: _FakeDb(total=0, fresh=0))
    link = lpc.check_feed(time.time())

    assert link.ok is False, "a starved feed on a live process is a genuine FAIL"
    assert "ALIVE" in link.detail and "no ticks" in link.detail
    # The age matters: a 2-minute-old process cannot fill a 10-minute window.
    assert "production up" in link.detail


def test_a_healthy_feed_still_prints_exactly_what_it_always_printed(monkeypatch):
    monkeypatch.setattr(lpc, "_production_procs", lambda: [time.time() - 3600.0])
    monkeypatch.setattr(lpc, "_db", lambda: _FakeDb(total=150, fresh=15))
    link = lpc.check_feed(time.time())

    assert link.ok is True
    assert link.detail == "150 ticks/10min across 15 fresh symbols", (
        "the healthy wording is load-bearing for other passes' comparisons")


def test_two_production_processes_are_named_on_the_pass_line(monkeypatch):
    now = time.time()
    monkeypatch.setattr(lpc, "_production_procs", lambda: [now - 3600.0, now - 60.0])
    monkeypatch.setattr(lpc, "_db", lambda: _FakeDb(total=150, fresh=15))
    link = lpc.check_feed(now)

    assert link.ok is True
    assert "2 production processes alive" in link.detail


def test_missing_psutil_is_unknown_not_a_dead_host(monkeypatch):
    monkeypatch.setattr(lpc, "_production_procs", lambda: None)
    monkeypatch.setattr(lpc, "_db", lambda: _FakeDb(total=150, fresh=15))
    link = lpc.check_feed(time.time())

    assert link.ok is True, "an absent dependency is not evidence of a dead host"
    assert "psutil missing" in link.detail


def test_log_silence_is_read_from_the_real_bracketed_line_format(tmp_path, monkeypatch):
    """The stamp is NOT at offset 0 -- it is '[2026-09-23 07:22:14] [INFO] ...'.

    The first cut of this helper sliced line[:19] and got '[2026-09-23 07:22:1',
    which fails to parse and returns None. None renders as 'log unreadable',
    which is exactly the ambiguity this whole change removes -- so pin it.
    """
    log = tmp_path / "system.log"
    log.write_text(
        "[2026-09-23 07:00:00] [INFO] production.console: governor: pausing\r\n"
        "[2026-09-23 07:22:14] [INFO] production.console: governor: resuming\r\n",
        encoding="utf-8")
    monkeypatch.setattr(lpc, "SYSTEM_LOG", log)

    now = time.mktime(time.strptime("2026-09-23 07:32:14", "%Y-%m-%d %H:%M:%S"))
    assert lpc._log_silent_minutes(now) == pytest.approx(10.0, abs=0.01)


def test_an_unreadable_log_does_not_crash_the_link(monkeypatch, tmp_path):
    monkeypatch.setattr(lpc, "SYSTEM_LOG", tmp_path / "does-not-exist.log")
    monkeypatch.setattr(lpc, "_production_procs", lambda: [])
    link = lpc.check_feed(time.time())
    assert link.ok is None
    assert "PRODUCTION IS NOT RUNNING" in link.detail


class _FakeDb:
    """Answers the two counts check_feed asks for, in the order it asks them."""

    def __init__(self, total: int, fresh: int) -> None:
        self._answers = [[(fresh,)], [(total,)]]

    def execute(self, sql, params=None):
        return self._answers.pop(0)
