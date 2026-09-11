"""The live gate must never decide on old evidence without saying so.

TWO FAILURES ARE PINNED HERE, AND THEY ARE NOT THE SAME BUG.

1. ``_prime_confusion_windows_blocking`` read ``self._train_lock`` directly, so
   a pipeline built by ``TrainingPipeline.__new__`` -- the idiom the test suite
   uses to skip a heavy ``__init__`` -- raised
   ``AttributeError("'TrainingPipeline' object has no attribute '_train_lock'")``
   out of the confusion refresh. ``live_readiness_report`` caught it and logged,
   at ERROR, "judging on the cached report, which may be stale", which named a
   live-money risk for what was a construction bug in the caller. 145 of those
   lines were in the last 50MB of logs/system.log on 2026-09-10.

2. The real one. ``live_readiness_report`` called ``ensure_confusion_fresh()``
   and DISCARDED its return value. That return is False on every path where the
   refresh did not happen and nothing raised -- offloaded to a worker because
   the caller is on the asyncio loop (the normal production case), suppressed by
   single-flight or the min-gap, blocked on the GuardianLease or the train lock,
   or short of min_samples. In all of those the gate judged live readiness on the
   cached report and logged nothing at all. Measured 2026-09-11 00:06,
   data/reports/confusion_matrices.json was 5968 seconds old against the
   900-second CONFUSION_REFRESH_MAX_AGE the same code enforces, silently.

Every assertion below is on captured ``log_message`` calls rather than on
logs/system.log, deliberately: that file is production's, the bus that writes it
is an async queue shared with a running system, and a test that greps it both
pollutes the operator's log and races the flush.
"""

from __future__ import annotations

import time

import pytest

import trading.pipeline as pipeline_module
from trading.pipeline import CONFUSION_WINDOW_BUCKETS, TrainingPipeline


@pytest.fixture()
def captured_logs(monkeypatch) -> list:
    """Intercept log_message so nothing here reaches the production log."""
    records: list = []

    def _capture(source, message, *, severity="info", details=None):
        records.append({"source": source, "message": message, "severity": severity, "details": details})

    monkeypatch.setattr(pipeline_module, "log_message", _capture)
    return records


def _pipeline_stub() -> TrainingPipeline:
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline._last_confusion_summary = {}
    pipeline._last_sample_meta = {}
    pipeline._confusion_windows = {label: seconds for label, seconds in CONFUSION_WINDOW_BUCKETS}
    pipeline.decision_threshold = 0.3
    pipeline.active_accuracy = 0.0
    pipeline.max_false_positive_rate = 0.15
    pipeline.min_ghost_win_rate = 0.55
    pipeline.min_realized_margin = 0.0
    pipeline._last_candidate_feedback = {}
    pipeline._last_confusion_refresh = 0.0
    pipeline._last_confusion_report = {
        "5m": {
            "precision": 0.4911,
            "recall": 0.5391,
            "samples": 510,
            "threshold": 0.3,
            "false_positive_rate": 0.5629,
            "f1_score": 0.513,
        }
    }
    return pipeline


def _warnings_about_staleness(records: list) -> list:
    return [
        r
        for r in records
        if r["severity"] in {"warning", "error", "critical"}
        and "judging on a confusion report" in r["message"]
    ]


def test_a_pipeline_built_without_init_names_its_caller_instead_of_raising(captured_logs, monkeypatch) -> None:
    """Cause (2): a caller that bypasses __init__, named at WARNING, not an AttributeError.

    Against the old code this raised AttributeError out of the refresh; the
    assertion that it returns False is what fails there.
    """
    # The lease is checked before the lock, so an unavailable lease would short
    # the function out before it ever reads _train_lock and the test would pass
    # for the wrong reason. Take it out of the question.
    monkeypatch.setattr(pipeline_module, "GuardianLease", None)

    pipeline = _pipeline_stub()
    assert not hasattr(pipeline, "_train_lock")

    assert pipeline._prime_confusion_windows_blocking(force=True) is False

    named = [
        r
        for r in captured_logs
        if r["severity"] == "warning" and "__init__" in r["message"] and "_train_lock" in r["message"]
    ]
    assert named, f"the construction bug was not named at WARNING: {captured_logs}"
    # ...and it must NOT be reported as a stale-report risk, which is what the
    # old ERROR line did.
    assert not _warnings_about_staleness(captured_logs)


def test_live_readiness_says_at_warning_when_it_judges_on_a_stale_report(captured_logs) -> None:
    """Failure (2): the silent path. ensure_confusion_fresh returned False and nobody said so."""
    pipeline = _pipeline_stub()
    # 5968s, the measured age of data/reports/confusion_matrices.json on
    # 2026-09-11 00:06, against the 900s default max age.
    pipeline._last_confusion_refresh = time.time() - 5968.0
    # The refresh did not run and did not raise -- the normal production outcome
    # when the work is offloaded to a worker off the asyncio loop.
    pipeline.ensure_confusion_fresh = lambda **_: False

    report = pipeline.live_readiness_report()

    said = _warnings_about_staleness(captured_logs)
    assert said, f"live readiness judged on a 5968s-old report in silence: {captured_logs}"
    assert said[0]["severity"] == "warning"
    assert "900" in said[0]["message"], said[0]["message"]
    assert report["confusion_stale"] is True
    assert report["confusion_age_sec"] == pytest.approx(5968.0, abs=5.0)
    assert report["confusion_max_age_sec"] == pytest.approx(900.0)


def test_a_fresh_confusion_report_is_judged_without_a_warning(captured_logs) -> None:
    """The warning must mean something: a report inside its own max age is silent."""
    pipeline = _pipeline_stub()
    pipeline._last_confusion_refresh = time.time()
    pipeline.ensure_confusion_fresh = lambda **_: True

    report = pipeline.live_readiness_report()

    assert not _warnings_about_staleness(captured_logs)
    assert report["confusion_stale"] is False
    assert report["confusion_age_sec"] < 900.0


def test_an_unrefreshed_but_still_young_report_is_not_called_stale(captured_logs) -> None:
    """A refresh that did not run is not by itself a staleness event.

    ensure_confusion_fresh returns False whenever it declined to refresh, and it
    declines whenever the cached report is still inside max_age via its own
    early return path being bypassed. Warning on every such call would reproduce
    the 30-lines-per-400KB spam that got the last warning ignored.
    """
    pipeline = _pipeline_stub()
    pipeline._last_confusion_refresh = time.time() - 60.0
    pipeline.ensure_confusion_fresh = lambda **_: False

    report = pipeline.live_readiness_report()

    assert not _warnings_about_staleness(captured_logs)
    assert report["confusion_stale"] is False


def test_the_no_confusion_data_verdict_carries_the_age_of_its_evidence(captured_logs, monkeypatch) -> None:
    """The fallback verdicts decide live readiness too, and returned no age at all.

    Against the old code this path returned a dict with only `ready` and
    `reason`, so a caller could not tell a fresh refusal from a ten-hour-old one.
    """
    pipeline = _pipeline_stub()
    pipeline._last_confusion_report = {}
    pipeline._last_confusion_refresh = time.time() - 5968.0
    pipeline.ensure_confusion_fresh = lambda **_: False
    monkeypatch.setattr(TrainingPipeline, "load_active_model", lambda self: None, raising=False)

    report = pipeline.live_readiness_report()

    assert report["ready"] is False
    assert "confusion_age_sec" in report, report
    assert report["confusion_stale"] is True
    assert _warnings_about_staleness(captured_logs)


def test_the_stale_warning_counts_the_verdicts_it_did_not_print(captured_logs) -> None:
    """Spacing the line must not hide how many verdicts rode on the old report."""
    pipeline = _pipeline_stub()
    pipeline._last_confusion_refresh = time.time() - 5968.0
    pipeline.ensure_confusion_fresh = lambda **_: False

    for _ in range(4):
        pipeline.live_readiness_report()

    said = _warnings_about_staleness(captured_logs)
    # Rate-limited to one line, not four...
    assert len(said) == 1, [r["message"] for r in said]
    # ...but the three suppressed verdicts are still counted by the next line.
    assert pipeline._confusion_stale_judgments == 3
