"""Live readiness must not be granted on the EXISTENCE OF A MODEL FILE.

THREE FAILURES ARE PINNED HERE AND THEY ARE ONE ITEM, NOT ONE BUG.

1. ABSENT WAS PRINTED AS STALE. ``_last_confusion_refresh`` starts at 0.0 and
   only the two writers that adopt a report stamp it. ``confusion_freshness``
   computed ``time.time() - 0.0`` and the warning line rendered that as a
   duration: logs/system.log:5833140 at 06:27:39 reads "live readiness is
   judging on a confusion report 1789122459s old against a 900s max age".
   1789122459 IS the unix epoch at that instant -- 56 years. The number is a
   SENTINEL meaning the field was never stamped, not an age.

2. THE DISK CACHE WAS NEVER RE-READ. Only ``__init__`` called
   ``_load_cached_confusion_report``, while ``_evaluate_and_promote`` wipes
   ``_last_confusion_report`` to ``{}`` on every build that yields nothing.
   After the first such wipe the process holds an empty report for the rest of
   its life and ``ensure_confusion_fresh`` orders a full rebuild -- minutes of
   CPU -- on every freshness check, while data/reports/confusion_matrices.json
   sits on disk minutes old.

3. THE FALLBACK GRANTED READINESS ON A FILE, AND THEN FAKED THE METRICS. With
   no confusion report and no candidate feedback, the fallback read: if every
   measured metric fails and a model is loadable from disk, declare mini_ready
   anyway -- and overwrite the two metrics that had just failed with values
   manufactured from the thresholds they failed. A measured false_positive_rate
   of 1.0 (the value used when there is no feedback at all) was published in the
   ``false_positive_rate`` key as 0.15.

Every assertion is on a pipeline stub and on captured ``log_message`` calls, not
on logs/system.log: that file is production's and a test that greps it both
pollutes the operator's log and races the flush.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import trading.pipeline as pipeline_module
from trading.pipeline import CONFUSION_WINDOW_BUCKETS, TrainingPipeline


@pytest.fixture()
def captured_logs(monkeypatch) -> list:
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
    pipeline._last_confusion_report = {}
    return pipeline


def _staleness_lines(records: list) -> list:
    return [
        r
        for r in records
        if r["severity"] in {"warning", "error", "critical"}
        and "live readiness is judging" in r["message"]
    ]


# ---------------------------------------------------------------- failure (1)


def test_a_never_stamped_pipeline_reports_absent_not_an_age() -> None:
    pipeline = _pipeline_stub()

    freshness = pipeline.confusion_freshness()

    assert freshness["confusion_never_loaded"] is True
    assert freshness["confusion_age_sec"] is None
    # ABSENT is not STALE. Stale means an age was measured and breached.
    assert freshness["confusion_stale"] is False


def test_the_warning_for_a_never_stamped_pipeline_prints_no_epoch(captured_logs) -> None:
    """The log line must not contain a number larger than 10^9.

    Against the old code the message carried 1789122459 -- the unix epoch
    formatted as a duration.
    """
    pipeline = _pipeline_stub()
    pipeline.ensure_confusion_fresh = lambda **_: False

    pipeline._announce_stale_confusion_judgment(refreshed=False, refresh_error="")

    said = _staleness_lines(captured_logs)
    assert said, f"a never-loaded confusion report was judged in silence: {captured_logs}"
    assert "no confusion report has ever been loaded" in said[0]["message"], said[0]["message"]
    numbers = [int(tok) for tok in __import__("re").findall(r"\d+", said[0]["message"])]
    assert not [n for n in numbers if n > 10**9], said[0]["message"]


def test_a_measured_age_still_prints_as_an_age(captured_logs) -> None:
    """The absent branch must not swallow the real staleness warning."""
    pipeline = _pipeline_stub()
    pipeline._last_confusion_report = {"5m": {"precision": 0.4}}
    pipeline._last_confusion_refresh = time.time() - 5968.0
    pipeline.ensure_confusion_fresh = lambda **_: False

    freshness = pipeline._announce_stale_confusion_judgment(refreshed=False, refresh_error="")

    assert freshness["confusion_never_loaded"] is False
    assert freshness["confusion_stale"] is True
    said = _staleness_lines(captured_logs)
    assert said and "900s max age" in said[0]["message"], said


# ---------------------------------------------------------------- failure (2)


def test_an_empty_in_memory_report_picks_up_a_fresh_file_without_priming(tmp_path, monkeypatch) -> None:
    """The freshness path must re-read the disk cache before ordering a rebuild.

    Against the old code ensure_confusion_fresh went straight to
    prime_confusion_windows and this asserts it did NOT have to.
    """
    monkeypatch.chdir(tmp_path)
    reports = tmp_path / "data" / "reports"
    reports.mkdir(parents=True)
    (reports / "confusion_matrices.json").write_text(
        json.dumps(
            {
                "updated_at": time.time() - 60.0,
                "decision_threshold": 0.3,
                "confusion": {"5m": {"precision": 0.61, "recall": 0.5, "samples": 400}},
            }
        ),
        encoding="utf-8",
    )

    pipeline = _pipeline_stub()
    primed: list = []

    def _never(*args, **kwargs):
        primed.append(True)
        return False

    pipeline.prime_confusion_windows = _never
    pipeline._summarize_confusion_report = lambda report: {"horizons": list(report)}
    pipeline._adjust_horizon_bias_from_confusion = lambda: None
    pipeline._build_transition_plan = lambda: {}

    assert pipeline.ensure_confusion_fresh() is True
    assert not primed, "a full rebuild was ordered although the disk cache was 60s old"
    assert pipeline._last_confusion_report, pipeline._last_confusion_report
    assert pipeline.confusion_freshness()["confusion_never_loaded"] is False


def test_a_stale_file_on_disk_still_orders_a_refresh(tmp_path, monkeypatch) -> None:
    """The re-read must not become a way to ride an old file forever."""
    monkeypatch.chdir(tmp_path)
    reports = tmp_path / "data" / "reports"
    reports.mkdir(parents=True)
    (reports / "confusion_matrices.json").write_text(
        json.dumps(
            {
                "updated_at": time.time() - 5968.0,
                "confusion": {"5m": {"precision": 0.61}},
            }
        ),
        encoding="utf-8",
    )

    pipeline = _pipeline_stub()
    primed: list = []
    pipeline.prime_confusion_windows = lambda **kwargs: (primed.append(True), False)[1]
    pipeline._summarize_confusion_report = lambda report: {"horizons": list(report)}
    pipeline._adjust_horizon_bias_from_confusion = lambda: None
    pipeline._build_transition_plan = lambda: {}

    assert pipeline.ensure_confusion_fresh() is False
    assert primed, "a 5968s-old file was accepted against a 900s max age"


# ---------------------------------------------------------------- failure (3)


def _fallback_report(pipeline, captured_logs, monkeypatch, *, has_model: bool = True):
    monkeypatch.delenv("LIVE_ALLOW_MODEL_FILE_AS_EVIDENCE", raising=False)
    monkeypatch.delenv("LIVE_ALLOW_MINI_READY", raising=False)
    monkeypatch.setattr(
        TrainingPipeline,
        "load_active_model",
        lambda self: object() if has_model else None,
        raising=False,
    )
    pipeline.ensure_confusion_fresh = lambda **_: False
    return pipeline.live_readiness_report()


def test_an_absent_confusion_report_cannot_grant_live_readiness(captured_logs, monkeypatch) -> None:
    """THE RED TEST. Empty report, empty feedback, a model file on disk, no flag.

    Against today's code this returns ready=True and mini_ready=True on nothing
    but the model file existing.
    """
    pipeline = _pipeline_stub()

    report = _fallback_report(pipeline, captured_logs, monkeypatch, has_model=True)

    assert report["ready"] is False, report
    assert report["mini_ready"] is False, report
    assert report["reason"] == "no_model_evidence", report
    assert report["mini_reason"] == "no_model_evidence", report


def test_the_fallback_reports_the_measured_false_positive_rate_not_the_threshold(
    captured_logs, monkeypatch
) -> None:
    """1.0 is the measured value when there is no feedback. 0.15 is a threshold."""
    pipeline = _pipeline_stub()

    report = _fallback_report(pipeline, captured_logs, monkeypatch, has_model=True)

    assert report["false_positive_rate"] == pytest.approx(1.0), report
    assert report["mini_false_positive_rate"] == pytest.approx(1.0), report
    assert report["false_positive_rate_floor_applied"] is False, report
    assert report["ghost_win_rate_floor_applied"] is False, report
    assert report["model_file_grant"] is False, report


def test_the_model_file_grant_survives_only_behind_its_named_flag(captured_logs, monkeypatch) -> None:
    """Kept, but OFF by default and self-declaring when on."""
    pipeline = _pipeline_stub()
    monkeypatch.setenv("LIVE_ALLOW_MODEL_FILE_AS_EVIDENCE", "1")
    monkeypatch.delenv("LIVE_ALLOW_MINI_READY", raising=False)
    monkeypatch.setattr(TrainingPipeline, "load_active_model", lambda self: object(), raising=False)
    pipeline.ensure_confusion_fresh = lambda **_: False

    report = pipeline.live_readiness_report()

    assert report["mini_ready"] is True, report
    assert report["model_file_grant"] is True, report
    # ...and even then the measurement is still the measurement.
    assert report["false_positive_rate"] == pytest.approx(1.0), report
    assert report["false_positive_rate_floor_applied"] is True, report
    assert report["false_positive_rate_floor_value"] == pytest.approx(0.15), report


def test_no_model_on_disk_is_refused_the_same_way(captured_logs, monkeypatch) -> None:
    pipeline = _pipeline_stub()

    report = _fallback_report(pipeline, captured_logs, monkeypatch, has_model=False)

    assert report["ready"] is False, report
    assert report["mini_ready"] is False, report
