from __future__ import annotations

import pytest

from trading.pipeline import CONFUSION_WINDOW_BUCKETS, TrainingPipeline


def test_live_readiness_rebuilds_summary_when_missing() -> None:
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
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
    pipeline._last_confusion_summary = {}
    pipeline._last_sample_meta = {}
    pipeline._confusion_windows = {label: seconds for label, seconds in CONFUSION_WINDOW_BUCKETS}
    pipeline.decision_threshold = 0.3
    pipeline.active_accuracy = 0.0
    pipeline.max_false_positive_rate = 0.15
    pipeline.min_ghost_win_rate = 0.55
    pipeline._last_candidate_feedback = {}

    report = pipeline.live_readiness_report()

    assert report["horizon"] == "5m"
    assert report["mini_precision"] == pytest.approx(0.4911)
    assert report["mini_samples"] == 510
