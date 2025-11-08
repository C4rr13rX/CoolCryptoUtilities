from __future__ import annotations

import numpy as np

from trading.metrics import ConfusionMatrixSummary, confusion_from_scores, confusion_sweep


def test_confusion_from_scores_counts() -> None:
    scores = np.array([0.9, 0.8, 0.4, 0.2], dtype=np.float64)
    labels = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    summary = confusion_from_scores(scores, labels, threshold=0.5)
    assert isinstance(summary, ConfusionMatrixSummary)
    assert summary.tp == 1
    assert summary.fp == 1
    assert summary.tn == 1
    assert summary.fn == 1
    report = summary.report()
    assert 0.0 <= report["precision"] <= 1.0


def test_confusion_sweep_returns_multiple_thresholds() -> None:
    scores = [0.7, 0.6, 0.3, 0.9]
    labels = [1, 0, 0, 1]
    sweep = confusion_sweep(scores, labels, thresholds=[0.4, 0.8])
    assert 0.4 in sweep and 0.8 in sweep
    assert sweep[0.4].samples == len(scores)
    assert sweep[0.8].tp <= sweep[0.4].tp
