from __future__ import annotations

from trading.pipeline import TrainingPipeline


def test_horizon_forecast_uses_profile() -> None:
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline._last_dataset_meta = {
        "horizon_profile": {
            "300": {"mean_return": 0.01, "positive_ratio": 0.6, "samples": 120},
            "2592000": {"mean_return": 0.05, "positive_ratio": 0.55, "samples": 80},
        },
        "lookahead_median_sec": 300,
    }
    pipeline._last_sample_meta = {"records": [{"lookahead_sec": 300}, {"lookahead_sec": 300}]}

    forecast = TrainingPipeline.horizon_forecast(pipeline, predicted_return=0.02, current_price=200.0)

    assert forecast["base_lookahead_sec"] == 300
    horizon_map = forecast["forecast"]
    assert "5m" in horizon_map
    assert "1mth" in horizon_map
    assert horizon_map["5m"]["price"] > 0


class _DummyLoader:
    def __init__(self, windows: tuple[int, ...]) -> None:
        self._horizon_windows = windows


def _pipeline_with_windows(windows: tuple[int, ...]) -> TrainingPipeline:
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline._horizon_targets = {"short": 32.0, "mid": 24.0, "long": 12.0}
    pipeline.data_loader = _DummyLoader(windows)
    return pipeline


def test_effective_horizon_targets_drop_missing_long_bucket() -> None:
    pipeline = _pipeline_with_windows((300, 900, 3600, 86400))

    targets = pipeline._effective_horizon_targets()

    assert targets["long"] == 0.0
    assert targets["short"] == 32.0
    assert targets["mid"] == 24.0


def test_effective_horizon_targets_keep_long_when_available() -> None:
    pipeline = _pipeline_with_windows((300, 900, 86400, 604800))

    targets = pipeline._effective_horizon_targets()

    assert targets["long"] == 12.0
    assert targets["mid"] == 24.0


def test_effective_horizon_targets_drop_short_when_absent() -> None:
    pipeline = _pipeline_with_windows((3600, 604800))

    targets = pipeline._effective_horizon_targets()

    assert targets["short"] == 0.0
    assert targets["long"] == 12.0


def test_effective_horizon_targets_drop_mid_when_absent() -> None:
    pipeline = _pipeline_with_windows((300, 604800))

    targets = pipeline._effective_horizon_targets()

    assert targets["short"] == 32.0
    assert targets["mid"] == 0.0
    assert targets["long"] == 12.0
