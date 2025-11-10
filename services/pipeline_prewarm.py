from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Sequence

from services.logging_utils import log_message

_PIPELINE_LOCK = threading.Lock()
_PIPELINE: Optional["TrainingPipeline"] = None


def _ensure_pipeline() -> "TrainingPipeline":
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            from trading.pipeline import TrainingPipeline  # Lazy import; heavy dependency

            _PIPELINE = TrainingPipeline()
    return _PIPELINE


def prewarm_training_pipeline(*, focus_assets: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """
    Initialise the shared TrainingPipeline singleton (if needed) and warm both
    the dataset cache and the news cache so the production manager's first
    ghost cycle does not stall on I/O.
    """

    pipeline = _ensure_pipeline()
    focus = list(focus_assets or [])
    summary: Dict[str, Any] = {
        "focus_assets": focus,
        "timestamp": time.time(),
        "dataset_ready": False,
        "news_ready": False,
        "iteration": getattr(pipeline, "iteration", 0),
    }
    try:
        dataset_ready = pipeline.warm_dataset_cache(focus_assets=focus or None, oversample=False)
        news_ready = pipeline.reinforce_news_cache(focus_assets=focus or None)
        summary["dataset_ready"] = bool(dataset_ready)
        summary["news_ready"] = bool(news_ready)
        log_message(
            "training",
            "pipeline prewarm complete",
            severity="info",
            details={
                "dataset_ready": summary["dataset_ready"],
                "news_ready": summary["news_ready"],
                "focus_assets": focus,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        summary["error"] = str(exc)
        log_message("training", f"pipeline prewarm failed: {exc}", severity="warning")
    return summary
