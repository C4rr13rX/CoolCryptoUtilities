from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple


class ReadinessProbe:
    """
    Lightweight harness to exercise the training/ghost → live readiness flow
    locally with minimal data. It keeps defaults conservative so it can run
    in restricted environments without pulling large datasets.
    """

    def __init__(
        self,
        *,
        env_overrides: Dict[str, str] | None = None,
        output_path: Path | str = Path("runtime/readiness_probe.json"),
    ) -> None:
        self.output_path = Path(output_path)
        self.env_overrides: Dict[str, str] = {
            # Prefer synthetic/small slices to avoid long cold-starts.
            "HISTORICAL_DATA_DIR": "data/intermediate/synthetic_ohlcv",
            "HISTORICAL_DATA_ROOT": "data/intermediate/synthetic_ohlcv",
            # Keep horizon set small for faster confusion windows.
            "TRAINING_HORIZON_WINDOWS_SEC": "300,900,3600",
            # Avoid disk cache rebuild churn during probing.
            "DATASET_DISK_CACHE": "0",
            "DATA_FILE_CACHE_LIMIT": "4",
            # Limit news lookback for faster enrichment; safe override.
            "NEWS_HORIZON_SEC": "3600",
            # Disable live trading while probing.
            "ENABLE_LIVE_TRADING": "0",
            "AUTO_PROMOTE_LIVE": "0",
            # Silence TF/absl noise.
            "TF_CPP_MIN_LOG_LEVEL": "3",
        }
        if env_overrides:
            self.env_overrides.update(env_overrides)

    def _apply_env(self) -> None:
        for key, value in self.env_overrides.items():
            os.environ.setdefault(key, str(value))

    def run_once(self) -> Dict[str, Any]:
        """
        Build the pipeline, collect readiness + transition data, and persist
        a small report to runtime/readiness_probe.json.
        """
        self._apply_env()
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        start = time.perf_counter()
        from trading.pipeline import TrainingPipeline  # import after env overrides

        pipeline = TrainingPipeline()
        focus_assets, focus_stats = pipeline.ghost_focus_assets()
        readiness = pipeline.live_readiness_report()
        transition = pipeline.ghost_live_transition_plan()
        elapsed = time.perf_counter() - start
        payload: Dict[str, Any] = {
            "focus_assets": focus_assets,
            "focus_stats": focus_stats,
            "readiness": readiness,
            "transition_plan": transition,
            "elapsed_seconds": elapsed,
            "iteration": pipeline.iteration,
            "model_dir": str(pipeline.model_dir),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        print(f"[probe] completed in {elapsed:.2f}s; report -> {self.output_path}")
        return payload


if __name__ == "__main__":
    ReadinessProbe().run_once()
