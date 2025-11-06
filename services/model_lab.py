from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import tensorflow as tf

from db import TradingDatabase, get_db
from trading.pipeline import TrainingPipeline


@dataclass
class LabJobConfig:
    train_files: List[str]
    eval_files: List[str]
    epochs: int = 1
    batch_size: int = 16


class ModelLabRunner:
    def __init__(self, *, db: Optional[TradingDatabase] = None) -> None:
        self.db = db or get_db()
        self.base_dir = Path("data") / "historical_ohlcv"
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._status: Dict[str, Any] = {
            "running": False,
            "progress": 0.0,
            "message": "idle",
            "started_at": None,
            "result": None,
            "error": None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_files(self) -> List[Dict[str, Any]]:
        root = self.base_dir.resolve()
        if not root.exists():
            return []
        records: List[Dict[str, Any]] = []
        for path in sorted(root.rglob("*.json")):
            try:
                stat = path.stat()
            except OSError:
                continue
            rel = path.relative_to(root)
            chain = rel.parts[0] if len(rel.parts) > 1 else "unknown"
            symbol = path.stem.split("_", 1)[-1].upper()
            records.append(
                {
                    "path": str(rel).replace("\\", "/"),
                    "chain": chain,
                    "symbol": symbol,
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
        return records

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._status))

    def start_job(self, config: LabJobConfig) -> None:
        with self._lock:
            if self._status.get("running"):
                raise RuntimeError("Model lab job already running.")
            self._status.update(
                {
                    "running": True,
                    "progress": 0.0,
                    "message": "initialising",
                    "started_at": time.time(),
                    "result": None,
                    "error": None,
                }
            )
        thread = threading.Thread(target=self._run_job, args=(config,), daemon=True)
        self._thread = thread
        thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_paths(self, entries: Sequence[str]) -> List[str]:
        root = self.base_dir.resolve()
        resolved: List[str] = []
        for item in entries:
            path = Path(item)
            if not path.is_absolute():
                path = (root / path).resolve()
            else:
                path = path.resolve()
            if not str(path).startswith(str(root)):
                continue
            if path.is_file() and path.suffix.lower() == ".json":
                resolved.append(str(path))
        return resolved

    def _relative_path(self, path: str) -> str:
        root = self.base_dir.resolve()
        try:
            rel = Path(path).resolve().relative_to(root)
            return rel.as_posix()
        except ValueError:
            return Path(path).name

    def _update_status(self, **kwargs: Any) -> None:
        with self._lock:
            self._status.update(kwargs)

    def _run_job(self, config: LabJobConfig) -> None:
        train_files = self._resolve_paths(config.train_files)
        eval_files = self._resolve_paths(config.eval_files)
        train_rel = [self._relative_path(path) for path in train_files]
        eval_rel = [self._relative_path(path) for path in eval_files]
        managed_pause = False
        original_flag = None
        try:
            original_flag = self.db.get_control_flag("pipeline_pause")
            if not original_flag or str(original_flag).strip().lower() not in {"1", "true", "yes", "paused"}:
                self.db.set_control_flag("pipeline_pause", "1")
                managed_pause = True
            self._update_status(progress=0.05, message="Pausing training pipeline")
            time.sleep(0.5)

            pipeline = TrainingPipeline(db=self.db)
            lab_model: Optional[tf.keras.Model] = None
            train_summary: Optional[Dict[str, Any]] = None
            eval_summary: Optional[Dict[str, Any]] = None

            if train_files:
                self._update_status(progress=0.15, message="Building training dataset")
                lab_model, train_metrics, train_info = pipeline.lab_train_on_files(
                    train_files,
                    epochs=max(1, config.epochs),
                    batch_size=max(8, config.batch_size),
                )
                train_info["files"] = train_rel
                train_summary = {
                    "metrics": train_metrics,
                    "info": train_info,
                }
                self._update_status(progress=0.65, message="Training completed")
            else:
                lab_model = pipeline.ensure_active_model()
                self._update_status(progress=0.25, message="Using active model weights")

            if eval_files:
                if lab_model is None:
                    lab_model = pipeline.ensure_active_model()
                self._update_status(progress=0.7, message="Running evaluation")
                eval_metrics = pipeline.lab_evaluate_on_files(
                    lab_model,
                    eval_files,
                    batch_size=max(8, config.batch_size),
                )
                eval_summary = {"metrics": eval_metrics, "files": eval_rel}
                self._update_status(progress=0.95, message="Evaluation complete")

            self._update_status(
                progress=1.0,
                message="Model lab run complete",
                running=False,
                result={
                    "train": train_summary,
                    "evaluation": eval_summary,
                    "train_files": train_rel,
                    "eval_files": eval_rel,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._update_status(
                running=False,
                progress=1.0,
                message="Model lab run failed",
                error=str(exc),
            )
        finally:
            if managed_pause:
                try:
                    self.db.clear_control_flag("pipeline_pause")
                except Exception:
                    pass
            if original_flag and not managed_pause:
                # ensure original flag value persists
                try:
                    self.db.set_control_flag("pipeline_pause", original_flag)
                except Exception:
                    pass
            if not managed_pause and not original_flag:
                try:
                    self.db.clear_control_flag("pipeline_pause")
                except Exception:
                    pass
            with self._lock:
                self._thread = None


_LAB_RUNNER: Optional[ModelLabRunner] = None
_LAB_LOCK = threading.Lock()


def get_model_lab_runner() -> ModelLabRunner:
    global _LAB_RUNNER
    with _LAB_LOCK:
        if _LAB_RUNNER is None:
            _LAB_RUNNER = ModelLabRunner()
    return _LAB_RUNNER
