from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from db import TradingDatabase, get_db
from services.news_lab import collect_news_for_files

if TYPE_CHECKING:  # pragma: no cover - import-heavy modules only for type hints
    from trading.pipeline import TrainingPipeline
    import tensorflow as tf


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
            "log": [],
            "history": [],
            "events": [],
            "snapshot": {},
        }
        self._history: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []

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

    def resolve_paths(self, entries: Sequence[str]) -> List[str]:
        return self._resolve_paths(entries)

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
                    "log": [],
                    "history": list(self._history),
                    "job_type": "model_lab",
                    "events": [],
                    "snapshot": {},
                }
            )
            self._events = []
        self._update_snapshot(
            config={
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "train_files": list(config.train_files),
                "eval_files": list(config.eval_files),
            },
            started_at=self._status.get("started_at"),
        )
        self._record_event(
            "info",
            "job_started",
            {
                "train_files": list(config.train_files),
                "eval_files": list(config.eval_files),
                "epochs": config.epochs,
                "batch_size": config.batch_size,
            },
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
            self._status["history"] = list(self._history)

    def _append_log(self, line: str) -> None:
        with self._lock:
            log = self._status.get("log", [])
            log.append(line)
            if len(log) > 400:
                log = log[-400:]
            self._status["log"] = log
            self._status["history"] = list(self._history)

    def _record_event(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        event = {
            "ts": time.time(),
            "level": level,
            "message": message,
        }
        if context:
            event["context"] = context
        self._events.append(event)
        self._events = self._events[-128:]
        with self._lock:
            snapshot = dict(self._status.get("snapshot") or {})
            snapshot_events = list(snapshot.get("events", []))
            snapshot_events.append(event)
            snapshot["events"] = snapshot_events[-128:]
            self._status["snapshot"] = snapshot
            self._status["events"] = list(self._events)
        self._append_log(f"[{level.upper()}] {message}")

    def _update_snapshot(self, **fields: Any) -> None:
        with self._lock:
            snapshot = dict(self._status.get("snapshot") or {})
            snapshot.update(fields)
            self._status["snapshot"] = snapshot

    def _emit_trace_events(self, trace: Optional[List[Dict[str, Any]]], *, prefix: str = "") -> None:
        if not trace:
            return
        for entry in trace[:32]:
            stage = entry.get("stage", "trace")
            level = entry.get("level") or ("error" if entry.get("status") == "error" else "info")
            ctx = {k: v for k, v in entry.items() if k not in {"stage", "level"}}
            name = f"{prefix}_{stage}" if prefix else stage
            self._record_event(level, name, ctx or None)

    def _should_pause_pipeline(self, pipeline: Optional["TrainingPipeline"] = None) -> bool:
        flag = None
        try:
            flag = self.db.get_control_flag("production_manager_active")
        except Exception:
            flag = None
        if self._flag_truthy(flag):
            return True
        # When production manager isn't running, avoid pausing the pipeline to keep lab jobs responsive.
        return False

    @staticmethod
    def _flag_truthy(flag: Optional[Any]) -> bool:
        if flag is None:
            return False
        if isinstance(flag, (int, float)):
            return bool(flag)
        return str(flag).strip().lower() in {"1", "true", "yes", "on", "running"}

    def _record_history(self, success: bool) -> None:
        with self._lock:
            entry = {
                "started_at": self._status.get("started_at"),
                "finished_at": self._status.get("finished_at") or time.time(),
                "status": "success" if success else "failure",
                "message": self._status.get("message"),
                "error": self._status.get("error"),
                "job_type": self._status.get("job_type", "model_lab"),
                "train_files": (self._status.get("result") or {}).get("train", {}).get("info", {}).get("files"),
                "eval_files": (self._status.get("result") or {}).get("evaluation", {}).get("files"),
                "result": self._status.get("result"),
                "log": list(self._status.get("log", [])),
            }
            self._history.append(entry)
            self._history = self._history[-50:]
            self._status["history"] = list(self._history)

    def _run_job(self, config: LabJobConfig) -> None:
        train_files = self._resolve_paths(config.train_files)
        eval_files = self._resolve_paths(config.eval_files)
        train_rel = [self._relative_path(path) for path in train_files]
        eval_rel = [self._relative_path(path) for path in eval_files]
        from trading.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(db=self.db)
        managed_pause = False
        original_flag = None
        should_pause = self._should_pause_pipeline(pipeline)
        self._update_snapshot(
            config={
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "train_files": train_rel,
                "eval_files": eval_rel,
            }
        )
        news_summary: Optional[Dict[str, Any]] = None
        lab_model: Optional[Any] = None
        train_summary: Optional[Dict[str, Any]] = None
        eval_summary: Optional[Dict[str, Any]] = None
        error_trace: Optional[str] = None
        try:
            if should_pause:
                original_flag = self.db.get_control_flag("pipeline_pause")
                if not original_flag or str(original_flag).strip().lower() not in {"1", "true", "yes", "paused"}:
                    self.db.set_control_flag("pipeline_pause", "1")
                    managed_pause = True
                self._update_status(progress=0.05, message="Pausing training pipeline")
                time.sleep(0.5)
                self._record_event(
                    "info",
                    "pipeline_pause_engaged",
                    {
                        "managed_pause": managed_pause,
                        "existing_flag": original_flag,
                    },
                )
            else:
                self._update_status(progress=0.05, message="Pipeline idle; proceeding without pause")
                self._record_event(
                    "info",
                    "pipeline_pause_skipped",
                    {"reason": "production_manager_inactive"},
                )

            news_targets = sorted({*train_rel, *eval_rel})
            if news_targets:
                self._update_status(progress=0.18, message="Collecting market context")
                self._record_event(
                    "info",
                    "collecting_news_context",
                    {"datasets": len(news_targets), "files": news_targets},
                )
                try:
                    def _news_progress(entry: Dict[str, Any]) -> None:
                        stage = entry.get("stage", "trace")
                        level = entry.get("level") or "info"
                        context = {k: v for k, v in entry.items() if k not in {"stage", "level"}}
                        self._record_event(level, f"news_{stage}", context or None)

                    news_summary = collect_news_for_files(
                        news_targets,
                        db=self.db,
                        hours_before=24,
                        hours_after=24,
                        cache_ttl_sec=2 * 3600,
                        progress_cb=_news_progress,
                    )
                    total_news = len(news_summary.get("items", []))
                    source_span = len(news_summary.get("sources", []))
                    self._record_event(
                        "info",
                        "news_context_ready",
                        {
                            "articles": total_news,
                            "sources": source_span,
                            "symbols": news_summary.get("symbols") or [],
                        },
                    )
                    self._emit_trace_events(news_summary.get("trace"), prefix="news")
                    items = news_summary.get("items") or []
                    if items:
                        sample = items[: min(2, len(items))]
                        for idx, entry in enumerate(sample):
                            self._record_event(
                                "info",
                                f"news_sample_{idx+1}",
                                {
                                    "title": (entry.get("title") or "")[:160],
                                    "source": entry.get("source") or entry.get("origin"),
                                    "sentiment": entry.get("sentiment"),
                                    "timestamp": entry.get("datetime") or entry.get("timestamp"),
                                },
                            )
                    else:
                        self._record_event("info", "news_sample_none", {})
                    self._update_status(progress=0.2, message="Market context captured")
                    self._update_snapshot(news=news_summary)
                except Exception as news_exc:
                    self._record_event(
                        "error",
                        "news_context_failed",
                        {"error": str(news_exc)},
                    )
                    self._update_status(progress=0.2, message="Market context skipped")

            if train_files:
                self._update_status(progress=0.15, message="Building training dataset")
                self._record_event(
                    "info",
                    "building_training_dataset",
                    {"files": train_rel, "count": len(train_files)},
                )
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
                self._record_event(
                    "info",
                    "training_completed",
                    {"metrics": train_metrics, "samples": train_info.get("samples")},
                )
            else:
                lab_model = pipeline.ensure_active_model()
                self._update_status(progress=0.25, message="Using active model weights")
                self._record_event(
                    "info",
                    "training_skipped",
                    {"reason": "no_training_files"},
                )

            if eval_files:
                if lab_model is None:
                    lab_model = pipeline.ensure_active_model()
                self._update_status(progress=0.7, message="Running evaluation")
                self._record_event(
                    "info",
                    "evaluation_started",
                    {"files": eval_rel, "count": len(eval_files)},
                )
                eval_metrics = pipeline.lab_evaluate_on_files(
                    lab_model,
                    eval_files,
                    batch_size=max(8, config.batch_size),
                )
                eval_summary = {"metrics": eval_metrics, "files": eval_rel}
                self._update_status(progress=0.95, message="Evaluation complete")
                self._record_event(
                    "info",
                    "evaluation_completed",
                    {"metrics": eval_metrics},
                )

            self._update_status(
                progress=1.0,
                message="Model lab run complete",
                running=False,
                result={
                    "train": train_summary,
                    "evaluation": eval_summary,
                    "train_files": train_rel,
                    "eval_files": eval_rel,
                    "news": news_summary,
                },
                finished_at=time.time(),
            )
            self._update_snapshot(result=self._status.get("result"))
            self._record_event(
                "info",
                "job_completed",
                {
                    "train_metrics": train_summary.get("metrics") if train_summary else None,
                    "eval_metrics": eval_summary.get("metrics") if eval_summary else None,
                },
            )
            self._record_history(True)
        except Exception as exc:  # pragma: no cover - defensive
            error_trace = traceback.format_exc()
            self._update_status(
                running=False,
                progress=1.0,
                message="Model lab run failed",
                error=str(exc),
                error_detail=error_trace,
                finished_at=time.time(),
            )
            self._update_snapshot(error={"message": str(exc), "trace": error_trace})
            self._record_event(
                "error",
                "job_failed",
                {
                    "error": str(exc),
                },
            )
            self._record_history(False)
        finally:
            if managed_pause:
                try:
                    self.db.clear_control_flag("pipeline_pause")
                except Exception:
                    pass
            if original_flag and not managed_pause:
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
