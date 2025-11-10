from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any, Dict, Optional, Sequence

from trading.pipeline import TrainingPipeline
from trading.selector import GhostTradingSupervisor
from trading.metrics import MetricStage, FeedbackSeverity
from services.background_workers import TokenDownloadSupervisor
from services.task_orchestrator import ParallelTaskManager
from services.logging_utils import log_message
from services.idle_work import IdleWorkManager
from services.heartbeat import HeartbeatFile


class ProductionManager:
    def __init__(self) -> None:
        self.pipeline = TrainingPipeline()
        self.supervisor = GhostTradingSupervisor(pipeline=self.pipeline)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()
        self._download_supervisor: Optional[TokenDownloadSupervisor] = TokenDownloadSupervisor(db=self.pipeline.db)
        self.task_manager = ParallelTaskManager(system_profile=self.pipeline.system_profile)
        self._cycle_thread: Optional[threading.Thread] = None
        self._cycle_interval = float(os.getenv("PRODUCTION_CYCLE_INTERVAL", "45"))
        self._active_flag_key = "production_manager_active"
        self.idle_worker = IdleWorkManager(db=self.pipeline.db)
        self.heartbeat = HeartbeatFile(label="production_manager")
        self._startup_prewarm: Dict[str, Any] = {}
        self._startup_prewarm_reported = False

    def start(self) -> None:
        if self.is_running:
            log_message("production", "manager already running.")
            self._set_active_flag(True)
            return
        self._stop.clear()
        self._startup_prewarm = self._prewarm_pipeline()
        self._startup_prewarm_reported = False
        self.supervisor.build()
        try:
            self._loop_thread = threading.Thread(target=self._run_supervisor_loop, daemon=True)
            self._loop_thread.start()
            self.task_manager.start()
            self._cycle_thread = threading.Thread(target=self._cycle_loop, daemon=True)
            self._cycle_thread.start()
            self._set_active_flag(True)
            self.heartbeat.update(
                "running",
                metadata={
                    "iteration": self.pipeline.iteration,
                    "cycle_interval": self._cycle_interval,
                    "startup_prewarm": self._startup_prewarm,
                },
            )
            self._startup_prewarm_reported = True
            log_message("production", "manager started.")
        except Exception:
            self._set_active_flag(False)
            self.heartbeat.update("error", metadata={"reason": "startup_failed"})
            raise

    def stop(self, timeout: float = 15.0) -> None:
        if not self.is_running:
            log_message("production", "manager is not running.")
            return
        self._stop.set()
        if self._loop and self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.supervisor.stop(), self._loop)
            try:
                fut.result(timeout=timeout)
            except Exception as exc:
                log_message("production", f"supervisor stop error: {exc}", severity="error")
        if self._loop_thread:
            self._loop_thread.join(timeout=timeout)
        if self._cycle_thread:
            self._cycle_thread.join(timeout=timeout)
        self.task_manager.stop()
        self._loop = None
        self._loop_thread = None
        self._cycle_thread = None
        self._set_active_flag(False)
        self.heartbeat.update("stopped", metadata={"iteration": self.pipeline.iteration})
        self.heartbeat.clear()
        log_message("production", "manager stopped.")

    @property
    def is_running(self) -> bool:
        return self._cycle_thread is not None and self._cycle_thread.is_alive()

    def _run_supervisor_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.supervisor.start())
        except Exception as exc:
            log_message("production", f"supervisor loop error: {exc}", severity="error")
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None

    def _set_active_flag(self, active: bool) -> None:
        try:
            if active:
                self.pipeline.db.set_control_flag(self._active_flag_key, "1")
            else:
                self.pipeline.db.clear_control_flag(self._active_flag_key)
        except Exception as exc:
            log_message("production", f"unable to update active flag: {exc}", severity="warning")

    def _cycle_loop(self) -> None:
        while not self._stop.is_set():
            cycle_id = str(int(time.time()))
            focus_assets, _ = self.pipeline.ghost_focus_assets()
            readiness = self.pipeline.live_readiness_report()
            bias = self._safe_horizon_bias()
            metadata = {"focus_assets": focus_assets, "readiness": readiness, "horizon_bias": bias}
            if not self._startup_prewarm_reported and self._startup_prewarm:
                metadata["startup_prewarm"] = self._startup_prewarm
                self._startup_prewarm_reported = True
            self.heartbeat.update(
                "running",
                metadata={
                    "cycle": cycle_id,
                    "focus_assets": focus_assets,
                    "queue_depth": self.task_manager.pending_tasks,
                    "live_ready": readiness.get("ready") if readiness else False,
                    "live_precision": (readiness or {}).get("precision"),
                    "live_samples": (readiness or {}).get("samples"),
                    "horizon_bias": bias,
                },
            )
            self.task_manager.submit("data_ingest", self._task_data_ingest, cycle_id=cycle_id)
            self.task_manager.submit(
                "news_enrichment",
                self._task_news_enrichment,
                cycle_id=cycle_id,
                kwargs={"focus_assets": focus_assets},
                metadata=metadata,
            )
            self.task_manager.submit(
                "dataset_warmup",
                self._task_dataset_warmup,
                cycle_id=cycle_id,
                kwargs={"focus_assets": focus_assets},
                metadata=metadata,
            )
            self.task_manager.submit(
                "candidate_training",
                self._task_candidate_training,
                cycle_id=cycle_id,
            )
            self.task_manager.submit(
                "ghost_metrics",
                self._task_ghost_metrics,
                cycle_id=cycle_id,
            )
            self.task_manager.submit(
                "scheduler_refresh",
                self._task_scheduler_refresh,
                cycle_id=cycle_id,
            )
            self.task_manager.submit(
                "telemetry_flush",
                self._task_telemetry_flush,
                cycle_id=cycle_id,
            )
            self.task_manager.submit(
                "background_refresh",
                self._task_background_refresh,
                cycle_id=cycle_id,
            )
            if self._stop.wait(self._cycle_interval):
                break

    # ------------------------------------------------------------------
    # Parallel task implementations
    # ------------------------------------------------------------------

    def _task_data_ingest(self) -> None:
        if self._download_supervisor:
            self._download_supervisor.run_cycle()

    def _task_news_enrichment(self, focus_assets: Optional[Sequence[str]] = None) -> None:
        self.pipeline.reinforce_news_cache(focus_assets)

    def _task_dataset_warmup(self, focus_assets: Optional[Sequence[str]] = None) -> None:
        self.pipeline.warm_dataset_cache(focus_assets=focus_assets, oversample=False)

    def _task_candidate_training(self) -> None:
        result = self.pipeline.train_candidate()
        status = result.get("status") if isinstance(result, dict) else None
        if status and status not in {"trained", "skipped"}:
            log_message("production", f"training status: {status}", severity="warning")

    def _task_ghost_metrics(self) -> None:
        trades = self.pipeline.metrics.ghost_trade_snapshot(limit=500, lookback_sec=self.pipeline.focus_lookback_sec)
        aggregate = self.pipeline.metrics.aggregate_trade_metrics(trades)
        self.pipeline.metrics.record(
            MetricStage.GHOST_TRADING,
            aggregate,
            category="orchestrator_snapshot",
        )

    def _task_scheduler_refresh(self) -> None:
        # Placeholder hook for bus/passenger scheduling or pair rotation.
        self.pipeline.metrics.feedback(
            "scheduler",
            severity=FeedbackSeverity.INFO,
            label="cycle_tick",
            details={"ts": time.time()},
        )

    def _task_telemetry_flush(self) -> None:
        summary = {
            "active_accuracy": self.pipeline.active_accuracy,
            "decision_threshold": self.pipeline.decision_threshold,
            "temperature_scale": self.pipeline.temperature_scale,
            "iteration": self.pipeline.iteration,
        }
        self.pipeline.metrics.record(
            MetricStage.PIPELINE,
            summary,
            category="telemetry_flush",
        )

    def _task_background_refresh(self) -> None:
        worked = self.idle_worker.run_next_job()
        if not worked:
            log_message("idle-work", "no background job ready this cycle", severity="info")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prewarm_pipeline(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        try:
            focus_assets, _ = self.pipeline.ghost_focus_assets()
            dataset_ready = bool(self.pipeline.warm_dataset_cache(focus_assets=focus_assets or None, oversample=False))
            news_ready = bool(self.pipeline.reinforce_news_cache(focus_assets=focus_assets or None))
            payload = {
                "focus_assets": focus_assets[:8],
                "dataset_ready": dataset_ready,
                "news_ready": news_ready,
            }
            if not dataset_ready or not news_ready:
                payload["note"] = "warmup_incomplete"
        except Exception as exc:
            log_message("production", f"pipeline prewarm failed: {exc}", severity="warning")
            payload = {"error": str(exc)}
        return payload

    def _safe_horizon_bias(self) -> Dict[str, float]:
        try:
            bias = self.pipeline.horizon_bias()
        except Exception:
            return {}
        result: Dict[str, float] = {}
        for key, value in bias.items():
            try:
                result[key] = float(value)
            except Exception:
                continue
        return result


if __name__ == "__main__":
    try:
        from services.process_names import set_process_name
    except Exception:  # pragma: no cover
        def set_process_name(_: str) -> None:
            return
    set_process_name("Production Manager")
    manager = ProductionManager()
    try:
        manager.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()
