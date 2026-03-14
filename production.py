from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

from trading.metrics import MetricStage, FeedbackSeverity
from services.background_workers import TokenDownloadSupervisor
from services.task_orchestrator import ParallelTaskManager
from services.logging_utils import log_message
from services.idle_work import IdleWorkManager
from services.heartbeat import HeartbeatFile
from services.env_loader import EnvLoader
from services.secure_settings import build_process_env
from services.stable_bank_notify import notifier as _stable_bank_notifier
from services.multi_wallet import multi_wallet_manager as _multi_wallet_mgr
from services.resource_governor import governor as _governor, Priority as _Priority
from services.delegation_client import DelegationClient

if TYPE_CHECKING:
    from trading.pipeline import TrainingPipeline
    from trading.selector import GhostTradingSupervisor


@dataclass(frozen=True)
class ScheduledTask:
    name: str
    func: Callable[..., Any]
    kwargs: Optional[Dict[str, Any]] = None


class ProductionManager:
    _env_loaded = False

    def __init__(self) -> None:
        # Delay heavy ML imports until absolutely necessary so lightweight CI
        # (without TensorFlow) can still import this module for unit tests.
        from trading.pipeline import TrainingPipeline
        from trading.selector import GhostTradingSupervisor

        self._ensure_secure_env()
        self.pipeline = TrainingPipeline()
        self.supervisor = GhostTradingSupervisor(pipeline=self.pipeline)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()
        if os.getenv("DISABLE_DATA_INGEST", "0").lower() in {"1", "true", "yes", "on"}:
            self._download_supervisor = None
        else:
            self._download_supervisor = TokenDownloadSupervisor(db=self.pipeline.db)
        self.task_manager = ParallelTaskManager(system_profile=self.pipeline.system_profile)
        self._cycle_thread: Optional[threading.Thread] = None
        self._cycle_interval = float(os.getenv("PRODUCTION_CYCLE_INTERVAL", "45"))
        forced_pending = os.getenv("PRODUCTION_MAX_PENDING_FORCE")
        base_pending = int(os.getenv("PRODUCTION_MAX_PENDING", "64"))
        if forced_pending is not None:
            self._max_pending = max(1, int(forced_pending))
        else:
            self._max_pending = base_pending
            if self.pipeline.system_profile.memory_pressure:
                # Allow a slightly higher queue to avoid constant skips, but cap for stability unless force is set.
                self._max_pending = min(self._max_pending, 14)
                self._cycle_interval = max(self._cycle_interval, 70.0)
            elif self.pipeline.system_profile.is_low_power:
                self._max_pending = min(self._max_pending, 16)
                self._cycle_interval = max(self._cycle_interval, 55.0)
            else:
                self._max_pending = min(self._max_pending, 24)
        self._ingest_cadence = int(os.getenv("PRODUCTION_INGEST_CADENCE", "3"))
        self._news_cadence = int(os.getenv("PRODUCTION_NEWS_CADENCE", "2"))
        self._training_cadence = int(os.getenv("PRODUCTION_TRAINING_CADENCE", "2"))
        self._background_floor = int(os.getenv("PRODUCTION_BACKGROUND_FLOOR", "4"))
        self._min_samples_for_live = int(os.getenv("PRODUCTION_MIN_SAMPLES_FOR_LIVE", "64"))
        if os.getenv("TRAIN_LIGHTWEIGHT", "0").lower() in {"1", "true", "yes", "on"}:
            # On lightweight profiles, allow live gating sooner to reach ghost mode faster.
            self._training_cadence = 1
            self._min_samples_for_live = min(self._min_samples_for_live, 32)
        self._active_flag_key = "production_manager_active"
        self.idle_worker = IdleWorkManager(db=self.pipeline.db)
        self.heartbeat = HeartbeatFile(label="production_manager")
        self._startup_prewarm: Dict[str, Any] = {}
        self._startup_prewarm_reported = False
        self._task_directives: Dict[str, bool] = {}
        self._cycle_index = 0
        self._training_pause_logged = False
        self._heavy_backlog_threshold = max(2, self._max_pending // 2)
        log_message(
            "production",
            "orchestrator configured",
            details={
                "download_enabled": bool(self._download_supervisor),
                "max_pending": self._max_pending,
                "cycle_interval": self._cycle_interval,
                "heavy_backlog_threshold": self._heavy_backlog_threshold,
                "system_profile": {
                    "cpu": getattr(self.pipeline.system_profile, "cpu_count", None),
                    "memory_gb": getattr(self.pipeline.system_profile, "total_memory_gb", None),
                    "low_power": getattr(self.pipeline.system_profile, "is_low_power", None),
                    "memory_pressure": getattr(self.pipeline.system_profile, "memory_pressure", None),
                },
            },
        )
        self._backlog_strikes = 0
        self._task_backoff_until: Dict[str, float] = {}
        self._task_threads: Dict[str, threading.Thread] = {}
        self._task_thread_start: Dict[str, float] = {}
        # Delegation — offload tasks to remote Revenir service hosts
        self._delegation_enabled = os.getenv("DELEGATION_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
        self._delegation_client: Optional[DelegationClient] = None
        # Task types eligible for delegation (heavy or parallelisable)
        self._delegatable_tasks = {
            "data_ingest", "news_enrichment", "dataset_warmup",
            "candidate_training", "background_refresh",
        }

    def start(self) -> None:
        if self.is_running:
            log_message("production", "manager already running.")
            self._set_active_flag(True)
            return
        self._stop.clear()
        _governor.start()
        # Start delegation client if enabled and hosts exist
        if self._delegation_enabled:
            try:
                env = {k: os.environ.get(k, "") for k in [
                    "ALCHEMY_API_KEY", "ANKR_API_KEY", "THEGRAPH_API_KEY",
                    "CRYPTOPANIC_API_KEY",
                ]}
                self._delegation_client = DelegationClient(secure_env=env)
                self._delegation_client.on_result(self._handle_delegation_result)
                self._delegation_client.start()
                log_message("production", "delegation client started")
            except Exception as exc:
                log_message("production", f"delegation client failed to start: {exc}", severity="warning")
                self._delegation_client = None
        self._wallet_bootstrap = self._try_wallet_bootstrap()
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
        if self._delegation_client:
            self._delegation_client.stop()
            self._delegation_client = None
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
            if not self._loop_thread or not self._loop_thread.is_alive():
                self.heartbeat.update("error", metadata={"reason": "ghost_loop_stopped"})
                log_message("production", "ghost supervisor loop stopped; ending cycle loop", severity="error")
                self._stop.set()
                break
            _governor.wait_if_pressured(label="production_cycle", max_wait=60.0, priority=_Priority.HIGH)
            cycle_id = str(int(time.time()))
            focus_assets, _ = self.pipeline.ghost_focus_assets()
            readiness = self.pipeline.live_readiness_report()
            bias = self._safe_horizon_bias()
            deficit = self._safe_horizon_deficit()
            metadata = {
                "focus_assets": focus_assets,
                "readiness": readiness,
                "horizon_bias": bias,
                "horizon_deficit": deficit,
            }
            if not self._startup_prewarm_reported and self._startup_prewarm:
                metadata["startup_prewarm"] = self._startup_prewarm
                self._startup_prewarm_reported = True
            pending = self.task_manager.pending_tasks
            active = self.task_manager.active_tasks
            backlog = self.task_manager.workload_depth
            heavy_backlog = backlog >= self._heavy_backlog_threshold
            self._task_directives = self._compute_task_directives(readiness, backlog)
            self.heartbeat.update(
                "running",
                metadata={
                    "cycle": cycle_id,
                    "focus_assets": focus_assets,
                    "queue_depth": pending,
                    "workload_depth": backlog,
                    "active_tasks": active,
                    "live_ready": readiness.get("ready") if readiness else False,
                    "live_precision": (readiness or {}).get("precision"),
                    "live_samples": (readiness or {}).get("samples"),
                    "horizon_bias": bias,
                    "horizon_deficit": deficit,
                    "max_concurrent": getattr(self.task_manager, "max_concurrent", None),
                    "directives": self._task_directives,
                    "heavy_backlog": heavy_backlog,
                    "max_pending": self._max_pending,
                    "heavy_backlog_threshold": self._heavy_backlog_threshold,
                },
            )
            log_message(
                "production",
                f"cycle {cycle_id} status",
                details={
                    "pending": pending,
                    "active": active,
                    "backlog": backlog,
                    "max_pending": self._max_pending,
                    "heavy_backlog": heavy_backlog,
                    "heavy_backlog_threshold": self._heavy_backlog_threshold,
                },
            )
            # ── Stable bank threshold notification ──────────────────
            try:
                from db import get_db
                _db = get_db()
                _state = _db.load_state() or {}
                _ghost = _state.get("ghost_trading") or {}
                _bank_usd = float(_ghost.get("stable_bank", 0.0))
                if _bank_usd > 0:
                    _stable_bank_notifier.check(_bank_usd)
            except Exception:
                pass  # notification is best-effort, never break the cycle
            # ── Multi-wallet auto-creation ────────────────────────────
            try:
                if _multi_wallet_mgr.enabled():
                    _mw_wallets = _multi_wallet_mgr.load_wallets()
                    if _mw_wallets:
                        _mw_balances: Dict[str, float] = {}
                        for _mw in _mw_wallets:
                            _mw_balances[_mw.address.lower()] = float(
                                _ghost.get("stable_bank", 0.0)
                            ) if _mw.index == 0 else 0.0
                        _multi_wallet_mgr.check_and_create(_mw_balances)
            except Exception:
                pass  # multi-wallet is best-effort
            # ────────────────────────────────────────────────────────
            # Dynamically reduce max pending under pressure
            effective_max_pending = self._max_pending
            if _governor.should_throttle(_Priority.HIGH):
                effective_max_pending = max(2, self._max_pending // 3)
            elif _governor.should_throttle(_Priority.NORMAL):
                effective_max_pending = max(4, self._max_pending // 2)
            if backlog >= effective_max_pending:
                self._backlog_strikes += 1
                log_message(
                    "production",
                    f"skipping cycle {cycle_id}: backlog {backlog} >= {self._max_pending}",
                    severity="warning",
                )
                if self._backlog_strikes >= 3 and hasattr(self.task_manager, "reset_queues"):
                    try:
                        self.task_manager.reset_queues()  # type: ignore[attr-defined]
                        log_message("production", "backlog flushed after repeated skips", severity="warning")
                        self._backlog_strikes = 0
                    except Exception:
                        pass
                if self._stop.wait(self._cycle_interval):
                    break
                continue
            throttled_this_cycle = False
            if heavy_backlog:
                # When backlog is heavy, focus on lightweight maintenance only.
                self._task_directives.update({
                    "ingest": False,
                    "news": False,
                    "dataset": False,
                    "training": False,
                    "background": False,
                })
                self._backlog_strikes += 1
            for scheduled in self._plan_cycle_tasks(focus_assets=focus_assets):
                if heavy_backlog and scheduled.name in {
                    "data_ingest",
                    "news_enrichment",
                    "dataset_warmup",
                    "candidate_training",
                    "background_refresh",
                }:
                    throttled_this_cycle = True
                    self.task_manager.mark_skipped(
                        scheduled.name,
                        cycle_id=cycle_id,
                        reason="skipped_backlog",
                        metadata=metadata,
                    )
                    continue
                self.task_manager.submit(
                    scheduled.name,
                    scheduled.func,
                    cycle_id=cycle_id,
                    kwargs=scheduled.kwargs,
                    metadata=metadata,
                )
            if throttled_this_cycle:
                log_message(
                    "production",
                    f"cycle {cycle_id} throttled due to backlog {backlog}",
                    severity="info",
                    details={"heavy_backlog_threshold": self._heavy_backlog_threshold},
                )
            # ── Delegation offload ──────────────────────────────────
            # When backlog is building up or resources are pressured, try to
            # offload eligible tasks to remote delegation hosts.
            self._try_delegate_tasks(
                cycle_id=cycle_id,
                focus_assets=focus_assets,
                backlog=backlog,
                heavy_backlog=heavy_backlog,
            )
            # ────────────────────────────────────────────────────────
            if self._stop.wait(self._cycle_interval):
                break
            self._backlog_strikes = max(0, self._backlog_strikes - 1)

    # ------------------------------------------------------------------
    # Cycle planning
    # ------------------------------------------------------------------

    def _compute_task_directives(self, readiness: Optional[Dict[str, Any]], backlog: int) -> Dict[str, bool]:
        self._cycle_index += 1
        samples = int((readiness or {}).get("samples") or 0)
        data_starved = not (readiness or {}).get("ready", False) or samples < self._min_samples_for_live
        high_backlog = backlog >= max(2, self._max_pending // 2)
        directives = {
            "ingest": data_starved or (self._cycle_index % self._ingest_cadence == 0),
            "news": data_starved or (self._cycle_index % self._news_cadence == 0),
            "dataset": True,
            "training": (not data_starved and not high_backlog and (self._cycle_index % self._training_cadence == 0)),
            "background": not data_starved and backlog < max(self._background_floor, self._max_pending // 2),
        }
        if data_starved and high_backlog:
            directives["training"] = False
            directives["background"] = False
        return directives

    def _plan_cycle_tasks(self, focus_assets: Optional[Sequence[str]]) -> List[ScheduledTask]:
        directives = self._task_directives or {}
        tasks: List[ScheduledTask] = [
            ScheduledTask("data_ingest", self._task_data_ingest),
            ScheduledTask("news_enrichment", self._task_news_enrichment, {"focus_assets": focus_assets}),
            ScheduledTask("dataset_warmup", self._task_dataset_warmup, {"focus_assets": focus_assets}),
            ScheduledTask("candidate_training", self._task_candidate_training),
            ScheduledTask("ghost_metrics", self._task_ghost_metrics),
            ScheduledTask("scheduler_refresh", self._task_scheduler_refresh),
            ScheduledTask("telemetry_flush", self._task_telemetry_flush),
        ]
        if directives.get("background", False):
            tasks.append(ScheduledTask("background_refresh", self._task_background_refresh))
        return tasks

    # ------------------------------------------------------------------
    # Parallel task implementations
    # ------------------------------------------------------------------

    def _task_data_ingest(self) -> None:
        if not self._task_directives.get("ingest", True):
            return
        if self._download_supervisor:
            self._download_supervisor.run_cycle()

    def _task_news_enrichment(self, focus_assets: Optional[Sequence[str]] = None) -> None:
        if not self._task_directives.get("news", True):
            return
        timeout = float(os.getenv("NEWS_ENRICH_TIMEOUT", "120"))
        backoff = float(os.getenv("NEWS_ENRICH_BACKOFF_SEC", "300"))
        # Keep the news task under the orchestrator timeout so we do not strand threads.
        pipeline_budget = getattr(self.pipeline, "_news_enrich_budget", timeout)
        max_budget = min(timeout * 0.8, timeout - 5.0) if timeout > 10 else timeout * 0.7
        max_budget = max(5.0, max_budget)
        budget_cap = max(5.0, min(pipeline_budget, max_budget))
        deadline = time.time() + budget_cap
        if self._backoff_active("news_enrichment"):
            resume_in = max(0.0, self._task_backoff_until.get("news_enrichment", 0.0) - time.time())
            log_message(
                "production",
                "news_enrichment in backoff; skipping this cycle",
                severity="info",
                details={"resume_in": round(resume_in, 2)},
            )
            return
        self._run_with_timeout(
            lambda: self.pipeline.reinforce_news_cache(focus_assets, deadline=deadline),
            timeout=timeout,
            label="news_enrichment",
            backoff_on_timeout=backoff,
        )

    def _task_dataset_warmup(self, focus_assets: Optional[Sequence[str]] = None) -> None:
        if not self._task_directives.get("dataset", True):
            return
        self.pipeline.warm_dataset_cache(focus_assets=focus_assets, oversample=False)

    def _task_candidate_training(self) -> None:
        if not self._task_directives.get("training", True):
            if not self._training_pause_logged:
                log_message("production", "candidate training paused for resource budget", severity="info")
                self._training_pause_logged = True
            return
        self._training_pause_logged = False
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
        try:
            self.pipeline.metrics.feedback(
                "scheduler",
                severity=FeedbackSeverity.INFO,
                label="cycle_tick",
                details={"ts": time.time()},
            )
        except Exception as exc:
            log_message(
                "production",
                f"scheduler feedback failed: {exc}",
                severity="warning",
            )
        try:
            # Persist the latest readiness snapshot so ghost→live gating reflects current market data.
            readiness = self.pipeline.live_readiness_report()
            if readiness:
                self.pipeline._persist_live_readiness_snapshot()  # type: ignore[attr-defined]
        except Exception as exc:
            log_message("production", f"readiness snapshot failed: {exc}", severity="warning")

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
        if not self._task_directives.get("background", False):
            return
        worked = self.idle_worker.run_next_job()
        if not worked:
            log_message("idle-work", "no background job ready this cycle", severity="info")

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def _try_delegate_tasks(
        self,
        cycle_id: str,
        focus_assets: Optional[Sequence[str]],
        backlog: int,
        heavy_backlog: bool,
    ) -> None:
        """
        Greedily fill remote hosts with delegatable tasks. Always attempts
        delegation when there are pending tasks and remote hosts are reachable —
        no longer gates on heavy_backlog.
        """
        if not self._delegation_client:
            return

        _TASK_TO_DIRECTIVE = {
            "data_ingest": "ingest",
            "news_enrichment": "news",
            "dataset_warmup": "dataset",
            "candidate_training": "training",
            "background_refresh": "background",
            "ghost_trading": "ghost_trading",
            "ghost_metrics": "ghost_metrics",
        }

        # Build a priority queue of tasks to delegate
        # Priority: candidate_training > data_ingest > ghost_trading > ghost_metrics > rest
        _TASK_PRIORITY = {
            "candidate_training": 0,
            "data_ingest": 1,
            "ghost_trading": 2,
            "ghost_metrics": 3,
            "dataset_warmup": 4,
            "news_enrichment": 5,
            "background_refresh": 6,
        }
        eligible_tasks = []
        for task_type in self._delegatable_tasks:
            directive_key = _TASK_TO_DIRECTIVE.get(task_type, task_type)
            if not self._task_directives.get(directive_key, False):
                continue
            eligible_tasks.append(task_type)
        eligible_tasks.sort(key=lambda t: _TASK_PRIORITY.get(t, 99))

        if not eligible_tasks:
            return

        # Query available capacity across all hosts
        try:
            hosts = self._delegation_client.get_available_hosts()
        except Exception:
            hosts = []
        if not hosts:
            return

        total_available = sum(h.get("headroom", 0) for h in hosts)
        if total_available <= 0:
            return

        # Greedy fill: keep dispatching until hosts are full or we run out of tasks
        delegated_count = 0
        slots_remaining = total_available
        task_idx = 0
        max_dispatch_per_cycle = max(total_available, 10)

        while slots_remaining > 0 and delegated_count < max_dispatch_per_cycle:
            if task_idx >= len(eligible_tasks):
                # Wrap around to dispatch more of the same types if slots remain
                # (e.g. multiple candidate_training tasks in parallel)
                if delegated_count == 0:
                    break  # No tasks were dispatched at all, stop
                task_idx = 0
                # Don't loop forever — only re-dispatch if there's real work
                if delegated_count >= len(eligible_tasks) * 2:
                    break

            task_type = eligible_tasks[task_idx]
            task_idx += 1

            payload = self._build_delegation_payload(task_type, focus_assets)
            try:
                tid = self._delegation_client.dispatch(task_type, payload)
                if tid:
                    delegated_count += 1
                    slots_remaining -= 1
                    log_message(
                        "production",
                        f"delegated {task_type} to remote host",
                        details={"task_id": tid[:8], "cycle": cycle_id},
                    )
                else:
                    # Host returned None — likely full, stop trying this round
                    slots_remaining = 0
            except Exception as exc:
                log_message(
                    "production",
                    f"delegation dispatch failed for {task_type}: {exc}",
                    severity="warning",
                )
                slots_remaining = 0  # Stop on error

        if delegated_count:
            log_message(
                "production",
                f"cycle {cycle_id}: delegated {delegated_count} tasks to remote hosts",
                details={"backlog": backlog, "total_available": total_available},
            )

    def _build_delegation_payload(
        self, task_type: str, focus_assets: Optional[Sequence[str]]
    ) -> Dict[str, Any]:
        """Build the payload dict sent to a delegation host for a given task type."""
        payload: Dict[str, Any] = {}
        if focus_assets:
            payload["focus_assets"] = list(focus_assets[:20])
        payload["iteration"] = self.pipeline.iteration

        if task_type == "data_ingest":
            payload["action"] = "download_cycle"
        elif task_type == "news_enrichment":
            payload["action"] = "reinforce_news"
        elif task_type == "dataset_warmup":
            payload["action"] = "warm_cache"
        elif task_type == "candidate_training":
            payload["action"] = "train_candidate"
        elif task_type == "background_refresh":
            payload["action"] = "idle_job"

        return payload

    # ------------------------------------------------------------------
    # Delegation result ingestion
    # ------------------------------------------------------------------

    def _handle_delegation_result(
        self, task_id: str, task_type: str, status: str, result: Optional[Dict[str, Any]]
    ) -> None:
        """Callback fired when a delegated task completes. Ingests results back
        into the local pipeline."""
        if status != "completed" or not result:
            log_message(
                "production",
                f"delegation result: {task_type} {status}",
                severity="warning" if status == "failed" else "info",
                details={"task_id": task_id[:8], "error": (result or {}).get("error")},
            )
            return

        try:
            handler = self._RESULT_HANDLERS.get(task_type)
            if handler:
                handler(self, task_id, result)
            else:
                log_message("production", f"no result handler for {task_type}", severity="debug")
        except Exception as exc:
            log_message(
                "production",
                f"result ingestion failed for {task_type}: {exc}",
                severity="warning",
                details={"task_id": task_id[:8]},
            )

    def _ingest_candidate_training(self, task_id: str, result: Dict[str, Any]) -> None:
        """Import a remotely-trained model candidate into the local pipeline."""
        if not result.get("trained"):
            return

        model_file = result.get("model_file")
        evaluation = result.get("evaluation", {})
        score = result.get("score", 0.0)
        params = result.get("params", {})

        log_message(
            "production",
            f"received trained model from delegation: score={score:.4f}",
            details={"task_id": task_id[:8], "params": params, "model_file": model_file},
        )

        # If a model file was returned, attempt to fetch it from the remote
        if model_file and self._delegation_client:
            try:
                self._fetch_result_file(task_id, model_file, "models")
            except Exception as exc:
                log_message("production", f"model file transfer failed: {exc}", severity="warning")

        # Record evaluation for the optimizer
        if evaluation and hasattr(self.pipeline, 'optimizer'):
            signals = result.get("signals", {})
            if signals:
                try:
                    self.pipeline.optimizer.update(
                        params,
                        score,
                        signals=signals,
                    )
                    log_message("production", "remote training result fed to optimizer")
                except Exception as exc:
                    log_message("production", f"optimizer update from remote failed: {exc}", severity="warning")

    def _ingest_data_ingest(self, task_id: str, result: Dict[str, Any]) -> None:
        """Merge remotely downloaded OHLCV data into local data store."""
        details = result.get("details", [])
        pairs_with_data = [d for d in details if d.get("records", 0) > 0]

        if pairs_with_data:
            log_message(
                "production",
                f"data ingest complete: {len(pairs_with_data)} pairs with data",
                details={"task_id": task_id[:8], "total": result.get("pairs_processed", 0)},
            )
            # Fetch result files from remote
            for d in pairs_with_data:
                fname = d.get("file")
                if fname and self._delegation_client:
                    try:
                        self._fetch_result_file(task_id, fname, "data/historical_ohlcv")
                    except Exception:
                        pass

    def _ingest_ghost_metrics(self, task_id: str, result: Dict[str, Any]) -> None:
        """Record ghost trading metrics from remote computation."""
        metrics = result.get("metrics", {})
        if not metrics:
            return

        log_message(
            "production",
            f"ghost metrics received: {metrics.get('total_trades', 0)} trades, "
            f"win_rate={metrics.get('win_rate', 0):.2%}",
            details={"task_id": task_id[:8], "metrics": metrics},
        )

        # Feed metrics to pipeline if available
        if hasattr(self.pipeline, 'metrics'):
            try:
                self.pipeline.metrics.record(
                    MetricStage.GHOST_TRADING,
                    {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                    category="delegation_aggregate",
                    meta={"task_id": task_id[:8], "source": "delegation"},
                )
            except Exception:
                pass

    def _ingest_ghost_trading(self, task_id: str, result: Dict[str, Any]) -> None:
        """Record ghost trading simulation results."""
        summary = result.get("summary", {})
        trades = result.get("trades", [])

        if not summary and not trades:
            return

        log_message(
            "production",
            f"ghost trading sim: {summary.get('total_trades', 0)} trades, "
            f"win_rate={summary.get('win_rate', 0):.2%}, "
            f"total_profit={summary.get('total_profit', 0):.4f}",
            details={"task_id": task_id[:8]},
        )

    def _ingest_background_refresh(self, task_id: str, result: Dict[str, Any]) -> None:
        """Handle background refresh results."""
        job_type = result.get("job_type", "unknown")
        log_message(
            "production",
            f"background refresh ({job_type}) complete",
            details={"task_id": task_id[:8], "result": result},
        )

    def _fetch_result_file(self, task_id: str, filename: str, dest_subdir: str) -> Optional[Path]:
        """Download a result file from the remote host that completed the task."""
        if not self._delegation_client:
            return None

        try:
            from web.delegation.models import DelegatedTask
            db_task = DelegatedTask.objects.filter(task_id=task_id).first()
            if not db_task or not db_task.host:
                return None

            host = db_task.host
            url_path = f"/tasks/{task_id}/files/{filename}"
            data = self._delegation_client._http_get(host.host, host.port, url_path, host.api_token)
            if data and isinstance(data, bytes):
                dest_dir = Path(dest_subdir)
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / filename
                dest.write_bytes(data)
                log_message("production", f"downloaded result file: {filename}")
                return dest
        except Exception as exc:
            log_message("production", f"file download failed: {exc}", severity="warning")
        return None

    _RESULT_HANDLERS = {
        "candidate_training": _ingest_candidate_training,
        "data_ingest": _ingest_data_ingest,
        "ghost_metrics": _ingest_ghost_metrics,
        "ghost_trading": _ingest_ghost_trading,
        "background_refresh": _ingest_background_refresh,
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _backoff_active(self, label: str, now: Optional[float] = None) -> bool:
        now = now or time.time()
        until = self._task_backoff_until.get(label, 0.0)
        if until and now < until:
            return True
        if until and now >= until:
            self._task_backoff_until.pop(label, None)
        return False

    def _run_with_timeout(
        self,
        func: Callable[[], Any],
        *,
        timeout: float,
        label: str,
        backoff_on_timeout: Optional[float] = None,
    ) -> bool:
        """
        Execute a callable with a soft timeout to avoid blocking the entire
        orchestrator (e.g. slow news feeds). Returns False on timeout or error.
        """
        if self._backoff_active(label):
            return False
        existing = self._task_threads.get(label)
        if existing and existing.is_alive():
            start_ts = self._task_thread_start.get(label, 0.0)
            age = time.time() - start_ts if start_ts else None
            stale_after = max(timeout * 2.0, timeout + (backoff_on_timeout or 0.0))
            if age and age > stale_after:
                log_message(
                    "production",
                    f"{label} thread stale after {age:.1f}s; detaching and retrying",
                    severity="warning",
                    details={"age_sec": round(age, 2), "stale_after": round(stale_after, 2)},
                )
                self._task_threads.pop(label, None)
                self._task_thread_start.pop(label, None)
            else:
                resume_in = None
                if backoff_on_timeout and backoff_on_timeout > 0:
                    resume_in = backoff_on_timeout
                    self._task_backoff_until[label] = time.time() + backoff_on_timeout
                log_message(
                    "production",
                    f"{label} still running; skipping re-entry",
                    severity="warning",
                    details={
                        "age_sec": round(age or 0.0, 2),
                        "stale_after": round(stale_after, 2),
                        "resume_in": round(resume_in or 0.0, 2) if resume_in else None,
                    },
                )
                return False
        result = {"ok": False}
        error: Dict[str, Any] = {}

        def _target() -> None:
            try:
                outcome = func()
                result["ok"] = bool(outcome)
            except Exception as exc:  # pragma: no cover - defensive guard
                error["exc"] = exc

        thread = threading.Thread(target=_target, name=f"{label}-runner", daemon=True)
        self._task_threads[label] = thread
        self._task_thread_start[label] = time.time()
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            log_message("production", f"{label} exceeded {timeout:.0f}s; continuing without result", severity="warning")
            if backoff_on_timeout and backoff_on_timeout > 0:
                self._task_backoff_until[label] = time.time() + backoff_on_timeout
                log_message(
                    "production",
                    f"{label} entering backoff after timeout",
                    severity="info",
                    details={"resume_in": round(backoff_on_timeout, 2)},
                )
            # Detach stale thread so future cycles are not blocked on the same handle.
            self._task_threads.pop(label, None)
            self._task_thread_start.pop(label, None)
            return False
        self._task_threads.pop(label, None)
        self._task_thread_start.pop(label, None)
        if error.get("exc"):
            log_message("production", f"{label} failed: {error['exc']}", severity="warning")
            return False
        return result["ok"]

    def _try_wallet_bootstrap(self) -> Dict[str, Any]:
        """
        Auto-discover wallet holdings and generate tradeable pairs on startup.
        Runs before supervisor.build() so watchlists are populated from wallet.
        Skipped if no MNEMONIC/PRIVATE_KEY is configured or if disabled via env.
        """
        if os.getenv("SKIP_WALLET_BOOTSTRAP", "0").lower() in {"1", "true", "yes", "on"}:
            return {"skipped": True, "reason": "disabled"}
        mnemonic = os.getenv("MNEMONIC", "").strip()
        pk = os.getenv("PRIVATE_KEY", "").strip()
        if not mnemonic and not pk:
            log_message("production", "wallet bootstrap skipped: no MNEMONIC or PRIVATE_KEY configured", severity="info")
            return {"skipped": True, "reason": "no_wallet_credentials"}
        try:
            from services.wallet_bootstrap import auto_bootstrap
            result = auto_bootstrap()
            log_message("production", "wallet bootstrap complete", details={
                "pairs": result.get("pairs_generated", []),
                "total_usd": result.get("total_usd", 0),
                "elapsed": result.get("elapsed_sec"),
            })
            return result
        except Exception as exc:
            log_message("production", f"wallet bootstrap failed (non-fatal): {exc}", severity="warning")
            return {"skipped": True, "reason": f"error: {exc}"}

    def _prewarm_pipeline(self) -> Dict[str, Any]:
        skip_default = "1" if self.pipeline.system_profile.memory_pressure else "0"
        if (os.getenv("SKIP_PREWARM") or skip_default).lower() in {"1", "true", "yes", "on"}:
            log_message(
                "production",
                "prewarm skipped for resource budget",
                details={"memory_pressure": self.pipeline.system_profile.memory_pressure},
            )
            return {"skipped": True, "reason": "resource_budget"}
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

    def _safe_horizon_deficit(self) -> Dict[str, float]:
        dataset_meta = getattr(self.pipeline, "_last_dataset_meta", {})
        if not isinstance(dataset_meta, dict):
            return {}
        deficits = dataset_meta.get("horizon_deficit")
        if not isinstance(deficits, dict):
            return {}
        result: Dict[str, float] = {}
        for bucket in ("short", "mid", "long"):
            if bucket not in deficits:
                continue
            try:
                value = float(deficits.get(bucket, 0.0))
            except (TypeError, ValueError):
                continue
            result[bucket] = max(0.0, value)
        return result

    # ------------------------------------------------------------------
    # Secure settings / env hydration
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_secure_env(cls) -> None:
        if cls._env_loaded:
            return
        # EnvLoader.load() hydrates from SecureVault (primary) with .env fallback.
        try:
            EnvLoader.load()
        except Exception:
            pass
        try:
            if "DJANGO_SETTINGS_MODULE" not in os.environ:
                os.environ["DJANGO_SETTINGS_MODULE"] = "coolcrypto_dashboard.settings"
            try:
                import django
                if not getattr(django.apps, "apps", None) or not django.apps.apps.ready:  # type: ignore[attr-defined]
                    django.setup()
            except Exception:
                pass
        except Exception:
            pass
        # If EnvLoader already hydrated from vault, just derive stream env and log.
        # Otherwise try one more time directly.
        try:
            if os.getenv("SECURE_ENV_HYDRATED") != "1":
                env = build_process_env()
                derived = cls._derive_stream_env(env)
                env.update(derived)
                os.environ.update(env)
                os.environ["SECURE_ENV_HYDRATED"] = "1"
            else:
                env = {k: os.environ[k] for k in os.environ if not k.startswith("_")}
            cls._debug_env(env)
            print("[production] env hydrated; wss template", env.get("MARKET_WS_TEMPLATE"), "wss", env.get("GLOBAL_WSS_URL") or env.get("BASE_WSS_URL"))
            chain_label = (env.get("PRIMARY_CHAIN") or os.getenv("PRIMARY_CHAIN", "base")).upper()
            log_message(
                "production",
                "secure settings loaded",
                details={
                    "alchemy": bool(env.get("ALCHEMY_API_KEY")),
                    "wss": bool(env.get("GLOBAL_WSS_URL") or env.get(f"{chain_label}_WSS_URL")),
                },
            )
        except Exception as exc:
            log_message("production", f"secure settings not loaded: {exc}", severity="warning")
        finally:
            cls._env_loaded = True

    @staticmethod
    def _derive_stream_env(env: Dict[str, str]) -> Dict[str, str]:
        """
        Build chain-specific RPC/WSS defaults from vault-stored Alchemy keys so
        market streams and on-chain listeners come up without manual env wiring.
        Falls back to public RPC/WSS when no key is present.
        """
        updates: Dict[str, str] = {}
        chain = (env.get("PRIMARY_CHAIN") or os.getenv("PRIMARY_CHAIN", "base")).lower()
        try:
            from balances import CHAIN_CONFIG
            cfg = CHAIN_CONFIG.get(chain)
            if not cfg:
                return updates
            env_alc = cfg.get("env_alchemy_url")
            api_key = (env.get("ALCHEMY_API_KEY") or os.getenv("ALCHEMY_API_KEY") or "").strip()
            slug = cfg.get("alchemy_slug")
            if api_key and slug:
                rpc = f"https://{slug}.g.alchemy.com/v2/{api_key}"
                wss = f"wss://{slug}.g.alchemy.com/v2/{api_key}"
                chain_prefix = chain.upper()
                updates.setdefault(f"{chain_prefix}_RPC_URL", rpc)
                updates.setdefault(f"{chain_prefix}_WSS_URL", wss)
                updates.setdefault("GLOBAL_RPC_URL", rpc)
                updates.setdefault("GLOBAL_WSS_URL", wss)
                if env_alc:
                    updates.setdefault(env_alc, rpc)
            if not updates:
                public_rpcs = cfg.get("public_rpcs") or []
                if public_rpcs:
                    rpc = str(public_rpcs[0]).strip()
                    if rpc:
                        wss = rpc.replace("https://", "wss://").replace("http://", "ws://") if rpc.startswith("http") else ""
                        chain_prefix = chain.upper()
                        updates.setdefault(f"{chain_prefix}_RPC_URL", rpc)
                        if wss:
                            updates.setdefault(f"{chain_prefix}_WSS_URL", wss)
                        updates.setdefault("GLOBAL_RPC_URL", rpc)
                        if wss:
                            updates.setdefault("GLOBAL_WSS_URL", wss)
                        if env_alc:
                            updates.setdefault(env_alc, rpc)
        except Exception:
            return updates
        return updates

    @staticmethod
    def _debug_env(env: Dict[str, Any]) -> None:
        path = Path("logs/stream_debug.log")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        payload = {
            "ts": time.time(),
            "label": "env_load",
            "BASE_WSS_URL": env.get("BASE_WSS_URL"),
            "GLOBAL_WSS_URL": env.get("GLOBAL_WSS_URL"),
            "MARKET_WS_TEMPLATE": env.get("MARKET_WS_TEMPLATE"),
            "MARKET_WS_SUBSCRIBE": env.get("MARKET_WS_SUBSCRIBE"),
        }
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception:
            try:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(str(payload) + "\n")
            except Exception:
                return


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
