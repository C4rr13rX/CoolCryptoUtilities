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
            if not self._loop_thread or not self._loop_thread.is_alive():
                self.heartbeat.update("error", metadata={"reason": "ghost_loop_stopped"})
                log_message("production", "ghost supervisor loop stopped; ending cycle loop", severity="error")
                self._stop.set()
                break
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
            if backlog >= self._max_pending:
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
        try:
            env = build_process_env()
            derived = cls._derive_stream_env(env)
            env.update(derived)
            os.environ.update(env)
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
