from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from db import get_db
from services.cron_profile import load_profile, save_profile
from services.guardian_lock import GuardianLease
from services.logging_utils import log_message
from services.secure_settings import build_process_env, default_env_user

STATE_PATH = Path("runtime/cron/state.json")
LOG_SOURCE = "internal-cron"


def _ensure_state_dir() -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"tasks": {}}
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"tasks": {}}
    if not isinstance(payload, dict):
        return {"tasks": {}}
    payload.setdefault("tasks", {})
    return payload


def _save_state(state: Dict[str, Any]) -> None:
    _ensure_state_dir()
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _now_ts() -> float:
    return time.time()


def _apply_env(env: Dict[str, str]) -> None:
    for key, value in env.items():
        if value is None:
            continue
        os.environ[key] = str(value)


@dataclass
class TaskOutcome:
    status: str
    message: str


class InternalCronSupervisor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lease: Optional[GuardianLease] = None
        self._poll_interval = float(os.getenv("CRON_POLL_INTERVAL", "30"))
        self._state = _load_state()
        self._status: Dict[str, Any] = {
            "running": False,
            "last_cycle": None,
            "errors": 0,
        }
        self._force_run: set[str] = set()

    # ------------------------------------------------------------------ public API
    def ensure_running(self) -> None:
        if os.getenv("CRON_AUTO_DISABLED") == "1" or os.getenv("INTERNAL_CRON_DISABLED") == "1":
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run_loop, name="internal-cron", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=10.0)
            self._thread = None
            if self._lease:
                self._lease.release()
                self._lease = None
        with self._lock:
            self._status["running"] = False

    def status(self) -> Dict[str, Any]:
        profile = load_profile()
        with self._lock:
            tasks = dict(self._state.get("tasks") or {})
            running = bool(self._thread and self._thread.is_alive())
            status = dict(self._status)
        return {
            "running": running,
            "profile": profile,
            "tasks": tasks,
            "status": status,
            "state_path": str(STATE_PATH),
        }

    def update_profile(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = load_profile(force=True)
        merged = dict(current)
        merged.update(updates or {})
        return save_profile(merged)

    def run_once(self, task_id: Optional[str] = None) -> None:
        if task_id:
            self._force_run.add(task_id)
        else:
            for task in load_profile().get("tasks", []):
                tid = str(task.get("id") or "").strip()
                if tid:
                    self._force_run.add(tid)

    # ------------------------------------------------------------------ loop
    def _run_loop(self) -> None:
        lease = GuardianLease("internal-cron", poll_interval=0.5)
        if not lease.acquire(cancel_event=self._stop):
            log_message(LOG_SOURCE, "cron lease busy; skipping start", severity="warning")
            return
        self._lease = lease
        log_message(LOG_SOURCE, "internal cron active", severity="info")
        try:
            while not self._stop.is_set():
                profile = load_profile()
                if not profile.get("enabled", True):
                    with self._lock:
                        self._status["running"] = False
                    if self._stop.wait(self._poll_interval):
                        break
                    continue
                with self._lock:
                    self._status["running"] = True
                now = _now_ts()
                for task in profile.get("tasks", []):
                    if self._stop.is_set():
                        break
                    task_id = str(task.get("id") or "").strip()
                    if not task_id or not task.get("enabled", True):
                        continue
                    state = self._get_task_state(task_id)
                    if state.get("running"):
                        continue
                    if task_id in self._force_run or self._task_due(task, state, now):
                        self._force_run.discard(task_id)
                        self._execute_task(task, profile)
                with self._lock:
                    self._status["last_cycle"] = now
                if self._stop.wait(self._poll_interval):
                    break
        finally:
            with self._lock:
                self._status["running"] = False
            if self._lease:
                self._lease.release()
                self._lease = None

    # ------------------------------------------------------------------ task helpers
    def _get_task_state(self, task_id: str) -> Dict[str, Any]:
        tasks = self._state.setdefault("tasks", {})
        if task_id not in tasks:
            tasks[task_id] = {
                "last_run": None,
                "last_status": "never",
                "last_error": None,
                "last_message": None,
                "last_duration_s": None,
                "running": False,
                "next_run": None,
            }
        return tasks[task_id]

    def _task_due(self, task: Dict[str, Any], state: Dict[str, Any], now: float) -> bool:
        next_run = state.get("next_run")
        if next_run is None:
            interval = int(task.get("interval_minutes") or 60) * 60
            jitter = int(task.get("jitter_seconds") or 0)
            state["next_run"] = now + (random.randint(0, jitter) if jitter > 0 else 0)
            _save_state(self._state)
            return True
        return now >= float(next_run)

    def _execute_task(self, task: Dict[str, Any], profile: Dict[str, Any]) -> None:
        task_id = str(task.get("id") or "unknown")
        state = self._get_task_state(task_id)
        state["running"] = True
        state["last_status"] = "running"
        state["last_message"] = "running"
        _save_state(self._state)
        start = _now_ts()
        outcome = TaskOutcome(status="success", message="completed")
        try:
            outcome = self._run_task(task, profile)
        except Exception as exc:  # pragma: no cover - defensive
            outcome = TaskOutcome(status="error", message=str(exc))
            log_message(LOG_SOURCE, f"task {task_id} failed: {exc}", severity="error")
            with self._lock:
                self._status["errors"] = int(self._status.get("errors", 0)) + 1
        finally:
            end = _now_ts()
            state["running"] = False
            state["last_run"] = end
            state["last_status"] = outcome.status
            state["last_message"] = outcome.message
            state["last_duration_s"] = round(end - start, 2)
            interval = int(task.get("interval_minutes") or 60) * 60
            jitter = int(task.get("jitter_seconds") or 0)
            if outcome.status.startswith("skipped"):
                backoff = min(3600, max(300, interval // 4))
                state["next_run"] = end + backoff
            else:
                state["next_run"] = end + interval + (random.randint(0, jitter) if jitter > 0 else 0)
            _save_state(self._state)

    # ------------------------------------------------------------------ task runner
    def _run_task(self, task: Dict[str, Any], profile: Dict[str, Any]) -> TaskOutcome:
        task_id = str(task.get("id") or "task")
        user = default_env_user()
        env = build_process_env(user)
        _apply_env(env)
        has_mnemonic = bool(env.get("MNEMONIC") or "")
        if task.get("requires_mnemonic") and not has_mnemonic:
            msg = "mnemonic missing; skipping"
            log_message(LOG_SOURCE, f"{task_id}: {msg}", severity="warning")
            return TaskOutcome(status="skipped_mnemonic", message=msg)

        production_active = False
        try:
            production_active = bool(get_db().get_control_flag("production_manager_active"))
        except Exception:
            production_active = False

        steps = [str(step) for step in (task.get("steps") or []) if step]
        context: Dict[str, Any] = {"discovery": None, "candidates": []}
        for step in steps:
            if self._stop.is_set():
                return TaskOutcome(status="skipped", message="stop requested")
            if step == "discovery":
                context["discovery"] = self._run_discovery(profile)
            elif step == "watchlists":
                context["candidates"] = self._update_watchlists(profile)
            elif step == "downloads":
                if production_active:
                    log_message(LOG_SOURCE, f"{task_id}: downloads skipped (production active)", severity="info")
                else:
                    self._run_downloads(profile)
            elif step == "news":
                self._run_news(profile, context)
            elif step == "training":
                if production_active:
                    log_message(LOG_SOURCE, f"{task_id}: training skipped (production active)", severity="info")
                else:
                    self._run_training(profile)
            elif step == "guardian":
                self._run_guardian()
            elif step == "branddozer":
                self._run_branddozer()
            else:
                log_message(LOG_SOURCE, f"{task_id}: unknown step {step}", severity="warning")
        return TaskOutcome(status="success", message="completed")

    # ------------------------------------------------------------------ step implementations
    def _run_discovery(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        discovery_cfg = profile.get("discovery") or {}
        chains = discovery_cfg.get("chains") or None
        limit = int(discovery_cfg.get("limit") or 40)
        try:
            from services.discovery.coordinator import DiscoveryCoordinator
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"discovery unavailable: {exc}", severity="warning")
            return {"count": 0, "error": str(exc)}
        coordinator = DiscoveryCoordinator(chains=chains, limit=limit)
        results = coordinator.run()
        log_message(LOG_SOURCE, f"discovery fetched {len(results)} tokens", severity="info")
        return {"count": len(results)}

    def _update_watchlists(self, profile: Dict[str, Any]) -> List[str]:
        discovery_cfg = profile.get("discovery") or {}
        try:
            from datetime import timedelta
            from django.utils import timezone
            from discovery.models import DiscoveryEvent, DiscoveredToken, HoneypotCheck
            from services.watchlists import mutate_watchlist
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"watchlist update unavailable: {exc}", severity="warning")
            return []

        max_age = float(discovery_cfg.get("max_age_hours") or 36)
        min_liquidity = float(discovery_cfg.get("min_liquidity_usd") or 0)
        min_volume = float(discovery_cfg.get("min_volume_usd") or 0)
        min_change_1h = float(discovery_cfg.get("min_change_1h") or -1000)
        min_change_24h = float(discovery_cfg.get("min_change_24h") or -1000)
        max_tokens = int(discovery_cfg.get("max_tokens") or 25)
        target = str(discovery_cfg.get("watchlist_target") or "stream").lower()
        also_ghost = bool(discovery_cfg.get("also_add_to_ghost", True))

        cutoff = timezone.now() - timedelta(hours=max_age)
        events = (
            DiscoveryEvent.objects.filter(created_at__gte=cutoff)
            .filter(liquidity_usd__gte=min_liquidity)
            .filter(volume_24h__gte=min_volume)
            .filter(bull_score__gte=min_change_1h)
            .filter(price_change_24h__gte=min_change_24h)
            .order_by("-volume_24h", "-liquidity_usd", "-created_at")
        )
        symbols = []
        seen = set()
        for event in events:
            symbol = (event.symbol or "").upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
            if len(symbols) >= max_tokens:
                break

        if not symbols:
            return []

        # Filter out known honeypots/rejections.
        statuses = {
            tok.symbol.upper(): tok.status
            for tok in DiscoveredToken.objects.filter(symbol__in=symbols)
        }
        latest_checks: Dict[str, HoneypotCheck] = {}
        for check in HoneypotCheck.objects.filter(symbol__in=symbols).order_by("symbol", "-created_at"):
            sym = check.symbol.upper()
            if sym not in latest_checks:
                latest_checks[sym] = check
        filtered = []
        for symbol in symbols:
            status = statuses.get(symbol)
            if status in {"honeypot", "rejected"}:
                continue
            check = latest_checks.get(symbol)
            if check and check.verdict == "honeypot":
                continue
            filtered.append(symbol)

        if not filtered:
            return []

        mutate_watchlist(target, add=filtered)
        if also_ghost:
            mutate_watchlist("ghost", add=filtered)
        log_message(
            LOG_SOURCE,
            f"watchlists updated ({target})",
            details={"count": len(filtered), "symbols": filtered[:8]},
        )
        return filtered

    def _run_downloads(self, profile: Dict[str, Any]) -> None:
        downloads_cfg = profile.get("downloads") or {}
        chains = downloads_cfg.get("chains") or []
        max_pairs = int(downloads_cfg.get("max_pairs") or 0)
        if chains:
            os.environ["DOWNLOAD_WORKER_CHAINS"] = ",".join([str(ch).lower() for ch in chains])
        if max_pairs:
            os.environ["DOWNLOAD_MAX_PAIRS"] = str(max_pairs)
        try:
            from services.background_workers import TokenDownloadSupervisor
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"download supervisor unavailable: {exc}", severity="warning")
            return
        supervisor = TokenDownloadSupervisor(db=get_db())
        supervisor.run_cycle()
        log_message(LOG_SOURCE, "download supervisor cycle completed", severity="info")

    def _run_news(self, profile: Dict[str, Any], context: Dict[str, Any]) -> None:
        news_cfg = profile.get("news") or {}
        lookback_hours = int(news_cfg.get("lookback_hours") or 72)
        max_pages = news_cfg.get("max_pages")
        max_tokens = int(news_cfg.get("max_tokens") or 8)
        tokens = context.get("candidates") or []
        if tokens:
            tokens = list(tokens)[:max_tokens]
        try:
            from datetime import datetime, timedelta, timezone
            from services.news_lab import collect_news_for_terms
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"news collector unavailable: {exc}", severity="warning")
            return
        if not tokens:
            return
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)
        result = collect_news_for_terms(tokens=tokens, start=start, end=end, max_pages=max_pages)
        log_message(
            LOG_SOURCE,
            "news batch complete",
            details={"tokens": tokens, "articles": len(result.get("items", []))},
        )

    def _run_training(self, profile: Dict[str, Any]) -> None:
        training_cfg = profile.get("training") or {}
        if not training_cfg.get("enabled", True):
            return
        try:
            from trading.pipeline import TrainingPipeline
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"training unavailable: {exc}", severity="warning")
            return
        pipeline = TrainingPipeline(db=get_db())
        pipeline.train_candidate()
        log_message(LOG_SOURCE, "training candidate cycle completed", severity="info")

    def _run_guardian(self) -> None:
        try:
            from services.guardian_supervisor import guardian_supervisor
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"guardian unavailable: {exc}", severity="warning")
            return
        guardian_supervisor.ensure_running()
        guardian_supervisor.run_once()

    def _run_branddozer(self) -> None:
        try:
            from services.branddozer_runner import branddozer_manager
            from services.branddozer_state import list_projects
        except Exception as exc:  # pragma: no cover
            log_message(LOG_SOURCE, f"branddozer unavailable: {exc}", severity="warning")
            return
        projects = list_projects()
        started = 0
        for project in projects:
            if not project.get("enabled"):
                continue
            branddozer_manager.start(project.get("id"))
            started += 1
        if started:
            log_message(LOG_SOURCE, f"branddozer ensured {started} project(s)", severity="info")


cron_supervisor = InternalCronSupervisor()
