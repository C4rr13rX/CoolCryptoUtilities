from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from services.guardian_lock import GuardianLease
from services.guardian_status import update_production_state
from services.logging_utils import log_message
from services.pipeline_prewarm import prewarm_training_pipeline

try:  # pragma: no cover - defensive import when Django app not loaded yet.
    from web.opsconsole.manager import manager as console_manager
except Exception as exc:  # pragma: no cover - makes failure visible in status metadata.
    console_manager = None  # type: ignore[assignment]
    _CONSOLE_IMPORT_ERROR = exc
else:
    _CONSOLE_IMPORT_ERROR = None

HEARTBEAT_PATH = Path("logs/production_manager_heartbeat.json")


class ProductionSupervisor:
    """
    Coordinates a *single* production manager instance by delegating lifecycle
    control to the wallet console process (main.py). This keeps console logs in
    sync with what the Django wallet panel displays while still allowing the
    guardian to auto-restart the bot if it crashes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lease: Optional[GuardianLease] = None
        self._status: Dict[str, Any] = {
            "running": False,
            "last_start": None,
            "errors": 0,
        }
        self._poll_interval = float(os.getenv("PRODUCTION_SUPERVISOR_INTERVAL", "15"))
        self._heartbeat_ttl = float(os.getenv("PRODUCTION_HEARTBEAT_TTL", "150"))
        self._restart_cooldown = float(os.getenv("PRODUCTION_RESTART_COOLDOWN", "45"))
        self._last_boot_cmd = 0.0
        self._prewarm_interval = float(os.getenv("PRODUCTION_PREWARM_INTERVAL", "600"))
        self._last_prewarm = 0.0

    # ------------------------------------------------------------------ public API
    def ensure_running(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="production-supervisor", daemon=True)
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
        self._stop_via_console()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    # ------------------------------------------------------------------ internals
    def _run(self) -> None:
        if console_manager is None:
            log_message(
                "production",
                f"console manager unavailable: {_CONSOLE_IMPORT_ERROR}",
                severity="error",
            )
            return

        lease = GuardianLease("production-manager", poll_interval=1.0)
        if not lease.acquire(cancel_event=self._stop):
            log_message("production", "unable to acquire production-manager lease", severity="warning")
            return
        self._lease = lease
        log_message("production", "supervisor active (console-controlled)", severity="info")
        try:
            while not self._stop.is_set():
                running, metadata = self._report_state()
                if not running:
                    self._maybe_restart(metadata)
                if self._stop.wait(self._poll_interval):
                    break
        finally:
            self._set_status(False)
            update_production_state(False, {"note": "stopped"})
            if self._lease:
                self._lease.release()
                self._lease = None

    # ------------------------------------------------------------------ helpers
    def _report_state(self) -> Tuple[bool, Dict[str, Any]]:
        heartbeat = self._read_heartbeat()
        running = self._heartbeat_is_fresh(heartbeat)
        metadata: Dict[str, Any] = {}
        if heartbeat:
            metadata.update(heartbeat)
        else:
            metadata["note"] = "heartbeat_missing"
        metadata.setdefault("note", "running" if running else "awaiting_heartbeat")
        self._set_status(running)
        update_production_state(running, metadata)
        return running, metadata

    def _set_status(self, running: bool) -> None:
        with self._lock:
            self._status.update(
                {
                    "running": running,
                    "last_start": time.time() if running else self._status.get("last_start"),
                }
            )

    def _increment_error(self) -> None:
        with self._lock:
            self._status["errors"] = int(self._status.get("errors", 0) or 0) + 1

    def _read_heartbeat(self) -> Optional[Dict[str, Any]]:
        if not HEARTBEAT_PATH.exists():
            return None
        try:
            return json.loads(HEARTBEAT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _heartbeat_is_fresh(self, heartbeat: Optional[Dict[str, Any]]) -> bool:
        if not heartbeat:
            return False
        try:
            ts = float(heartbeat.get("timestamp", 0))
        except (TypeError, ValueError):
            return False
        if not ts:
            return False
        age = time.time() - ts
        status = str(heartbeat.get("status") or "").lower()
        return age <= self._heartbeat_ttl and status in {"running", "starting"}

    def _maybe_restart(self, metadata: Dict[str, Any]) -> None:
        now = time.time()
        if now - self._last_boot_cmd < self._restart_cooldown:
            return
        self._last_boot_cmd = now
        self._prewarm_pipeline(metadata)
        try:
            console_status = console_manager.status()
            if console_status.get("status") not in {"running", "started"}:
                start_result = console_manager.start()
                if start_result.get("status") == "error":
                    raise RuntimeError(start_result.get("message", "console start failed"))
                time.sleep(1.0)
            result = console_manager.send("7")
            if result.get("status") != "sent":
                log_message("production", f"console unable to start manager: {result}", severity="warning")
                self._increment_error()
            else:
                log_message(
                    "production",
                    "guardian requested production manager start via console",
                    severity="info",
                )
        except Exception as exc:
            metadata.setdefault("error", str(exc))
            log_message("production", f"console start failed: {exc}", severity="error")
            self._increment_error()

    def _prewarm_pipeline(self, metadata: Dict[str, Any]) -> None:
        now = time.time()
        if now - self._last_prewarm < self._prewarm_interval:
            return
        try:
            summary = prewarm_training_pipeline(focus_assets=metadata.get("focus_assets") or None)
        except Exception as exc:
            log_message("production", f"pipeline prewarm skipped: {exc}", severity="warning")
            return
        self._last_prewarm = now
        metadata.setdefault("prewarm", summary)

    def _stop_via_console(self) -> None:
        if console_manager is None:
            return
        try:
            result = console_manager.send("8")
            if result.get("status") == "sent":
                log_message("production", "guardian requested production manager stop", severity="info")
        except Exception:
            return


production_supervisor = ProductionSupervisor()
