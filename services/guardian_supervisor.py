from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from monitoring_guardian.guardian import DEFAULT_CONFIG, Guardian, load_config
from monitoring_guardian.prompt_text import DEFAULT_GUARDIAN_PROMPT
from opsconsole.manager import manager as console_manager
from services.guardian_lock import GuardianLease
from services.guardian_state import (
    consume_one_time_prompt,
    get_guardian_settings,
    set_one_time_prompt,
    update_guardian_settings,
)
from services.logging_utils import log_message

CONFIG_PATH = Path("monitoring_guardian/config.json")


class GuardianSupervisor:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._console_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._guardian: Optional[Guardian] = None
        self._lease: Optional[GuardianLease] = None
        self._status_lock = threading.Lock()
        self._status: Dict[str, Any] = {
            "running": False,
            "last_report": None,
            "findings": [],
        }
        self._bootstrap_lock = threading.Lock()

    # ------------------------------------------------------------------ public API
    def start_if_enabled(self) -> None:
        settings = get_guardian_settings()
        if settings.get("enabled"):
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        with self._bootstrap_lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._ensure_console_running()
            self._thread = threading.Thread(target=self._run_guardian, name="guardian-loop", daemon=True)
            self._thread.start()
            self._start_console_monitor()

    def stop(self) -> None:
        self._stop.set()
        guardian = self._guardian
        if guardian:
            guardian.shutdown.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._thread = None
        if self._lease:
            self._lease.release()
            self._lease = None
        if self._console_thread:
            self._console_thread.join(timeout=5.0)
        self._console_thread = None

    def set_enabled(self, enabled: bool) -> Dict[str, Any]:
        settings = update_guardian_settings({"enabled": bool(enabled)})
        if enabled:
            self.start()
        else:
            self.stop()
        return settings

    def update_prompt(self, prompt: str) -> Dict[str, Any]:
        settings = update_guardian_settings({"default_prompt": prompt})
        return settings

    def update_interval(self, minutes: int) -> Dict[str, Any]:
        minutes = max(10, min(int(minutes), 720))
        settings = update_guardian_settings({"interval_minutes": minutes})
        guardian = self._guardian
        if guardian:
            guardian.report_interval = minutes * 60
        return settings

    def run_once(self, prompt: Optional[str] = None) -> None:
        if prompt:
            set_one_time_prompt(prompt)
        guardian = self._guardian
        if guardian:
            guardian.request_report()
        else:
            self.start()

    def status(self) -> Dict[str, Any]:
        with self._status_lock:
            status = dict(self._status)
        status["settings"] = get_guardian_settings()
        status["console"] = console_manager.status()
        status["running"] = bool(self._guardian and not self._guardian.shutdown.is_set())
        return status

    def ensure_running(self) -> None:
        settings = get_guardian_settings()
        if not settings.get("enabled"):
            return
        if not self._thread or not self._thread.is_alive():
            self.start()
        else:
            self._ensure_console_running()

    # ------------------------------------------------------------------ internal helpers
    def _run_guardian(self) -> None:
        lease = GuardianLease("guardian-process", poll_interval=2.0)
        if not lease.acquire(cancel_event=self._stop):
            log_message("guardian", "guardian lease unavailable; aborting startup", severity="warning")
            return
        self._lease = lease
        try:
            config = load_config(CONFIG_PATH)
        except Exception:
            config = DEFAULT_CONFIG.copy()  # type: ignore[name-defined]
        settings = get_guardian_settings()
        config["report_interval_minutes"] = settings.get("interval_minutes", 120)
        guardian = Guardian(
            config,
            prompt_provider=self._prompt_provider,
            status_hook=self._record_status,
        )
        self._guardian = guardian
        log_message("guardian", "supervisor started", severity="info")
        try:
            guardian.run()
        except Exception as exc:
            log_message("guardian", f"guardian loop crashed: {exc}", severity="error")
        finally:
            self._guardian = None
            lease.release()
            self._lease = None
            with self._status_lock:
                self._status["running"] = False

    def _prompt_provider(self) -> str:
        prompt = consume_one_time_prompt()
        if prompt:
            return prompt
        settings = get_guardian_settings()
        return settings.get("default_prompt") or DEFAULT_GUARDIAN_PROMPT

    def _record_status(self, payload: Dict[str, Any]) -> None:
        with self._status_lock:
            self._status.update(
                {
                    "running": True,
                    "last_report": payload.get("timestamp"),
                    "findings": payload.get("findings", []),
                }
            )

    def _start_console_monitor(self) -> None:
        if self._console_thread and self._console_thread.is_alive():
            return
        self._console_thread = threading.Thread(target=self._console_loop, name="guardian-console", daemon=True)
        self._console_thread.start()

    def _ensure_console_running(self) -> None:
        try:
            status = console_manager.status()
            if status.get("status") != "running":
                console_manager.start()
        except Exception as exc:
            log_message("guardian", f"console bootstrap error: {exc}", severity="warning")

    def _console_loop(self) -> None:
        while not self._stop.is_set():
            try:
                status = console_manager.status()
                if status.get("status") != "running":
                    console_manager.start()
            except Exception as exc:
                log_message("guardian", f"console monitor error: {exc}", severity="warning")
            for _ in range(30):
                if self._stop.is_set():
                    break
                time.sleep(1.0)


guardian_supervisor = GuardianSupervisor()
