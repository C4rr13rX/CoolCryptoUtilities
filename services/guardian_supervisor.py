from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from monitoring_guardian.prompt_text import DEFAULT_GUARDIAN_PROMPT
from opsconsole.manager import manager as console_manager
from services.guardian_state import (
    get_guardian_settings,
    set_one_time_prompt,
    update_guardian_settings,
)
from services.guardian_status import (
    request_guardian_run,
    snapshot_status as guardian_queue_snapshot,
)
from services.guardian_lock import GuardianLease
from services.logging_utils import log_message
from services.production_supervisor import production_supervisor
from services.secure_settings import build_process_env

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_LOG = Path("runtime/guardian/guardian.log")


def _python_bin() -> str:
    candidate = os.getenv("PYTHON_BIN")
    if candidate:
        return candidate
    virtual_env = os.getenv("VIRTUAL_ENV")
    if virtual_env:
        return str(Path(virtual_env) / "bin" / "python")
    import sys

    return sys.executable


class GuardianSupervisor:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._console_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lease: Optional[GuardianLease] = None
        self._process: Optional[subprocess.Popen] = None
        self._status_lock = threading.Lock()
        self._status: Dict[str, Any] = {
            "running": False,
            "pid": None,
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
            if not self._acquire_lease():
                log_message("guardian", "another guardian supervisor already active; skipping bootstrap", severity="warning")
                return
            self._stop.clear()
            self._ensure_console_running()
            production_supervisor.ensure_running()
            self._thread = threading.Thread(target=self._guardian_loop, name="guardian-process", daemon=True)
            self._thread.start()
            self._start_console_monitor()

    def stop(self) -> None:
        self._stop.set()
        proc = self._process
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        self._process = None
        if self._thread:
            self._thread.join(timeout=5.0)
        self._thread = None
        if self._console_thread:
            self._console_thread.join(timeout=5.0)
        self._console_thread = None
        production_supervisor.stop()
        with self._status_lock:
            self._status["running"] = False
            self._status["pid"] = None
        self._release_lease()

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
        return settings

    def run_once(self, prompt: Optional[str] = None) -> None:
        if prompt:
            set_one_time_prompt(prompt)
        request_guardian_run()

    def status(self) -> Dict[str, Any]:
        with self._status_lock:
            status = dict(self._status)
        status["settings"] = get_guardian_settings()
        status["console"] = console_manager.status()
        status["running"] = bool(self._process and self._process.poll() is None)
        status["queue"] = guardian_queue_snapshot()
        status["production"] = production_supervisor.status()
        status["log_path"] = str(GUARDIAN_LOG)
        return status

    def ensure_running(self) -> None:
        settings = get_guardian_settings()
        if not settings.get("enabled"):
            return
        if not self._thread or not self._thread.is_alive():
            self.start()
        else:
            self._ensure_console_running()
            production_supervisor.ensure_running()

    def _acquire_lease(self) -> bool:
        if self._lease:
            return True
        lease = GuardianLease("guardian-supervisor", poll_interval=0.5)
        if not lease.acquire(cancel_event=self._stop):
            return False
        self._lease = lease
        return True

    def _release_lease(self) -> None:
        if not self._lease:
            return
        try:
            self._lease.release()
        finally:
            self._lease = None

    # ------------------------------------------------------------------ process helpers
    def _guardian_loop(self) -> None:
        env = build_process_env()
        cmd = [_python_bin(), "-m", "monitoring_guardian.guardian"]
        GUARDIAN_LOG.parent.mkdir(parents=True, exist_ok=True)
        while not self._stop.is_set():
            try:
                log_handle = GUARDIAN_LOG.open("a", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                    text=True,
                )
                self._process = proc
                with self._status_lock:
                    self._status["running"] = True
                    self._status["pid"] = proc.pid
                log_message("guardian", f"started guardian PID {proc.pid}", severity="info")
                while proc.poll() is None and not self._stop.is_set():
                    time.sleep(1.0)
                if self._stop.is_set():
                    break
                log_message("guardian", f"guardian exited with code {proc.returncode}; restarting in 5s", severity="warning")
                time.sleep(5.0)
            except Exception as exc:
                log_message("guardian", f"unable to launch guardian process: {exc}", severity="error")
                time.sleep(5.0)
            finally:
                with self._status_lock:
                    self._status["running"] = False
                self._process = None
        # Loop exited (stop requested); release lease so another process can take over later.
        self._release_lease()

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
