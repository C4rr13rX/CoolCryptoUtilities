from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from production import ProductionManager
from services.guardian_lock import GuardianLease
from services.guardian_status import update_production_state
from services.logging_utils import log_message

HEARTBEAT_PATH = Path("logs/production_manager_heartbeat.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = REPO_ROOT / "main.py"
MAIN_LOG_PATH = REPO_ROOT / "runtime" / "main_autostart.log"


class ProductionSupervisor:
    """
    Keeps a single ProductionManager instance alive without relying on the wallet
    console/menu. Intended to be driven by guardian so option 7 is effectively
    always running once enabled.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lease: Optional[GuardianLease] = None
        self._manager: Optional[ProductionManager] = None
        self._status: Dict[str, Any] = {
            "running": False,
            "last_start": None,
            "errors": 0,
        }
        self._poll_interval = float(os.getenv("PRODUCTION_SUPERVISOR_INTERVAL", "15"))
        self._heartbeat_ttl = float(os.getenv("PRODUCTION_HEARTBEAT_TTL", "150"))
        self._restart_cooldown = float(os.getenv("PRODUCTION_RESTART_COOLDOWN", "45"))
        self._last_boot = 0.0
        self._main_process_enabled = os.getenv("PRODUCTION_SUPERVISOR_MAIN_PROCESS", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._last_main_launch = 0.0

    def ensure_running(self) -> None:
        if os.environ.get("PRODUCTION_AUTO_DISABLED") == "1":
            # Explicitly requested to keep production manager off.
            self.stop()
            return
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
        manager = self._manager
        if manager and manager.is_running:
            try:
                manager.stop()
            except Exception:
                pass
        self._manager = None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        lease = GuardianLease("production-manager", poll_interval=0.5)
        if not lease.acquire(cancel_event=self._stop):
            log_message("production", "unable to acquire production-manager lease", severity="warning")
            return
        self._lease = lease
        log_message("production", "production supervisor active", severity="info")
        try:
            while not self._stop.is_set():
                if self._main_process_enabled:
                    self._ensure_main_process()
                running, heartbeat = self._check_heartbeat()
                if not running:
                    self._restart_manager(heartbeat)
                else:
                    update_production_state(True, heartbeat)
                if self._stop.wait(self._poll_interval):
                    break
        finally:
            self._set_status(False)
            update_production_state(False, {"note": "stopped"})
            if self._lease:
                self._lease.release()
                self._lease = None
            manager = self._manager
            if manager and manager.is_running:
                try:
                    manager.stop()
                except Exception:
                    pass
            self._manager = None

    def _restart_manager(self, metadata: Dict[str, Any]) -> None:
        now = time.time()
        if now - self._last_boot < self._restart_cooldown:
            return
        self._last_boot = now
        if self._manager and self._manager.is_running:
            try:
                self._manager.stop()
            except Exception:
                pass
        self._manager = None
        if self._main_process_enabled:
            if self._ensure_main_process(force_start=True):
                update_production_state(False, {"note": "launching_main_process"})
                return
            log_message(
                "production",
                "main.py launch failed; falling back to in-process manager",
                severity="warning",
            )
        try:
            manager = ProductionManager()
        except Exception as exc:
            metadata.setdefault("error", str(exc))
            log_message("production", f"unable to instantiate manager: {exc}", severity="error")
            self._increment_error()
            return
        self._manager = manager
        try:
            manager.start()
            self._set_status(True)
            update_production_state(True, {"note": "running"})
            log_message("production", "production manager started", severity="info")
        except Exception as exc:
            metadata.setdefault("error", str(exc))
            log_message("production", f"manager start failed: {exc}", severity="error")
            self._increment_error()
            self._manager = None

    # ------------------------------------------------------------------
    def _check_heartbeat(self) -> tuple[bool, Dict[str, Any]]:
        heartbeat = self._read_heartbeat()
        running = self._heartbeat_is_fresh(heartbeat)
        metadata: Dict[str, Any] = heartbeat or {}
        metadata.setdefault("note", "running" if running else "awaiting_heartbeat")
        self._set_status(running)
        return running, metadata

    def _read_heartbeat(self) -> Dict[str, Any]:
        if not HEARTBEAT_PATH.exists():
            return {}
        try:
            data = json.loads(HEARTBEAT_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _heartbeat_is_fresh(self, heartbeat: Dict[str, Any]) -> bool:
        ts = heartbeat.get("timestamp")
        if ts is None:
            return False
        try:
            age = time.time() - float(ts)
        except (TypeError, ValueError):
            return False
        status = str(heartbeat.get("status") or "").lower()
        return age <= self._heartbeat_ttl and status in {"running", "starting"}

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

    def _ensure_main_process(self, *, force_start: bool = False) -> bool:
        if not self._main_process_enabled:
            return False
        if self._main_process_running():
            return True
        if not force_start and time.time() - self._last_main_launch < self._restart_cooldown:
            return False
        if self._launch_main_process():
            self._last_main_launch = time.time()
            return True
        return False

    def _main_process_running(self) -> bool:
        try:
            import psutil  # type: ignore
        except Exception:
            psutil = None  # type: ignore

        if psutil is not None:
            for proc in psutil.process_iter(["cmdline"]):
                try:
                    cmdline = proc.info.get("cmdline") or []
                except (psutil.NoSuchProcess, psutil.AccessDenied):  # type: ignore[attr-defined]
                    continue
                if any("main.py" in str(part) for part in cmdline):
                    return True
            return False

        try:
            output = subprocess.check_output(["ps", "-eo", "args"], text=True)
        except Exception:
            return False
        return any("main.py" in line for line in output.splitlines())

    def _launch_main_process(self) -> bool:
        if not MAIN_PATH.exists():
            log_message("production", "main.py not found; cannot start main process.", severity="error")
            return False
        python_bin = os.getenv("PYTHON_BIN") or sys.executable
        cmd = [python_bin, str(MAIN_PATH), "--action", "start_production", "--stay-alive"]
        env = os.environ.copy()
        env.setdefault("ALLOW_SQLITE_FALLBACK", "1")
        MAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with MAIN_LOG_PATH.open("a", encoding="utf-8") as handle:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                    env=env,
                )
            log_message("production", f"launched main.py pid={proc.pid}", severity="info")
            return True
        except Exception as exc:
            log_message("production", f"failed to launch main.py: {exc}", severity="error")
            return False


production_supervisor = ProductionSupervisor()
