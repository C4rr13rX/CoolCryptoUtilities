from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from production import ProductionManager
from services.guardian_lock import GuardianLease
from services.guardian_status import update_production_state
from services.logging_utils import log_message


class ProductionSupervisor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._manager: Optional[ProductionManager] = None
        self._lease: Optional[GuardianLease] = None
        self._status: Dict[str, Any] = {
            "running": False,
            "last_start": None,
            "errors": 0,
        }

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

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    # ------------------------------------------------------------------ internals
    def _run(self) -> None:
        lease = GuardianLease("production-manager", poll_interval=1.0)
        if not lease.acquire(cancel_event=self._stop):
            log_message("production", "unable to acquire production-manager lease", severity="warning")
            return
        self._lease = lease
        manager = ProductionManager()
        self._manager = manager
        try:
            manager.start()
            self._set_status(True)
            update_production_state(True, {"note": "started by guardian"})
            while not self._stop.wait(10.0):
                if not manager.is_running:
                    log_message("production", "manager not running, attempting restart", severity="warning")
                    manager.start()
                    self._set_status(True)
                    update_production_state(True, {"note": "auto-restart"})
        except Exception as exc:
            log_message("production", f"manager crashed: {exc}", severity="error")
            self._increment_error()
        finally:
            try:
                manager.stop()
            except Exception:
                pass
            self._manager = None
            self._set_status(False)
            update_production_state(False, {"note": "stopped"})
            if self._lease:
                self._lease.release()
                self._lease = None

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


production_supervisor = ProductionSupervisor()
