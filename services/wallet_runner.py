from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from services.guardian_status import snapshot_status as guardian_snapshot
from services.secure_settings import build_process_env
from services.wallet_actions import list_wallet_actions

REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_python() -> str:
    candidate = os.getenv("PYTHON_BIN")
    if candidate:
        return candidate
    virtual_env = os.getenv("VIRTUAL_ENV")
    if virtual_env:
        if os.name == "nt":
            return str(Path(virtual_env) / "Scripts" / "python.exe")
        return str(Path(virtual_env) / "bin" / "python")
    import sys

    return sys.executable


class WalletActionRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._status: Dict[str, Any] = {
            "running": False,
            "action": None,
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "message": "idle",
            "log": [],
        }
        self._history: list[Dict[str, Any]] = []

    def status(self) -> Dict[str, Any]:
        with self._lock:
            payload = dict(self._status)
            payload["history"] = list(self._history)
            payload["actions"] = list_wallet_actions()
        try:
            guardian_state = guardian_snapshot()
            payload["production"] = guardian_state.get("production", {})
        except Exception:
            payload["production"] = {"running": False, "updated_at": None, "metadata": {}}
        return payload

    def run(self, action: str, payload: Optional[Dict[str, Any]] = None, user=None) -> None:
        with self._lock:
            if self._status.get("running"):
                raise RuntimeError("wallet action already running")
            self._status.update(
                {
                    "running": True,
                    "action": action,
                    "payload": payload or {},
                    "started_at": time.time(),
                    "finished_at": None,
                    "returncode": None,
                    "message": "running",
                    "log": [],
                }
            )
        thread = threading.Thread(target=self._worker, args=(action, payload or {}, user), daemon=True)
        self._thread = thread
        thread.start()

    # ------------------------------------------------------------------ internal
    def _worker(self, action: str, payload: Dict[str, Any], user=None) -> None:
        env = build_process_env(user)
        python_bin = _default_python()
        cmd = [python_bin, "-u", "main.py", "--action", action]
        if payload:
            cmd.extend(["--payload", json.dumps(payload)])
        log_lines: list[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._finalize(False, f"spawn failed: {exc}", -1, log_lines)
            return

        assert proc.stdout is not None
        for line in proc.stdout:
            log_lines.append(line.rstrip())
            self._append_log(line.rstrip())
        proc.wait()
        success = proc.returncode == 0
        message = "completed" if success else f"failed ({proc.returncode})"
        self._finalize(success, message, proc.returncode, log_lines)

    def _append_log(self, line: str) -> None:
        with self._lock:
            log = self._status.get("log", []) or []
            log.append(line)
            self._status["log"] = log[-400:]

    def _finalize(self, success: bool, message: str, returncode: Optional[int], log_lines: list[str]) -> None:
        with self._lock:
            self._status.update(
                {
                    "running": False,
                    "message": message,
                    "returncode": returncode,
                    "finished_at": time.time(),
                }
            )
            history_entry = {
                "action": self._status.get("action"),
                "payload": self._status.get("payload", {}),
                "message": message,
                "returncode": returncode,
                "started_at": self._status.get("started_at"),
                "finished_at": self._status.get("finished_at"),
                "log": log_lines[-200:],
            }
            self._history.append(history_entry)
            self._history = self._history[-20:]


wallet_runner = WalletActionRunner()
