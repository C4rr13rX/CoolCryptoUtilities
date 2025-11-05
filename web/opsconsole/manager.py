from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = os.getenv("PYTHON_BIN")
if not DEFAULT_PYTHON:
    DEFAULT_PYTHON = os.getenv("VIRTUAL_ENV", "")
    if DEFAULT_PYTHON:
        DEFAULT_PYTHON = str(Path(DEFAULT_PYTHON) / "bin" / "python")
    else:
        import sys

        DEFAULT_PYTHON = sys.executable
DEFAULT_COMMAND = [DEFAULT_PYTHON, "-u", "main.py"]
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "console.log"


class ConsoleProcessManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._started_at: Optional[float] = None

    def start(self, command: Optional[list[str]] = None) -> Dict[str, str]:
        cmd = command or DEFAULT_COMMAND
        with self._lock:
            if self._process and self._process.poll() is None:
                return {"status": "running", "pid": str(self._process.pid)}
            logfile = LOG_PATH.open("a", encoding="utf-8")
            env = os.environ.copy()
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                    env=env,
                )
            except Exception as exc:
                logfile.close()
                return {"status": "error", "message": str(exc)}
            self._process = proc
            self._started_at = time.time()
            return {"status": "started", "pid": str(proc.pid)}

    def stop(self, graceful_timeout: float = 5.0) -> Dict[str, str]:
        with self._lock:
            if not self._process or self._process.poll() is not None:
                return {"status": "stopped"}
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGINT)
            except Exception:
                self._process.terminate()
            waited = 0.0
            while self._process.poll() is None and waited < graceful_timeout:
                time.sleep(0.25)
                waited += 0.25
            if self._process.poll() is None:
                self._process.kill()
            return {"status": "stopped"}

    def status(self) -> Dict[str, Optional[str]]:
        with self._lock:
            if not self._process:
                return {"status": "idle", "pid": None, "uptime": None}
            running = self._process.poll() is None
            uptime = None
            if running and self._started_at:
                uptime = f"{time.time() - self._started_at:.1f}"
            return {
                "status": "running" if running else "exited",
                "pid": str(self._process.pid),
                "returncode": None if running else str(self._process.returncode),
                "uptime": uptime,
            }

    def tail(self, lines: int = 200) -> list[str]:
        if not LOG_PATH.exists():
            return []
        with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as handle:
            dq: deque[str] = deque(handle, maxlen=lines)
        return list(dq)


manager = ConsoleProcessManager()
