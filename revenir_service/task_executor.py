"""Execute delegated tasks in isolated subprocesses with resource tracking."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .resource_monitor import ResourceMonitor

logger = logging.getLogger("revenir.executor")


class TaskResult:
    __slots__ = (
        "task_id", "task_type", "status", "result", "result_files",
        "error", "peak_cpu", "peak_memory_mb", "duration_seconds",
    )

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.task_type = ""
        self.status = "running"
        self.result: Dict = {}
        self.result_files: List[str] = []
        self.error = ""
        self.peak_cpu = 0.0
        self.peak_memory_mb = 0.0
        self.duration_seconds = 0.0

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "result": self.result,
            "result_files": self.result_files,
            "error": self.error,
            "peak_cpu_percent": self.peak_cpu,
            "peak_memory_mb": self.peak_memory_mb,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class TaskExecutor:
    """Manages concurrent task execution with resource awareness."""

    def __init__(
        self,
        work_dir: Path,
        monitor: ResourceMonitor,
        max_concurrent: int = 2,
        callback_url: str = "",
        api_token: str = "",
    ) -> None:
        self._work_dir = work_dir
        self._monitor = monitor
        self._max_concurrent = max_concurrent
        self._callback_url = callback_url
        self._api_token = api_token
        self._lock = threading.Lock()
        self._active: Dict[str, threading.Thread] = {}
        self._results: Dict[str, TaskResult] = {}
        self._work_dir.mkdir(parents=True, exist_ok=True)
        (self._work_dir / "tasks").mkdir(exist_ok=True)
        (self._work_dir / "results").mkdir(exist_ok=True)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @max_concurrent.setter
    def max_concurrent(self, value: int) -> None:
        self._max_concurrent = max(1, value)

    def can_accept(self) -> bool:
        if self._monitor.should_pause:
            return False
        with self._lock:
            return len(self._active) < self._max_concurrent

    def submit(self, task_id: str, task_type: str, payload: Dict, env_keys: Dict[str, str]) -> bool:
        """Submit a task for execution. Returns False if at capacity."""
        if not self.can_accept():
            return False

        result = TaskResult(task_id)
        result.task_type = task_type

        with self._lock:
            if task_id in self._active:
                return False
            self._results[task_id] = result
            thread = threading.Thread(
                target=self._run_task,
                args=(task_id, task_type, payload, env_keys, result),
                daemon=True,
                name=f"task-{task_id[:8]}",
            )
            self._active[task_id] = thread

        thread.start()
        return True

    def get_result(self, task_id: str) -> Optional[Dict]:
        with self._lock:
            r = self._results.get(task_id)
        if r:
            return r.to_dict()
        # Check disk
        path = self._work_dir / "results" / f"{task_id}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    def active_tasks(self) -> List[Dict]:
        with self._lock:
            return [
                {"task_id": tid, "task_type": r.task_type, "status": r.status}
                for tid, r in self._results.items()
                if r.status == "running"
            ]

    def _run_task(
        self,
        task_id: str,
        task_type: str,
        payload: Dict,
        env_keys: Dict[str, str],
        result: TaskResult,
    ) -> None:
        start = time.time()
        task_dir = self._work_dir / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        peak_cpu = 0.0
        peak_mem = 0.0

        try:
            # Write task spec
            spec_path = task_dir / "spec.json"
            spec_path.write_text(json.dumps({
                "task_id": task_id,
                "task_type": task_type,
                "payload": payload,
            }, indent=2), encoding="utf-8")

            # Build environment with forwarded API keys
            env = os.environ.copy()
            env.update(env_keys)
            env["REVENIR_TASK_ID"] = task_id
            env["REVENIR_TASK_TYPE"] = task_type
            env["REVENIR_TASK_DIR"] = str(task_dir)
            env["REVENIR_RESULT_DIR"] = str(self._work_dir / "results")

            # Run the task runner script
            runner = Path(__file__).parent / "task_runner.py"
            cmd = [sys.executable, "-u", str(runner), str(spec_path)]
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(task_dir),
                encoding="utf-8",
                errors="replace",
            )

            # Monitor resources while task runs
            log_path = task_dir / "output.log"
            with log_path.open("w", encoding="utf-8") as log_f:
                while proc.poll() is None:
                    line = proc.stdout.readline()
                    if line:
                        log_f.write(line)
                        log_f.flush()
                    snap = self._monitor.snapshot()
                    peak_cpu = max(peak_cpu, snap["cpu_percent"])
                    peak_mem = max(peak_mem, snap.get("our_rss_mb", 0))
                    if self._monitor.should_pause:
                        logger.warning("task %s: system under pressure, waiting...", task_id[:8])
                        time.sleep(2)

                # Drain remaining output
                remaining = proc.stdout.read()
                if remaining:
                    log_f.write(remaining)

            duration = time.time() - start
            result.duration_seconds = duration
            result.peak_cpu = peak_cpu
            result.peak_memory_mb = peak_mem

            if proc.returncode == 0:
                # Read result file if the runner wrote one
                result_path = task_dir / "result.json"
                if result_path.exists():
                    try:
                        result.result = json.loads(result_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                # Collect any output files
                out_dir = task_dir / "output"
                if out_dir.exists():
                    for f in out_dir.iterdir():
                        result.result_files.append(str(f))
                result.status = "completed"
            else:
                result.status = "failed"
                result.error = f"Process exited with code {proc.returncode}"
                # Grab last 20 lines of output for error context
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    result.error += "\n" + "\n".join(lines[-20:])
                except Exception:
                    pass

        except Exception as exc:
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            result.duration_seconds = time.time() - start
            result.peak_cpu = peak_cpu
            result.peak_memory_mb = peak_mem

        finally:
            # Persist result to disk
            result_path = self._work_dir / "results" / f"{task_id}.json"
            try:
                result_path.write_text(
                    json.dumps(result.to_dict(), indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception:
                pass

            with self._lock:
                self._active.pop(task_id, None)

            # Notify callback if configured
            if self._callback_url:
                self._send_callback(result)

    def _send_callback(self, result: TaskResult) -> None:
        """POST result back to the main server with retry + backoff."""
        import urllib.request

        max_retries = 5
        backoff = 5.0  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                data = json.dumps(result.to_dict()).encode("utf-8")
                req = urllib.request.Request(
                    self._callback_url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._api_token}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    logger.info("callback sent for task %s: %d", result.task_id[:8], resp.status)
                return  # success
            except Exception as exc:
                logger.warning(
                    "callback attempt %d/%d failed for task %s: %s",
                    attempt, max_retries, result.task_id[:8], exc,
                )
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)  # cap at 2 minutes

        logger.error(
            "callback exhausted retries for task %s — result saved to disk",
            result.task_id[:8],
        )
