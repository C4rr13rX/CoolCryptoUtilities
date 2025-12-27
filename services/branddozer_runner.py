from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import shutil
from django.db import close_old_connections

from tools.codex_session import CodexSession
from services.branddozer_state import get_project, list_projects, update_project_fields
from services.logging_utils import log_message

LOG_ROOT = Path("runtime/branddozer")
LOG_ROOT.mkdir(parents=True, exist_ok=True)

BASE_INSTRUCTIONS = (
    "Act as a full-access Codex agent working on the specified project root. "
    "For each run: stay in a fix/test/fix/test loop until errors are resolved; "
    "generate or update CLI test harnesses for every GUI function; take Chromium screenshots when it helps you assess UI/UX; "
    "use scripts/branddozer_ui_capture.py to capture UI screenshots when needed; "
    "avoid destructive actions; summarize changes; and stop cleanly when done."
)


class BrandDozerManager:
    def __init__(self) -> None:
        self._threads: Dict[str, threading.Thread] = {}
        self._stops: Dict[str, threading.Event] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self._status: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------ lifecycle
    def start(self, project_id: str) -> Dict[str, str]:
        with self._lock:
            if project_id in self._threads and self._threads[project_id].is_alive():
                return {"status": "running"}
            if not shutil.which("codex"):
                msg = "codex CLI not available on PATH; cannot start BrandDozer cycle."
                log_message("branddozer", msg, severity="error")
                self._status[project_id] = {"state": "error", "last_message": msg}
                return {"status": "error", "detail": msg}
            stop_event = threading.Event()
            self._locks.setdefault(project_id, threading.Lock())
            thread = threading.Thread(target=self._run_loop, args=(project_id, stop_event), name=f"branddozer-{project_id}", daemon=True)
            self._threads[project_id] = thread
            self._stops[project_id] = stop_event
            thread.start()
            self._status[project_id] = {"state": "running"}
        update_project_fields(project_id, {"enabled": True})
        return {"status": "started"}

    def stop(self, project_id: str) -> Dict[str, str]:
        with self._lock:
            stop = self._stops.get(project_id)
            if stop:
                stop.set()
            thread = self._threads.get(project_id)
        if thread:
            thread.join(timeout=5.0)
        update_project_fields(project_id, {"enabled": False})
        with self._lock:
            self._status[project_id] = {"state": "stopped"}
        return {"status": "stopped"}

    def stop_all(self) -> None:
        for pid in list(self._stops.keys()):
            self.stop(pid)

    # ------------------------------------------------------------------ query
    def snapshot(self) -> List[Dict[str, str]]:
        snap: List[Dict[str, str]] = []
        for project in list_projects():
            pid = project.get("id")
            status = self._status.get(pid, {})
            snap.append(
                {
                    "id": pid,
                    "running": bool(pid and pid in self._threads and self._threads[pid].is_alive()),
                    "state": status.get("state", "idle"),
                    "last_run": status.get("last_run") or project.get("last_run"),
                    "last_message": status.get("last_message", ""),
                }
            )
        return snap

    def tail_log(self, project_id: str, limit: int = 200) -> List[str]:
        project = get_project(project_id)
        if not project:
            return []
        path = Path(project.get("log_path") or "")
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()
        except Exception:
            return []
        return [line.rstrip("\n") for line in lines[-limit:]]

    # ------------------------------------------------------------------ workers
    def _run_loop(self, project_id: str, stop: threading.Event) -> None:
        close_old_connections()
        project = get_project(project_id)
        if not project:
            return
        interval = int(project.get("interval_minutes") or 120) * 60
        lock = self._locks.get(project_id) or threading.Lock()
        while not stop.is_set():
            project = get_project(project_id)
            if not project:
                break
            try:
                acquired = lock.acquire(timeout=5.0)
                if not acquired:
                    log_message("branddozer", f"skip cycle: lock busy for {project_id}", severity="warning")
                    continue
                try:
                    self._run_cycle(project, stop)
                finally:
                    lock.release()
                update_project_fields(project_id, {"last_run": time.time()})
                with self._lock:
                    self._status[project_id] = {"state": "running", "last_run": time.time(), "last_message": "cycle complete"}
            except Exception as exc:  # pragma: no cover - best effort resilience
                log_message("branddozer", f"cycle error for {project_id}: {exc}", severity="error")
                with self._lock:
                    self._status[project_id] = {"state": "error", "last_message": str(exc)}
            if stop.wait(interval):
                break
            close_old_connections()
        close_old_connections()
        with self._lock:
            self._status[project_id] = {"state": "stopped"}

    def _run_cycle(self, project: Dict[str, str], stop: threading.Event) -> None:
        if stop.is_set():
            return
        prompts: List[tuple[str, str]] = []
        default_prompt = (project.get("default_prompt") or "").strip()
        if default_prompt:
            prompts.append(("default", default_prompt))
        for idx, intr in enumerate(project.get("interjections") or []):
            intr_text = (intr or "").strip()
            if intr_text:
                prompts.append((f"interjection-{idx+1}", intr_text))
        if not prompts:
            return
        for label, prompt in prompts:
            if stop.is_set():
                break
            self._run_prompt(project, prompt, label)
            with self._lock:
                self._status[project["id"]] = {"state": "running", "last_message": f"{label} done"}

    def _run_prompt(self, project: Dict[str, str], prompt: str, label: str) -> str:
        if not shutil.which("codex"):
            raise RuntimeError("codex CLI not available on PATH")
        root = project.get("root_path") or "."
        session_name = f"branddozer-{project.get('id')}"
        transcript_dir = LOG_ROOT / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        session = CodexSession(
            session_name,
            transcript_dir=transcript_dir,
            sandbox_mode="danger-full-access",
            approval_policy="never",
            model="gpt-5.1-codex-max",
            reasoning_effort="xhigh",
            read_timeout_s=None,
            workdir=str(root),
        )
        header = f"[BrandDozer Project: {project.get('name')}]\nRoot: {root}\nMode: {label}\n{BASE_INSTRUCTIONS}\n"
        full_prompt = f"{header}\n{prompt}"
        log_path = Path(project.get("log_path") or LOG_ROOT / f"{project.get('id')}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        divider = "=" * 60
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{divider}\n{ts} :: {project.get('name')} :: {label}\nPROMPT:\n{prompt}\nOUTPUT:\n")
            handle.flush()

            def _stream_writer(chunk: str) -> None:
                handle.write(chunk)
                handle.flush()

            output = session.send(full_prompt, stream=True, stream_callback=_stream_writer)
            if output and not output.endswith("\n"):
                handle.write("\n")
            handle.write("\n")
        return output

    def _append_log(self, project: Dict[str, str], label: str, prompt: str, output: Optional[str]) -> None:
        log_path = Path(project.get("log_path") or LOG_ROOT / f"{project.get('id')}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        text = output or "[no output]"
        divider = "=" * 60
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{divider}\n{ts} :: {project.get('name')} :: {label}\nPROMPT:\n{prompt}\nOUTPUT:\n{text}\n")


branddozer_manager = BrandDozerManager()
