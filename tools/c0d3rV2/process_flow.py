from __future__ import annotations

import argparse
import difflib
import re
import os
import getpass
import sys
import time
import datetime
import threading
import queue
import subprocess
import urllib.request
import hashlib
import shutil
import platform
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent


os.environ["C0D3R_RUNTIME_ROOT"] = str((PROJECT_ROOT / "runtime" / "c0d3r").resolve())

class ProcessFlow:
    def __init__(self, userRequest: str):
        self.userRequest = userRequest
        self.response = None
        
    def _runtime_root(self) -> Path:
        override = os.getenv("C0D3R_RUNTIME_ROOT")
        if override:
            return Path(override).expanduser().resolve()
        return (PROJECT_ROOT / "runtime" / "c0d3r").resolve()

    def _auto_context_commands_enabled(self) -> bool:
        return os.getenv("C0D3R_AUTO_CONTEXT_COMMANDS", "0").strip().lower() in {"1", "true", "yes", "on"}


    def _runtime_path(self,*parts: str) -> Path:
        return self._runtime_root().joinpath(*parts)


    def _detect_shells(self) -> List[str]:
        shells = [
            ("pwsh", "pwsh"),
            ("powershell", "powershell"),
            ("cmd", "cmd"),
            ("bash", "bash"),
            ("sh", "sh"),
            ("zsh", "zsh"),
        ]
        available = []
        for name, exe in shells:
            if shutil.which(exe):
                available.append(name)
        return available


    def _detect_tools(self) -> Dict[str, str]:
        tools = ("python", "pip", "git", "node", "npm", "npx", "yarn", "pnpm", "uv", "rg")
        found: Dict[str, str] = {}
        for tool in tools:
            path = shutil.which(tool) or ""
            found[tool] = path
        return found


    def _system_time_info(self) -> dict:
        try:
            now = datetime.datetime.now().astimezone()
            return {
                "local_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": now.tzname() or "",
                "utc_offset": now.strftime("%z") or "",
            }
        except Exception:
            return {}


    def _weather_summary(self) -> str:
        if os.getenv("C0D3R_WEATHER", "1").strip().lower() in {"0", "false", "no", "off"}:
            return ""
        url = os.getenv("C0D3R_WEATHER_URL", "https://wttr.in/?format=1").strip()
        if not url:
            return ""
        timeout_s = float(os.getenv("C0D3R_WEATHER_TIMEOUT_S", "5.0") or "1.0")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "c0d3r/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                text = resp.read().decode("utf-8", errors="ignore").strip()
            return text[:200] if text else ""
        except Exception:
            return ""

    def _environment_context_block(self, workdir: Path) -> str:
        lines = ["Environment:"]
        lines.append(f"- platform: {sys.platform}")
        lines.append(f"- os_name: {os.name}")
        lines.append(f"- cwd: {workdir}")
        try:
            lines.append(f"- project_root: {workdir.resolve()}")
        except Exception:
            pass
        shells = self._detect_shells()
        lines.append(f"- shells: {', '.join(shells) if shells else '(none found)'}")
        tools = self._detect_tools()
        for name, path in tools.items():
            lines.append(f"- tool.{name}: {path or 'missing'}")
        time_info = self._system_time_info()
        if time_info:
            if time_info.get("local_time"):
                lines.append(f"- local_time: {time_info['local_time']}")
            if time_info.get("timezone"):
                lines.append(f"- timezone: {time_info['timezone']}")
            if time_info.get("utc_offset"):
                lines.append(f"- utc_offset: {time_info['utc_offset']}")
        weather = self._weather_summary()
        if weather:
            lines.append(f"- weather: {weather}")
        try:
            from services.system_probe import collect_system_probe

            probe = collect_system_probe(cwd=workdir)
            lines.append(f"- is_admin: {probe.is_admin}")
            lines.append(f"- cpu_count: {probe.cpu_count}")
            lines.append(f"- total_memory_gb: {probe.total_memory_gb}")
            lines.append(f"- hostname: {probe.hostname}")
            lines.append(f"- network_available: {probe.network_available}")
        except Exception:
            pass
        lines.append("- local_tools: datalab + wallet meta commands available")
        lines.append("- datalab.meta: ::datalab_tables | ::datalab_query {json} | ::datalab_news {json} | ::datalab_web {json}")
        lines.append("- wallet.meta: ::wallet_login | ::wallet_logout | ::wallet_actions | ::wallet_lookup {json} | ::wallet_send {json} | ::wallet_swap {json} | ::wallet_bridge {json}")
        lines.append(
            "- vm.meta: ::vm_status | ::vm_catalog | ::vm_latest {json} | ::vm_bootstrap {json} | ::vm_update {json} | ::vm_fetch {json} | ::vm_create {json} | ::vm_unattended {json} | ::vm_autopilot {json} | ::vm_start {json} | ::vm_stop {json} | ::vm_wait {json} | ::vm_ready {json} | ::vm_screenshot {json} | ::vm_mouse {json} | ::vm_type {json} | ::vm_keys {json} | ::vm_exec {json} | ::vm_guest {json} | ::vm_tail {json} | ::vm_obstacle {json}"
        )
        lines.append("- vm.notes: vm_screenshot + vm_obstacle auto-attach images for the next model step")
        lines.append("- documents: auto-attach supported files (pdf/csv/doc/docx/xls/xlsx/html/txt/md) when referenced or via --doc")
        return "\n".join(lines)



    def _summary_paths(self, session_id: str | None = None) -> tuple[Path, Path]:
        if session_id:
            return (self._runtime_path(f"summary_{session_id}.json"), self._runtime_path(f"summary_{session_id}.txt"))
        return (self._runtime_path("summary.json"), self._runtime_path("summary.txt"))


    def _load_summary_bundle(self, session_id: str | None = None) -> dict:
        summary_json, summary_txt = self._summary_paths(session_id)
        def _trim_200_words(text: str) -> str:
            words = text.split()
            if len(words) > 200:
                return " ".join(words[:200])
            return text

        strict = os.getenv("C0D3R_SUMMARY_SESSION_STRICT", "1").strip().lower() not in {"0", "false", "no", "off"}
        if summary_json.exists():
            try:
                payload = json.loads(summary_json.read_text(encoding="utf-8", errors="ignore"))
                stored_session = str(payload.get("session_id") or "").strip()
                if session_id and stored_session and stored_session != session_id:
                    return {"summary": "", "key_points": []}
                if session_id and not stored_session and strict:
                    return {"summary": "", "key_points": []}
                summary = str(payload.get("summary") or "").strip()
                summary = _trim_200_words(summary)
                points = payload.get("key_points") or []
                if not isinstance(points, list):
                    points = []
                points = [str(p).strip() for p in points if str(p).strip()]
                return {"summary": summary, "key_points": points}
            except Exception:
                pass
        if session_id and strict:
            return {"summary": "", "key_points": []}
        summary = ""
        if summary_txt.exists():
            try:
                summary = summary_txt.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                summary = ""
        summary = _trim_200_words(summary)
        points = self._extract_key_points(summary, limit=10)
        return {"summary": summary, "key_points": points}


    def _save_summary_bundle(self, bundle: dict, *, session_id: str | None = None) -> None:
        summary_json, summary_txt = self._summary_paths(session_id)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = str(bundle.get("summary") or "").strip()
        words = summary.split()
        if len(words) > 200:
            summary = " ".join(words[:200])
        points = bundle.get("key_points") or []
        if not isinstance(points, list):
            points = []
        points = [str(p).strip() for p in points if str(p).strip()]
        if len(points) > 10:
            points = points[:10]
        payload = {"summary": summary, "key_points": points, "session_id": session_id or ""}
        try:
            summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass
        try:
            summary_txt.write_text(summary, encoding="utf-8")
        except Exception:
            pass


    def _load_rolling_summary(self, max_chars: int = 2000, *, session_id: str | None = None) -> str:
        bundle = self._load_summary_bundle(session_id=session_id)
        summary = str(bundle.get("summary") or "").strip()
        if not summary:
            return ""
        if max_chars and len(summary) > max_chars:
            return summary[:max_chars]
        return summary


    def _extract_key_points(self, summary_or_points, limit: int = 6) -> List[str]:
        if not summary_or_points:
            return []
        if isinstance(summary_or_points, list):
            points = [str(p).strip() for p in summary_or_points if str(p).strip()]
            return points[:limit]
        summary = str(summary_or_points or "")
        points = []
        for line in summary.splitlines():
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith(("-", "*", "•")):
                points.append(raw.lstrip("-*• ").strip())
        if points:
            return points[:limit]
        # Fallback: split by sentence-ish boundaries.
        sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
        for sent in sentences:
            if sent.strip():
                points.append(sent.strip())
            if len(points) >= limit:
                break
        return points[:limit]


    def _key_points_block(self, summary_or_points) -> str:
        points = self._extract_key_points(summary_or_points, limit=10)
        if not points:
            return ""
        lines = ["Key points:"]
        for item in points:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def _run_parallel_tasks(self, tasks: list[tuple[str, callable]], max_workers: int = 3) -> list[tuple[str, object]]:
        """
        Run independent tasks in parallel; return list of (label, result_text).
        """
        if not tasks:
            return []
        results: list[tuple[str, str]] = []
        use_parallel = os.getenv("C0D3R_PARALLEL_TASKS", "1").strip().lower() not in {"0", "false", "no", "off"}
        if not use_parallel or len(tasks) == 1:
            for label, fn in tasks:
                try:
                    results.append((label, fn()))
                except Exception:
                    results.append((label, None))
            return results
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_map = {ex.submit(fn): label for label, fn in tasks}
                for fut in as_completed(future_map):
                    label = future_map[fut]
                    try:
                        results.append((label, fut.result()))
                    except Exception:
                        results.append((label, None))
        except Exception:
            for label, fn in tasks:
                try:
                    results.append((label, fn()))
                except Exception:
                    results.append((label, None))
        return results



    def _build_context_block(self, workdir: Path) -> str:
        lines = [
            "Context:",
            f"- cwd: {workdir}",
            f"- os: {os.name}",
        ]
        try:
            lines.append(f"- project_root: {workdir.resolve()}")
        except Exception:
            pass
        lines.append(self._environment_context_block(workdir))
        env_session = os.getenv("C0D3R_SESSION_ID")
        session_id = session_id or (env_session.strip() if env_session else None)
        bundle = self._load_summary_bundle(session_id=session_id) if session_id else {"summary": "", "key_points": []}
        summary = str(bundle.get("summary") or "").strip()
        if summary:
            lines.append("Rolling summary:\n" + summary)
            key_points = self._key_points_block(bundle.get("key_points") or summary)
            if key_points:
                lines.append(key_points)
        if self._auto_context_commands_enabled():
            try:
                from services.framework_catalog import detect_frameworks
                frameworks = detect_frameworks(workdir)
                if frameworks:
                    lines.append(f"- frameworks: {', '.join(frameworks)}")
                else:
                    lines.append("- frameworks: (none detected)")
            except Exception:
                lines.append("- frameworks: (unknown)")
            # Parallelize independent context probes.
            tasks = []
            # tasks.append(("git_status", lambda: run_command("git status -sb", cwd=workdir)))
            # tasks.append(("git_root", lambda: run_command("git rev-parse --show-toplevel", cwd=workdir)))
            def _ls_cmd() -> str:
                if os.name == "nt":
                    if shutil.which("pwsh") or shutil.which("powershell"):
                        return "Get-ChildItem -Name"
                    return f'{sys.executable} -c "import os;print(\'\\n\'.join(os.listdir(\'.\')))"'
                if shutil.which("ls"):
                    return "ls -1"
                return f'{sys.executable} -c "import os;print(\'\\n\'.join(os.listdir(\'.\')))"'
            # tasks.append(("ls", lambda: run_command(_ls_cmd(), cwd=workdir)))
            results = self._run_parallel_tasks([(name, fn) for name, fn in tasks], max_workers=3)
            result_map = {name: res for name, res in results}
            if "git_status" in result_map:
                code, stdout, stderr = result_map["git_status"]
                if stdout.strip():
                    lines.append("git status -sb:")
                    lines.append(stdout.strip()[:2000])
                if stderr.strip():
                    lines.append("git status stderr:")
                    lines.append(stderr.strip()[:500])
            if "git_root" in result_map:
                code, stdout, stderr = result_map["git_root"]
                if stdout.strip():
                    lines.append(f"repo root: {stdout.strip()}")
            if "ls" in result_map:
                code, stdout, stderr = result_map["ls"]
                if stdout.strip():
                    lines.append("top-level files:")
                    lines.append("\n".join(stdout.strip().splitlines()[:80]))
        return "\n".join(lines)
        
    def _step_2_inject_context(self):
        context = self._build_context_block(Path.cwd())
        print(f"{sys.argv[0]} {context}")
    def _step_3_orchestration(self):
        # Placeholder for the main orchestration logic of the process flow.
        pass
    
    def main(self):
        self._step_2_inject_context()
    
if __name__ == "__main__":
    process_flow = ProcessFlow(userRequest="Example request")
    process_flow.main()