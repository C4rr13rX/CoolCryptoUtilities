from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any


class ContextBuilder:
    """
    Step 2: Gathers and assembles the full context block injected into every
    model call.

    Context has five layers:
      Local        — fast, no model calls: system info, cwd, time/date, weather.
      Memory       — rolling summary + top-10 key points (persisted between turns).
      Transcript   — as much recent user/model exchange as fits in the char budget.
      Tools        — descriptions of every tool the Orchestrator can dispatch.
                     ALWAYS included so the AI can decide which tools to use.
      Accumulated  — every tool output from the current task tree.  This is the
                     mechanism by which tools "send feedback loops to each other."
    """

    DEFAULT_MAX_TRANSCRIPT_CHARS: int = 8000
    DEFAULT_MAX_ACCUMULATED_CHARS: int = 6000

    def __init__(
        self,
        workdir: Path,
        *,
        session_id: str | None = None,
        lt_memory: Any | None = None,
        st_memory: Any | None = None,
        tool_descriptions: list[dict] | None = None,
        summary_bundle: dict | None = None,
        accumulated_results: str = "",
        task_tree_summary: str = "",
    ) -> None:
        self.workdir = workdir
        self.session_id = session_id
        self.lt_memory = lt_memory
        self.st_memory = st_memory
        self.tool_descriptions = tool_descriptions or []
        self.summary_bundle = summary_bundle or {}
        self.accumulated_results = accumulated_results
        self.task_tree_summary = task_tree_summary

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self) -> str:
        sections: list[str] = [
            self._local_section(),
            self._memory_section(),
            self._transcript_section(),
            self._tools_section(),
            self._accumulated_section(),
            self._tree_section(),
        ]
        return "\n\n".join(s for s in sections if s.strip())

    # ------------------------------------------------------------------
    # Layer 1 — Local context (synchronous, no model calls)
    # ------------------------------------------------------------------

    def _local_section(self) -> str:
        lines = ["[System Context]"]
        lines.append(f"cwd: {self.workdir}")
        lines.append(f"os: {os.name} / {sys.platform}")
        lines.append(f"python: {sys.executable}")

        time_line = self._time_line()
        if time_line:
            lines.append(time_line)

        shells = self._available_shells()
        if shells:
            lines.append(f"shells: {', '.join(shells)}")

        tools = self._available_tools()
        for name, path in tools.items():
            lines.append(f"tool.{name}: {path or 'missing'}")

        weather = self._weather()
        if weather:
            lines.append(f"weather: {weather}")

        probe = self._system_probe()
        if probe:
            lines.extend(probe)

        return "\n".join(lines)

    @staticmethod
    def _time_line() -> str:
        try:
            now = datetime.datetime.now().astimezone()
            return (
                f"local_time: {now.strftime('%Y-%m-%d %H:%M:%S')} "
                f"({now.tzname() or ''}{now.strftime('%z') or ''})"
            )
        except Exception:
            return ""

    @staticmethod
    def _available_shells() -> list[str]:
        candidates = ["pwsh", "powershell", "cmd", "bash", "sh", "zsh"]
        return [s for s in candidates if shutil.which(s)]

    @staticmethod
    def _available_tools() -> dict[str, str]:
        names = (
            "python", "pip", "git", "node", "npm", "npx",
            "yarn", "pnpm", "uv", "rg",
        )
        return {n: shutil.which(n) or "" for n in names}

    @staticmethod
    def _weather() -> str:
        if os.getenv("C0D3R_WEATHER", "1").strip().lower() in {
            "0", "false", "no", "off",
        }:
            return ""
        url = os.getenv(
            "C0D3R_WEATHER_URL", "https://wttr.in/?format=1"
        ).strip()
        timeout = float(os.getenv("C0D3R_WEATHER_TIMEOUT_S", "3.0") or "3.0")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "c0d3r/2.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="ignore").strip()[:200]
        except Exception:
            return ""

    @staticmethod
    def _system_probe() -> list[str]:
        try:
            from services.system_probe import collect_system_probe
            probe = collect_system_probe()
            return [
                f"hostname: {probe.hostname}",
                f"cpu_count: {probe.cpu_count}",
                f"total_memory_gb: {probe.total_memory_gb:.1f}",
                f"is_admin: {probe.is_admin}",
                f"network_available: {probe.network_available}",
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Layer 2 — Memory context (rolling summary + key points)
    # ------------------------------------------------------------------

    def _memory_section(self) -> str:
        # Prefer modular STMemory if available; fall back to raw bundle.
        if self.st_memory and hasattr(self.st_memory, "build_memory_section"):
            return self.st_memory.build_memory_section()

        parts: list[str] = []

        summary = str(self.summary_bundle.get("summary") or "").strip()
        if summary:
            parts.append(f"[Rolling Summary]\n{summary}")

        key_points: list[str] = [
            str(p).strip()
            for p in (self.summary_bundle.get("key_points") or [])
            if str(p).strip()
        ]
        if key_points:
            pts = "\n".join(f"- {p}" for p in key_points[:10])
            parts.append(f"[Key Points]\n{pts}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Layer 3 — Transcript (fill as much as the char budget allows)
    # ------------------------------------------------------------------

    def _transcript_section(self) -> str:
        """Include as much recent conversation as fits in the budget."""
        max_chars = int(
            os.getenv(
                "C0D3R_TRANSCRIPT_CHARS",
                str(self.DEFAULT_MAX_TRANSCRIPT_CHARS),
            )
        )

        # Prefer modular STMemory with smart budgeting.
        if self.st_memory and hasattr(self.st_memory, "build_transcript_section"):
            return self.st_memory.build_transcript_section(budget=max_chars)

        # Fallback: read from LT memory (legacy path).
        if not self.lt_memory:
            return ""
        recent = self.lt_memory.recent(
            limit=20, session_id=self.session_id or ""
        )
        if not recent:
            return ""

        lines: list[str] = ["[Recent Transcript]"]
        char_count = 0
        for entry in reversed(recent):
            user = str(entry.get("user", "")).strip()
            model = str(entry.get("model", "")).strip()
            block = f"User: {user[:500]}\nAssistant: {model[:500]}"
            if char_count + len(block) > max_chars:
                break
            lines.append(block)
            char_count += len(block)

        return "\n".join(lines) if len(lines) > 1 else ""

    # ------------------------------------------------------------------
    # Layer 4 — Tool descriptions (ALWAYS included)
    # ------------------------------------------------------------------

    def _tools_section(self) -> str:
        if not self.tool_descriptions:
            return ""
        lines = ["[Available Tools]"]
        for tool in self.tool_descriptions:
            lines.append(
                f"- {tool.get('name', '?')}: {tool.get('description', '')}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Layer 5 — Accumulated tool results (cross-tool feedback loop)
    # ------------------------------------------------------------------

    def _accumulated_section(self) -> str:
        """
        Inject all tool outputs collected so far across the task tree.

        This is the mechanism by which tools send feedback loops to each
        other: web_search results are visible when equation_matrix runs,
        matrix results are visible when executor runs, etc.
        """
        if not self.accumulated_results:
            return ""
        return self.accumulated_results

    # ------------------------------------------------------------------
    # Layer 6 — Task tree position (branch context)
    # ------------------------------------------------------------------

    def _tree_section(self) -> str:
        if not self.task_tree_summary:
            return ""
        return f"[Task Tree]\n{self.task_tree_summary}"
