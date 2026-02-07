from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

NOTES_ROOT = Path("runtime/branddozer/notes")


@dataclass
class WorkspaceContext:
    root: Path
    os_name: str
    shell: str
    python: str
    notes_path: Path


def detect_shell() -> str:
    if os.name == "nt":
        if shutil.which("pwsh"):
            return "pwsh"
        if shutil.which("powershell"):
            return "powershell"
        return "cmd"
    shell = os.getenv("SHELL")
    if shell and Path(shell).exists():
        return shell
    if shutil.which("bash"):
        return "bash"
    if shutil.which("sh"):
        return "sh"
    return shell or "bash"


def build_context(root: Path, *, notes_name: str) -> WorkspaceContext:
    shell = detect_shell()
    python = os.getenv("VIRTUAL_ENV") or ""
    if python:
        if os.name == "nt":
            python = str(Path(python) / "Scripts" / "python.exe")
        else:
            python = str(Path(python) / "bin" / "python")
    else:
        python = "python" if os.name == "nt" else "python3"
    notes_path = NOTES_ROOT / f"{notes_name}.md"
    return WorkspaceContext(
        root=root,
        os_name=platform.system(),
        shell=shell,
        python=python,
        notes_path=notes_path,
    )


def init_notes(context: WorkspaceContext, *, extra: Optional[List[str]] = None) -> None:
    NOTES_ROOT.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Agent Notes ({context.root})",
        "",
        f"- OS: {context.os_name}",
        f"- Shell: {context.shell}",
        f"- Python: {context.python}",
        f"- Initialized: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Current state",
        "- Pending:",
        "  - [ ] Review project structure",
        "  - [ ] Run smoke tests (if configured)",
        "",
    ]
    if extra:
        lines.extend(extra)
        lines.append("")
    context.notes_path.write_text("\n".join(lines), encoding="utf-8")


def append_notes(context: WorkspaceContext, text: str) -> None:
    NOTES_ROOT.mkdir(parents=True, exist_ok=True)
    with context.notes_path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def read_notes(context: WorkspaceContext) -> str:
    if not context.notes_path.exists():
        return ""
    return context.notes_path.read_text(encoding="utf-8", errors="ignore")


def run_command(
    command: str,
    *,
    cwd: Optional[Path] = None,
    timeout_s: int = 900,
    allow_shell: bool = True,
) -> Tuple[int, str, str]:
    if not command:
        return 1, "", "empty command"
    if os.name == "nt":
        if allow_shell:
            if shutil.which("pwsh"):
                cmd = ["pwsh", "-NoProfile", "-Command", command]
            elif shutil.which("powershell"):
                cmd = ["powershell", "-NoProfile", "-Command", command]
            else:
                cmd = ["cmd", "/c", command]
        else:
            cmd = shlex.split(command)
    else:
        if allow_shell:
            shell = os.getenv("SHELL")
            if shell and Path(shell).exists():
                cmd = [shell, "-lc", command]
            elif shutil.which("bash"):
                cmd = ["bash", "-lc", command]
            else:
                cmd = ["sh", "-lc", command]
        else:
            cmd = shlex.split(command)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


__all__ = ["WorkspaceContext", "build_context", "init_notes", "append_notes", "read_notes", "run_command"]
