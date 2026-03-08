from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class Executor:
    """
    Runs shell / PowerShell / cmd commands on behalf of the model.

    Commands come from the AI.  This is the only class that touches the OS
    to execute them.  Results are returned as (return_code, stdout, stderr)
    tuples — nothing is printed here.
    """

    DEFAULT_TIMEOUT_S: int = 120

    def __init__(self, workdir: Path, timeout_s: int | None = None) -> None:
        self.workdir = workdir
        self.timeout_s = timeout_s or self.DEFAULT_TIMEOUT_S

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, command: str, *, cwd: Path | None = None) -> tuple[int, str, str]:
        """Execute a command and return (return_code, stdout, stderr)."""
        effective_cwd = cwd or self.workdir
        if os.name == "nt":
            return self._run_windows(command, cwd=effective_cwd)
        return self._run_posix(command, cwd=effective_cwd)

    # ------------------------------------------------------------------
    # Platform runners
    # ------------------------------------------------------------------

    def _run_posix(self, command: str, *, cwd: Path) -> tuple[int, str, str]:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {self.timeout_s}s"
        except Exception as exc:
            return 1, "", str(exc)

    def _run_windows(self, command: str, *, cwd: Path) -> tuple[int, str, str]:
        """Prefer pwsh -> powershell -> cmd. Always passes -NoProfile."""
        if shutil.which("pwsh"):
            shell_cmd = ["pwsh", "-NoProfile", "-Command", command]
        elif shutil.which("powershell"):
            shell_cmd = ["powershell", "-NoProfile", "-Command", command]
        else:
            shell_cmd = ["cmd", "/c", command]
        try:
            result = subprocess.run(
                shell_cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                encoding="utf-8",
                errors="replace",
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {self.timeout_s}s"
        except Exception as exc:
            return 1, "", str(exc)
