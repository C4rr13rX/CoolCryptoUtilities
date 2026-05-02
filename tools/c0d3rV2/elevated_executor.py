"""tools/c0d3rV2/elevated_executor.py — Executor with optional UAC/sudo path.

Wraps the standard Executor.  When a command is run with elevate=True,
spawns it via tools/c0d3rV2/elevation.py which uses the host OS's native
authentication dialog (UAC, polkit, osascript).  No password is ever
captured by this code — the OS asks the user and decides.
"""
from __future__ import annotations

from pathlib import Path

from executor import Executor
import elevation


class ElevatedExecutor(Executor):
    """Executor variant that supports `elevate=True` on each .run() call.

    Non-elevated runs delegate to the parent Executor unchanged, so this
    is a pure superset — wherever a standard Executor is expected, an
    ElevatedExecutor can take its place.
    """

    def run(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        elevate: bool = False,
    ) -> tuple[int, str, str]:
        if not elevate:
            return super().run(command, cwd=cwd)
        try:
            return elevation.run_elevated(
                command,
                cwd=cwd or self.workdir,
                timeout_s=self.timeout_s,
            )
        except elevation.ElevationUnavailable as exc:
            return 1, "", f"elevation unavailable: {exc}"

    def elevation_method(self) -> str:
        return elevation.elevation_method()
