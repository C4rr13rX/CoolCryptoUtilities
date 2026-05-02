"""wizard_chat/agent_runner.py — agent-mode runner for wizard-chat.

Mirror of tools/c0d3rV2/web_runner.py, but wires in the Executor (and a
new ElevatedShellTool that uses tools/c0d3rV2/elevation.py).  This file
is what makes "agent mode" different from regular wizard chat — it gives
the C0d3rV2 orchestrator the ability to actually run shell commands on
the host instead of only doing read-only inference.

Safety model:
  • Designed for personal/intranet deployments — every agent endpoint
    that imports this is expected to enforce localhost-only itself.
  • Each shell tool call is appended to web/logs/agent_audit.log along
    with the AI's chosen command and the resulting return code.
  • Elevated commands trigger the host OS's native auth dialog (UAC,
    polkit, osascript).  No password ever flows through Python.
  • No credential caching — every elevated command requires fresh OS
    authentication.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any


# Wire up tools/c0d3rV2 onto sys.path the same way web_runner.py does.
_HERE = Path(__file__).resolve().parent
_WEB_ROOT = _HERE.parent
_PROJECT_ROOT = _WEB_ROOT.parent
_TOOLS_ROOT = _PROJECT_ROOT / "tools"
_C0D3R_ROOT = _TOOLS_ROOT / "c0d3rV2"
for _p in (str(_PROJECT_ROOT), str(_TOOLS_ROOT), str(_C0D3R_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_RUNTIME_ROOT = _PROJECT_ROOT / "runtime" / "c0d3rv2"
_AUDIT_LOG = _WEB_ROOT / "logs" / "agent_audit.log"

# In-process flow cache: {session_key → ProcessFlow}.  Separate from the
# read-only cache in web_runner.py so the two modes don't share state.
_FLOW_CACHE: dict[str, Any] = {}
_AUDIT_LOCK = threading.Lock()


# ── Audit log ───────────────────────────────────────────────────────────────

def _audit(event: str, **fields: Any) -> None:
    """Append a JSON line to web/logs/agent_audit.log.  Best-effort:
    if the directory or file can't be written, fail silently rather than
    blocking the agent."""
    try:
        _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event,
            **fields,
        }
        with _AUDIT_LOCK:
            with _AUDIT_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


# ── Shell tools (wrap the Executor so every call is audited) ────────────────

def _build_shell_tools(workdir: Path, session_key: str) -> tuple[Any, Any]:
    """Return (regular_shell_tool, admin_shell_tool).  Both wrap an
    ElevatedExecutor that audits every command."""
    from elevated_executor import ElevatedExecutor
    from tool_registry import Tool

    executor = ElevatedExecutor(workdir)

    class AuditedShellTool(Tool):
        name = "executor"
        description = (
            "Run a non-privileged terminal command (PowerShell on Windows, "
            "bash/sh on Linux/macOS).  Use for reading files, running "
            "scripts, listing directories, calling installed CLIs, etc.  "
            "Returns stdout, stderr, and return code.  For commands that "
            "require admin/root (installing system packages, writing to "
            "system directories, modifying services), use shell_admin "
            "instead."
        )
        use_when = (
            "Use for any OS-level task that does NOT require admin/root.  "
            "Inspect files, run tests, call git, build projects, query "
            "system state.  If a command fails with a permissions error, "
            "consider whether shell_admin is appropriate."
        )
        params_schema = {"command": "str — the shell command to execute"}

        def execute(self, params: dict) -> dict:
            command = str(params.get("command", ""))
            if not command:
                return {"error": "No command provided"}
            code, stdout, stderr = executor.run(command, elevate=False)
            _audit("shell_run", session=session_key, elevated=False,
                   command=command, return_code=code,
                   stdout_len=len(stdout), stderr_len=len(stderr))
            return {"return_code": code, "stdout": stdout, "stderr": stderr}

    class AuditedAdminShellTool(Tool):
        name = "shell_admin"
        description = (
            "Run a shell command with ELEVATED privileges (admin/root).  "
            "On Windows this triggers a UAC dialog the user must approve.  "
            "On Linux this triggers the polkit dialog (pkexec) or a "
            "graphical sudo askpass.  On macOS this triggers the system "
            "auth dialog via osascript.  In every case the host OS asks "
            "the user for credentials directly — this tool never sees the "
            "password.  Use sparingly: only for tasks that genuinely need "
            "admin (installing system packages, writing to /etc, "
            "modifying registry, restarting system services)."
        )
        use_when = (
            "Use ONLY when the task cannot be done without admin/root: "
            "  • installing OS-level packages (apt, dnf, brew, choco, winget)\n"
            "  • writing to system config (/etc, %WINDIR%, system Library)\n"
            "  • starting/stopping system services\n"
            "  • modifying firewall/network config\n"
            "  • changing file ownership outside the user's home\n"
            "Always prefer the regular executor when possible — it does "
            "not require user authentication and does not interrupt the "
            "user with a password prompt."
        )
        params_schema = {
            "command": "str — the shell command to execute as admin",
            "reason": "str — short explanation of why admin is required, "
                      "shown to the user via the audit log",
        }

        def execute(self, params: dict) -> dict:
            command = str(params.get("command", ""))
            reason = str(params.get("reason", ""))
            if not command:
                return {"error": "No command provided"}
            _audit("shell_admin_request", session=session_key,
                   command=command, reason=reason,
                   method=executor.elevation_method())
            code, stdout, stderr = executor.run(command, elevate=True)
            _audit("shell_admin_run", session=session_key,
                   command=command, reason=reason, return_code=code,
                   stdout_len=len(stdout), stderr_len=len(stderr),
                   cancelled=(code == 126))
            return {
                "return_code": code,
                "stdout": stdout,
                "stderr": stderr,
                "elevation_method": executor.elevation_method(),
                "user_cancelled": code == 126,
            }

    return AuditedShellTool(), AuditedAdminShellTool()


# ── Flow construction ───────────────────────────────────────────────────────

def _make_session(backend: str, session_key: str) -> Any:
    """Same backend selection as web_runner._make_session — wizard first,
    bedrock fallback.  Duplicated here to avoid coupling the agent flow to
    the read-only flow's import paths."""
    backend = (backend or "wizard").lower().strip()
    if backend == "wizard":
        from tools.wizard_session import WizardSession
        probe = WizardSession.probe()
        if probe["online"]:
            return WizardSession(
                session_name=f"agent-{session_key[:16]}",
                transcript_dir=_RUNTIME_ROOT / "transcripts",
            )
        backend = "bedrock"
    if backend in ("bedrock", "c0d3r", "coder"):
        from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings
        settings = c0d3r_default_settings()
        for key in ("stream_default", "transcript_enabled", "event_store_enabled",
                    "diagnostics_enabled", "research_report_enabled"):
            settings.pop(key, None)
        return C0d3rSession(
            session_name=f"c0d3rv2-agent-{session_key[:16]}",
            transcript_dir=_RUNTIME_ROOT / "transcripts",
            stream_default=False,
            transcript_enabled=False,
            event_store_enabled=False,
            diagnostics_enabled=False,
            db_sync_enabled=False,
            **settings,
        )
    raise ValueError(f"Unknown backend: {backend!r}.")


def _build_flow(session_key: str, workdir: Path, backend: str,
                  allow_admin: bool) -> Any:
    """Same as web_runner._build_flow but with shell tools added (and
    optionally the admin variant)."""
    from process_flow import ProcessFlow
    from tool_registry import ToolRegistry, WebSearchTool, MemorySearchTool, MatrixSearchTool
    from lt_mem import LongTermMemory
    from web_search import WebSearch

    session = _make_session(backend, session_key)
    rt = _RUNTIME_ROOT
    rt.mkdir(parents=True, exist_ok=True)
    lt_memory = LongTermMemory(rt)

    tools = ToolRegistry()
    tools.register(WebSearchTool(WebSearch(session)))
    tools.register(MemorySearchTool(lt_memory))
    tools.register(MatrixSearchTool())

    shell_tool, admin_tool = _build_shell_tools(workdir, session_key)
    tools.register(shell_tool)
    if allow_admin:
        tools.register(admin_tool)

    flow = ProcessFlow(
        session=session,
        workdir=workdir,
        tools=tools,
        session_id=session_key,
        lt_memory=lt_memory,
    )
    _audit("flow_built", session=session_key, backend=backend,
           workdir=str(workdir), admin_enabled=allow_admin)
    return flow


# ── Public entry point ──────────────────────────────────────────────────────

def run(
    prompt: str,
    *,
    session_key: str,
    workdir: Path | None = None,
    backend: str = "wizard",
    system_context: str = "",
    reset: bool = False,
    allow_admin: bool = False,
) -> str:
    """Run a single agent turn with executor (and optionally shell_admin)
    available as tools.

    session_key   Unique key per user+session.
    workdir       Defaults to the CoolCryptoUtilities project root.
    backend       "wizard" or "bedrock"; falls through if wizard offline.
    allow_admin   If True, the shell_admin tool is registered so the AI
                  can request OS-level elevation.  Each elevated command
                  still requires fresh user authentication via UAC/polkit/
                  osascript.
    """
    if workdir is None:
        workdir = _PROJECT_ROOT

    if reset and session_key in _FLOW_CACHE:
        _audit("flow_reset", session=session_key)
        del _FLOW_CACHE[session_key]

    flow = _FLOW_CACHE.get(session_key)
    # Rebuild the flow if the admin-enabled flag flipped since last call —
    # we don't want a stale flow keeping the admin tool registered after
    # the user toggled the UI off.
    if flow is not None and getattr(flow, "_agent_admin_enabled", None) != allow_admin:
        _audit("flow_rebuild_admin_toggle", session=session_key,
               new_admin=allow_admin)
        del _FLOW_CACHE[session_key]
        flow = None
    if flow is None:
        flow = _build_flow(session_key, workdir, backend, allow_admin)
        flow._agent_admin_enabled = allow_admin
        _FLOW_CACHE[session_key] = flow

    if system_context and system_context.strip():
        flow._pending_system = system_context.strip()
    else:
        flow._pending_system = ""
    _patch_session_context(flow, system_context)

    augmented = flow.step_2_inject_context(prompt)
    flow._context = augmented

    from orchestrator import Orchestrator
    from petal_system import PetalManager

    orchestrator = Orchestrator(
        session=flow.session,
        tools=flow.tools,
        context=flow._context,
        petals=flow.petals or PetalManager(),
    )
    _audit("agent_turn_start", session=session_key,
           prompt_len=len(prompt), admin_enabled=allow_admin)
    results, tree = orchestrator.run(prompt)
    flow._update_memory(prompt, results, tree)
    _audit("agent_turn_end", session=session_key,
           result_count=len(results))

    parts = [r.output for r in results if r.output and r.output.strip()]
    return "\n\n".join(parts) if parts else (
        "[c0d3rv2-agent] Task complete — no text output generated."
    )


def _patch_session_context(flow: Any, system_context: str) -> None:
    if not system_context or not system_context.strip():
        return
    original_send = flow.session.send

    def _wrapped_send(prompt, *, stream=False, system="", **kwargs):
        combined = system_context.strip()
        if system:
            combined = f"{combined}\n\n{system}"
        return original_send(prompt, stream=stream, system=combined, **kwargs)

    flow.session.send = _wrapped_send


def elevation_method() -> str:
    """Expose the current host's elevation mechanism for UI display."""
    from elevated_executor import ElevatedExecutor
    return ElevatedExecutor(_PROJECT_ROOT).elevation_method()
