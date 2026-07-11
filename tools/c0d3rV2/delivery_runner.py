"""tools/c0d3rV2/delivery_runner.py — C0d3rV2 runner for BrandDozer delivery context.

This is the server-side (background job) counterpart to web_runner.py.
Unlike the web runner, delivery sessions run with full tool access:
  - file_read     — read any file in the project
  - file_write    — create or patch files
  - executor      — run shell/powershell commands (tests, builds, linters)
  - web_search    — DuckDuckGo research
  - memory_search — long-term memory recall
  - equation_matrix — mathematical/scientific equations
  - file_locate   — Hazy Hash contextual file lookup

This is what replaced Codex CLI: a real agent loop that can:
  1. Read the existing codebase
  2. Plan changes
  3. Write code to files
  4. Run tests/builds to verify
  5. Iterate until done

Usage (from branddozer_delivery.py):
    from tools.c0d3rV2.delivery_runner import run_delivery_turn, probe_wizard_node

    output = run_delivery_turn(
        prompt="Add a dark mode toggle to the settings page",
        session_key=f"branddozer:{run.id}:dev",
        workdir=project_root,
        backend="wizard",
        system_context=f"Project: {run.project.name}\\nContext: {run.context}",
        reset=False,
    )
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_TOOLS_ROOT = _HERE.parent
_PROJECT_ROOT = _TOOLS_ROOT.parent
for _p in (str(_PROJECT_ROOT), str(_TOOLS_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_WEB_ROOT = _PROJECT_ROOT / "web"
if _WEB_ROOT.exists() and str(_WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(_WEB_ROOT))

_RUNTIME_ROOT = _PROJECT_ROOT / "runtime" / "c0d3rv2_delivery"

# Per-session cache: {session_key → ProcessFlow}
_FLOW_CACHE: dict[str, Any] = {}


def _build_delivery_flow(session_key: str, workdir: Path, backend: str = "wizard") -> Any:
    """Wire up a full-capability ProcessFlow for delivery/background-job use."""
    from process_flow import ProcessFlow
    from tool_registry import (
        ToolRegistry, WebSearchTool, MemorySearchTool, MatrixSearchTool,
        ExecutorTool, FileReadTool, FileWriteTool, FileLocateTool,
        DirectoryEnsureTool, WorkspaceScaffoldTool, EnvironmentBootstrapTool,
        ScientificMethodTool,
    )
    from lt_mem import LongTermMemory
    from side_load_st_mem_file_location import STSideLoadedMemory
    from side_load_lt_mem_file_location import LTSideLoadedMemory
    from web_search import WebSearch
    from executor import Executor

    session = _make_session(backend, session_key, workdir)
    rt = _RUNTIME_ROOT
    rt.mkdir(parents=True, exist_ok=True)

    lt_memory = LongTermMemory(rt)
    st_memory = STSideLoadedMemory(session_key, rt)
    lt_side_memory = LTSideLoadedMemory(rt)
    executor = Executor(workdir)

    tools = ToolRegistry()
    tools.register(FileReadTool(workdir))
    tools.register(FileWriteTool(workdir))
    tools.register(DirectoryEnsureTool(workdir))
    tools.register(WorkspaceScaffoldTool(workdir))
    web_search = WebSearch(session)
    tools.register(EnvironmentBootstrapTool(workdir))
    tools.register(ExecutorTool(executor))
    tools.register(WebSearchTool(web_search))
    tools.register(ScientificMethodTool(web_search, runtime_dir=rt))
    tools.register(MemorySearchTool(lt_memory))
    tools.register(MatrixSearchTool())
    tools.register(FileLocateTool(st_memory, lt_side_memory, workdir=workdir))

    flow = ProcessFlow(
        session=session,
        workdir=workdir,
        tools=tools,
        session_id=session_key,
        lt_memory=lt_memory,
        st_memory=st_memory,
        lt_side_memory=lt_side_memory,
    )
    return flow


def _make_session(backend: str, session_key: str, workdir: Path) -> Any:
    from tools.ai_backend_mode import freeloader_mode_active

    backend = (backend or "wizard").lower().strip()
    if freeloader_mode_active():
        backend = "freeloader"

    if backend in ("freeloader", "agentthefreeloader", "agent_the_freeloader"):
        from tools.c0d3rV2.plugins.agent_the_freeloader import AgentTheFreeloaderSession
        allowed_raw = os.getenv("C0D3R_DELIVERY_ATF_MODELS", "").strip()
        allowed = [item.strip() for item in allowed_raw.split(",") if item.strip()] or None
        return AgentTheFreeloaderSession(
            session_name=f"freeloader-delivery-{session_key[:24]}",
            transcript_dir=_RUNTIME_ROOT / "transcripts",
            workdir=workdir,
            allowed_models=allowed,
            timeout_s=float(os.getenv("C0D3R_DELIVERY_ATF_TIMEOUT_S", "30")),
            max_attempts=max(1, int(os.getenv("C0D3R_DELIVERY_ATF_ATTEMPTS", "1"))),
        )

    if backend == "wizard":
        from tools.wizard_session import WizardSession
        probe = WizardSession.probe()
        if probe["online"]:
            return WizardSession(
                session_name=f"delivery-{session_key[:24]}",
                transcript_dir=_RUNTIME_ROOT / "transcripts",
                workdir=workdir,
            )
        print(
            f"[c0d3rv2-delivery] W1z4rD node offline ({probe['error']}); "
            "falling back to Bedrock.",
            flush=True,
        )
        backend = "bedrock"

    if backend in ("bedrock", "c0d3r", "coder"):
        try:
            from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings
            settings = c0d3r_default_settings()
            for key in ("stream_default", "transcript_enabled", "event_store_enabled",
                        "diagnostics_enabled", "research_report_enabled"):
                settings.pop(key, None)
            return C0d3rSession(
                session_name=f"c0d3rv2-delivery-{session_key[:24]}",
                transcript_dir=_RUNTIME_ROOT / "transcripts",
                stream_default=False,
                transcript_enabled=False,
                event_store_enabled=False,
                diagnostics_enabled=False,
                db_sync_enabled=False,
                workdir=str(workdir),
                **settings,
            )
        except Exception as exc:
            raise RuntimeError(f"C0d3rV2 delivery runner: no AI backend available — {exc}") from exc

    raise ValueError(f"Unknown backend: {backend!r}")


def run_delivery_turn(
    prompt: str,
    *,
    session_key: str,
    workdir: Path | None = None,
    backend: str = "wizard",
    system_context: str = "",
    reset: bool = False,
) -> str:
    """
    Run one delivery turn of the C0d3rV2 agent with full file+executor access.

    The agent can read/write code files and run shell commands — equivalent to
    what Codex CLI did, but driven by the C0d3rV2 orchestrator loop so each
    tool call feeds back into the next decision.

    Returns the agent's final text output.
    """
    if workdir is None:
        workdir = _PROJECT_ROOT

    if reset and session_key in _FLOW_CACHE:
        del _FLOW_CACHE[session_key]

    flow = _FLOW_CACHE.get(session_key)
    if flow is None:
        flow = _build_delivery_flow(session_key, workdir, backend=backend)
        _FLOW_CACHE[session_key] = flow

    begin_turn = getattr(flow.session, "begin_turn", None)
    if callable(begin_turn):
        atomic = "unattended atomic workday job" in system_context.lower()
        budget_name = "C0D3R_ATOMIC_MAX_MODEL_CALLS" if atomic else "C0D3R_MAX_MODEL_CALLS"
        default_budget = "12" if atomic else "64"
        begin_turn(max(1, int(os.getenv(budget_name, default_budget))))

    if system_context and system_context.strip():
        flow._pending_system = system_context.strip()
        _patch_session_context(flow, system_context)
    else:
        flow._pending_system = ""

    augmented = flow.step_2_inject_context(prompt)
    # step_2_inject_context stores the system-only context on flow._context.
    # Do not replace it with the returned "context + user request" string or
    # every orchestrator call receives the complete user prompt twice.

    from orchestrator import Orchestrator
    from petal_system import PetalManager

    orchestrator = Orchestrator(
        session=flow.session,
        tools=flow.tools,
        context=flow._context,
        petals=flow.petals or PetalManager(),
    )
    results, tree = orchestrator.run(prompt)
    flow._last_results = results
    flow._last_tree = tree
    flow._update_memory(prompt, results, tree)

    parts = [
        r.output for r in results
        if r.output and r.output.strip() and not getattr(r, "tool_outputs", None)
    ]
    if parts:
        return "\n\n".join(parts)

    tool_events = list(tree.accumulated_results()) if tree is not None else []
    tool_summary = _delivery_tool_summary(tool_events)
    if tool_summary:
        return tool_summary
    raise RuntimeError(
        "C0d3rV2 delivery produced no user-facing result and no successful "
        "write/scaffold evidence."
    )


def _delivery_tool_summary(tool_events: list[dict]) -> str:
    """Return a local completion summary only when tools prove delivery happened."""
    successful: list[dict] = []
    for event in tool_events:
        tool = str(event.get("tool") or "")
        if tool not in {"file_write", "directory_ensure", "workspace_scaffold", "environment_bootstrap", "executor"}:
            continue
        result = event.get("result") or {}
        if result.get("error"):
            continue
        if (
            result.get("status")
            or result.get("path")
            or result.get("paths")
            or (tool == "executor" and result.get("return_code") == 0)
        ):
            successful.append(event)
    if not successful:
        return ""

    lines = ["[c0d3rv2-delivery] Completed tool-backed delivery:"]
    for event in successful[-12:]:
        tool = str(event.get("tool") or "")
        result = event.get("result") or {}
        if tool == "workspace_scaffold":
            lines.append(
                f"- workspace_scaffold: {result.get('framework_count', 0)} frameworks, "
                f"{result.get('written_count', 0)} files under {result.get('workdir', '')}"
            )
        elif tool == "environment_bootstrap":
            lines.append(
                f"- environment_bootstrap: {result.get('preset', '')} "
                f"status={result.get('status', '')} under {result.get('workdir', '')}"
            )
        elif tool == "directory_ensure":
            lines.append(f"- directory_ensure: {result.get('count', 0)} directories")
        elif tool == "file_write":
            lines.append(f"- file_write: {result.get('path', '')}")
        elif tool == "executor":
            stdout = str(result.get("stdout") or "").strip().replace("\r\n", "\n")
            first_line = stdout.splitlines()[0] if stdout else "return_code=0"
            lines.append(f"- executor: {first_line[:180]}")
    return "\n".join(lines)


def run_delivery_turn_detailed(
    prompt: str,
    *,
    session_key: str,
    workdir: Path | None = None,
    backend: str = "wizard",
    system_context: str = "",
    reset: bool = False,
) -> dict:
    """Run a turn and include ATF route metadata for durable supervisors."""
    output = run_delivery_turn(
        prompt,
        session_key=session_key,
        workdir=workdir,
        backend=backend,
        system_context=system_context,
        reset=reset,
    )
    flow = _FLOW_CACHE.get(session_key)
    session = getattr(flow, "session", None)
    route_history = list(getattr(session, "route_history", []) or [])
    selected = [
        item
        for trace in route_history
        for item in trace
        if item.get("outcome") == "selected"
    ]
    tree = getattr(flow, "_last_tree", None)
    tool_events = list(tree.accumulated_results()) if tree is not None else []
    artifact_models: list[dict] = []
    seen_artifact_models: set[tuple[str, str]] = set()
    for event in tool_events:
        if event.get("tool") != "file_write":
            continue
        result = event.get("result") or {}
        if result.get("error"):
            continue
        attribution = result.get("_attribution") or {}
        identity = (str(attribution.get("provider") or ""), str(attribution.get("model") or ""))
        if all(identity) and identity not in seen_artifact_models:
            seen_artifact_models.add(identity)
            artifact_models.append({"provider": identity[0], "model": identity[1], "phase": "artifact_write"})
    return {
        "output": output,
        "route_history": route_history,
        "session_error": str(getattr(session, "last_error", "") or ""),
        "turn_model_calls": int(getattr(session, "_turn_calls", 0) or 0),
        "artifact_models": artifact_models,
        "tool_events": [
            {
                "branch": event.get("branch"), "tool": event.get("tool"),
                "error": str((event.get("result") or {}).get("error") or "")[:1000],
                "path": str((event.get("result") or {}).get("path") or ""),
                "return_code": (event.get("result") or {}).get("return_code"),
                "status": str((event.get("result") or {}).get("status") or ""),
                "similarity": (event.get("result") or {}).get("similarity"),
                "payload_normalized": bool((event.get("result") or {}).get("payload_normalized")),
                "attribution": (event.get("result") or {}).get("_attribution") or {},
            }
            for event in tool_events[-200:]
        ],
        "models": [
            {
                "provider": item.get("provider"),
                "model": item.get("model"),
                "phase": item.get("phase", ""),
            }
            for item in selected
        ],
    }


def probe_wizard_node() -> dict:
    from tools.wizard_session import WizardSession
    return WizardSession.probe()


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
