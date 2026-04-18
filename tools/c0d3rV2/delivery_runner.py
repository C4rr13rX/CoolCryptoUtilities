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
    )
    from lt_mem import LongTermMemory
    from st_memory import STMemory
    from web_search import WebSearch
    from executor import Executor

    session = _make_session(backend, session_key, workdir)
    rt = _RUNTIME_ROOT
    rt.mkdir(parents=True, exist_ok=True)

    lt_memory = LongTermMemory(rt)
    st_memory = STMemory()
    executor = Executor(workdir)

    tools = ToolRegistry()
    tools.register(FileReadTool(workdir))
    tools.register(FileWriteTool(workdir))
    tools.register(ExecutorTool(executor))
    tools.register(WebSearchTool(WebSearch(session)))
    tools.register(MemorySearchTool(lt_memory))
    tools.register(MatrixSearchTool())
    tools.register(FileLocateTool(st_memory, lt_memory))

    flow = ProcessFlow(
        session=session,
        workdir=workdir,
        tools=tools,
        session_id=session_key,
        lt_memory=lt_memory,
    )
    return flow


def _make_session(backend: str, session_key: str, workdir: Path) -> Any:
    backend = (backend or "wizard").lower().strip()

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

    if system_context and system_context.strip():
        flow._pending_system = system_context.strip()
        _patch_session_context(flow, system_context)
    else:
        flow._pending_system = ""

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
    results, tree = orchestrator.run(prompt)
    flow._update_memory(prompt, results, tree)

    parts = [r.output for r in results if r.output and r.output.strip()]
    return "\n\n".join(parts) if parts else "[c0d3rv2-delivery] Turn complete — no text output."


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
