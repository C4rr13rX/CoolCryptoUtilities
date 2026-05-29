"""tools/c0d3rV2/web_runner.py — C0d3rV2 runner for Django web context.

Strips the CLI-specific concerns (TUI, header, PTY, stdin) and exposes a
simple synchronous run(prompt) → str interface suitable for calling from a
Django view.

The runner creates a lightweight ProcessFlow:
  - AI backend: WizardSession against the merged W1z4rDV1510n main
    node (localhost:8090 by default, routed through /brain/chat) by
    default; falls back to Bedrock C0d3rSession if the wizard node
    is unset or offline.  Subsequent fallbacks: Claude (Anthropic
    API) and OpenAI.  The brain may be a §18 cluster head — the
    same URL transparently uses the ring.
  - Tools available in web context: web_search, memory_search,
    equation_matrix.  The executor and vm_playground tools are omitted
    because they could allow arbitrary shell execution via the web UI.
  - LT memory is loaded from disk (per-user session_id) so memory builds up
    across web turns.
  - Session state (summary bundle) is cached in-process by session_key so
    subsequent turns have rolling context.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Ensure V2 package and project root are importable from any call site.
_HERE = Path(__file__).resolve().parent
_TOOLS_ROOT = _HERE.parent
_PROJECT_ROOT = _TOOLS_ROOT.parent
for _p in (str(_PROJECT_ROOT), str(_TOOLS_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_WEB_ROOT = _PROJECT_ROOT / "web"
if _WEB_ROOT.exists() and str(_WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(_WEB_ROOT))

_RUNTIME_ROOT = _PROJECT_ROOT / "runtime" / "c0d3rv2"

# In-process cache: {session_key → ProcessFlow}
_FLOW_CACHE: dict[str, Any] = {}


def _build_flow(session_key: str, workdir: Path, backend: str = "wizard") -> Any:
    """Wire up a ProcessFlow for web use — one per user+session."""
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

    flow = ProcessFlow(
        session=session,
        workdir=workdir,
        tools=tools,
        session_id=session_key,
        lt_memory=lt_memory,
    )
    return flow


def _make_session(backend: str, session_key: str) -> Any:
    """
    Create an AI session for the given backend preference.

    Cascading fallback when the request is "auto" or the preferred
    backend is unavailable: wizard → bedrock → claude → openai.
    """
    from tools.ai_session import resolve_with_fallback
    chosen = resolve_with_fallback(backend)

    if chosen == "openai":
        from tools.openai_session import OpenAISession
        return OpenAISession(
            session_name=f"c0d3rv2-web-{session_key[:16]}",
            transcript_dir=_RUNTIME_ROOT / "transcripts",
            transcript_enabled=False,
        )

    if chosen == "claude":
        from tools.claude_session import ClaudeSession
        return ClaudeSession(
            session_name=f"c0d3rv2-web-{session_key[:16]}",
            transcript_dir=_RUNTIME_ROOT / "transcripts",
            transcript_enabled=False,
        )

    if chosen == "wizard":
        from tools.wizard_session import WizardSession
        return WizardSession(
            session_name=f"web-{session_key[:16]}",
            transcript_dir=_RUNTIME_ROOT / "transcripts",
        )

    # bedrock (default fallback when nothing else available)
    try:
        from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings
        settings = c0d3r_default_settings()
        for key in ("stream_default", "transcript_enabled", "event_store_enabled",
                    "diagnostics_enabled", "research_report_enabled"):
            settings.pop(key, None)
        return C0d3rSession(
            session_name=f"c0d3rv2-web-{session_key[:16]}",
            transcript_dir=_RUNTIME_ROOT / "transcripts",
            stream_default=False,
            transcript_enabled=False,
            event_store_enabled=False,
            diagnostics_enabled=False,
            db_sync_enabled=False,
            **settings,
        )
    except Exception as exc:
        raise RuntimeError(f"C0d3rV2 web runner: no AI backend available — {exc}") from exc


def run(
    prompt: str,
    *,
    session_key: str,
    workdir: Path | None = None,
    backend: str = "wizard",
    system_context: str = "",
    reset: bool = False,
) -> str:
    """
    Run the C0d3rV2 agent with the given prompt.

    session_key  Unique key per user+session for process flow caching.
    workdir      Project root (defaults to CoolCryptoUtilities root).
    backend      "wizard" (default), "bedrock", or "openai".
    system_context  Extra context prepended by the view (conversation history,
                    user metadata, etc.).
    reset        If True, clear the cached ProcessFlow for this session.

    Returns the agent's final output as a single string.
    """
    if workdir is None:
        workdir = _PROJECT_ROOT

    if reset and session_key in _FLOW_CACHE:
        del _FLOW_CACHE[session_key]

    flow = _FLOW_CACHE.get(session_key)
    if flow is None:
        flow = _build_flow(session_key, workdir, backend=backend)
        _FLOW_CACHE[session_key] = flow

    # Inject system context into the prompt so it reaches the session.send()
    # system parameter via ProcessFlow → Orchestrator.
    if system_context and system_context.strip():
        flow._pending_system = system_context.strip()
    else:
        flow._pending_system = ""

    # Patch the session to carry system context on this call.
    _patch_session_context(flow, system_context)

    # Run a single turn (non-interactive, not REPL).
    augmented = flow.step_2_inject_context(prompt)
    flow._context = augmented  # also update on flow for orchestrator

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

    # Collect output
    parts = [r.output for r in results if r.output and r.output.strip()]
    return "\n\n".join(parts) if parts else (
        "[c0d3rv2] Task complete — no text output generated."
    )


def probe_wizard_node() -> dict:
    """Utility for views and health checks."""
    from tools.wizard_session import WizardSession
    return WizardSession.probe()


def _patch_session_context(flow: Any, system_context: str) -> None:
    """
    Wrap the flow's session.send() to always inject system_context on the
    next call.  This is a non-invasive shim — we replace .send temporarily
    for the duration of this web turn.
    """
    if not system_context or not system_context.strip():
        return
    original_send = flow.session.send

    def _wrapped_send(prompt, *, stream=False, system="", **kwargs):
        combined_system = system_context.strip()
        if system:
            combined_system = f"{combined_system}\n\n{system}"
        return original_send(prompt, stream=stream, system=combined_system, **kwargs)

    flow.session.send = _wrapped_send
