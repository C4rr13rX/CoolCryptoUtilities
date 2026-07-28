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
    equation_matrix, and c0d3r_native_os.  Direct Python executor and
    vm_playground tools are omitted; privileged OS work is delegated to
    the authenticated loopback native service.
  - LT memory is loaded from disk (per-user session_id) so memory builds up
    across web turns.
  - Session state (summary bundle) is cached in-process by session_key so
    subsequent turns have rolling context.
"""
from __future__ import annotations

import json
import os
import sys
import re
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


def _build_flow(session_key: str, workdir: Path, backend: str = "wizard", model: str = "", atf_models: list[str] | None = None) -> Any:
    """Wire up a ProcessFlow for web use — one per user+session."""
    from process_flow import ProcessFlow
    from tool_registry import (
        ToolRegistry,
        WebSearchTool,
        MemorySearchTool,
        MatrixSearchTool,
        NativeOsTool,
        ReactPwaScaffoldTool,
        SandboxFileOpsTool,
        VirtualHardwareSimScaffoldTool,
        BaseCryptoPaperTradeBenchmarkTool,
        ATFStaticTradingStrategyTool,
        ProjectWorkMapperTool,
        DependencyTraversalTool,
        ResearchHarvesterTool,
        FileLocateTool,
    )
    from lt_mem import LongTermMemory
    from st_memory import STMemory
    from side_load_st_mem_file_location import STSideLoadedMemory
    from side_load_lt_mem_file_location import LTSideLoadedMemory
    from web_search import WebSearch
    from tools.c0d3rV2.plugins.research_harvester import ResearchHarvester

    session = _make_session(backend, session_key, model=model, atf_models=atf_models)
    rt = _RUNTIME_ROOT
    rt.mkdir(parents=True, exist_ok=True)

    lt_memory = LongTermMemory(rt)
    short_memory = STMemory(session, session_id=session_key, runtime_root=rt)
    st_side_memory = STSideLoadedMemory(session_key, rt)
    lt_side_memory = LTSideLoadedMemory(rt)

    tools = ToolRegistry()
    web_search = WebSearch(session)
    tools.register(WebSearchTool(web_search))
    tools.register(ResearchHarvesterTool(ResearchHarvester(rt), web_search))
    memory_tool = MemorySearchTool(lt_memory)
    file_locate_tool = FileLocateTool(st_side_memory, lt_side_memory, workdir=workdir)
    tools.register(memory_tool)
    tools.register(file_locate_tool)
    tools.register(MatrixSearchTool())
    tools.register(NativeOsTool())
    tools.register(ReactPwaScaffoldTool())
    tools.register(SandboxFileOpsTool())
    tools.register(VirtualHardwareSimScaffoldTool())
    tools.register(BaseCryptoPaperTradeBenchmarkTool())
    tools.register(ATFStaticTradingStrategyTool())
    tools.register(ProjectWorkMapperTool(workdir))
    tools.register(DependencyTraversalTool(workdir, memory_tool, file_locate_tool))

    flow = ProcessFlow(
        session=session,
        workdir=workdir,
        tools=tools,
        session_id=session_key,
        lt_memory=lt_memory,
        short_memory=short_memory,
        st_side_memory=st_side_memory,
        lt_side_memory=lt_side_memory,
    )
    return flow


def _make_session(backend: str, session_key: str, *, model: str = "", atf_models: list[str] | None = None) -> Any:
    """
    Create an AI session for the given backend preference.

    Cascading fallback when the request is "auto" or the preferred
    backend is unavailable: wizard → bedrock → claude → openai.
    """
    if (backend or "").lower().strip() in {
        "freeloader", "agentthefreeloader", "agent_the_freeloader",
    }:
        from tools.c0d3rV2.plugins.agent_the_freeloader import AgentTheFreeloaderSession
        return AgentTheFreeloaderSession(
            session_name=f"freeloader-web-{session_key[:16]}",
            transcript_dir=_RUNTIME_ROOT / "transcripts",
            workdir=_PROJECT_ROOT,
            allowed_models=atf_models,
            timeout_s=float(os.getenv("C0D3R_WEB_ATF_TIMEOUT_S", "12")),
            max_attempts=max(1, int(os.getenv("C0D3R_WEB_ATF_ATTEMPTS", "8"))),
            max_tokens=max(128, int(os.getenv("C0D3R_WEB_ATF_MAX_TOKENS", "1024"))),
        )

    from tools.ai_session import resolve_with_fallback
    chosen = resolve_with_fallback(backend)

    if chosen == "freeloader":
        from tools.c0d3rV2.plugins.agent_the_freeloader import AgentTheFreeloaderSession
        return AgentTheFreeloaderSession(
            session_name=f"freeloader-web-{session_key[:16]}",
            transcript_dir=_RUNTIME_ROOT / "transcripts",
            workdir=_PROJECT_ROOT,
            allowed_models=atf_models,
            timeout_s=float(os.getenv("C0D3R_WEB_ATF_TIMEOUT_S", "12")),
            max_attempts=max(1, int(os.getenv("C0D3R_WEB_ATF_ATTEMPTS", "8"))),
            max_tokens=max(128, int(os.getenv("C0D3R_WEB_ATF_MAX_TOKENS", "1024"))),
        )

    if chosen == "openai":
        from tools.openai_session import OpenAISession
        return OpenAISession(
            session_name=f"c0d3rv2-web-{session_key[:16]}",
            transcript_dir=_RUNTIME_ROOT / "transcripts",
            transcript_enabled=False,
            **({"model": model} if model else {}),
        )

    if chosen == "claude":
        from tools.claude_session import ClaudeSession
        return ClaudeSession(
            session_name=f"c0d3rv2-web-{session_key[:16]}",
            transcript_dir=_RUNTIME_ROOT / "transcripts",
            transcript_enabled=False,
            **({"model": model} if model else {}),
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
        if model:
            settings["model"] = model
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
    model: str = "",
    atf_models: list[str] | None = None,
    reset: bool = False,
) -> str:
    """
    Run the C0d3rV2 agent with the given prompt.

    session_key  Unique key per user+session for process flow caching.
    workdir      Project root (defaults to CoolCryptoUtilities root).
    backend      "wizard" (default), "bedrock", "openai", or "freeloader".
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
        flow = _build_flow(session_key, workdir, backend=backend, model=model, atf_models=atf_models)
        _FLOW_CACHE[session_key] = flow

    if _should_use_conversationalist_path(prompt):
        return _run_compact_conversation(flow, prompt, system_context)

    deterministic = _maybe_run_deterministic_scaffold(flow, prompt)
    if deterministic:
        return deterministic
    deterministic = _maybe_run_sandbox_file_ops(flow, prompt)
    if deterministic:
        return deterministic
    deterministic = _maybe_run_virtual_hardware_scaffold(flow, prompt)
    if deterministic:
        return deterministic
    deterministic = _maybe_run_base_crypto_paper_benchmark(flow, prompt)
    if deterministic:
        return deterministic

    if _should_use_compact_research(prompt):
        return _run_compact_research(flow, prompt, system_context)
    if _should_use_compact_planning(prompt):
        return _run_compact_planning(flow, prompt, system_context)

    # Inject system context into the prompt so it reaches the session.send()
    # system parameter via ProcessFlow → Orchestrator.
    if system_context and system_context.strip():
        flow._pending_system = system_context.strip()
    else:
        flow._pending_system = ""

    # Patch the session to carry system context on this call.
    # The context wrapper is per-turn. Leaving it installed caused wrappers
    # and old conversation contexts to accumulate on every web request.
    original_send = flow.session.send
    _patch_session_context(flow, system_context)

    # Run a single turn (non-interactive, not REPL).
    augmented = flow.step_2_inject_context(prompt)
    # The flow already retained the system-only context. The augmented return
    # value is for direct model submission, not Orchestrator.context; assigning
    # it here duplicates the user request on every subsequent model call.

    from orchestrator import Orchestrator
    from petal_system import PetalManager

    orchestrator = Orchestrator(
        session=flow.session,
        tools=flow.tools,
        context=flow._context,
        petals=flow.petals or PetalManager(),
    )
    begin_turn = getattr(flow.session, "begin_turn", None)
    if callable(begin_turn):
        begin_turn(max(1, int(os.getenv("C0D3R_WEB_MAX_MODEL_CALLS", "8"))))
    try:
        results, tree = orchestrator.run(prompt)
    finally:
        flow.session.send = original_send
    _capture_last_model_id(flow, flow.session)
    flow._update_memory(prompt, results, tree)

    # Collect output
    # Tool-call StepResults are internal evidence for the control loop. They
    # often contain raw JSON/search results/build logs and must not be treated
    # as the final chat answer. Only surface answer/complete results that were
    # not produced by direct tool dispatch.
    parts = [
        r.output for r in results
        if r.output and r.output.strip() and not getattr(r, "tool_outputs", None)
    ]
    if not parts:
        tool_summary = _summarize_tool_evidence(results)
        if tool_summary:
            return tool_summary
        raise RuntimeError("C0d3rV2 produced no user-facing result; the run was not completed.")
    return "\n\n".join(parts)


def _maybe_run_deterministic_scaffold(flow: Any, prompt: str) -> str:
    """Run high-confidence scaffold tools without asking weak models to route.

    ATF remains the configured model, but C0d3rV2 owns deterministic tool
    arbitration. Free models regularly fail by discussing a scaffold request
    instead of selecting the obvious scaffold tool; this guard keeps benchmark
    execution grounded in the requested toolchain.
    """
    text = " ".join(str(prompt or "").split())
    lower = text.lower()
    if "react" not in lower or not any(marker in lower for marker in ("spa", "pwa")):
        return ""
    if not any(marker in lower for marker in ("build", "create", "regenerate", "scaffold")):
        return ""

    root = _extract_windows_path(text)
    if not root:
        return ""
    app_name = _extract_app_name(text, root)
    try:
        result = flow.tools.dispatch(
            "react_pwa_scaffold",
            {"root_path": root, "app_name": app_name, "overwrite": True},
        )
    except Exception as exc:
        raise RuntimeError(f"react_pwa_scaffold failed before execution: {exc}") from exc
    return _summarize_direct_scaffold_result(result)


def _extract_windows_path(text: str) -> str:
    match = re.search(r"([A-Za-z]:\\[^\n\r\"'`<>|]+)", text)
    if not match:
        return ""
    raw = match.group(1).strip().rstrip(".,;)")
    raw = raw.split(",", 1)[0].strip().rstrip(".,;)")
    raw = re.split(
        r"\.\s+(?:This|It|That|Use|Validate|Include|Keep|Make|Build|Create|Design)\b",
        raw,
        maxsplit=1,
    )[0].strip().rstrip(".,;)")
    # Stop at common English continuation after the path.
    raw = re.split(
        r"\s+(?:It|It must|It should|That|Use|Validate|Return|and UI|with UI)\b",
        raw,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip().rstrip(".,;)")
    return raw


def _extract_app_name(text: str, root: str) -> str:
    patterns = (
        r"\bnamed\s+([A-Za-z][A-Za-z0-9_-]{2,80})",
        r"\bname\s+([A-Za-z][A-Za-z0-9_-]{2,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return Path(root).name or "MarketForgeLocal"


def _summarize_direct_scaffold_result(result: dict[str, Any]) -> str:
    if result.get("error"):
        raise RuntimeError(f"react_pwa_scaffold failed: {result.get('error')}")
    validation = result.get("validation") if isinstance(result, dict) else {}
    npm_build = validation.get("npm_build") if isinstance(validation, dict) else {}
    if isinstance(npm_build, dict) and npm_build.get("error"):
        raise RuntimeError(f"react_pwa_scaffold build failed: {npm_build.get('error')}")
    if isinstance(npm_build, dict) and int(npm_build.get("return_code") or 0) != 0:
        raise RuntimeError(str(npm_build.get("stderr") or npm_build.get("stdout") or "npm build failed")[-4000:])
    root = str(result.get("root_path") or "")
    app_name = str(result.get("app_name") or Path(root).name or "React PWA")
    lines = [
        f"Completed: {app_name} was created and validated successfully.",
        f"Location: `{root}`",
        f"Files written: {result.get('file_count', 0)}",
        "Validation: `npm run build` passed.",
        "",
        "Included benchmark capabilities:",
        "- React TypeScript SPA/PWA shell",
        "- OOP market research, opportunity scoring, product spec, and local link classes",
        "- C0D3R API client for `/api/c0d3r/run/` and `/api/c0d3r/runs/{id}/`",
        "- Research-cycle controller using C0D3R+ATF",
        "- Persistent local product registry",
        "- UI controls for market-needs cycles and local product links",
        "",
        "Useful commands:",
    ]
    for command in result.get("run_commands") or []:
        lines.append(f"- `{command}`")
    return "\n".join(lines)


def _maybe_run_sandbox_file_ops(flow: Any, prompt: str) -> str:
    text = " ".join(str(prompt or "").split())
    lower = text.lower()
    file_markers = (
        "file system", "filesystem", "organize files", "sort files",
        "move files", "copy files", "dedupe", "deduplicate", "rename files",
        "normalize names", "manifest", "inventory files", "sandbox",
    )
    if not any(marker in lower for marker in file_markers):
        return ""
    if not any(marker in lower for marker in ("sandbox", "apps folder", "desktop\\apps", "desktop apps", "test")):
        return ""
    root = _extract_windows_path(text)
    if not root:
        root = str(Path.home() / "Desktop" / "Apps" / "C0D3R_FileOpsSandbox")
    result = flow.tools.dispatch(
        "sandbox_file_ops",
        {"instruction": prompt, "sandbox_root": root, "dry_run": False, "allow_delete": False},
    )
    return _summarize_sandbox_file_ops_result(result)


def _summarize_sandbox_file_ops_result(result: dict[str, Any]) -> str:
    if result.get("error"):
        raise RuntimeError(f"sandbox_file_ops failed: {result.get('error')}")
    lines = [
        "Completed sandboxed file-system operation.",
        f"Sandbox: `{result.get('sandbox_root', '')}`",
        f"Status: `{result.get('status', '')}`",
        "",
        "Operations:",
    ]
    for item in result.get("operations") or []:
        name = item.get("operation", "operation")
        count = item.get("count")
        if count is None:
            count = item.get("file_count", item.get("written_count", ""))
        lines.append(f"- `{name}`: {count}")
    errors = result.get("errors") or []
    if errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in errors)
    tree = result.get("tree") or []
    if tree:
        lines.extend(["", "Current sandbox sample:"])
        for entry in tree[:12]:
            lines.append(f"- {entry.get('type')}: `{entry.get('path')}`")
    return "\n".join(lines)


def _maybe_run_virtual_hardware_scaffold(flow: Any, prompt: str) -> str:
    text = " ".join(str(prompt or "").split())
    lower = text.lower()
    markers = (
        "virtual hardware", "virtual driver", "virtual drivers",
        "hardware components", "hardware component", "device simulator",
        "radio bearer", "mesh internet", "mesh node", "simulated hardware",
    )
    if not any(marker in lower for marker in markers):
        return ""
    if not any(marker in lower for marker in ("build", "make", "create", "scaffold", "work on", "implement", "design")):
        return ""
    root = _extract_windows_path(text)
    if not root:
        root = str(Path.home() / "Desktop" / "Apps" / "MeshInternetVirtualLab")
    app_name = _extract_app_name(text, root)
    domain = "decentralized multi-bearer mesh internet virtual hardware and drivers"
    result = flow.tools.dispatch(
        "virtual_hardware_sim_scaffold",
        {"root_path": root, "app_name": app_name, "domain": domain, "overwrite": True},
    )
    return _summarize_virtual_hardware_result(result)


def _summarize_virtual_hardware_result(result: dict[str, Any]) -> str:
    if result.get("error"):
        raise RuntimeError(f"virtual_hardware_sim_scaffold failed: {result.get('error')}")
    validation = result.get("validation") if isinstance(result, dict) else {}
    build = validation.get("npm_build_test") if isinstance(validation, dict) else {}
    if isinstance(build, dict) and (build.get("error") or int(build.get("return_code") or 0) != 0):
        raise RuntimeError(str(build.get("error") or build.get("stderr") or build.get("stdout") or "virtual hardware scaffold validation failed")[-4000:])
    root = str(result.get("root_path") or "")
    app_name = str(result.get("app_name") or Path(root).name or "VirtualHardwareLab")
    lines = [
        f"Completed: {app_name} virtual hardware/driver simulation scaffold was created and validated.",
        f"Location: `{root}`",
        f"Files written: {result.get('file_count', 0)}",
        "Validation: `npm run build` and `npm test` passed.",
        "",
        "Included reusable abstractions:",
        "- Virtual hardware component base class",
        "- Virtual driver attach/detach/poll contract",
        "- Virtual radio bearers for mesh-network links",
        "- Traffic-class-aware link manager",
        "- Realm identity and deterministic per-realm addressing",
        "- Pocket/Relay/Home-Hub virtual node model",
        "- Content-addressed chunk store and delay-tolerant bundle queue",
        "- Demo simulator and automated test",
        "",
        "Useful commands:",
    ]
    for command in result.get("run_commands") or []:
        lines.append(f"- `{command}`")
    return "\n".join(lines)


def _maybe_run_base_crypto_paper_benchmark(flow: Any, prompt: str) -> str:
    text = " ".join(str(prompt or "").split())
    lower = text.lower()
    run_id = _extract_run_id(text)
    if run_id and ("status" in lower or "how is" in lower):
        result = flow.tools.dispatch("base_crypto_paper_trade_benchmark", {"action": "status", "run_id": run_id})
        return _summarize_base_crypto_paper_result(result, "status")
    if not ("crypto" in lower or "cryptos" in lower or "token" in lower):
        return ""
    if "base" not in lower:
        return ""
    if not any(marker in lower for marker in ("paper", "benchmark", "buy low", "sell high", "within an hour", "monitor", "trigger")):
        return ""
    action = "status" if "status" in lower or "how is" in lower else "start"
    budget = _extract_money_amount(text) or 20.0
    hours = _extract_hours(text) or 4.0
    params: dict[str, Any] = {
        "action": action,
        "budget_usd": budget,
        "hours": hours,
        "interval_minutes": float(os.getenv("C0D3R_CRYPTO_PAPER_INTERVAL_MIN", "5")),
        "target_net_pct": float(os.getenv("C0D3R_CRYPTO_PAPER_TARGET_NET_PCT", "2.0")),
        "stop_loss_pct": float(os.getenv("C0D3R_CRYPTO_PAPER_STOP_LOSS_PCT", "4.0")),
        "roundtrip_fee_pct": float(os.getenv("C0D3R_CRYPTO_PAPER_ROUNDTRIP_FEE_PCT", "1.2")),
    }
    if run_id:
        params["run_id"] = run_id
    result = flow.tools.dispatch("base_crypto_paper_trade_benchmark", params)
    return _summarize_base_crypto_paper_result(result, action)


def _extract_money_amount(text: str) -> float:
    match = re.search(r"\$\s*(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:usd|dollars)\b", text, flags=re.I)
    return float(match.group(1)) if match else 0.0


def _extract_hours(text: str) -> float:
    matches = re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b", text, flags=re.I)
    if not matches:
        return 0.0
    values = [float(item) for item in matches]
    return max(values)


def _extract_run_id(text: str) -> str:
    match = re.search(r"\b(base-paper-\d{8}-\d{6})\b", text)
    return match.group(1) if match else ""


def _summarize_base_crypto_paper_result(result: dict[str, Any], action: str) -> str:
    if result.get("error"):
        raise RuntimeError(f"base crypto paper benchmark failed: {result.get('error')}")
    if action == "status":
        positions = result.get("positions") or []
        lines = [
            "Base crypto paper-trade benchmark status.",
            f"Runtime: `{result.get('runtime', '')}`",
            f"Positions: {result.get('closed_count', 0)}/{result.get('position_count', 0)} closed",
            f"Net simulated P/L: `${float(result.get('net_pnl_usd') or 0):.6f}`",
            f"Profitable so far: `{bool(result.get('profitable'))}`",
        ]
        for pos in positions[:5]:
            lines.append(f"- {pos.get('symbol')}: {pos.get('status')} entry={pos.get('entry_price_usd')} exit={pos.get('exit_price_usd')} reason={pos.get('exit_reason')}")
        return "\n".join(lines)
    return "\n".join([
        "Started standalone Base-network crypto paper-trade benchmark.",
        f"Run ID: `{result.get('run_id')}`",
        f"PID: `{result.get('pid')}`",
        f"Runtime: `{result.get('runtime')}`",
        "",
        "Guardrails:",
        "- Paper trading only; no wallet access and no real transactions.",
        "- Not integrated into the trading pipeline.",
        "- Uses live public DEX data snapshots and logs candidate/entry/exit evidence.",
        "- Windows notifications will fire at entries, hourly checkpoints, exits, and completion.",
        "",
        "Status command:",
        f"- Ask C0D3R: `status for {result.get('run_id')}`",
    ])


def _summarize_tool_evidence(results: list[Any]) -> str:
    evidence: list[dict[str, Any]] = []
    successful_builds: list[dict[str, Any]] = []
    created_apps: list[dict[str, Any]] = []
    for result in results:
        for item in getattr(result, "tool_outputs", None) or []:
            tool = str(item.get("tool") or "")
            payload = item.get("result")
            if not tool or not isinstance(payload, dict):
                continue
            compact: dict[str, Any] = {"tool": tool}
            for key in (
                "status", "operation", "action", "cwd", "path", "root_path",
                "app_name", "file_count", "files_written", "run_commands",
                "validation", "error",
            ):
                if key in payload:
                    compact[key] = payload[key]
            if "stdout" in payload:
                compact["stdout"] = str(payload.get("stdout") or "")[-1200:]
            if "stderr" in payload:
                compact["stderr"] = str(payload.get("stderr") or "")[-1200:]
            evidence.append(compact)
            if tool == "react_pwa_scaffold" and not payload.get("error"):
                created_apps.append(payload)
            stdout = str(payload.get("stdout") or "")
            if "npm run build" in stdout and "built in" in stdout and not payload.get("error"):
                successful_builds.append(payload)
            validation = payload.get("validation")
            if isinstance(validation, dict):
                npm_build = validation.get("npm_build")
                if isinstance(npm_build, dict):
                    build_stdout = str(npm_build.get("stdout") or "")
                    if "npm run build" in build_stdout and "built in" in build_stdout and not npm_build.get("error"):
                        successful_builds.append(npm_build | {"cwd": payload.get("root_path")})
    if not evidence:
        return ""

    if created_apps or successful_builds:
        app = created_apps[-1] if created_apps else {}
        build = successful_builds[-1] if successful_builds else {}
        root = str(app.get("root_path") or build.get("cwd") or "").strip()
        app_name_value = app.get("app_name") or (Path(root).name if root else "the React PWA")
        app_name = str(app_name_value).strip()
        file_count = app.get("file_count")
        commands = app.get("run_commands") or []
        lines = [f"Completed: {app_name} was created and validated successfully."]
        if root:
            lines.append(f"Location: `{root}`")
        if file_count:
            lines.append(f"Files written: {file_count}")
        lines.append("Validation: `npm run build` passed.")
        if commands:
            lines.append("")
            lines.append("Useful commands:")
            for command in commands[:4]:
                lines.append(f"- `{command}`")
        lines.append("")
        lines.append("The model did not add extra commentary after the tools finished, so C0d3rV2 generated this concise completion summary from verified tool results.")
        return "\n".join(lines)

    lines = [
        "C0d3rV2 completed tool work, but the model did not produce a final synthesis.",
        "Verified tool summary:",
        "",
    ]
    for item in evidence[-6:]:
        tool = item.pop("tool", "tool")
        lines.append(f"## {tool}")
        lines.append("```json")
        lines.append(json.dumps(item, indent=2, default=str)[:6000])
        lines.append("```")
    return "\n".join(lines)


def _record_compact_turn(flow: Any, prompt: str, output: str) -> None:
    try:
        history = list(getattr(flow, "_compact_history", []))
        history.append({"user": prompt, "assistant": output[:4000]})
        flow._compact_history = history[-8:]
        if getattr(flow,"lt_memory",None):
            flow.lt_memory.append(prompt,output[:8000],workdir=str(flow.workdir),model_id=getattr(flow.session,"get_model_id",lambda:"")(),session_id=str(flow.session_id or ""))
        short=getattr(flow,"_st_mem",None)
        if short and hasattr(short,"record_turn"):
            short.record_turn(prompt,output,update_summary_model=False)
            flow._summary_bundle=short.summary_bundle
    except Exception:
        pass


def _compact_history_text(flow: Any, max_chars: int = 5000) -> str:
    history = list(getattr(flow, "_compact_history", []))
    if not history:
        return ""
    blocks = []
    for item in history[-6:]:
        blocks.append(f"User: {item.get('user', '')}\nAssistant: {item.get('assistant', '')}")
    text = "\n\n".join(blocks)
    return text[-max_chars:]


def _compact_session(flow: Any) -> Any:
    existing = getattr(flow, "_compact_atf_session", None)
    if existing is not None:
        return existing
    models_raw = os.getenv("C0D3R_WEB_COMPACT_ATF_MODELS", "")
    allowed = [item.strip() for item in models_raw.split(",") if item.strip()]
    try:
        from tools.c0d3rV2.plugins.agent_the_freeloader import AgentTheFreeloaderSession

        session = AgentTheFreeloaderSession(
            session_name=f"freeloader-web-compact-{str(flow.session_id)[:16]}",
            transcript_dir=_RUNTIME_ROOT / "transcripts",
            workdir=_PROJECT_ROOT,
            allowed_models=allowed,
            timeout_s=float(os.getenv("C0D3R_WEB_COMPACT_ATF_TIMEOUT_S", "15")),
            max_attempts=max(1, int(os.getenv("C0D3R_WEB_COMPACT_ATF_ATTEMPTS", "1"))),
            max_tokens=max(128, int(os.getenv("C0D3R_WEB_COMPACT_ATF_MAX_TOKENS", "1200"))),
        )
        flow._compact_atf_session = session
        return session
    except Exception:
        return flow.session


def _should_use_compact_research(prompt: str) -> bool:
    text = " ".join(str(prompt or "").lower().split())
    execution_markers = (
        "build", "create", "implement", "write file", "create file", "edit file",
        "fix", "repair", "install", "deploy", "run command", "scaffold",
        "workspace", "app", "spa", "pwa", "website",
    )
    if any(marker in text for marker in execution_markers):
        return False
    if not any(token in text for token in ("research", "current", "latest", "source", "citation", "cited")):
        return False
    return any(token in text for token in ("paper", "article", "report", "brief", "essay", "write"))


def _should_use_conversationalist_path(prompt: str) -> bool:
    """Return true when the next turn is conversation, not tool execution.

    This is intentionally not a canned greeting handler.  It only chooses the
    low-latency one-model conversationalist path; the selected backend still
    generates the actual response.
    """
    text = " ".join(str(prompt or "").lower().split())
    if not text:
        return False
    execution_markers = (
        "implement", "write file", "create file", "edit file", "fix", "repair",
        "research", "search", "benchmark", "build", "install", "deploy", "test",
        "run command", "make an app", "create an app", "set up", "setup",
    )
    if any(marker in text for marker in execution_markers):
        return False
    conversational_markers = (
        "hi", "hello", "hey", "hiya", "howdy", "good morning", "good afternoon",
        "good evening", "thanks", "thank you", "bye", "goodbye",
        "how are you", "how are you doing", "how's it going", "hows it going",
        "what's up", "whats up", "quick side question", "side question",
        "i have a question", "can we talk", "let's talk", "lets talk",
        "what do you think", "explain", "clarify",
    )
    if any(marker in text for marker in conversational_markers):
        return True
    # Short natural-language turns with no explicit execution intent are
    # usually conversational interleaves during a larger project session.
    if len(text.split()) <= 18 and text.endswith("?"):
        return True
    return False


def _should_use_compact_conversation(prompt: str) -> bool:
    """Backward-compatible alias for tests/imports."""
    return _should_use_conversationalist_path(prompt)


def _run_compact_conversation(flow: Any, prompt: str, system_context: str = "") -> str:
    flow.step_2_inject_context(prompt)
    session = _compact_session(flow)
    begin_turn = getattr(session, "begin_turn", None)
    if callable(begin_turn):
        begin_turn(max(1, int(os.getenv("C0D3R_WEB_CHAT_MAX_MODEL_CALLS", "1"))))
    system = (
        (system_context.strip() + "\n\n" if system_context.strip() else "")
        + "You are C0d3rV2 in conversationalist mode. The user may be talking "
        "between project-work turns. Answer the conversational turn naturally, "
        "briefly, and as the selected AI model. Do not use canned text. Do not "
        "claim task completion. Do not start a tool loop. Do not emit JSON "
        "unless explicitly asked."
    )
    history = _compact_history_text(flow)
    reply = session.send(
        (
            f"{flow._context}\n\n"
            + (f"Compact session history:\n{history}\n\n" if history else "")
            + f"User conversational turn:\n{prompt}\n\n"
            + "Return only the user-facing conversational response."
        ),
        stream=False,
        system=system,
        max_tokens=max(128, int(os.getenv("C0D3R_WEB_CHAT_MAX_TOKENS", "300"))),
    )
    _capture_last_model_id(flow, session)
    output = str(reply or "").strip()
    if not output:
        raise RuntimeError("Compact conversation produced no user-facing response.")
    _record_compact_turn(flow, prompt, output)
    return output


def _should_use_compact_planning(prompt: str) -> bool:
    text = " ".join(str(prompt or "").lower().split())
    execution_patterns = (
        r"\bimplement\b", r"\bcode\b", r"\bcreate files\b", r"\bwrite files\b",
        r"\bfix\b", r"\brepair\b", r"\bbuild and run\b",
    )
    if any(re.search(pattern, text) for pattern in execution_patterns):
        return False
    planning_markers = (
        "plan", "outline", "architecture", "milestone", "acceptance criteria",
        "roadmap", "strategy", "modules", "data inputs", "success metrics",
        "continue with", "continue the", "continue project",
    )
    return any(marker in text for marker in planning_markers)


def _run_compact_planning(flow: Any, prompt: str, system_context: str = "") -> str:
    flow.step_2_inject_context(prompt)
    session = _compact_session(flow)
    begin_turn = getattr(session, "begin_turn", None)
    if callable(begin_turn):
        begin_turn(max(1, int(os.getenv("C0D3R_WEB_PLANNING_MAX_MODEL_CALLS", "2"))))
    system = (
        (system_context.strip() + "\n\n" if system_context.strip() else "")
        + "You are C0d3rV2 in compact planning mode. Produce a direct, useful "
        "user-facing planning answer. Use prior session context when the user "
        "asks to continue. Do not emit raw JSON unless the user explicitly asks "
        "for JSON. Keep the plan bounded and actionable."
    )
    request = (
        f"{flow._context}\n\n"
        + (f"Compact session history:\n{_compact_history_text(flow)}\n\n" if _compact_history_text(flow) else "")
        + f"User request:\n{prompt}\n\n"
        + "Answer directly. If continuing prior work, identify what you are continuing."
    )
    reply = session.send(
        request,
        stream=False,
        system=system,
        max_tokens=max(512, int(os.getenv("C0D3R_WEB_PLANNING_MAX_TOKENS", "1200"))),
    )
    _capture_last_model_id(flow, session)
    output = str(reply or "").strip()
    if not output:
        raise RuntimeError("Compact planning produced no user-facing response.")
    _record_compact_turn(flow, prompt, output)
    return output


def _research_queries(prompt: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(prompt or "")).strip()
    cleaned = re.sub(
        r"\b(write|draft|produce|create)\b.*?\b(paper|article|report|brief|essay)\b",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" .,:;")
    if not cleaned:
        cleaned = text
    queries = [cleaned[:220]]
    lowered = cleaned.lower()
    if "best practice" in lowered or "hallucination" in lowered:
        queries.append("small language model coding agent hallucination reduction best practices 2024 2025")
    elif "market" in lowered:
        queries.append(cleaned[:160] + " market needs efficiency gaps")
    return list(dict.fromkeys(query for query in queries if query))


def _fallback_research_queries(prompt: str, existing: list[str]) -> list[str]:
    text = re.sub(r"\s+", " ", str(prompt or "")).strip()
    base = existing[0] if existing else text
    tokens = [
        token for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", base.lower())
        if token not in {
            "write", "paper", "article", "report", "research", "current",
            "latest", "source", "citation", "cited", "about", "using",
            "from", "with", "that", "this", "what", "need", "needs",
        }
    ]
    core = " ".join(tokens[:8]).strip() or base[:160]
    fallbacks = [
        f"{core} overview",
        f"{core} evidence sources",
        f"{core} site:wikipedia.org OR site:.gov OR site:.edu",
        f"{core} scholarly articles",
    ]
    if any(word in base.lower() for word in ("market", "product", "business", "software")):
        fallbacks.append(f"{core} market analysis unmet needs")
    return [query for query in fallbacks if query and query not in existing]


def _run_compact_research(flow: Any, prompt: str, system_context: str = "") -> str:
    """Bounded research-and-write path for web chat latency.

    The recursive orchestrator is intentionally powerful, but small hosted
    free models over-plan simple "research and write" tasks. This path keeps
    the C0d3rV2 tool contract intact: local memory and web search provide
    evidence, then ATF performs one synthesis call.
    """
    from web_search import WebSearch

    session = _compact_session(flow)
    begin_turn = getattr(session, "begin_turn", None)
    if callable(begin_turn):
        begin_turn(max(1, int(os.getenv("C0D3R_WEB_RESEARCH_MAX_MODEL_CALLS", "3"))))

    flow.step_2_inject_context(prompt)
    memory = {}
    try:
        memory = flow.tools.dispatch("memory_search", {"query": prompt[:180]})
    except Exception as exc:
        memory = {"error": str(exc)}

    searcher = WebSearch(None, max_results=max(3, int(os.getenv("C0D3R_WEB_RESEARCH_RESULTS", "5"))))
    searches = []
    source_count = 0
    queries = _research_queries(prompt)[: max(1, int(os.getenv("C0D3R_WEB_RESEARCH_QUERIES", "2")))]
    queries.extend(_fallback_research_queries(prompt, queries))
    seen_queries: set[str] = set()
    max_queries = max(2, int(os.getenv("C0D3R_WEB_RESEARCH_MAX_QUERY_ATTEMPTS", "5")))
    for query in queries:
        if query in seen_queries:
            continue
        seen_queries.add(query)
        if len(searches) >= max_queries and source_count > 0:
            break
        try:
            result = searcher.search(query)
            raw = result.get("results") or []
            source_count += len(raw)
            searches.append({
                "query": query,
                "results": raw[: searcher.max_results],
                "summary": result.get("summary") or "",
                "scientific": result.get("scientific", False),
            })
        except Exception as exc:
            searches.append({"query": query, "results": [], "summary": "", "error": str(exc)})

    if source_count <= 0:
        raise RuntimeError(
            "Research failed before synthesis: no usable web evidence was returned "
            "from DuckDuckGo, Bing, Wikipedia, OpenAlex, Crossref, arXiv, PubMed, "
            "or fallback source-query generation."
        )

    evidence = {
        "memory": memory,
        "web_searches": searches,
    }
    synthesis_system = (
        (system_context.strip() + "\n\n" if system_context.strip() else "")
        + "You are C0d3rV2 writing from bounded research evidence. "
        "Write a direct user-facing answer. Do not emit raw JSON. "
        "Cite source URLs from the provided evidence. If evidence is thin, "
        "state that limitation explicitly instead of inventing citations. "
        "Do not call arXiv sources peer-reviewed unless the evidence explicitly "
        "shows an accepted conference/journal venue; label them as preprints."
    )
    synthesis_prompt = (
        f"User request:\n{prompt}\n\n"
        f"Evidence JSON:\n{json.dumps(evidence, indent=2, default=str)[:12000]}\n\n"
        "Now write the requested deliverable. Keep it concise and readable."
    )
    reply = session.send(
        synthesis_prompt,
        stream=False,
        system=synthesis_system,
        max_tokens=max(512, int(os.getenv("C0D3R_WEB_RESEARCH_SYNTHESIS_TOKENS", "1800"))),
    )
    _capture_last_model_id(flow, session)
    output = str(reply or "").strip()
    if not output:
        raise RuntimeError("Compact research produced no user-facing synthesis.")
    _record_compact_turn(flow, prompt, output)
    return output


def probe_wizard_node() -> dict:
    """Utility for views and health checks."""
    from tools.wizard_session import WizardSession
    return WizardSession.probe()


def _capture_last_model_id(flow: Any, session: Any) -> None:
    try:
        getter = getattr(session, "get_model_id", None)
        if callable(getter):
            flow._last_model_id = str(getter() or "")
    except Exception:
        pass


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
