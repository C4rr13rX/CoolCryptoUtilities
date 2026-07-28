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
import json
import re
import sys
import time
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
        ScientificMethodTool, BrandDozerProductCycleTool, ProductArtifactMaterializerTool,
        ProjectWorkMapperTool, DependencyTraversalTool,
        ClassRefinementBenchmarkTool,
        ResearchHarvesterTool,
    )
    from lt_mem import LongTermMemory
    from side_load_st_mem_file_location import STSideLoadedMemory
    from side_load_lt_mem_file_location import LTSideLoadedMemory
    from st_memory import STMemory
    from web_search import WebSearch
    from tools.c0d3rV2.plugins.research_harvester import ResearchHarvester
    from executor import Executor

    session = _make_session(backend, session_key, workdir)
    rt = _RUNTIME_ROOT
    rt.mkdir(parents=True, exist_ok=True)

    lt_memory = LongTermMemory(rt)
    st_memory = STSideLoadedMemory(session_key, rt)
    short_memory = STMemory(session, session_id=session_key, runtime_root=rt)
    lt_side_memory = LTSideLoadedMemory(rt)
    executor = Executor(workdir)

    tools = ToolRegistry()
    tools.register(FileReadTool(workdir))
    tools.register(FileWriteTool(workdir))
    tools.register(DirectoryEnsureTool(workdir))
    tools.register(WorkspaceScaffoldTool(workdir))
    web_search = WebSearch(session)
    research_harvester = ResearchHarvester(rt)
    tools.register(EnvironmentBootstrapTool(workdir))
    tools.register(ExecutorTool(executor))
    tools.register(WebSearchTool(web_search))
    tools.register(ResearchHarvesterTool(research_harvester, web_search))
    tools.register(ScientificMethodTool(web_search, runtime_dir=rt))
    tools.register(BrandDozerProductCycleTool())
    tools.register(ProductArtifactMaterializerTool(workdir))
    tools.register(ProjectWorkMapperTool(workdir))
    tools.register(ClassRefinementBenchmarkTool())
    memory_tool = MemorySearchTool(lt_memory)
    file_locate_tool = FileLocateTool(st_memory, lt_side_memory, workdir=workdir)
    tools.register(memory_tool)
    tools.register(MatrixSearchTool())
    tools.register(file_locate_tool)
    tools.register(DependencyTraversalTool(workdir, memory_tool, file_locate_tool))

    flow = ProcessFlow(
        session=session,
        workdir=workdir,
        tools=tools,
        session_id=session_key,
        lt_memory=lt_memory,
        short_memory=short_memory,
        st_side_memory=st_memory,
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
            max_tokens=max(
                2048,
                int(os.getenv("C0D3R_DELIVERY_ATF_MAX_TOKENS", "4096")),
            ),
            timeout_s=float(os.getenv("C0D3R_DELIVERY_ATF_TIMEOUT_S", "30")),
            max_attempts=max(1, int(os.getenv("C0D3R_DELIVERY_ATF_ATTEMPTS", "3"))),
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
        bounded_research = (
            "bounded read-only archival-research role" in system_context.lower()
        )
        if bounded_research:
            budget_name = "C0D3R_RESEARCH_MAX_MODEL_CALLS"
            default_budget = "2"
        else:
            budget_name = "C0D3R_ATOMIC_MAX_MODEL_CALLS" if atomic else "C0D3R_MAX_MODEL_CALLS"
            default_budget = "12" if atomic else "64"
        begin_turn(max(1, int(os.getenv(budget_name, default_budget))))

    if system_context and system_context.strip():
        flow._pending_system = system_context.strip()
        _patch_session_context(flow, system_context)
    else:
        flow._pending_system = ""

    bounded_research_result = _bounded_read_only_research_delivery(
        prompt, flow, system_context
    )
    if bounded_research_result:
        return bounded_research_result

    prompt = _inject_dependency_evidence(prompt, flow, system_context)
    prompt = _inject_research_evidence(prompt, flow, system_context, project_key=session_key)

    read_only_result = _read_only_evidence_delivery(prompt, flow)
    if read_only_result:
        return read_only_result

    atomic_result = _atomic_contract_delivery(prompt, flow, workdir, system_context)
    if atomic_result:
        return atomic_result

    augmented = flow.step_2_inject_context(prompt)
    # step_2_inject_context stores the system-only context on flow._context.
    # Do not replace it with the returned "context + user request" string or
    # every orchestrator call receives the complete user prompt twice.
    # Preserve the caller's execution-mode contract in the Orchestrator context
    # as well as the provider system prompt. Previously this was patched only
    # into the model session, so Orchestrator never saw "unattended atomic
    # workday job" and incorrectly re-expanded each persisted BrandDozer step.
    if system_context and system_context.strip():
        flow._context = system_context.strip() + "\n\n" + flow._context

    from orchestrator import Orchestrator
    from petal_system import PetalManager

    orchestrator = Orchestrator(
        session=flow.session,
        tools=flow.tools,
        context=flow._context,
        petals=flow.petals or PetalManager(),
    )
    results, tree = orchestrator.run(prompt)
    flow._last_refined_outline = dict(getattr(orchestrator, "refined_outline", {}) or {})
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
    if "unattended atomic workday job" in system_context.lower():
        # An atomic BrandDozer iteration is allowed to fail, but the failure
        # must remain explicit so the outer test->fix loop can persist it as
        # correction evidence. Raising here previously terminated the entire
        # delivery run before its deterministic smoke gate could respond.
        errors = [str(getattr(result, "error", "") or "").strip() for result in results]
        detail = " | ".join(item for item in errors if item)[:4000]
        lowered_detail = detail.lower()
        # Do not burn another provider call when the entire turn produced only
        # protocol hallucinations/acknowledgements and no actionable evidence.
        # The supervisor treats this explicit signal as a capacity cooldown and
        # resumes the same bounded step later.
        if detail and any(marker in lowered_detail for marker in (
            "protocol_hallucination", "unparseable structured response",
            "no user-facing result", "corrective model violated the advertised tool schema",
            "exhausted eligible fallbacks", "model-call budget exhausted",
        )) and not any(
            "file_write:" in item.lower() and "rejected" not in item.lower()
            for item in errors
        ):
            raise RuntimeError("ATF sane-response cooldown required: " + detail[:1800])
        return (
            "[c0d3rv2] Atomic work package incomplete; validation/correction required: "
            + (detail or "no successful mutation or user-facing result was produced")
        )
    raise RuntimeError(
        "C0d3rV2 delivery produced no user-facing result and no successful "
        "write/scaffold evidence."
    )


def _read_only_evidence_delivery(prompt: str, flow: Any) -> str:
    """Answer explicit evidence-only requests without entering a mutation planner."""
    lowered = " ".join(str(prompt or "").lower().split())
    explicitly_read_only = any(marker in lowered for marker in (
        "read-only", "read only", "do not write", "do not modify", "without modifying",
    ))
    has_evidence = any(marker in lowered for marker in (
        "injection packet", "dependency/regression injection", "evidence_files",
    ))
    if not explicitly_read_only or not has_evidence:
        return ""
    system = (
        "You are C0d3rV2's bounded evidence-synthesis path. Answer the request directly from the "
        "supplied, hashed project evidence. Do not plan a project, call mutation tools, or claim "
        "unsupported facts. Preserve the response schema requested by the user. Return the final "
        "answer now; an evidence answer is successful completion and requires no file write."
    )
    try:
        raw = flow.session.send(prompt, stream=False, system=system)
    except Exception:
        raise
    output = str(raw or "").strip()
    if not output:
        raise RuntimeError("C0d3rV2 evidence synthesis produced no user-facing answer.")
    acknowledgements = (
        "how can i help", "ready to help", "please provide", "need more information",
    )
    if any(marker in output.lower() for marker in acknowledgements):
        raise RuntimeError("C0d3rV2 evidence synthesis returned an acknowledgement instead of an answer.")
    # Normalize provider formatting at the C0d3rV2 boundary. In particular,
    # consumers asking for JSON must not receive Markdown-fenced JSON.
    from model_response_normalizer import ModelResponseNormalizer
    parsed = ModelResponseNormalizer().parse(output)
    if parsed.valid and isinstance(parsed.value, (dict, list)):
        output = json.dumps(parsed.value, ensure_ascii=False, indent=2)
    return output


def _bounded_read_only_research_delivery(
    prompt: str, flow: Any, system_context: str
) -> str:
    """Run research synthesis directly, without the mutation-agent planner.

    Source discovery and document fetching are performed by Brand Dozer before
    this call.  Entering the general C0D3R planner here can turn one bounded
    literature package into dozens of tool-protocol turns even though the only
    valid product is a JSON object.
    """
    if "bounded read-only archival-research role" not in system_context.lower():
        return ""
    system = (
        f"{system_context.strip()}\n\n"
        "You are C0D3R V2's bounded archival-research synthesis path. Work only "
        "from the assignment and supplied candidate evidence. Do not plan a "
        "software project, call tools, modify files, or return a tool protocol. "
        "Return exactly one complete JSON object matching the requested schema, "
        "without Markdown fences or commentary."
    )
    from model_response_normalizer import ModelResponseNormalizer

    raw = flow.session.send(prompt, stream=False, system=system)
    output = str(raw or "").strip()
    parsed = ModelResponseNormalizer().parse(output)
    if parsed.valid and isinstance(parsed.value, dict):
        return json.dumps(parsed.value, ensure_ascii=False)

    # One scope-locked repair is enough to recover truncated/fenced prose.  A
    # second failed response is surfaced to Brand Dozer's quarantine instead
    # of allowing an unbounded autonomous loop.
    repair_prompt = (
        "Repair your previous response. Return exactly one complete JSON object "
        "that satisfies every requested key and evidence constraint in the "
        "original assignment. Do not add prose or Markdown.\n\n"
        f"ORIGINAL ASSIGNMENT:\n{prompt}\n\n"
        f"INVALID RESPONSE:\n{output[:16000]}"
    )
    repaired = flow.session.send(repair_prompt, stream=False, system=system)
    repaired_output = str(repaired or "").strip()
    repaired_parsed = ModelResponseNormalizer().parse(repaired_output)
    if repaired_parsed.valid and isinstance(repaired_parsed.value, dict):
        return json.dumps(repaired_parsed.value, ensure_ascii=False)
    raise RuntimeError(
        "C0D3R bounded research route returned no complete JSON object after "
        "one scope-locked repair."
    )


def _atomic_contract_delivery(prompt: str, flow: Any, workdir: Path, system_context: str) -> str:
    """One-artifact compiler path: refine, synthesize, write, validate, repair."""
    from outline_refiner import OutlineRefiner

    if not OutlineRefiner._contract_ready(prompt):
        return ""
    targets = re.findall(
        r"(?<![\w.-])([\w./\\-]+\.(?:py|ts|tsx|js|jsx|rs|go|java|cpp|c|h|php|pl))\b",
        prompt, flags=re.IGNORECASE,
    )
    targets = [item.replace("\\", "/") for item in targets if not Path(item).name.lower().startswith("test_")]
    if len(set(targets)) != 1:
        return ""
    target = targets[0]
    _atomic_trace({"event":"start","target":target,"workdir":str(workdir)})
    outline = OutlineRefiner(workdir=workdir, passes=4).refine(prompt, prompt)
    if not (outline.get("quality") or {}).get("passed"):
        raise RuntimeError("Atomic artifact planning quality gate failed")
    flow._last_refined_outline = outline
    test_match = re.search(r"\b(test_[\w.-]+\.py)\b", prompt, flags=re.IGNORECASE)
    validator = f'& "{sys.executable}" {test_match.group(1)}' if test_match else ""
    observed_inputs: list[str] = []
    for named_path in ["contract.json", test_match.group(1) if test_match else ""]:
        if not named_path:
            continue
        observed = flow.tools.dispatch("file_read", {"path": named_path})
        content = str(observed.get("content") or "") if isinstance(observed, dict) else ""
        if content:
            observed_inputs.append(f"OBSERVED {named_path}:\n{content[:12000]}")
    observed_context = "\n\n".join(observed_inputs)
    suffix = Path(target).suffix.lower()
    language = {".py":"Python", ".ts":"TypeScript", ".tsx":"TypeScript TSX", ".js":"JavaScript", ".jsx":"JavaScript JSX", ".rs":"Rust", ".go":"Go", ".java":"Java", ".cpp":"C++", ".c":"C", ".h":"C/C++ header", ".php":"PHP", ".pl":"Perl"}.get(suffix, "source")
    prior_error = ""
    for attempt in range(1, 3):
        repair = f"\nThe prior validator failed:\n{prior_error}\nReturn a corrected complete file." if prior_error else ""
        reply = flow.session.send(
            prompt=(
                f"Produce the complete contents of {target} for this exact contract. "
                f"Return only {language} source, without markdown or explanation.\n\n"
                f"{prompt}\n\n{observed_context}{repair}"
            ),
            system=(
                f"{system_context}\nYou are C0d3rV2's atomic artifact compiler. Preserve the public contract and scope exactly. "
                "All necessary contract and test inputs are included below; do not request tools. "
                "Validate numeric domains and error behavior; avoid placeholders."
            ),
            stream=False,
        )
        code = _extract_source(reply, language)
        _atomic_trace({"event":"model_response","target":target,"attempt":attempt,"chars":len(code),"model":getattr(flow.session,"get_model_id",lambda:"unknown")(),"route":getattr(flow.session,"last_route",[])})
        written = flow.tools.dispatch("file_write", {"path": target, "content": code, "create_dirs": True})
        _atomic_trace({"event":"write","target":target,"attempt":attempt,"result":written})
        if written.get("error"):
            prior_error = str(written["error"])
            continue
        if not validator:
            return f"Created {target} through the selected model after four scope-locked refinement passes."
        checked = flow.tools.dispatch("executor", {"command": validator})
        _atomic_trace({"event":"validate","target":target,"attempt":attempt,"result":checked})
        if not checked.get("error") and checked.get("return_code") == 0:
            reporter = getattr(flow.session, "report_outcome", None)
            if callable(reporter):
                reporter(success=True, reason=f"atomic_validator_passed:{validator}")
            return (
                f"Created and validated {target} through the selected model in {attempt} synthesis call(s).\n"
                f"Validation: {validator} exited 0."
            )
        prior_error = str(checked.get("error") or checked.get("stderr") or checked.get("stdout") or "validation failed")[-3000:]
        reporter = getattr(flow.session, "report_outcome", None)
        if callable(reporter):
            reporter(success=False, reason=f"atomic_validator_failed:{prior_error[:500]}")
    raise RuntimeError(f"Atomic artifact validation failed after two selected-model calls: {prior_error}")


def _atomic_trace(payload: dict[str, Any]) -> None:
    try:
        path = _RUNTIME_ROOT / "atomic_contract.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts":time.time(),**payload}, default=str, ensure_ascii=True)+"\n")
    except Exception:
        pass


def _extract_source(reply: str, language: str) -> str:
    raw = str(reply or "").strip()
    if "```" not in raw:
        return raw
    blocks = raw.split("```")
    for block in blocks:
        cleaned = block.strip()
        if cleaned.lower().startswith((language.lower(), "python", "typescript", "javascript", "rust", "java", "cpp", "c++", "php", "perl")):
            return cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
    return next((block.strip() for block in blocks if block.strip()), raw)


def _delivery_tool_summary(tool_events: list[dict]) -> str:
    """Return a local completion summary only when tools prove delivery happened."""
    successful: list[dict] = []
    for event in tool_events:
        tool = str(event.get("tool") or "")
        if tool not in {
            "file_write", "directory_ensure", "workspace_scaffold",
            "environment_bootstrap", "executor", "branddozer_product_cycle",
            "class_refinement_benchmark",
        }:
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
        elif tool == "branddozer_product_cycle":
            state = result.get("state") or {}
            lines.append(
                f"- branddozer_product_cycle: status={result.get('status')} "
                f"cycle={state.get('cycle')} verified={result.get('verified')}"
            )
        elif tool == "class_refinement_benchmark":
            lines.append(
                f"- class_refinement_benchmark: passed={result.get('passed')}/"
                f"{result.get('count')} pass_rate={result.get('pass_rate')}"
            )
    return "\n".join(lines)


def _direct_verified_delivery_route(prompt: str, flow: Any, workdir: Path) -> str:
    lowered = (prompt or "").lower()
    product_markers = (
        "branddozer digital product", "digital product continuous refinement",
        "continuous refinement cycle", "market needs", "base sepolia",
        "product-loop", "product loop",
    )
    wants_product_cycle = (
        any(marker in lowered for marker in product_markers)
        and any(marker in lowered for marker in ("product", "market", "storefront", "crypto", "base"))
    )
    if wants_product_cycle:
        result = flow.tools.dispatch("branddozer_product_cycle", {
            "root_path": str(workdir),
            "cycles": 1,
        })
        if result.get("error"):
            raise RuntimeError(str(result.get("error")))
        return (
            "[c0d3rv2-delivery] Completed verified BrandDozer product cycle:\n"
            f"- status: {result.get('status')}\n"
            f"- verified: {result.get('verified')}\n"
            f"- output: {result.get('output')}\n"
            f"- workspace: {(result.get('state') or {}).get('workspace')}"
        )

    class_markers = (
        "class generation", "class-generation", "class benchmark",
        "class refinement", "make classes", "produce classes",
        "represents a dog", "represents a bicycle", "bird development",
    )
    if any(marker in lowered for marker in class_markers):
        result = flow.tools.dispatch("class_refinement_benchmark", {
            "count": 1 if "300" not in lowered else 8,
            "attempts": 2,
        })
        if result.get("error"):
            raise RuntimeError(str(result.get("error")))
        return (
            "[c0d3rv2-delivery] Completed verified ATF class refinement benchmark:\n"
            f"- passed: {result.get('passed')}/{result.get('count')}\n"
            f"- pass_rate: {result.get('pass_rate')}\n"
            f"- guide: {result.get('guide_path')}\n"
            f"- results: {result.get('results_path')}"
        )

    return ""


def _inject_dependency_evidence(prompt: str, flow: Any, system_context: str) -> str:
    """Inject a bounded causal file graph before an unattended mutation."""
    if "unattended atomic workday job" not in str(system_context).lower():
        return prompt
    paths: list[str] = []
    match = re.search(r"Validator-directed repair paths:\s*(\[[^\]]*\])", prompt, re.IGNORECASE)
    if match:
        try:
            paths = [str(item) for item in json.loads(match.group(1))]
        except Exception:
            paths = []
    failures = re.findall(
        r'"(?:focus_failure|active_failures)"[\s\S]{0,1800}?"message"\s*:\s*"((?:\\.|[^"\\])*)"',
        prompt,
    )
    next_steps = re.findall(r"Next step:\s*([^\n]+)", prompt)
    query = " ".join([*(next_steps[-1:] or []), *(failures[-2:] or []), *paths[:8]])[:1000]
    if not query.strip():
        return prompt
    try:
        packet = flow.tools.dispatch("dependency_traversal", {
            "action": "inject", "query": query, "paths": paths,
            "depth": 3, "max_nodes": 48, "failures": failures[-6:],
        })
    except Exception as exc:
        packet = {"error": str(exc)}
    if packet.get("error") or not packet.get("change_surface"):
        return prompt
    return prompt + "\n\n[C0d3rV2 dependency/regression injection]\n" + json.dumps(
        packet, indent=2, default=str,
    )[:6500]


def _inject_research_evidence(prompt: str, flow: Any, system_context: str, *, project_key: str = "") -> str:
    """Ground unattended repairs from local-first, bounded web evidence."""
    if "unattended atomic workday job" not in str(system_context).lower():
        return prompt
    if os.getenv("C0D3R_AUTO_RESEARCH", "1").strip().lower() in {"0", "false", "off", "no"}:
        return prompt
    messages = re.findall(
        r'"focus_failure"\s*:\s*\{[\s\S]{0,1200}?"message"\s*:\s*"((?:\\.|[^"\\])*)"',
        prompt,
    )
    patterns = re.findall(
        r'"recommended_pattern"\s*:\s*"((?:\\.|[^"\\])*)"', prompt,
    )
    next_steps = re.findall(r"Next step:\s*([^\n]+)", prompt)
    def decode(value: str) -> str:
        try:
            return json.loads('"' + value + '"')
        except Exception:
            return value
    terms = [decode(messages[-1])] if messages else []
    if patterns:
        terms.append(decode(patterns[-1]))
    if next_steps:
        terms.append(next_steps[-1])
    query = " ".join(terms).strip()[:900]
    query = re.sub(r"(?:[A-Za-z]:)?[\\/]?(?:src|app|lib|tests?)[\\/][\w./\\-]+(?::\d+(?::\d+)?)?", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"[\u276f|]+|\s+", " ", query).strip()
    if not query:
        return prompt
    try:
        if project_key:
            flow.tools.dispatch("research_harvester", {
                "action": "project_configure", "project_key": project_key,
                "query": query, "max_depth": 0, "max_pages": 5,
                "coverage_target": 0.7, "refresh_seconds": 86_400, "max_rounds": 2,
            })
            evidence = flow.tools.dispatch("research_harvester", {
                "action": "project_refresh", "project_key": project_key, "limit": 4,
            })
        else:
            evidence = flow.tools.dispatch("research_harvester", {
                "action": "research", "query": query, "max_depth": 0,
                "max_pages": 5, "limit": 4, "same_origin": True,
            })
    except Exception as exc:
        evidence = {"error": str(exc)}
    retrieval = evidence.get("retrieval") or {}
    results = retrieval.get("results") or []
    if not results:
        return prompt
    compact_sources = [{
        "title": item.get("title"), "url": item.get("url"),
        "passage": str(item.get("passage") or "")[:1200],
        "content_sha256": item.get("content_sha256"),
        "authority_score": item.get("authority_score"),
    } for item in results[:2]]
    packet = {
        "schema": "c0d3r.retrieval-evidence/v1",
        "query": query,
        "coverage": retrieval.get("coverage"),
        "sources": compact_sources,
        "instruction": (
            "Use these fetched passages as implementation evidence. Preserve source URLs/hashes; "
            "do not copy examples blindly or treat discovery snippets as verified facts."
        ),
    }
    return prompt + "\n\n[C0d3rV2 local-first research evidence]\n" + json.dumps(packet, indent=2, default=str)[:4500]


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
        "refined_outline": dict(getattr(flow, "_last_refined_outline", {}) or {}),
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
