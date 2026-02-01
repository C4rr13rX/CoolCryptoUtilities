#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import queue
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Tuple
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_INSTALL_ATTEMPTS: set[str] = set()


def _live_log_enabled() -> bool:
    return os.getenv("C0D3R_LIVE_LOG", "1").strip().lower() not in {"0", "false", "no", "off"}


def _diag_log(message: str) -> None:
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        path = Path("runtime/c0d3r/diagnostics.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {message}\n")
    except Exception:
        pass


def _emit_live(message: str) -> None:
    if not _live_log_enabled():
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        log_path = Path("runtime/c0d3r/live.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="c0d3r",
        description="c0d3r CLI (Bedrock-backed) - run prompts against a working directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text. If omitted, reads from stdin.")
    parser.add_argument("-C", "--cd", dest="workdir", help="Working directory (defaults to current directory).")
    parser.add_argument("--model", help="Override Bedrock model id or inference profile.")
    parser.add_argument("--reasoning", default=None, help="Reasoning effort (low, medium, high, extra_high).")
    parser.add_argument("--profile", help="AWS profile to use (defaults to auto-detect).")
    parser.add_argument("--region", help="AWS region (default: from env or us-east-1).")
    parser.add_argument("--research", action="store_true", help="Enable web research + synthesis.")
    parser.add_argument("--image", action="append", help="Image path(s) for multimodal review.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output.")
    parser.add_argument("--tool-loop", action="store_true", help="Enable local command execution loop.")
    parser.add_argument("--no-tools", action="store_true", help="Disable local command execution loop.")
    parser.add_argument("--no-context", action="store_true", help="Disable automatic repo context summary.")
    parser.add_argument("--scientific", action="store_true", help="Enable scientific-method analysis mode.")
    parser.add_argument("--no-scientific", action="store_true", help="Disable scientific-method analysis mode.")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _emit_live("boot: start")
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    _emit_live(f"boot: prompt loaded (len={len(prompt)})")
    # Silence noisy startup warnings for CLI usage.
    os.environ.setdefault("C0D3R_QUIET_STARTUP", "1")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    from services.env_loader import EnvLoader
    from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings
    from services.agent_workspace import run_command
    from services.system_probe import system_probe_context

    _emit_live("boot: env load")
    EnvLoader.load()

    _emit_live("boot: settings build")
    settings = c0d3r_default_settings()
    if args.model:
        settings["model"] = args.model
    if args.reasoning:
        settings["reasoning_effort"] = args.reasoning
    if args.profile:
        settings["profile"] = args.profile
    if args.region:
        settings["region"] = args.region
    if args.research:
        settings["research"] = True

    _emit_live("boot: resolve workdir")
    workdir = Path(args.workdir or os.getcwd()).resolve()
    os.environ.setdefault("C0D3R_ROOT_CWD", str(workdir))
    if _requires_rigorous_constraints(prompt):
        settings["rigorous_mode"] = True
    scientific = args.scientific or (os.getenv("C0D3R_SCIENTIFIC_MODE", "1").strip().lower() not in {"0", "false", "no", "off"})
    if args.no_scientific:
        scientific = False
    _emit_live("boot: build context")
    context_block = ""
    if not args.no_context:
        context_block = _build_context_block(workdir, run_command)
    tool_loop = args.tool_loop or (os.getenv("C0D3R_TOOL_LOOP", "1").strip().lower() not in {"0", "false", "no", "off"})
    if args.no_tools:
        tool_loop = False

    _emit_live("boot: probe context")
    probe_block = system_probe_context(workdir)
    if context_block:
        context_block = f"{probe_block}\n{context_block}"
    else:
        context_block = probe_block
    if context_block:
        prompt = f"{context_block}\n\nUser request:\n{prompt}"
    settings = dict(settings)
    if "stream_default" in settings:
        settings["stream_default"] = settings.get("stream_default") and not args.no_stream
    _emit_live("boot: session init")
    session = C0d3rSession(
        session_name="c0d3r-cli",
        transcript_dir=Path("runtime/c0d3r/transcripts"),
        workdir=str(workdir),
        **settings,
    )
    usage = UsageTracker(model_id=session.get_model_id())
    header = HeaderRenderer(usage)
    header.render()
    _emit_live("boot: header rendered")
    _refresh_pricing_cache(session, header, session.get_model_id())
    math_block = ""
    if settings.get("math_grounding", True):
        _emit_live("boot: math grounding")
        math_block = _math_grounding_block(session, prompt, workdir)
    if math_block:
        prompt = f"{math_block}\n\n{prompt}"
    _emit_live("boot: enter repl")
    return _run_repl(
        session,
        usage,
        workdir,
        run_command,
        scientific=scientific,
        tool_loop=tool_loop,
        initial_prompt=prompt or None,
        header=header,
    )


def _build_context_block(workdir: Path, run_command) -> str:
    lines = [
        "[context]",
        f"- cwd: {workdir}",
        f"- os: {os.name}",
    ]
    code, stdout, stderr = run_command("git status -sb", cwd=workdir)
    if stdout.strip():
        lines.append("git status -sb:")
        lines.append(stdout.strip()[:2000])
    if stderr.strip():
        lines.append("git status stderr:")
        lines.append(stderr.strip()[:500])
    code, stdout, stderr = run_command("git rev-parse --show-toplevel", cwd=workdir)
    if stdout.strip():
        lines.append(f"repo root: {stdout.strip()}")
    code, stdout, stderr = run_command("Get-ChildItem -Name", cwd=workdir) if os.name == "nt" else run_command("ls -1", cwd=workdir)
    if stdout.strip():
        lines.append("top-level files:")
        lines.append("\n".join(stdout.strip().splitlines()[:80]))
    return "\n".join(lines)


def _run_tool_loop(
    session: C0d3rSession,
    prompt: str,
    workdir: Path,
    run_command,
    *,
    images: List[str] | None,
    stream: bool,
    stream_callback,
    usage_tracker,
) -> str:
    max_steps = int(os.getenv("C0D3R_TOOL_STEPS", "4"))
    full_completion = _requires_full_completion(prompt)
    unlimited = full_completion and os.getenv("C0D3R_FULL_UNLIMITED", "1").strip().lower() not in {"0", "false", "no", "off"}
    if full_completion and not unlimited:
        max_steps = max(max_steps, int(os.getenv("C0D3R_FULL_STEPS", "8")))
    history: List[str] = []
    gap_score = 0
    last_error = ""
    log_path = Path("runtime/c0d3r/tool_loop.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base_snapshot = _snapshot_projects_dir(prompt)
    created_dirs: List[str] = []
    known_projects: List[Path] = []
    success = False
    any_success = False
    tests_ran = False
    tests_ok = False
    require_tests = _requires_tests(prompt)
    allow_new_root_dirs = _requires_new_projects_dir(prompt) or _prompt_allows_new_dirs(prompt)
    require_benchmark = _requires_benchmark(prompt)
    consecutive_no_progress = 0
    test_failures = 0
    model_timeouts = 0
    unbounded_resolved = False
    behavior_log: list[dict] = []
    step = 0
    no_command_count = 0
    while True:
        step += 1
        if not unlimited and step > max_steps:
            break
        usage_tracker.set_status("planning", f"step {step+1}/{max_steps}")
        _emit_live(f"tool_loop step {step+1}/{max_steps}: preparing model prompt")
        _capture_behavior_snapshot(
            behavior_log,
            step=step,
            no_progress=consecutive_no_progress,
            test_failures=test_failures,
            model_timeouts=model_timeouts,
            note="prepare_prompt",
        )
        if gap_score >= 2:
            _emit_live("tool_loop: gap score high; running targeted research")
            research_note = _pre_research(session, prompt)
            if research_note:
                history.append("[research]\n" + research_note[:2000])
                _append_research_notes(research_note)
        enforce_progress = consecutive_no_progress >= 2 and full_completion
        base_request = _strip_context_block(prompt)
        tool_prompt = (
            "You can run local shell commands to inspect the repo and validate work. "
            "Return ONLY JSON with keys: commands (list of strings) and final (string or empty). "
            "If you are done, set final to your response and commands to []. "
            "Always create minimal unit tests with representative sample (foo) data and run them "
            "after each file write, and again before declaring success. "
            "Do not finalize until tests pass.\n"
            + (
                "This request requires rigorous mathematical/engineering constraints. "
                "Use explicit assumptions, invariants, and verification steps. "
                "Prioritize research, formal reasoning, and validation before coding. "
                "Do NOT shortcut with stubs or placeholders.\n"
                if _requires_rigorous_constraints(prompt)
                else ""
            )
            + "Create and maintain a checklist of requirements in your reasoning. "
            "Before finalizing, explicitly verify each checklist item is complete. "
            "If incomplete, keep working and iterate. "
            "It is OK to loop across tasks if they depend on each other.\n"
            "Create and update these files in runtime/c0d3r:\n"
            "- plan.md (expanded plan)\n"
            "- checklist.md (requirements checklist + verification notes)\n"
            "- bibliography.md (APA citations for any research/software used)\n"
            "Do not finalize until these files exist and are updated.\n"
            "Work inside the current working directory. Do NOT create new project directories "
            "or switch to other folders unless the user explicitly asks. "
            "Prefer modifying existing files in the target project and running its tests.\n"
            "If any errors or missing knowledge are detected, perform targeted research and "
            "record source->decision->code in checklist.md before continuing.\n"
            "If code/data is derived from research, add APA-formatted citations in code comments "
            "and append them to runtime/c0d3r/bibliography.md.\n"
            "If third-party libraries or datasets are added, record license obligations in "
            "runtime/c0d3r/bibliography.md and include required license files/notices in the repo.\n"
            + (
                "The request includes 'outperforms' or 'novel' claims. "
                "You must create a benchmark script that compares your solver to a baseline "
                "on the same generated dataset, and write results to runtime/c0d3r/benchmarks.json "
                "including keys: generated_by, baseline, candidate, dataset, metrics. "
                "Do not finalize without that file.\n"
                if require_benchmark
                else ""
            )
            + ("You must modify at least one project file this step; runtime-only edits are not allowed.\n" if enforce_progress else "")
            + "Meta commands available:\n"
            "- ::bg <command> (run long-lived command in background, returns pid)\n"
            "- ::wait_http <url> <seconds> (poll until HTTP 200 or timeout)\n"
            "- ::sleep <seconds>\n"
            "- ::kill <pid>\n"
            "- ::cd <path> (change working directory for subsequent commands)\n"
            "Be concise and execution-focused. Always execute commands before returning final.\n"
            "Use this pattern for Ionic projects:\n"
            "1) ::cd C:/Users/Adam/Projects\n"
            "2) npx @ionic/cli@latest start <appname> tabs --type=angular --no-interactive --no-confirm --no-git\n"
            "3) ::cd <appname>\n"
            "4) ::bg npx @ionic/cli@latest serve --no-open --port <port>\n"
            "5) ::wait_http http://localhost:<port> 120\n"
            f"Step {step + 1}/{max_steps}.\n"
            f"Request:\n{base_request}\n"
            + ("\n\nRecent outputs:\n" + "\n".join(history[-6:]) if history else "")
            + (f"\n\n[gap_score]\n{gap_score}\n[last_error]\n{last_error}" if gap_score or last_error else "")
        )
        if model_timeouts >= 2:
            tool_prompt = (
                "Return ONLY JSON with keys: commands (list of strings) and final (string or empty). "
                "Provide 2-6 concrete shell commands. Do not return an empty list.\n"
                f"Request:\n{base_request}\n"
            )
        usage_tracker.add_input(tool_prompt)
        _emit_live("tool_loop: calling model for commands")
        _diag_log("tool_loop: model call start")
        if _requires_rigorous_constraints(prompt):
            command_model = os.getenv("C0D3R_COMMAND_MODEL", "mistral.mistral-large-3-675b-instruct")
            saved_model = getattr(session._c0d3r, "model_id", "")
            saved_multi = getattr(session._c0d3r, "multi_model", True)
            saved_rigorous = getattr(session._c0d3r, "rigorous_mode", False)
            saved_profile = getattr(session._c0d3r, "inference_profile", "")
            try:
                session._c0d3r.model_id = command_model
                session._c0d3r.multi_model = False
                session._c0d3r.rigorous_mode = False
                session._c0d3r.inference_profile = ""
                response = _call_with_timeout(
                    session._safe_send,
                    timeout_s=_model_timeout_s(),
                    kwargs={"prompt": tool_prompt, "stream": stream, "images": images, "stream_callback": stream_callback},
                )
            finally:
                session._c0d3r.model_id = saved_model
                session._c0d3r.multi_model = saved_multi
                session._c0d3r.rigorous_mode = saved_rigorous
                session._c0d3r.inference_profile = saved_profile
        else:
            response = _call_with_timeout(
                session._safe_send,
                timeout_s=_model_timeout_s(),
                kwargs={"prompt": tool_prompt, "stream": stream, "images": images, "stream_callback": stream_callback},
            )
        if response is None:
            _emit_live("tool_loop: model call timed out")
            _diag_log("tool_loop: model call timeout")
            timeout_val = _model_timeout_s()
            if timeout_val is None:
                last_error = "model call returned no response (timeouts disabled)"
            else:
                last_error = f"model call timed out after {timeout_val}s"
            model_timeouts += 1
            if model_timeouts >= 3:
                fallback_model = os.getenv("C0D3R_TOOL_FALLBACK_MODEL", "mistral.mistral-large-3-675b-instruct")
                try:
                    session._c0d3r.rigorous_mode = False
                    session._c0d3r.model_id = fallback_model
                    _emit_live(f"tool_loop: switching to fallback model {fallback_model}")
                except Exception:
                    pass
            history.append("note: model call timed out; retrying with shorter context")
            mini_prompt = (
                "Return ONLY JSON with keys: commands (list of strings) and final (string or empty). "
                "Focus on executing the request with minimal steps.\n"
                f"Request:\n{prompt}\n"
            )
            if _requires_rigorous_constraints(prompt):
                command_model = os.getenv("C0D3R_COMMAND_MODEL", "mistral.mistral-large-3-675b-instruct")
                saved_model = getattr(session._c0d3r, "model_id", "")
                saved_multi = getattr(session._c0d3r, "multi_model", True)
                saved_rigorous = getattr(session._c0d3r, "rigorous_mode", False)
                saved_profile = getattr(session._c0d3r, "inference_profile", "")
                try:
                    session._c0d3r.model_id = command_model
                    session._c0d3r.multi_model = False
                    session._c0d3r.rigorous_mode = False
                    session._c0d3r.inference_profile = ""
                    response = _call_with_timeout(
                        session._safe_send,
                        timeout_s=max(10.0, _model_timeout_value() / 2),
                        kwargs={"prompt": mini_prompt, "stream": stream, "images": images, "stream_callback": stream_callback},
                    )
                finally:
                    session._c0d3r.model_id = saved_model
                    session._c0d3r.multi_model = saved_multi
                    session._c0d3r.rigorous_mode = saved_rigorous
                    session._c0d3r.inference_profile = saved_profile
            else:
                response = _call_with_timeout(
                    session._safe_send,
                    timeout_s=max(10.0, _model_timeout_value() / 2),
                    kwargs={"prompt": mini_prompt, "stream": stream, "images": images, "stream_callback": stream_callback},
                )
            if response is None:
                _emit_live("tool_loop: minimal prompt timed out")
                _diag_log("tool_loop: minimal prompt timeout")
                last_error = "model call timed out (minimal prompt)"
                model_timeouts += 1
                if model_timeouts >= 5:
                    _emit_live("tool_loop: repeated timeouts; backing off for 3s")
                    time.sleep(3.0)
                continue
        _diag_log("tool_loop: model call complete")
        commands, final = _extract_commands(response)
        _emit_live(f"tool_loop: model returned {len(commands)} commands, final={'yes' if final else 'no'}")
        if final and not commands:
            if _requires_commands_for_task(prompt) and not (success or any_success):
                history.append("note: commands required for this task; no final allowed without verified success")
                continue
            # require explicit completion signal
            if "complete" not in final.lower():
                history.append("note: final must state completion and checklist verified")
                continue
            if full_completion and not _plan_and_checklist_present():
                history.append("note: plan/checklist files missing; create runtime/c0d3r/plan.md and checklist.md")
                continue
            if full_completion and not _plan_is_substantial():
                history.append("note: plan.md is empty/insufficient; expand it before finalizing")
                continue
            if full_completion and not _checklist_has_mapping():
                history.append("note: checklist must include source->decision->code mapping")
                continue
            if full_completion and not _checklist_is_complete():
                history.append("note: checklist has unchecked items; finish all requirements before finalizing")
                continue
            if full_completion and require_benchmark and not _benchmark_evidence_present():
                history.append("note: benchmark evidence missing; create runtime/c0d3r/benchmarks.json before finalizing")
                continue
            if full_completion and require_tests and not _tests_passed_recently():
                history.append("note: recent tests did not pass; re-run tests and fix failures before finalizing")
                continue
            if full_completion and _disallow_placeholder_code(workdir):
                history.append("note: placeholder implementation detected; implement real logic before finalizing")
                continue
        if full_completion and _has_empty_tests(workdir):
            history.append("note: empty test files detected; add real tests before finalizing")
            continue
        if full_completion and _requires_rigorous_constraints(prompt) and not _unbounded_math_ready():
            history.append("note: missing equations/research links; unbounded resolver must populate them")
            continue
        if full_completion and _requires_rigorous_constraints(prompt) and not _equation_graph_ready():
            history.append("note: equation graph is empty; ingestion must populate it before finalizing")
            continue
        return final
        if not commands and not final:
            no_command_count += 1
            if full_completion:
                history.append("note: no commands returned; running research to fill gaps")
                _emit_live("tool_loop: no commands; triggering research")
                research_note = _pre_research(session, prompt)
                if research_note:
                    history.append("[research]\n" + research_note[:2000])
                _emit_live("tool_loop: requesting command generator due to empty commands")
                alt_model = os.getenv("C0D3R_COMMAND_MODEL", "anthropic.claude-opus-4-20250514-v1:0")
                saved_model = getattr(session._c0d3r, "model_id", "")
                saved_rigorous = getattr(session._c0d3r, "rigorous_mode", False)
                try:
                    session._c0d3r.rigorous_mode = False
                    resolver = getattr(session._c0d3r, "_resolve_profile_cached", None)
                    resolved_model = resolver(alt_model) if callable(resolver) else alt_model
                    session._c0d3r.model_id = resolved_model
                    saved_profile = getattr(session._c0d3r, "inference_profile", "")
                    session._c0d3r.inference_profile = ""
                    base_request = _strip_context_block(prompt)
                    emergency_prompt = (
                        "Return ONLY JSON with keys: commands (list of strings) and final (string or empty). "
                        "Provide 2-5 concrete shell commands to make progress on the task. "
                        "Do NOT return an empty commands list.\n"
                        f"CWD: {workdir}\n"
                        f"Request:\n{base_request}\n"
                    )
                    try:
                        raw = session._c0d3r._invoke_model(resolved_model, emergency_prompt, images=images)
                    except Exception:
                        raw = ""
                    if raw:
                        commands, final = _extract_commands(raw)
                        _save_empty_commands_response(raw)
                finally:
                    session._c0d3r.model_id = saved_model
                    session._c0d3r.rigorous_mode = saved_rigorous
                    try:
                        session._c0d3r.inference_profile = saved_profile
                    except Exception:
                        pass
                if not commands:
                    _emit_live("tool_loop: command generator still empty; using fallback inspection commands")
                    commands = _fallback_inspection_commands(workdir)
                    final = ""
            else:
                return response
        if commands:
            no_command_count = 0
        if no_command_count >= 2:
            _emit_live("tool_loop: no commands twice; switching to command-generator model")
            alt_model = os.getenv("C0D3R_COMMAND_MODEL", "anthropic.claude-opus-4-20250514-v1:0")
            saved_model = getattr(session._c0d3r, "model_id", "")
            saved_rigorous = getattr(session._c0d3r, "rigorous_mode", False)
            try:
                session._c0d3r.rigorous_mode = False
                session._c0d3r.model_id = alt_model
                emergency_prompt = (
                    "Return ONLY JSON with keys: commands (list of strings) and final (string or empty). "
                    "Provide 2-5 concrete shell commands to make progress on the task. "
                    "Do NOT return an empty commands list.\n"
                    f"Request:\n{prompt}\n"
                )
                response = _call_with_timeout(
                    session._safe_send,
                    timeout_s=max(20.0, _model_timeout_value()),
                    kwargs={"prompt": emergency_prompt, "stream": stream, "images": images, "stream_callback": stream_callback},
                )
                if response:
                    commands, final = _extract_commands(response)
            finally:
                session._c0d3r.model_id = saved_model
                session._c0d3r.rigorous_mode = saved_rigorous
            if not commands:
                _emit_live("tool_loop: command-generator still empty; running fallback inspection commands")
                fallback = _fallback_inspection_commands(workdir)
                commands = fallback
                final = ""
        if commands and _commands_only_runtime(commands):
            history.append("note: commands only touch runtime/c0d3r; must operate on project files too")
            _emit_live("tool_loop: commands only touched runtime; inserting inspection commands")
            commands = _fallback_inspection_commands(workdir) + commands
        wrote_project = False
        written_files: set[str] = set()
        defer_tests = os.getenv("C0D3R_DEFER_TESTS", "1").strip().lower() not in {"0", "false", "no", "off"}
        pending_test_targets: List[Path | None] = []
        for cmd in commands[:8]:
            usage_tracker.set_status("executing", cmd)
            _emit_live(f"exec: {cmd}")
            if cmd.lower().startswith("cd "):
                cmd = "::cd " + cmd[3:].strip()
            if os.name == "nt" and ("<<" in cmd or cmd.strip().lower().startswith("cat ")):
                history.append("note: bash heredoc/cat redirection blocked on Windows; use echo/set-content")
                _append_tool_log(log_path, cmd, 1, "", "blocked: heredoc not supported")
                continue
            if ("mkdir" in cmd.lower() or "new-item" in cmd.lower()) and created_dirs:
                history.append("note: directory already created; do not create more directories")
                _append_tool_log(log_path, cmd, 1, "", "mkdir blocked; reuse existing directory")
                continue
            if (("mkdir" in cmd.lower() or "new-item -itemtype directory" in cmd.lower()) and
                not allow_new_root_dirs and _mkdir_targets_root(cmd, workdir)):
                history.append("note: creating new top-level directories is blocked; use existing project folders")
                _append_tool_log(log_path, cmd, 1, "", "mkdir blocked at repo root")
                continue
            if _is_type_nul(cmd) and not _is_runtime_command(cmd):
                history.append("note: type nul file resets are blocked; write real content instead")
                _append_tool_log(log_path, cmd, 1, "", "blocked: type nul > file")
                continue
            if _is_benchmark_echo(cmd):
                history.append("note: benchmarks.json must be generated by running a benchmark script, not echo")
                _append_tool_log(log_path, cmd, 1, "", "blocked: echo > benchmarks.json")
                continue
            if _requires_rigorous_constraints(prompt) and cmd.lower().strip().startswith("touch ") and "tests" in cmd.lower():
                history.append("note: empty test file creation blocked; write real tests instead")
                _append_tool_log(log_path, cmd, 1, "", "blocked: touch test file")
                continue
            if _requires_rigorous_constraints(prompt) and cmd.lower().strip().startswith("touch ") and ".py" in cmd.lower():
                history.append("note: empty source file creation blocked; write real code instead")
                _append_tool_log(log_path, cmd, 1, "", "blocked: touch sat source")
                continue
            if "pip install" in cmd.lower() and ("sat" in prompt.lower()) and ("django" in cmd.lower()):
                history.append("note: pip install for Django packages blocked for SAT task; use confcutdir for tests")
                _append_tool_log(log_path, cmd, 1, "", "blocked: unrelated pip install")
                continue
            if "pytest" in cmd.lower() and _requires_rigorous_constraints(prompt):
                history.append("note: pytest command from model blocked; c0d3r runs targeted tests with confcutdir")
                _append_tool_log(log_path, cmd, 1, "", "blocked: pytest command")
                continue
            if _is_write_command(cmd):
                target_path = _infer_written_path(cmd, workdir)
                if target_path:
                    key = str(target_path.resolve())
                    if key in written_files:
                        history.append("note: repeated writes to the same file in one step are blocked")
                        _append_tool_log(log_path, cmd, 1, "", "blocked: repeated write same file")
                        continue
                    written_files.add(key)
            if "ionic serve" in cmd and not cmd.startswith("::bg"):
                history.append("note: ionic serve must be run with ::bg to avoid blocking")
                _append_tool_log(log_path, cmd, 1, "", "ionic serve blocked; use ::bg")
                continue
            if _is_pip_install(cmd) and not _requires_pip_install(prompt):
                history.append("note: pip install skipped; not required for skeleton setup")
                _append_tool_log(log_path, cmd, 1, "", "pip install skipped")
                continue
            if "ionic start ." in cmd:
                history.append("note: ionic start . is not allowed; use ionic start <appname> from parent dir")
                _append_tool_log(log_path, cmd, 1, "", "ionic start . blocked")
                continue
            if "ionic start" in cmd and "--no-interactive" not in cmd:
                history.append("note: ionic start must include --no-interactive --no-confirm")
                _append_tool_log(log_path, cmd, 1, "", "ionic start missing flags")
                continue
            if ("npm install" in cmd or "ionic serve" in cmd) and not (workdir / "package.json").exists():
                history.append("note: package.json missing; project not created yet")
                _append_tool_log(log_path, cmd, 1, "", "blocked; package.json missing")
                continue
            if cmd.startswith("::"):
                code, stdout, stderr, new_cwd = _execute_meta_command(cmd, workdir)
                if new_cwd:
                    workdir = new_cwd
                _append_tool_log(log_path, cmd, code, stdout, stderr)
            else:
                handled, code, stdout, stderr = _execute_file_command(cmd, workdir)
                if handled:
                    _append_tool_log(log_path, cmd, code, stdout, stderr)
                else:
                    normalized = _normalize_command(cmd, workdir)
                    if not normalized.strip():
                        history.append("note: command skipped (no-op after normalization)")
                        continue
                    handled, code, stdout, stderr = _execute_file_command(normalized, workdir)
                    if handled:
                        _append_tool_log(log_path, normalized, code, stdout, stderr)
                    else:
                        code, stdout, stderr = run_command(normalized, cwd=workdir, timeout_s=_command_timeout_s(normalized))
                        _append_tool_log(log_path, normalized, code, stdout, stderr)
            if code != 0 and (stderr or stdout):
                err_blob = stderr if stderr.strip() else stdout
                remediated, _, _, _ = _attempt_auto_fix(cmd, err_blob, workdir, run_command, log_path)
                if remediated:
                    # Re-run the original command after remediation.
                    code, stdout, stderr = run_command(cmd, cwd=workdir, timeout_s=_command_timeout_s(cmd))
                    _append_tool_log(log_path, cmd, code, stdout, stderr)
            _emit_live(f"exec result: exit={code}")
            if stdout.strip():
                _emit_live(f"stdout: {stdout.strip()[:800]}")
            if stderr.strip():
                _emit_live(f"stderr: {stderr.strip()[:800]}")
            _append_evidence(cmd, code, stdout, stderr)
            if stderr.strip():
                last_error = stderr.strip()[:1200]
                gap_score = max(gap_score, _gap_score(stderr))
            if require_tests and _is_write_command(cmd):
                target = _infer_written_path(cmd, workdir)
                if _should_test_write(target):
                    if not _is_runtime_path(target):
                        wrote_project = True
                    if defer_tests:
                        pending_test_targets.append(target)
                    else:
                        _emit_live(f"post-write: running targeted tests for {target or 'project'}")
                        tests_ran, tests_ok = _run_tests_for_project(workdir, run_command, usage_tracker, log_path, target=target)
                        if not tests_ok:
                            history.append("error: tests failed after write; fix and re-run before continuing")
                            continue
            snippet = f"$ {cmd}\n(exit {code})\n{stdout.strip()}\n{stderr.strip()}".strip()
            history.append(snippet[:4000])
            if (workdir / "package.json").exists() and workdir not in known_projects:
                known_projects.append(workdir)
            if base_snapshot and _requires_new_projects_dir(prompt):
                new_dirs = _diff_projects_dir(base_snapshot)
                if new_dirs:
                    created_dirs = new_dirs
            if cmd.startswith("::wait_http") and code == 0 and known_projects:
                success = True
                any_success = True
                if require_tests:
                    tests_ran, tests_ok = _run_tests_for_project(known_projects[-1], run_command, usage_tracker, log_path)
                    if not tests_ok:
                        history.append("error: tests failed; fix and re-run tests before finalizing")
                        success = False
                        continue
                return f"Success: ionic serve reachable; project at {known_projects[-1]}"
            if _requires_skeleton(prompt):
                skeleton = _find_skeleton_root(workdir)
                if skeleton:
                    success = True
                    any_success = True
                    if require_tests:
                        tests_ran, tests_ok = _run_tests_for_project(Path(skeleton), run_command, usage_tracker, log_path)
                        if not tests_ok:
                            history.append("error: tests failed; fix and re-run tests before finalizing")
                            success = False
                            continue
                    return f"Success: skeleton created at {skeleton}"
            if code != 0:
                history.append("error: previous command failed; analyze stderr and retry with correction")
            else:
                any_success = True
            if _is_write_command(cmd) and not _is_runtime_command(cmd):
                wrote_project = True
        if full_completion and not wrote_project:
            history.append("note: no project files were written this step; continue with concrete file edits")
            consecutive_no_progress += 1
        else:
            consecutive_no_progress = 0
        if require_tests and defer_tests and pending_test_targets:
            unique_targets: List[Path | None] = []
            for target in pending_test_targets:
                if target not in unique_targets:
                    unique_targets.append(target)
            for target in unique_targets:
                _emit_live(f"post-write: running targeted tests for {target or 'project'}")
                tests_ran, tests_ok = _run_tests_for_project(workdir, run_command, usage_tracker, log_path, target=target)
                if not tests_ok:
                    history.append("error: tests failed after write; fix and re-run before continuing")
                    test_failures += 1
                    break
        if _unbounded_trigger(consecutive_no_progress, test_failures, model_timeouts) and not unbounded_resolved:
            _emit_live("unbounded: detected spiral; building bounded objective matrix")
            unbounded_payload = _resolve_unbounded_request(session, prompt)
            if unbounded_payload:
                unbounded_resolved = True
                history.append("[unbounded]\n" + unbounded_payload.get("bounded_task", "").strip())
                _append_unbounded_matrix(unbounded_payload)
                prompt = _apply_unbounded_constraints(prompt, unbounded_payload)
                _apply_behavior_insights(unbounded_payload, behavior_log)
        if full_completion:
            _append_verification_snapshot(workdir, run_command, history)
            _update_system_map(workdir)
            ok = _run_quality_checks(workdir, run_command, usage_tracker, log_path)
            if not ok:
                history.append("error: quality/security checks failed; fix and re-run")
                continue
            if not _spec_validator():
                history.append("error: spec validator failed (checklist incomplete or missing verification)")
                continue
        # post-check: if prompt asks for new dir under Projects, ensure one exists
        if base_snapshot and _requires_new_projects_dir(prompt):
            new_dirs = _diff_projects_dir(base_snapshot)
            if not new_dirs:
                history.append("note: no new directory detected under C:/Users/Adam/Projects")
    return history[-1] if history else "No output."


def _normalize_command(cmd: str, workdir: Path) -> str:
    """
    Normalize common commands for Windows PowerShell.
    """
    if os.name != "nt":
        return cmd
    # Split chained commands for PowerShell.
    if "&&" in cmd:
        parts = [p.strip() for p in cmd.split("&&") if p.strip()]
        normalized_parts = [_normalize_command(p, workdir) for p in parts]
        normalized_parts = [p for p in normalized_parts if p]
        return " ; ".join(normalized_parts)
    # Remove shell activation steps that don't apply in non-interactive runs.
    if "venv\\Scripts\\activate" in cmd or "venv/Scripts/activate" in cmd:
        cmd = cmd.replace("venv\\Scripts\\activate", "").replace("venv/Scripts/activate", "").strip()
    # Expand multi-arg mkdir into separate New-Item calls.
    if cmd.lower().startswith("mkdir "):
        parts = [p for p in cmd.split()[1:] if p not in {"-p", "--parents"}]
        if parts:
            targets = [p.strip().strip('"') for p in parts]
            commands = []
            for target in targets:
                target_path = (workdir / target) if not Path(target).is_absolute() else Path(target)
                if target_path.exists():
                    continue
                if target_path.resolve() == workdir.resolve():
                    continue
                commands.append(f'New-Item -ItemType Directory -Force "{target}"')
            if not commands:
                return ""
            return " ; ".join(commands)
    # Handle echo redirection for file creation/appends.
    if cmd.lower().startswith("echo.") and ">" in cmd:
        # Empty file write (cmd style). Convert to Set-Content.
        target = cmd.split(">", 1)[1].strip()
        return f'Set-Content -Encoding UTF8 -Path "{target}" -Value ""'
    if cmd.lower().startswith("echo ") and (">" in cmd):
        try:
            if ">>" in cmd:
                left, right = cmd.split(">>", 1)
                content = left[len("echo ") :].strip()
                target = right.strip()
                return f'Add-Content -Encoding UTF8 -Path "{target}" -Value "{content}"'
            if ">" in cmd:
                left, right = cmd.split(">", 1)
                content = left[len("echo ") :].strip()
                target = right.strip()
                return f'Set-Content -Encoding UTF8 -Path "{target}" -Value "{content}"'
        except Exception:
            return cmd
    if cmd.lower().startswith("type nul >"):
        target = cmd.split(">", 1)[1].strip()
        return f'Set-Content -Encoding UTF8 -Path "{target}" -Value ""'
    # Convert touch to New-Item -ItemType File
    if cmd.lower().startswith("touch "):
        parts = cmd.split()[1:]
        if parts:
            commands = []
            for target in parts:
                target_path = (workdir / target) if not Path(target).is_absolute() else Path(target)
                if target_path.exists():
                    continue
                commands.append(f'New-Item -ItemType File -Force "{target}"')
            if not commands:
                return ""
            return " ; ".join(commands)
    # Ensure Set-Content/Add-Content values are safe by translating to python file write.
    lower = cmd.lower()
    if lower.startswith("dir "):
        # Translate common cmd-style dir flags to PowerShell.
        if "/b" in lower and "/a-d" in lower:
            return "Get-ChildItem -File -Name"
        if "/b" in lower:
            return "Get-ChildItem -Name"
    if lower.startswith("set-content") or lower.startswith("add-content"):
        try:
            path_token = '-Path "'
            value_token = '-Value "'
            if path_token in cmd and value_token in cmd:
                path_part = cmd.split(path_token, 1)[1]
                path = path_part.split('"', 1)[0]
                value_part = cmd.split(value_token, 1)[1]
                value = value_part.rsplit('"', 1)[0]
                import base64 as _b64

                encoded = _b64.b64encode(value.encode("utf-8")).decode("ascii")
                mode = "a" if lower.startswith("add-content") else "w"
                return (
                    'python -c "import base64, pathlib; '
                    f'p=pathlib.Path(r\'{path}\'); p.parent.mkdir(parents=True, exist_ok=True); '
                    f'p.write_text(base64.b64decode(\'{encoded}\').decode(\'utf-8\'), encoding=\'utf-8\') '
                    f'if \'{mode}\'==\'w\' else p.open(\'a\', encoding=\'utf-8\').write(base64.b64decode(\'{encoded}\').decode(\'utf-8\'))"'
                )
        except Exception:
            return cmd
    return cmd


def _run_scientific_loop(
    session: C0d3rSession,
    prompt: str,
    workdir: Path,
    run_command,
    *,
    images: List[str] | None,
    stream: bool,
    stream_callback,
    usage_tracker,
) -> str:
    # Observe: fixed inspection bundle (empirical evidence)
    bundle = []
    for cmd in [
        "git status -sb",
        "git log -1 --oneline",
        "rg -n \"TODO|FIXME|HACK\" -S",
        "Get-ChildItem -Name" if os.name == "nt" else "ls -1",
    ]:
        code, stdout, stderr = run_command(cmd, cwd=workdir)
        bundle.append(
            f"$ {cmd}\n(exit {code})\n{(stdout or '').strip()}\n{(stderr or '').strip()}"
        )
    evidence = "\n\n".join(bundle)

    analysis_prompt = (
        "Use the scientific method. Do not speculate. "
        "Use ONLY the provided empirical evidence. "
        "Return ONLY JSON with keys:\n"
        "observations (list of strings),\n"
        "hypotheses (list of strings),\n"
        "tests (list of strings),\n"
        "findings (list of {severity, claim, evidence_cmd, evidence_excerpt, files}),\n"
        "conclusion (string),\n"
        "next_steps (list of strings).\n"
        "Every finding must cite evidence_cmd and evidence_excerpt from the bundle.\n\n"
        f"User request:\n{prompt}\n\n"
        f"Evidence bundle:\n{evidence}\n"
    )
    usage_tracker.add_input(analysis_prompt)
    response = _call_with_timeout(
        session._safe_send,
        timeout_s=_model_timeout_s(),
        kwargs={
            "prompt": analysis_prompt,
            "stream": stream,
            "images": images,
            "evidence_bundle": evidence,
            "stream_callback": stream_callback,
        },
    )
    if response is None:
        return "Scientific analysis timed out; retry with a smaller evidence bundle."
    commands, final = _extract_commands(response)
    if final:
        return final
    return response


def _extract_commands(text: str) -> Tuple[List[str], str]:
    try:
        import json

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                payload_list = json.loads(cleaned)
                if isinstance(payload_list, list):
                    commands = [str(c) for c in payload_list if str(c).strip()]
                    return commands, ""
            except Exception:
                pass
        payload = json.loads(_extract_json(cleaned))
        commands = payload.get("commands") or []
        if isinstance(commands, list):
            commands = [str(c) for c in commands if str(c).strip()]
        else:
            commands = []
        final = str(payload.get("final") or "").strip()
        return commands, final
    except Exception:
        try:
            import re

            commands: List[str] = []
            cmd_match = re.search(r'"commands"\s*:\s*\[(.*?)\]', text, re.S)
            if cmd_match:
                inner = cmd_match.group(1)
                commands = [c.strip().strip('"') for c in inner.split(",") if c.strip()]
            if not commands:
                # Fallback: parse bullet/numbered lists as commands
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                for ln in lines:
                    if ln.startswith(("-", "*")):
                        commands.append(ln.lstrip("-* ").strip())
                    elif re.match(r"^\d+\.\s+", ln):
                        commands.append(re.sub(r"^\d+\.\s+", "", ln))
            final_match = re.search(r'"final"\s*:\s*"(.+)"', text, re.S)
            if final_match:
                final = final_match.group(1).strip()
                return commands, final
        except Exception:
            pass
        return [], ""


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "{}"
    return text[start : end + 1]


def _safe_json(text: str):
    try:
        import json as _json

        return _json.loads(_extract_json(text))
    except Exception:
        return {}


def _strip_context_block(prompt: str) -> str:
    if not prompt:
        return ""
    marker = "User request:"
    if marker in prompt:
        return prompt.split(marker, 1)[-1].strip()
    return prompt.strip()


def _math_grounding_block(session: C0d3rSession, prompt: str, workdir: Path) -> str:
    """
    Convert the prompt into mathematical form, solve with local tools when possible,
    and return a grounding block to prepend to the request.
    """
    base_request = _strip_context_block(prompt)
    system = (
        "Return ONLY JSON with keys: variables (list), unknowns (list), "
        "equations (list of strings), constraints (list), "
        "mapping (list), research_questions (list). "
        "Express the request as math; if you cannot form explicit equations, "
        "provide symbolic placeholders AND at least 3 research_questions to "
        "identify missing constants/relations."
    )
    def _heartbeat(label: str, stop_event: threading.Event, interval: float = 5.0) -> threading.Thread:
        def _runner():
            tick = 0
            while not stop_event.is_set():
                _emit_live(f"{label}... ({tick * interval:.0f}s)")
                stop_event.wait(interval)
                tick += 1

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return t

    _emit_live("math_grounding: model pass 1 (equations)")
    try:
        stop = threading.Event()
        _heartbeat("math_grounding: awaiting model pass 1", stop)
        raw = session.send(prompt=base_request, stream=False, system=system)
        stop.set()
        payload = _safe_json(raw)
    except Exception as exc:
        try:
            stop.set()
        except Exception:
            pass
        _emit_live(f"math_grounding: model pass 1 error: {exc}")
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    research_questions = payload.get("research_questions") or []
    if not research_questions:
        research_questions = [
            f"Find equations, constants, or constraints needed to model: {base_request}"
        ]
    research_notes = ""
    _emit_live("math_grounding: research starting")
    try:
        stop = threading.Event()
        _heartbeat("math_grounding: awaiting research", stop)
        # Use internal research pipeline to build evidence
        queries = "\n".join(str(q) for q in research_questions[:6])
        research_notes = session._c0d3r._research(queries)
        stop.set()
    except Exception:
        try:
            stop.set()
        except Exception:
            pass
        research_notes = ""
    _emit_live("math_grounding: research complete")
    if research_notes:
        _emit_live("math_grounding: model pass 2 (equations from research)")
        system2 = (
            "Return ONLY JSON with keys: equations (list), constraints (list), "
            "gap_fill_steps (list), research_links (list). "
            "Convert the research into explicit equations and constraints."
        )
        try:
            raw2 = session.send(
                prompt=f"Request:\n{base_request}\n\nResearch:\n{research_notes}",
                stream=False,
                system=system2,
            )
            payload2 = _safe_json(raw2)
            if isinstance(payload2, dict):
                for key in ("equations", "constraints", "gap_fill_steps", "research_links"):
                    if payload2.get(key):
                        payload[key] = payload2.get(key)
        except Exception:
            pass
        if not payload.get("research_links"):
            try:
                links = re.findall(r"https?://[^\s\]\)\"']+", research_notes)
                if links:
                    payload["research_links"] = links
            except Exception:
                pass
        try:
            from services.equation_graph import ingest_equations, extract_equations

            if not payload.get("equations"):
                payload["equations"] = extract_equations(research_notes)
            for link in payload.get("research_links") or []:
                ingest_equations(
                    research_notes,
                    source_title="Research Notes",
                    source_url=str(link),
                    discipline_tags=["cross-disciplinary"],
                    raw_excerpt=research_notes[:1000],
                )
        except Exception:
            pass

    solutions = []
    try:
        import sympy as _sp

        eqs = []
        symbols = {}
        for eq in payload.get("equations") or []:
            try:
                if "=" in eq:
                    left, right = eq.split("=", 1)
                    expr = _sp.Eq(_sp.sympify(left), _sp.sympify(right))
                else:
                    expr = _sp.sympify(eq)
                eqs.append(expr)
            except Exception:
                continue
        for var in payload.get("unknowns") or []:
            try:
                symbols[str(var)] = _sp.symbols(str(var))
            except Exception:
                continue
        if eqs and symbols:
            try:
                sol = _sp.solve(eqs, list(symbols.values()), dict=True)
                solutions = sol if isinstance(sol, list) else [sol]
            except Exception:
                solutions = []
    except Exception:
        solutions = []

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request": base_request,
        "variables": payload.get("variables") or [],
        "unknowns": payload.get("unknowns") or [],
        "equations": payload.get("equations") or [],
        "constraints": payload.get("constraints") or [],
        "gap_fill_steps": payload.get("gap_fill_steps") or [],
        "research_links": payload.get("research_links") or [],
        "solutions": solutions,
    }
    try:
        out_dir = Path("runtime/c0d3r")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "math_grounding.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        with (out_dir / "math_grounding_history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass
    block = [
        "[math_grounding]",
        f"variables: {record['variables']}",
        f"unknowns: {record['unknowns']}",
        f"equations: {record['equations']}",
        f"constraints: {record['constraints']}",
        f"gap_fill_steps: {record['gap_fill_steps']}",
        f"research_links: {record['research_links']}",
        f"solutions: {record['solutions']}",
    ]
    return "\n".join(block)


def _typewriter_callback(usage, header=None, controller=None):
    delay_ms = float(os.getenv("C0D3R_TYPEWRITER_MS", "8") or "8")
    delay_s = max(0.0, delay_ms / 1000.0)
    if header:
        header.render()
        header.resume()

    def _callback(chunk: str) -> None:
        if controller and controller.interrupted:
            return
        usage.add_output(chunk)
        if header:
            header.update()
        for ch in chunk:
            if controller and controller.interrupted:
                return
            sys.stdout.write(ch)
            sys.stdout.flush()
            if ch.strip():
                time.sleep(delay_s)

    return _callback


def _typewriter_print(text: str, usage, header=None, controller=None) -> None:
    cb = _typewriter_callback(usage, header=header, controller=controller)
    cb(text)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")
        sys.stdout.flush()


def _save_empty_commands_response(text: str) -> None:
    try:
        if not text:
            return
        path = Path("runtime/c0d3r/empty_commands_response.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n--- {ts} ---\n")
            fh.write(text.strip() + "\n")
    except Exception:
        pass


def _render_json_response(text: str) -> str:
    payload = _safe_json(text)
    if not isinstance(payload, dict):
        return ""
    if "status_updates" in payload:
        lines = []
        for item in payload.get("status_updates") or []:
            lines.append(f"[working] {item}")
        final = str(payload.get("final") or "").strip()
        if final:
            lines.append("\n" + final if lines else final)
        return "\n".join(lines).strip()
    if "final" in payload:
        return str(payload.get("final") or "")
    if "conclusion" in payload:
        lines = []
        if payload.get("observations"):
            lines.append("Observations:")
            for item in payload.get("observations") or []:
                lines.append(f"- {item}")
        if payload.get("findings"):
            lines.append("\nFindings:")
            for item in payload.get("findings") or []:
                claim = item.get("claim") if isinstance(item, dict) else item
                lines.append(f"- {claim}")
        lines.append("\nConclusion:")
        lines.append(str(payload.get("conclusion") or "").strip())
        steps = payload.get("next_steps") or []
        if steps:
            lines.append("\nNext steps:")
            for step in steps:
                lines.append(f"- {step}")
        return "\n".join([l for l in lines if l]).strip()
    return ""


def _run_repl(
    session: C0d3rSession,
    usage: UsageTracker,
    workdir: Path,
    run_command,
    *,
    scientific: bool,
    tool_loop: bool,
    initial_prompt: str | None = None,
    header: "HeaderRenderer | None" = None,
) -> int:
    from services.conversation_memory import ConversationMemory

    memory = ConversationMemory(Path("runtime/c0d3r/conversation.jsonl"))
    summary_path = Path("runtime/c0d3r/summary.txt")
    summary = summary_path.read_text(encoding="utf-8", errors="ignore") if summary_path.exists() else ""
    pending = initial_prompt
    oneshot = os.getenv("C0D3R_ONESHOT", "").strip().lower() in {"1", "true", "yes", "on"}
    minimal = oneshot or os.getenv("C0D3R_MINIMAL_CONTEXT", "").strip().lower() in {"1", "true", "yes", "on"}
    allow_interrupt = sys.stdin.isatty() and initial_prompt is None
    header = header or HeaderRenderer(usage)
    budget = BudgetTracker(header.budget_usd)
    budget.enabled = header.budget_enabled
    header.render()
    _emit_live("repl: header rendered")
    while True:
        try:
            header.freeze()
            prompt = pending if pending is not None else _read_input(workdir)
            header.resume()
            pending = None
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0
        _emit_live(f"repl: prompt received (len={len(prompt)})")
        if not prompt.strip():
            continue
        if prompt.strip().lower() in {"/exit", "/quit"}:
            return 0
        if prompt.startswith("!"):
            cmd = prompt[1:].strip()
            if cmd.startswith("cd "):
                new_dir = cmd[3:].strip().strip('"')
                workdir = (workdir / new_dir).resolve() if not Path(new_dir).is_absolute() else Path(new_dir).resolve()
                os.chdir(workdir)
                print(f"[cwd] {workdir}")
                continue
            code, stdout, stderr = run_command(cmd, cwd=workdir)
            if stdout.strip():
                print(stdout.strip())
            if stderr.strip():
                print(stderr.strip())
            continue

        context = ""
        if not minimal:
            context = memory.build_context(summary)
        try:
            from services.system_probe import system_probe_context
            probe_block = system_probe_context(workdir)
            context = f"{probe_block}\n{context}" if context else probe_block
        except Exception:
            pass
        if not minimal:
            recall_hits = memory.search_if_referenced(prompt, limit=5)
            if recall_hits:
                context = context + "\n\n[recall]\n" + "\n".join(recall_hits)
        system = (
            "Role: You are a senior software engineer. The user is a project manager. "
            "Return ONLY JSON with keys: status_updates (list of short strings) and final (string). "
            "Provide brief, concrete status updates before the final response. "
            "Avoid private chain-of-thought; summarize reasoning at a high level."
        )
        full_prompt = f"{context}\n\nUser:\n{prompt}"
        usage.add_input(full_prompt)
        print("[status] Planning...")
        print("[status] Executing...")
        _emit_live("repl: starting request")
        usage.set_status("planning", "routing + research")
        mode = _decide_mode(session, prompt, default_scientific=scientific, default_tool_loop=tool_loop)
        research_summary = ""
        if mode in {"tool_loop", "scientific"}:
            usage.set_status("research", "gathering references")
            if minimal or os.getenv("C0D3R_DISABLE_PRERESEARCH", "").strip().lower() in {"1", "true", "yes", "on"}:
                research_summary = ""
            else:
                _emit_live("repl: pre-research starting")
                research_summary = _pre_research(session, prompt)
                _emit_live(f"repl: pre-research complete (len={len(research_summary)})")
                _emit_live("repl: pre-research complete")
        controller = InterruptController()
        if allow_interrupt:
            controller.start()
        try:
            if mode == "scientific":
                usage.set_status("executing", "scientific analysis")
                _emit_live("repl: scientific loop starting")
                response = _run_scientific_loop(
                    session, full_prompt, workdir, run_command, images=None, stream=False, stream_callback=None, usage_tracker=usage
                )
                _emit_live("repl: scientific loop complete")
            elif mode == "tool_loop":
                usage.set_status("executing", "local commands")
                _emit_live("repl: tool loop starting")
                response = _run_tool_loop(
                    session, f"{full_prompt}\n\n[research]\n{research_summary}" if research_summary else full_prompt,
                    workdir, run_command, images=None, stream=False, stream_callback=None, usage_tracker=usage
                )
                _emit_live("repl: tool loop complete")
            else:
                usage.set_status("executing", "direct response")
                _emit_live("repl: direct model call starting")
                response = _call_with_timeout(
                    session._safe_send,
                    timeout_s=_model_timeout_s(),
                    kwargs={"prompt": full_prompt, "stream": False, "system": system},
                )
                if response is None:
                    response = "Model call timed out. Try again or adjust C0D3R_MODEL_TIMEOUT_S."
                _emit_live("repl: direct model call complete")
        finally:
            controller.stop()
        rendered = _render_json_response(response) or response
        # update budget usage
        try:
            from services.bedrock_pricing import estimate_cost
            in_cost, out_cost = estimate_cost(usage.model_id or "", usage.input_tokens, usage.output_tokens)
            if in_cost is not None and out_cost is not None:
                total_cost = in_cost + out_cost
                # approximate incremental spend since last check
                budget.add_spend(max(0.0, total_cost - budget.spent_usd))
        except Exception:
            pass
        if controller.interrupted and allow_interrupt:
            print("\n[pause] Input detected. Add your note and press Enter to resume.")
            note = input("> ").strip()
            if note:
                continuation = (
                    f"{context}\n\nUser:\n{prompt}\n\nAssistant (partial):\n{rendered}\n\n"
                    f"User interruption note: {note}\n\nContinue and finish the response."
                )
                usage.add_input(continuation)
                response = session.send(continuation, stream=False, system=system)
                rendered = _render_json_response(response) or response
        _typewriter_print(rendered, usage, header=header, controller=controller)
        sys.stdout.write("\n")
        sys.stdout.flush()
        if budget.exceeded():
            choice = _prompt_budget_choice(budget.budget_usd)
            if choice == "continue":
                budget.reset(budget.budget_usd)
            elif choice == "set":
                new_budget = _prompt_budget_value(budget.budget_usd)
                budget.reset(new_budget)
                header.budget_usd = new_budget
            elif choice == "indefinite":
                budget.disable()
                header.budget_enabled = False
            else:
                return 0
        if not minimal:
            memory.append(prompt, response)
            summary = memory.update_summary(summary, prompt, response, session)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(summary, encoding="utf-8")
        if oneshot:
            return 0


def _read_input(workdir: Path) -> str:
    return input(f"[{workdir}]> ")


_PRICING_REFRESHED: set[str] = set()


def _is_pricing_stale(pricing) -> bool:
    if not pricing or not getattr(pricing, "as_of", None):
        return True
    raw = str(pricing.as_of)
    try:
        if len(raw) == 7:
            as_of = datetime.datetime.strptime(raw, "%Y-%m")
        else:
            as_of = datetime.datetime.strptime(raw, "%Y-%m-%d")
        now = datetime.datetime.utcnow()
        return as_of.year != now.year or as_of.month != now.month
    except Exception:
        return True


def _refresh_pricing_cache(session: C0d3rSession, header: "HeaderRenderer", model_id: str) -> None:
    from services.bedrock_pricing import lookup_pricing, ModelPricing, cache_pricing, refresh_pricing_from_aws

    if not model_id or model_id in _PRICING_REFRESHED:
        return
    _PRICING_REFRESHED.add(model_id)
    pricing = lookup_pricing(model_id)
    if pricing and not _is_pricing_stale(pricing):
        return
    _emit_live("pricing: fetch from AWS pricing page")
    try:
        aws_pricing = refresh_pricing_from_aws(model_id)
        if aws_pricing:
            _emit_live(
                f"pricing: AWS pricing parsed in={aws_pricing.input_per_1k} out={aws_pricing.output_per_1k} as_of={aws_pricing.as_of}"
            )
            header.update()
            return
    except Exception:
        pass
    _emit_live("pricing: research starting")
    query = (
        f"Find the latest published pricing (current month if available) for AWS Bedrock model {model_id}. "
        "Return authoritative source URL."
    )
    research = ""
    stop = threading.Event()
    def _heartbeat():
        tick = 0
        while not stop.is_set():
            _emit_live(f"pricing: waiting on research... ({tick * 5:.0f}s)")
            stop.wait(5.0)
            tick += 1
    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()
    def _do_research():
        nonlocal research
        try:
            _emit_live(f"pricing: search query => {query}")
            research = session._c0d3r._research(query)
        except Exception:
            research = ""
        finally:
            stop.set()
    th = threading.Thread(target=_do_research, daemon=True)
    th.start()
    th.join(timeout=8.0)
    if th.is_alive():
        _emit_live("pricing: slow research; continuing with cached rates and updating async")
        # Let it finish in background; return without blocking header.
        return
    if research:
        preview = research.strip().replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300] + "..."
        _emit_live(f"pricing: research preview => {preview}")
    system = (
        "Return ONLY JSON with keys: input_per_1k (number), output_per_1k (number), "
        "source_url (string), as_of (YYYY-MM or YYYY-MM-DD). "
        "Use the latest month published. If unknown, return null values."
    )
    try:
        response = session.send(prompt=f"Research:\n{research}\n\nModel: {model_id}", stream=False, system=system)
        payload = _safe_json(response)
    except Exception:
        payload = {}
    try:
        inp = float(payload.get("input_per_1k"))
        outp = float(payload.get("output_per_1k"))
        source = str(payload.get("source_url") or "")
        as_of = str(payload.get("as_of") or "")
        if inp > 0 and outp > 0 and source and as_of:
            cache_pricing(model_id, ModelPricing(inp, outp, source, as_of))
            _emit_live("pricing: cache updated")
            header.update()
        else:
            _emit_live("pricing: research incomplete; keeping existing rates")
    except Exception:
        _emit_live("pricing: research parse failed")


def _prompt_budget_choice(budget_usd: float) -> str:
    print(f"\nBudget reached (${budget_usd:.2f}). Choose:")
    print("1) Continue with current budget")
    print("2) Set a new budget")
    print("3) Indefinite (no budget)")
    print("4) Exit")
    choice = input("> ").strip().lower()
    if choice in {"1", "continue", "c"}:
        return "continue"
    if choice in {"2", "set", "s"}:
        return "set"
    if choice in {"3", "indefinite", "i"}:
        return "indefinite"
    return "exit"


def _prompt_budget_value(current: float) -> float:
    try:
        val = input(f"New budget USD (current {current:.2f}): ").strip()
        if not val:
            return current
        return float(val)
    except Exception:
        return current


def _decide_mode(session: C0d3rSession, prompt: str, *, default_scientific: bool, default_tool_loop: bool) -> str:
    """
    Ask a routing model to choose the best execution mode.
    """
    if os.getenv("C0D3R_FORCE_DIRECT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return "direct"
    router_prompt = (
        "Return ONLY JSON: {\"mode\": \"tool_loop|direct|scientific\"}.\n"
        "Choose scientific for repo inspection/summaries with evidence bundles. "
        "Choose tool_loop for tasks requiring local commands (create/run/install/build/serve). "
        "Choose direct for pure discussion.\n"
        f"User request: {prompt}"
    )
    try:
        response = session.send(router_prompt, stream=False)
        payload = _safe_json(response)
        mode = str(payload.get("mode") or "").strip().lower()
        if mode in {"tool_loop", "direct", "scientific"}:
            return mode
    except Exception:
        pass
    if default_scientific and "inspect" in prompt.lower():
        return "scientific"
    if default_tool_loop:
        return "tool_loop"
    return "direct"


def _pre_research(session: C0d3rSession, prompt: str) -> str:
    research_prompt = (
        "Perform preliminary research for the task. Summarize the most relevant framework/library setup steps, "
        "best practices, and any constraints. Return 6-10 bullets.\n\n"
        f"Task: {prompt}"
    )
    try:
        _emit_live("pre_research: model call starting")
        stop = threading.Event()
        def _heartbeat():
            tick = 0
            while not stop.is_set():
                _emit_live(f"pre_research: waiting... ({tick * 5:.0f}s)")
                stop.wait(5.0)
                tick += 1
        t = threading.Thread(target=_heartbeat, daemon=True)
        t.start()
        response = _call_with_timeout(
            session._safe_send,
            timeout_s=_model_timeout_s(),
            kwargs={"prompt": research_prompt, "stream": False, "research_override": True},
        )
        stop.set()
        if response:
            _append_bibliography_from_text(response)
        return response or ""
    except Exception:
        try:
            stop.set()
        except Exception:
            pass
        return ""


def _model_timeout_s() -> float | None:
    # Default to no timeouts unless explicitly enabled.
    no_timeouts = os.getenv("C0D3R_NO_TIMEOUTS", "1").strip().lower() not in {"0", "false", "no", "off"}
    if no_timeouts:
        return None
    try:
        raw = os.getenv("C0D3R_MODEL_TIMEOUT_S", "-1") or "-1"
        val = float(raw)
        if val <= 0:
            return None
        return val
    except Exception:
        return 60.0


def _model_timeout_value(default: float = 60.0) -> float:
    val = _model_timeout_s()
    return default if val is None else val


def _call_with_timeout(fn, *, timeout_s: float | None, kwargs: dict) -> str | None:
    if timeout_s is None:
        try:
            return fn(**kwargs)
        except Exception as exc:
            _diag_log(f"model error: {exc}")
            return None
    result: dict = {}
    error: dict = {}

    def _runner():
        try:
            result["value"] = fn(**kwargs)
        except Exception as exc:
            error["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        _diag_log(f"model timeout after {timeout_s}s")
        return None
    if error:
        _diag_log(f"model error: {error.get('error')}")
        return None
    return result.get("value")


def _execute_meta_command(cmd: str, workdir: Path) -> Tuple[int, str, str, Path | None]:
    parts = cmd.split(maxsplit=2)
    if len(parts) < 2:
        return 1, "", "invalid meta command", None
    action = parts[0][2:]
    if action == "sleep":
        try:
            seconds = float(parts[1])
        except Exception:
            return 1, "", "invalid sleep duration", None
        time.sleep(seconds)
        return 0, f"slept {seconds}s", "", None
    if action == "wait_http":
        if len(parts) < 3:
            return 1, "", "usage: ::wait_http <url> <seconds>", None
        url = parts[1]
        try:
            timeout = float(parts[2])
        except Exception:
            return 1, "", "invalid timeout", None
        start = time.time()
        while time.time() - start < timeout:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return 0, f"HTTP 200 {url}", "", None
            except Exception:
                time.sleep(0.5)
        return 1, "", f"timeout waiting for {url}", None
    if action == "bg":
        if len(parts) < 2:
            return 1, "", "usage: ::bg <command>", None
        command = parts[1] if len(parts) == 2 else parts[1] + " " + parts[2]
        try:
            log_dir = Path("runtime/c0d3r/bg_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"bg_{ts}.log"
            log_fh = log_path.open("ab")
            proc = subprocess.Popen(command, cwd=str(workdir), shell=True, stdout=log_fh, stderr=log_fh)
            return 0, f"started pid {proc.pid} (log {log_path})", "", None
        except Exception as exc:
            return 1, "", str(exc), None
    if action == "kill":
        if len(parts) < 2:
            return 1, "", "usage: ::kill <pid>", None
        try:
            pid = int(parts[1])
        except Exception:
            return 1, "", "invalid pid", None
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
            else:
                os.kill(pid, 9)
            return 0, f"killed {pid}", "", None
        except Exception as exc:
            return 1, "", str(exc), None
    if action == "cd":
        if len(parts) < 2:
            return 1, "", "usage: ::cd <path>", None
        new_dir = parts[1] if len(parts) == 2 else parts[1] + " " + parts[2]
        new_path = (workdir / new_dir).resolve() if not Path(new_dir).is_absolute() else Path(new_dir).resolve()
        if new_dir.strip().rstrip("\\/") == workdir.name:
            # prevent accidental recursive nesting
            return 0, f"cwd unchanged (already in {workdir})", "", workdir
        if not new_path.exists():
            return 1, "", f"path not found: {new_path}", None
        if os.getenv("C0D3R_LOCK_CWD", "1").strip().lower() not in {"0", "false", "no", "off"}:
            root = Path(os.getenv("C0D3R_ROOT_CWD", str(workdir))).resolve()
            try:
                new_path.relative_to(root)
            except Exception:
                return 1, "", f"cd blocked outside project root: {root}", None
        return 0, f"cwd -> {new_path}", "", new_path
    return 1, "", f"unknown meta command {action}", None


def _requires_commands_for_task(prompt: str) -> bool:
    lower = (prompt or "").lower()
    keywords = ("create", "start", "run", "serve", "install", "build", "generate")
    return any(k in lower for k in keywords)


def _requires_new_projects_dir(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "c:/users/adam/projects" in lower and "create" in lower and "directory" in lower


def _requires_skeleton(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "skeleton" in lower or "initialize" in lower


def _find_skeleton_root(workdir: Path) -> str | None:
    try:
        candidates = [workdir] + [p for p in workdir.iterdir() if p.is_dir()]
        for root in candidates:
            if (root / "README.md").exists() and (root / "requirements.txt").exists() and (root / "src").exists():
                return str(root)
    except Exception:
        return None
    return None


def _requires_full_completion(prompt: str) -> bool:
    lower = (prompt or "").lower()
    keywords = (
        "overhaul",
        "finish",
        "fully",
        "complete",
        "end-to-end",
        "production-ready",
        "test and fix",
        "until you get no errors",
        "until there are no errors",
    )
    return any(k in lower for k in keywords)


def _plan_and_checklist_present() -> bool:
    base = Path("runtime/c0d3r")
    return (base / "plan.md").exists() and (base / "checklist.md").exists()


def _checklist_has_mapping() -> bool:
    path = Path("runtime/c0d3r/checklist.md")
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return "source->decision->code" in text


def _checklist_is_complete() -> bool:
    path = Path("runtime/c0d3r/checklist.md")
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text.strip()) < 50:
        return False
    if text.count("- [") < 3:
        return False
    return "[ ]" not in text


def _plan_is_substantial() -> bool:
    path = Path("runtime/c0d3r/plan.md")
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return len(text.strip()) >= 50


def _commands_only_runtime(commands: List[str]) -> bool:
    if not commands:
        return False
    for cmd in commands:
        lower = (cmd or "").lower()
        if "runtime\\c0d3r" in lower or "runtime/c0d3r" in lower:
            continue
        if "plan.md" in lower or "checklist.md" in lower:
            continue
        return False
    return True


def _unbounded_math_ready() -> bool:
    """
    Ensure unbounded resolver produced equations + research links and they are persisted.
    """
    try:
        payload_path = Path("runtime/c0d3r/unbounded_payload.json")
        if not payload_path.exists():
            return False
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        equations = payload.get("equations") or []
        links = payload.get("research_links") or []
        if not equations or not links:
            return False
        return True
    except Exception:
        return False


def _equation_graph_ready() -> bool:
    try:
        from core.models import Equation

        return Equation.objects.exists()
    except Exception:
        return False


def _prompt_allows_new_dirs(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "new directory" in lower or "new folder" in lower or "create a new directory" in lower


def _requires_benchmark(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "outperform" in lower or "outperforms" in lower or "novel" in lower


def _requires_rigorous_constraints(prompt: str) -> bool:
    """
    Detect requests that demand rigorous mathematical/engineering constraints
    and should trigger heavy research + verification.
    """
    lower = (prompt or "").lower()
    triggers = [
        "novel",
        "outperform",
        "outperforms",
        "algorithm",
        "theorem",
        "proof",
        "unsat",
        "sat ",
        "satisfiability",
        "optimal",
        "optimality",
        "correctness",
    ]
    return any(t in lower for t in triggers)


def _benchmark_evidence_present() -> bool:
    path = Path("runtime/c0d3r/benchmarks.json")
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _diag_log(f"benchmark parse failed: {exc}")
        return False
    if not isinstance(payload, dict):
        return False
    # Require explicit metadata from the benchmark script
    if "generated_by" not in payload:
        return False
    if not ("baseline" in payload and "candidate" in payload):
        return False
    log_path = Path("runtime/c0d3r/evidence.log")
    if not log_path.exists():
        return False
    log_text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    if "benchmark" in log_text and "python" in log_text and "exit 0" in log_text:
        return True
    return False


def _tests_passed_recently() -> bool:
    log_path = Path("runtime/c0d3r/evidence.log")
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    # Find last pytest invocation
    lines = text.splitlines()
    last_pytest_idx = None
    for idx, line in enumerate(lines):
        if "$ " in line and "pytest" in line:
            last_pytest_idx = idx
    if last_pytest_idx is None:
        return False
    for line in lines[last_pytest_idx:last_pytest_idx + 4]:
        if "(exit 0)" in line:
            return True
    return False


def _mkdir_targets_root(cmd: str, workdir: Path) -> bool:
    try:
        lower = cmd.lower()
        targets: List[str] = []
        if lower.startswith("mkdir "):
            targets = [t for t in cmd.split()[1:] if not t.startswith("-")]
        elif "new-item -itemtype directory" in lower:
            import re
            matches = re.findall(r'New-Item -ItemType Directory -Force "([^"]+)"', cmd, re.I)
            targets = matches or []
        for target in targets:
            if not target:
                continue
            target_path = (workdir / target) if not Path(target).is_absolute() else Path(target)
            try:
                rel = target_path.resolve().relative_to(workdir.resolve())
            except Exception:
                continue
            if len(rel.parts) == 1 and not target_path.exists():
                return True
    except Exception:
        return False
    return False


def _attempt_auto_fix(cmd: str, stderr: str, workdir: Path, run_command, log_path: Path) -> Tuple[bool, int, str, str]:
    lower = (stderr or "").lower()
    if "nameerror" in lower and "name 'os' is not defined" in lower and "python -c" in cmd.lower():
        fixed = cmd.replace("python -c \"", "python -c \"import os; ", 1)
        code, stdout, err = run_command(fixed, cwd=workdir, timeout_s=_command_timeout_s(fixed))
        _append_tool_log(log_path, fixed, code, stdout, err)
        return True, code, stdout, err
    if "importerror" in lower or "module not found" in lower or "_pytest" in lower:
        path = _extract_path_from_error(stderr)
        if path:
            try:
                target = Path(path)
                if not target.is_absolute():
                    target = (workdir / target).resolve()
                try:
                    target.relative_to(workdir.resolve())
                except Exception:
                    target = None
                if target:
                    pkg_dir = target.parent
                    for _ in range(2):
                        init_path = pkg_dir / "__init__.py"
                        if not init_path.exists():
                            init_path.write_text("", encoding="utf-8")
                        pkg_dir = pkg_dir.parent
                    return True, 0, f"added __init__.py near {target}", ""
            except Exception:
                pass
        # Fallback: ensure tests and core packages are importable.
        try:
            tests_dir = workdir / "tests"
            if tests_dir.exists():
                (tests_dir / "__init__.py").write_text("", encoding="utf-8")
                for sub in tests_dir.iterdir():
                    if sub.is_dir():
                        init = sub / "__init__.py"
                        if not init.exists():
                            init.write_text("", encoding="utf-8")
            core_dir = workdir / "core"
            if core_dir.exists():
                (core_dir / "__init__.py").write_text("", encoding="utf-8")
                for sub in core_dir.iterdir():
                    if sub.is_dir():
                        init = sub / "__init__.py"
                        if not init.exists():
                            init.write_text("", encoding="utf-8")
            return True, 0, "added __init__.py to tests packages", ""
        except Exception:
            pass
    return False, 0, "", ""


def _extract_path_from_error(stderr: str) -> str | None:
    try:
        for line in stderr.splitlines():
            if line.strip().startswith("ImportError while importing test module"):
                continue
            if line.strip().startswith("Hint:") or "Hint:" in line:
                continue
            if ".py" in line and "\\" in line:
                frag = line.strip().strip("'\"")
                if Path(frag).exists():
                    return frag
            if ".py" in line and ":" in line:
                parts = line.split(":")
                for part in parts:
                    if part.strip().endswith(".py") and Path(part.strip()).exists():
                        return part.strip()
    except Exception:
        return None
    return None


def _is_runtime_path(path: Path | None) -> bool:
    if path is None:
        return False
    rel = str(path).replace("\\", "/")
    return "/runtime/c0d3r/" in rel


def _is_runtime_command(cmd: str) -> bool:
    lower = (cmd or "").lower()
    return "runtime\\c0d3r" in lower or "runtime/c0d3r" in lower


def _is_type_nul(cmd: str) -> bool:
    lower = (cmd or "").lower().strip()
    return lower.startswith("type nul") or "type nul >" in lower


def _ps_prefix(cmd: str) -> str:
    if os.name != "nt":
        return cmd
    stripped = cmd.lstrip()
    if stripped.startswith("& "):
        return cmd
    if stripped.startswith("\""):
        return f"& {cmd}"
    return cmd


def _is_benchmark_echo(cmd: str) -> bool:
    lower = (cmd or "").lower()
    return ("benchmarks.json" in lower) and ("echo " in lower or "set-content" in lower or "add-content" in lower)


def _disallow_placeholder_code(workdir: Path) -> bool:
    try:
        for path in workdir.rglob("*.py"):
            if "runtime" in str(path).replace("\\", "/"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            lower = text.lower()
            if "todo" in lower:
                return True
            if "return true  # simplified" in lower or "return true # simplified" in lower:
                return True
            if "return false  # simplified" in lower or "return false # simplified" in lower:
                return True
            if "pass  # stub" in lower or "pass  # todo" in lower:
                return True
        return False
    except Exception:
        return False


def _has_empty_tests(workdir: Path) -> bool:
    try:
        tests_dir = workdir / "tests"
        if not tests_dir.exists():
            return True
        for path in tests_dir.rglob("test*.py"):
            if path.stat().st_size == 0:
                return True
        return False
    except Exception:
        return True


def _unbounded_trigger(no_progress: int, test_failures: int, model_timeouts: int) -> bool:
    return (no_progress >= 2) or (test_failures >= 3) or (model_timeouts >= 2)


def _resolve_unbounded_request(session: C0d3rSession, prompt: str) -> dict | None:
    system = (
        "Return ONLY JSON with keys: branches (list), matrix (list of rows), "
        "integrated_mechanics (list), equations (list), gap_fill_steps (list), "
        "research_links (list), anomalies (list), hypotheses (list), "
        "experiments (list), decision_criteria (list), candidate_nodes (list), "
        "selected_node (object), bounded_task (string), constraints (list), next_steps (list). "
        "branches must name 4 disciplines + critical thinking psychology + neuroscience of engineering. "
        "Treat the problem as a mathematical traversal of a multi-disciplinary knowledge matrix. "
        "First integrate mechanics across disciplines (integrated_mechanics), then express them as "
        "explicit equations (equations). Fill gaps between equations by proposing bridging equations "
        "and tests (gap_fill_steps). Only then list anomalies/paradoxes nearest to those mechanics, "
        "and derive hypotheses that reconcile paradoxes with proven mechanics. "
        "Use experiments to refine hypotheses. Build candidate_nodes as objects with fields: "
        "{id, rationale, supporting_mechanics, tests, score_0_1}. "
        "selected_node must be the best-scoring candidate and explain why it matches the request. "
        "matrix rows should be compact (discipline, axis_x, axis_y, axis_z, axis_w, insight). "
        "bounded_task must be a concrete, testable task with explicit completion criteria."
    )
    try:
        response = session.send(prompt=f"Unbounded request:\n{prompt}", stream=False, system=system)
        payload = _safe_json(response)
        if not isinstance(payload, dict):
            return None
        if not payload.get("selected_node") and payload.get("candidate_nodes"):
            try:
                nodes = payload.get("candidate_nodes") or []
                best = max(nodes, key=lambda n: float(n.get("score_0_1") or 0))
                payload["selected_node"] = best
            except Exception:
                pass
        # If no hypotheses, prompt once more for next-step refinement.
        if not (payload.get("hypotheses") or payload.get("bounded_task")):
            followup = session.send(
                prompt=(
                    f"Unbounded request:\n{prompt}\n\n"
                    "You returned no hypotheses. Provide integrated_mechanics, anomalies, "
                    "equations, gap_fill_steps, candidate_nodes, selected_node, and a bounded_task with explicit completion criteria."
                ),
                stream=False,
                system=system,
            )
            follow_payload = _safe_json(followup)
            if isinstance(follow_payload, dict):
                payload = follow_payload
        _persist_unbounded_payload(payload)
        return payload
    except Exception:
        return None


def _append_unbounded_matrix(payload: dict) -> None:
    try:
        out_dir = Path("runtime/c0d3r")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "unbounded_matrix.md"
        history_path = out_dir / "unbounded_matrix_history.jsonl"
        lines = ["# Unbounded Request Matrix", ""]
        branches = payload.get("branches") or []
        if branches:
            lines.append("## Branches")
            for b in branches:
                lines.append(f"- {b}")
            lines.append("")
        matrix = payload.get("matrix") or []
        if matrix:
            lines.append("## Matrix")
            for row in matrix:
                if isinstance(row, dict):
                    lines.append(f"- {row}")
                else:
                    lines.append(f"- {row}")
            lines.append("")
        mechanics = payload.get("integrated_mechanics") or []
        if mechanics:
            lines.append("## Integrated Mechanics")
            for item in mechanics:
                lines.append(f"- {item}")
            lines.append("")
        equations = payload.get("equations") or []
        if equations:
            lines.append("## Equations")
            for item in equations:
                lines.append(f"- {item}")
            lines.append("")
        gap_steps = payload.get("gap_fill_steps") or []
        if gap_steps:
            lines.append("## Gap Fill Steps")
            for item in gap_steps:
                lines.append(f"- {item}")
            lines.append("")
        research_links = payload.get("research_links") or []
        if research_links:
            lines.append("## Research Links")
            for item in research_links:
                lines.append(f"- {item}")
            lines.append("")
        anomalies = payload.get("anomalies") or []
        if anomalies:
            lines.append("## Anomalies/Paradoxes")
            for a in anomalies:
                lines.append(f"- {a}")
            lines.append("")
        hypotheses = payload.get("hypotheses") or []
        if hypotheses:
            lines.append("## Hypotheses")
            for h in hypotheses:
                lines.append(f"- {h}")
            lines.append("")
        candidate_nodes = payload.get("candidate_nodes") or []
        if candidate_nodes:
            lines.append("## Candidate Nodes")
            for node in candidate_nodes:
                lines.append(f"- {node}")
            lines.append("")
        selected = payload.get("selected_node")
        if selected:
            lines.append("## Selected Node")
            lines.append(f"- {selected}")
            lines.append("")
        experiments = payload.get("experiments") or []
        if experiments:
            lines.append("## Experiments / Thought Tests")
            for exp in experiments:
                lines.append(f"- {exp}")
            lines.append("")
        criteria = payload.get("decision_criteria") or []
        if criteria:
            lines.append("## Decision Criteria")
            for item in criteria:
                lines.append(f"- {item}")
            lines.append("")
        next_steps = payload.get("next_steps") or []
        if next_steps:
            lines.append("## Next Steps")
            for step in next_steps:
                lines.append(f"- {step}")
            lines.append("")
        constraints = payload.get("constraints") or []
        if constraints:
            lines.append("## Constraints")
            for c in constraints:
                lines.append(f"- {c}")
            lines.append("")
        bounded = str(payload.get("bounded_task") or "").strip()
        if bounded:
            lines.append("## Bounded Task")
            lines.append(bounded)
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "branches": payload.get("branches") or [],
            "matrix": payload.get("matrix") or [],
            "integrated_mechanics": payload.get("integrated_mechanics") or [],
            "equations": payload.get("equations") or [],
            "gap_fill_steps": payload.get("gap_fill_steps") or [],
            "research_links": payload.get("research_links") or [],
            "anomalies": payload.get("anomalies") or [],
            "hypotheses": payload.get("hypotheses") or [],
            "experiments": payload.get("experiments") or [],
            "decision_criteria": payload.get("decision_criteria") or [],
            "bounded_task": bounded,
            "constraints": payload.get("constraints") or [],
            "payload": payload,
            "candidate_nodes": payload.get("candidate_nodes") or [],
            "selected_node": payload.get("selected_node") or {},
        }
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        _write_matrix_db(record)
    except Exception:
        pass


def _apply_unbounded_constraints(prompt: str, payload: dict) -> str:
    constraints = payload.get("constraints") or []
    bounded = str(payload.get("bounded_task") or "").strip()
    mechanics = payload.get("integrated_mechanics") or []
    equations = payload.get("equations") or []
    gap_steps = payload.get("gap_fill_steps") or []
    research_links = payload.get("research_links") or []
    hypotheses = payload.get("hypotheses") or []
    experiments = payload.get("experiments") or []
    block = []
    if bounded:
        block.append(f"Bounded task: {bounded}")
    if mechanics:
        block.append("Integrated mechanics:")
        for item in mechanics:
            block.append(f"- {item}")
    if equations:
        block.append("Equations:")
        for item in equations:
            block.append(f"- {item}")
    if gap_steps:
        block.append("Gap fill steps:")
        for item in gap_steps:
            block.append(f"- {item}")
    if research_links:
        block.append("Research links:")
        for item in research_links:
            block.append(f"- {item}")
    if hypotheses:
        block.append("Hypotheses:")
        for item in hypotheses:
            block.append(f"- {item}")
    if experiments:
        block.append("Experiments:")
        for item in experiments:
            block.append(f"- {item}")
    if constraints:
        block.append("Constraints:")
        for c in constraints:
            block.append(f"- {c}")
    if not block:
        return prompt
    return f"{prompt}\n\n[unbounded_resolution]\n" + "\n".join(block)


def _persist_unbounded_payload(payload: dict) -> None:
    try:
        out_dir = Path("runtime/c0d3r")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "unbounded_payload.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _capture_behavior_snapshot(
    log: list[dict],
    *,
    step: int,
    no_progress: int,
    test_failures: int,
    model_timeouts: int,
    note: str,
) -> None:
    try:
        log.append(
            {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "step": step,
                "no_progress": no_progress,
                "test_failures": test_failures,
                "model_timeouts": model_timeouts,
                "note": note,
            }
        )
        if len(log) % 3 == 0:
            path = Path("runtime/c0d3r/behavior_log.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                for item in log[-3:]:
                    fh.write(json.dumps(item) + "\n")
    except Exception:
        pass


def _apply_behavior_insights(payload: dict, behavior_log: list[dict]) -> None:
    """
    Use integrated mechanics + hypotheses to turn failures into fixes.
    This doesn't execute commands; it stores insights for the next model call.
    """
    try:
        path = Path("runtime/c0d3r/behavior_insights.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Behavior Insights", ""]
        mechanics = payload.get("integrated_mechanics") or []
        if mechanics:
            lines.append("## Integrated Mechanics")
            for item in mechanics:
                lines.append(f"- {item}")
            lines.append("")
        anomalies = payload.get("anomalies") or []
        if anomalies:
            lines.append("## Anomalies")
            for item in anomalies:
                lines.append(f"- {item}")
            lines.append("")
        hypotheses = payload.get("hypotheses") or []
        if hypotheses:
            lines.append("## Hypotheses")
            for item in hypotheses:
                lines.append(f"- {item}")
            lines.append("")
        equations = payload.get("equations") or []
        if equations:
            lines.append("## Equations")
            for item in equations:
                lines.append(f"- {item}")
            lines.append("")
        gap_steps = payload.get("gap_fill_steps") or []
        if gap_steps:
            lines.append("## Gap Fill Steps")
            for item in gap_steps:
                lines.append(f"- {item}")
            lines.append("")
        lines.append("## Behavior Signals")
        for item in behavior_log[-5:]:
            lines.append(f"- {item}")
        path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


def _append_bibliography_from_text(text: str) -> None:
    try:
        urls = []
        for token in (text or "").split():
            if token.startswith("http://") or token.startswith("https://"):
                urls.append(token.strip().rstrip(").,;"))
        if not urls:
            return
        out = Path("runtime/c0d3r/bibliography.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        existing = out.read_text(encoding="utf-8", errors="ignore") if out.exists() else ""
        lines = []
        for url in urls:
            if url in existing:
                continue
            title = url.split("//", 1)[-1].split("/", 1)[0]
            entry = f"Unknown Author. (n.d.). {title}. {url}. (Accessed {time.strftime('%Y-%m-%d')})."
            lines.append(entry)
        if lines:
            with out.open("a", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(line + "\n")
    except Exception:
        pass


def _write_matrix_db(record: dict) -> None:
    """
    Best-effort write to Django DB if available.
    """
    try:
        import os as _os
        if not _os.getenv("DJANGO_SETTINGS_MODULE"):
            _os.environ["DJANGO_SETTINGS_MODULE"] = "coolcrypto_dashboard.settings"
        import django
        django.setup()
        from core.models import UnboundedMatrixRecord
        UnboundedMatrixRecord.objects.create(
            prompt=record.get("bounded_task") or "",
            branches=record.get("branches") or [],
            matrix=record.get("matrix") or [],
            integrated_mechanics=record.get("integrated_mechanics") or [],
            equations=record.get("equations") or [],
            gap_fill_steps=record.get("gap_fill_steps") or [],
            research_links=record.get("research_links") or [],
            anomalies=record.get("anomalies") or [],
            hypotheses=record.get("hypotheses") or [],
            experiments=record.get("experiments") or [],
            decision_criteria=record.get("decision_criteria") or [],
            bounded_task=record.get("bounded_task") or "",
            constraints=record.get("constraints") or [],
            payload=record,
        )
    except Exception:
        return


def _install_pytest(python_exec: str | None, project_root: Path, run_command, log_path: Path) -> None:
    try:
        pip_cmd = f"\"{python_exec}\" -m pip install pytest" if python_exec else "python -m pip install pytest"
        pip_cmd = _ps_prefix(pip_cmd)
        code, stdout, stderr = run_command(pip_cmd, cwd=project_root, timeout_s=_command_timeout_s(pip_cmd))
        _append_tool_log(log_path, pip_cmd, code, stdout, stderr)
    except Exception:
        pass


def _append_verification_snapshot(workdir: Path, run_command, history: list[str]) -> None:
    try:
        code, stdout, stderr = run_command("git status -sb", cwd=workdir)
        snippet = f"$ git status -sb\n(exit {code})\n{stdout.strip()}\n{stderr.strip()}".strip()
        history.append(snippet[:2000])
        code, stdout, stderr = run_command("Get-ChildItem -Name" if os.name == "nt" else "ls -1", cwd=workdir)
        snippet = f"$ list\n(exit {code})\n{stdout.strip()}\n{stderr.strip()}".strip()
        history.append(snippet[:2000])
    except Exception:
        pass


def _update_system_map(workdir: Path) -> None:
    try:
        root = workdir.resolve()
        files = []
        tests = []
        symbols = {"classes": {}, "functions": {}, "imports": {}}
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            rel = str(path.relative_to(root))
            files.append(rel)
            if rel.startswith("tests") or rel.endswith("_test.py") or rel.startswith("test_"):
                tests.append(rel)
            if path.suffix == ".py":
                try:
                    import ast
                    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
                    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
                    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
                    imps = []
                    for n in tree.body:
                        if isinstance(n, ast.Import):
                            imps.extend([a.name for a in n.names])
                        if isinstance(n, ast.ImportFrom):
                            if n.module:
                                imps.append(n.module)
                    symbols["classes"][rel] = classes
                    symbols["functions"][rel] = funcs
                    symbols["imports"][rel] = sorted(set(imps))
                except Exception:
                    pass
        payload = {
            "root": str(root),
            "files": sorted(files),
            "tests": sorted(tests),
            "symbols": symbols,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out = Path("runtime/c0d3r/system_map.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _append_evidence(cmd: str, code: int, stdout: str, stderr: str) -> None:
    try:
        out = Path("runtime/c0d3r/evidence.log")
        out.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with out.open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] $ {cmd}\n(exit {code})\n{stdout.strip()}\n{stderr.strip()}\n\n")
    except Exception:
        pass


def _append_research_notes(text: str) -> None:
    try:
        out = Path("runtime/c0d3r/research_notes.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with out.open("a", encoding="utf-8") as fh:
            fh.write(f"\n## {ts}\n{text}\n")
    except Exception:
        pass


def _gap_score(stderr: str) -> int:
    lower = (stderr or "").lower()
    score = 0
    triggers = ("no module named", "modulenotfounderror", "not found", "deprecated", "version", "unknown", "attributeerror")
    for t in triggers:
        if t in lower:
            score += 1
    return score


def _run_quality_checks(workdir: Path, run_command, usage_tracker, log_path: Path) -> bool:
    python_exec = _resolve_project_python(workdir) or "python"
    ok = True
    handler_profile = _detect_handler(workdir)
    # Lint
    lint_cmds = []
    if handler_profile:
        handler, profile = handler_profile
        lint_cmds = handler.lint(profile)
    if not lint_cmds:
        lint_cmds = [f"\"{python_exec}\" -m ruff ."]
    for lint_cmd in lint_cmds:
        code, stdout, stderr = run_command(lint_cmd, cwd=workdir, timeout_s=_command_timeout_s(lint_cmd))
        _append_tool_log(log_path, lint_cmd, code, stdout, stderr)
        if code != 0 and _auto_install_enabled():
            if "ruff" in lint_cmd:
                install_cmd = f"\"{python_exec}\" -m pip install ruff"
                run_command(install_cmd, cwd=workdir, timeout_s=_command_timeout_s(install_cmd))
                code, stdout, stderr = run_command(lint_cmd, cwd=workdir, timeout_s=_command_timeout_s(lint_cmd))
                _append_tool_log(log_path, lint_cmd, code, stdout, stderr)
        ok = ok and code == 0
    # Security
    sec_cmds = []
    if handler_profile:
        handler, profile = handler_profile
        sec_cmds = handler.security(profile)
    if not sec_cmds:
        sec_cmds = [f"\"{python_exec}\" -m bandit -r ."]
    for sec_cmd in sec_cmds:
        code, stdout, stderr = run_command(sec_cmd, cwd=workdir, timeout_s=_command_timeout_s(sec_cmd))
        _append_tool_log(log_path, sec_cmd, code, stdout, stderr)
        if code != 0 and _auto_install_enabled():
            if "bandit" in sec_cmd:
                install_cmd = f"\"{python_exec}\" -m pip install bandit"
                run_command(install_cmd, cwd=workdir, timeout_s=_command_timeout_s(install_cmd))
                code, stdout, stderr = run_command(sec_cmd, cwd=workdir, timeout_s=_command_timeout_s(sec_cmd))
                _append_tool_log(log_path, sec_cmd, code, stdout, stderr)
        ok = ok and code == 0
    # Dependency audit (if requirements exist)
    if (workdir / "requirements.txt").exists():
        audit_cmd = f"\"{python_exec}\" -m pip_audit -r requirements.txt"
        code, stdout, stderr = run_command(audit_cmd, cwd=workdir, timeout_s=_command_timeout_s(audit_cmd))
        _append_tool_log(log_path, audit_cmd, code, stdout, stderr)
        if code != 0 and _auto_install_enabled():
            install_cmd = f"\"{python_exec}\" -m pip install pip-audit"
            run_command(install_cmd, cwd=workdir, timeout_s=_command_timeout_s(install_cmd))
            code, stdout, stderr = run_command(audit_cmd, cwd=workdir, timeout_s=_command_timeout_s(audit_cmd))
            _append_tool_log(log_path, audit_cmd, code, stdout, stderr)
        ok = ok and code == 0
    return ok


def _spec_validator() -> bool:
    checklist = Path("runtime/c0d3r/checklist.md")
    if not checklist.exists():
        return False
    text = checklist.read_text(encoding="utf-8", errors="ignore")
    if "- [ ]" in text:
        return False
    if "Verified" not in text and "verified" not in text:
        return False
    if "source->decision->code" not in text.lower():
        return False
    return True


def _is_pip_install(cmd: str) -> bool:
    lower = cmd.lower()
    return "pip install" in lower or "python -m pip install" in lower


def _requires_pip_install(prompt: str) -> bool:
    lower = (prompt or "").lower()
    keywords = ("install", "dependencies", "pip", "run", "serve", "start", "execute", "setup env", "virtualenv")
    return any(k in lower for k in keywords)


def _requires_tests(prompt: str) -> bool:
    lower = (prompt or "").lower()
    keywords = ("create", "setup", "initialize", "build", "generate", "scaffold", "project")
    return any(k in lower for k in keywords)


def _run_tests_for_project(
    project_root: Path,
    run_command,
    usage_tracker,
    log_path: Path,
    *,
    target: Path | None = None,
) -> Tuple[bool, bool]:
    tests_ran = False
    tests_ok = False
    usage_tracker.set_status("testing", f"tests in {project_root}")
    handler_profile = _detect_handler(project_root)
    python_exec = _resolve_project_python(project_root)
    _ensure_foo_data(project_root)
    if target:
        _ensure_test_stub(project_root, target)
    # Python tests
    if handler_profile and handler_profile[1].language == "python":
        handler, profile = handler_profile
        for cmd in handler.tests_for_path(profile, target):
            tests_ran = True
            code, stdout, stderr = run_command(cmd, cwd=project_root, timeout_s=_command_timeout_s(cmd))
            _append_tool_log(log_path, cmd, code, stdout, stderr)
            if code == 0:
                tests_ok = True
            else:
                return tests_ran, False
        if handler.full_tests(profile):
            for cmd in handler.full_tests(profile):
                tests_ran = True
                code, stdout, stderr = run_command(cmd, cwd=project_root, timeout_s=_command_timeout_s(cmd))
                _append_tool_log(log_path, cmd, code, stdout, stderr)
                if code == 0:
                    tests_ok = True
                else:
                    return tests_ran, False
        return tests_ran, tests_ok
    if (project_root / "pytest.ini").exists() or (project_root / "tests").exists():
        tests_ran = True
        cmd = f"\"{python_exec}\" -m pytest" if python_exec else "python -m pytest"
        if target:
            rel = None
            try:
                rel = target.resolve().relative_to(project_root.resolve())
            except Exception:
                rel = None
            if rel:
                # Try direct test file match or module path.
                test_candidate = project_root / "tests" / (rel.stem if rel.suffix else rel.name)
                if rel.suffix:
                    test_candidate = project_root / "tests" / rel.name
                if test_candidate.exists():
                    cmd = f"{cmd} \"{test_candidate}\""
                else:
                    cmd = f"{cmd} -k \"{rel.stem}\""
                # Avoid repo-wide conftest interference.
                confcut = test_candidate.parent if test_candidate.exists() else (project_root / "tests")
                cmd = f"{cmd} --confcutdir=\"{confcut}\""
        cmd = _ps_prefix(cmd)
        code, stdout, stderr = run_command(cmd, cwd=project_root, timeout_s=_command_timeout_s(cmd))
        _append_tool_log(log_path, cmd, code, stdout, stderr)
        if code == 0:
            tests_ok = True
        else:
            if "no module named pytest" in (stdout + "\n" + stderr).lower():
                _install_pytest(python_exec, project_root, run_command, log_path)
                code, stdout, stderr = run_command(cmd, cwd=project_root, timeout_s=_command_timeout_s(cmd))
                _append_tool_log(log_path, cmd, code, stdout, stderr)
                if code == 0:
                    return tests_ran, True
            # Retry with confcutdir if Django fixtures break local tests.
            text = (stdout + "\n" + stderr).lower()
            if "django_settings_module" in text or "corsheaders" in text:
                confcut = project_root / "tests"
                if target:
                    try:
                        confcut = target.resolve().parent
                    except Exception:
                        confcut = project_root / "tests"
                retry = _ps_prefix(f"{cmd} --confcutdir=\"{confcut}\"")
                code, stdout, stderr = run_command(retry, cwd=project_root, timeout_s=_command_timeout_s(retry))
                _append_tool_log(log_path, retry, code, stdout, stderr)
                if code == 0:
                    return tests_ran, True
            key = str(project_root.resolve())
            if _auto_install_enabled() and (project_root / "requirements.txt").exists() and key not in _INSTALL_ATTEMPTS:
                _INSTALL_ATTEMPTS.add(key)
                _emit_live("tests failed; attempting pip install -r requirements.txt (one-time)")
                pip_cmd = f"\"{python_exec}\" -m pip install -r requirements.txt" if python_exec else "python -m pip install -r requirements.txt"
                run_command(pip_cmd, cwd=project_root, timeout_s=_command_timeout_s(pip_cmd))
            return tests_ran, False
    # Node/JS tests
    pkg = project_root / "package.json"
    if pkg.exists():
        try:
            import json as _json
            payload = _json.loads(pkg.read_text(encoding="utf-8"))
            scripts = payload.get("scripts") or {}
            dev_deps = payload.get("devDependencies") or {}
            deps = payload.get("dependencies") or {}
            has_vitest = "vitest" in dev_deps or "vitest" in deps
            has_jest = "jest" in dev_deps or "jest" in deps
            if target and str(target).lower().endswith((".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte")):
                if has_vitest:
                    tests_ran = True
                    cmd = f"npx vitest run \"{target}\""
                    code, stdout, stderr = run_command(cmd, cwd=project_root, timeout_s=_command_timeout_s(cmd))
                    _append_tool_log(log_path, cmd, code, stdout, stderr)
                    if code == 0:
                        tests_ok = True
                    else:
                        return tests_ran, False
                elif has_jest:
                    tests_ran = True
                    cmd = f"npx jest \"{target}\""
                    code, stdout, stderr = run_command(cmd, cwd=project_root, timeout_s=_command_timeout_s(cmd))
                    _append_tool_log(log_path, cmd, code, stdout, stderr)
                    if code == 0:
                        tests_ok = True
                    else:
                        return tests_ran, False
            if "test" in scripts:
                tests_ran = True
                cmd = "npm test -- --watch=false"
                code, stdout, stderr = run_command(cmd, cwd=project_root, timeout_s=_command_timeout_s(cmd))
                _append_tool_log(log_path, cmd, code, stdout, stderr)
                if code == 0:
                    tests_ok = True
                else:
                    return tests_ran, False
        except Exception:
            pass
    return tests_ran, tests_ok


def _resolve_project_python(project_root: Path) -> str | None:
    # Prefer local virtual envs
    for candidate in (project_root / ".venv" / "Scripts" / "python.exe", project_root / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    # Unix-style
    for candidate in (project_root / ".venv" / "bin" / "python", project_root / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return None


def _detect_handler(project_root: Path):
    try:
        from services.handlers.registry import detect_profile
        return detect_profile(project_root)
    except Exception:
        return None


def _auto_install_enabled() -> bool:
    return os.getenv("C0D3R_AUTO_INSTALL", "1").strip().lower() not in {"0", "false", "no", "off"}


def _is_write_command(cmd: str) -> bool:
    lower = (cmd or "").lower()
    write_tokens = (
        "set-content",
        "add-content",
        "echo ",
        "type nul",
        "new-item -itemtype file",
        "touch ",
        "python -c",
        "write_text",
        "write(",
    )
    return any(token in lower for token in write_tokens)


def _infer_written_path(cmd: str, workdir: Path) -> Path | None:
    try:
        lower = cmd.lower()
        if " -path " in lower:
            start = lower.find(" -path ")
            frag = cmd[start + len(" -path "):].strip()
            if frag.startswith('"') and '"' in frag[1:]:
                path = frag.split('"', 2)[1]
            else:
                path = frag.split()[0]
            return (workdir / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if "touch " in lower:
            path = cmd.split("touch", 1)[1].strip().split()[0]
            return (workdir / path).resolve()
        if "new-item -itemtype file" in lower:
            parts = cmd.split('"')
            if len(parts) >= 2:
                return (workdir / parts[1]).resolve()
        if "set-content" in lower or "add-content" in lower:
            if '-Path "' in cmd:
                path = cmd.split('-Path "', 1)[1].split('"', 1)[0]
                return (workdir / path).resolve()
        return None
    except Exception:
        return None


def _should_test_write(target: Path | None) -> bool:
    if target is None:
        return True
    try:
        rel = str(target).replace("\\", "/")
        if "/runtime/c0d3r/" in rel:
            return False
        if target.suffix.lower() in {".md", ".txt", ".log"}:
            return False
        return True
    except Exception:
        return True


def _ensure_foo_data(project_root: Path) -> None:
    try:
        path = project_root / "runtime" / "c0d3r" / "foo_data.json"
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        sample = {"foo": "bar", "count": 1, "items": ["alpha", "beta"]}
        path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    except Exception:
        pass


def _ensure_test_stub(project_root: Path, target: Path) -> None:
    if target.suffix != ".py":
        return
    tests_dir = project_root / "tests"
    if not tests_dir.exists():
        return
    name = target.stem
    test_path = tests_dir / f"test_{name}.py"
    if test_path.exists():
        return
    rel = None
    try:
        rel = target.resolve().relative_to(project_root.resolve())
    except Exception:
        return
    module = str(rel).replace("\\", ".").replace("/", ".")
    if module.endswith(".py"):
        module = module[:-3]
    content = (
        "import json\n"
        "from pathlib import Path\n\n"
        f"def test_{name}_foo_data():\n"
        "    data_path = Path('runtime/c0d3r/foo_data.json')\n"
        "    payload = json.loads(data_path.read_text())\n"
        f"    __import__('{module}')\n"
        "    assert payload['foo'] == 'bar'\n"
    )
    try:
        test_path.write_text(content, encoding="utf-8")
    except Exception:
        pass


def _fallback_inspection_commands(workdir: Path) -> List[str]:
    if os.name == "nt":
        return [
            "Get-ChildItem -Name",
            "Get-Content main.py -TotalCount 200",
            "Get-Content ui.kv -TotalCount 200",
            "Get-ChildItem modules -Name",
        ]
    return [
        "ls -1",
        "sed -n '1,200p' main.py",
        "sed -n '1,200p' ui.kv",
        "ls -1 modules",
    ]


def _command_timeout_s(cmd: str) -> int:
    try:
        base = int(os.getenv("C0D3R_CMD_TIMEOUT_S", "180") or "180")
    except Exception:
        base = 180
    lower = cmd.lower()
    if "pip install" in lower or "npm install" in lower:
        return max(base, 300)
    if "ionic serve" in lower:
        return max(base, 120)
    return base


def _snapshot_projects_dir(prompt: str) -> List[str]:
    if "c:/users/adam/projects" not in (prompt or "").lower():
        return []
    base = Path("C:/Users/Adam/Projects")
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def _diff_projects_dir(snapshot: List[str]) -> List[str]:
    base = Path("C:/Users/Adam/Projects")
    if not base.exists():
        return []
    current = sorted([p.name for p in base.iterdir() if p.is_dir()])
    return [name for name in current if name not in snapshot]


def _append_tool_log(path: Path, cmd: str, code: int, stdout: str, stderr: str) -> None:
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"$ {cmd}\n(exit {code})\n{stdout.strip()}\n{stderr.strip()}\n\n")
    except Exception:
        pass


def _execute_file_command(cmd: str, workdir: Path) -> Tuple[bool, int, str, str]:
    """
    Handle common file/dir commands locally to avoid PowerShell quoting pitfalls.
    """
    lower = cmd.strip().lower()
    # mkdir / New-Item
    if lower.startswith("new-item -itemtype directory") or lower.startswith("mkdir "):
        try:
            targets: List[str] = []
            if lower.startswith("mkdir "):
                targets = [t for t in cmd.split()[1:] if not t.startswith("-")]
            else:
                import re
                matches = re.findall(r'New-Item -ItemType Directory -Force "([^"]+)"', cmd, re.I)
                targets = matches or []
            created = []
            for target in targets:
                if not target:
                    continue
                target_path = (workdir / target) if not Path(target).is_absolute() else Path(target)
                if target_path.name == workdir.name:
                    continue
                if target_path.exists():
                    continue
                target_path.mkdir(parents=True, exist_ok=True)
                created.append(str(target_path))
            return True, 0, "\n".join(created), ""
        except Exception as exc:
            return True, 1, "", str(exc)
    # touch / New-Item file
    if lower.startswith("touch ") or "new-item -itemtype file" in lower:
        try:
            targets: List[str] = []
            if lower.startswith("touch "):
                targets = cmd.split()[1:]
            else:
                import re
                matches = re.findall(r'New-Item -ItemType File -Force "([^"]+)"', cmd, re.I)
                targets = matches or []
            created = []
            for target in targets:
                if not target:
                    continue
                target_path = (workdir / target) if not Path(target).is_absolute() else Path(target)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if not target_path.exists():
                    target_path.write_text("", encoding="utf-8")
                created.append(str(target_path))
            return True, 0, "\n".join(created), ""
        except Exception as exc:
            return True, 1, "", str(exc)
    # Set-Content / Add-Content
    if lower.startswith("set-content") or lower.startswith("add-content"):
        try:
            import re

            path_match = re.search(r'-Path\s+"([^"]+)"', cmd, re.I)
            value_match = re.search(r'-Value\s+(.+)$', cmd, re.I | re.S)
            if not path_match:
                return False, 0, "", ""
            path = path_match.group(1)
            value = value_match.group(1) if value_match else ""
            value = value.strip()
            # strip wrapping quotes
            for _ in range(2):
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
            value = value.replace("\\n", "\n")
            # Fix malformed path that contains content + redirection.
            if ">" in path:
                content_part, target_part = path.split(">", 1)
                content_part = content_part.strip()
                target_part = target_part.strip().strip('"').strip()
                if target_part:
                    target_part = target_part.split()[0]
                if content_part.startswith("=") and value and value not in content_part:
                    content_part = f"{value}{content_part}"
                path = target_part
                value = content_part
            target_path = (workdir / path) if not Path(path).is_absolute() else Path(path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if lower.startswith("add-content"):
                with target_path.open("a", encoding="utf-8") as fh:
                    fh.write(value)
            else:
                target_path.write_text(value, encoding="utf-8")
            return True, 0, str(target_path), ""
        except Exception as exc:
            return True, 1, "", str(exc)
    # echo ... > file (PowerShell-friendly)
    if lower.startswith("echo ") and ">" in cmd:
        try:
            import re
            match = re.match(r"^echo\s+([\s\S]+)\s>\s(.+)$", cmd, re.IGNORECASE)
            if not match:
                return False, 0, "", ""
            left = "echo " + match.group(1).strip()
            right = match.group(2).strip()
            append = right.startswith(">")
            if append:
                right = right[1:].strip()
            # unwrap target path
            target = right.strip().strip('"').strip("'")
            # extract content from echo
            if left.lower().startswith("echo "):
                content = left[5:].strip()
            else:
                content = left.strip()
            if (content.startswith("'") and content.endswith("'")) or (content.startswith('"') and content.endswith('"')):
                content = content[1:-1]
            content = content.replace("\\n", "\n")
            target_path = (workdir / target) if not Path(target).is_absolute() else Path(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if append:
                with target_path.open("a", encoding="utf-8") as fh:
                    fh.write(content)
            else:
                target_path.write_text(content, encoding="utf-8")
            return True, 0, str(target_path), ""
        except Exception as exc:
            return True, 1, "", str(exc)
    return False, 0, "", ""


class InterruptController:
    def __init__(self) -> None:
        self.interrupted = False
        self._q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._watch_input, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.2)

    def _watch_input(self) -> None:
        if os.name == "nt":
            self._watch_input_windows()
        else:
            self._watch_input_posix()

    def _watch_input_windows(self) -> None:
        try:
            import msvcrt
        except Exception:
            return
        while not self._stop.is_set():
            if msvcrt.kbhit():
                _ = msvcrt.getwch()
                self.interrupted = True
                return
            time.sleep(0.05)

    def _watch_input_posix(self) -> None:
        import select
        while not self._stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                _ = sys.stdin.readline()
                self.interrupted = True
                return


class UsageTracker:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.input_chars = 0
        self.output_chars = 0
        self.input_tokens = 0.0
        self.output_tokens = 0.0
        self.status = "idle"
        self.last_action = ""

    def add_input(self, text: str) -> None:
        self.input_chars += len(text or "")
        self.input_tokens = self.input_chars / 4.0

    def add_output(self, text: str) -> None:
        self.output_chars += len(text or "")
        self.output_tokens = self.output_chars / 4.0

    def set_status(self, status: str, action: str = "") -> None:
        self.status = status
        if action:
            self.last_action = action


class BudgetTracker:
    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self.enabled = True

    def add_spend(self, amount: float) -> None:
        if not self.enabled:
            return
        self.spent_usd += max(0.0, amount)

    def exceeded(self) -> bool:
        return self.enabled and self.spent_usd >= self.budget_usd

    def reset(self, new_budget: float | None = None) -> None:
        if new_budget is not None:
            self.budget_usd = new_budget
        self.spent_usd = 0.0

    def disable(self) -> None:
        self.enabled = False


class HeaderRenderer:
    def __init__(self, usage: UsageTracker) -> None:
        self.usage = usage
        self.enabled = os.getenv("C0D3R_HEADER", "1").strip().lower() not in {"0", "false", "no", "off"}
        self.ansi_ok = sys.stdout.isatty() and os.getenv("C0D3R_HEADER_ANSI", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self._last = ""
        self._frozen = False
        self.budget_usd = float(os.getenv("C0D3R_BUDGET_USD", "10.0") or "10.0")
        self.budget_enabled = os.getenv("C0D3R_BUDGET_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
        if self.budget_usd < 0:
            self.budget_enabled = False

    def freeze(self) -> None:
        self._frozen = True

    def resume(self) -> None:
        self._frozen = False

    def render(self) -> None:
        if not self.enabled:
            return
        header = self._build_header()
        if self.ansi_ok:
            sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(header)
        sys.stdout.flush()
        self._last = header

    def update(self) -> None:
        if not self.enabled or self._frozen:
            return
        header = self._build_header()
        if header == self._last:
            return
        if not self.ansi_ok:
            return
        # Save cursor, move to home, write header, restore cursor
        sys.stdout.write("\x1b7\x1b[H")
        sys.stdout.write(header)
        sys.stdout.write("\x1b8")
        sys.stdout.flush()
        self._last = header

    def _build_header(self) -> str:
        from services.bedrock_pricing import estimate_cost, lookup_pricing

        model = self.usage.model_id or "unknown"
        in_cost, out_cost = estimate_cost(model, self.usage.input_tokens, self.usage.output_tokens)
        pricing = lookup_pricing(model)
        if in_cost is None or out_cost is None:
            cost_line = f"Est. cost: N/A (no pricing for {model})"
        else:
            cost_line = f"Est. cost: ${in_cost + out_cost:.6f} (in ${in_cost:.6f} / out ${out_cost:.6f})"
        if pricing:
            rate_line = (
                f"Rates: ${pricing.input_per_1k:.4f}/1K in, ${pricing.output_per_1k:.4f}/1K out"
                f" | as of {pricing.as_of}"
            )
        else:
            rate_line = "Rates: unknown"
        token_line = f"Tokens est: in {self.usage.input_tokens:.0f} / out {self.usage.output_tokens:.0f}"
        status_line = f"Status: {self.usage.status}" + (f" | {self.usage.last_action}" if self.usage.last_action else "")
        budget_line = f"Budget: ${self.budget_usd:.2f}"
        header = (
            f"c0d3r session | model: {model}\n"
            f"{token_line} | {cost_line} | {budget_line}\n"
            f"{rate_line}\n"
            f"{status_line}\n"
            f"{'-' * 70}\n"
        )
        return header


if __name__ == "__main__":
    raise SystemExit(main())
