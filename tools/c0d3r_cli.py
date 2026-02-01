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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    # Silence noisy startup warnings for CLI usage.
    os.environ.setdefault("C0D3R_QUIET_STARTUP", "1")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    from services.env_loader import EnvLoader
    from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings
    from services.agent_workspace import run_command
    from services.system_probe import system_probe_context

    EnvLoader.load()

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

    workdir = Path(args.workdir or os.getcwd()).resolve()
    scientific = args.scientific or (os.getenv("C0D3R_SCIENTIFIC_MODE", "1").strip().lower() not in {"0", "false", "no", "off"})
    if args.no_scientific:
        scientific = False
    context_block = ""
    if not args.no_context:
        context_block = _build_context_block(workdir, run_command)
    tool_loop = args.tool_loop or (os.getenv("C0D3R_TOOL_LOOP", "1").strip().lower() not in {"0", "false", "no", "off"})
    if args.no_tools:
        tool_loop = False

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
    session = C0d3rSession(
        session_name="c0d3r-cli",
        transcript_dir=Path("runtime/c0d3r/transcripts"),
        workdir=str(workdir),
        **settings,
    )
    usage = UsageTracker(model_id=session.get_model_id())
    return _run_repl(
        session,
        usage,
        workdir,
        run_command,
        scientific=scientific,
        tool_loop=tool_loop,
        initial_prompt=prompt or None,
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
    history: List[str] = []
    log_path = Path("runtime/c0d3r/tool_loop.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base_snapshot = _snapshot_projects_dir(prompt)
    created_dirs: List[str] = []
    known_projects: List[Path] = []
    success = False
    tests_ran = False
    tests_ok = False
    require_tests = _requires_tests(prompt)
    for step in range(max_steps):
        usage_tracker.set_status("planning", f"step {step+1}/{max_steps}")
        tool_prompt = (
            "You can run local shell commands to inspect the repo and validate work. "
            "Return ONLY JSON with keys: commands (list of strings) and final (string or empty). "
            "If you are done, set final to your response and commands to []. "
            "Always create minimal unit tests and run them before declaring success. "
            "Do not finalize until tests pass.\n"
            "Meta commands available:\n"
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
            f"Request:\n{prompt}\n"
            + ("\n\nRecent outputs:\n" + "\n".join(history[-6:]) if history else "")
        )
        usage_tracker.add_input(tool_prompt)
        response = _call_with_timeout(
            session.send,
            timeout_s=_model_timeout_s(),
            kwargs={"prompt": tool_prompt, "stream": stream, "images": images, "stream_callback": stream_callback},
        )
        if response is None:
            history.append("note: model call timed out; retrying with shorter context")
            continue
        commands, final = _extract_commands(response)
        if final and not commands:
            if _requires_commands_for_task(prompt) and not success:
                history.append("note: commands required for this task; no final allowed without verified success")
                continue
            return final
        if not commands and not final:
            return response
        for cmd in commands[:8]:
            usage_tracker.set_status("executing", cmd)
            if cmd.lower().startswith("cd "):
                cmd = "::cd " + cmd[3:].strip()
            if ("mkdir" in cmd.lower() or "new-item" in cmd.lower()) and created_dirs:
                history.append("note: directory already created; do not create more directories")
                _append_tool_log(log_path, cmd, 1, "", "mkdir blocked; reuse existing directory")
                continue
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
                    if require_tests:
                        tests_ran, tests_ok = _run_tests_for_project(Path(skeleton), run_command, usage_tracker, log_path)
                        if not tests_ok:
                            history.append("error: tests failed; fix and re-run tests before finalizing")
                            success = False
                            continue
                    return f"Success: skeleton created at {skeleton}"
            if code != 0:
                history.append("error: previous command failed; analyze stderr and retry with correction")
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
    # Expand multi-arg mkdir into separate New-Item calls.
    if cmd.lower().startswith("mkdir "):
        parts = cmd.split()[1:]
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
        session.send,
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

        payload = json.loads(_extract_json(text))
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
) -> int:
    from services.conversation_memory import ConversationMemory

    memory = ConversationMemory(Path("runtime/c0d3r/conversation.jsonl"))
    summary_path = Path("runtime/c0d3r/summary.txt")
    summary = summary_path.read_text(encoding="utf-8", errors="ignore") if summary_path.exists() else ""
    pending = initial_prompt
    allow_interrupt = sys.stdin.isatty() and initial_prompt is None
    header = HeaderRenderer(usage)
    header.render()
    while True:
        try:
            header.freeze()
            prompt = pending if pending is not None else _read_input(workdir)
            header.resume()
            pending = None
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0
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

        context = memory.build_context(summary)
        try:
            from services.system_probe import system_probe_context
            probe_block = system_probe_context(workdir)
            context = f"{probe_block}\n{context}" if context else probe_block
        except Exception:
            pass
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
        usage.set_status("planning", "routing + research")
        mode = _decide_mode(session, prompt, default_scientific=scientific, default_tool_loop=tool_loop)
        research_summary = ""
        if mode in {"tool_loop", "scientific"}:
            usage.set_status("research", "gathering references")
            research_summary = _pre_research(session, prompt)
        controller = InterruptController()
        if allow_interrupt:
            controller.start()
        try:
            if mode == "scientific":
                usage.set_status("executing", "scientific analysis")
                response = _run_scientific_loop(
                    session, full_prompt, workdir, run_command, images=None, stream=False, stream_callback=None, usage_tracker=usage
                )
            elif mode == "tool_loop":
                usage.set_status("executing", "local commands")
                response = _run_tool_loop(
                    session, f"{full_prompt}\n\n[research]\n{research_summary}" if research_summary else full_prompt,
                    workdir, run_command, images=None, stream=False, stream_callback=None, usage_tracker=usage
                )
            else:
                usage.set_status("executing", "direct response")
                response = session.send(full_prompt, stream=False, system=system)
        finally:
            controller.stop()
        rendered = _render_json_response(response) or response
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
        memory.append(prompt, response)
        summary = memory.update_summary(summary, prompt, response, session)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary, encoding="utf-8")


def _read_input(workdir: Path) -> str:
    return input(f"[{workdir}]> ")


def _decide_mode(session: C0d3rSession, prompt: str, *, default_scientific: bool, default_tool_loop: bool) -> str:
    """
    Ask a routing model to choose the best execution mode.
    """
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
        response = _call_with_timeout(
            session.send,
            timeout_s=_model_timeout_s(),
            kwargs={"prompt": research_prompt, "stream": False, "research_override": True},
        )
        return response or ""
    except Exception:
        return ""


def _model_timeout_s() -> float:
    try:
        return float(os.getenv("C0D3R_MODEL_TIMEOUT_S", "60") or "60")
    except Exception:
        return 60.0


def _call_with_timeout(fn, *, timeout_s: float, kwargs: dict) -> str | None:
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
        return None
    if error:
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
) -> Tuple[bool, bool]:
    tests_ran = False
    tests_ok = False
    usage_tracker.set_status("testing", f"tests in {project_root}")
    # Python tests
    if (project_root / "pytest.ini").exists() or (project_root / "tests").exists():
        tests_ran = True
        cmd = "python -m pytest"
        code, stdout, stderr = run_command(cmd, cwd=project_root, timeout_s=_command_timeout_s(cmd))
        _append_tool_log(log_path, cmd, code, stdout, stderr)
        if code == 0:
            tests_ok = True
        else:
            return tests_ran, False
    # Node tests
    pkg = project_root / "package.json"
    if pkg.exists():
        try:
            import json as _json
            payload = _json.loads(pkg.read_text(encoding="utf-8"))
            scripts = payload.get("scripts") or {}
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
        header = (
            f"c0d3r session | model: {model}\n"
            f"{token_line} | {cost_line}\n"
            f"{rate_line}\n"
            f"{status_line}\n"
            f"{'-' * 70}\n"
        )
        return header


if __name__ == "__main__":
    raise SystemExit(main())
