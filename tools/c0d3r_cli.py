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
    from services.env_loader import EnvLoader
    from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings
    from services.agent_workspace import run_command

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

    if context_block:
        prompt = f"{context_block}\n\nUser request:\n{prompt}"
    settings = dict(settings)
    if "stream_default" in settings:
        settings["stream_default"] = settings.get("stream_default") and not args.no_stream
    session = C0d3rSession(
        session_name="c0d3r-cli",
        transcript_dir=Path("runtime/c0d3r/transcripts"),
        read_timeout_s=None,
        workdir=str(workdir),
        **settings,
    )
    usage = UsageTracker(model_id=session.get_model_id())
    stream = not args.no_stream
    typewriter = _typewriter_callback(usage) if stream else None
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
    base_snapshot = _snapshot_projects_dir(prompt)
    for step in range(max_steps):
        tool_prompt = (
            "You can run local shell commands to inspect the repo and validate work. "
            "Return ONLY JSON with keys: commands (list of strings) and final (string or empty). "
            "If you are done, set final to your response and commands to []. "
            "Meta commands available:\n"
            "- ::bg <command> (run long-lived command in background, returns pid)\n"
            "- ::wait_http <url> <seconds> (poll until HTTP 200 or timeout)\n"
            "- ::sleep <seconds>\n"
            "- ::kill <pid>\n"
            "- ::cd <path> (change working directory for subsequent commands)\n"
            "Be concise and execution-focused. Always execute commands before returning final.\n"
            f"Step {step + 1}/{max_steps}.\n"
            f"Request:\n{prompt}\n"
            + ("\n\nRecent outputs:\n" + "\n".join(history[-6:]) if history else "")
        )
        usage_tracker.add_input(tool_prompt)
        response = session.send(tool_prompt, stream=stream, images=images, stream_callback=stream_callback)
        commands, final = _extract_commands(response)
        if final and not commands:
            if _requires_commands_for_task(prompt):
                history.append("note: commands required for this task; no final allowed without execution")
                continue
            return final
        if not commands and not final:
            return response
        for cmd in commands[:8]:
            if cmd.startswith("::"):
                code, stdout, stderr, new_cwd = _execute_meta_command(cmd, workdir)
                if new_cwd:
                    workdir = new_cwd
            else:
                code, stdout, stderr = run_command(cmd, cwd=workdir)
            snippet = f"$ {cmd}\n(exit {code})\n{stdout.strip()}\n{stderr.strip()}".strip()
            history.append(snippet[:4000])
        # post-check: if prompt asks for new dir under Projects, ensure one exists
        if base_snapshot and _requires_new_projects_dir(prompt):
            new_dirs = _diff_projects_dir(base_snapshot)
            if not new_dirs:
                history.append("note: no new directory detected under C:/Users/Adam/Projects")
    return history[-1] if history else "No output."


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
    response = session.send(
        analysis_prompt,
        stream=stream,
        images=images,
        evidence_bundle=evidence,
        stream_callback=stream_callback,
    )
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


def _typewriter_callback(usage, controller=None):
    delay_ms = float(os.getenv("C0D3R_TYPEWRITER_MS", "8") or "8")
    delay_s = max(0.0, delay_ms / 1000.0)
    header = HeaderRenderer(usage)
    header.render()

    def _callback(chunk: str) -> None:
        if controller and controller.interrupted:
            return
        usage.add_output(chunk)
        header.update()
        for ch in chunk:
            if controller and controller.interrupted:
                return
            sys.stdout.write(ch)
            sys.stdout.flush()
            if ch.strip():
                time.sleep(delay_s)

    return _callback


def _typewriter_print(text: str, usage, controller=None) -> None:
    cb = _typewriter_callback(usage, controller=controller)
    cb(text)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")
        sys.stdout.flush()


def _render_json_response(text: str) -> str:
    payload = _safe_json(text)
    if not isinstance(payload, dict):
        return ""
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
    while True:
        try:
            prompt = pending if pending is not None else _read_input(workdir)
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
        recall_hits = memory.search_if_referenced(prompt, limit=5)
        if recall_hits:
            context = context + "\n\n[recall]\n" + "\n".join(recall_hits)
        system = (
            "Role: You are a senior software engineer. The user is a project manager. "
            "Provide brief status updates, then a final response. Avoid private chain-of-thought; "
            "summarize reasoning at a high level."
        )
        full_prompt = f"{context}\n\nUser:\n{prompt}"
        usage.add_input(full_prompt)
        print("[status] Planning...")
        print("[status] Executing...")
        controller = InterruptController()
        if allow_interrupt:
            controller.start()
        try:
            if scientific:
                response = _run_scientific_loop(
                    session, full_prompt, workdir, run_command, images=None, stream=False, stream_callback=None, usage_tracker=usage
                )
            elif tool_loop:
                response = _run_tool_loop(
                    session, full_prompt, workdir, run_command, images=None, stream=False, stream_callback=None, usage_tracker=usage
                )
            else:
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
        _typewriter_print(rendered, usage, controller=controller)
        sys.stdout.write("\n")
        sys.stdout.flush()
        memory.append(prompt, response)
        summary = memory.update_summary(summary, prompt, response, session)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary, encoding="utf-8")


def _read_input(workdir: Path) -> str:
    return input(f"[{workdir}]> ")


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
            proc = subprocess.Popen(command, cwd=str(workdir), shell=True)
            return 0, f"started pid {proc.pid}", "", None
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

    def add_input(self, text: str) -> None:
        self.input_chars += len(text or "")
        self.input_tokens = self.input_chars / 4.0

    def add_output(self, text: str) -> None:
        self.output_chars += len(text or "")
        self.output_tokens = self.output_chars / 4.0


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
        if not self.enabled:
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
        header = (
            f"c0d3r session | model: {model}\n"
            f"{token_line} | {cost_line}\n"
            f"{rate_line}\n"
        )
        return header


if __name__ == "__main__":
    raise SystemExit(main())
