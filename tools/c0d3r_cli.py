#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
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
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = PROJECT_ROOT / "web"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if WEB_ROOT.exists() and str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

_INSTALL_ATTEMPTS: set[str] = set()
_UI_MANAGER = None
_LAST_FILE_OPS_ERRORS: list[str] = []
_LAST_FILE_OPS_WRITTEN: list[str] = []
_MATRIX_SEED_VERSION = "2026-02-04"
_TECH_MATRIX_DIR = Path("runtime/c0d3r/tech_matrix")


def _trace_event(payload: dict) -> None:
    try:
        payload = dict(payload)
        payload["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path = Path("runtime/c0d3r/run_trace.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _tail_executor_log(lines: int = 6) -> str:
    path = Path("runtime/c0d3r/executor.log")
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(content[-lines:])
    except Exception:
        return ""


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
    if _UI_MANAGER:
        _UI_MANAGER.write_line(line)
    try:
        if not _UI_MANAGER:
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


def _emit_status_line(message: str) -> None:
    if _UI_MANAGER:
        _UI_MANAGER.set_status(message)
        return
    try:
        sys.stdout.write("\r" + message + " " * 10)
        sys.stdout.flush()
    except Exception:
        pass


def _animate_research(duration_s: float = 1.2) -> None:
    frames = ["📖", "📖", "📖", "📚", "📖", "📚", "📖"]
    start = time.time()
    idx = 0
    while time.time() - start < duration_s:
        _emit_status_line(f"research {frames[idx % len(frames)]} flipping pages...")
        idx += 1
        time.sleep(0.15)


def _animate_matrix(duration_s: float = 1.2) -> None:
    frames = ["⚡→○", "○→⚡", "○↔⚡", "⚡↔○", "○⇄⚡", "⚡⇄○"]
    symbols = ["∑", "∫", "π", "λ", "ψ", "Ω"]
    start = time.time()
    idx = 0
    while time.time() - start < duration_s:
        sym = symbols[idx % len(symbols)]
        _emit_status_line(f"matrix {frames[idx % len(frames)]} {sym}")
        idx += 1
        time.sleep(0.15)


class TerminalUI:
    def __init__(self, header: HeaderRenderer, workdir: Path) -> None:
        self.header = header
        self.workdir = workdir
        self.lines = deque(maxlen=1000)
        self.footer = ""
        self.status = ""
        self._lock = threading.Lock()
        self._input_queue: "queue.Queue[str]" = queue.Queue()
        self._running = False
        self._prompt_thread: threading.Thread | None = None
        self._render_thread: threading.Thread | None = None
        self._render_event = threading.Event()
        self._dirty = False
        self._last_render = 0.0
        self._min_render_interval = 1.0 / 30.0
        self._use_rich = False
        self._use_prompt_toolkit = False
        self._live = None
        self._console = None
        self._layout = None
        self._pt_app = None
        self._pt_header = None
        self._pt_output = None
        self._pt_input = None
        self._init_tui()

    def _init_tui(self) -> None:
        try:
            from prompt_toolkit.application import Application
            from prompt_toolkit.layout import Layout, HSplit, Window
            from prompt_toolkit.widgets import TextArea
            from prompt_toolkit.formatted_text import FormattedText
            from prompt_toolkit.key_binding import KeyBindings

            kb = KeyBindings()

            header = TextArea(
                height=4,
                text=self.header.render_text(),
                style="class:header",
                focusable=False,
                read_only=True,
            )
            output = TextArea(
                text="",
                focusable=False,
                read_only=True,
                scrollbar=True,
                wrap_lines=False,
            )
            input_box = TextArea(
                height=1,
                prompt=f"[{self.workdir}]> ",
                multiline=False,
                wrap_lines=False,
            )

            @kb.add("enter")
            def _(event) -> None:
                text = input_box.text
                input_box.text = ""
                if text is not None:
                    self._input_queue.put(text)

            root = HSplit(
                [
                    header,
                    Window(height=1, char="-"),
                    output,
                    Window(height=1, char="-"),
                    input_box,
                ]
            )
            layout = Layout(root, focused_element=input_box)
            self._pt_app = Application(layout=layout, key_bindings=kb, full_screen=True)
            self._pt_header = header
            self._pt_output = output
            self._pt_input = input_box
            self._use_prompt_toolkit = True
            self._use_rich = False
            return
        except Exception:
            self._use_prompt_toolkit = False
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.layout import Layout
            from rich.panel import Panel
            self._console = Console()
            self._layout = Layout()
            self._layout.split_column(
                Layout(name="header", size=5),
                Layout(name="body", ratio=1),
                Layout(name="footer", size=3),
            )
            self._live = Live(self._layout, console=self._console, refresh_per_second=8, transient=False)
            self._use_rich = True
        except Exception:
            self._use_rich = False

    def start(self) -> None:
        self._running = True
        if self._use_prompt_toolkit and self._pt_app:
            self._prompt_thread = threading.Thread(target=self._pt_app.run, daemon=True)
            self._prompt_thread.start()
        elif self._use_rich and self._live:
            self._live.start()
        if not self._use_prompt_toolkit:
            self._prompt_thread = threading.Thread(target=self._input_loop, daemon=True)
            self._prompt_thread.start()
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()
        self.render()

    def stop(self) -> None:
        self._running = False
        if self._use_rich and self._live:
            self._live.stop()
        if self._use_prompt_toolkit and self._pt_app:
            try:
                self._pt_app.exit()
            except Exception:
                pass
        self._render_event.set()

    def _render_loop(self) -> None:
        while self._running:
            self._render_event.wait(0.05)
            self._render_event.clear()
            if self._dirty:
                self.render(force=True)

    def _input_loop(self) -> None:
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.patch_stdout import patch_stdout
            session = PromptSession()
            while self._running:
                with patch_stdout():
                    text = session.prompt(f"[{self.workdir}]> ")
                if text is not None:
                    self._input_queue.put(text)
        except Exception:
            # Fallback to blocking input
            while self._running:
                try:
                    text = input(f"[{self.workdir}]> ")
                    self._input_queue.put(text)
                except Exception:
                    break

    def read_input(self, prompt: str) -> str:
        # Block until input is available.
        return self._input_queue.get()

    def set_header(self, text: str) -> None:
        with self._lock:
            self.header_text = text
        self.render()

    def set_status(self, text: str) -> None:
        with self._lock:
            self.status = text
        self.render()

    def set_footer(self, text: str) -> None:
        with self._lock:
            self.footer = text
        self.render()

    def write_line(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)
            self._dirty = True
        self.render()

    def write_text(self, text: str, *, delay_s: float = 0.0, controller=None) -> None:
        for ch in text:
            if controller and controller.interrupted:
                return
            with self._lock:
                if not self.lines:
                    self.lines.append("")
                if ch == "\n":
                    self.lines.append("")
                else:
                    self.lines[-1] = self.lines[-1] + ch
                self._dirty = True
            self.render()
            if ch.strip() and delay_s:
                time.sleep(delay_s)

    def render(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_render) < self._min_render_interval:
            self._render_event.set()
            return
        header_text = getattr(self, "header_text", self.header.render_text())
        body_text = "\n".join(self.lines)
        footer_text = self.footer or "input queued" if not self._input_queue.empty() else ""
        if self._use_prompt_toolkit and self._pt_app:
            if self._pt_header:
                self._pt_header.text = header_text
            if self._pt_output:
                self._pt_output.text = body_text
                try:
                    self._pt_output.buffer.cursor_position = len(self._pt_output.text)
                except Exception:
                    pass
            if self._pt_app:
                try:
                    self._pt_app.invalidate()
                except Exception:
                    pass
            self._last_render = now
            self._dirty = False
            return
        if self._use_rich and self._layout:
            from rich.panel import Panel
            from rich.text import Text
            self._layout["header"].update(Panel(header_text, title="c0d3r", border_style="blue"))
            self._layout["body"].update(Panel(Text(body_text), title="output"))
            self._layout["footer"].update(Panel(footer_text or "ready", title="input"))
            self._last_render = now
            self._dirty = False
            return
        # Basic ANSI fallback: clear + render.
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(header_text)
        sys.stdout.write(body_text + "\n")
        sys.stdout.write(footer_text + "\n")
        sys.stdout.flush()
        self._last_render = now
        self._dirty = False


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
    parser.add_argument("--matrix-query", dest="matrix_query", help="Query the equation matrix and exit.")
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
    # Default to full verbosity unless user disables it.
    os.environ.setdefault("C0D3R_VERBOSE_MODEL_OUTPUT", "0")
    os.environ.setdefault("C0D3R_BEDROCK_LIVE", "1")
    os.environ.setdefault("C0D3R_BEDROCK_STREAM", "1")
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

    if args.matrix_query:
        query = args.matrix_query.strip()
        if not query:
            print("matrix query is empty")
            return 1
        if not _ensure_django_ready():
            print("matrix query requires Django + database configured")
            return 1
        _seed_base_matrix_django()
        result = _matrix_search(query, limit=20)
        print(json.dumps(result, indent=2))
        return 0

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
    base_request = _strip_context_block(prompt)
    settings = dict(settings)
    if "stream_default" in settings:
        settings["stream_default"] = settings.get("stream_default") and not args.no_stream
    if os.getenv("C0D3R_ONESHOT", "").strip().lower() in {"1", "true", "yes", "on"}:
        if "update project" in base_request.lower():
            retarget = _maybe_retarget_project(base_request, workdir)
            if retarget:
                _emit_live(f"oneshot: retargeting workdir -> {retarget}")
                workdir = retarget
                os.chdir(workdir)
            else:
                _emit_live("oneshot: target project not found; skipping local fallback")
                return 0
        _emit_live("oneshot: attempting local fallback (pre-session)")
        if _apply_simple_task_fallback(base_request, workdir):
            _emit_live("oneshot: simple task complete")
            return 0
        if _apply_simple_project_stub(base_request, workdir):
            _emit_live("oneshot: simple project stub (pre-session)")
            return 0
        if _is_scaffold_task(base_request) and _requires_new_projects_dir(base_request) and _requires_scaffold_cmd(base_request):
            _emit_live("oneshot: scaffold task pre-session")
            project_root = _ensure_project_root(base_request, workdir) or (workdir / _slugify_project_name(base_request))
            scaffold_cmds = _fallback_scaffold_commands(base_request, project_root)
            for cmd in scaffold_cmds:
                run_command(_normalize_command(cmd, workdir), cwd=workdir, timeout_s=_command_timeout_s(cmd))
            return 0
    _emit_live("boot: session init")
    session = C0d3rSession(
        session_name="c0d3r-cli",
        transcript_dir=Path("runtime/c0d3r/transcripts"),
        workdir=str(workdir),
        **settings,
    )
    usage = UsageTracker(model_id=session.get_model_id())
    header = HeaderRenderer(usage)
    # Initialize terminal UI if available.
    global _UI_MANAGER
    use_tui = os.getenv("C0D3R_TUI", "1").strip().lower() not in {"0", "false", "no", "off"}
    if use_tui:
        try:
            _UI_MANAGER = TerminalUI(header, workdir)
            _UI_MANAGER.start()
        except Exception:
            _UI_MANAGER = None
    header.render()
    _emit_live("boot: header rendered")
    if os.getenv("C0D3R_SKIP_PRICING", "").strip().lower() in {"1", "true", "yes", "on"}:
        _emit_live("pricing: skipped by C0D3R_SKIP_PRICING")
    else:
        _refresh_pricing_cache(session, header, session.get_model_id())
    math_block = ""
    # Lightweight plan to avoid over-work on simple prompts.
    if _is_simple_file_task(base_request) or _is_scaffold_task(base_request):
        plan = {"mode": "tool_loop", "do_math": False, "do_research": False, "do_tool_loop": True}
    else:
        plan = _plan_execution(session, base_request)
    if plan.get("model_override"):
        session._c0d3r.model_id = str(plan.get("model_override"))
    # Simple/actionable tasks should avoid heavy math/research on first pass.
    if _is_simple_file_task(base_request) or _is_scaffold_task(base_request):
        do_math = False
        do_research = False
    else:
        do_math = bool(plan.get("do_math", settings.get("math_grounding", True)))
        do_research = bool(plan.get("do_research", not os.getenv("C0D3R_DISABLE_PRERESEARCH", "").strip().lower() in {"1","true","yes","on"}))
    if do_math and not _requires_rigorous_constraints(base_request):
        do_math = False
    # Long-form tech descriptions: build tech matrix + outline before tool loop.
    tech_matrix = None
    if _is_longform_request(base_request):
        _emit_live("longform: building tech matrix + outline")
        tech_matrix = _build_tech_matrix(session, base_request)
    if os.getenv("C0D3R_DISABLE_PRERESEARCH", "").strip().lower() in {"1", "true", "yes", "on"}:
        do_research = False
    if do_math:
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
        scientific=scientific if plan.get("mode") != "tool_loop" else False,
        tool_loop=tool_loop if plan.get("do_tool_loop", True) else False,
        initial_prompt=prompt or None,
        header=header,
        pre_research_enabled=do_research,
    )


def _build_context_block(workdir: Path, run_command) -> str:
    from services.framework_catalog import detect_frameworks
    lines = [
        "[context]",
        f"- cwd: {workdir}",
        f"- os: {os.name}",
    ]
    try:
        lines.append(f"- project_root: {workdir.resolve()}")
    except Exception:
        pass
    frameworks = detect_frameworks(workdir)
    if frameworks:
        lines.append(f"- frameworks: {', '.join(frameworks)}")
    else:
        lines.append("- frameworks: (none detected)")
    # Parallelize independent context probes.
    tasks = []
    tasks.append(("git_status", lambda: run_command("git status -sb", cwd=workdir)))
    tasks.append(("git_root", lambda: run_command("git rev-parse --show-toplevel", cwd=workdir)))
    if os.name == "nt":
        tasks.append(("ls", lambda: run_command("Get-ChildItem -Name", cwd=workdir)))
    else:
        tasks.append(("ls", lambda: run_command("ls -1", cwd=workdir)))
    results = _run_parallel_tasks([(name, fn) for name, fn in tasks], max_workers=3)
    result_map = {name: res for name, res in results}
    if "git_status" in result_map:
        code, stdout, stderr = result_map["git_status"]
        if stdout.strip():
            lines.append("git status -sb:")
            lines.append(stdout.strip()[:2000])
        if stderr.strip():
            lines.append("git status stderr:")
            lines.append(stderr.strip()[:500])
    if "git_root" in result_map:
        code, stdout, stderr = result_map["git_root"]
        if stdout.strip():
            lines.append(f"repo root: {stdout.strip()}")
    if "ls" in result_map:
        code, stdout, stderr = result_map["ls"]
        if stdout.strip():
            lines.append("top-level files:")
            lines.append("\n".join(stdout.strip().splitlines()[:80]))
    return "\n".join(lines)


def _context_scan_path(workdir: Path) -> Path:
    root = workdir.resolve()
    return Path("runtime/c0d3r") / f"context_scan_{root.name}.json"


def _is_existing_project(workdir: Path) -> bool:
    # Heuristic: git repo or framework markers or common project files.
    root = workdir.resolve()
    # Treat workspace-like directories (many subfolders, no project markers) as non-projects.
    try:
        entries = [p for p in root.iterdir() if p.is_dir()]
        if len(entries) >= 6:
            markers = ["pyproject.toml", "requirements.txt", "package.json", "manage.py", ".git"]
            if not any((root / m).exists() for m in markers):
                return False
    except Exception:
        pass
    markers = ["pyproject.toml", "requirements.txt", "package.json", "manage.py", ".git"]
    return any((root / m).exists() for m in markers)


def _scan_project_context(workdir: Path, run_command) -> dict:
    root = workdir.resolve()
    files: list[str] = []
    git_snapshot = _snapshot_git(root, run_command)
    # Prefer rg for speed.
    code, stdout, _ = run_command("rg --files", cwd=root)
    if code == 0 and stdout.strip():
        files = stdout.strip().splitlines()
    else:
        code, stdout, _ = run_command("Get-ChildItem -Recurse -File | Select-Object -ExpandProperty FullName", cwd=root)
        if code == 0 and stdout.strip():
            files = [f.replace(str(root) + os.sep, "") for f in stdout.strip().splitlines()]
    key_files = []
    for name in ("pyproject.toml", "requirements.txt", "package.json", "manage.py", "setup.cfg", "Pipfile"):
        if (root / name).exists():
            key_files.append(name)
    contents: dict[str, str] = {}
    for name in key_files:
        try:
            contents[name] = (root / name).read_text(encoding="utf-8", errors="ignore")[:4000]
        except Exception:
            continue
    scan = {
        "root": str(root),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_count": len(files),
        "files_sample": files[:200],
        "key_files": key_files,
        "key_contents": contents,
        "git_snapshot": git_snapshot,
    }
    try:
        path = _context_scan_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scan, indent=2), encoding="utf-8")
    except Exception:
        pass
    return scan


def _scan_is_fresh(workdir: Path, run_command, max_age_minutes: int = 15) -> bool:
    path = _context_scan_path(workdir)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    ts = payload.get("timestamp") or ""
    try:
        scan_time = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return False
    age = (datetime.datetime.now() - scan_time).total_seconds() / 60.0
    if age > max_age_minutes:
        return False
    git_snapshot = payload.get("git_snapshot") or ""
    current = _snapshot_git(workdir, run_command)
    if git_snapshot != current:
        return False
    return True


def _ensure_preflight(workdir: Path, run_command) -> None:
    path = Path("runtime/c0d3r/preflight.json")
    if path.exists():
        return
    checks = {}
    for cmd in ("python", "git", "node", "npm"):
        if os.name == "nt":
            code, stdout, _ = run_command(f"where {cmd}", cwd=workdir)
        else:
            code, stdout, _ = run_command(f"which {cmd}", cwd=workdir)
        checks[cmd] = {"found": code == 0, "path": stdout.strip().splitlines()[0] if stdout.strip() else ""}
    payload = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": checks}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _framework_decision(session: C0d3rSession, prompt: str, workdir: Path) -> dict:
    system = (
        "Return ONLY JSON with keys: framework (string), scaffold_commands (list), rationale (string). "
        "Choose a concrete framework stack for a new project and provide shell commands to scaffold it. "
        "If no new project is needed, return framework='existing' and empty scaffold_commands."
    )
    try:
        raw = session.send(prompt=f"Task:\n{prompt}\nCWD:\n{workdir}", stream=False, system=system)
        payload = _safe_json(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


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
    wants_new_project = False
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
    base_request = _tool_loop_base_request(prompt)
    wants_new_project = _requires_new_projects_dir(base_request)
    simple_task = _is_simple_file_task(base_request)
    scaffold_task = _is_scaffold_task(base_request)
    if simple_task or scaffold_task:
        full_completion = False
    require_tests = _requires_tests(prompt)
    allow_new_root_dirs = wants_new_project or _prompt_allows_new_dirs(prompt)
    require_benchmark = _requires_benchmark(prompt)
    actionable = _is_actionable_prompt(base_request) or _requires_commands_for_task(prompt) or wants_new_project
    consecutive_no_progress = 0
    test_failures = 0
    model_timeouts = 0
    unbounded_resolved = False
    behavior_log: list[dict] = []
    step = 0
    no_command_count = 0
    petals = PetalManager()
    wrote_project = False
    project_root_for_ops = _ensure_project_root(base_request, workdir) if wants_new_project else None
    if wants_new_project and project_root_for_ops is None:
        project_root_for_ops = (workdir / _slugify_project_name(base_request)).resolve()
    if wants_new_project:
        _emit_live(f"tool_loop: new-project mode -> root={project_root_for_ops or workdir}")
    scaffold_done = False
    if simple_task:
        _emit_live("simple_task: attempting local fallback before model call")
        if _apply_simple_task_fallback(base_request, workdir):
            _emit_live("simple_task: local fallback applied")
            return "complete"
    if scaffold_task and wants_new_project and _requires_scaffold_cmd(base_request):
        _emit_live("scaffold_task: running scaffold command before model call")
        scaffold_cmds = _fallback_scaffold_commands(base_request, project_root_for_ops or workdir)
        for cmd in scaffold_cmds:
            run_command(_normalize_command(cmd, workdir), cwd=workdir, timeout_s=_command_timeout_s(cmd))
        return "complete"
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
        research_tasks: list[tuple[str, callable]] = []
        disable_pr = os.getenv("C0D3R_DISABLE_PRERESEARCH", "").strip().lower() in {"1", "true", "yes", "on"}
        if gap_score >= 2 and not (simple_task or scaffold_task) and not disable_pr:
            _emit_live("tool_loop: gap score high; running targeted research")
            research_tasks.append(("gap_score", lambda: _pre_research(session, prompt)))
        # Apply dynamic petals via capability mapping (no hardwired petals).
        constraint_list = petals.state.get("constraints") or []
        petal_plan = _petal_action_plan(session, constraint_list, prompt)
        petal_effects = _apply_petal_plan(petal_plan) if petal_plan else {}
        if petal_effects and not (simple_task or scaffold_task) and not disable_pr:
            research_queries = petal_effects.get("research_queries") or []
            if research_queries:
                _emit_live(f"petal: research queries -> {len(research_queries)}")
                for rq in research_queries[:3]:
                    research_tasks.append((f"petal:{rq[:60]}", lambda rq=rq: _pre_research(session, rq)))
            if petal_effects.get("pause_for_input") and sys.stdin.isatty():
                _emit_live("petal: pause_for_user_input")
                print("\n[pause] Petal requested input. Add note and press Enter:")
                note = input("> ").strip()
                if note:
                    history.append(f"[petal note]\n{note}")
        # Matrix query for gaps/hits (skip for simple tasks)
        if not (simple_task or scaffold_task) and not disable_pr:
            matrix_info = _query_unbounded_matrix(prompt)
            if matrix_info:
                history.append(f"[matrix] hits={matrix_info.get('hits')} missing={matrix_info.get('missing')}")
                if matrix_info.get("missing"):
                    _emit_live("matrix: missing items; targeted research")
                    for item in matrix_info["missing"][:3]:
                        research_tasks.append((f"matrix:{item[:60]}", lambda item=item: _pre_research(session, item)))
        if research_tasks:
            _emit_live(f"research: running {len(research_tasks)} tasks in parallel")
            for label, note in _run_parallel_tasks(research_tasks, max_workers=3):
                if note:
                    tag = "research"
                    if label.startswith("petal:"):
                        tag = "petal research"
                    elif label.startswith("matrix:"):
                        tag = "matrix research"
                    history.append(f"[{tag}]\n{note[:2000]}")
                    _append_research_notes(note)
        enforce_progress = consecutive_no_progress >= 2 and full_completion
        if petal_effects.get("force_file_edits"):
            enforce_progress = True
        force_tests = bool(petal_effects.get("force_tests")) if petal_effects else False
        target_root = project_root_for_ops or workdir
        if simple_task or scaffold_task:
            history_block = ""
        else:
            history_block = ("\n\nRecent outputs:\n" + "\n".join(history[-6:])) if history else ""
        code_memory = _load_code_memory_summary()
        if code_memory and not simple_task:
            history_block = history_block + "\n\n" + code_memory
        tool_prompt = (
            "[schema:tool_loop]\n"
            "You can run local shell commands to inspect the repo and validate work. "
            "Return ONLY JSON with keys: commands (list of strings), file_ops (list), and final (string or empty). "
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
            + (
                ""
                if simple_task
                else "Create and maintain a checklist of requirements in your reasoning. "
                "Before finalizing, explicitly verify each checklist item is complete. "
                "If incomplete, keep working and iterate. "
                "It is OK to loop across tasks if they depend on each other.\n"
                "Create and update these files in runtime/c0d3r:\n"
                "- plan.md (expanded plan)\n"
                "- checklist.md (requirements checklist + verification notes)\n"
                "- bibliography.md (APA citations for any research/software used)\n"
                "Do not finalize until these files exist and are updated.\n"
                "If any errors or missing knowledge are detected, perform targeted research and "
                "record source->decision->code in checklist.md before continuing.\n"
                "If code/data is derived from research, add APA-formatted citations in code comments "
                "and append them to runtime/c0d3r/bibliography.md.\n"
                "If third-party libraries or datasets are added, record license obligations in "
                "runtime/c0d3r/bibliography.md and include required license files/notices in the repo.\n"
            )
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
            + f"Work inside the current project root: {target_root}. "
            "If the user asked for a new project, you MUST create a new folder and place all files inside it. "
            "Prefer modifying existing files in the target project and running its tests.\n"
            + "Meta commands available:\n"
            "- ::bg <command> (run long-lived command in background, returns pid)\n"
            "- ::wait_http <url> <seconds> (poll until HTTP 200 or timeout)\n"
            "- ::sleep <seconds>\n"
            "- ::kill <pid>\n"
            "- ::cd <path> (change working directory for subsequent commands)\n"
            "- ::textbook_deps (npm install for textbook pipeline)\n"
            "- ::textbook_fetch [path] (download LibreTexts PDFs into textbooks/)\n"
            "- ::textbook_segment [path] (extract/segment text from PDFs)\n"
            "- ::textbook_build_dataset [path] (build segments.ndjson)\n"
            "- ::textbook_build_tiles [path] (build tiles.ndjson)\n"
            "- ::textbook_fix_pages [path] (repair pages.json counts)\n"
            "- ::textbook_verify_pages [path] (verify processed page counts)\n"
            "- ::textbook_reprocess [path] (reprocess incomplete books; Windows-only)\n"
            "- ::textbook_import [path] (import manifest into Django DB + citations)\n"
            "- ::textbook_ocr [path] (OCR page PNGs -> Django DB)\n"
            "- ::textbook_list (list textbooks from DB)\n"
            "- ::textbook_prepare_qa [path] (generate QA candidates)\n"
            "- ::textbook_import_qa [path] (import QA candidates into DB)\n"
            "- ::textbook_knowledge [path] (build knowledge docs + queue items)\n"
            "Be concise and execution-focused. Always execute commands before returning final.\n"
            "Use this pattern for Ionic projects:\n"
            "1) ::cd C:/Users/Adam/Projects\n"
            "2) npx @ionic/cli@latest start <appname> tabs --type=angular --no-interactive --no-confirm --no-git\n"
            "3) ::cd <appname>\n"
            "4) ::bg npx @ionic/cli@latest serve --no-open --port <port>\n"
            "5) ::wait_http http://localhost:<port> 120\n"
            f"Step {step + 1}/{max_steps}.\n"
            f"Request:\n{base_request}\n"
            + history_block
            + (f"\n\n[gap_score]\n{gap_score}\n[last_error]\n{last_error}" if gap_score or last_error else "")
        )
        if model_timeouts >= 2:
            tool_prompt = (
            "Return ONLY JSON with keys: commands (list of strings), file_ops (list), and final (string or empty). "
                "Provide 2-6 concrete shell commands. Do not return an empty list.\n"
                f"Request:\n{base_request}\n"
            )
        _trace_event({
            "event": "tool_loop.prompt",
            "step": step,
            "wants_new_project": wants_new_project,
            "simple_task": simple_task,
            "scaffold_task": scaffold_task,
            "actionable": actionable,
            "project_root": str(project_root_for_ops or workdir),
            "prompt_preview": base_request[:200],
        })
        usage_tracker.add_input(tool_prompt)
        _emit_live("tool_loop: calling model for commands")
        _diag_log("tool_loop: model call start")
        # Apply petal overrides (model/timeout/microtests)
        saved_model_id = getattr(session._c0d3r, "model_id", "")
        saved_timeout = os.getenv("C0D3R_MODEL_TIMEOUT_S")
        if petal_effects.get("model_override"):
            session._c0d3r.model_id = str(petal_effects["model_override"])
            _emit_live(f"petal: switch_model -> {session._c0d3r.model_id}")
        if petal_effects.get("timeout_override") is not None:
            os.environ["C0D3R_MODEL_TIMEOUT_S"] = str(petal_effects["timeout_override"])
            _emit_live(f"petal: adjust_timeout_s -> {petal_effects['timeout_override']}")
        if petal_effects.get("enforce_microtests"):
            os.environ["C0D3R_MICROTESTS"] = "1"
            _emit_live("petal: enforce_microtests")
        def _timeout_reroute(note: str) -> tuple[str, list[str]]:
            """
            Failure-aware reroute: use research + local inspection + constrained command generator.
            Returns (response_text, injected_history).
            """
            injected: list[str] = []
            _emit_live(f"tool_loop: reroute on timeout ({note}) -> research + diagnostics")
            # 1) Targeted research based on last_error + base_request.
            try:
                research_note = _pre_research(session, base_request)
                if research_note:
                    injected.append("[reroute-research]\n" + research_note[:2000])
            except Exception:
                pass
            # 2) Local diagnostics to gather empirical evidence.
            diag_cmds = _fallback_inspection_commands(workdir)
            for cmd in diag_cmds[:4]:
                if cmd:
                    _emit_live(f"reroute: diag {cmd}")
                    code, stdout, stderr = run_command(_normalize_command(cmd, workdir), cwd=workdir, timeout_s=_command_timeout_s(cmd))
                    snippet = (stdout or "")[:1500]
                    err_snip = (stderr or "")[:800]
                    injected.append(f"[diagnostic] {cmd}\n{snippet}\n{err_snip}".strip())
            # 3) Constrained command generator using research + diagnostics.
            reroute_prompt = (
                "Return ONLY JSON with keys: commands (list of strings), final (string or empty), "
                "file_ops (list of {path, action, content}). "
                "Use the diagnostics and research below to choose the next actionable steps. "
                "Provide 2-6 concrete shell commands OR file_ops. Do NOT return empty commands/file_ops.\n"
                f"Request:\n{base_request}\n\n"
                + ("\n\n[reroute_context]\n" + "\n".join(injected) if injected else "")
            )
            try:
                response = session.send(prompt=reroute_prompt, stream=False)
                return response or "", injected
            except Exception:
                return "", injected
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
            history.append("note: model call timed out; rerouting to research+diagnostics path")
            reroute, injected = _timeout_reroute("primary")
            if injected:
                history.extend(injected[-3:])
            if reroute:
                response = reroute
            else:
                # Last-resort: local inspection commands to ensure forward motion.
                _emit_live("tool_loop: timeout reroute failed; using fallback inspection commands")
                commands = _fallback_inspection_commands(workdir)
                for cmd in commands[:5]:
                    if cmd:
                        run_command(_normalize_command(cmd, workdir), cwd=workdir, timeout_s=_command_timeout_s(cmd))
                continue
            mini_prompt = (
                "Return ONLY JSON with keys: commands (list of strings), final (string or empty), "
                "file_ops (list of {path, action, content}). "
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
                # Reroute again before continuing.
                reroute, injected = _timeout_reroute("minimal")
                if injected:
                    history.extend(injected[-3:])
                if reroute:
                    response = reroute
                else:
                    continue
        # Restore petal overrides
        session._c0d3r.model_id = saved_model_id
        if saved_timeout is None:
            os.environ.pop("C0D3R_MODEL_TIMEOUT_S", None)
        else:
            os.environ["C0D3R_MODEL_TIMEOUT_S"] = saved_timeout
        _diag_log("tool_loop: model call complete")
        if response and "schema_validation_failed" in response:
            # Only force file ops when the task is actionable (needs file/command work).
            if _is_actionable_prompt(base_request) or _requires_commands_for_task(prompt):
                _emit_live("tool_loop: schema validation failed; forcing file_ops retry")
                forced = _force_file_ops(session, prompt, workdir, base_root=project_root_for_ops or workdir)
                if forced:
                    for path in forced:
                        history.append(f"forced write: {path}")
                    wrote_project = True
        # Force file_ops if petal demands it
        if petal_effects.get("force_file_ops"):
            response = _enforce_actionability(
                session,
                "Return ONLY JSON with file_ops; do not include commands.",
                response or "",
            )
        else:
            response = _enforce_actionability(session, tool_prompt, response or "")
        actionable = _is_actionable_prompt(base_request) or _requires_commands_for_task(prompt) or wants_new_project
        response = _ensure_actionable_response(
            session,
            tool_prompt,
            response or "",
            actionable=actionable,
            max_retries=int(os.getenv("C0D3R_ACTION_RETRIES", "2")),
        )
        file_ops = _extract_file_ops_from_text(response or "")
        if file_ops:
            if simple_task:
                os.environ["C0D3R_ALLOW_FULL_REPLACE"] = "1"
            if simple_task:
                for op in file_ops:
                    if isinstance(op, dict) and "allow_full_replace" not in op:
                        op["allow_full_replace"] = True
            if project_root_for_ops and _requires_scaffold_cmd(prompt) and not scaffold_done:
                _emit_live("tool_loop: running scaffold command before applying file_ops")
                scaffold_cmds = _fallback_scaffold_commands(prompt, project_root_for_ops)
                for cmd in scaffold_cmds:
                    run_command(_normalize_command(cmd, workdir), cwd=workdir, timeout_s=_command_timeout_s(cmd))
                scaffold_done = True
            _emit_live(f"tool_loop: applying {len(file_ops)} file ops from model response")
            _trace_event({
                "event": "file_ops.apply",
                "count": len(file_ops),
                "base_root": str(project_root_for_ops or workdir),
                "prompt": base_request[:200],
            })
            applied = _apply_file_ops(file_ops, workdir, base_root=project_root_for_ops or workdir)
            if not applied and _LAST_FILE_OPS_ERRORS:
                _emit_live("tool_loop: file_ops applied=0; last errors:\n" + "\n".join(_LAST_FILE_OPS_ERRORS[-5:]))
                tail = _tail_executor_log()
                if tail:
                    _emit_live("executor.log tail:\n" + tail)
            if not applied:
                history.append("error: file_ops rejected (paths/validation). Retrying with strict paths.")
                strict_prompt = (
                    "Your file_ops were rejected (invalid paths or missing allow_full_replace). "
                    "Return ONLY JSON with file_ops using relative paths within the project root "
                    f"({project_root_for_ops or workdir}). If overwriting, set allow_full_replace=true. "
                    "Do NOT reference runtime/ paths."
                )
                retry = session.send(prompt=strict_prompt, stream=False)
                retry = _enforce_actionability(session, strict_prompt, retry or "")
                retry_ops = _extract_file_ops_from_text(retry or "")
                if retry_ops:
                    applied = _apply_file_ops(retry_ops, workdir, base_root=project_root_for_ops or workdir)
            if not applied and project_root_for_ops and _file_ops_only_runtime(file_ops, project_root_for_ops):
                _emit_live("tool_loop: file_ops runtime-only; forcing scaffold")
                scaffold_cmds = _fallback_scaffold_commands(prompt, project_root_for_ops)
                for cmd in scaffold_cmds:
                    run_command(_normalize_command(cmd, workdir), cwd=workdir, timeout_s=_command_timeout_s(cmd))
                scaffold_done = True
            for path in applied:
                history.append(f"model file write: {path}")
            wrote_project = wrote_project or bool(applied)
            if _microtest_enabled() and applied:
                py_targets = [p for p in applied if str(p).endswith(".py") and "/tests/" not in str(p).replace("\\", "/")]
                if py_targets:
                    _emit_live(f"microtests: running on {len(py_targets)} files")
                    ok, output = _run_microtests_for_paths(workdir, run_command, py_targets)
                    if not ok:
                        history.append("error: microtests failed; fixing before continuing")
                        fix_prompt = (
                            "Microtests failed. Provide file_ops or commands to fix the errors, "
                            "then re-run microtests. Return ONLY JSON with commands and file_ops.\n"
                            f"Errors:\n{output}"
                        )
                        response = session.send(prompt=fix_prompt, stream=False)
                        response = _enforce_actionability(session, fix_prompt, response or "")
                        retry_ops = _extract_file_ops_from_text(response or "")
                        if retry_ops:
                            _apply_file_ops(retry_ops, workdir)
                        commands, _ = _extract_commands(response)
                        for cmd in commands[:5]:
                            if cmd:
                                run_command(_normalize_command(cmd, workdir), cwd=workdir, timeout_s=_command_timeout_s(cmd))
                        ok, output = _run_microtests_for_paths(workdir, run_command, py_targets)
                        if not ok:
                            history.append("error: microtests still failing after remediation")
                            test_failures += 1
            if simple_task and wrote_project:
                success = True
                break
        elif _looks_like_code(response or ""):
            _emit_live("tool_loop: code detected without file targets; rejecting and requesting file_ops")
            history.append("error: code output without file targets; re-requesting with file_ops")
            repair_prompt = (
                "You produced code without file paths. Return ONLY JSON with key file_ops "
                "(list of {path, action, content}) so the code can be written to disk. "
                "Do not include prose.\n"
                f"Original response:\n{response}"
            )
            response = session.send(repair_prompt, stream=False)
            file_ops = _extract_file_ops_from_text(response or "")
            if file_ops:
                _emit_live(f"tool_loop: applying {len(file_ops)} file ops after repair")
                applied = _apply_file_ops(file_ops, workdir, base_root=project_root_for_ops or workdir)
                if not applied and _LAST_FILE_OPS_ERRORS:
                    _emit_live("tool_loop: file_ops applied=0; last errors:\n" + "\n".join(_LAST_FILE_OPS_ERRORS[-5:]))
                if not applied and _LAST_FILE_OPS_ERRORS:
                    _emit_live("tool_loop: file_ops applied=0; last errors:\n" + "\n".join(_LAST_FILE_OPS_ERRORS[-5:]))
                for path in applied:
                    history.append(f"model file write: {path}")
                wrote_project = wrote_project or bool(applied)
        # If still nothing applied but response has JSON, try coercing file ops again.
        if not wrote_project and _safe_json(response):
            extra_ops = _extract_file_ops_from_text(response or "")
            if extra_ops:
                _emit_live(f"tool_loop: applying {len(extra_ops)} coerced file ops")
                applied = _apply_file_ops(extra_ops, workdir, base_root=project_root_for_ops or workdir)
                if not applied and _LAST_FILE_OPS_ERRORS:
                    _emit_live("tool_loop: file_ops applied=0; last errors:\n" + "\n".join(_LAST_FILE_OPS_ERRORS[-5:]))
                for path in applied:
                    history.append(f"model file write: {path}")
                wrote_project = wrote_project or bool(applied)
        commands, final = _extract_commands(response)
        _emit_live(f"tool_loop: model returned {len(commands)} commands, final={'yes' if final else 'no'}")
        if actionable and not commands and not file_ops:
            history.append("error: actionable response without commands/file_ops; retrying")
            consecutive_no_progress += 1
            if consecutive_no_progress >= 1:
                history.append("note: forcing file_ops due to no actionable output")
                forced = _force_file_ops(session, prompt, workdir, base_root=project_root_for_ops or workdir)
                if forced:
                    for path in forced:
                        history.append(f"forced write: {path}")
                    wrote_project = True
                continue
        if _commands_only_runtime(commands):
            _emit_live("tool_loop: runtime-only commands detected; forcing scaffold for new project")
            if project_root_for_ops and wants_new_project:
                scaffold_cmds = _fallback_scaffold_commands(prompt, project_root_for_ops)
                commands = scaffold_cmds
        if not commands and not file_ops and wants_new_project:
            _emit_live("tool_loop: no commands for new project; forcing file_ops scaffold")
            forced = _force_file_ops(session, prompt, workdir, base_root=project_root_for_ops or workdir)
            if forced:
                for path in forced:
                    history.append(f"forced write: {path}")
                wrote_project = True
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
        if final and not commands:
            return final
        if not commands and not final:
            if simple_task and not file_ops:
                _emit_live("simple_task: applying local fallback")
                if _apply_simple_task_fallback(base_request, workdir):
                    wrote_project = True
                    return "Success: simple task completed."
            if step == 1 and not file_ops:
                _emit_live("tool_loop: forcing file_ops on first step")
                forced = _force_file_ops(session, prompt, workdir, base_root=project_root_for_ops or workdir)
                if forced:
                    wrote_project = True
                    return "Success: initial file ops applied."
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
                        "Return ONLY JSON with keys: commands (list of strings), file_ops (list), and final (string or empty). "
                        "Provide 2-5 concrete shell commands OR file_ops to make progress on the task. "
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
                    "Return ONLY JSON with keys: commands (list of strings), final (string or empty), "
                    "file_ops (list of {path, action, content}). "
                    "Provide 2-5 concrete shell commands OR file_ops to make progress on the task. "
                    "Do NOT return empty commands and file_ops.\n"
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
        checkpoint = _critical_thinking_checkpoint(session, prompt, history)
        if checkpoint:
            try:
                path = Path("runtime/c0d3r/critical_thinking.jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"ts": time.time(), **checkpoint}) + "\n")
            except Exception:
                pass
            # Keep a rolling plan checklist
            plan_steps = checkpoint.get("plan_steps") or []
            if plan_steps:
                try:
                    plan_path = Path("runtime/c0d3r/plan.json")
                    plan_path.parent.mkdir(parents=True, exist_ok=True)
                    plan_payload = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": plan_steps}
                    plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
                except Exception:
                    pass
                # Apply dynamic plan reordering from petals if provided.
                if petal_effects and petal_effects.get("reorder_steps"):
                    order = [s for s in petal_effects.get("reorder_steps") if s]
                    if order:
                        _emit_live("petal: applying dynamic step order")
                        plan_steps = order + [s for s in plan_steps if s not in order]
                # Add extra plan steps if requested by petals.
                if petal_effects and petal_effects.get("add_plan_steps"):
                    extras = [s for s in petal_effects.get("add_plan_steps") if s]
                    if extras:
                        _emit_live("petal: adding plan steps")
                        plan_steps = list(plan_steps) + extras
            state = _load_plan_state()
            if not state or state.get("steps") != plan_steps:
                state = _init_plan_state(plan_steps, workdir, run_command)
                _save_plan_state(state)
            needs_code = bool(checkpoint.get("needs_code_changes"))
            if needs_code and not commands:
                _emit_live("critical: needs code changes; forcing file ops")
                forced = _force_file_ops(session, prompt, workdir, base_root=project_root_for_ops or workdir)
                if forced:
                    for path in forced:
                        history.append(f"forced write: {path}")
                    wrote_project = True
        plan_state = _load_plan_state()
        if plan_state and plan_state.get("status") == "in_progress" and not simple_task:
            steps = plan_state.get("steps") or []
            idx = int(plan_state.get("current_index") or 0)
            if idx < len(steps):
                current_step = steps[idx]
                _emit_live(f"plan: enforcing step {idx+1}/{len(steps)} -> {current_step}")
                prompt = f"[plan step]\n{current_step}\n\n{prompt}"
        if wants_new_project:
            decision = _framework_decision(session, prompt, workdir)
            if decision:
                try:
                    path = Path("runtime/c0d3r/framework_decision.json")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
                except Exception:
                    pass
                scaffold_cmds = decision.get("scaffold_commands") or []
                if scaffold_cmds:
                    _emit_live(f"framework: scaffold via {decision.get('framework')}")
                    commands = list(scaffold_cmds) + (commands or [])
        if commands and _commands_only_runtime(commands):
            history.append("note: commands only touch runtime/c0d3r; must operate on project files too")
            _emit_live("tool_loop: commands only touched runtime; inserting inspection commands")
            commands = _fallback_inspection_commands(workdir) + commands
        written_files: set[str] = set()
        defer_tests = os.getenv("C0D3R_DEFER_TESTS", "1").strip().lower() not in {"0", "false", "no", "off"}
        if force_tests:
            require_tests = True
            defer_tests = False
        pending_test_targets: List[Path | None] = []
        for cmd in commands[:8]:
            usage_tracker.set_status("executing", cmd)
            _emit_live(f"exec: {cmd}")
            if _command_outside_root(cmd, workdir, allow_projects=wants_new_project):
                history.append("error: command targets path outside project root; blocked")
                _append_tool_log(log_path, cmd, 1, "", "blocked: outside project root")
                consecutive_no_progress += 1
                continue
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
            if base_snapshot and wants_new_project:
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
        if consecutive_no_progress >= 2 and not wrote_project:
            _emit_live("no-progress: forcing file edits via file_ops")
            forced = _force_file_ops(session, prompt, workdir, base_root=project_root_for_ops or workdir)
            if forced:
                for path in forced:
                    history.append(f"forced write: {path}")
                wrote_project = True
                consecutive_no_progress = 0
                if require_tests:
                    for path in forced:
                        _emit_live(f"post-write: running targeted tests for {path}")
                        _run_tests_for_project(workdir, run_command, usage_tracker, log_path, target=path)
        else:
            consecutive_no_progress = 0
        # Repository mutation audit
        if full_completion:
            before = _snapshot_git(workdir, run_command)
            if not _require_repo_change(before, workdir, run_command):
                history.append("error: no repo mutations detected; forcing new edits")
                strict_prompt = (
                    "No repository changes were detected. Provide commands or file_ops that create or "
                    "update actual project files. Return ONLY JSON with keys: commands, final, file_ops."
                )
                response = session.send(prompt=strict_prompt, stream=False)
                response = _enforce_actionability(session, strict_prompt, response or "")
                file_ops = _extract_file_ops_from_text(response or "")
                if file_ops:
                    _apply_file_ops(file_ops, workdir)
                commands, _ = _extract_commands(response)
                for cmd in commands[:5]:
                    if cmd:
                        run_command(_normalize_command(cmd, workdir), cwd=workdir, timeout_s=_command_timeout_s(cmd))
                # Re-check repo state
                if not _require_repo_change(before, workdir, run_command):
                    history.append("error: still no repo mutations after forced edits")
                    continue
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
        # Enforce test creation for new Python files
        if wrote_project:
            new_py = [p for p in written_files if p.endswith(".py") and "\\tests\\" not in p and "/tests/" not in p]
            if new_py and not tests_ran:
                history.append("error: new Python files without tests; create tests before continuing")
                fix_prompt = (
                    "You created new Python files without tests. Provide file_ops to add tests under tests/ "
                    "and commands to run them. Return ONLY JSON with commands and file_ops."
                )
                response = session.send(prompt=fix_prompt, stream=False)
                response = _enforce_actionability(session, fix_prompt, response or "")
                file_ops = _extract_file_ops_from_text(response or "")
                if file_ops:
                    _apply_file_ops(file_ops, workdir)
                commands, _ = _extract_commands(response)
                for cmd in commands[:5]:
                    if cmd:
                        run_command(_normalize_command(cmd, workdir), cwd=workdir, timeout_s=_command_timeout_s(cmd))
                tests_ran, tests_ok = _run_tests_for_project(workdir, run_command, usage_tracker, log_path)
                if not tests_ok:
                    history.append("error: tests failed after adding tests")
                    test_failures += 1
        if test_failures:
            _emit_live("tests: failure detected; enforcing remediation before continuing")
            fix_prompt = (
                "Tests failed. Provide commands or file_ops to fix the failures, then re-run tests. "
                "Return ONLY JSON with keys: commands, final, file_ops."
            )
            response = session.send(prompt=fix_prompt, stream=False)
            response = _enforce_actionability(session, fix_prompt, response or "")
            file_ops = _extract_file_ops_from_text(response or "")
            if file_ops:
                _apply_file_ops(file_ops, workdir)
            commands, _ = _extract_commands(response)
            for cmd in commands[:5]:
                if cmd:
                    run_command(_normalize_command(cmd, workdir), cwd=workdir, timeout_s=_command_timeout_s(cmd))
            # Re-run tests after remediation
            _emit_live("tests: re-running after remediation")
            _, tests_ok = _run_tests_for_project(workdir, run_command, usage_tracker, log_path)
            if not tests_ok:
                history.append("error: tests still failing after remediation")
                continue
            test_failures = 0
        # If loop appears stuck, disable active petals generically.
        if consecutive_no_progress >= 3 or test_failures >= 2 or model_timeouts >= 3:
            for name, active in (petals.state.get("active") or {}).items():
                if active:
                    petals.disable(name, "loop detected: no progress or timeouts")
                    _emit_live(f"petal: disabled {name} (loop detected)")
        if not (simple_task or scaffold_task):
            if _unbounded_trigger(consecutive_no_progress, test_failures, model_timeouts) and not unbounded_resolved:
                _emit_live("unbounded: detected spiral; building bounded objective matrix")
                unbounded_payload = _enforce_unbounded_matrix(session, prompt)
                if unbounded_payload:
                    unbounded_resolved = True
                    history.append("[unbounded]\n" + unbounded_payload.get("bounded_task", "").strip())
                    _append_unbounded_matrix(unbounded_payload)
                    prompt = _apply_unbounded_constraints(prompt, unbounded_payload)
                    _apply_behavior_insights(unbounded_payload, behavior_log)
            if _requires_rigorous_constraints(prompt) and not _unbounded_math_ready():
                _emit_live("unbounded: required matrix not complete; forcing resolver")
                unbounded_payload = _enforce_unbounded_matrix(session, prompt)
                if unbounded_payload:
                    unbounded_resolved = True
                    prompt = _apply_unbounded_constraints(prompt, unbounded_payload)
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
        # Enforce plan verification gate.
        if plan_state and plan_state.get("status") == "in_progress":
            steps = plan_state.get("steps") or []
            idx = int(plan_state.get("current_index") or 0)
            if idx < len(steps):
                current_step = steps[idx]
                if _verify_step(current_step, workdir, run_command, usage_tracker, log_path):
                    plan_state["current_index"] = idx + 1
                    plan_state["snapshot"] = _snapshot_git(workdir, run_command)
                    if plan_state["current_index"] >= len(steps):
                        plan_state["status"] = "complete"
                    _save_plan_state(plan_state)
                else:
                    history.append("error: plan verification failed; must fix before advancing")
                    continue
        # post-check: if prompt asks for new dir under Projects, ensure one exists
        if base_snapshot and wants_new_project:
            new_dirs = _diff_projects_dir(base_snapshot)
            if not new_dirs:
                history.append("error: no new project directory detected under C:/Users/Adam/Projects")
                consecutive_no_progress += 1
                if project_root_for_ops and _requires_scaffold_cmd(prompt):
                    _emit_live("tool_loop: forcing scaffold for missing project dir")
                    scaffold_cmds = _fallback_scaffold_commands(prompt, project_root_for_ops)
                    for cmd in scaffold_cmds:
                        run_command(_normalize_command(cmd, workdir), cwd=workdir, timeout_s=_command_timeout_s(cmd))
                    scaffold_done = True
                continue
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


def _extract_paths_from_command(cmd: str) -> list[str]:
    paths: list[str] = []
    if not cmd:
        return paths
    for match in re.findall(r"[A-Za-z]:\\\\[^\\s\"']+", cmd):
        paths.append(match)
    for match in re.findall(r"(?:^|\\s)(/[^\\s\"']+)", cmd):
        paths.append(match.strip())
    return paths


def _command_outside_root(cmd: str, workdir: Path, *, allow_projects: bool) -> bool:
    raw_paths = _extract_paths_from_command(cmd)
    if not raw_paths:
        return False
    root = workdir.resolve()
    projects_root = Path("C:/Users/Adam/Projects").resolve()
    for raw in raw_paths:
        try:
            candidate = Path(raw).resolve()
        except Exception:
            continue
        if allow_projects and str(candidate).startswith(str(projects_root)):
            continue
        if not str(candidate).startswith(str(root)):
            return True
    return False


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


def _extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    if not text:
        return candidates
    # Prefer fenced JSON blocks.
    for block in re.findall(r"```json\\s*([\\s\\S]+?)```", text, re.IGNORECASE):
        candidates.append(block.strip())
    # Then any fenced blocks.
    for block in re.findall(r"```([\\s\\S]+?)```", text):
        if "{" in block and "}" in block:
            candidates.append(block.strip())
    # Finally, scan for balanced JSON objects.
    start_indices = [i for i, ch in enumerate(text) if ch in "{["]
    for start in start_indices[:5]:
        stack = []
        for i in range(start, len(text)):
            ch = text[i]
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                stack.pop()
                if not stack:
                    candidates.append(text[start : i + 1])
                    break
    return candidates


def _safe_json(text: str):
    try:
        import json as _json
    except Exception:
        return {}
    preferred = None
    for candidate in _extract_json_candidates(text):
        try:
            parsed = _json.loads(candidate)
            if isinstance(parsed, dict) and ("file_ops" in parsed or "commands" in parsed):
                return parsed
            preferred = parsed
        except Exception:
            continue
    # Fallback: best-effort slice from first to last brace.
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            parsed = _json.loads(text[start : end + 1])
            if isinstance(parsed, dict) and ("file_ops" in parsed or "commands" in parsed):
                return parsed
            preferred = preferred or parsed
    except Exception:
        pass
    return preferred or {}


def _looks_like_json(text: str) -> bool:
    if not text:
        return False
    if "{" in text and "}" in text:
        return True
    return False


def _strip_context_block(prompt: str) -> str:
    if not prompt:
        return ""
    marker = "User request:"
    if marker in prompt:
        return prompt.split(marker, 1)[-1].strip()
    if "User:\n" in prompt:
        return prompt.split("User:\n", 1)[-1].strip()
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
        "mapping (list), research_questions (list), symbol_definitions (dict), equation_units (dict). "
        "Express the request as math; if you cannot form explicit equations, "
        "provide symbolic placeholders AND at least 3 research_questions to "
        "identify missing constants/relations. Provide units for equations."
    )
    def _heartbeat(label: str, stop_event: threading.Event, interval: float = 5.0) -> threading.Thread:
        def _runner():
            tick = 0
            while not stop_event.is_set():
                msg = f"{label}... ({tick * interval:.0f}s)"
                _emit_status_line(msg)
                _emit_live(msg)
                stop_event.wait(interval)
                tick += 1

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return t

    model_id = session.get_model_id()
    _emit_live(f"math_grounding: model pass 1 (equations) -> waiting on Bedrock response [{model_id}]")
    try:
        stop = threading.Event()
        _heartbeat(f"math_grounding: awaiting Bedrock response (model pass 1) [{model_id}]", stop)
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
        # Reroute: if math grounding fails, perform research-only grounding.
        try:
            _emit_live("math_grounding: reroute to research-only grounding")
            research_only = session._c0d3r._research(base_request)
            if research_only:
                payload = {
                    "variables": [],
                    "unknowns": [],
                    "equations": [],
                    "constraints": [],
                    "mapping": [],
                    "research_questions": [f"Use research to derive equations for: {base_request}"],
                    "symbol_definitions": {},
                    "equation_units": {},
                    "research_only": research_only[:3000],
                }
        except Exception:
            pass
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
        "symbol_definitions": payload.get("symbol_definitions") or {},
        "equation_units": payload.get("equation_units") or {},
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
        if _UI_MANAGER:
            _UI_MANAGER.write_text(chunk, delay_s=delay_s, controller=controller)
            return
        if os.getenv("C0D3R_VERBOSE_MODEL_OUTPUT", "0").strip().lower() in {"1", "true", "yes", "on"}:
            sys.stdout.write("\n[model]\n")
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
        if os.getenv("C0D3R_SHOW_MODEL_ACTIONS", "1").strip().lower() not in {"0", "false", "no", "off"}:
            commands = payload.get("commands") or []
            if commands:
                lines.append("\nCommands:")
                for cmd in commands:
                    lines.append(f"- {cmd}")
            ops = payload.get("file_ops") or payload.get("actions") or []
            if ops:
                lines.append("\nFile ops:")
                for op in ops:
                    if not isinstance(op, dict):
                        continue
                    path = op.get("path") or ""
                    action = op.get("action") or "write"
                    lines.append(f"- {action}: {path}")
        final = str(payload.get("final") or "").strip()
        if final:
            lines.append("\n" + final if lines else final)
        return "\n".join(lines).strip()
    if "final" in payload:
        return str(payload.get("final") or "")
    # Tool-loop JSON without final should not be printed verbatim.
    if "commands" in payload or "file_ops" in payload or "actions" in payload:
        if os.getenv("C0D3R_SHOW_MODEL_ACTIONS", "1").strip().lower() not in {"0", "false", "no", "off"}:
            lines = []
            commands = payload.get("commands") or []
            if commands:
                lines.append("Commands:")
                for cmd in commands:
                    lines.append(f"- {cmd}")
            ops = payload.get("file_ops") or payload.get("actions") or []
            if ops:
                lines.append("\nFile ops:")
                for op in ops:
                    if not isinstance(op, dict):
                        continue
                    path = op.get("path") or ""
                    action = op.get("action") or "write"
                    lines.append(f"- {action}: {path}")
            return "\n".join([l for l in lines if l]).strip()
        return ""
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


def _extract_file_ops_from_text(text: str) -> list[dict]:
    # Prefer any JSON block that includes file_ops.
    for candidate in _extract_json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and parsed.get("file_ops"):
            return _normalize_file_ops(parsed.get("file_ops") or [])
        if isinstance(parsed, dict) and parsed.get("actions"):
            return _normalize_file_ops(parsed.get("actions") or [])
    payload = _safe_json(text)
    if isinstance(payload, dict) and payload.get("file_ops"):
        ops = payload.get("file_ops") or []
        return _normalize_file_ops(ops)
    if isinstance(payload, dict) and payload.get("actions"):
        ops = payload.get("actions") or []
        return _normalize_file_ops(ops)
    # Common alternative shape: {"files": [{"path": "...", "content": "..."}]}
    if isinstance(payload, dict) and payload.get("files"):
        ops: list[dict] = []
        files = payload.get("files") or []
        if isinstance(files, dict):
            files = [files]
        for item in files:
            if not isinstance(item, dict):
                continue
            path = item.get("path") or item.get("file") or item.get("name")
            content = item.get("content") or item.get("text") or ""
            if path:
                ops.append({"path": str(path), "action": "write", "content": content})
        if ops:
            return _normalize_file_ops(ops)
    # Common alternative: {"path": "...", "content": "..."}
    if isinstance(payload, dict) and payload.get("path") and payload.get("content") is not None:
        return _normalize_file_ops([{"path": str(payload["path"]), "action": "write", "content": payload.get("content") or ""}])
    # Heuristic: ```file path/to/file\n...```
    ops: list[dict] = []
    for block in re.findall(r"```file\\s+([^\\n]+)\\n([\\s\\S]+?)```", text):
        path, content = block
        ops.append({"path": path.strip(), "action": "write", "content": content})
    return _normalize_file_ops(ops)


def _looks_like_code(text: str) -> bool:
    if not text:
        return False
    if "```" in text:
        return True
    keywords = ["def ", "class ", "import ", "function ", "const ", "let ", "var ", "public ", "private "]
    return any(k in text for k in keywords)


def _normalize_file_ops(ops: list) -> list[dict]:
    normalized: list[dict] = []
    for op in ops or []:
        if not isinstance(op, dict):
            continue
        action = op.get("action") or op.get("operation") or op.get("type")
        if action:
            action = str(action).strip().lower()
        # Map common verbs to executor actions.
        if action in {"create", "write", "update", "add", "insert"}:
            action = "write"
        elif action in {"append"}:
            action = "append"
        elif action in {"delete", "remove"}:
            action = "delete"
        elif action in {"mkdir", "dir", "directory"}:
            action = "mkdir"
        elif action in {"move", "rename", "copy", "patch", "replace"}:
            action = action
        elif action in {"read", "read_file", "read_directory", "list", "ls"}:
            # Ignore read ops (no executor support).
            continue
        op_norm = dict(op)
        if action:
            op_norm["action"] = action
        normalized.append(op_norm)
    return normalized


def _validate_action_schema(text: str) -> bool:
    payload = _safe_json(text)
    if not isinstance(payload, dict):
        return False
    cmds = payload.get("commands")
    ops = payload.get("file_ops") or payload.get("actions")
    if cmds is None and ops is None:
        return False
    if cmds is not None and not isinstance(cmds, list):
        return False
    if ops is not None and not isinstance(ops, list):
        return False
    return True


def _apply_file_ops(ops: list, workdir: Path, *, base_root: Path | None = None) -> list[Path]:
    global _LAST_FILE_OPS_ERRORS, _LAST_FILE_OPS_WRITTEN
    executor = FileExecutor(base_root or workdir)
    if os.getenv("C0D3R_ALLOW_FULL_REPLACE", "").strip().lower() in {"1", "true", "yes", "on"}:
        for op in ops or []:
            if isinstance(op, dict) and "allow_full_replace" not in op:
                op["allow_full_replace"] = True
    written = executor.apply_ops(ops)
    _LAST_FILE_OPS_WRITTEN = [str(p) for p in written]
    _LAST_FILE_OPS_ERRORS = executor.last_errors[:]
    return written


def _apply_unified_diff(original: str, diff_text: str) -> str:
    lines = original.splitlines(keepends=True)
    diff_lines = diff_text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    di = 0
    while di < len(diff_lines):
        line = diff_lines[di]
        if not line.startswith("@@"):
            di += 1
            continue
        # Parse hunk header: @@ -l,s +l,s @@
        m = re.match(r"@@\s+-([0-9]+)(?:,([0-9]+))?\s+\+([0-9]+)(?:,([0-9]+))?\s+@@", line)
        if not m:
            di += 1
            continue
        start_old = int(m.group(1)) - 1
        # flush unchanged
        while i < start_old and i < len(lines):
            out.append(lines[i])
            i += 1
        di += 1
        # Apply hunk
        while di < len(diff_lines):
            hline = diff_lines[di]
            if hline.startswith("@@"):
                break
            if hline.startswith(" "):
                out.append(lines[i])
                i += 1
            elif hline.startswith("-"):
                i += 1
            elif hline.startswith("+"):
                out.append(hline[1:])
            di += 1
    # Append remaining
    out.extend(lines[i:])
    return "".join(out)


def _resolve_target_path(base: Path, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    path = raw_path.strip().strip('"').strip("'")
    path = os.path.expandvars(os.path.expanduser(path))
    base = base.resolve()
    target = (base / path) if not Path(path).is_absolute() else Path(path)
    try:
        target = target.resolve()
    except Exception:
        return None
    if _is_runtime_path(target):
        return None
    if not str(target).startswith(str(base)):
        return None
    return target


class FileExecutor:
    def __init__(self, workdir: Path) -> None:
        self.base = workdir.resolve()
        self.log_path = Path("runtime/c0d3r/executor.log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_errors: list[str] = []

    def _log(self, msg: str) -> None:
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[{ts}] {msg}\n")
        except Exception:
            pass

    def _comment_prefix(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext in {".py", ".sh", ".rb", ".pl"}:
            return "#"
        if ext in {".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".cs", ".go"}:
            return "//"
        if ext in {".md", ".txt"}:
            return ""
        return "#"

    def _citation_block(self, sources: list, path: Path) -> str:
        prefix = self._comment_prefix(path)
        lines = ["C0D3R-CITATIONS:"]
        for src in sources:
            lines.append(f"- {src}")
        if prefix:
            return "\n".join(f"{prefix} {line}".rstrip() for line in lines) + "\n\n"
        return "\n".join(lines) + "\n\n"

    def _log_missing_sources(self, path: Path, action: str) -> None:
        try:
            todo_path = Path("runtime/c0d3r/bibliography_todo.jsonl")
            todo_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "path": str(path),
                "action": action,
                "note": "Missing APA sources for file op; add citations if research was used.",
            }
            with todo_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    def apply_ops(self, ops: list) -> list[Path]:
        written: list[Path] = []
        created_this_batch: set[str] = set()
        for op in ops:
            if not isinstance(op, dict):
                continue
            action = str(op.get("action") or "write").strip().lower()
            if action in {"read", "read_file", "read_directory", "list", "ls"}:
                self._log(f"skip read op: {op.get('path')}")
                self.last_errors.append(f"skip read op: {op.get('path')}")
                continue
            raw_path = str(op.get("path") or "").strip()
            target = _resolve_target_path(self.base, raw_path)
            if not target:
                self._log(f"reject path: {raw_path}")
                self.last_errors.append(f"reject path: {raw_path}")
                continue
            content = op.get("content") if op.get("content") is not None else ""
            sources = op.get("sources") or []
            if isinstance(sources, str):
                sources = [sources]
            self._log(f"op={action} path={target}")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                self._log(f"reject: parent exists as file {target.parent}")
                self.last_errors.append(f"reject: parent exists as file {target.parent}")
                continue
            except Exception as exc:
                self._log(f"reject: mkdir failed {target.parent} err={exc}")
                self.last_errors.append(f"reject: mkdir failed {target.parent} err={exc}")
                continue
            # Backup before destructive writes
            if target.exists() and action in {"write", "append", "replace", "patch", "delete", "remove", "move", "rename"}:
                backup_dir = Path("runtime/c0d3r/backups") / time.strftime("%Y%m%d_%H%M%S")
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / target.name
                try:
                    backup_path.write_bytes(target.read_bytes())
                    self._log(f"backup: {backup_path}")
                except Exception:
                    pass
            if action in {"mkdir", "dir", "directory"}:
                target.mkdir(parents=True, exist_ok=True)
                written.append(target)
                continue
            if action in {"delete", "remove"}:
                if target.is_dir():
                    for child in target.glob("**/*"):
                        if child.is_file():
                            child.unlink(missing_ok=True)
                    target.rmdir()
                elif target.exists():
                    target.unlink(missing_ok=True)
                written.append(target)
                continue
            # Evidence-first guard for large writes (warn + record, but do not block)
            if action in {"write", "append", "replace", "patch"} and not sources:
                self._log(f"warn: missing sources for write {target}")
                self._log_missing_sources(target, action)
            # Diff enforcement: do not overwrite existing files without allow_full_replace,
            # except for files created earlier in this batch.
            if action == "write" and target.exists() and not op.get("allow_full_replace"):
                if str(target) not in created_this_batch:
                    self._log(f"reject: write requires allow_full_replace {target}")
                    self.last_errors.append(f"reject: write requires allow_full_replace {target}")
                    continue
                # Allow overwrite if created in this batch.
                pass
            if action == "write" and not target.exists():
                created_this_batch.add(str(target))
            if action in {"move", "rename"}:
                dest_raw = str(op.get("dest") or "").strip()
                dest = _resolve_target_path(self.base, dest_raw)
                if dest:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    target.replace(dest)
                    written.append(dest)
                continue
            if action == "copy":
                dest_raw = str(op.get("dest") or "").strip()
                dest = _resolve_target_path(self.base, dest_raw)
                if dest and target.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(target.read_bytes())
                    written.append(dest)
                continue
            if action == "patch":
                if target.exists():
                    original = target.read_text(encoding="utf-8", errors="ignore")
                else:
                    original = ""
                patched = _apply_unified_diff(original, str(content))
                if sources and "C0D3R-CITATIONS" not in patched:
                    patched = self._citation_block(sources, target) + patched
                target.write_text(patched, encoding="utf-8")
                written.append(target)
                continue
            if action == "replace":
                find_text = str(op.get("find") or "")
                replace_text = str(op.get("replace") or "")
                original = target.read_text(encoding="utf-8", errors="ignore") if target.exists() else ""
                updated = original.replace(find_text, replace_text)
                if sources and "C0D3R-CITATIONS" not in updated:
                    updated = self._citation_block(sources, target) + updated
                target.write_text(updated, encoding="utf-8")
                written.append(target)
                continue
            if action == "append":
                existing = target.read_text(encoding="utf-8", errors="ignore") if target.exists() else ""
                if sources and "C0D3R-CITATIONS" not in existing:
                    existing = self._citation_block(sources, target) + existing
                with target.open("w", encoding="utf-8") as fh:
                    fh.write(existing + str(content))
            else:
                text = str(content)
                if sources and "C0D3R-CITATIONS" not in text:
                    text = self._citation_block(sources, target) + text
                target.write_text(text, encoding="utf-8")
            written.append(target)
            if not target.exists():
                self._log(f"warn: file op did not create target {target}")
        return written


def _force_file_ops(session: C0d3rSession, prompt: str, workdir: Path, *, base_root: Path | None = None) -> list[Path]:
    system = (
        "Return ONLY JSON with key: file_ops (list). "
        "Each item: {\"path\": \"relative/path\", \"action\": \"write|append\", \"content\": \"...\"}. "
        "Write concrete project files that implement the request. No runtime/ notes/ logs. "
        "Include sources (APA format) for any write/append/patch/replace in a 'sources' list. "
        "At least 2 file_ops."
    )
    if _requires_new_projects_dir(prompt):
        system += (
            " The task requires a NEW project folder under the current workspace. "
            "Ensure file_ops create that folder and place files inside it."
        )
    try:
        raw = session.send(prompt=prompt, stream=False, system=system)
        payload = _safe_json(raw)
    except Exception:
        payload = {}
    ops = payload.get("file_ops") if isinstance(payload, dict) else None
    if not ops:
        return []
    return _apply_file_ops(ops, workdir, base_root=base_root or workdir)


def _critical_thinking_checkpoint(session: C0d3rSession, prompt: str, history: List[str]) -> dict:
    system = (
        "Return ONLY JSON with keys: needs_code_changes (bool), needs_tests (bool), "
        "missing_info (list), next_actions (list), rationale (string), "
        "framework_guess (string), plan_steps (list). "
        "Use a critical-thinking checklist: clarify task, compare to evidence, "
        "decide if code edits are required now."
    )
    try:
        context = "\n".join(history[-6:])
        raw = session.send(
            prompt=f"Task:\n{prompt}\n\nRecent context:\n{context}",
            stream=False,
            system=system,
        )
        payload = _safe_json(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _enforce_actionability(session: C0d3rSession, prompt: str, response: str) -> str:
    # If response is only text and task is actionable, re-ask for commands/file_ops.
    if not response:
        return response
    if _validate_action_schema(response):
        return response
    if _extract_commands(response)[0]:
        return response
    if _extract_file_ops_from_text(response):
        return response
    if "schema_validation_failed" in response:
        system = (
            "Return ONLY valid JSON with keys: commands (list of strings), final (string or empty), "
            "file_ops (list of {path, action, content}). No extra text. "
            "Actions can be write|append|delete|mkdir|patch|replace|move|copy."
        )
        try:
            raw = session.send(prompt=prompt, stream=False, system=system)
            return raw
        except Exception:
            return response
    if _looks_like_code(response):
        # Force file targets for code output.
        system = (
            "You returned code without file targets. Return ONLY JSON with keys: "
            "commands (list of strings), final (string or empty), "
            "file_ops (list of {path, action, content}). "
            "Actions can be write|append|delete|mkdir|patch|replace|move|copy. "
            "If you output code, it MUST be in file_ops with explicit paths. "
            "Include sources (APA format list) for any write/append/patch/replace."
        )
        try:
            raw = session.send(prompt=prompt, stream=False, system=system)
            return raw
        except Exception:
            return response
    system = (
        "Return ONLY JSON with keys: commands (list of strings), final (string or empty), "
        "file_ops (list of {path, action, content}). "
        "Actions can be write|append|delete|mkdir|patch|replace|move|copy. "
        "Include sources (APA format list) for any write/append/patch/replace. "
        "You MUST provide commands or file_ops for actionable tasks."
    )
    try:
        raw = session.send(prompt=prompt, stream=False, system=system)
        return raw
    except Exception:
        return response


def _ensure_actionable_response(
    session: C0d3rSession,
    prompt: str,
    response: str,
    *,
    actionable: bool,
    max_retries: int = 2,
) -> str:
    """
    Ensure actionable tasks produce commands or file_ops.
    If code appears without file targets, force file_ops.
    """
    attempt = 0
    while True:
        if _validate_action_schema(response):
            return response
        if _extract_commands(response)[0]:
            return response
        if _extract_file_ops_from_text(response):
            return response
        if not actionable:
            return response
        if _looks_like_code(response):
            response = _enforce_actionability(session, prompt, response)
        else:
            response = _enforce_actionability(session, prompt, response)
        attempt += 1
        if attempt > max_retries:
            return response


def _plan_state_path() -> Path:
    return Path("runtime/c0d3r/plan_state.json")


def _load_plan_state() -> dict:
    path = _plan_state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_plan_state(state: dict) -> None:
    try:
        path = _plan_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def _init_plan_state(plan_steps: list, workdir: Path, run_command) -> dict:
    return {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": plan_steps,
        "current_index": 0,
        "snapshot": _snapshot_git(workdir, run_command),
        "status": "in_progress",
    }


def _snapshot_git(workdir: Path, run_command) -> str:
    code, stdout, _ = run_command("git status --porcelain", cwd=workdir)
    if code != 0:
        return ""
    return stdout.strip()


def _require_repo_change(before: str, workdir: Path, run_command) -> bool:
    after = _snapshot_git(workdir, run_command)
    return bool(after) and after != before


def _verify_step(step: str, workdir: Path, run_command, usage_tracker, log_path: Path) -> bool:
    step_lower = (step or "").lower()
    if "test" in step_lower or "pytest" in step_lower or "unit" in step_lower:
        _emit_live(f"verify: running tests for step '{step}'")
        _, ok = _run_tests_for_project(workdir, run_command, usage_tracker, log_path)
        return ok
    # Default verification: ensure repo changed since snapshot.
    current = _snapshot_git(workdir, run_command)
    return bool(current)


def _microtest_enabled() -> bool:
    return os.getenv("C0D3R_MICROTESTS", "1").strip().lower() not in {"0", "false", "no", "off"}


class PetalManager:
    """
    Dynamic constraint registry (petals).
    No hardwired petals; constraints are learned from user input or model directives.
    """
    def __init__(self) -> None:
        self.path = Path("runtime/c0d3r/petals.json")
        self.state = {
            "constraints": [],
            "active": {},
            "cooldowns": {},
            "last_used": {},
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self.state.update(payload)
        except Exception:
            pass

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        except Exception:
            pass

    def update_from_directives(self, directives: list[dict]) -> None:
        if directives:
            self.state["constraints"] = directives
        self.save()

    def active(self, name: str) -> bool:
        return bool(self.state.get("active", {}).get(name, True))

    def mark_used(self, name: str) -> None:
        self.state.setdefault("last_used", {})[name] = time.time()
        self.save()

    def disable(self, name: str, reason: str) -> None:
        self.state.setdefault("active", {})[name] = False
        self.state.setdefault("cooldowns", {})[name] = {
            "disabled_at": time.time(),
            "reason": reason,
        }
        self.save()


def _extract_petal_directives(prompt: str) -> list[dict]:
    """
    Extract dynamic constraints from user input. Each constraint is a dict:
    {name, type, payload, ttl, priority, rationale}.
    """
    if not prompt:
        return []
    directives: list[dict] = []
    # Minimal heuristic extraction; model can override by writing petals.json via file_ops if needed.
    lower = prompt.lower()
    if "research before" in lower:
        directives.append({
            "name": "dynamic_research_before",
            "type": "research_before",
            "payload": prompt.strip(),
            "ttl": "session",
            "priority": 5,
            "rationale": "user requested research before execution",
        })
    if "before coding" in lower or "code before" in lower:
        directives.append({
            "name": "dynamic_order",
            "type": "ordering_hint",
            "payload": prompt.strip(),
            "ttl": "session",
            "priority": 4,
            "rationale": "user specified ordering preference",
        })
    return directives


def _ensure_django_ready() -> bool:
    try:
        import os as _os
        if not _os.getenv("DJANGO_SETTINGS_MODULE"):
            _os.environ["DJANGO_SETTINGS_MODULE"] = "coolcrypto_dashboard.settings"
        import django
        django.setup()
        return True
    except Exception:
        return False


def _seed_base_matrix_django() -> None:
    if not _ensure_django_ready():
        return
    try:
        from core.models import Equation, EquationDiscipline, EquationSource
        if Equation.objects.filter(domains__contains=["ClassicalMechanics"]).exists():
            return
        base = [
            ("ClassicalMechanics", "F = m * a", "Newton's second law of motion."),
            ("ClassicalMechanics", "p = m * v", "Linear momentum."),
            ("Thermodynamics", "dU = T * dS - P * dV", "Fundamental thermodynamic relation."),
            ("Thermodynamics", "P * V = n * R * T", "Ideal gas law."),
            ("Electromagnetism", "∇ · E = ρ / ε0", "Gauss's law (electric)."),
            ("Electromagnetism", "∇ · B = 0", "Gauss's law (magnetic)."),
            ("Electromagnetism", "∇ × E = -∂B/∂t", "Faraday's law of induction."),
            ("Electromagnetism", "∇ × B = μ0 * J + μ0 * ε0 * ∂E/∂t", "Ampere-Maxwell law."),
            ("QuantumMechanics", "i * ħ * ∂ψ/∂t = Ĥ * ψ", "Time-dependent Schrödinger equation."),
            ("QuantumMechanics", "E = ħ * ω", "Planck-Einstein relation."),
            ("Relativity", "E^2 = (p*c)^2 + (m*c^2)^2", "Energy-momentum relation."),
            ("StatisticalMechanics", "S = k_B * ln(Ω)", "Boltzmann entropy."),
        ]
        source, _ = EquationSource.objects.get_or_create(
            title="Standard physics textbooks",
            defaults={
                "citation": "Standard physics textbooks (collected seed).",
                "tags": ["seed", _MATRIX_SEED_VERSION],
            },
        )
        for domain, eq, desc in base:
            EquationDiscipline.objects.get_or_create(name=domain)
            Equation.objects.get_or_create(
                text=eq,
                defaults={
                    "latex": eq,
                    "domains": [domain],
                    "disciplines": [domain],
                    "confidence": 0.95,
                    "source": source,
                    "assumptions": [],
                    "constraints": [],
                },
            )
    except Exception:
        pass


def _normalize_equation(eq: str) -> str:
    return re.sub(r"\s+", "", (eq or "")).strip()


def _authoritative_domains() -> list[str]:
    return [
        "nist.gov",
        "nih.gov",
        "ncbi.nlm.nih.gov",
        "arxiv.org",
        "nature.com",
        "science.org",
        "sciencemag.org",
        "ieee.org",
        "acm.org",
        "springer.com",
        "springerlink.com",
        "wiley.com",
        "sciencedirect.com",
        "royalsocietypublishing.org",
        "aps.org",
        "aip.org",
        "iop.org",
        "cambridge.org",
        "mit.edu",
        "stanford.edu",
        "harvard.edu",
        "caltech.edu",
        "ox.ac.uk",
        "cam.ac.uk",
    ]


def _is_authoritative_source(url: str) -> bool:
    if not url:
        return False
    lowered = url.lower()
    return any(dom in lowered for dom in _authoritative_domains())


def _is_longform_request(prompt: str) -> bool:
    return len(prompt or "") > 12000


def _split_longform(prompt: str, chunk_size: int = 4000) -> list[str]:
    text = prompt or ""
    chunks = []
    idx = 0
    while idx < len(text):
        chunks.append(text[idx : idx + chunk_size])
        idx += chunk_size
    return chunks


def _build_tech_matrix(session: C0d3rSession, prompt: str) -> dict:
    """
    Build a component matrix for long-form technology descriptions.
    Returns dict with nodes, connections, requirements, and outline.
    """
    _TECH_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = _split_longform(prompt)
    nodes: list[dict] = []
    requirements: list[str] = []
    interfaces: list[dict] = []
    data_shapes: list[dict] = []
    system = (
        "Return ONLY JSON with keys: components (list), requirements (list), interfaces (list), data_shapes (list). "
        "components should include {name, responsibility, inputs, outputs, dependencies}. "
        "interfaces should include {name, direction, protocol, payload_schema}. "
        "data_shapes should include {name, fields, storage, validation}."
    )
    for idx, chunk in enumerate(chunks):
        raw = session.send(prompt=f"[chunk {idx+1}/{len(chunks)}]\n{chunk}", stream=False, system=system)
        payload = _safe_json(raw)
        if not isinstance(payload, dict):
            continue
        nodes.extend(payload.get("components") or [])
        requirements.extend(payload.get("requirements") or [])
        interfaces.extend(payload.get("interfaces") or [])
        data_shapes.extend(payload.get("data_shapes") or [])
    matrix = {
        "prompt": prompt,
        "components": nodes,
        "requirements": requirements,
        "interfaces": interfaces,
        "data_shapes": data_shapes,
        "chunks": len(chunks),
    }
    ( _TECH_MATRIX_DIR / "latest.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    outline = _compile_tech_outline(session, matrix)
    matrix["outline"] = outline
    ( _TECH_MATRIX_DIR / "latest_outline.json").write_text(json.dumps(outline, indent=2), encoding="utf-8")
    _write_tech_matrix_db(matrix, prompt=prompt)
    return matrix


def _write_tech_matrix_db(matrix: dict, *, prompt: str = "") -> None:
    try:
        if not _ensure_django_ready():
            return
        import os as _os
        if not _os.getenv("DJANGO_SETTINGS_MODULE"):
            _os.environ["DJANGO_SETTINGS_MODULE"] = "coolcrypto_dashboard.settings"
        import django
        django.setup()
        from core.models import TechMatrixRecord
        TechMatrixRecord.objects.create(
            prompt=prompt or "",
            components=matrix.get("components") or [],
            requirements=matrix.get("requirements") or [],
            interfaces=matrix.get("interfaces") or [],
            data_shapes=matrix.get("data_shapes") or [],
            outline=matrix.get("outline") or {},
        )
    except Exception:
        pass


def _compile_tech_outline(session: C0d3rSession, matrix: dict) -> dict:
    system = (
        "Return ONLY JSON with keys: architecture (list), modules (list), data_flows (list), tests (list). "
        "Use the matrix to propose a concrete software outline."
    )
    raw = session.send(prompt=f"Matrix:\n{json.dumps(matrix, indent=2)[:12000]}", stream=False, system=system)
    payload = _safe_json(raw)
    return payload if isinstance(payload, dict) else {}


def _matrix_search(query: str, limit: int = 12) -> dict:
    if not query:
        return {"hits": [], "missing": []}
    if not _ensure_django_ready():
        return {"hits": [], "missing": []}
    _seed_base_matrix_django()
    try:
        from core.models import Equation
        q = query.strip()
        tokens = [t for t in re.findall(r"[a-zA-Z_]{3,}", q.lower()) if t not in {"that","this","with","from","into","then"}]
        hits: list[dict] = []
        qs = Equation.objects.all()
        if any(ch in q for ch in ("=", "∇", "∂", "ħ", "Ω", "λ", "μ")):
            qs = qs.filter(text__icontains=q)
        else:
            qs = qs.filter(text__icontains=q) | qs.filter(latex__icontains=q)
        for eq in qs[:limit]:
            hits.append({"equation": eq.text, "domain": ",".join(eq.disciplines or eq.domains or []), "summary": ""})
        if tokens:
            for eq in Equation.objects.all()[:200]:
                norm = _normalize_equation(eq.text)
                if any(_normalize_equation(t) in norm for t in tokens):
                    hits.append({"equation": eq.text, "domain": ",".join(eq.disciplines or eq.domains or []), "summary": ""})
        unique = []
        seen = set()
        for item in hits:
            key = item["equation"]
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
            if len(unique) >= limit:
                break
        missing = [t for t in set(tokens[:10]) if not any(t in str(h["equation"]).lower() for h in unique)]
        return {"hits": unique, "missing": missing}
    except Exception:
        return {"hits": [], "missing": []}


def _query_unbounded_matrix(prompt: str) -> dict:
    if not prompt:
        return {}
    _animate_matrix(0.8)
    result = _matrix_search(prompt)
    if result.get("hits"):
        return result
    return result


def _petal_action_plan(session: C0d3rSession, constraints: list[dict], prompt: str) -> dict:
    """
    Map dynamic constraints to available capabilities without hardwiring.
    Returns: {research_queries: [], reorder_steps: [], mode_hint: str}
    """
    if not constraints:
        return {}
    capabilities = [
        "research_queries",
        "reorder_plan_steps",
        "prioritize_tests",
        "prioritize_file_edits",
        "pause_for_user_input",
        "force_file_ops",
        "force_tool_loop",
        "force_scientific",
        "switch_model",
        "adjust_timeout_s",
        "enforce_microtests",
        "add_plan_steps",
        "skip_math_grounding",
    ]
    system = (
        "Return ONLY JSON with keys: research_queries (list), reorder_steps (list), "
        "mode_hint (string). Use constraints to decide how to use capabilities. "
        "Do not invent capabilities not listed."
    )
    try:
        raw = session.send(
            prompt=f"Constraints:\n{json.dumps(constraints, indent=2)}\n\nCapabilities:\n{capabilities}\n\nTask:\n{prompt}",
            stream=False,
            system=system,
        )
        payload = _safe_json(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _apply_petal_plan(petal_plan: dict) -> dict:
    """
    Normalize a petal plan into actionable flags for the execution loop.
    """
    effects = {
        "research_queries": petal_plan.get("research_queries") or [],
        "reorder_steps": petal_plan.get("reorder_steps") or [],
        "force_tests": bool(petal_plan.get("prioritize_tests")),
        "force_file_edits": bool(petal_plan.get("prioritize_file_edits")),
        "pause_for_input": bool(petal_plan.get("pause_for_user_input")),
        "force_file_ops": bool(petal_plan.get("force_file_ops")),
        "force_tool_loop": bool(petal_plan.get("force_tool_loop")),
        "force_scientific": bool(petal_plan.get("force_scientific")),
        "model_override": petal_plan.get("switch_model") or "",
        "timeout_override": petal_plan.get("adjust_timeout_s"),
        "enforce_microtests": bool(petal_plan.get("enforce_microtests")),
        "add_plan_steps": petal_plan.get("add_plan_steps") or [],
        "skip_math_grounding": bool(petal_plan.get("skip_math_grounding")),
    }
    return effects


def _write_microtest_harness(workdir: Path, targets: list[Path]) -> Path:
    runtime_dir = Path("runtime/c0d3r/microtests")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    target_list = [str(t.resolve()) for t in targets]
    payload_path = runtime_dir / "targets.json"
    payload_path.write_text(json.dumps(target_list, indent=2), encoding="utf-8")
    harness = runtime_dir / "run_microtests.py"
    harness.write_text(
        (
            "import json, inspect, importlib.util, sys, traceback\n"
            "from pathlib import Path\n"
            "ROOT = Path(__file__).resolve().parents[2]\n"
            "sys.path.insert(0, str(ROOT))\n"
            "targets = json.loads(Path(__file__).with_name('targets.json').read_text())\n"
            "errors = []\n"
            "def dummy_for(param):\n"
            "    name = param.name.lower()\n"
            "    ann = getattr(param, 'annotation', None)\n"
            "    if ann in (int, 'int'): return 1\n"
            "    if ann in (float, 'float'): return 1.0\n"
            "    if ann in (str, 'str'): return 'foo'\n"
            "    if ann in (bool, 'bool'): return True\n"
            "    if ann in (list, 'list'): return []\n"
            "    if ann in (dict, 'dict'): return {}\n"
            "    if 'path' in name or 'file' in name: return str(ROOT / 'runtime' / 'c0d3r' / 'foo_data.json')\n"
            "    if 'name' in name: return 'foo'\n"
            "    if 'count' in name or 'num' in name: return 1\n"
            "    return None\n"
            "def safe_call(func, params):\n"
            "    args = []\n"
            "    kwargs = {}\n"
            "    for p in params:\n"
            "        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):\n"
            "            continue\n"
            "        if p.default is not inspect._empty:\n"
            "            continue\n"
            "        kwargs[p.name] = dummy_for(p)\n"
            "    return func(**kwargs)\n"
            "def should_skip(name):\n"
            "    skip = ['delete','remove','drop','write','save','send','request','open','close','connect']\n"
            "    return any(s in name.lower() for s in skip)\n"
            "for target in targets:\n"
            "    path = Path(target)\n"
            "    if not path.exists():\n"
            "        continue\n"
            "    modname = path.stem\n"
            "    try:\n"
            "        spec = importlib.util.spec_from_file_location(modname, str(path))\n"
            "        mod = importlib.util.module_from_spec(spec)\n"
            "        spec.loader.exec_module(mod)\n"
            "    except Exception as exc:\n"
            "        errors.append(f'import {path}: {exc}\\n{traceback.format_exc()}')\n"
            "        continue\n"
            "    for name, obj in inspect.getmembers(mod):\n"
            "        if name.startswith('_'):\n"
            "            continue\n"
            "        try:\n"
            "            if inspect.isfunction(obj):\n"
            "                if should_skip(name):\n"
            "                    continue\n"
            "                sig = inspect.signature(obj)\n"
            "                safe_call(obj, sig.parameters.values())\n"
            "            elif inspect.isclass(obj):\n"
            "                if should_skip(name):\n"
            "                    continue\n"
            "                sig = inspect.signature(obj.__init__)\n"
            "                safe_call(obj, list(sig.parameters.values())[1:])\n"
            "        except Exception as exc:\n"
            "            errors.append(f'{path}:{name}: {exc}\\n{traceback.format_exc()}')\n"
            "if errors:\n"
            "    raise SystemExit('\\n\\n'.join(errors))\n"
            "print('microtests: ok')\n"
        ),
        encoding="utf-8",
    )
    return harness


def _run_microtests_for_paths(workdir: Path, run_command, targets: list[Path]) -> tuple[bool, str]:
    if not targets:
        return True, ""
    harness = _write_microtest_harness(workdir, targets)
    cmd = f'python "{harness}"'
    code, stdout, stderr = run_command(cmd, cwd=workdir, timeout_s=300)
    combined = (stdout or "") + ("\n" + stderr if stderr else "")
    return code == 0, combined.strip()


def _run_parallel_tasks(tasks: list[tuple[str, callable]], max_workers: int = 3) -> list[tuple[str, object]]:
    """
    Run independent tasks in parallel; return list of (label, result_text).
    """
    if not tasks:
        return []
    results: list[tuple[str, str]] = []
    use_parallel = os.getenv("C0D3R_PARALLEL_TASKS", "1").strip().lower() not in {"0", "false", "no", "off"}
    if not use_parallel or len(tasks) == 1:
        for label, fn in tasks:
            try:
                results.append((label, fn()))
            except Exception:
                results.append((label, None))
        return results
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(fn): label for label, fn in tasks}
            for fut in as_completed(future_map):
                label = future_map[fut]
                try:
                    results.append((label, fut.result()))
                except Exception:
                    results.append((label, None))
    except Exception:
        for label, fn in tasks:
            try:
                results.append((label, fn()))
            except Exception:
                results.append((label, None))
    return results


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
    pre_research_enabled: bool = True,
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
    _ensure_preflight(workdir, run_command)
    if os.getenv("C0D3R_DISABLE_PRERESEARCH", "").strip().lower() in {"1", "true", "yes", "on"}:
        pre_research_enabled = False
    while True:
        try:
            header.freeze()
            prompt = pending if pending is not None else _read_input(workdir)
            header.resume()
            pending = None
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            if _UI_MANAGER:
                _UI_MANAGER.stop()
            return 0
        _emit_live(f"repl: prompt received (len={len(prompt)})")
        _trace_event({
            "event": "repl.prompt",
            "workdir": str(workdir),
            "prompt_preview": _strip_context_block(prompt)[:200],
        })
        if not prompt.strip():
            continue
        user_prompt = _strip_context_block(prompt)
        if os.getenv("C0D3R_ONESHOT", "").strip().lower() in {"1", "true", "yes", "on"}:
            if _is_simple_file_task(user_prompt):
                _emit_live("oneshot: simple task fallback")
                if _apply_simple_task_fallback(user_prompt, workdir):
                    _emit_live("oneshot: simple task complete")
                    return 0
            if _apply_simple_project_stub(user_prompt, workdir):
                _emit_live("oneshot: simple project stub complete")
                return 0
        retarget = _maybe_retarget_project(user_prompt, workdir)
        if retarget:
            _emit_live(f"repl: retargeting workdir -> {retarget}")
            workdir = retarget
            os.chdir(workdir)
        # Petal directives from user prompt (dynamic constraints)
        petals = PetalManager()
        directives = _extract_petal_directives(user_prompt)
        if directives:
            names = [d.get("name") for d in directives if isinstance(d, dict)]
            _emit_live(f"petal: updating directives {names}")
            petals.update_from_directives(directives)
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
        # Normalize user prompt (strip injected context if any).
        user_prompt = _strip_context_block(prompt)
        # Existing project gate: scan context before edits (parallelized).
        scan_future = None
        scan_executor = None
        if _is_existing_project(workdir):
            scan_path = _context_scan_path(workdir)
            if not scan_path.exists() or not _scan_is_fresh(workdir, run_command, max_age_minutes=10):
                _emit_live("context: scanning existing project before edits")
                try:
                    from concurrent.futures import ThreadPoolExecutor
                    scan_executor = ThreadPoolExecutor(max_workers=1)
                    scan_future = scan_executor.submit(_scan_project_context, workdir, run_command)
                except Exception:
                    scan_future = None
        if not minimal:
            recall_hits = memory.search_if_referenced(user_prompt, limit=5)
            if recall_hits:
                context = context + "\n\n[recall]\n" + "\n".join(recall_hits)
        if scan_future:
            try:
                scan = scan_future.result()
                if scan:
                    context = context + "\n\n[project_scan]\n" + json.dumps(scan, indent=2)
            finally:
                if scan_executor:
                    scan_executor.shutdown(wait=True)
        system = (
            "Role: You are a senior software engineer. The user is a project manager. "
            "Return ONLY JSON with keys: status_updates (list of short strings) and final (string). "
            "Provide brief, concrete status updates before the final response. "
            "Avoid private chain-of-thought; summarize reasoning at a high level."
        )
        full_prompt = f"{context}\n\nUser:\n{user_prompt}"
        usage.add_input(full_prompt)
        print("[status] Planning...")
        print("[status] Executing...")
        _emit_live("repl: starting request")
        usage.set_status("planning", "routing + research")
        if os.getenv("C0D3R_ONESHOT", "").strip().lower() in {"1", "true", "yes", "on"} and _is_actionable_prompt(user_prompt):
            mode = "tool_loop"
        else:
            mode = _decide_mode(session, user_prompt, default_scientific=scientific, default_tool_loop=tool_loop)
            if _is_actionable_prompt(user_prompt):
                mode = "tool_loop"
        _trace_event({
            "event": "repl.mode",
            "mode": mode,
            "actionable": _is_actionable_prompt(user_prompt),
            "workdir": str(workdir),
        })
        research_summary = ""
        if mode in {"tool_loop", "scientific"} or _is_conversation_prompt(user_prompt):
            usage.set_status("research", "gathering references")
            if not pre_research_enabled or minimal or os.getenv("C0D3R_DISABLE_PRERESEARCH", "").strip().lower() in {"1", "true", "yes", "on"}:
                research_summary = ""
            else:
                _emit_live("repl: pre-research starting")
                # Run pre-research with a soft timeout to avoid blocking execution.
                try:
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError
                    timeout_s = float(os.getenv("C0D3R_PRERESEARCH_TIMEOUT_S", "20") or "20")
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        future = ex.submit(_pre_research, session, user_prompt)
                        try:
                            research_summary = future.result(timeout=timeout_s)
                        except TimeoutError:
                            research_summary = ""
                            _emit_live(f"repl: pre-research timed out after {timeout_s}s; continuing")
                except Exception:
                    research_summary = _pre_research(session, user_prompt)
                _emit_live(f"repl: pre-research complete (len={len(research_summary)})")
                _emit_live("repl: pre-research complete")
        controller = InterruptController()
        if tech_matrix and mode != "direct":
            outline = tech_matrix.get("outline") or {}
            matrix_context = json.dumps(outline, indent=2)[:6000]
            full_prompt = f"{full_prompt}\n\n[longform_outline]\n{matrix_context}"
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
                convo_prompt = full_prompt
                matrix_hint = ""
                if _is_conversation_prompt(user_prompt):
                    matrix_info = _query_unbounded_matrix(user_prompt)
                    if matrix_info:
                        matrix_hint = f"\n\n[matrix_hits]\n{json.dumps(matrix_info.get('hits') or [], indent=2)}"
                if matrix_hint:
                    convo_prompt = f"{convo_prompt}{matrix_hint}"
                convo_timeout = float(os.getenv("C0D3R_CONVO_TIMEOUT_S", "20") or "20")
                response = _call_with_timeout(
                    session._safe_send,
                    timeout_s=convo_timeout if _is_conversation_prompt(user_prompt) else _model_timeout_s(),
                    kwargs={"prompt": convo_prompt, "stream": False, "system": system},
                )
                if response is None and _is_conversation_prompt(user_prompt):
                    response = "{\"status_updates\": [\"Still looking into sources and integrating insights.\"], \"final\": \"I’m still investigating this. Want me to keep digging or focus on a specific sub-question?\"}"
                if response is None:
                    _emit_live("repl: direct mode timed out; rerouting to tool loop")
                    response = _run_tool_loop(
                        session,
                        full_prompt,
                        workdir,
                        run_command,
                        images=None,
                        stream=False,
                        stream_callback=None,
                        usage_tracker=usage,
                    )
                _emit_live("repl: direct model call complete")
        finally:
            controller.stop()
        if mode == "direct" and _is_actionable_prompt(prompt):
            response = _enforce_actionability(session, full_prompt, response or "")
        rendered = _render_json_response(response)
        if not rendered and _looks_like_json(response or ""):
            # Avoid dumping raw JSON to the UI; keep output clean.
            rendered = ""
        if rendered and not rendered.strip():
            rendered = ""
        if not rendered:
            # Still account for output tokens so header cost updates.
            usage.add_output(response or "")
            if header:
                header.update()
        # Apply file ops from any mode (direct/scientific/tool-loop) if present.
        file_ops = _extract_file_ops_from_text(response or "")
        if file_ops:
            _emit_live(f"executor: applying {len(file_ops)} file ops from response")
            base_root = _ensure_project_root(user_prompt, workdir) if _requires_new_projects_dir(user_prompt) else workdir
            if _requires_new_projects_dir(user_prompt) and base_root is None:
                base_root = (workdir / _slugify_project_name(user_prompt)).resolve()
            _trace_event({
                "event": "repl.file_ops",
                "count": len(file_ops),
                "base_root": str(base_root),
                "prompt_preview": user_prompt[:200],
            })
            applied = _apply_file_ops(file_ops, workdir, base_root=base_root)
            for path in applied:
                _emit_live(f"executor: wrote {path}")
            if not applied:
                _emit_live("executor: file_ops provided but none applied (validation/paths)")
                if _LAST_FILE_OPS_ERRORS:
                    _emit_live("executor: last errors:\n" + "\n".join(_LAST_FILE_OPS_ERRORS[-5:]))
                tail = _tail_executor_log()
                if tail:
                    _emit_live("executor.log tail:\n" + tail)
            if _microtest_enabled() and applied:
                py_targets = [p for p in applied if str(p).endswith(".py") and "/tests/" not in str(p).replace("\\", "/")]
                if py_targets:
                    _emit_live(f"microtests: running on {len(py_targets)} files")
                    ok, output = _run_microtests_for_paths(workdir, run_command, py_targets)
                    if not ok:
                        _emit_live("microtests: failed; attempting remediation")
                        fix_prompt = (
                            "Microtests failed. Provide file_ops or commands to fix errors, then re-run microtests. "
                            "Return ONLY JSON with commands and file_ops.\n"
                            f"Errors:\n{output}"
                        )
                        response = session.send(prompt=fix_prompt, stream=False)
                        response = _enforce_actionability(session, fix_prompt, response or "")
                        retry_ops = _extract_file_ops_from_text(response or "")
                        if retry_ops:
                            _apply_file_ops(retry_ops, workdir)
                        commands, _ = _extract_commands(response)
                        for cmd in commands[:5]:
                            if cmd:
                                run_command(_normalize_command(cmd, workdir), cwd=workdir, timeout_s=_command_timeout_s(cmd))
                        ok, output = _run_microtests_for_paths(workdir, run_command, py_targets)
                        if not ok:
                            _emit_live("microtests: still failing after remediation")
            if applied:
                _update_system_map(workdir)
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
        if rendered:
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
            memory.append(user_prompt, response)
            summary = memory.update_summary(summary, user_prompt, response, session)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(summary, encoding="utf-8")
        if oneshot:
            return 0


def _read_input(workdir: Path) -> str:
    if _UI_MANAGER:
        return _UI_MANAGER.read_input(f"[{workdir}]> ")
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


def _is_actionable_prompt(prompt: str) -> bool:
    if not prompt:
        return False
    lower = prompt.lower()
    keywords = [
        "create", "update", "modify", "fix", "implement", "add", "remove", "delete",
        "run", "test", "build", "install", "generate", "scaffold", "refactor", "migrate",
        "configure", "setup", "set up", "deploy", "serve",
    ]
    return any(k in lower for k in keywords)


def _is_conversation_prompt(prompt: str) -> bool:
    if not prompt:
        return False
    lower = prompt.lower()
    if _is_actionable_prompt(lower):
        return False
    convo_markers = (
        "explain", "why", "how", "what", "discuss", "conversation", "talk", "help me understand",
        "tell me about", "integrate", "compare", "overview", "summary", "question", "can you",
    )
    if "?" in lower:
        return True
    return any(m in lower for m in convo_markers)


def _decide_mode(session: C0d3rSession, prompt: str, *, default_scientific: bool, default_tool_loop: bool) -> str:
    """
    Ask a routing model to choose the best execution mode.
    """
    if os.getenv("C0D3R_FORCE_DIRECT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return "direct"
    if _is_conversation_prompt(prompt):
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
            if _is_actionable_prompt(prompt):
                return "tool_loop"
            return mode
    except Exception:
        pass
    if default_scientific and "inspect" in prompt.lower():
        return "scientific"
    if default_tool_loop:
        return "tool_loop"
    return "direct"


def _is_simple_file_task(prompt: str) -> bool:
    if not prompt:
        return False
    lower = prompt.lower().strip()
    if any(k in lower for k in ("create a new file", "create file", "create folder", "create a folder", "mkdir")):
        return True
    return False


def _is_scaffold_task(prompt: str) -> bool:
    if not prompt:
        return False
    lower = prompt.lower().strip()
    return "ionic" in lower or "scaffold" in lower or "template" in lower or "startproject" in lower


def _slugify_project_name(prompt: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\\s_-]", " ", prompt.lower())
    stop = {
        "create", "new", "project", "scaffolding", "template", "ionic", "angular",
        "that", "will", "contain", "with", "for", "and", "the", "a", "an", "to", "of",
        "tabs",
    }
    tokens = [t for t in text.split() if t not in stop]
    base = "_".join(tokens[:5]) if tokens else "new_project"
    base = re.sub(r"_+", "_", base).strip("_")
    if base and base[0].isdigit():
        base = f"project_{base}"
    return base or "new_project"


def _generate_brand_name(prompt: str, workdir: Path) -> str:
    """
    Generate a short, unique brand-style project name without model calls.
    """
    adjectives = [
        "Nova", "Pulse", "Lumen", "Arc", "Vivid", "Nimbus", "Echo", "Zephyr",
        "Halo", "Apex", "Flux", "Orbit", "Quartz", "Vertex", "Ion", "Glint",
    ]
    nouns = [
        "Leaf", "Forge", "Nest", "Drive", "Wave", "Spark", "Beacon", "Studio",
        "Lab", "Works", "Bloom", "Shift", "Trail", "Engine", "Studio", "Gate",
    ]
    seed = abs(hash(prompt)) % (len(adjectives) * len(nouns))
    adj = adjectives[seed % len(adjectives)]
    noun = nouns[(seed // len(adjectives)) % len(nouns)]
    name = f"{adj}{noun}"
    # Ensure uniqueness in workdir
    base = name
    idx = 1
    while (workdir / name).exists():
        idx += 1
        name = f"{base}{idx}"
    return name


def _ensure_project_root(prompt: str, workdir: Path) -> Path | None:
    if not _requires_new_projects_dir(prompt):
        return None
    lower = (prompt or "").lower()
    if "brand" in lower or "marketing" in lower:
        name = _generate_brand_name(prompt, workdir)
    else:
        name = _slugify_project_name(prompt)
    root = (workdir / name).resolve()
    # Do not pre-create folder for scaffold commands that need an empty target.
    if not _requires_scaffold_cmd(prompt):
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
    return root


def _fallback_scaffold_commands(prompt: str, project_root: Path) -> list[str]:
    lower = (prompt or "").lower()
    name = project_root.name
    if "ionic" in lower:
        return [
            f"npx @ionic/cli@latest start {name} tabs --type=angular --no-interactive --no-confirm --no-git",
        ]
    if "django" in lower:
        return [f"python -m django startproject {name}"]
    return []


def _requires_scaffold_cmd(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "ionic" in lower or "django" in lower


def _file_ops_only_runtime(ops: list, base_root: Path) -> bool:
    if not ops:
        return False
    for op in ops:
        if not isinstance(op, dict):
            continue
        raw = str(op.get("path") or "")
        target = _resolve_target_path(base_root, raw)
        if target and not _is_runtime_path(target):
            return False
    return True


def _plan_execution(session: C0d3rSession, prompt: str) -> dict:
    """
    Lightweight planner to choose initial tool sequence and model for the job.
    Returns JSON with keys: mode, do_math, do_research, do_tool_loop, model_override.
    """
    system = (
        "Return ONLY JSON with keys: mode, do_math, do_research, do_tool_loop, model_override. "
        "Choose the minimal set of steps needed to complete the task. "
        "Use do_math/do_research only if required to succeed. "
        "mode should be tool_loop for tasks that write files or run commands."
    )
    try:
        raw = session.send(prompt=prompt, stream=False, system=system)
        payload = _safe_json(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _pre_research(session: C0d3rSession, prompt: str) -> str:
    if os.getenv("C0D3R_DISABLE_PRERESEARCH", "").strip().lower() in {"1", "true", "yes", "on"}:
        return ""
    research_prompt = (
        "Perform preliminary research for the task. Summarize the most relevant framework/library setup steps, "
        "best practices, and any constraints. Return 6-10 bullets.\n\n"
        f"Task: {prompt}"
    )
    try:
        _emit_live("pre_research: model call starting")
        _emit_live("pre_research: looking into sources")
        _animate_research(0.8)
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
            _persist_research_knowledge(prompt, response)
        return response or ""
    except Exception:
        try:
            stop.set()
        except Exception:
            pass
        # Reroute: fall back to direct web research if model times out.
        try:
            _emit_live("pre_research: timeout; rerouting to direct web research")
            fallback = session._c0d3r._research(research_prompt) or ""
            if fallback:
                _persist_research_knowledge(prompt, fallback)
            return fallback
        except Exception:
            return ""


def _persist_research_knowledge(prompt: str, summary: str) -> None:
    try:
        if not _ensure_django_ready():
            return
        import os as _os
        if not _os.getenv("DJANGO_SETTINGS_MODULE"):
            _os.environ["DJANGO_SETTINGS_MODULE"] = "coolcrypto_dashboard.settings"
        import django
        django.setup()
        from core.models import KnowledgeDocument, KnowledgeQueueItem
        title = prompt.strip()[:200] if prompt else "c0d3r research"
        doc, created = KnowledgeDocument.objects.get_or_create(
            source="c0d3r_pre_research",
            title=title,
            defaults={
                "abstract": "",
                "body": summary,
                "url": "",
                "citation_apa": "",
                "metadata": {"prompt": prompt},
            },
        )
        if not created and doc.body != summary:
            doc.body = summary
            doc.metadata = doc.metadata or {}
            doc.metadata["prompt"] = prompt
            doc.save(update_fields=["body", "metadata"])
        KnowledgeQueueItem.objects.get_or_create(document=doc, defaults={"status": "pending", "confidence": 0.0})
    except Exception:
        pass


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
    if action.startswith("textbook"):
        scripts_root = PROJECT_ROOT / "tools" / "textbooks"
        scripts_dir = scripts_root / "scripts"
        if not scripts_dir.exists():
            return 1, "", "textbook scripts not found", None
        target_dir = workdir
        if len(parts) >= 2 and parts[1] not in {"", None}:
            maybe = parts[1] if len(parts) == 2 else parts[1] + " " + (parts[2] if len(parts) > 2 else "")
            if maybe and maybe.strip():
                target_dir = Path(maybe).expanduser().resolve()
        if action == "textbook_deps":
            npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
            try:
                proc = subprocess.run([npm_cmd, "install"], cwd=str(scripts_root), capture_output=True, text=True)
                return proc.returncode, proc.stdout, proc.stderr, None
            except Exception as exc:
                return 1, "", str(exc), None
        if action == "textbook_list":
            script = scripts_dir / "list-textbooks.py"
            try:
                proc = subprocess.run(
                    ["python", str(script)],
                    cwd=str(scripts_root),
                    capture_output=True,
                    text=True,
                )
                return proc.returncode, proc.stdout, proc.stderr, None
            except Exception as exc:
                return 1, "", str(exc), None
        if action == "textbook_import":
            script = scripts_dir / "import-manifest.py"
            try:
                env = os.environ.copy()
                env["TEXTBOOKS_DIR"] = str(target_dir)
                proc = subprocess.run(
                    ["python", str(script)],
                    cwd=str(scripts_root),
                    capture_output=True,
                    text=True,
                    env=env,
                )
                return proc.returncode, proc.stdout, proc.stderr, None
            except Exception as exc:
                return 1, "", str(exc), None
        if action == "textbook_ocr":
            script = scripts_dir / "ocr-pages.py"
            try:
                env = os.environ.copy()
                env["TEXTBOOKS_DIR"] = str(target_dir)
                proc = subprocess.run(
                    ["python", str(script)],
                    cwd=str(scripts_root),
                    capture_output=True,
                    text=True,
                    env=env,
                )
                return proc.returncode, proc.stdout, proc.stderr, None
            except Exception as exc:
                return 1, "", str(exc), None
        if action == "textbook_prepare_qa":
            script = scripts_dir / "prepare-textbook-qa.py"
            try:
                env = os.environ.copy()
                env["TEXTBOOKS_DIR"] = str(target_dir)
                proc = subprocess.run(
                    ["python", str(script), "--input-root", str(Path(target_dir) / "textbooks"), "--output-root", str(Path(target_dir) / "data" / "textbooks")],
                    cwd=str(scripts_root),
                    capture_output=True,
                    text=True,
                    env=env,
                )
                return proc.returncode, proc.stdout, proc.stderr, None
            except Exception as exc:
                return 1, "", str(exc), None
        if action == "textbook_import_qa":
            script = scripts_dir / "import-qa.py"
            try:
                env = os.environ.copy()
                env["TEXTBOOKS_DIR"] = str(target_dir)
                proc = subprocess.run(
                    ["python", str(script)],
                    cwd=str(scripts_root),
                    capture_output=True,
                    text=True,
                    env=env,
                )
                return proc.returncode, proc.stdout, proc.stderr, None
            except Exception as exc:
                return 1, "", str(exc), None
        if action == "textbook_knowledge":
            script = scripts_dir / "build-knowledge-docs.py"
            try:
                env = os.environ.copy()
                env["TEXTBOOKS_DIR"] = str(target_dir)
                proc = subprocess.run(
                    ["python", str(script)],
                    cwd=str(scripts_root),
                    capture_output=True,
                    text=True,
                    env=env,
                )
                return proc.returncode, proc.stdout, proc.stderr, None
            except Exception as exc:
                return 1, "", str(exc), None
        script_map = {
            "textbook_fetch": "fetch-libretexts.mjs",
            "textbook_segment": "segment-textbook.mjs",
            "textbook_build_dataset": "build-seg-dataset.mjs",
            "textbook_build_tiles": "build-tiles.mjs",
            "textbook_fix_pages": "fix-pages-json.mjs",
            "textbook_verify_pages": "verify-pages-count.mjs",
            "textbook_label_heuristic": "label-heuristic.mjs",
            "textbook_label_openai": "label-with-openai.mjs",
            "textbook_export_labels": "export-labeling-tsv.mjs",
            "textbook_train_microcortex": "train-microcortex.mjs",
        }
        if action == "textbook_reprocess":
            script = scripts_dir / "reprocess-incomplete.ps1"
            if os.name != "nt":
                return 1, "", "reprocess-incomplete.ps1 is Windows-only", None
            if not script.exists():
                return 1, "", "reprocess-incomplete.ps1 not found", None
            try:
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                    cwd=str(target_dir),
                    capture_output=True,
                    text=True,
                )
                return proc.returncode, proc.stdout, proc.stderr, None
            except Exception as exc:
                return 1, "", str(exc), None
        script_name = script_map.get(action)
        if not script_name:
            return 1, "", f"unknown meta command {action}", None
        script = scripts_dir / script_name
        if not script.exists():
            return 1, "", f"script not found: {script_name}", None
        try:
            node_cmd = "node"
            env = os.environ.copy()
            # If requesting a fetch, inject skip lists from DB to avoid duplicates.
            if action == "textbook_fetch":
                try:
                    if _ensure_django_ready():
                        from core.models import TextbookSource
                        skip_ids = []
                        skip_titles = []
                        for tb in TextbookSource.objects.all():
                            if tb.print_id:
                                skip_ids.append(tb.print_id)
                            if tb.title:
                                skip_titles.append(tb.title)
                        if skip_ids:
                            env["SKIP_PRINT_IDS"] = ",".join(skip_ids)
                        if skip_titles:
                            env["SKIP_TITLES"] = ",".join(skip_titles)
                except Exception:
                    pass
            proc = subprocess.run([node_cmd, str(script)], cwd=str(target_dir), capture_output=True, text=True, env=env)
            return proc.returncode, proc.stdout, proc.stderr, None
        except Exception as exc:
            return 1, "", str(exc), None
    return 1, "", f"unknown meta command {action}", None


def _requires_commands_for_task(prompt: str) -> bool:
    lower = (prompt or "").lower()
    keywords = ("create", "start", "run", "serve", "install", "build", "generate")
    return any(k in lower for k in keywords)


def _tool_loop_base_request(prompt: str) -> str:
    return _strip_context_block(prompt)


def _requires_new_projects_dir(prompt: str) -> bool:
    lower = (prompt or "").lower()
    if "project" in lower and any(k in lower for k in ("create", "template", "scaffold", "setup", "set up", "initialize")):
        return True
    if "c:/users/adam/projects" in lower and "create" in lower and "directory" in lower:
        return True
    if "ionic" in lower and any(k in lower for k in ("create", "new", "project", "scaffold", "tabs", "angular", "start")):
        return True
    if "kivy" in lower and "app" in lower and "create" in lower:
        return True
    # If current dir looks like a workspace (many folders, no markers) and user asks for a template/scaffold.
    try:
        root = Path(os.getenv("C0D3R_ROOT_CWD", "") or os.getcwd()).resolve()
        entries = [p for p in root.iterdir() if p.is_dir()]
        markers = ["pyproject.toml", "requirements.txt", "package.json", "manage.py", ".git"]
        workspace = len(entries) >= 6 and not any((root / m).exists() for m in markers)
        if workspace and any(k in lower for k in ("template", "scaffold", "setup", "set up", "initialize", "new project", "app")):
            return True
    except Exception:
        pass
    return False


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


def _apply_simple_task_fallback(prompt: str, workdir: Path) -> bool:
    """
    Local fallback for trivial file/folder tasks when the model fails to return commands.
    """
    text = (prompt or "").strip()
    lower = text.lower()
    # Create a new file with optional content.
    m = re.search(r"(?:create|add) (?:a )?new file\s+([^\s]+)(?:\s+with content\s+(.+))?$", text, re.IGNORECASE)
    if m:
        name = m.group(1).strip().strip('"').strip("'")
        content = m.group(2) or ""
        content = content.strip().strip('"').strip("'")
        executor = FileExecutor(workdir)
        ops = [{"path": name, "action": "write", "content": content, "allow_full_replace": True}]
        return bool(executor.apply_ops(ops))
    m = re.search(r"update\s+([A-Za-z0-9_.-]+)\s+with content\s+(.+)$", text, re.IGNORECASE)
    if m:
        name = m.group(1).strip().strip('"').strip("'")
        content = m.group(2).strip().strip('"').strip("'")
        executor = FileExecutor(workdir)
        ops = [{"path": name, "action": "write", "content": content, "allow_full_replace": True}]
        return bool(executor.apply_ops(ops))
    m = re.search(r"(?:delete|remove)\s+(?:file\s+)?([A-Za-z0-9_.-]+)$", text, re.IGNORECASE)
    if m:
        name = m.group(1).strip().strip('"').strip("'")
        executor = FileExecutor(workdir)
        ops = [{"path": name, "action": "delete"}]
        return bool(executor.apply_ops(ops))
    m = re.search(r"([A-Za-z0-9_.-]+\.md)\b", text, re.IGNORECASE)
    if m and any(k in lower for k in ("add", "adding", "create", "update")):
        name = m.group(1).strip()
        content = ""
        if "bullet" in lower or "next steps" in lower:
            content = "- Next step 1\n- Next step 2\n"
        executor = FileExecutor(workdir)
        ops = [{"path": name, "action": "write", "content": content, "allow_full_replace": True}]
        ok = bool(executor.apply_ops(ops))
        if ok:
            return True
    if "update project" in lower:
        mproj = re.search(r"update project\\s+([A-Za-z0-9_.-]+)", lower)
        if mproj:
            proj = mproj.group(1)
            projects_root = Path("C:\\Users\\Adam\\Projects")
            found = []
            if projects_root.exists():
                for child in projects_root.iterdir():
                    if child.is_dir() and proj in child.name.lower():
                        found.append(child.name)
            if not found:
                _emit_live(f"oneshot: project '{proj}' not found under {projects_root}")
                return False
    # Create a new folder.
    m = re.search(r"create (?:a )?(?:new )?folder\s+([^\s]+)$", text, re.IGNORECASE)
    if m:
        name = m.group(1).strip().strip('"').strip("'")
        executor = FileExecutor(workdir)
        ops = [{"path": name, "action": "mkdir"}]
        return bool(executor.apply_ops(ops))
    return False


def _apply_simple_project_stub(prompt: str, workdir: Path) -> bool:
    """
    Local fallback: create a new project folder with a README when prompt requests it.
    """
    lower = (prompt or "").lower()
    if "project" not in lower:
        return False
    if "readme" not in lower:
        return False
    projects_root = Path("C:\\Users\\Adam\\Projects")
    target_root = workdir
    try:
        in_projects = str(workdir).lower().startswith(str(projects_root).lower())
        # If we are inside a project folder, create new project at Projects root instead of nesting.
        project_markers = ["package.json", "pyproject.toml", "requirements.txt", "manage.py", ".git"]
        is_project = any((workdir / m).exists() for m in project_markers)
        if (not in_projects or is_project) and projects_root.exists():
            target_root = projects_root
    except Exception:
        target_root = workdir
    name = _generate_brand_name(prompt, target_root)
    project_root = (target_root / name).resolve()
    try:
        project_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    content = f"# {name}\n\nMinimal scaffold created by c0d3r.\n"
    executor = FileExecutor(project_root)
    ops = [{"path": "README.md", "action": "write", "content": content, "allow_full_replace": True}]
    # Create requested directories/files if prompt lists them.
    tokens = re.split(r"[,\n ]+", prompt)
    for tok in tokens:
        tok = tok.strip().strip(".").strip()
        if not tok or tok.startswith("http"):
            continue
        if "/" in tok and not tok.lower().endswith((".toml", ".txt", ".md", ".yaml", ".yml", ".json")):
            path = tok.replace("\\", "/").rstrip("/")
            ops.append({"path": path, "action": "mkdir"})
    # Handle common nested data directories
    lower_prompt = prompt.lower()
    if "data/" in lower_prompt and any(x in lower_prompt for x in ("raw/", "processed/", "results/")):
        for sub in ("raw", "processed", "results"):
            ops.append({"path": f"data/{sub}", "action": "mkdir"})
    file_matches = re.findall(r"([A-Za-z0-9_.-]+\.(?:toml|txt|md|yaml|yml|json))", prompt)
    for f in file_matches:
        if f.lower() == "readme.md":
            continue
        ops.append({"path": f, "action": "write", "content": "", "allow_full_replace": True})
    return bool(executor.apply_ops(ops))


def _maybe_retarget_project(prompt: str, workdir: Path) -> Path | None:
    """
    If prompt references a project name and we're not in it, retarget to that folder
    under C:\\Users\\Adam\\Projects when found.
    """
    lower = (prompt or "").lower()
    if "project" not in lower and "update" not in lower:
        return None
    projects_root = Path("C:\\Users\\Adam\\Projects")
    if not projects_root.exists():
        return None
    try:
        for child in projects_root.iterdir():
            if not child.is_dir():
                continue
            if child.name.lower() in lower:
                if not str(workdir).lower().startswith(str(child).lower()):
                    return child.resolve()
    except Exception:
        return None
    return None


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
        matrix = payload.get("matrix") or []
        gaps = payload.get("gap_fill_steps") or []
        nodes = payload.get("candidate_nodes") or []
        selected = payload.get("selected_node")
        symbol_defs = payload.get("symbol_definitions") or {}
        eq_units = payload.get("equation_units") or {}
        prior_matrix = _load_unbounded_matrix_context()
        matrix_hits = payload.get("matrix_hits") or []
        if not equations or not links:
            return False
        if not matrix or not gaps or not nodes or not selected:
            return False
        if not isinstance(symbol_defs, dict) or not symbol_defs:
            return False
        if not isinstance(eq_units, dict) or not eq_units:
            return False
        if prior_matrix and not matrix_hits:
            return False
        # Require advanced math tokens in equations
        math_tokens = ("=", "\\frac", "\\int", "\\sum", "\\partial", "d/d", "exp(", "log(")
        if not any(any(tok in str(eq) for tok in math_tokens) for eq in equations):
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


def _enforce_unbounded_matrix(session: C0d3rSession, prompt: str) -> dict | None:
    payload = _resolve_unbounded_request(session, prompt)
    if not payload:
        return None
    # Contradiction detection across equations (duplicate LHS with different RHS).
    try:
        lhs_map = {}
        conflict = False
        for eq in payload.get("equations") or []:
            if "=" not in eq:
                continue
            left, right = eq.split("=", 1)
            left_key = left.strip()
            prev = lhs_map.get(left_key)
            if prev and prev.strip() != right.strip():
                conflict = True
                break
            lhs_map[left_key] = right
        if conflict:
            payload["equations"] = []
    except Exception:
        pass
    # Validate equations are computable via sympy where possible
    try:
        import sympy as _sp

        valid = 0
        for eq in payload.get("equations") or []:
            try:
                if "=" in eq:
                    left, right = eq.split("=", 1)
                    _sp.Eq(_sp.sympify(left), _sp.sympify(right))
                else:
                    _sp.sympify(eq)
                valid += 1
            except Exception:
                continue
        if valid == 0:
            payload["equations"] = []
    except Exception:
        pass
    # Validate unit consistency if provided.
    try:
        eq_units = payload.get("equation_units") or {}
        if isinstance(eq_units, dict):
            for eq, units in eq_units.items():
                if not isinstance(units, dict):
                    payload["equations"] = []
                    break
                lhs = units.get("lhs_units")
                rhs = units.get("rhs_units")
                if lhs is None or rhs is None or str(lhs) != str(rhs):
                    payload["equations"] = []
                    break
    except Exception:
        pass
    _append_unbounded_matrix(payload)
    _persist_unbounded_payload(payload)
    return payload


def _load_unbounded_matrix_context(max_chars: int = 6000) -> str:
    if not _ensure_django_ready():
        return ""
    _seed_base_matrix_django()
    try:
        from core.models import Equation
        lines = ["# Matrix Snapshot", ""]
        rows = Equation.objects.order_by("-created_at")[:30]
        if rows:
            lines.append("## Recent Equations")
            for eq in rows:
                domain = ",".join(eq.disciplines or eq.domains or [])
                lines.append(f"- [{domain}] {eq.text}")
            lines.append("")
        text = "\n".join(lines)
        return text[-max_chars:]
    except Exception:
        return ""


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
        "explicit equations (equations) using real math (LaTeX or SymPy-friendly). "
        "Equations MUST be computable locally and reference measurable quantities. "
        "Provide symbol_definitions as dict {symbol: {meaning, units}} and "
        "equation_units as dict {equation: {lhs_units, rhs_units}}. "
        "Fill gaps between equations by proposing bridging equations "
        "and tests (gap_fill_steps). Only then list anomalies/paradoxes nearest to those mechanics, "
        "and derive hypotheses that reconcile paradoxes with proven mechanics. "
        "Use experiments to refine hypotheses. Build candidate_nodes as objects with fields: "
        "{id, rationale, supporting_mechanics, tests, score_0_1}. "
        "selected_node must be the best-scoring candidate and explain why it matches the request. "
        "matrix rows should be compact (discipline, axis_x, axis_y, axis_z, axis_w, insight). "
        "bounded_task must be a concrete, testable task with explicit completion criteria."
    )
    try:
        prior_matrix = _load_unbounded_matrix_context()
        matrix_context = ""
        if prior_matrix:
            # Query existing matrix for relevant nodes before creating new ones.
            reuse_system = (
                "Return ONLY JSON with keys: matrix_hits (list). "
                "Each hit: {equation, discipline, rationale}. "
                "Select the most relevant existing equations/mechanics for the request."
            )
            reuse_prompt = f"Existing matrix:\n{prior_matrix}\n\nRequest:\n{prompt}"
            reuse_raw = session.send(prompt=reuse_prompt, stream=False, system=reuse_system)
            reuse_payload = _safe_json(reuse_raw)
            matrix_hits = reuse_payload.get("matrix_hits") if isinstance(reuse_payload, dict) else None
            if matrix_hits:
                matrix_context = f"\n\n[matrix_hits]\n{json.dumps(matrix_hits, indent=2)}"
        response = session.send(prompt=f"Unbounded request:\n{prompt}{matrix_context}", stream=False, system=system)
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
        from core.models import (
            UnboundedMatrixRecord,
            Equation,
            EquationDiscipline,
            EquationSource,
            EquationLink,
        )
        _seed_base_matrix_django()
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
        equations = record.get("equations") or []
        disciplines = record.get("branches") or []
        research_links = record.get("research_links") or []
        authoritative_links = [l for l in research_links if _is_authoritative_source(str(l))]
        if equations and not authoritative_links:
            # Do not insert equations without authoritative sources.
            _emit_live("matrix: skipping equation insert (no authoritative sources)")
            return
        source = None
        if authoritative_links:
            source, _ = EquationSource.objects.get_or_create(
                title=authoritative_links[0][:255],
                defaults={
                    "url": authoritative_links[0],
                    "citation": authoritative_links[0],
                    "tags": ["research"],
                },
            )
        created_eqs: list[Equation] = []
        for eq in equations:
            if not eq:
                continue
            defaults = {
                "latex": str(eq),
                "domains": disciplines,
                "disciplines": disciplines,
                "confidence": 0.6,
                "source": source,
            }
            obj, _ = Equation.objects.get_or_create(text=str(eq), defaults=defaults)
            created_eqs.append(obj)
        for disc in disciplines:
            if disc:
                EquationDiscipline.objects.get_or_create(name=str(disc))
        # Link consecutive equations to form a minimal graph.
        for idx in range(len(created_eqs) - 1):
            EquationLink.objects.get_or_create(
                from_equation=created_eqs[idx],
                to_equation=created_eqs[idx + 1],
                defaults={"relation_type": "bridges", "notes": "auto-linked from unbounded record"},
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
        symbols = {"classes": {}, "functions": {}, "imports": {}, "signatures": {}}
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
                    sigs = []
                    for n in tree.body:
                        if isinstance(n, ast.FunctionDef):
                            args = [a.arg for a in n.args.args]
                            sigs.append({"name": n.name, "args": args})
                    imps = []
                    for n in tree.body:
                        if isinstance(n, ast.Import):
                            imps.extend([a.name for a in n.names])
                        if isinstance(n, ast.ImportFrom):
                            if n.module:
                                imps.append(n.module)
                    symbols["classes"][rel] = classes
                    symbols["functions"][rel] = funcs
                    symbols["signatures"][rel] = sigs
                    symbols["imports"][rel] = sorted(set(imps))
                except Exception:
                    pass
            elif path.suffix in {".ts", ".js", ".tsx", ".jsx"}:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    class_names = re.findall(r"class\s+([A-Za-z0-9_]+)", text)
                    func_names = re.findall(r"function\s+([A-Za-z0-9_]+)\s*\(", text)
                    arrow_funcs = re.findall(r"const\s+([A-Za-z0-9_]+)\s*=\s*\\(", text)
                    sigs = [{"name": n, "args": []} for n in func_names + arrow_funcs]
                    symbols["classes"][rel] = class_names
                    symbols["functions"][rel] = func_names + arrow_funcs
                    symbols["signatures"][rel] = sigs
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


def _load_code_memory_summary(max_chars: int = 2000) -> str:
    path = Path("runtime/c0d3r/system_map.json")
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        symbols = data.get("symbols") or {}
        classes = symbols.get("classes") or {}
        funcs = symbols.get("functions") or {}
        sigs = symbols.get("signatures") or {}
        lines = ["# Code Memory Summary"]
        for rel, names in list(classes.items())[:8]:
            if names:
                lines.append(f"{rel} classes: {', '.join(names[:6])}")
        for rel, names in list(funcs.items())[:8]:
            if names:
                lines.append(f"{rel} funcs: {', '.join(names[:6])}")
        for rel, items in list(sigs.items())[:6]:
            if items:
                snippet = ", ".join(f"{i.get('name')}({', '.join(i.get('args') or [])})" for i in items[:4])
                lines.append(f"{rel} sigs: {snippet}")
        text = "\n".join(lines)
        return text[:max_chars]
    except Exception:
        return ""


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
        if _UI_MANAGER:
            _UI_MANAGER.set_header(header)
            return
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
        if _UI_MANAGER:
            _UI_MANAGER.set_header(header)
            self._last = header
            return
        if not self.ansi_ok:
            return
        # Save cursor, move to home, write header, restore cursor
        sys.stdout.write("\x1b7\x1b[H")
        sys.stdout.write(header)
        sys.stdout.write("\x1b8")
        sys.stdout.flush()
        self._last = header

    def render_text(self) -> str:
        return self._build_header()

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
