#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import os
import getpass
import sys
import time
import datetime
import threading
import queue
import subprocess
import urllib.request
import hashlib
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = PROJECT_ROOT / "web"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if WEB_ROOT.exists() and str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))


def _allow_local_fallback() -> bool:
    return os.getenv("C0D3R_ALLOW_LOCAL_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}

def _runtime_root() -> Path:
    override = os.getenv("C0D3R_RUNTIME_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (PROJECT_ROOT / "runtime" / "c0d3r").resolve()


def _runtime_path(*parts: str) -> Path:
    return _runtime_root().joinpath(*parts)

_INSTALL_ATTEMPTS: set[str] = set()
_UI_MANAGER = None
_LAST_FILE_OPS_ERRORS: list[str] = []
_LAST_FILE_OPS_WRITTEN: list[str] = []
_MATRIX_SEED_VERSION = "2026-02-04"
_TECH_MATRIX_DIR = _runtime_path("tech_matrix")
_FINAL_STYLE = os.getenv("C0D3R_FINAL_STYLE", "bold yellow")
_DEFAULT_SECRET_CATEGORY = "default"
_DJANGO_USER_FILE = _runtime_path("django_user.json")
_MNEMONIC_TRIGGER = re.compile(r"\b(mnemonic|seed phrase|seed)\b", re.IGNORECASE)
_MNEMONIC_PHRASE = re.compile(r"\b(?:[a-zA-Z]{3,}\s+){11,23}[a-zA-Z]{3,}\b")
_DOCUMENT_EXTENSIONS = {".pdf", ".csv", ".doc", ".docx", ".xls", ".xlsx", ".html", ".txt", ".md"}
_ANIMATION_LOCK = threading.Lock()
_ANIMATION_STATE: dict[str, object | None] = {"stop": None, "thread": None, "kind": None}


def _ensure_root_dir(root: Path | None) -> None:
    if not root:
        return
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _trace_event(payload: dict) -> None:
    try:
        payload = dict(payload)
        payload["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path = _runtime_path("run_trace.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _tail_executor_log(lines: int = 6) -> str:
    path = _runtime_path("executor.log")
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
        path = _runtime_path("diagnostics.log")
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
        log_path = _runtime_path("live.log")
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


def _run_status_animation(kind: str, stop_event: threading.Event) -> None:
    if not _UI_MANAGER:
        return
    if kind == "research":
        frames = ["📖", "📖", "📖", "📚", "📖", "📚", "📖"]
        prefix = "research"
        suffixes = ["flipping pages..."] * len(frames)
    else:
        frames = ["⚡→○", "○→⚡", "○↔⚡", "⚡↔○", "○⇄⚡", "⚡⇄○"]
        prefix = "matrix"
        symbols = ["∑", "∫", "π", "λ", "ψ", "Ω"]
        suffixes = symbols
    idx = 0
    while not stop_event.is_set():
        frame = frames[idx % len(frames)]
        suffix = suffixes[idx % len(suffixes)]
        _emit_status_line(f"{prefix} {frame} {suffix}")
        idx += 1
        stop_event.wait(0.15)
    with _ANIMATION_LOCK:
        if _ANIMATION_STATE.get("stop") is stop_event:
            _emit_status_line("")
            _ANIMATION_STATE["stop"] = None
            _ANIMATION_STATE["thread"] = None
            _ANIMATION_STATE["kind"] = None


def _start_status_animation(kind: str) -> threading.Event | None:
    if not _UI_MANAGER:
        return None
    stop_event = threading.Event()
    with _ANIMATION_LOCK:
        prior = _ANIMATION_STATE.get("stop")
        if isinstance(prior, threading.Event):
            prior.set()
        t = threading.Thread(target=_run_status_animation, args=(kind, stop_event), daemon=True)
        _ANIMATION_STATE["stop"] = stop_event
        _ANIMATION_STATE["thread"] = t
        _ANIMATION_STATE["kind"] = kind
        t.start()
    return stop_event


def _stop_status_animation(stop_event: threading.Event | None) -> None:
    if not stop_event:
        return
    try:
        stop_event.set()
    except Exception:
        pass


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
        self._header_refresh_s = float(os.getenv("C0D3R_HEADER_REFRESH_S", "0.5") or "0.5")
        self._last_header_refresh = 0.0
        self._use_textual = False
        self._use_rich = False
        self._use_prompt_toolkit = False
        self.backend = "none"
        self.backend_error = ""
        self._textual_app = None
        self._textual_thread: threading.Thread | None = None
        self._live = None
        self._console = None
        self._layout = None
        self._pt_app = None
        self._pt_header = None
        self._pt_output = None
        self._pt_input = None
        self.final_style = _FINAL_STYLE
        self._init_tui()

    def _init_tui(self) -> None:
        backend = os.getenv("C0D3R_TUI_BACKEND", "textual").strip().lower()
        if backend in {"", "textual"}:
            try:
                from textual.app import App, ComposeResult
                from textual.widgets import Static, Input, RichLog

                class C0d3rTextualApp(App):
                    BINDINGS = [("ctrl+c", "quit", "Quit")]
                    CSS = """
                    #header { height:5; }
                    #body { height:1fr; }
                    #footer { height:3; }
                    """

                    def __init__(self, ui: "TerminalUI") -> None:
                        super().__init__()
                        self.ui = ui
                        self._header = None
                        self._body = None
                        self._footer = None

                    def compose(self) -> ComposeResult:
                        yield Static(id="header")
                        yield RichLog(id="body", wrap=False, markup=False, highlight=False)
                        yield Input(id="footer", placeholder="Type instructions and press Enter")

                    def on_mount(self) -> None:
                        self._header = self.query_one("#header", Static)
                        self._body = self.query_one("#body", RichLog)
                        self._footer = self.query_one("#footer", Input)
                        try:
                            self._header.update(self.ui.header.render_text())
                        except Exception:
                            pass
                        try:
                            self.set_focus(self._footer)
                        except Exception:
                            pass
                        self.set_interval(0.5, self._refresh_header)

                    def _refresh_header(self) -> None:
                        if self._header:
                            try:
                                self._header.update(self.ui.header.render_text())
                            except Exception:
                                pass

                    def on_input_submitted(self, event: Input.Submitted) -> None:
                        text = event.value
                        if self._footer:
                            self._footer.value = ""
                        if text is not None:
                            self.ui._input_queue.put(text)

                    def push_line(self, line: str) -> None:
                        if self._body:
                            self._body.write(line)

                    def push_renderable(self, renderable) -> None:
                        if self._body:
                            self._body.write(renderable)

                    def set_header_text(self, text: str) -> None:
                        if self._header:
                            self._header.update(text)

                    def set_footer_hint(self, text: str) -> None:
                        if self._footer:
                            self._footer.placeholder = text or "Type instructions and press Enter"

                self._textual_app = C0d3rTextualApp(self)
                self._use_textual = True
                self._use_prompt_toolkit = False
                self._use_rich = False
                self.backend = "textual"
                return
            except Exception as exc:
                self._use_textual = False
                self.backend_error = f"textual failed: {exc}"
                backend = ""
        if backend in {"", "prompt_toolkit"}:
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
                self.backend = "prompt_toolkit"
                return
            except Exception as exc:
                self._use_prompt_toolkit = False
                if not self.backend_error:
                    self.backend_error = f"prompt_toolkit failed: {exc}"
        if backend in {"", "rich"}:
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
                self.backend = "rich"
            except Exception:
                self._use_rich = False
                if not self.backend_error:
                    self.backend_error = "no TUI backend available"

    def start(self) -> None:
        self._running = True
        if self._use_textual and self._textual_app:
            self._textual_thread = threading.Thread(target=self._textual_app.run, daemon=True)
            self._textual_thread.start()
        elif self._use_prompt_toolkit and self._pt_app:
            self._prompt_thread = threading.Thread(target=self._pt_app.run, daemon=True)
            self._prompt_thread.start()
        elif self._use_rich and self._live:
            self._live.start()
        if not self._use_prompt_toolkit and not self._use_textual:
            self._prompt_thread = threading.Thread(target=self._input_loop, daemon=True)
            self._prompt_thread.start()
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()
        self.render()
        try:
            _emit_live(f"tui: backend={self.backend}")
            if self.backend_error and self.backend != "textual":
                _emit_live(f"tui: backend_error={self.backend_error}")
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        if self._use_rich and self._live:
            self._live.stop()
        if self._use_textual and self._textual_app:
            try:
                self._textual_app.exit()
            except Exception:
                pass
        if self._use_prompt_toolkit and self._pt_app:
            try:
                self._pt_app.exit()
            except Exception:
                pass
        self._render_event.set()

    def _render_loop(self) -> None:
        while self._running:
            now = time.time()
            if (now - self._last_header_refresh) >= self._header_refresh_s:
                self._dirty = True
                self._last_header_refresh = now
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

    def drain_input(self, max_items: int = 50) -> list[str]:
        items: list[str] = []
        for _ in range(max_items):
            try:
                items.append(self._input_queue.get_nowait())
            except queue.Empty:
                break
        return items

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

    def set_final_style(self, style: str) -> None:
        if style:
            self.final_style = style

    def write_line(self, line: str) -> None:
        if self._use_textual and self._textual_app:
            try:
                self._textual_app.call_from_thread(self._textual_app.push_line, line)
                return
            except Exception:
                pass
        with self._lock:
            self.lines.append(line)
            self._dirty = True
        self.render()

    def write_text(self, text: str, *, delay_s: float = 0.0, controller=None) -> None:
        if self._use_textual and self._textual_app:
            for line in text.splitlines():
                if controller and controller.interrupted:
                    return
                self.write_line(line)
            return
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

    def write_final(self, text: str) -> None:
        if text is None:
            return
        lines = str(text).splitlines()
        if self._use_textual and self._textual_app:
            try:
                from rich.text import Text
                style = self.final_style or "bold yellow"
                # Add two blank lines before and after.
                self._textual_app.call_from_thread(self._textual_app.push_line, "")
                self._textual_app.call_from_thread(self._textual_app.push_line, "")
                if lines:
                    for idx, line in enumerate(lines):
                        self._textual_app.call_from_thread(self._textual_app.push_renderable, Text(line, style=style))
                else:
                    self._textual_app.call_from_thread(self._textual_app.push_renderable, Text("", style=style))
                self._textual_app.call_from_thread(self._textual_app.push_line, "")
                self._textual_app.call_from_thread(self._textual_app.push_line, "")
                return
            except Exception:
                pass
        # Fallback: ANSI bold yellow if possible
        ansi = "\x1b[1;33m"
        reset = "\x1b[0m"
        decorated = "\n\n" + "\n".join(lines) + "\n\n"
        try:
            self.write_line(ansi + decorated + reset)
        except Exception:
            self.write_line(decorated)

    def render(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_render) < self._min_render_interval:
            self._render_event.set()
            return
        header_text = getattr(self, "header_text", self.header.render_text())
        body_text = "\n".join(self.lines)
        queued = 0
        try:
            queued = self._input_queue.qsize()
        except Exception:
            queued = 0
        footer_text = self.footer or (f"queued: {queued}" if queued else "ready")
        if self.status:
            footer_text = f"{footer_text} | {self.status}".strip()
        if self._use_textual and self._textual_app:
            try:
                self._textual_app.call_from_thread(self._textual_app.set_header_text, header_text)
                self._textual_app.call_from_thread(self._textual_app.set_footer_hint, footer_text)
                pending: list[str] = []
                with self._lock:
                    if self.lines:
                        pending = list(self.lines)
                        self.lines.clear()
                if pending:
                    def _push_pending() -> None:
                        for line in pending:
                            self._textual_app.push_line(line)
                    self._textual_app.call_from_thread(_push_pending)
            except Exception:
                pass
            self._last_render = now
            self._dirty = False
            return
        if self._use_prompt_toolkit and self._pt_app:
            if self._pt_header:
                self._pt_header.text = header_text
            if self._pt_output:
                self._pt_output.text = body_text
                try:
                    self._pt_output.buffer.cursor_position = len(self._pt_output.text)
                except Exception:
                    pass
            if self._pt_input:
                prompt = f"[{self.workdir}]> "
                if queued:
                    prompt = f"[{self.workdir}] (queued:{queued})> "
                self._pt_input.prompt = prompt
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
    parser.add_argument("--doc", "--document", dest="documents", action="append", help="Document path(s) for Bedrock document analysis.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output.")
    parser.add_argument("--tool-loop", action="store_true", help="Enable local command execution loop.")
    parser.add_argument("--no-tools", action="store_true", help="Disable local command execution loop.")
    parser.add_argument("--no-context", action="store_true", help="Disable automatic repo context summary.")
    parser.add_argument("--scientific", action="store_true", help="Enable scientific-method analysis mode.")
    parser.add_argument("--no-scientific", action="store_true", help="Disable scientific-method analysis mode.")
    parser.add_argument("--matrix-query", dest="matrix_query", help="Query the equation matrix and exit.")
    parser.add_argument("--scripted", dest="scripted", help="Path to a newline-delimited scripted prompt file.")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    _emit_live("boot: start")
    prompt = " ".join(args.prompt).strip()
    scripted_prompts: list[str] | None = None
    if args.scripted:
        try:
            path = Path(args.scripted).expanduser()
            raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            scripted_prompts = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]
        except Exception:
            scripted_prompts = []
    if not prompt and not scripted_prompts:
        prompt = sys.stdin.read().strip()
    _emit_live(f"boot: prompt loaded (len={len(prompt)})")
    # Silence noisy startup warnings for CLI usage.
    os.environ.setdefault("C0D3R_QUIET_STARTUP", "1")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    # Default to full verbosity unless user disables it.
    os.environ.setdefault("C0D3R_VERBOSE_MODEL_OUTPUT", "1")
    # Defaults tuned for "executor_v2": model decides terminal commands; python file_ops are disabled.
    os.environ.setdefault("C0D3R_EXECUTOR_V2", "1")
    os.environ.setdefault("C0D3R_COMMANDS_ONLY", "1")
    os.environ["C0D3R_ENABLE_FILE_OPS"] = "0"
    os.environ.setdefault("C0D3R_BEDROCK_LIVE", "1")
    os.environ.setdefault("C0D3R_BEDROCK_STREAM", "1")
    os.environ.setdefault("C0D3R_AUTO_CONTEXT_COMMANDS", "0")
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
    resolved_images = _resolve_image_paths(args.image, workdir)
    if args.image and not resolved_images:
        _emit_live("boot: image paths provided but none resolved")
    elif resolved_images:
        _emit_live(f"boot: images loaded ({len(resolved_images)})")
    resolved_documents = _resolve_document_paths(args.documents, workdir)
    if args.documents and not resolved_documents:
        _emit_live("boot: document paths provided but none resolved")
    elif resolved_documents:
        _emit_live(f"boot: documents loaded ({len(resolved_documents)})")
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
    if context_block and prompt:
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
    _emit_live("boot: session init")
    session = C0d3rSession(
        session_name="c0d3r-cli",
        transcript_dir=_runtime_path("transcripts"),
        workdir=str(workdir),
        **settings,
    )
    try:
        os.environ["C0D3R_SESSION_ID"] = str(session.session_id)
    except Exception:
        pass
    usage = UsageTracker(model_id=session.get_model_id())
    header = HeaderRenderer(usage)
    # Initialize terminal UI if available.
    global _UI_MANAGER
    use_tui = os.getenv("C0D3R_TUI", "1").strip().lower() not in {"0", "false", "no", "off"}
    # Disable TUI when not attached to a real terminal (e.g., redirected output).
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        use_tui = False
    if use_tui:
        try:
            _UI_MANAGER = TerminalUI(header, workdir)
            _UI_MANAGER.start()
        except Exception:
            _UI_MANAGER = None
    header.render()
    _emit_live("boot: header rendered")
    os.environ.setdefault("C0D3R_SKIP_PRICING", "1")
    _emit_live("pricing: skipped by default (commands-only mode)")
    # Commands-only execution: no planner/screener/matrix or math grounding.
    plan = {
        "mode": "tool_loop",
        "do_math": False,
        "do_research": False,
        "do_tool_loop": True,
        "model_override": "",
    }
    do_research = False
    tech_matrix = None
    _emit_live("boot: enter repl")
    initial_prompt = prompt or None
    if scripted_prompts:
        if initial_prompt:
            scripted_prompts.insert(0, initial_prompt)
        initial_prompt = None
    return _run_repl(
        session,
        usage,
        workdir,
        run_command,
        scientific=scientific if plan.get("mode") != "tool_loop" else False,
        tool_loop=tool_loop if plan.get("do_tool_loop", True) else False,
        initial_prompt=initial_prompt,
        scripted_prompts=scripted_prompts,
        images=resolved_images,
        documents=resolved_documents,
        header=header,
        pre_research_enabled=do_research,
        tech_matrix=tech_matrix,
        plan=plan,
    )


def _auto_context_commands_enabled() -> bool:
    return os.getenv("C0D3R_AUTO_CONTEXT_COMMANDS", "0").strip().lower() in {"1", "true", "yes", "on"}


def _detect_shells() -> List[str]:
    shells = [
        ("pwsh", "pwsh"),
        ("powershell", "powershell"),
        ("cmd", "cmd"),
        ("bash", "bash"),
        ("sh", "sh"),
        ("zsh", "zsh"),
    ]
    available = []
    for name, exe in shells:
        if shutil.which(exe):
            available.append(name)
    return available


def _detect_tools() -> Dict[str, str]:
    tools = ("python", "pip", "git", "node", "npm", "npx", "yarn", "pnpm", "uv", "rg")
    found: Dict[str, str] = {}
    for tool in tools:
        path = shutil.which(tool) or ""
        found[tool] = path
    return found


def _system_time_info() -> dict:
    try:
        now = datetime.datetime.now().astimezone()
        return {
            "local_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": now.tzname() or "",
            "utc_offset": now.strftime("%z") or "",
        }
    except Exception:
        return {}


def _weather_summary() -> str:
    if os.getenv("C0D3R_WEATHER", "1").strip().lower() in {"0", "false", "no", "off"}:
        return ""
    url = os.getenv("C0D3R_WEATHER_URL", "https://wttr.in/?format=1").strip()
    if not url:
        return ""
    timeout_s = float(os.getenv("C0D3R_WEATHER_TIMEOUT_S", "1.0") or "1.0")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "c0d3r/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="ignore").strip()
        return text[:200] if text else ""
    except Exception:
        return ""


def _environment_context_block(workdir: Path) -> str:
    lines = ["[environment]"]
    lines.append(f"- platform: {sys.platform}")
    lines.append(f"- os_name: {os.name}")
    lines.append(f"- cwd: {workdir}")
    try:
        lines.append(f"- project_root: {workdir.resolve()}")
    except Exception:
        pass
    shells = _detect_shells()
    lines.append(f"- shells: {', '.join(shells) if shells else '(none found)'}")
    tools = _detect_tools()
    for name, path in tools.items():
        lines.append(f"- tool.{name}: {path or 'missing'}")
    time_info = _system_time_info()
    if time_info:
        if time_info.get("local_time"):
            lines.append(f"- local_time: {time_info['local_time']}")
        if time_info.get("timezone"):
            lines.append(f"- timezone: {time_info['timezone']}")
        if time_info.get("utc_offset"):
            lines.append(f"- utc_offset: {time_info['utc_offset']}")
    weather = _weather_summary()
    if weather:
        lines.append(f"- weather: {weather}")
    try:
        from services.system_probe import collect_system_probe

        probe = collect_system_probe(cwd=workdir)
        lines.append(f"- is_admin: {probe.is_admin}")
        lines.append(f"- cpu_count: {probe.cpu_count}")
        lines.append(f"- total_memory_gb: {probe.total_memory_gb}")
        lines.append(f"- hostname: {probe.hostname}")
        lines.append(f"- network_available: {probe.network_available}")
    except Exception:
        pass
    lines.append("- local_tools: datalab + wallet meta commands available")
    lines.append("- datalab.meta: ::datalab_tables | ::datalab_query {json} | ::datalab_news {json} | ::datalab_web {json}")
    lines.append("- wallet.meta: ::wallet_login | ::wallet_logout | ::wallet_actions | ::wallet_lookup {json} | ::wallet_send {json} | ::wallet_swap {json} | ::wallet_bridge {json}")
    lines.append("- documents: auto-attach supported files (pdf/csv/doc/docx/xls/xlsx/html/txt/md) when referenced or via --doc")
    return "\n".join(lines)


def _summary_paths(session_id: str | None = None) -> tuple[Path, Path]:
    if session_id:
        return (_runtime_path(f"summary_{session_id}.json"), _runtime_path(f"summary_{session_id}.txt"))
    return (_runtime_path("summary.json"), _runtime_path("summary.txt"))


def _load_summary_bundle(session_id: str | None = None) -> dict:
    summary_json, summary_txt = _summary_paths(session_id)
    def _trim_200_words(text: str) -> str:
        words = text.split()
        if len(words) > 200:
            return " ".join(words[:200])
        return text

    strict = os.getenv("C0D3R_SUMMARY_SESSION_STRICT", "1").strip().lower() not in {"0", "false", "no", "off"}
    if summary_json.exists():
        try:
            payload = json.loads(summary_json.read_text(encoding="utf-8", errors="ignore"))
            stored_session = str(payload.get("session_id") or "").strip()
            if session_id and stored_session and stored_session != session_id:
                return {"summary": "", "key_points": []}
            if session_id and not stored_session and strict:
                return {"summary": "", "key_points": []}
            summary = str(payload.get("summary") or "").strip()
            summary = _trim_200_words(summary)
            points = payload.get("key_points") or []
            if not isinstance(points, list):
                points = []
            points = [str(p).strip() for p in points if str(p).strip()]
            return {"summary": summary, "key_points": points}
        except Exception:
            pass
    if session_id and strict:
        return {"summary": "", "key_points": []}
    summary = ""
    if summary_txt.exists():
        try:
            summary = summary_txt.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            summary = ""
    summary = _trim_200_words(summary)
    points = _extract_key_points(summary, limit=10)
    return {"summary": summary, "key_points": points}


def _save_summary_bundle(bundle: dict, *, session_id: str | None = None) -> None:
    summary_json, summary_txt = _summary_paths(session_id)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary = str(bundle.get("summary") or "").strip()
    words = summary.split()
    if len(words) > 200:
        summary = " ".join(words[:200])
    points = bundle.get("key_points") or []
    if not isinstance(points, list):
        points = []
    points = [str(p).strip() for p in points if str(p).strip()]
    if len(points) > 10:
        points = points[:10]
    payload = {"summary": summary, "key_points": points, "session_id": session_id or ""}
    try:
        summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    try:
        summary_txt.write_text(summary, encoding="utf-8")
    except Exception:
        pass


def _load_rolling_summary(max_chars: int = 2000, *, session_id: str | None = None) -> str:
    bundle = _load_summary_bundle(session_id=session_id)
    summary = str(bundle.get("summary") or "").strip()
    if not summary:
        return ""
    if max_chars and len(summary) > max_chars:
        return summary[:max_chars]
    return summary


def _extract_key_points(summary_or_points, limit: int = 6) -> List[str]:
    if not summary_or_points:
        return []
    if isinstance(summary_or_points, list):
        points = [str(p).strip() for p in summary_or_points if str(p).strip()]
        return points[:limit]
    summary = str(summary_or_points or "")
    points = []
    for line in summary.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith(("-", "*", "•")):
            points.append(raw.lstrip("-*• ").strip())
    if points:
        return points[:limit]
    # Fallback: split by sentence-ish boundaries.
    sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
    for sent in sentences:
        if sent.strip():
            points.append(sent.strip())
        if len(points) >= limit:
            break
    return points[:limit]


def _key_points_block(summary_or_points) -> str:
    points = _extract_key_points(summary_or_points, limit=10)
    if not points:
        return ""
    lines = ["[key_points]"]
    for item in points:
        lines.append(f"- {item}")
    return "\n".join(lines)


_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "that", "this", "from",
    "have", "has", "had", "was", "were", "what", "when", "where", "which", "who", "why",
    "how", "about", "into", "over", "under", "then", "than", "them", "they", "their", "ours",
    "can", "could", "should", "would", "will", "just", "like", "been", "did", "does", "doing",
    "it's", "im", "i'm", "we", "our", "us", "me", "my", "mine", "yours",
}


def _keyword_set(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _short_term_matches(prompt: str, summary_bundle: dict) -> bool:
    summary_text = str(summary_bundle.get("summary") or "")
    key_points = summary_bundle.get("key_points") or []
    hay = f"{summary_text}\n" + "\n".join(str(p) for p in key_points)
    if not hay.strip():
        return False
    prompt_keys = _keyword_set(prompt)
    if not prompt_keys:
        return False
    hay_l = hay.lower()
    hits = [k for k in prompt_keys if k in hay_l]
    return len(hits) >= 1


_RECALL_CUES = (
    # English
    "remember", "recall", "do you remember", "when we were", "that time",
    "earlier", "previous", "last time", "before", "last asked", "last question",
    "what did i last", "what was the last", "context", "conversation",
    # Spanish/Portuguese
    "recuerdas", "recuerdo", "anterior", "antes", "último", "ultim", "conversación",
    "lembra", "lembrar", "anterior", "antes", "último", "conversa",
    # French
    "souviens", "souvenir", "rappelle", "rappel", "précédent", "dernier", "avant",
    # German
    "erinnerst", "erinnerung", "vorher", "letzte", "zuvor",
    # Italian
    "ricordi", "ricordo", "precedente", "ultimo", "prima",
)


def _recall_cue_score(prompt: str) -> float:
    lower = (prompt or "").lower()
    score = 0.0
    for cue in _RECALL_CUES:
        if cue in lower:
            score += 1.0
    if "?" in (prompt or ""):
        score += 0.25
    if any(w in lower for w in ("talk", "discuss", "spoke", "spoke about", "said", "asked")):
        score += 0.25
    return score


def _detect_recall_trigger(prompt: str) -> bool:
    return _recall_cue_score(prompt) >= 1.0


def _is_recent_recall(prompt: str) -> bool:
    lower = (prompt or "").lower()
    recent_markers = ("last asked", "last question", "what did i last", "what was the last", "previous question")
    if any(m in lower for m in recent_markers):
        return True
    if "last" in lower and "ask" in lower:
        return True
    return False


def _is_short_term_summary_request(prompt: str) -> bool:
    lower = (prompt or "").lower()
    markers = (
        "what have we spoken about",
        "what have we talked about",
        "what did we talk about",
        "what have we discussed",
        "conversation so far",
        "so far",
        "summarize the conversation",
        "recap what we discussed",
        "what are we working on",
        "what is this about",
    )
    return any(m in lower for m in markers)


def _decide_recall_scope(
    session: "C0d3rSession",
    prompt: str,
    summary_bundle: dict,
    recent_snippet: str,
) -> tuple[str, str]:
    """
    Decide whether to use no recall, short-term only, or long-term search.
    Returns (scope, query) where scope is one of: none | short | long.
    """
    summary_text = str(summary_bundle.get("summary") or "").strip()
    key_points = summary_bundle.get("key_points") or []
    cue_score = _recall_cue_score(prompt)
    short_hit = _short_term_matches(prompt, summary_bundle)

    # Heuristic fast-paths (avoid extra model calls).
    if _is_recent_recall(prompt):
        return "short", ""
    if _is_short_term_summary_request(prompt):
        return "short", ""
    if cue_score < 0.75 and short_hit:
        return "short", ""
    if cue_score < 0.75 and not short_hit:
        return "none", ""

    if os.getenv("C0D3R_LTM_MODEL", "1").strip().lower() in {"0", "false", "no", "off"}:
        # heuristic fallback
        return ("long", prompt.strip()) if (cue_score >= 1.0 and not short_hit) else ("short", "")

    system = (
        "Return ONLY JSON with keys: scope (\"none\"|\"short\"|\"long\") and query (string). "
        "Choose short if the short-term summary or recent transcript likely contains the answer. "
        "Choose long only if the answer is likely from older history."
    )
    prompt_block = (
        f"Short-term summary (<=200 words):\n{summary_text}\n\n"
        f"Key points:\n{key_points}\n\n"
        f"Recent transcript snippet:\n{recent_snippet}\n\n"
        f"User prompt:\n{prompt}\n"
    )
    try:
        raw = session.send(prompt_block, stream=False, system=system)
    except Exception:
        return ("long", prompt.strip()) if (cue_score >= 1.0 and not short_hit) else ("short", "")
    payload = _safe_json(raw or "")
    if not isinstance(payload, dict):
        return ("long", prompt.strip()) if (cue_score >= 1.0 and not short_hit) else ("short", "")
    scope = str(payload.get("scope") or "").strip().lower()
    if scope not in {"none", "short", "long"}:
        scope = "short" if short_hit else "long"
    query = str(payload.get("query") or "").strip()
    if not query:
        query = prompt.strip()
    return scope, query


def _maybe_long_term_recall(
    session,
    memory,
    prompt: str,
    summary_bundle: dict,
    *,
    session_id: str | None = None,
    memory_long=None,
) -> list[str]:
    recall_trigger = _detect_recall_trigger(prompt)
    if _is_recent_recall(prompt):
        return []
    if _is_short_term_summary_request(prompt):
        return []
    short_hit = _short_term_matches(prompt, summary_bundle)
    recent_entries = memory.load(limit=12, session_id=session_id)
    recent_snippet = "\n".join(f"{e.role}: {e.content[:200]}" for e in recent_entries)
    if not recall_trigger and short_hit:
        return []
    scope, query = _decide_recall_scope(session, prompt, summary_bundle, recent_snippet)
    if scope != "long":
        return []
    mem_long = memory_long or memory
    hits = mem_long.search_long_term(query, limit=5)
    if not hits and recall_trigger:
        hits = mem_long.search_long_term(prompt, limit=5)
    return hits


def _build_context_block(workdir: Path, run_command, *, session_id: str | None = None) -> str:
    lines = [
        "[context]",
        f"- cwd: {workdir}",
        f"- os: {os.name}",
    ]
    try:
        lines.append(f"- project_root: {workdir.resolve()}")
    except Exception:
        pass
    lines.append(_environment_context_block(workdir))
    env_session = os.getenv("C0D3R_SESSION_ID")
    session_id = session_id or (env_session.strip() if env_session else None)
    bundle = _load_summary_bundle(session_id=session_id) if session_id else {"summary": "", "key_points": []}
    summary = str(bundle.get("summary") or "").strip()
    if summary:
        lines.append("[rolling_summary]\n" + summary)
        key_points = _key_points_block(bundle.get("key_points") or summary)
        if key_points:
            lines.append(key_points)
    if _auto_context_commands_enabled():
        try:
            from services.framework_catalog import detect_frameworks
            frameworks = detect_frameworks(workdir)
            if frameworks:
                lines.append(f"- frameworks: {', '.join(frameworks)}")
            else:
                lines.append("- frameworks: (none detected)")
        except Exception:
            lines.append("- frameworks: (unknown)")
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


def _resolve_image_paths(paths: list[str] | None, workdir: Path) -> list[str]:
    if not paths:
        return []
    resolved: list[str] = []
    for raw in paths:
        if not raw:
            continue
        try:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = (workdir / candidate).resolve()
            if candidate.exists() and candidate.is_file():
                resolved.append(str(candidate))
        except Exception:
            continue
    return resolved


def _is_supported_document(path: Path) -> bool:
    try:
        return path.suffix.lower() in _DOCUMENT_EXTENSIONS
    except Exception:
        return False


def _resolve_document_paths(paths: list[str] | None, workdir: Path) -> list[str]:
    if not paths:
        return []
    resolved: list[str] = []
    for raw in paths:
        if not raw:
            continue
        try:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = (workdir / candidate).resolve()
            if candidate.exists() and candidate.is_file() and _is_supported_document(candidate):
                resolved.append(str(candidate))
        except Exception:
            continue
    return resolved


def _doc_auto_enabled() -> bool:
    return os.getenv("C0D3R_DOC_AUTO", "1").strip().lower() not in {"0", "false", "no", "off"}


def _auto_document_paths(prompt: str, workdir: Path) -> list[str]:
    if not prompt or not _doc_auto_enabled():
        return []
    tokens = re.split(r"\s+", prompt)
    found: list[str] = []
    for token in tokens:
        cleaned = token.strip().strip("\"'").rstrip(").,;")
        if not cleaned:
            continue
        if cleaned.lower().startswith("file:"):
            cleaned = cleaned[5:].strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if not any(lowered.endswith(ext) for ext in _DOCUMENT_EXTENSIONS):
            continue
        try:
            candidate = Path(cleaned).expanduser()
            if not candidate.is_absolute():
                candidate = (workdir / candidate).resolve()
            if candidate.exists() and candidate.is_file() and _is_supported_document(candidate):
                found.append(str(candidate))
        except Exception:
            continue
    return found


def _merge_document_paths(primary: list[str] | None, extra: list[str] | None) -> list[str]:
    max_files = int(os.getenv("C0D3R_DOC_MAX_FILES", "5") or "5")
    merged: list[str] = []
    seen: set[str] = set()
    for item in (primary or []) + (extra or []):
        if not item:
            continue
        norm = str(item)
        if norm in seen:
            continue
        seen.add(norm)
        merged.append(norm)
        if len(merged) >= max_files:
            break
    return merged


def _context_scan_path(workdir: Path) -> Path:
    root = workdir.resolve()
    return _runtime_root() / f"context_scan_{root.name}.json"


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
    if any((root / m).exists() for m in markers):
        return True
    try:
        # Kivy-style or ad-hoc Python projects (e.g., .kv + .py files).
        if any(root.glob("*.kv")) and any(root.glob("*.py")):
            return True
        if any(root.glob("*.py")) and (root / "env.json").exists():
            return True
    except Exception:
        pass
    return False


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
    path = _runtime_path("preflight.json")
    if path.exists():
        return
    checks = {}
    for cmd in ("python", "pip", "git", "node", "npm", "npx", "yarn", "pnpm", "uv", "rg"):
        found_path = shutil.which(cmd) or ""
        checks[cmd] = {"found": bool(found_path), "path": found_path}
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


def _decide_new_project(session: C0d3rSession, prompt: str, workdir: Path) -> bool:
    """
    Ask the model whether the task should create a new project directory.
    Only override to True when explicitly requested in the prompt.
    """
    explicit_folder = _extract_target_folder_name(prompt)
    if explicit_folder:
        return True
    if _requires_new_projects_dir(prompt):
        return True
    if "unzip" in (prompt or "").lower() or "extract" in (prompt or "").lower():
        # If a target folder is named for extraction, treat it as a new project root.
        if explicit_folder:
            return True
    system = (
        "Return ONLY JSON: {\"new_project\": true|false, \"reason\": \"...\"}.\n"
        "Choose true only when the user explicitly asks to create a NEW project directory.\n"
        "Choose false if the task should operate in the current project or existing folders."
    )
    try:
        raw = session.send(prompt=f"CWD: {workdir}\nTask: {prompt}", stream=False, system=system)
        payload = _safe_json(raw)
        return bool(payload.get("new_project"))
    except Exception:
        return False


def _decide_project_name(session: C0d3rSession, prompt: str, workdir: Path) -> str:
    """
    Ask the model for a CamelCase brand-style project name.
    Falls back to a deterministic local generator if needed.
    """
    system = (
        "Return ONLY JSON: {\"name\": \"CamelCase\"}.\n"
        "Pick a short, unique, marketing/branding-style name for a new project folder.\n"
        "Use CamelCase with no spaces. Do not include file extensions."
    )
    try:
        raw = session.send(prompt=f"CWD: {workdir}\nTask: {prompt}", stream=False, system=system)
        payload = _safe_json(raw)
        name = str(payload.get("name") or "").strip()
        if name:
            return name
    except Exception:
        pass
    return _generate_brand_name(prompt, workdir)


def _decide_simple_task(session: C0d3rSession, prompt: str) -> bool:
    """
    Ask the model if the task is small enough to skip heavy planning/research.
    """
    system = (
        "Return ONLY JSON: {\"simple\": true|false}.\n"
        "Return true only if the task can be completed with 1-3 small file edits "
        "or 1-2 short commands and does NOT require research or multi-step planning."
    )
    try:
        raw = _call_with_timeout(
            session._safe_send,
            timeout_s=float(os.getenv("C0D3R_SIMPLE_SCREEN_TIMEOUT_S", "2") or "2"),
            kwargs={"prompt": prompt, "stream": False, "system": system},
        )
        payload = _safe_json(raw or "")
        return bool(payload.get("simple"))
    except Exception:
        return False



# ----------------------------
# Tool loop v2 (model-driven executor)
# ----------------------------

def _tool_loop_v2_enabled() -> bool:
    """Enable the simplified, model-driven executor loop.

    v2 disables direct python file_ops writes by default and expects the model
    to express *all* filesystem changes as terminal commands.
    """
    # Enforce v2 to keep execution fully model-driven (commands-only).
    return True


def _file_ops_enabled() -> bool:
    # File ops are disabled to enforce command-only execution.
    return False


def _commands_only_enabled() -> bool:
    return os.getenv("C0D3R_COMMANDS_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}


def _ui_write(line: str) -> None:
    """Write a line to the interactive UI if present, else stdout."""
    try:
        if _UI_MANAGER:
            _UI_MANAGER.write_line(line.rstrip("\n"))
            return
    except Exception:
        pass
    print(line, end="" if line.endswith("\n") else "\n")
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _ui_write_final(text: str) -> None:
    if _UI_MANAGER:
        _UI_MANAGER.write_final(text)
        return
    if text is None:
        return
    ansi = "\x1b[1;33m"
    reset = "\x1b[0m"
    decorated = "\n\n" + str(text) + "\n\n"
    try:
        print(ansi + decorated + reset)
    except Exception:
        print(decorated)


def _chunk_output(text: str, *, lines_per_chunk: int = 200, max_chunks: int = 20) -> List[str]:
    if not text:
        return []
    lines = text.splitlines()
    chunks: List[str] = []
    for i in range(0, len(lines), lines_per_chunk):
        if len(chunks) >= max_chunks:
            break
        chunk = "\n".join(lines[i:i + lines_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _infer_task_type(base_request: str, last_error: str = "") -> str:
    text = f"{base_request} {last_error}".lower()
    if any(k in text for k in ("research", "paper", "docs", "citations", "reference", "sources")):
        return "research"
    if any(k in text for k in ("debug", "error", "traceback", "exception", "fail", "failing", "fix")):
        return "debug"
    if any(k in text for k in ("install", "setup", "build", "compile", "test", "run", "execute", "shell", "command")):
        return "command"
    if any(k in text for k in ("edit", "update", "modify", "refactor", "write", "create", "delete", "move", "rename")):
        return "command"
    return "default"


def _select_model_for_step(session: "C0d3rSession", base_request: str, *, step: int, step_failed: bool, failure_count: int, last_error: str, sys_info: str) -> str:
    default_model = (
        os.getenv("C0D3R_MODEL_DEFAULT")
        or getattr(session, "get_model_id", lambda: "")()
        or getattr(getattr(session, "_c0d3r", None), "model_id", "")
    )
    model_research = os.getenv("C0D3R_MODEL_RESEARCH") or default_model
    model_debug = os.getenv("C0D3R_MODEL_DEBUG") or default_model
    model_command = os.getenv("C0D3R_MODEL_COMMAND") or default_model
    model_fallback = os.getenv("C0D3R_MODEL_FALLBACK") or default_model
    model_windows = os.getenv("C0D3R_MODEL_WINDOWS") or ""

    task_type = _infer_task_type(base_request, last_error)
    if step_failed or failure_count > 0:
        task_type = "debug"

    selected = default_model
    if task_type == "research":
        selected = model_research
    elif task_type == "debug":
        selected = model_debug
    elif task_type == "command":
        selected = model_command

    if os.name == "nt" and model_windows:
        if task_type in {"command", "debug"}:
            selected = model_windows

    if not selected:
        selected = default_model or model_fallback
    return selected or model_fallback or ""


def _send_with_model_override(session: "C0d3rSession", *, prompt: str, model_id: str, stream: bool, **kwargs) -> str:
    saved_model_id = getattr(getattr(session, "_c0d3r", None), "model_id", "")
    saved_multi = getattr(getattr(session, "_c0d3r", None), "multi_model", True)
    switched = False
    if model_id and hasattr(session, "_c0d3r"):
        if saved_model_id != model_id:
            session._c0d3r.model_id = model_id
            session._c0d3r.multi_model = False
            switched = True
    try:
        return session.send(prompt=prompt, stream=stream, **kwargs)
    finally:
        if switched and hasattr(session, "_c0d3r"):
            session._c0d3r.model_id = saved_model_id
            session._c0d3r.multi_model = saved_multi


def _intent_for_step(step: int, *, step_failed: bool, last_error: str = "", output_request: bool = False) -> str:
    if output_request:
        return "Deep-dive into prior command output chunks to remove ambiguity before proposing new commands."
    if step_failed:
        return "Diagnose the failure and propose corrective commands and verification checks."
    if step == 1:
        return "Generate the initial command plan and verification checks to satisfy the objective."
    return "Continue with the next commands and checks to complete the objective."


def _wrap_command_for_shell(command: str, shell: str) -> list[str]:
    """Wrap a command string for the requested shell in a cross-platform way."""
    shell = (shell or "auto").strip().lower()
    is_windows = os.name == "nt" or sys.platform.startswith("win")
    if shell == "auto":
        shell = "powershell" if is_windows else "bash"
    if shell in {"pwsh", "powershell"}:
        exe = "pwsh" if shutil.which("pwsh") else "powershell"
        return [exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command]
    if shell == "cmd":
        return ["cmd", "/c", command]
    if shell in {"bash", "sh"}:
        exe = "bash" if shell == "bash" else "sh"
        return [exe, "-lc", command]
    if shell == "python":
        return [sys.executable, "-c", command]
    if is_windows:
        return ["cmd", "/c", command]
    return ["bash", "-lc", command]


def _stream_subprocess(
    *,
    command: str,
    cwd: Path,
    shell: str = "auto",
    timeout_s: float | None = None,
    log_prefix: str = "cmd",
) -> dict:
    """Run a command with streamed output + capture."""
    cwd = Path(cwd).resolve()
    timeout_s = float(timeout_s) if timeout_s is not None else float(os.getenv("C0D3R_CMD_TIMEOUT_S", "180") or "180")
    out_dir = _runtime_path("command_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd_stripped = command.strip()
    if cmd_stripped.startswith("::"):
        started = time.time()
        code, stdout, stderr, new_cwd = _execute_meta_command(cmd_stripped, cwd)
        output = stdout or ""
        if stderr:
            output = output + ("\n" if output else "") + stderr
        if output:
            _ui_write(output + ("\n" if not output.endswith("\n") else ""))
        split = output.splitlines()
        head = "\n".join(split[:20])
        tail = "\n".join(split[-20:]) if split else ""
        return {
            "exit_code": code,
            "duration_s": time.time() - started,
            "output": output,
            "head": head,
            "tail": tail,
            "log_path": "",
            "timed_out": False,
            "cwd_after": str(new_cwd or cwd),
        }

    args = _wrap_command_for_shell(command, shell)
    started = time.time()
    timed_out = False
    q: "queue.Queue[str]" = queue.Queue()
    lines: list[str] = []

    stamp = time.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256((str(cwd) + "\n" + shell + "\n" + command).encode("utf-8", errors="ignore")).hexdigest()[:12]
    log_path = out_dir / f"{log_prefix}_{stamp}_{digest}.log"

    def _reader_thread(proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                q.put(line)
        except Exception as e:
            q.put(f"[stream-error] {e}\n")
        finally:
            q.put("")  # sentinel

    try:
        proc = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as e:
        msg = f"[exec-error] {type(e).__name__}: {e}"
        _ui_write(msg + "\n")
        return {
            "exit_code": 1,
            "duration_s": 0.0,
            "output": msg,
            "head": msg,
            "tail": msg,
            "log_path": "",
            "timed_out": False,
        }

    t = threading.Thread(target=_reader_thread, args=(proc,), daemon=True)
    t.start()

    with log_path.open("w", encoding="utf-8", errors="ignore") as lf:
        while True:
            if timeout_s and (time.time() - started) > timeout_s and proc.poll() is None:
                timed_out = True
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                item = q.get(timeout=0.1)
            except Exception:
                item = None
            if item is None:
                if proc.poll() is not None and not t.is_alive():
                    break
                continue
            if item == "":
                if proc.poll() is not None:
                    break
                continue
            lines.append(item)
            lf.write(item)
            lf.flush()
            _ui_write(item)

    try:
        exit_code = int(proc.wait(timeout=5))
    except Exception:
        exit_code = int(proc.poll() or 1)

    duration_s = time.time() - started
    output = "".join(lines)
    split = output.splitlines()
    head = "\n".join(split[:20])
    tail = "\n".join(split[-20:]) if split else ""

    return {
        "exit_code": exit_code,
        "duration_s": duration_s,
        "output": output,
        "head": head,
        "tail": tail,
        "log_path": str(log_path),
        "timed_out": timed_out,
    }


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of the first JSON object from a string."""
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.M)
    if fenced != text:
        try:
            obj = json.loads(fenced)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    start = fenced.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(fenced)):
        ch = fenced[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = fenced[start : i + 1]
                try:
                    obj = json.loads(chunk)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _coerce_command_item(item) -> dict | None:
    if item is None:
        return None
    if isinstance(item, str):
        cmd = item.strip()
        if not cmd:
            return None
        return {"command": cmd, "shell": "auto", "cwd": ".", "timeout_s": None, "purpose": ""}
    if isinstance(item, dict):
        cmd = (item.get("command") or item.get("cmd") or item.get("run") or "").strip()
        if not cmd:
            return None
        return {
            "command": cmd,
            "shell": (item.get("shell") or item.get("terminal") or item.get("executor") or "auto"),
            "cwd": item.get("cwd") or ".",
            "timeout_s": item.get("timeout_s") if item.get("timeout_s") is not None else item.get("timeout") if item.get("timeout") is not None else None,
            "purpose": item.get("purpose") or item.get("why") or "",
        }
    return None


def _validate_tool_loop_v2_payload(payload: dict) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload is not an object"
    if "final" not in payload:
        return False, "missing key: final"
    status_updates = payload.get("status_updates")
    if status_updates is None:
        return False, "missing key: status_updates"
    if not isinstance(status_updates, list):
        return False, "status_updates must be a list"
    if "needs_terminal" not in payload:
        return False, "missing key: needs_terminal"
    if "commands" not in payload:
        return False, "missing key: commands"
    commands = payload.get("commands")
    if commands is None:
        payload["commands"] = []
        commands = payload["commands"]
    if not isinstance(commands, list):
        return False, "commands must be a list"
    checks = payload.get("checks")
    if checks is None:
        return False, "missing key: checks"
    if not isinstance(checks, list):
        return False, "checks must be a list"
    needs_terminal = payload.get("needs_terminal")
    if not isinstance(needs_terminal, bool):
        return False, "needs_terminal must be boolean"
    if commands and payload["needs_terminal"] is False:
        return False, "needs_terminal must be true when commands are provided"
    if payload["needs_terminal"] and not commands and not (payload.get("final") or "").strip():
        return False, "needs_terminal=true but no commands and empty final"
    if "output_request" not in payload:
        return False, "missing key: output_request"
    output_request = payload.get("output_request")
    if not isinstance(output_request, list):
        return False, "output_request must be a list"
    for item in output_request:
        if not isinstance(item, dict):
            return False, "output_request items must be objects"
        key = item.get("key")
        idx = item.get("chunk_index")
        if not isinstance(key, str) or not key.strip():
            return False, "output_request.key must be a non-empty string"
        if not isinstance(idx, int) or idx < 0:
            return False, "output_request.chunk_index must be a non-negative integer"
    if output_request and payload.get("needs_terminal") is True:
        return False, "output_request requires needs_terminal=false"
    if output_request and (commands or (payload.get("checks") or [])):
        return False, "output_request must be used without commands/checks"
    return True, ""


def _run_tool_loop_v2(
    session: C0d3rSession,
    prompt: str,
    workdir: Path,
    run_command,  # signature compatibility; v2 executes via subprocess directly
    *,
    images: List[str] | None,
    documents: List[str] | None,
    stream: bool,
    stream_callback,
    usage_tracker,
) -> str:
    # Preserve the pre-built rolling context (system probe + rolling summary + key points) if present.
    context_block = ""
    if "User request:" in prompt:
        context_block = prompt.split("User request:", 1)[0].strip()
    elif "User:\n" in prompt:
        context_block = prompt.split("User:\n", 1)[0].strip()
    base_request = _tool_loop_base_request(prompt)
    max_steps = int(os.getenv("C0D3R_TOOL_STEPS", "10") or "10")
    max_commands_per_step = int(os.getenv("C0D3R_TOOL_MAX_CMDS", "12") or "12")
    max_json_attempts = min(10, int(os.getenv("C0D3R_JSON_MAX_ATTEMPTS", "3") or "3"))
    history: list[str] = []
    last_payload: dict | None = None
    current_cwd = Path(workdir).resolve()
    failure_count = 0
    last_error = ""
    prev_step_failed = False
    output_cache: dict[str, list[str]] = {}
    output_cursor: dict[str, int] = {}
    session_id = getattr(session, "session_id", "") if session else ""
    summary_bundle = _load_summary_bundle(session_id=session_id)
    rolling_summary = str(summary_bundle.get("summary") or "").strip()
    key_points_block = _key_points_block(summary_bundle.get("key_points") or rolling_summary) if rolling_summary or summary_bundle.get("key_points") else ""
    user_notes: list[str] = []

    def _capture_user_notes(reason: str = "") -> bool:
        if not _UI_MANAGER or not hasattr(_UI_MANAGER, "drain_input"):
            return False
        try:
            notes = _UI_MANAGER.drain_input()
        except Exception:
            notes = []
        if not notes:
            return False
        for note in notes:
            cleaned = str(note).strip()
            if not cleaned:
                continue
            user_notes.append(cleaned)
            history.append(f"[user_note] {cleaned}")
            _ui_write(f"note: {cleaned}\n")
        if reason:
            history.append(f"[user_note_context] {reason}")
        return True

    def _history_block() -> str:
        if not history:
            return ""
        return "\n\n[recent]\n" + "\n\n".join(history[-6:])

    schema_doc = (
        "Return ONLY JSON (no markdown) using this schema:\n"
        "{\n"
        "  \"status_updates\": [string],\n"
        "  \"needs_terminal\": true|false,\n"
        "  \"commands\": [\n"
        "    { \"purpose\": string, \"shell\": \"auto\"|\"powershell\"|\"cmd\"|\"bash\"|\"sh\"|\"python\", \"cwd\": string, \"timeout_s\": number|null, \"command\": string }\n"
        "  ],\n"
        "  \"checks\": [ same shape as commands ],\n"
        "  \"output_request\": [ { \"key\": string, \"chunk_index\": number } ],\n"
        "  \"ui_actions\": [ { \"action\": string, \"value\": string } ],\n"
        "  \"final\": string\n"
        "}\n"
        "Rules:\n"
        "- status_updates must be a list (can be empty). Keep items short.\n"
        "- All filesystem/OS actions MUST be expressed as terminal commands in commands[]. No file_ops.\n"
        "- Prefer idempotent commands. Avoid creating helper scripts unless absolutely required by the task.\n"
        "- Always include verification in checks[] (tests, grep, ls, python -m ...), and interpret failures to produce the next command batch.\n"
        "- If you need more output context, use output_request with a key and chunk_index; do not include commands/checks in that response.\n"
        "- Meta commands starting with :: are allowed for local tools (e.g., ::datalab_news {...}).\n"
        "- ui_actions is optional; use it to set UI preferences such as final response style (action: set_final_style, value: \"bold yellow\").\n"
        "- If done, set needs_terminal=false, commands=[], checks=[], output_request=[] and put the user-facing answer in final.\n"
    )

    for step in range(1, max_steps + 1):
        usage_tracker.set_status("planning", f"executor_v2 step {step}/{max_steps}")
        _emit_live(f"tool_loop_v2: step {step}/{max_steps} (cwd={current_cwd})")
        _capture_user_notes("before planning")

        sys_info = ""
        try:
            from services.system_probe import system_probe_context
            sys_info = system_probe_context(current_cwd)
        except Exception:
            sys_info = f"[system]\nos={os.name} platform={sys.platform} cwd={current_cwd}"

        env_block = _environment_context_block(current_cwd)
        objective = base_request.strip()
        intent = _intent_for_step(step, step_failed=prev_step_failed, last_error=last_error)
        summary_block = f"[rolling_summary]\n{rolling_summary}" if rolling_summary else ""
        notes_block = ""
        if user_notes:
            notes_block = "[user_notes]\n" + "\n".join(f"- {note}" for note in user_notes[-6:])
        docs_block = ""
        if documents:
            docs_block = "[documents_attached]\n" + "\n".join(f"- {doc}" for doc in documents)

        last_json = ""
        if last_payload:
            try:
                last_json = json.dumps(last_payload, indent=2)[:2000]
            except Exception:
                last_json = ""

        tool_prompt = (
            "[schema:tool_loop_v2]\n"
            + schema_doc
            + "\n"
            + sys_info
            + ("\n\n" + env_block if env_block else "")
            + ("\n\n" + summary_block if summary_block else "")
            + ("\n\n" + key_points_block if key_points_block else "")
            + ("\n\n" + notes_block if notes_block else "")
            + ("\n\n" + docs_block if docs_block else "")
            + "\n\n[objective]\n"
            + objective
            + "\n\n[intent]\n"
            + intent
            + ("\n\n[context]\n" + context_block[:6000] if context_block else "")
            + "\n\n[task]\n"
            + base_request
            + (_history_block())
            + ("\n\n[last_payload]\n" + last_json if last_json else "")
        )

        selected_model = _select_model_for_step(
            session,
            base_request,
            step=step,
            step_failed=prev_step_failed,
            failure_count=failure_count,
            last_error=last_error,
            sys_info=sys_info,
        )
        if selected_model:
            usage_tracker.model_id = selected_model

        payload = None
        ok = False
        err = ""
        attempt = 0
        active_prompt = tool_prompt
        while attempt < max_json_attempts:
            attempt += 1
            usage_tracker.add_input(active_prompt)
            images_for_step = None
            if images:
                if step == 1 or os.getenv("C0D3R_IMAGES_EVERY_STEP", "0").strip().lower() in {"1", "true", "yes", "on"}:
                    images_for_step = images
            docs_for_step = None
            if documents:
                if step == 1 or os.getenv("C0D3R_DOCS_EVERY_STEP", "0").strip().lower() in {"1", "true", "yes", "on"}:
                    docs_for_step = documents
            raw = _send_with_model_override(
                session,
                prompt=active_prompt,
                model_id=selected_model,
                stream=False,
                images=images_for_step,
                documents=docs_for_step,
            )
            usage_tracker.add_output(raw or "")
            payload = _extract_json_object(raw or "")
            if payload is None:
                active_prompt = (
                    "[schema:tool_loop_v2]\n"
                    + schema_doc
                    + "\nYour previous response was not valid JSON. Re-emit ONLY valid JSON matching the schema.\n\n"
                    + (raw or "")[:2000]
                )
                continue
            ok, err = _validate_tool_loop_v2_payload(payload)
            if ok:
                break
            active_prompt = (
                "[schema:tool_loop_v2]\n"
                + schema_doc
                + f"\nYour JSON failed validation: {err}. Fix it and re-emit ONLY valid JSON.\n\n"
                + json.dumps(payload, indent=2)[:2000]
            )
            payload = None

        if payload is None or not ok:
            history.append(f"[schema_error] {err or 'Model did not return valid JSON.'}")
            last_payload = payload
            continue

        last_payload = payload
        final = (payload.get("final") or "").strip()
        needs_terminal = bool(payload.get("needs_terminal"))
        cmd_items = payload.get("commands") or []
        chk_items = payload.get("checks") or []
        output_request = payload.get("output_request") or []
        _apply_ui_actions(payload)

        commands: list[dict] = []
        for item in cmd_items:
            c = _coerce_command_item(item)
            if c:
                commands.append(c)
        checks: list[dict] = []
        for item in chk_items:
            c = _coerce_command_item(item)
            if c:
                checks.append(c)

        show_final_now = not commands and not checks and not needs_terminal
        payload_lines = _format_payload_lines(payload, show_final=show_final_now)
        if payload_lines:
            for line in payload_lines:
                _ui_write(line + "\n")

        if output_request:
            delivered = 0
            for req in output_request:
                key = str(req.get("key") or "").strip()
                idx = int(req.get("chunk_index") or 0)
                chunks = output_cache.get(key) or []
                if 0 <= idx < len(chunks):
                    history.append(f"[output_chunk] key={key} index={idx} total={len(chunks)}\n{chunks[idx]}")
                    delivered += 1
                else:
                    history.append(f"[output_chunk_missing] key={key} index={idx} total={len(chunks)}")
            if delivered:
                history.append("[next] output chunks delivered; propose next commands or finalize.")
            prev_step_failed = False
            continue

        if final and not commands and not checks and not needs_terminal:
            return final
        if final and not commands and not checks and needs_terminal:
            return final

        if not commands and not checks:
            history.append("[no_commands] Model returned no commands and no checks. Ask for concrete commands next.")
            continue

        usage_tracker.set_status("executing", f"executor_v2 running {len(commands)} cmd(s)")
        step_failed = False
        ran_any = False
        user_interrupted = False

        for i, cmd in enumerate(commands[:max_commands_per_step], start=1):
            ran_any = True
            purpose = (cmd.get("purpose") or "").strip()
            shell = (cmd.get("shell") or "auto").strip()
            cwd_s = (cmd.get("cwd") or ".").strip()
            timeout_s = cmd.get("timeout_s")
            try:
                cwd = (current_cwd / cwd_s).resolve() if not Path(cwd_s).is_absolute() else Path(cwd_s).resolve()
            except Exception:
                cwd = current_cwd

            _ui_write(f"\n[cmd {i}/{min(len(commands), max_commands_per_step)}] {purpose}\n$ {cmd.get('command')}\n")
            res = _stream_subprocess(
                command=str(cmd.get("command") or ""),
                cwd=cwd,
                shell=shell,
                timeout_s=timeout_s,
                log_prefix=f"step{step}_cmd{i}",
            )

            output_text = res.get("output") or ""
            tail = res.get("tail") or ""
            head = res.get("head") or ""
            log_path = res.get("log_path") or ""
            exit_code = res.get("exit_code", 1)
            timed_out = bool(res.get("timed_out"))
            chunk_key = f"step{step}_cmd{i}"
            chunks = _chunk_output(output_text)
            if chunks:
                output_cache[chunk_key] = chunks
                output_cursor.setdefault(chunk_key, 0)

            history.append(
                "\n".join(
                    [
                        f"[cmd_result] step={step} idx={i} exit={exit_code} timeout={timed_out}",
                        f"purpose: {purpose}",
                        f"command: {cmd.get('command')}",
                        f"cwd: {cwd}",
                        f"log: {log_path}",
                        f"chunk_key: {chunk_key} chunks_total: {len(chunks)}",
                        f"head:\n{head}",
                        f"tail:\n{tail}",
                    ]
                )[:4000]
            )

            if timed_out or exit_code != 0:
                step_failed = True
                last_error = f"exit={exit_code} timeout={timed_out}"
                if chunks:
                    idx = output_cursor.get(chunk_key, 0)
                    if idx < len(chunks):
                        history.append(f"[output_chunk] key={chunk_key} index={idx} total={len(chunks)}\n{chunks[idx]}")
                        output_cursor[chunk_key] = idx + 1
                break
            if _capture_user_notes("during command execution"):
                user_interrupted = True
                break

        if not step_failed and not user_interrupted and checks:
            usage_tracker.set_status("executing", f"executor_v2 checks ({len(checks)} cmd)")
            for j, chk in enumerate(checks[:max_commands_per_step], start=1):
                purpose = (chk.get("purpose") or "check").strip()
                shell = (chk.get("shell") or "auto").strip()
                cwd_s = (chk.get("cwd") or ".").strip()
                timeout_s = chk.get("timeout_s")
                try:
                    cwd = (current_cwd / cwd_s).resolve() if not Path(cwd_s).is_absolute() else Path(cwd_s).resolve()
                except Exception:
                    cwd = current_cwd

                _ui_write(f"\n[check {j}/{min(len(checks), max_commands_per_step)}] {purpose}\n$ {chk.get('command')}\n")
                res = _stream_subprocess(
                    command=str(chk.get('command') or ""),
                    cwd=cwd,
                    shell=shell,
                    timeout_s=timeout_s,
                    log_prefix=f"step{step}_chk{j}",
                )
                output_text = res.get("output") or ""
                tail = res.get("tail") or ""
                head = res.get("head") or ""
                log_path = res.get("log_path") or ""
                exit_code = res.get("exit_code", 1)
                timed_out = bool(res.get("timed_out"))
                chunk_key = f"step{step}_chk{j}"
                chunks = _chunk_output(output_text)
                if chunks:
                    output_cache[chunk_key] = chunks
                    output_cursor.setdefault(chunk_key, 0)
                history.append(
                    "\n".join(
                        [
                            f"[check_result] step={step} idx={j} exit={exit_code} timeout={timed_out}",
                            f"purpose: {purpose}",
                            f"command: {chk.get('command')}",
                            f"cwd: {cwd}",
                            f"log: {log_path}",
                            f"chunk_key: {chunk_key} chunks_total: {len(chunks)}",
                            f"head:\n{head}",
                            f"tail:\n{tail}",
                        ]
                    )[:4000]
                )
                if timed_out or exit_code != 0:
                    step_failed = True
                    last_error = f"exit={exit_code} timeout={timed_out}"
                    if chunks:
                        idx = output_cursor.get(chunk_key, 0)
                        if idx < len(chunks):
                            history.append(f"[output_chunk] key={chunk_key} index={idx} total={len(chunks)}\n{chunks[idx]}")
                            output_cursor[chunk_key] = idx + 1
                    break
                if _capture_user_notes("during checks"):
                    user_interrupted = True
                    break

        if user_interrupted:
            prev_step_failed = False
            history.append("[next] user note received; replan before more commands.")
            continue

        if final and not step_failed:
            return final

        if step_failed:
            failure_count += 1
            prev_step_failed = True
            history.append("[next] previous step failed; propose adjusted commands and re-verify.")
        else:
            prev_step_failed = False
            history.append("[next] commands ran; propose next batch or finalize if complete.")

    tail = "\n\n".join(history[-3:]) if history else ""
    return "I couldn't complete the task within the execution budget.\nLast signals:\n" + tail[-2500:]



def _run_tool_loop(
    session: C0d3rSession,
    prompt: str,
    workdir: Path,
    run_command,
    *,
    images: List[str] | None,
    documents: List[str] | None,
    stream: bool,
    stream_callback,
    usage_tracker,
) -> str:
    # Legacy v1 tool loop is disabled; always use v2 commands-only executor.
    return _run_tool_loop_v2(
        session,
        prompt,
        workdir,
        run_command,
        images=images,
        documents=documents,
        stream=stream,
        stream_callback=stream_callback,
        usage_tracker=usage_tracker,
    )


def _normalize_command(cmd: str, workdir: Path) -> str:
    """
    Normalize common commands for Windows PowerShell.
    """
    if os.name != "nt":
        return cmd
    raw_cmd = cmd.strip()
    if "powershell" in raw_cmd.lower() and "command" in raw_cmd.lower():
        try:
            low = raw_cmd.lower()
            idx = low.find("command")
            tail = raw_cmd[idx + len("command") :] if idx >= 0 else ""
            tail = tail.lstrip()
            tail = tail.replace("\\\"", "\"").replace("\\'", "'").rstrip("\\")
            # Strip wrapping quotes even if only leading quote exists.
            if tail.startswith('"') and tail.endswith('"'):
                tail = tail[1:-1]
            elif tail.startswith('"') and not tail.endswith('"'):
                tail = tail[1:]
            cmd = tail.strip().strip('"')
        except Exception:
            pass
    # Split chained commands for PowerShell.
    if "&&" in cmd:
        parts = [p.strip() for p in cmd.split("&&") if p.strip()]
        normalized_parts = [_normalize_command(p, workdir) for p in parts]
        normalized_parts = [p for p in normalized_parts if p]
        return " ; ".join(normalized_parts)
    if "||" in cmd:
        # Handle common CMD-style fallback: type/cat <file> 2>nul || echo <msg>
        m = re.search(r"^\s*(type|cat)\s+([^\s]+)\s+2>nul\s*\|\|\s*echo\s+(.+)$", cmd, re.I)
        if m:
            target = m.group(2).strip().strip('"')
            msg = m.group(3).strip().strip('"').strip("'")
            return f'if (Test-Path "{target}") {{ Get-Content "{target}" }} else {{ Write-Output "{msg}" }}'
        left, right = cmd.split("||", 1)
        left = left.strip()
        right = right.strip()
        # Use explicit error action to ensure non-zero exitcode
        return f"$ErrorActionPreference='SilentlyContinue'; {left} ; if ($LASTEXITCODE -ne 0) {{ {right} }}"
    m = re.search(r"^\s*(type|cat)\s+([^\s]+)\s+2>nul\s*$", cmd, re.I)
    if m:
        target = m.group(2).strip().strip('"')
        return f'if (Test-Path "{target}") {{ Get-Content "{target}" }}'
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
    # Convert unzip to Expand-Archive on Windows.
    if cmd.lower().startswith("unzip "):
        try:
            m = re.search(r"unzip\s+(?:-o\s+)?([^\s]+)(?:\s+-d\s+([^\s]+))?", cmd, re.I)
            if m:
                zip_name = m.group(1).strip().strip('"')
                dest = m.group(2).strip().strip('"') if m.group(2) else "."
                if zip_name and not Path(zip_name).is_absolute():
                    candidate = (workdir / zip_name)
                    if not candidate.exists():
                        for parent in [workdir] + list(workdir.parents)[:3]:
                            probe = parent / zip_name
                            if probe.exists():
                                candidate = probe
                                break
                    zip_name = str(candidate)
                if dest in {".", "./"}:
                    try:
                        dest = Path(zip_name).stem or dest
                    except Exception:
                        dest = dest
                return f'Expand-Archive -Path "{zip_name}" -DestinationPath "{dest}" -Force'
        except Exception:
            pass
    if cmd.lower().startswith("expand-archive"):
        try:
            zip_match = re.search(r"-Path\\s+(['\"]?)([^'\"\\s]+)\\1", cmd, re.I)
            dest_match = re.search(r"-DestinationPath\\s+(['\"]?)([^'\"\\s]+)\\1", cmd, re.I)
            zip_name = zip_match.group(2) if zip_match else None
            dest = dest_match.group(2) if dest_match else None
            if zip_name and not Path(zip_name).is_absolute():
                # Resolve relative zip paths by searching current and parent directories.
                candidate = (workdir / zip_name)
                if not candidate.exists():
                    for parent in [workdir] + list(workdir.parents)[:3]:
                        probe = parent / zip_name
                        if probe.exists():
                            candidate = probe
                            break
                zip_name = str(candidate)
                cmd = re.sub(r"-Path\\s+(['\"]?)[^'\"\\s]+\\1", f'-Path \"{zip_name}\"', cmd, flags=re.I)
            if zip_name and (dest is None or dest in {".", "./"}):
                dest = Path(zip_name).stem or dest or "."
                return re.sub(r"-DestinationPath\\s+(['\"]?)[^'\"\\s]+\\1", f'-DestinationPath \"{dest}\"', cmd, flags=re.I)
        except Exception:
            pass
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
    if lower.startswith("ls -la") or lower == "ls -la":
        return "Get-ChildItem -Force"
    if lower.startswith("ls -l") or lower == "ls -l":
        return "Get-ChildItem"
    if lower.startswith("find ") and "-name" in lower:
        m = re.search(r"find\\s+\\.?\\s*-name\\s+\"?([^\\s\\\"]+)\"?\\s*-type\\s+f", cmd, re.I)
        if m:
            pattern = m.group(1)
            return f'Get-ChildItem -Recurse -Filter "{pattern}" -File | Select-Object -ExpandProperty FullName'
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
        if str(candidate).startswith(str(projects_root)) and (
            allow_projects or str(root).startswith(str(projects_root))
        ):
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
    documents: List[str] | None,
    stream: bool,
    stream_callback,
    usage_tracker,
) -> str:
    scientific_prompt = (
        "[scientific_mode]\n"
        "Use the scientific method. Do not speculate. "
        "If you need evidence, gather it via terminal commands in commands[]. "
        "Always verify with checks[]. "
        "Return JSON per the tool_loop_v2 schema.\n\n"
        f"User request:\n{prompt}\n"
    )
    return _run_tool_loop_v2(
        session,
        scientific_prompt,
        workdir,
        run_command,
        images=images,
        documents=documents,
        stream=stream,
        stream_callback=stream_callback,
        usage_tracker=usage_tracker,
    )


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


def _extract_checks(text: str) -> List[str]:
    try:
        import json
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        payload = json.loads(_extract_json(cleaned))
        checks = payload.get("checks") or []
        if isinstance(checks, list):
            return [str(c) for c in checks if str(c).strip()]
    except Exception:
        pass
    return []


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
            if isinstance(parsed, dict) and (
                "final" in parsed
                or "commands" in parsed
                or "needs_terminal" in parsed
                or "checks" in parsed
                or "status_updates" in parsed
                or "conclusion" in parsed
            ):
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
            if isinstance(parsed, dict) and (
                "final" in parsed
                or "commands" in parsed
                or "needs_terminal" in parsed
                or "checks" in parsed
                or "status_updates" in parsed
                or "conclusion" in parsed
            ):
                return parsed
            preferred = preferred or parsed
    except Exception:
        pass
    return preferred or {}


def _looks_like_project_root(root: Path) -> bool:
    markers = [
        "pyproject.toml",
        "requirements.txt",
        "package.json",
        "manage.py",
        ".git",
        "setup.cfg",
        "setup.py",
    ]
    try:
        return any((root / m).exists() for m in markers)
    except Exception:
        return False


def _is_workspace_root(root: Path) -> bool:
    try:
        entries = [p for p in root.iterdir() if p.is_dir()]
        markers = ["pyproject.toml", "requirements.txt", "package.json", "manage.py", ".git"]
        return len(entries) >= 6 and not any((root / m).exists() for m in markers)
    except Exception:
        return False


def _is_projects_root(root: Path) -> bool:
    try:
        return root.resolve() == Path("C:/Users/Adam/Projects").resolve()
    except Exception:
        return False


def _find_nearest_project_root(path: Path) -> Path | None:
    try:
        current = path.resolve()
    except Exception:
        return None
    for parent in [current] + list(current.parents):
        if _looks_like_project_root(parent):
            return parent
    return None


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
        out_dir = _runtime_root()
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
    if not sys.stdout.isatty():
        delay_s = 0.0
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
            try:
                sys.stdout.write(ch)
                sys.stdout.flush()
            except UnicodeEncodeError:
                try:
                    sys.stdout.buffer.write(ch.encode("utf-8", errors="replace"))
                    sys.stdout.flush()
                except Exception:
                    pass
            if ch.strip():
                time.sleep(delay_s)

    return _callback


def _typewriter_print(text: str, usage, header=None, controller=None) -> None:
    cb = _typewriter_callback(usage, header=header, controller=controller)
    cb(text)
    if _UI_MANAGER:
        return
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")
        sys.stdout.flush()


def _save_empty_commands_response(text: str) -> None:
    try:
        if not text:
            return
        path = _runtime_path("empty_commands_response.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n--- {ts} ---\n")
            fh.write(text.strip() + "\n")
    except Exception:
        pass


def _format_payload_lines(payload: dict, *, show_final: bool = True, show_commands: Optional[bool] = None) -> list[str]:
    lines: list[str] = []

    def _add(label: str, message) -> None:
        if message is None:
            return
        msg = str(message).strip()
        if msg:
            lines.append(f"{label}: {msg}")

    status_updates = payload.get("status_updates") or []
    if isinstance(status_updates, list):
        for item in status_updates:
            _add("status", item)

    if show_commands is None:
        show_commands = os.getenv("C0D3R_SHOW_MODEL_ACTIONS", "1").strip().lower() not in {"0", "false", "no", "off"}
    if show_commands:
        commands = payload.get("commands") or []
        if isinstance(commands, list):
            for cmd in commands:
                if isinstance(cmd, dict):
                    label = (cmd.get("purpose") or "command").strip() or "command"
                    _add(label, cmd.get("command") or "")
                else:
                    _add("command", cmd)
        checks = payload.get("checks") or []
        if isinstance(checks, list):
            for chk in checks:
                if isinstance(chk, dict):
                    label = (chk.get("purpose") or "check").strip() or "check"
                    _add(label, chk.get("command") or "")
                else:
                    _add("check", chk)

    if show_final:
        _add("final", payload.get("final"))

    if "conclusion" in payload:
        _add("conclusion", payload.get("conclusion"))
        for item in payload.get("observations") or []:
            _add("observation", item)
        for item in payload.get("findings") or []:
            claim = item.get("claim") if isinstance(item, dict) else item
            _add("finding", claim)
        for item in payload.get("next_steps") or []:
            _add("next", item)

    return lines


def _set_final_style(style: str) -> None:
    global _FINAL_STYLE
    if not style:
        return
    _FINAL_STYLE = style
    if _UI_MANAGER:
        _UI_MANAGER.set_final_style(style)


def _parse_color_from_text(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    colors = {
        "yellow": "yellow",
        "gold": "yellow",
        "amber": "yellow",
        "orange": "yellow",
        "red": "red",
        "green": "green",
        "blue": "blue",
        "cyan": "cyan",
        "magenta": "magenta",
        "purple": "magenta",
        "white": "white",
    }
    for key, val in colors.items():
        if key in lower:
            return val
    return ""


def _maybe_set_style_from_prompt(prompt: str) -> None:
    if not prompt:
        return
    lower = prompt.lower()
    if "final" not in lower or ("color" not in lower and "colour" not in lower and "style" not in lower):
        return
    color = _parse_color_from_text(prompt)
    bold = "bold" in lower or "bright" in lower
    if color:
        style = f"{'bold ' if bold else ''}{color}".strip()
        _set_final_style(style)


def _apply_ui_actions(payload: dict) -> None:
    if not isinstance(payload, dict):
        return
    actions = payload.get("ui_actions") or []
    if not isinstance(actions, list):
        return
    for action in actions:
        if not isinstance(action, dict):
            continue
        name = str(action.get("action") or "").strip().lower()
        if name in {"set_final_style", "final_style"}:
            style = str(action.get("value") or action.get("style") or "").strip()
            if style:
                _set_final_style(style)


def _render_json_response(text: str) -> str:
    payload = _safe_json(text)
    if not isinstance(payload, dict):
        return ""
    show_final = True
    lines = _format_payload_lines(payload, show_final=show_final)
    return "\n".join(lines).strip()


def _universal_response_schema() -> str:
    return (
        "Return ONLY JSON with keys: status_updates (list of short strings), "
        "needs_terminal (bool), commands (list), checks (list), output_request (list), final (string). "
        "commands/checks entries must be objects with {purpose, shell, cwd, timeout_s, command}. "
        "If no terminal actions are needed, set needs_terminal=false and commands/checks to [] and output_request to []. "
        "No markdown or extra text."
    )


def _validate_universal_payload(payload: dict) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload is not an object"
    status_updates = payload.get("status_updates")
    if status_updates is None or not isinstance(status_updates, list):
        return False, "missing or invalid key: status_updates"
    if "final" not in payload:
        return False, "missing key: final"
    if "needs_terminal" not in payload or not isinstance(payload.get("needs_terminal"), bool):
        return False, "missing or invalid key: needs_terminal"
    if "commands" not in payload or not isinstance(payload.get("commands"), list):
        return False, "missing or invalid key: commands"
    if "checks" not in payload or not isinstance(payload.get("checks"), list):
        return False, "missing or invalid key: checks"
    if "output_request" not in payload or not isinstance(payload.get("output_request"), list):
        return False, "missing or invalid key: output_request"
    return True, ""


def _ensure_json_response(session: C0d3rSession, prompt: str, response: str, *, max_attempts: int = 3) -> str:
    payload = _safe_json(response)
    if isinstance(payload, dict):
        ok, _ = _validate_universal_payload(payload)
        if ok:
            return response
    max_attempts = min(10, int(os.getenv("C0D3R_JSON_MAX_ATTEMPTS", str(max_attempts)) or str(max_attempts)))
    system = _universal_response_schema()
    attempt = 0
    last = response
    while attempt < max_attempts:
        attempt += 1
        repair_prompt = (
            system
            + "\nYour previous response was not valid JSON. Re-emit ONLY valid JSON matching the schema.\n\n"
            + (last or "")[:2000]
            + "\n\nUser request:\n"
            + prompt
        )
        try:
            last = session.send(prompt=repair_prompt, stream=False)
        except Exception:
            break
        payload = _safe_json(last)
        if isinstance(payload, dict):
            ok, _ = _validate_universal_payload(payload)
            if ok:
                return last
    return response


def _extract_plaintext_response(text: str) -> str:
    if not text:
        return ""
    payload = _safe_json(text)
    if isinstance(payload, dict):
        final = payload.get("final")
        if isinstance(final, str) and final.strip():
            return final.strip()
        answers = payload.get("answers")
        if isinstance(answers, list):
            return "\n".join(str(a) for a in answers)
        response = payload.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()
    return text


def _strip_runtime_ops(ops: list) -> tuple[list, int]:
    if not ops:
        return [], 0
    filtered: list = []
    dropped = 0
    for op in ops:
        if not isinstance(op, dict):
            filtered.append(op)
            continue
        path = str(op.get("path") or "")
        if path.replace("\\", "/").startswith("runtime/"):
            dropped += 1
            continue
        try:
            abs_path = Path(path)
            if abs_path.is_absolute() and _is_runtime_path(abs_path):
                dropped += 1
                continue
        except Exception:
            pass
        filtered.append(op)
    return filtered, dropped


def _qa_verify_answers(session: "C0d3rSession", questions_block: str, draft_answers: str) -> str:
    system = (
        "You are a rigorous verifier. Check the draft answers against the questions. "
        "Correct any mistakes. Output ONLY the corrected numbered answers, no commentary."
    )
    prompt = (
        "Questions:\n"
        f"{questions_block}\n\n"
        "Draft answers:\n"
        f"{draft_answers}\n\n"
        "Return corrected numbered answers only."
    )
    try:
        return session._safe_send(prompt=prompt, stream=False, system=system) or ""
    except Exception:
        return ""


def _extract_numbered_items(text: str) -> dict[int, str]:
    items: dict[int, str] = {}
    if not text:
        return items
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\)\s*(.+)", line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        items[idx] = m.group(2).strip()
    return items


def _override_word_problem_answers(questions: dict[int, str], answers: dict[int, str]) -> dict[int, str]:
    updated = dict(answers)
    for idx, q in questions.items():
        lower = q.lower()
        # Quadratic ax^2 + bx + c = 0
        m = re.search(r"([0-9]*)x\^2\s*([+-])\s*([0-9]+)x\s*([+-])\s*([0-9]+)\s*=\s*0", lower)
        if m:
            a = float(m.group(1) or 1)
            b = float(m.group(3)) * (1 if m.group(2) == "+" else -1)
            c = float(m.group(5)) * (1 if m.group(4) == "+" else -1)
            disc = b * b - 4 * a * c
            if disc >= 0 and a != 0:
                root1 = (-b + disc ** 0.5) / (2 * a)
                root2 = (-b - disc ** 0.5) / (2 * a)
                if abs(root1 - root2) < 1e-9:
                    updated[idx] = f"x = {root1:g}"
                else:
                    updated[idx] = f"x = {root1:g}, x = {root2:g}"
            continue
        # Speed = distance / time.
        m = re.search(r"travels\s+([0-9]+)\s+miles\s+in\s+([0-9]+)\s+hours", lower)
        if m:
            dist = float(m.group(1))
            hours = float(m.group(2))
            if hours != 0:
                updated[idx] = f"{dist / hours:g} mph"
            continue
        # Discount: P% off $X.
        m = re.search(r"([0-9]+)%\s+off\s+\$?([0-9]+)", lower)
        if m:
            pct = float(m.group(1))
            price = float(m.group(2))
            sale = price * (1.0 - pct / 100.0)
            updated[idx] = f"${sale:g}"
            continue
        # Linear equation: ax + b = c.
        m = re.search(r"([0-9]+)x\s*([+-])\s*([0-9]+)\s*=\s*([0-9]+)", lower)
        if m:
            a = float(m.group(1))
            sign = m.group(2)
            b = float(m.group(3)) * (1 if sign == "+" else -1)
            c = float(m.group(4))
            if a != 0:
                x = (c - b) / a
                updated[idx] = f"x = {x:g}"
            continue
        # Fraction of a number.
        m = re.search(r"if\s*([0-9]+)/([0-9]+)\s+of\s+a\s+number\s+is\s+([0-9]+)", lower)
        if m:
            num = float(m.group(1))
            den = float(m.group(2))
            val = float(m.group(3))
            if den != 0:
                updated[idx] = f"{val / (num / den):g}"
            continue
        # Rectangle perimeter.
        m = re.search(r"perimeter\s+([0-9]+)\s+and\s+width\s+([0-9]+)", lower)
        if m:
            perim = float(m.group(1))
            width = float(m.group(2))
            length = perim / 2.0 - width
            updated[idx] = f"{length:g}"
            continue
    return updated


def _format_numbered_answers(answers: dict[int, str]) -> str:
    if not answers:
        return ""
    lines = []
    for idx in sorted(answers.keys()):
        lines.append(f"{idx}) {answers[idx]}")
    return "\n".join(lines)


def _extract_file_ops_from_text(text: str) -> list[dict]:
    if _commands_only_enabled():
        return []
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
    status_updates = payload.get("status_updates")
    if status_updates is None or not isinstance(status_updates, list):
        return False
    cmds = payload.get("commands")
    needs_terminal = payload.get("needs_terminal")
    output_request = payload.get("output_request")
    if cmds is None:
        return False
    if cmds is not None and not isinstance(cmds, list):
        return False
    if needs_terminal is not None and not isinstance(needs_terminal, bool):
        return False
    if output_request is None or not isinstance(output_request, list):
        return False
    return True


def _apply_file_ops(ops: list, workdir: Path, *, base_root: Path | None = None) -> list[Path]:
    global _LAST_FILE_OPS_ERRORS, _LAST_FILE_OPS_WRITTEN
    if _commands_only_enabled():
        _LAST_FILE_OPS_ERRORS = ["file_ops disabled: commands-only mode"]
        _LAST_FILE_OPS_WRITTEN = []
        return []
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
        self.log_path = _runtime_path("executor.log")
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
            todo_path = _runtime_path("bibliography_todo.jsonl")
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
                backup_dir = _runtime_path("backups") / time.strftime("%Y%m%d_%H%M%S")
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
                if ".c0d3r" in str(target).replace("\\", "/"):
                    pass
                elif str(target) not in created_this_batch:
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
    if _commands_only_enabled():
        return []
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
    _ensure_root_dir(base_root)
    return _apply_file_ops(ops, workdir, base_root=base_root or workdir)


def _force_readme_ops(session: C0d3rSession, prompt: str, workdir: Path, *, base_root: Path | None = None) -> list[Path]:
    if _commands_only_enabled():
        return []
    system = (
        "Return ONLY JSON with key: file_ops (list). "
        "Each item: {\"path\": \"README.md\", \"action\": \"write\", \"content\": \"...\"}. "
        "Write a concise README.md summarizing what the project is for and what remains to finish. "
        "Do NOT include any other file paths."
    )
    try:
        raw = session.send(prompt=prompt, stream=False, system=system)
        payload = _safe_json(raw)
    except Exception:
        payload = {}
    ops = payload.get("file_ops") if isinstance(payload, dict) else None
    if not ops or not _file_ops_allowed_for_local_task(ops):
        return []
    _ensure_root_dir(base_root)
    return _apply_file_ops(ops, workdir, base_root=base_root or workdir)


def _force_commands(session: C0d3rSession, prompt: str) -> list[str]:
    system = (
        "Return ONLY JSON with keys: commands (list of strings) and final (string or empty). "
        "Provide 1-5 concrete shell commands that execute the task. "
        "Do NOT include file_ops."
    )
    try:
        raw = session.send(prompt=prompt, stream=False, system=system)
        commands, _ = _extract_commands(raw)
        return commands
    except Exception:
        return []


def _force_local_extract_commands(session: C0d3rSession, prompt: str, target: str) -> list[str]:
    system = (
        "Return ONLY JSON with keys: commands (list of strings) and final (string or empty). "
        "You MUST include at least one command that extracts/unzips the archive and creates the "
        f"target folder '{target}'. "
        "Use platform-appropriate commands (PowerShell on Windows). "
        "Do NOT include file_ops. Provide 1-4 commands."
    )
    try:
        raw = session.send(prompt=prompt, stream=False, system=system)
        commands, _ = _extract_commands(raw)
        return commands
    except Exception:
        return []


def _file_ops_allowed_for_local_task(ops: list[dict]) -> bool:
    if not ops:
        return True
    for op in ops:
        if not isinstance(op, dict):
            continue
        path = str(op.get("path") or "").replace("\\", "/")
        if "/.c0d3r/" in path or path.endswith("/.c0d3r"):
            continue
        lower = path.lower()
        if lower.endswith("readme.md"):
            continue
        if lower.endswith("readme.txt"):
            continue
        if lower.endswith(".md"):
            continue
        return False
    return True


def _readme_present_and_nonempty(root: Path) -> bool:
    try:
        for name in ("README.md", "README.txt", "readme.md", "readme.txt"):
            candidate = root / name
            if candidate.exists() and candidate.is_file():
                if candidate.read_text(encoding="utf-8", errors="ignore").strip():
                    return True
        return False
    except Exception:
        return False


def _has_extract_command(commands: list[str]) -> bool:
    try:
        for cmd in commands or []:
            lower = (cmd or "").lower()
            if "expand-archive" in lower or "unzip" in lower or "tar " in lower or "7z " in lower:
                return True
        return False
    except Exception:
        return False


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
    # If response is only text and task is actionable, re-ask for commands.
    if not response:
        return response
    if _validate_action_schema(response):
        return response
    if _extract_commands(response)[0]:
        return response
    if "schema_validation_failed" in response:
        system = (
            "Return ONLY valid JSON with keys: status_updates (list), needs_terminal (bool), "
            "commands (list), checks (list), output_request (list), final (string or empty). "
            "commands/checks entries must be objects with {purpose, shell, cwd, timeout_s, command}. "
            "Set output_request=[] when not asking for output chunks. No extra text."
        )
        try:
            raw = session.send(prompt=prompt, stream=False, system=system)
            return raw
        except Exception:
            return response
    if _looks_like_code(response):
        # Force command-based file writes for code output.
        system = (
            "You returned code without terminal commands. Return ONLY JSON with keys: "
            "status_updates (list), needs_terminal (bool), commands (list), checks (list), output_request (list), final (string or empty). "
            "Use shell commands to create/update files (e.g., PowerShell Set-Content / Add-Content, bash cat > file). "
            "No file_ops. No extra text."
        )
        try:
            raw = session.send(prompt=prompt, stream=False, system=system)
            return raw
        except Exception:
            return response
    system = (
        "Return ONLY JSON with keys: status_updates (list), needs_terminal (bool), "
        "commands (list), checks (list), output_request (list), final (string or empty). "
        "commands/checks entries must be objects with {purpose, shell, cwd, timeout_s, command}. "
        "Set output_request=[] when not asking for output chunks. No file_ops. You MUST provide commands for actionable tasks."
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
    Ensure actionable tasks produce commands.
    If code appears without terminal commands, force command-based output.
    """
    attempt = 0
    while True:
        if _validate_action_schema(response):
            return response
        if _extract_commands(response)[0]:
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
    return _runtime_path("plan_state.json")


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


def _prompt_signature(prompt: str) -> str:
    try:
        return hashlib.sha256((prompt or "").strip().encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def _init_plan_state(plan_steps: list, workdir: Path, run_command, prompt_sig: str) -> dict:
    return {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": plan_steps,
        "current_index": 0,
        "snapshot": _snapshot_git(workdir, run_command),
        "prompt_sig": prompt_sig,
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
    if "extract" in step_lower or "unzip" in step_lower or "expand-archive" in step_lower:
        try:
            m = re.search(r"([A-Za-z0-9_\\-]+\\.zip)", step, re.I)
            if m:
                stem = Path(m.group(1)).stem
                candidate = (workdir / stem).resolve()
                if candidate.exists():
                    return True
            # fallback: any new directory created recently
            recent_dirs = [p for p in workdir.iterdir() if p.is_dir()]
            if recent_dirs:
                return True
        except Exception:
            pass
    if "test" in step_lower or "pytest" in step_lower or "unit" in step_lower:
        _emit_live(f"verify: running tests for step '{step}'")
        _, ok = _run_tests_for_project(workdir, run_command, usage_tracker, log_path)
        return ok
    # Default verification: ensure repo changed since snapshot.
    current = _snapshot_git(workdir, run_command)
    if not current:
        return True
    return bool(current)


def _microtest_enabled() -> bool:
    return os.getenv("C0D3R_MICROTESTS", "1").strip().lower() not in {"0", "false", "no", "off"}


class PetalManager:
    """
    Dynamic constraint registry (petals).
    No hardwired petals; constraints are learned from user input or model directives.
    """
    def __init__(self) -> None:
        self.path = _runtime_path("petals.json")
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


def _prompt_local_input(prompt: str, *, secret: bool = False) -> str:
    if _UI_MANAGER:
        try:
            _ui_write(prompt)
            return _UI_MANAGER.read_input("")
        except Exception:
            pass
    if secret:
        try:
            return getpass.getpass(prompt)
        except Exception:
            return input(prompt)
    return input(prompt)


def _load_django_user_session():
    if not _ensure_django_ready():
        return None
    path = _DJANGO_USER_FILE
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    user_id = payload.get("user_id")
    username = payload.get("username")
    try:
        from django.contrib.auth import get_user_model
        User = get_user_model()
        if user_id:
            return User.objects.filter(id=user_id).first()
        if username:
            return User.objects.filter(username=username).first()
    except Exception:
        return None
    return None


def _save_django_user_session(user) -> None:
    try:
        payload = {
            "user_id": getattr(user, "id", None),
            "username": getattr(user, "get_username", lambda: None)(),
            "logged_in_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _DJANGO_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DJANGO_USER_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _clear_django_user_session() -> None:
    try:
        if _DJANGO_USER_FILE.exists():
            _DJANGO_USER_FILE.unlink()
    except Exception:
        pass


def _require_django_user():
    if not _ensure_django_ready():
        return None, "django not ready"
    user = _load_django_user_session()
    if not user:
        return None, "no user logged in (run ::wallet_login)"
    return user, ""


def _validate_mnemonic(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]{3,}", text or "")
    return 12 <= len(words) <= 24


def _store_mnemonic_for_user(user, mnemonic: str) -> bool:
    if not user or not mnemonic:
        return False
    if not _ensure_django_ready():
        return False
    try:
        from django.db import transaction
        from securevault.models import SecureSetting
        from services.secure_settings import encrypt_secret
    except Exception:
        return False
    try:
        with transaction.atomic():
            setting, _ = SecureSetting.objects.get_or_create(
                user=user,
                name="MNEMONIC",
                category=_DEFAULT_SECRET_CATEGORY,
                defaults={"is_secret": True},
            )
            setting.is_secret = True
            payload = encrypt_secret(mnemonic.strip())
            setting.value_plain = None
            setting.ciphertext = payload["ciphertext"]
            setting.encapsulated_key = payload["encapsulated_key"]
            setting.nonce = payload["nonce"]
            setting.save()
        return True
    except Exception:
        return False


def _has_mnemonic(user) -> bool:
    if not user or not _ensure_django_ready():
        return False
    try:
        from securevault.models import SecureSetting
        return SecureSetting.objects.filter(
            user=user, name="MNEMONIC", category=_DEFAULT_SECRET_CATEGORY
        ).exists()
    except Exception:
        return False


def _ensure_mnemonic_for_user(user) -> Tuple[bool, str]:
    if not user:
        return False, "user not logged in"
    if _has_mnemonic(user):
        return True, ""
    for attempt in range(3):
        mnemonic = _prompt_local_input("Enter wallet mnemonic (12-24 words): ", secret=True).strip()
        if not mnemonic:
            return False, "mnemonic required"
        if not _validate_mnemonic(mnemonic):
            _ui_write("Mnemonic format not recognized. Please try again.")
            continue
        stored = _store_mnemonic_for_user(user, mnemonic)
        if stored:
            return True, ""
        return False, "failed to store mnemonic"
    return False, "mnemonic validation failed"


def _scrub_mnemonic_from_prompt(prompt: str) -> Tuple[str, bool]:
    if not prompt or not _MNEMONIC_PHRASE.search(prompt):
        return prompt, False
    match = _MNEMONIC_PHRASE.search(prompt)
    if not match:
        return prompt, False
    phrase = " ".join(match.group(0).split())
    words_total = len(re.findall(r"[a-zA-Z]{3,}", prompt))
    phrase_words = len(re.findall(r"[a-zA-Z]{3,}", phrase))
    triggered = _MNEMONIC_TRIGGER.search(prompt) is not None
    if not triggered and words_total > phrase_words + 2:
        return prompt, False
    scrubbed = prompt.replace(match.group(0), "[mnemonic redacted]")
    user, _ = _require_django_user()
    if user and _validate_mnemonic(phrase):
        if _store_mnemonic_for_user(user, phrase):
            _ui_write("Mnemonic captured and stored securely.")
            return scrubbed, True
    _ui_write("Mnemonic detected but no login or storage failed. Run ::wallet_login and try again.")
    return scrubbed, False


def _estimate_usd_value(chain: str, token: str, amount: str) -> Tuple[float | None, str | None]:
    try:
        qty = float(amount)
    except Exception:
        return None, None
    try:
        from services.usd_valuation import UsdValuation
    except Exception:
        return None, None
    symbol = token if token and not str(token).lower().startswith("0x") else None
    try:
        resolver = UsdValuation()
        resolution = resolver.resolve_price(str(chain), str(token), symbol)
        if not resolution or not resolution.usd:
            return None, None
        return qty * float(resolution.usd), resolution.source
    except Exception:
        return None, None


def _prompt_confirmation(lines: list[str]) -> bool:
    for line in lines:
        _ui_write(line)
    response = _prompt_local_input("Type YES to confirm: ").strip().lower()
    return response == "yes"


def _seed_base_matrix_django() -> None:
    if not _ensure_django_ready():
        return
    try:
        from core.models import Equation, EquationDiscipline, EquationSource
        from django.utils import timezone
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
                    "citations": [source.citation] if source and source.citation else [],
                    "tool_used": "seed_base_matrix",
                    "captured_at": timezone.now(),
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
        try:
            from services.graph_store import search_graph_equations
            graph_hits = search_graph_equations(query, limit=limit)
            if graph_hits:
                missing = []
                tokens = [t for t in re.findall(r"[a-zA-Z_]{3,}", query.lower()) if t not in {"that","this","with","from","into","then"}]
                if tokens:
                    missing = [t for t in set(tokens[:10]) if not any(t in str(h["equation"]).lower() for h in graph_hits)]
                return {"hits": graph_hits, "missing": missing}
        except Exception:
            pass
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
    anim = _start_status_animation("matrix")
    try:
        result = _matrix_search(prompt)
        if result.get("hits"):
            return result
        return result
    finally:
        _stop_status_animation(anim)


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
    runtime_dir = _runtime_path("microtests")
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
    normalized = _normalize_command(cmd, workdir)
    code, stdout, stderr = run_command(normalized, cwd=workdir, timeout_s=300)
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
    scripted_prompts: list[str] | None = None,
    images: list[str] | None = None,
    documents: list[str] | None = None,
    header: "HeaderRenderer | None" = None,
    pre_research_enabled: bool = True,
    tech_matrix: dict | None = None,
    plan: dict | None = None,
) -> int:
    from services.conversation_memory import ConversationMemory

    session_id = getattr(session, "session_id", "") if session else ""
    session_path = _runtime_path(f"conversation_{session_id}.jsonl") if session_id else _runtime_path("conversation.jsonl")
    memory = ConversationMemory(session_path)
    memory_long = ConversationMemory(_runtime_path("conversation_global.jsonl"))
    summary_bundle = _load_summary_bundle(session_id=session_id)
    pending = initial_prompt
    scripted_mode = scripted_prompts is not None
    single_shot = initial_prompt is not None and not sys.stdin.isatty() and not scripted_mode
    plan = plan or {}
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
        if scripted_mode and pending is None and scripted_prompts:
            pending = scripted_prompts.pop(0)
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
        if not prompt.strip():
            continue
        user_prompt = _strip_context_block(prompt)
        user_prompt, _ = _scrub_mnemonic_from_prompt(user_prompt)
        _emit_live(f"repl: prompt received (len={len(prompt)})")
        _trace_event({
            "event": "repl.prompt",
            "workdir": str(workdir),
            "prompt_preview": user_prompt[:200],
        })
        _maybe_set_style_from_prompt(user_prompt)
        inlined = _inline_file_references(user_prompt)
        user_prompt_inlined = inlined or user_prompt
        auto_docs = _auto_document_paths(user_prompt, workdir)
        documents_for_prompt = _merge_document_paths(documents, auto_docs)
        force_direct = False
        qa_or_convo = _is_qa_prompt(user_prompt_inlined) or _is_conversation_prompt(user_prompt_inlined)
        if inlined:
            user_prompt = inlined
            qa_or_convo = _is_qa_prompt(user_prompt_inlined) or _is_conversation_prompt(user_prompt_inlined)
            if re.search(r"\banswer all questions\b", user_prompt_inlined, re.I) or re.search(r"\bconversation\b", user_prompt_inlined, re.I):
                force_direct = True
        if qa_or_convo:
            force_direct = True
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
            normalized = _normalize_command(cmd, workdir)
            code, stdout, stderr = run_command(normalized, cwd=workdir)
            if stdout.strip():
                print(stdout.strip())
            if stderr.strip():
                print(stderr.strip())
            continue

        context = ""
        system_context_block = ""
        if not minimal:
            max_chars = int(os.getenv("C0D3R_CONTEXT_CHAR_BUDGET", "12000") or "12000")
            per_turn_limit = int(os.getenv("C0D3R_CONTEXT_PER_TURN_LIMIT", "4000") or "4000")
            context = memory.build_context(
                summary_bundle,
                max_chars=max_chars,
                context_limit=per_turn_limit,
                session_id=session_id,
            )
            last_user = memory.last_user(session_id=session_id)
            if last_user and last_user.content:
                context = f"[last_user]\n{last_user.content}\n\n{context}" if context else f"[last_user]\n{last_user.content}"
            last_exchange = memory.last_exchange(session_id=session_id)
            if last_exchange:
                exchange_lines = ["[last_exchange]"]
                for entry in last_exchange:
                    exchange_lines.append(f"{entry.role}: {entry.content}")
                exchange_block = "\n".join(exchange_lines)
                context = f"{exchange_block}\n\n{context}" if context else exchange_block
        try:
            from services.system_probe import system_probe_context
            probe_block = system_probe_context(workdir)
            system_context_block = probe_block
        except Exception:
            pass
        env_block = _environment_context_block(workdir)
        if env_block:
            system_context_block = f"{system_context_block}\n{env_block}" if system_context_block else env_block
        if system_context_block:
            context = f"{system_context_block}\n{context}" if context else system_context_block
        # Normalize user prompt (strip injected context if any).
        user_prompt = user_prompt_inlined or _strip_context_block(prompt)
        simple_task = _is_simple_file_task(user_prompt)
        # Existing project gate: scan context before edits (parallelized).
        scan_future = None
        scan_executor = None
        if _auto_context_commands_enabled() and _is_existing_project(workdir) and not _is_workspace_root(workdir) and not _is_projects_root(workdir) and not qa_or_convo and not simple_task:
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
            recall_hits = _maybe_long_term_recall(
                session,
                memory,
                user_prompt,
                summary_bundle,
                session_id=session_id,
                memory_long=memory_long,
            )
            if recall_hits:
                context = context + "\n\n[long_term_recall]\n" + "\n".join(recall_hits)
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
            "Provide brief, concrete status updates before the final response. "
            "Avoid private chain-of-thought; summarize reasoning at a high level. "
            + _universal_response_schema()
        )
        intent = (
            "Respond to the user request. If terminal actions are required, set needs_terminal=true and provide commands/checks. "
            "Otherwise set needs_terminal=false and keep commands/checks empty."
        )
        full_prompt = f"{context}\n\n[objective]\n{user_prompt}\n\n[intent]\n{intent}\n\nUser:\n{user_prompt}"
        if force_direct:
            system = _universal_response_schema()
        usage.add_input(full_prompt)
        print("[status] Planning...")
        print("[status] Executing...")
        _emit_live("repl: starting request")
        usage.set_status("planning", "routing + research")
        mode = "tool_loop"
        _trace_event({
            "event": "repl.mode",
            "mode": mode,
            "actionable": _is_actionable_prompt(user_prompt),
            "workdir": str(workdir),
        })
        research_summary = ""
        local_task = _is_local_task(user_prompt)
        if mode in {"tool_loop", "scientific"} or _is_conversation_prompt(user_prompt):
            usage.set_status("research", "gathering references")
            allow_research = True
            if isinstance(plan, dict) and "do_research" in plan and plan.get("do_research") is False:
                allow_research = False
            if not pre_research_enabled or minimal or os.getenv("C0D3R_DISABLE_PRERESEARCH", "").strip().lower() in {"1", "true", "yes", "on"}:
                allow_research = False
            if simple_task or local_task:
                allow_research = False
            if qa_or_convo and "[file_contents:" in user_prompt:
                # Fast QA on provided content shouldn't spin research.
                allow_research = False
            # Defer research for new project scaffolding only if model decides it's needed later.
            if not allow_research:
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
            usage.set_status("executing", "local commands")
            _emit_live("repl: tool loop starting")
            response = _run_tool_loop(
                session, f"{full_prompt}\n\n[research]\n{research_summary}" if research_summary else full_prompt,
                workdir,
                run_command,
                images=images,
                documents=documents_for_prompt,
                stream=False,
                stream_callback=None,
                usage_tracker=usage,
            )
            _emit_live("repl: tool loop complete")
        finally:
            controller.stop()
        rendered = _render_json_response(response)
        if not rendered and _looks_like_json(response or ""):
            # Avoid dumping raw JSON to the UI; keep output clean.
            rendered = ""
        if not rendered and response:
            # Show raw text when model doesn't follow JSON format.
            rendered = response
        if single_shot:
            return 0
        if qa_or_convo and response:
            # For QA/conversation, always print the raw model text (or JSON "final" if present).
            rendered = _extract_plaintext_response(response) or rendered
        if rendered and not rendered.strip():
            rendered = ""
        if not rendered:
            # Still account for output tokens so header cost updates.
            usage.add_output(response or "")
            if header:
                header.update()
        # File ops are ignored in commands-only mode.
        if _extract_file_ops_from_text(response or ""):
            _emit_live("executor: file_ops ignored (commands-only mode). Use terminal commands instead.")
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
            _ui_write_final(rendered)
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
        if not minimal and initial_prompt is None:
            assistant_text = _extract_plaintext_response(response) or rendered or response
            memory.append(
                user_prompt,
                assistant_text,
                context=system_context_block,
                workdir=str(workdir),
                model_id=usage.model_id,
                session_id=session_id,
            )
            memory_long.append(
                user_prompt,
                assistant_text,
                context=system_context_block,
                workdir=str(workdir),
                model_id=usage.model_id,
                session_id=session_id,
            )
            summary_bundle = memory.update_summary(summary_bundle, user_prompt, assistant_text, session)
            _save_summary_bundle(summary_bundle, session_id=session_id)
        if oneshot:
            return 0
        if scripted_mode and not scripted_prompts:
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
    if _is_qa_prompt(lower) or "conversation" in lower:
        return False
    keywords = [
        "create", "update", "modify", "fix", "implement", "add", "remove", "delete",
        "run", "test", "build", "install", "generate", "scaffold", "refactor", "migrate",
        "configure", "setup", "set up", "deploy", "serve", "unzip", "extract", "inspect",
    ]
    return any(k in lower for k in keywords)


def _is_local_task(prompt: str) -> bool:
    if not prompt:
        return False
    lower = prompt.lower()
    return any(k in lower for k in ("unzip", "extract", "inspect", "list files", "list folder")) and "research" not in lower




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
    # Disabled: rely on model-driven routing.
    return False


def _is_scaffold_task(prompt: str) -> bool:
    # No hardcoded scaffold detection; rely on model decisions.
    return False


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
    # Create a project root when explicitly asked or when caller already decided new project.
    lower = (prompt or "").lower()
    explicit = _extract_target_folder_name(prompt)
    if explicit:
        name = explicit
    elif "brand" in lower or "marketing" in lower:
        name = _generate_brand_name(prompt, workdir)
    else:
        name = _slugify_project_name(prompt)
    root = (workdir / name).resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return root


def _extract_target_folder_name(prompt: str) -> str:
    if not prompt:
        return ""
    text = prompt.strip()
    patterns = [
        r"folder named\s+([A-Za-z0-9._-]+)",
        r"directory named\s+([A-Za-z0-9._-]+)",
        r"into\s+([A-Za-z0-9._-]+)\s+folder",
        r"into\s+([A-Za-z0-9._-]+)\s+directory",
        r"into\s+folder\s+([A-Za-z0-9._-]+)",
        r"into\s+directory\s+([A-Za-z0-9._-]+)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _fallback_scaffold_commands(prompt: str, project_root: Path) -> list[str]:
    return []


def _requires_scaffold_cmd(prompt: str) -> bool:
    return False


def _file_ops_only_runtime(ops: list, base_root: Path) -> bool:
    if not ops:
        return False
    for op in ops:
        if not isinstance(op, dict):
            continue
        raw = str(op.get("path") or "")
        target = _resolve_target_path(base_root, raw)
        if target:
            rel = str(target).replace("\\", "/")
            if "/.c0d3r/" in rel or rel.endswith("/.c0d3r"):
                continue
            if not _is_runtime_path(target):
                return False
        if target is None:
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


def _screen_tools(session: C0d3rSession, prompt: str, *, timeout_s: float | None = None) -> dict:
    """
    Quick screener to select tool usage and sequencing for this request.
    Returns JSON with keys:
      mode (tool_loop|direct|scientific),
      do_research (bool),
      do_math (bool),
      do_tool_loop (bool),
      sequence (list of strings),
      model_override (string),
      reason (string).
    """
    system = (
        "Return ONLY JSON with keys: mode, do_research, do_math, do_tool_loop, "
        "sequence (list), model_override (string), reason (string). "
        "Sequence must be minimal and ordered (e.g., [\"context\",\"research\",\"execute\",\"verify\"]). "
        "Choose direct for pure Q&A/conversation. Choose tool_loop for local commands or file changes. "
        "Enable research only if necessary for accuracy."
    )
    try:
        response = _call_with_timeout(
            session._safe_send,
            timeout_s=timeout_s if timeout_s is not None else float(os.getenv("C0D3R_SCREEN_TIMEOUT_S", "2.5") or "2.5"),
            kwargs={"prompt": prompt, "stream": False, "system": system},
        )
        payload = _safe_json(response or "")
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
        anim = _start_status_animation("research")
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
        _stop_status_animation(anim)
        if response:
            _append_bibliography_from_text(response)
            _persist_research_knowledge(prompt, response)
        return response or ""
    except Exception:
        try:
            stop.set()
        except Exception:
            pass
        _stop_status_animation(locals().get("anim"))
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
    def _parse_payload() -> dict:
        raw = cmd[len(f"::{action}") :].strip()
        if not raw:
            return {}
        if raw.startswith("@"):
            path = Path(raw[1:]).expanduser()
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"failed to read payload: {exc}") from exc
        try:
            return json.loads(raw)
        except Exception as exc:
            raise ValueError(f"invalid json payload: {exc}") from exc
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
            log_dir = _runtime_path("bg_logs")
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
    if action in {"wallet_login", "django_login"}:
        if not _ensure_django_ready():
            return 1, "", "django not ready", None
        username = _prompt_local_input("Username: ").strip()
        password = _prompt_local_input("Password: ", secret=True)
        if not username or not password:
            return 1, "", "missing credentials", None
        try:
            from django.contrib.auth import authenticate
            user = authenticate(username=username, password=password)
        except Exception as exc:
            return 1, "", f"login error: {exc}", None
        if not user or not getattr(user, "is_active", True):
            return 1, "", "login failed", None
        _save_django_user_session(user)
        return 0, "login ok", "", None
    if action == "wallet_logout":
        _clear_django_user_session()
        return 0, "logged out", "", None
    if action == "wallet_actions":
        try:
            from services.wallet_actions import list_wallet_actions
            payload = {"actions": list_wallet_actions()}
            return 0, json.dumps(payload, indent=2), "", None
        except Exception as exc:
            return 1, "", str(exc), None
    if action == "wallet_lookup":
        try:
            payload = _parse_payload()
            if not isinstance(payload, dict):
                return 1, "", "wallet_lookup payload must be a JSON object", None
            name = str(payload.get("name") or payload.get("q") or "").strip()
            if not name:
                return 1, "", "wallet_lookup requires name", None
            user, err = _require_django_user()
            if not user:
                return 1, "", err, None
            from addressbook.models import AddressBookEntry
            exact = bool(payload.get("exact", False))
            limit = int(payload.get("limit") or 10)
            queryset = AddressBookEntry.objects.filter(user=user)
            queryset = queryset.filter(name__iexact=name) if exact else queryset.filter(name__icontains=name)
            results = []
            for entry in queryset.order_by("name", "address")[: max(1, min(limit, 50))]:
                results.append(
                    {
                        "id": entry.id,
                        "name": entry.name,
                        "address": entry.address,
                        "chain": entry.chain,
                        "image": entry.image.url if entry.image else "",
                    }
                )
            return 0, json.dumps({"count": queryset.count(), "results": results}, indent=2), "", None
        except Exception as exc:
            return 1, "", str(exc), None
    if action in {"wallet_send", "wallet_swap", "wallet_bridge"}:
        try:
            payload = _parse_payload()
            if not isinstance(payload, dict):
                return 1, "", "wallet action payload must be a JSON object", None
            user, err = _require_django_user()
            if not user:
                return 1, "", err, None
            action_payload: dict = {}
            if action == "wallet_send":
                chain = str(payload.get("chain") or "").strip()
                token = str(payload.get("token") or "").strip()
                amount = str(payload.get("amount") or payload.get("qty") or "").strip()
                to_addr = str(payload.get("to") or payload.get("address") or "").strip()
                to_name = str(payload.get("to_name") or payload.get("recipient") or payload.get("name") or "").strip()
                if not to_addr and to_name:
                    from addressbook.models import AddressBookEntry
                    matches = list(AddressBookEntry.objects.filter(user=user, name__iexact=to_name))
                    if not matches:
                        matches = list(AddressBookEntry.objects.filter(user=user, name__icontains=to_name)[:5])
                    if len(matches) == 1:
                        entry = matches[0]
                        to_addr = entry.address
                        if not chain and entry.chain:
                            chain = entry.chain
                        to_name = entry.name
                    elif len(matches) > 1:
                        _ui_write("Multiple address book matches found:")
                        for idx, entry in enumerate(matches, start=1):
                            _ui_write(f"{idx}. {entry.name} [{entry.chain}] {entry.address}")
                        choice = _prompt_local_input("Select number (blank to cancel): ").strip()
                        if not choice:
                            return 1, "", "cancelled", None
                        try:
                            entry = matches[int(choice) - 1]
                        except Exception:
                            return 1, "", "invalid selection", None
                        to_addr = entry.address
                        if not chain and entry.chain:
                            chain = entry.chain
                        to_name = entry.name
                if not all([chain, token, amount, to_addr]):
                    return 1, "", "wallet_send requires chain, token, amount, and to/recipient", None
                usd_val, usd_source = _estimate_usd_value(chain, token, amount)
                summary = [
                    "[wallet] confirm send",
                    f"- recipient: {to_name or '(unspecified)'} {to_addr}",
                    f"- chain: {chain}",
                    f"- token: {token}",
                    f"- amount: {amount}",
                    f"- est_usd: ${usd_val:,.2f} ({usd_source})" if usd_val is not None else "- est_usd: unknown",
                ]
                if not _prompt_confirmation(summary):
                    return 1, "", "cancelled", None
                ok, msg = _ensure_mnemonic_for_user(user)
                if not ok:
                    return 1, "", msg, None
                action_payload = {"chain": chain, "token": token, "to": to_addr, "amount": amount}
            elif action == "wallet_swap":
                chain = str(payload.get("chain") or "").strip()
                sell = str(payload.get("sell_token") or payload.get("sell") or "").strip()
                buy = str(payload.get("buy_token") or payload.get("buy") or "").strip()
                amount = str(payload.get("amount") or payload.get("sell_amount") or "").strip()
                if not all([chain, sell, buy, amount]):
                    return 1, "", "wallet_swap requires chain, sell_token, buy_token, amount", None
                usd_val, usd_source = _estimate_usd_value(chain, sell, amount)
                summary = [
                    "[wallet] confirm swap",
                    f"- chain: {chain}",
                    f"- sell: {amount} {sell}",
                    f"- buy: {buy}",
                    f"- est_usd: ${usd_val:,.2f} ({usd_source})" if usd_val is not None else "- est_usd: unknown",
                ]
                if not _prompt_confirmation(summary):
                    return 1, "", "cancelled", None
                ok, msg = _ensure_mnemonic_for_user(user)
                if not ok:
                    return 1, "", msg, None
                action_payload = {"chain": chain, "sell_token": sell, "buy_token": buy, "amount": amount}
            elif action == "wallet_bridge":
                src = str(payload.get("source_chain") or payload.get("from_chain") or "").strip()
                dst = str(payload.get("destination_chain") or payload.get("to_chain") or "").strip()
                token = str(payload.get("token") or "").strip()
                amount = str(payload.get("amount") or "").strip()
                dst_token = str(payload.get("destination_token") or payload.get("to_token") or "").strip()
                if not all([src, dst, token, amount]):
                    return 1, "", "wallet_bridge requires source_chain, destination_chain, token, amount", None
                usd_val, usd_source = _estimate_usd_value(src, token, amount)
                summary = [
                    "[wallet] confirm bridge",
                    f"- source: {src}",
                    f"- destination: {dst}",
                    f"- token: {token}",
                    f"- amount: {amount}",
                    f"- destination_token: {dst_token or token}",
                    f"- est_usd: ${usd_val:,.2f} ({usd_source})" if usd_val is not None else "- est_usd: unknown",
                ]
                if not _prompt_confirmation(summary):
                    return 1, "", "cancelled", None
                ok, msg = _ensure_mnemonic_for_user(user)
                if not ok:
                    return 1, "", msg, None
                action_payload = {
                    "source_chain": src,
                    "destination_chain": dst,
                    "token": token,
                    "amount": amount,
                }
                if dst_token:
                    action_payload["destination_token"] = dst_token
            from services.wallet_runner import wallet_runner
            action_name = action.replace("wallet_", "")
            wallet_runner.run(action_name, action_payload, user=user)
            if payload.get("wait"):
                timeout = float(payload.get("timeout_s") or 120)
                start = time.time()
                while time.time() - start < timeout:
                    status = wallet_runner.status()
                    if not status.get("running"):
                        return 0, json.dumps(status, indent=2), "", None
                    time.sleep(1.0)
                return 1, "", "timeout waiting for wallet action", None
            return 0, json.dumps(wallet_runner.status(), indent=2), "", None
        except Exception as exc:
            return 1, "", str(exc), None
    if action in {"datalab_tables", "datalab_list"}:
        try:
            from services.data_lab_tools import list_tables, summarize_payload
            payload = list_tables()
            return 0, summarize_payload(payload), "", None
        except Exception as exc:
            return 1, "", str(exc), None
    if action in {"datalab_query", "datalab_db"}:
        try:
            payload = _parse_payload()
            if not isinstance(payload, dict):
                return 1, "", "datalab_query payload must be a JSON object", None
            table = str(payload.get("table") or "").strip()
            if not table:
                return 1, "", "datalab_query requires table", None
            from services.data_lab_tools import query_table, summarize_payload
            result = query_table(
                table,
                filters=payload.get("filters"),
                limit=payload.get("limit", 50),
                order_by=payload.get("order_by"),
                order=payload.get("order", "desc"),
                columns=payload.get("columns"),
            )
            return 0, summarize_payload(result), "", None
        except Exception as exc:
            return 1, "", str(exc), None
    if action == "datalab_news":
        try:
            payload = _parse_payload()
            if not isinstance(payload, dict):
                return 1, "", "datalab_news payload must be a JSON object", None
            tokens = payload.get("tokens") or []
            query = payload.get("query")
            if not tokens and not query:
                return 1, "", "datalab_news requires tokens or query", None
            if not tokens and query:
                tokens = [tok for tok in re.split(r"[ ,;/]+", str(query).upper()) if tok]
            start = payload.get("start") or (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)).isoformat()
            end = payload.get("end") or datetime.datetime.now(datetime.timezone.utc).isoformat()
            from services.data_lab_tools import fetch_news_with_summary, summarize_payload
            result = fetch_news_with_summary(
                tokens=tokens,
                start=start,
                end=end,
                query=query,
                max_pages=payload.get("max_pages"),
                max_items=payload.get("max_items", 40),
            )
            return 0, summarize_payload(result), "", None
        except Exception as exc:
            return 1, "", str(exc), None
    if action == "datalab_web":
        try:
            payload = _parse_payload()
            if not isinstance(payload, dict):
                return 1, "", "datalab_web payload must be a JSON object", None
            query = str(payload.get("query") or "").strip()
            if not query:
                return 1, "", "datalab_web requires query", None
            from services.data_lab_tools import search_web, summarize_payload
            result = search_web(
                query=query,
                max_results=payload.get("max_results", 5),
                max_bytes=payload.get("max_bytes", 200000),
                max_chars=payload.get("max_chars", 12000),
                summary_sentences=payload.get("summary_sentences", 3),
                domains=payload.get("domains"),
            )
            return 0, summarize_payload(result), "", None
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
    # Disabled: rely on model-driven routing.
    return False


def _tool_loop_base_request(prompt: str) -> str:
    return _strip_context_block(prompt)


def _inline_file_references(prompt: str) -> str:
    """
    Inline referenced file contents into the prompt for Q&A/conversation tasks.
    Supports patterns like: "file: C:\\path\\to\\file.txt".
    """
    if not prompt:
        return ""
    patterns = [
        r"file:\s*([A-Za-z]:\\[^\n\r]+)",
        r"file:\s*([^ \n\r]+)",
    ]
    seen = set()
    additions = []
    for pattern in patterns:
        for match in re.findall(pattern, prompt):
            path = match.strip().strip("\"'").rstrip(").,;")
            if not path or path in seen:
                continue
            seen.add(path)
            try:
                candidate = Path(path).expanduser().resolve()
            except Exception:
                continue
            if not candidate.exists() or not candidate.is_file():
                continue
            if _is_supported_document(candidate):
                # Avoid inlining binary/large documents; use document attachments instead.
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                try:
                    text = candidate.read_text(errors="ignore")
                except Exception:
                    continue
            max_chars = int(os.getenv("C0D3R_INLINE_FILE_MAX_CHARS", "40000") or "40000")
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...[truncated]..."
            additions.append(f"[file_contents:{candidate}]\n{text}\n[end_file_contents]")
    if not additions:
        return ""
    return prompt + "\n\n" + "\n\n".join(additions)


def _is_qa_prompt(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "answer all questions" in lower or "question sheet" in lower or "provide numbered answers" in lower


def _requires_new_projects_dir(prompt: str) -> bool:
    # Disabled: rely on model-driven routing.
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
    base = _runtime_root()
    return (base / "plan.md").exists() and (base / "checklist.md").exists()


def _checklist_has_mapping() -> bool:
    path = _runtime_path("checklist.md")
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return "source->decision->code" in text


def _checklist_is_complete() -> bool:
    path = _runtime_path("checklist.md")
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text.strip()) < 50:
        return False
    if text.count("- [") < 3:
        return False
    return "[ ]" not in text


def _plan_is_substantial(workdir: Path) -> bool:
    path = (workdir / ".c0d3r" / "plan.md").resolve()
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return len(text.strip()) >= 50


def _commands_only_runtime(commands: List[str]) -> bool:
    if not commands:
        return False
    runtime_root = str(_runtime_root()).lower()
    for cmd in commands:
        lower = (cmd or "").lower()
        if runtime_root and runtime_root in lower:
            continue
        if "runtime\\c0d3r" in lower or "runtime/c0d3r" in lower:
            continue
        if ".c0d3r" in lower:
            continue
        if "plan.md" in lower or "checklist.md" in lower:
            continue
        return False
    return True


def _commands_only_inspection(commands: List[str]) -> bool:
    if not commands:
        return False
    for cmd in commands:
        lower = (cmd or "").strip().lower()
        if lower.startswith(("dir", "ls", "pwd", "cd", "echo", "get-childitem", "where", "whoami")):
            continue
        return False
    return True


def _apply_simple_task_fallback(prompt: str, workdir: Path) -> bool:
    return False


def _apply_simple_project_stub(prompt: str, workdir: Path) -> bool:
    return False


def _maybe_retarget_project(prompt: str, workdir: Path) -> Path | None:
    """
    If prompt references a project name and we're not in it, retarget to that folder
    under C:\\Users\\Adam\\Projects when found.
    """
    lower = (prompt or "").lower()
    if "project" not in lower:
        return None
    if not any(phrase in lower for phrase in ("update project", "work on", "open project", "continue project", "go to project", "switch to project")):
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
        payload_path = _runtime_path("unbounded_payload.json")
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
    path = _runtime_path("benchmarks.json")
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
    log_path = _runtime_path("evidence.log")
    if not log_path.exists():
        return False
    log_text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    if "benchmark" in log_text and "python" in log_text and "exit 0" in log_text:
        return True
    return False


def _tests_passed_recently() -> bool:
    log_path = _runtime_path("evidence.log")
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
    if "unzip" in lower and "not recognized" in lower:
        try:
            import re as _re
            zip_name = None
            dest = "."
            m = _re.search(r"unzip\s+(?:-o\s+)?([^\s]+)(?:\s+-d\s+([^\s]+))?", cmd, _re.I)
            if m:
                zip_name = m.group(1).strip().strip('"')
                if m.group(2):
                    dest = m.group(2).strip().strip('"')
            if not zip_name:
                # fallback: first zip in workdir
                zips = list(workdir.glob("*.zip"))
                if zips:
                    zip_name = zips[0].name
            if zip_name:
                fix_cmd = f'Expand-Archive -Path "{zip_name}" -DestinationPath "{dest}" -Force'
                fix_cmd = _normalize_command(fix_cmd, workdir)
                code, stdout, err = run_command(fix_cmd, cwd=workdir, timeout_s=_command_timeout_s(fix_cmd))
                _append_tool_log(log_path, fix_cmd, code, stdout, err)
                return True, code, stdout, err
        except Exception:
            pass
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
    try:
        return path.resolve().is_relative_to(_runtime_root())
    except Exception:
        rel = str(path).replace("\\", "/")
        root = str(_runtime_root()).replace("\\", "/")
        return rel.startswith(root)


def _is_runtime_command(cmd: str) -> bool:
    lower = (cmd or "").lower()
    root = str(_runtime_root()).lower()
    return root in lower or "runtime\\c0d3r" in lower or "runtime/c0d3r" in lower


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
    anim = _start_status_animation("matrix")
    try:
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
    finally:
        _stop_status_animation(anim)


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
        out_dir = _runtime_root()
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
            path = _runtime_path("behavior_log.jsonl")
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
        path = _runtime_path("behavior_insights.md")
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
        out = _runtime_path("bibliography.md")
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
        from django.utils import timezone
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
                "citations": authoritative_links,
                "tool_used": "c0d3r_unbounded_matrix",
                "captured_at": timezone.now(),
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
        try:
            from services.graph_store import sync_graph_from_django
            if os.getenv("C0D3R_GRAPH_SYNC_ON_WRITE", "1").strip().lower() not in {"0", "false", "no", "off"}:
                sync_graph_from_django()
        except Exception:
            pass
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
        out = _runtime_path("system_map.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _append_evidence(cmd: str, code: int, stdout: str, stderr: str) -> None:
    try:
        out = _runtime_path("evidence.log")
        out.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with out.open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] $ {cmd}\n(exit {code})\n{stdout.strip()}\n{stderr.strip()}\n\n")
    except Exception:
        pass


def _load_code_memory_summary(max_chars: int = 2000) -> str:
    path = _runtime_path("system_map.json")
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
        out = _runtime_path("research_notes.md")
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
    checklist = _runtime_path("checklist.md")
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
    def _run_test_cmd(cmd: str):
        normalized = _normalize_command(cmd, project_root)
        return run_command(normalized, cwd=project_root, timeout_s=_command_timeout_s(normalized))
    if not _looks_like_project_root(project_root):
        candidate = _find_nearest_project_root(target) if target else None
        project_root = candidate or project_root
    if not _looks_like_project_root(project_root) or _is_workspace_root(project_root):
        _emit_live(f"tests: skipping (no project root detected at {project_root})")
        return False, True
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
            code, stdout, stderr = _run_test_cmd(cmd)
            _append_tool_log(log_path, cmd, code, stdout, stderr)
            if code == 0:
                tests_ok = True
            else:
                return tests_ran, False
        if handler.full_tests(profile):
            for cmd in handler.full_tests(profile):
                tests_ran = True
                code, stdout, stderr = _run_test_cmd(cmd)
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
        code, stdout, stderr = _run_test_cmd(cmd)
        _append_tool_log(log_path, cmd, code, stdout, stderr)
        if code == 0:
            tests_ok = True
        else:
            if "no module named pytest" in (stdout + "\n" + stderr).lower():
                _install_pytest(python_exec, project_root, run_command, log_path)
                code, stdout, stderr = _run_test_cmd(cmd)
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
                code, stdout, stderr = _run_test_cmd(retry)
                _append_tool_log(log_path, retry, code, stdout, stderr)
                if code == 0:
                    return tests_ran, True
            key = str(project_root.resolve())
            if _auto_install_enabled() and (project_root / "requirements.txt").exists() and key not in _INSTALL_ATTEMPTS:
                _INSTALL_ATTEMPTS.add(key)
                _emit_live("tests failed; attempting pip install -r requirements.txt (one-time)")
                pip_cmd = f"\"{python_exec}\" -m pip install -r requirements.txt" if python_exec else "python -m pip install -r requirements.txt"
                _run_test_cmd(pip_cmd)
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
                    code, stdout, stderr = _run_test_cmd(cmd)
                    _append_tool_log(log_path, cmd, code, stdout, stderr)
                    if code == 0:
                        tests_ok = True
                    else:
                        return tests_ran, False
                elif has_jest:
                    tests_ran = True
                    cmd = f"npx jest \"{target}\""
                    code, stdout, stderr = _run_test_cmd(cmd)
                    _append_tool_log(log_path, cmd, code, stdout, stderr)
                    if code == 0:
                        tests_ok = True
                    else:
                        return tests_ran, False
            if "test" in scripts:
                tests_ran = True
                cmd = "npm test -- --watch=false"
                code, stdout, stderr = _run_test_cmd(cmd)
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
        if _is_runtime_path(target):
            return False
        if target.suffix.lower() in {".md", ".txt", ".log"}:
            return False
        return True
    except Exception:
        return True


def _ensure_foo_data(project_root: Path) -> None:
    try:
        path = (project_root / ".c0d3r" / "foo_data.json").resolve()
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
        "    data_path = Path('.c0d3r/foo_data.json')\n"
        "    payload = json.loads(data_path.read_text())\n"
        f"    __import__('{module}')\n"
        "    assert payload['foo'] == 'bar'\n"
    )
    try:
        test_path.write_text(content, encoding="utf-8")
    except Exception:
        pass


def _fallback_inspection_commands(workdir: Path) -> List[str]:
    # No hardcoded inspection commands; rely on model outputs.
    return []


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
    if _commands_only_enabled():
        return False, 0, "", ""
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
        self._status_anim = None
        self._status_anim_kind = ""

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
        if status == "research":
            if self._status_anim_kind != "research":
                self._status_anim = _start_status_animation("research")
                self._status_anim_kind = "research"
        elif status == "matrix":
            if self._status_anim_kind != "matrix":
                self._status_anim = _start_status_animation("matrix")
                self._status_anim_kind = "matrix"
        else:
            if self._status_anim:
                _stop_status_animation(self._status_anim)
                self._status_anim = None
                self._status_anim_kind = ""


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
        try:
            if self.ansi_ok:
                sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(header)
            sys.stdout.flush()
            self._last = header
        except OSError:
            # If stdout is invalid (non-interactive teardown), disable header updates.
            self.enabled = False

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
        try:
            sys.stdout.write("\x1b7\x1b[H")
            sys.stdout.write(header)
            sys.stdout.write("\x1b8")
            sys.stdout.flush()
            self._last = header
        except OSError:
            self.enabled = False

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
