#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
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
import platform
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from collections import deque

global _UI_MANAGER
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _live_log_enabled() -> bool:
    return os.getenv("C0D3R_LIVE_LOG", "1").strip().lower() not in {"0", "false", "no", "off"}

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = str(raw).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default

def _runtime_root() -> Path:
    override = os.getenv("C0D3R_RUNTIME_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (PROJECT_ROOT / "runtime" / "c0d3r").resolve()


def _runtime_path(*parts: str) -> Path:
    return _runtime_root().joinpath(*parts)

_SEARCH_MEMORY_PATH = _runtime_path("search_memory.jsonl")
_SEARCH_MEMORY_SESSION_TEMPLATE = "search_memory_{session_id}.jsonl"
_ENV_MEMORY_PATH = _runtime_path("env_command_memory.jsonl")
_SIDE_MEMORY_CACHE: dict[str, object] = {}

_INSTALL_ATTEMPTS: set[str] = set()
_UI_MANAGER = None
_LAST_FILE_OPS_ERRORS: list[str] = []
_LAST_FILE_OPS_WRITTEN: list[str] = []
_MATRIX_SEED_VERSION = "2026-02-04"
_TECH_MATRIX_DIR = _runtime_path("tech_matrix")
_FINAL_STYLE = os.getenv("C0D3R_FINAL_STYLE", "bold yellow")
_USER_STYLE = os.getenv("C0D3R_USER_STYLE", "yellow")
_DEFAULT_SECRET_CATEGORY = "default"
_DJANGO_USER_FILE = _runtime_path("django_user.json")
_MNEMONIC_TRIGGER = re.compile(r"\b(mnemonic|seed phrase|seed)\b", re.IGNORECASE)
_MNEMONIC_PHRASE = re.compile(r"\b(?:[a-zA-Z]{3,}\s+){11,23}[a-zA-Z]{3,}\b")
_HEARTBEAT_SESSION = None
_HEARTBEAT_MODEL_ID = ""
_HEARTBEAT_INTERVAL_S = _env_float("C0D3R_HEARTBEAT_MINUTES", 30.0) * 60.0
_HEARTBEAT_USE_MODEL = os.getenv("C0D3R_HEARTBEAT_USE_MODEL", "1").strip().lower() not in {"0", "false", "no", "off"}
_CONTROL_SYSTEM_PREFIX = (
    "You are operating as a closed-loop empirical systems-engineering control system. "
    "Frame decisions as hypotheses, constraints, state variables, and measurable acceptance criteria. "
    "Continuously compare observations to expectations and adapt via feedback loops and context injection, "
    "avoiding rigid heuristics except where operations are provably deterministic. "
    "Prioritize experimental validation, error bounds, and falsifiable checks. "
    "Return deterministic, schema-compliant JSON only. "
)


def _init_heartbeat(session) -> None:
    global _HEARTBEAT_SESSION, _HEARTBEAT_MODEL_ID, _HEARTBEAT_INTERVAL_S
    try:
        from tools.c0d3r_session import ROLE_FALLBACK_MODEL
    except Exception:
        ROLE_FALLBACK_MODEL = {}
    _HEARTBEAT_SESSION = session
    fallback = ""
    try:
        fallback = ROLE_FALLBACK_MODEL.get("worker", "")
    except Exception:
        fallback = ""
    _HEARTBEAT_MODEL_ID = os.getenv("C0D3R_HEARTBEAT_MODEL") or fallback or getattr(session, "get_model_id", lambda: "")()
    _HEARTBEAT_INTERVAL_S = _env_float("C0D3R_HEARTBEAT_MINUTES", 30.0) * 60.0
_DOCUMENT_EXTENSIONS = {".pdf", ".csv", ".doc", ".docx", ".xls", ".xlsx", ".html", ".txt", ".md"}
_ANIMATION_LOCK = threading.Lock()
_ANIMATION_STATE: dict[str, object | None] = {"stop": None, "thread": None, "kind": None}
_LIVE_NOTE_QUEUE: "queue.Queue[str]" = queue.Queue()
_PENDING_IMAGES: list[str] = []


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


def _control_system_prompt(text: str) -> str:
    return _CONTROL_SYSTEM_PREFIX + text



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
        self.budget_usd = float(os.getenv("C0D3R_BUDGET_USD", "50.0") or "10.0")
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

    def write_user(self, text: str) -> None:
        if text is None:
            return
        lines = str(text).splitlines() or [""]
        prefix = "User: "
        if self._use_textual and self._textual_app:
            try:
                from rich.text import Text
                style = _USER_STYLE or "bright_yellow"
                for idx, line in enumerate(lines):
                    label = prefix if idx == 0 else " " * len(prefix)
                    self._textual_app.call_from_thread(
                        self._textual_app.push_renderable,
                        Text(f"{label}{line}", style=style),
                    )
                return
            except Exception:
                pass
        ansi = "\x1b[93m"
        reset = "\x1b[0m"
        for idx, line in enumerate(lines):
            label = prefix if idx == 0 else " " * len(prefix)
            self.write_line(f"{ansi}{label}{line}{reset}")

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


usage = UsageTracker(model_id=session.get_model_id())
header = HeaderRenderer(usage)
# Initialize terminal UI if available.

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



