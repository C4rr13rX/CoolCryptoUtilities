from __future__ import annotations

import os
import sys
import time
import threading
import queue
from pathlib import Path
from collections import deque

from header_renderer import HeaderRenderer

_FINAL_STYLE: str = os.getenv("C0D3R_FINAL_STYLE", "bold yellow")
_USER_STYLE: str = os.getenv("C0D3R_USER_STYLE", "yellow")


class TerminalUI:
    """
    Multi-backend TUI for c0d3r V2.

    Tries Textual -> prompt_toolkit -> Rich -> ANSI fallback.
    Provides input queue, header refresh, and styled output methods.
    """

    def __init__(self, header: HeaderRenderer, workdir: Path) -> None:
        self.header = header
        self.workdir = workdir
        self.lines: deque[str] = deque(maxlen=1000)
        self.footer = ""
        self.status = ""
        self._lock = threading.Lock()
        self._input_queue: queue.Queue[str] = queue.Queue()
        self._running = False
        self._prompt_thread: threading.Thread | None = None
        self._render_thread: threading.Thread | None = None
        self._render_event = threading.Event()
        self._dirty = False
        self._last_render = 0.0
        self._min_render_interval = 1.0 / 30.0
        self._header_refresh_s = float(
            os.getenv("C0D3R_HEADER_REFRESH_S", "0.5") or "0.5"
        )
        self._last_header_refresh = 0.0
        # Backend flags
        self._use_textual = False
        self._use_rich = False
        self._use_prompt_toolkit = False
        self.backend = "none"
        self.backend_error = ""
        # Backend handles
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

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_tui(self) -> None:
        backend = os.getenv("C0D3R_TUI_BACKEND", "textual").strip().lower()

        if backend in {"", "textual"}:
            try:
                from textual.app import App, ComposeResult
                from textual.widgets import Static, Input, RichLog

                class C0d3rTextualApp(App):
                    BINDINGS = [("ctrl+c", "quit", "Quit")]
                    CSS = (
                        "#header { height:5; }\n"
                        "#body { height:1fr; }\n"
                        "#footer { height:3; }"
                    )

                    def __init__(self, ui: TerminalUI) -> None:
                        super().__init__()
                        self.ui = ui
                        self._header = None
                        self._body = None
                        self._footer = None

                    def compose(self) -> ComposeResult:
                        yield Static(id="header")
                        yield RichLog(
                            id="body", wrap=False,
                            markup=False, highlight=False,
                        )
                        yield Input(
                            id="footer",
                            placeholder="Type instructions and press Enter",
                        )

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
                                self._header.update(
                                    self.ui.header.render_text()
                                )
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
                            self._footer.placeholder = (
                                text or "Type instructions and press Enter"
                            )

                self._textual_app = C0d3rTextualApp(self)
                self._use_textual = True
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
                from prompt_toolkit.key_binding import KeyBindings

                kb = KeyBindings()
                pt_header = TextArea(
                    height=4, text=self.header.render_text(),
                    style="class:header", focusable=False, read_only=True,
                )
                pt_output = TextArea(
                    text="", focusable=False, read_only=True,
                    scrollbar=True, wrap_lines=False,
                )
                pt_input = TextArea(
                    height=1, prompt=f"[{self.workdir}]> ",
                    multiline=False, wrap_lines=False,
                )

                @kb.add("enter")
                def _(event) -> None:
                    text = pt_input.text
                    pt_input.text = ""
                    if text is not None:
                        self._input_queue.put(text)

                root = HSplit([
                    pt_header,
                    Window(height=1, char="-"),
                    pt_output,
                    Window(height=1, char="-"),
                    pt_input,
                ])
                layout = Layout(root, focused_element=pt_input)
                self._pt_app = Application(
                    layout=layout, key_bindings=kb, full_screen=True,
                )
                self._pt_header = pt_header
                self._pt_output = pt_output
                self._pt_input = pt_input
                self._use_prompt_toolkit = True
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
                self._console = Console()
                self._layout = Layout()
                self._layout.split_column(
                    Layout(name="header", size=5),
                    Layout(name="body", ratio=1),
                    Layout(name="footer", size=3),
                )
                self._live = Live(
                    self._layout, console=self._console,
                    refresh_per_second=8, transient=False,
                )
                self._use_rich = True
                self.backend = "rich"
            except Exception:
                self._use_rich = False
                if not self.backend_error:
                    self.backend_error = "no TUI backend available"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        if self._use_textual and self._textual_app:
            self._textual_thread = threading.Thread(
                target=self._textual_app.run, daemon=True,
            )
            self._textual_thread.start()
        elif self._use_prompt_toolkit and self._pt_app:
            self._prompt_thread = threading.Thread(
                target=self._pt_app.run, daemon=True,
            )
            self._prompt_thread.start()
        elif self._use_rich and self._live:
            self._live.start()

        if not self._use_prompt_toolkit and not self._use_textual:
            self._prompt_thread = threading.Thread(
                target=self._input_loop, daemon=True,
            )
            self._prompt_thread.start()

        self._render_thread = threading.Thread(
            target=self._render_loop, daemon=True,
        )
        self._render_thread.start()
        self.render()

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

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

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
            while self._running:
                try:
                    text = input(f"[{self.workdir}]> ")
                    self._input_queue.put(text)
                except Exception:
                    break

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def read_input(self, prompt: str = "") -> str:
        return self._input_queue.get()

    def drain_input(self, max_items: int = 50) -> list[str]:
        items: list[str] = []
        for _ in range(max_items):
            try:
                items.append(self._input_queue.get_nowait())
            except queue.Empty:
                break
        return items

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

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
                self._textual_app.call_from_thread(
                    self._textual_app.push_line, line,
                )
                return
            except Exception:
                pass
        with self._lock:
            self.lines.append(line)
            self._dirty = True
        self.render()

    def write_text(
        self, text: str, *, delay_s: float = 0.0, controller=None,
    ) -> None:
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
                self._textual_app.call_from_thread(
                    self._textual_app.push_line, "",
                )
                self._textual_app.call_from_thread(
                    self._textual_app.push_line, "",
                )
                for line in lines or [""]:
                    self._textual_app.call_from_thread(
                        self._textual_app.push_renderable,
                        Text(line, style=style),
                    )
                self._textual_app.call_from_thread(
                    self._textual_app.push_line, "",
                )
                self._textual_app.call_from_thread(
                    self._textual_app.push_line, "",
                )
                return
            except Exception:
                pass
        ansi = "\x1b[1;33m"
        reset = "\x1b[0m"
        decorated = "\n\n" + "\n".join(lines) + "\n\n"
        try:
            self.write_line(ansi + decorated + reset)
        except Exception:
            self.write_line(decorated)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

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
        footer_text = self.footer or (
            f"queued: {queued}" if queued else "ready"
        )
        if self.status:
            footer_text = f"{footer_text} | {self.status}".strip()

        if self._use_textual and self._textual_app:
            try:
                self._textual_app.call_from_thread(
                    self._textual_app.set_header_text, header_text,
                )
                self._textual_app.call_from_thread(
                    self._textual_app.set_footer_hint, footer_text,
                )
                pending: list[str] = []
                with self._lock:
                    if self.lines:
                        pending = list(self.lines)
                        self.lines.clear()
                if pending:
                    def _push() -> None:
                        for line in pending:
                            self._textual_app.push_line(line)
                    self._textual_app.call_from_thread(_push)
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
                    self._pt_output.buffer.cursor_position = len(
                        self._pt_output.text
                    )
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
            self._layout["header"].update(
                Panel(header_text, title="c0d3r", border_style="blue"),
            )
            self._layout["body"].update(
                Panel(Text(body_text), title="output"),
            )
            self._layout["footer"].update(
                Panel(footer_text or "ready", title="input"),
            )
            self._last_render = now
            self._dirty = False
            return

        # Basic ANSI fallback
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(header_text)
        sys.stdout.write(body_text + "\n")
        sys.stdout.write(footer_text + "\n")
        sys.stdout.flush()
        self._last_render = now
        self._dirty = False
