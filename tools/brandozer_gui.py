from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from kivy.app import App
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.textinput import TextInput

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRANDOZER_BIN = PROJECT_ROOT / "tools" / "brandozer_cli.py"
if not BRANDOZER_BIN.exists():
    BRANDOZER_BIN = PROJECT_ROOT / "bin" / "brandozer"


def _safe_float(value: str, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    value = value.strip()
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return None


def _safe_int(value: str, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    value = value.strip()
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return None


@dataclass
class FieldRefs:
    inputs: Dict[str, TextInput]
    toggles: Dict[str, CheckBox]
    spinners: Dict[str, Spinner]


class BrandozerGUI(App):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._output: TextInput
        self._settings_input: TextInput
        self._root_input: TextInput
        self._current_process = None
        self._lock = threading.Lock()
        self._command_fields: Dict[str, FieldRefs] = {}

    def build(self):
        root = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(10))

        header = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(36))
        header.add_widget(Label(text="BrandDozer Utility (Kivy)", size_hint_x=0.7, halign="left", valign="middle"))
        clear_btn = Button(text="Clear Output", size_hint_x=0.15)
        clear_btn.bind(on_release=lambda *_: self._clear_output())
        stop_btn = Button(text="Stop Command", size_hint_x=0.15)
        stop_btn.bind(on_release=lambda *_: self._stop_process())
        header.add_widget(clear_btn)
        header.add_widget(stop_btn)
        root.add_widget(header)

        root.add_widget(self._build_global_settings())
        root.add_widget(self._build_tabs())
        root.add_widget(self._build_output_area())

        return root

    # ------------------------------------------------------------------ UI builders
    def _build_global_settings(self) -> BoxLayout:
        layout = BoxLayout(orientation="vertical", size_hint_y=None, height=dp(110), spacing=dp(6))
        layout.add_widget(Label(text="Global settings", size_hint_y=None, height=dp(20), halign="left", valign="middle"))
        grid = GridLayout(cols=2, spacing=dp(6), size_hint_y=None)
        grid.bind(minimum_height=grid.setter("height"))

        self._settings_input = TextInput(text="coolcrypto_dashboard.settings", multiline=False)
        self._root_input = TextInput(text="", multiline=False, hint_text="Optional project root path")

        grid.add_widget(Label(text="Django settings module"))
        grid.add_widget(self._settings_input)
        grid.add_widget(Label(text="Project root (optional)"))
        grid.add_widget(self._root_input)

        layout.add_widget(grid)
        return layout

    def _build_tabs(self) -> TabbedPanel:
        tabs = TabbedPanel(do_default_tab=False)
        tabs.add_widget(self._tab_start())
        tabs.add_widget(self._tab_runs())
        tabs.add_widget(self._tab_status())
        tabs.add_widget(self._tab_workflow())
        tabs.add_widget(self._tab_tail())
        tabs.add_widget(self._tab_stop())
        tabs.add_widget(self._tab_projects())
        tabs.add_widget(self._tab_accounts())
        tabs.add_widget(self._tab_users())
        tabs.add_widget(self._tab_prompt())
        return tabs

    def _build_output_area(self) -> BoxLayout:
        layout = BoxLayout(orientation="vertical", size_hint_y=0.42)
        layout.add_widget(Label(text="Output", size_hint_y=None, height=dp(20), halign="left", valign="middle"))
        self._output = TextInput(readonly=True, multiline=True, text="", font_size=dp(12))
        layout.add_widget(self._output)
        return layout

    # ------------------------------------------------------------------ Tab helpers
    def _tab_start(self) -> TabbedPanelItem:
        item = TabbedPanelItem(text="start")
        layout, refs = self._form_layout(
            [
                ("prompt", "Prompt (required)", True),
                ("project_id", "Project id", False),
                ("name", "Project name", False),
                ("default_prompt", "Default prompt", True),
                ("run_id", "Run id", False),
                ("github_account_id", "GitHub account id", False),
                ("github_username", "GitHub username", False),
                ("inline_timeout", "Inline timeout seconds", False, "5.0"),
                ("codex_model", "Codex model override", False),
                ("smoke_test_cmd", "Solo smoke test command", False),
            ],
            toggles=[
                ("acceptance_required", "Acceptance required", True),
                ("research", "Research mode", False),
            ],
            spinners=[
                ("team_mode", "Team mode", ["full", "solo"], "full"),
                ("mode", "Mode", ["auto", "new", "existing"], "auto"),
                ("codex_reasoning", "Reasoning effort", ["medium", "high", "extra_high", "low"], "medium"),
            ],
        )
        refs.inputs["prompt"].hint_text = "Enter delivery prompt"
        refs.inputs["prompt"].height = dp(90)
        refs.inputs["prompt"].multiline = True
        run_btn = Button(text="Run start", size_hint_y=None, height=dp(40))
        run_btn.bind(on_release=lambda *_: self._run_start(refs))
        layout.add_widget(run_btn)
        item.add_widget(self._scrollable(layout))
        self._command_fields["start"] = refs
        return item

    def _tab_runs(self) -> TabbedPanelItem:
        item = TabbedPanelItem(text="runs")
        layout, refs = self._form_layout(
            [
                ("project_id", "Project id", False),
                ("limit", "Limit", False, "10"),
            ]
        )
        run_btn = Button(text="Run runs", size_hint_y=None, height=dp(40))
        run_btn.bind(on_release=lambda *_: self._run_runs(refs))
        layout.add_widget(run_btn)
        item.add_widget(self._scrollable(layout))
        self._command_fields["runs"] = refs
        return item

    def _tab_status(self) -> TabbedPanelItem:
        item = TabbedPanelItem(text="status")
        layout, refs = self._form_layout([("run_id", "Run id (required)", False)])
        run_btn = Button(text="Run status", size_hint_y=None, height=dp(40))
        run_btn.bind(on_release=lambda *_: self._run_status(refs))
        layout.add_widget(run_btn)
        item.add_widget(self._scrollable(layout))
        self._command_fields["status"] = refs
        return item

    def _tab_workflow(self) -> TabbedPanelItem:
        item = TabbedPanelItem(text="workflow")
        layout, refs = self._form_layout(
            [
                ("run_id", "Run id", False),
                ("history_limit", "History limit", False, "20"),
            ],
            toggles=[("show_gates", "Show gates", False)],
        )
        run_btn = Button(text="Run workflow", size_hint_y=None, height=dp(40))
        run_btn.bind(on_release=lambda *_: self._run_workflow(refs))
        layout.add_widget(run_btn)
        item.add_widget(self._scrollable(layout))
        self._command_fields["workflow"] = refs
        return item

    def _tab_tail(self) -> TabbedPanelItem:
        item = TabbedPanelItem(text="tail")
        layout, refs = self._form_layout(
            [
                ("run_id", "Run id (required)", False),
                ("lines", "Lines", False, "80"),
            ]
        )
        run_btn = Button(text="Run tail", size_hint_y=None, height=dp(40))
        run_btn.bind(on_release=lambda *_: self._run_tail(refs))
        layout.add_widget(run_btn)
        item.add_widget(self._scrollable(layout))
        self._command_fields["tail"] = refs
        return item

    def _tab_stop(self) -> TabbedPanelItem:
        item = TabbedPanelItem(text="stop")
        layout, refs = self._form_layout([("run_id", "Run id (required)", False)])
        run_btn = Button(text="Run stop", size_hint_y=None, height=dp(40))
        run_btn.bind(on_release=lambda *_: self._run_stop(refs))
        layout.add_widget(run_btn)
        item.add_widget(self._scrollable(layout))
        self._command_fields["stop"] = refs
        return item

    def _tab_projects(self) -> TabbedPanelItem:
        item = TabbedPanelItem(text="projects")
        layout, refs = self._form_layout([("limit", "Limit", False, "20")])
        run_btn = Button(text="Run projects", size_hint_y=None, height=dp(40))
        run_btn.bind(on_release=lambda *_: self._run_projects(refs))
        layout.add_widget(run_btn)
        item.add_widget(self._scrollable(layout))
        self._command_fields["projects"] = refs
        return item

    def _tab_accounts(self) -> TabbedPanelItem:
        item = TabbedPanelItem(text="accounts")
        layout, refs = self._form_layout(
            [
                ("username", "Username", False),
                ("limit", "Limit", False, "20"),
            ]
        )
        run_btn = Button(text="Run accounts", size_hint_y=None, height=dp(40))
        run_btn.bind(on_release=lambda *_: self._run_accounts(refs))
        layout.add_widget(run_btn)
        item.add_widget(self._scrollable(layout))
        self._command_fields["accounts"] = refs
        return item

    def _tab_users(self) -> TabbedPanelItem:
        item = TabbedPanelItem(text="users")
        layout, refs = self._form_layout([("limit", "Limit", False, "20")])
        run_btn = Button(text="Run users", size_hint_y=None, height=dp(40))
        run_btn.bind(on_release=lambda *_: self._run_users(refs))
        layout.add_widget(run_btn)
        item.add_widget(self._scrollable(layout))
        self._command_fields["users"] = refs
        return item

    def _tab_prompt(self) -> TabbedPanelItem:
        item = TabbedPanelItem(text="prompt")
        layout, refs = self._form_layout(
            [
                ("run_id", "Run id (required)", False),
                ("prompt", "Prompt (required)", True),
            ]
        )
        refs.inputs["prompt"].height = dp(90)
        refs.inputs["prompt"].multiline = True
        run_btn = Button(text="Run prompt", size_hint_y=None, height=dp(40))
        run_btn.bind(on_release=lambda *_: self._run_prompt(refs))
        layout.add_widget(run_btn)
        item.add_widget(self._scrollable(layout))
        self._command_fields["prompt"] = refs
        return item

    # ------------------------------------------------------------------ Form elements
    def _form_layout(
        self,
        inputs: List[tuple[str, str, bool, Optional[str] | None]],
        toggles: Optional[List[tuple[str, str, bool]]] = None,
        spinners: Optional[List[tuple[str, str, List[str], str]]] = None,
    ) -> tuple[BoxLayout, FieldRefs]:
        layout = BoxLayout(orientation="vertical", spacing=dp(6), padding=dp(6))
        grid = GridLayout(cols=2, spacing=dp(6), size_hint_y=None)
        grid.bind(minimum_height=grid.setter("height"))

        input_refs: Dict[str, TextInput] = {}
        toggle_refs: Dict[str, CheckBox] = {}
        spinner_refs: Dict[str, Spinner] = {}

        for name, label, multiline, default in inputs:
            grid.add_widget(Label(text=label))
            text = TextInput(text=default or "", multiline=multiline)
            input_refs[name] = text
            grid.add_widget(text)

        if spinners:
            for name, label, values, default in spinners:
                grid.add_widget(Label(text=label))
                spinner = Spinner(text=default, values=values)
                spinner_refs[name] = spinner
                grid.add_widget(spinner)

        if toggles:
            for name, label, default in toggles:
                grid.add_widget(Label(text=label))
                box = CheckBox(active=default)
                toggle_refs[name] = box
                grid.add_widget(box)

        layout.add_widget(grid)
        return layout, FieldRefs(inputs=input_refs, toggles=toggle_refs, spinners=spinner_refs)

    def _scrollable(self, content: BoxLayout) -> ScrollView:
        scroll = ScrollView()
        content.size_hint_y = None
        content.bind(minimum_height=content.setter("height"))
        scroll.add_widget(content)
        return scroll

    # ------------------------------------------------------------------ Command builders
    def _base_command(self) -> List[str]:
        cmd = [sys.executable, str(BRANDOZER_BIN)]
        settings = (self._settings_input.text or "").strip()
        if settings:
            cmd.extend(["--settings", settings])
        root = (self._root_input.text or "").strip()
        if root:
            cmd.extend(["--root", root])
        return cmd

    def _run_start(self, refs: FieldRefs) -> None:
        prompt = refs.inputs["prompt"].text.strip()
        if not prompt:
            self._append_output("ERROR: start requires a prompt.\n")
            return
        cmd = self._base_command() + ["start", prompt]
        self._maybe_add(cmd, "--project-id", refs.inputs["project_id"].text)
        self._maybe_add(cmd, "--name", refs.inputs["name"].text)
        cmd.extend(["--team-mode", refs.spinners["team_mode"].text])
        cmd.extend(["--mode", refs.spinners["mode"].text])
        self._maybe_add(cmd, "--default-prompt", refs.inputs["default_prompt"].text)
        self._maybe_add(cmd, "--run-id", refs.inputs["run_id"].text)
        if not refs.toggles["acceptance_required"].active:
            cmd.append("--no-acceptance")
        self._maybe_add(cmd, "--github-account-id", refs.inputs["github_account_id"].text)
        self._maybe_add(cmd, "--github-username", refs.inputs["github_username"].text)
        self._maybe_add(cmd, "--codex-model", refs.inputs["codex_model"].text)
        cmd.extend(["--codex-reasoning", refs.spinners["codex_reasoning"].text])
        self._maybe_add(cmd, "--smoke-test-cmd", refs.inputs["smoke_test_cmd"].text)
        inline_timeout = _safe_float(refs.inputs["inline_timeout"].text, 5.0)
        if inline_timeout is None:
            self._append_output("ERROR: inline timeout must be a number.\n")
            return
        cmd.extend(["--inline-timeout", str(inline_timeout)])
        if refs.toggles["research"].active:
            cmd.append("--research")
        self._run_command(cmd)

    def _run_runs(self, refs: FieldRefs) -> None:
        cmd = self._base_command() + ["runs"]
        self._maybe_add(cmd, "--project-id", refs.inputs["project_id"].text)
        limit = _safe_int(refs.inputs["limit"].text, 10)
        if limit is None:
            self._append_output("ERROR: limit must be an integer.\n")
            return
        cmd.extend(["--limit", str(limit)])
        self._run_command(cmd)

    def _run_status(self, refs: FieldRefs) -> None:
        run_id = refs.inputs["run_id"].text.strip()
        if not run_id:
            self._append_output("ERROR: status requires a run id.\n")
            return
        cmd = self._base_command() + ["status", run_id]
        self._run_command(cmd)

    def _run_workflow(self, refs: FieldRefs) -> None:
        cmd = self._base_command() + ["workflow"]
        self._maybe_add(cmd, "--run-id", refs.inputs["run_id"].text)
        if refs.toggles.get("show_gates") and refs.toggles["show_gates"].active:
            cmd.append("--show-gates")
        history = _safe_int(refs.inputs["history_limit"].text, 20)
        if history is None:
            self._append_output("ERROR: history limit must be an integer.\n")
            return
        cmd.extend(["--history-limit", str(history)])
        self._run_command(cmd)

    def _run_tail(self, refs: FieldRefs) -> None:
        run_id = refs.inputs["run_id"].text.strip()
        if not run_id:
            self._append_output("ERROR: tail requires a run id.\n")
            return
        cmd = self._base_command() + ["tail", run_id]
        lines = _safe_int(refs.inputs["lines"].text, 80)
        if lines is None:
            self._append_output("ERROR: lines must be an integer.\n")
            return
        cmd.extend(["--lines", str(lines)])
        self._run_command(cmd)

    def _run_stop(self, refs: FieldRefs) -> None:
        run_id = refs.inputs["run_id"].text.strip()
        if not run_id:
            self._append_output("ERROR: stop requires a run id.\n")
            return
        cmd = self._base_command() + ["stop", run_id]
        self._run_command(cmd)

    def _run_projects(self, refs: FieldRefs) -> None:
        cmd = self._base_command() + ["projects"]
        limit = _safe_int(refs.inputs["limit"].text, 20)
        if limit is None:
            self._append_output("ERROR: limit must be an integer.\n")
            return
        cmd.extend(["--limit", str(limit)])
        self._run_command(cmd)

    def _run_accounts(self, refs: FieldRefs) -> None:
        cmd = self._base_command() + ["accounts"]
        self._maybe_add(cmd, "--username", refs.inputs["username"].text)
        limit = _safe_int(refs.inputs["limit"].text, 20)
        if limit is None:
            self._append_output("ERROR: limit must be an integer.\n")
            return
        cmd.extend(["--limit", str(limit)])
        self._run_command(cmd)

    def _run_users(self, refs: FieldRefs) -> None:
        cmd = self._base_command() + ["users"]
        limit = _safe_int(refs.inputs["limit"].text, 20)
        if limit is None:
            self._append_output("ERROR: limit must be an integer.\n")
            return
        cmd.extend(["--limit", str(limit)])
        self._run_command(cmd)

    def _run_prompt(self, refs: FieldRefs) -> None:
        run_id = refs.inputs["run_id"].text.strip()
        prompt = refs.inputs["prompt"].text.strip()
        if not run_id or not prompt:
            self._append_output("ERROR: prompt requires run id and prompt text.\n")
            return
        cmd = self._base_command() + ["prompt", run_id, prompt]
        self._run_command(cmd)

    # ------------------------------------------------------------------ Process handling
    def _maybe_add(self, cmd: List[str], flag: str, value: str) -> None:
        value = (value or "").strip()
        if value:
            cmd.extend([flag, value])

    def _append_output(self, text: str) -> None:
        self._output.text += text
        self._output.cursor = (0, len(self._output.text.splitlines()))

    def _clear_output(self) -> None:
        self._output.text = ""

    def _stop_process(self) -> None:
        with self._lock:
            proc = self._current_process
        if proc and proc.poll() is None:
            proc.terminate()
            self._append_output("\n[stopping process]\n")

    def _run_command(self, cmd: List[str]) -> None:
        if not BRANDOZER_BIN.exists():
            self._append_output(f"ERROR: {BRANDOZER_BIN} not found.\n")
            return
        self._append_output(f"\n$ {' '.join(cmd)}\n")
        thread = threading.Thread(target=self._run_command_thread, args=(cmd,), daemon=True)
        thread.start()

    def _run_command_thread(self, cmd: List[str]) -> None:
        import subprocess

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
        except Exception as exc:
            Clock.schedule_once(lambda *_: self._append_output(f"ERROR: failed to start process: {exc}\n"))
            return

        with self._lock:
            self._current_process = proc

        if proc.stdout:
            for line in proc.stdout:
                Clock.schedule_once(lambda *_: self._append_output(line))

        code = proc.wait()
        Clock.schedule_once(lambda *_: self._append_output(f"[exit code {code}]\n"))
        with self._lock:
            if self._current_process is proc:
                self._current_process = None


def main() -> int:
    BrandozerGUI().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
