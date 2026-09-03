"""A window for the trading loop: watch it work, and talk to it while it does.

Two things this replaces. The loop's output lived in a console window that
could not be scrolled back through usefully, and the only way to steer the
agent was Tell.ps1 from a second terminal. Both live here now: the log tails
itself in the top pane, and anything typed below is queued for the next pass.

Messages go to data/agent_inbox.md, which the loop reads at the top of each
pass and archives, so a note written mid-pass lands on the next one and is
never delivered twice.

Nothing here blocks on a clock. The tail thread reads whatever has been
appended since it last looked; a pass that runs for two hours simply keeps
streaming.

    python scripts/loop_console.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import dearpygui.dearpygui as dpg

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "GetToLiveTrading.log"
INBOX = ROOT / "data" / "agent_inbox.md"
ARCHIVE = ROOT / "data" / "agent_inbox_archive.md"
SCORES = ROOT / "data" / "pass_scores.json"
PROMPT = ROOT / "data" / "behavior_prompt.md"

MAX_LINES = 4000          # keep the pane responsive on a long run
POLL_SEC = 0.5

# Everything the UI reads is produced on a worker thread and stored here.
#
# The render loop must never block. It used to call _loop_running() directly,
# which shells out to PowerShell and takes hundreds of milliseconds -- so
# every five seconds the window froze mid-frame and stopped responding to
# input. Same for the file reads. The UI thread now only reads these values.
_state = {
    "pos": 0,
    "lines": [],
    "stop": False,
    "running": False,
    "pending": 0,
    "score": "no passes scored yet",
    "lock": None,
}


# ----------------------------------------------------------------- tail --

def _tail_worker() -> None:
    """Stream new bytes from the log without ever re-reading the whole file."""
    while not _state["stop"]:
        _tail_once()
        time.sleep(POLL_SEC)


def _tail_once() -> None:
    try:
        if LOG.exists():
            size = LOG.stat().st_size
            # Truncated or rotated: start over from the top.
            if size < _state["pos"]:
                _state["pos"] = 0
            if size > _state["pos"]:
                with LOG.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(_state["pos"])
                    chunk = fh.read()
                    _state["pos"] = fh.tell()
                for line in chunk.splitlines():
                    if line.strip():
                        _state["lines"].append(line.rstrip())
                if len(_state["lines"]) > MAX_LINES:
                    del _state["lines"][:-MAX_LINES]
    except Exception:
        pass


def _status_worker() -> None:
    """Poll the slow things off the UI thread.

    Process listing shells out to PowerShell and the score file has to be
    parsed; neither belongs in a render frame. This runs on its own thread
    and only ever writes plain values into _state.
    """
    while not _state["stop"]:
        try:
            _state["running"] = _loop_running()
        except Exception:
            pass
        try:
            _state["pending"] = _pending_count()
        except Exception:
            pass
        try:
            _state["score"] = _latest_score()
        except Exception:
            pass
        # Slow on purpose: none of this changes fast, and each cycle costs a
        # process spawn.
        for _ in range(50):
            if _state["stop"]:
                return
            time.sleep(0.1)


def _pending_count() -> int:
    try:
        if not INBOX.exists():
            return 0
        return len([l for l in INBOX.read_text(encoding="utf-8",
                                               errors="replace").splitlines() if l.strip()])
    except Exception:
        return 0


def _latest_score() -> str:
    try:
        rows = json.loads(SCORES.read_text(encoding="utf-8"))
        if not rows:
            return "no passes scored yet"
        r = rows[-1]
        p = r.get("parts", {})
        return ("last pass %d/100  (progress %d, correctness %d, evidence %d, efficiency %d)"
                % (r.get("score", 0), p.get("progress", 0), p.get("correctness", 0),
                   p.get("evidence", 0), p.get("efficiency", 0)))
    except Exception:
        return "no scores yet"


def _loop_running() -> bool:
    """True when a GetToLiveTrading process is alive."""
    try:
        # CREATE_NO_WINDOW: without it every poll flashes a console window,
        # which is the exact behaviour we spent today removing elsewhere.
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "@(Get-CimInstance Win32_Process -Filter \"Name='powershell.exe'\" "
             "-ErrorAction SilentlyContinue | Where-Object "
             "{ $_.CommandLine -like '*GetToLiveTrading.ps1*' }).Count"],
            capture_output=True, text=True, creationflags=flags)
        return int((out.stdout or "0").strip() or 0) > 0
    except Exception:
        return False


# --------------------------------------------------------------- actions --

def send_message(sender=None, app_data=None, user_data=None) -> None:
    text = dpg.get_value("msg_input").strip()
    if not text:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        INBOX.parent.mkdir(parents=True, exist_ok=True)
        with INBOX.open("a", encoding="utf-8") as fh:
            fh.write("[%s] %s\n" % (stamp, text))
        dpg.set_value("msg_input", "")
        dpg.set_value("status_msg",
                      "queued at %s - delivered at the start of the next pass" % stamp)
    except Exception as exc:  # noqa: BLE001
        dpg.set_value("status_msg", "could not queue: %s" % exc)


def clear_inbox(sender=None, app_data=None, user_data=None) -> None:
    """Drop queued notes that have not been delivered yet."""
    try:
        if INBOX.exists():
            INBOX.unlink()
        dpg.set_value("status_msg", "cleared undelivered messages")
    except Exception as exc:  # noqa: BLE001
        dpg.set_value("status_msg", "could not clear: %s" % exc)


# ------------------------------------------------------------------- ui --

def build() -> None:
    dpg.create_context()

    with dpg.theme() as dark:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (18, 20, 24))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (12, 14, 17))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (205, 214, 224))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (38, 46, 58))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (54, 66, 82))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (26, 30, 37))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)

    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            dpg.add_text("R3V3N!R loop", color=(120, 200, 255))
            dpg.add_text("", tag="run_state")
            dpg.add_spacer(width=20)
            dpg.add_text("", tag="score_line", color=(150, 190, 150))

        dpg.add_separator()

        # --- live output -------------------------------------------------
        with dpg.child_window(tag="log_pane", height=-165, horizontal_scrollbar=True):
            dpg.add_text("waiting for output...", tag="log_text", wrap=0)

        dpg.add_separator()

        # --- message to the agent ---------------------------------------
        dpg.add_text("Message the agent (delivered at the start of the next pass):")
        dpg.add_input_text(tag="msg_input", multiline=True, height=60, width=-1,
                           hint="e.g. stop tuning money_button, get one trade settled first")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Send", width=110, callback=send_message)
            dpg.add_button(label="Clear queued", width=130, callback=clear_inbox)
            dpg.add_checkbox(label="follow tail", tag="follow", default_value=True)
            dpg.add_text("", tag="queued_count", color=(200, 180, 120))
        dpg.add_text("", tag="status_msg", color=(150, 170, 200))

    dpg.bind_theme(dark)
    dpg.create_viewport(title="R3V3N!R - trading loop console", width=1180, height=760)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("root", True)


def run() -> int:
    if not LOG.exists():
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.touch()

    build()
    threading.Thread(target=_tail_worker, daemon=True).start()
    threading.Thread(target=_status_worker, daemon=True).start()

    last_ui = 0.0

    # The render loop does no I/O and spawns no processes: it only copies
    # values the workers have already computed. That is what keeps the window
    # responsive while a pass runs for hours.
    while dpg.is_dearpygui_running():
        now = time.time()

        if now - last_ui > 0.25:
            last_ui = now
            dpg.set_value("log_text", "\n".join(_state["lines"][-MAX_LINES:])
                          or "waiting for output...")
            if dpg.get_value("follow"):
                dpg.set_y_scroll("log_pane", -1.0)

            n = _state["pending"]
            dpg.set_value("queued_count",
                          ("%d message(s) waiting" % n) if n else "")

            running = _state["running"]
            dpg.set_value("run_state", "RUNNING" if running else "NOT RUNNING")
            dpg.configure_item("run_state",
                               color=(120, 220, 140) if running else (230, 120, 120))
            dpg.set_value("score_line", _state["score"])

        dpg.render_dearpygui_frame()

    _state["stop"] = True
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
