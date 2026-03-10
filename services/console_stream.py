from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Iterable

from services.logging_utils import log_message

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pattern to strip nested log-bus prefixes so we relay the raw message only.
_PREFIX_RE = re.compile(
    r"^(?:\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*\[\w+\]\s*[\w.\-]+:\s*)+",
)


def _strip_log_prefix(text: str) -> str:
    """Remove accumulated log-bus prefixes to prevent recursive nesting."""
    m = _PREFIX_RE.match(text)
    if m:
        inner = text[m.end():]
        return inner.strip() if inner.strip() else text
    return text


class _FileStreamer:
    def __init__(self, path: Path, label: str) -> None:
        self.path = path
        self.label = label
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._own_tag = f"{label}.console"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        thread = threading.Thread(target=self._loop, name=f"{self.label}-stream", daemon=True)
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        last_inode = None
        handle = None
        while not self._stop.is_set():
            try:
                if not self.path.exists():
                    time.sleep(1.0)
                    continue
                stat = self.path.stat()
                inode = getattr(stat, "st_ino", None)
                if handle is None or (inode is not None and inode != last_inode):
                    if handle:
                        handle.close()
                    handle = self.path.open("r", encoding="utf-8", errors="ignore")
                    handle.seek(0, os.SEEK_END)
                    last_inode = inode
                line = handle.readline()
                if not line:
                    time.sleep(0.4)
                    continue
                text = line.rstrip("\n")
                if not text:
                    continue
                # Skip lines that are already wrapped by our own tag to avoid
                # recursive nesting (console.log -> streamer -> log_message ->
                # print -> console.log -> ...).
                if f"{self._own_tag}:" in text:
                    continue
                # Strip any existing log-bus prefix so we relay only the payload.
                cleaned = _strip_log_prefix(text)
                if cleaned:
                    log_message(self._own_tag, cleaned)
            except Exception as exc:
                log_message(self._own_tag, f"stream error: {exc}", severity="warning")
                time.sleep(1.5)
        if handle:
            handle.close()


_STREAMERS: list[_FileStreamer] = []
_STARTED = False


def start_console_streams(paths: Iterable[tuple[str, str]] | None = None) -> None:
    """
    Begin tailing the guardian + production console logs so operators can watch
    the output directly in the Django runserver console.
    """
    global _STARTED
    if _STARTED:
        return
    if os.getenv("CONSOLE_STREAM_DISABLE") == "1":
        return
    default_paths = [
        ("guardian", str(_PROJECT_ROOT / "runtime" / "guardian" / "transcripts" / "guardian-session.log")),
        ("production", str(_PROJECT_ROOT / "logs" / "console.log")),
    ]
    for label, abs_path in paths or default_paths:
        streamer = _FileStreamer(Path(abs_path), label)
        streamer.start()
        _STREAMERS.append(streamer)
    _STARTED = True
