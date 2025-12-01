from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterable

from services.logging_utils import log_message


class _FileStreamer:
    def __init__(self, path: Path, label: str) -> None:
        self.path = path
        self.label = label
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

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
                if text:
                    log_message(f"{self.label}.console", text)
            except Exception as exc:
                log_message(f"{self.label}.console", f"stream error: {exc}", severity="warning")
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
        ("guardian", "runtime/guardian/transcripts/guardian-session.log"),
        ("production", "logs/console.log"),
    ]
    for label, rel_path in paths or default_paths:
        streamer = _FileStreamer(Path(rel_path), label)
        streamer.start()
        _STREAMERS.append(streamer)
    _STARTED = True
