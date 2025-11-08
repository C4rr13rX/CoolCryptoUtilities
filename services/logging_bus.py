from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional
import os


@dataclass
class LogRecord:
    ts: float
    source: str
    severity: str
    message: str
    details: Optional[dict] = None


class LogBus:
    """
    Minimal logging relay that funnels messages through a single queue so the console
    (CLI or Django websocket) receives them in-order even when many threads emit logs.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[Optional[LogRecord]] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="log-bus", daemon=True)
        self._thread.start()
        log_path = Path(os.getenv("LOG_BUS_PATH", "logs/system.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = log_path
        self._file_lock = Lock()

    def publish(self, source: str, message: str, *, severity: str = "info", details: Optional[dict] = None) -> None:
        record = LogRecord(ts=time.time(), source=source, severity=severity.lower(), message=message, details=details)
        self._queue.put(record)

    def stop(self, timeout: float = 2.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while True:
            record = self._queue.get()
            if record is None:
                break
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.ts))
            payload = f"[{stamp}] [{record.severity.upper()}] {record.source}: {record.message}"
            if record.details:
                payload += f" -> {record.details}"
            print(payload, file=sys.stdout, flush=True)
            with self._file_lock:
                with self._log_path.open("a", encoding="utf-8") as handle:
                    handle.write(payload + "\n")


_GLOBAL_BUS = LogBus()


def log_message(source: str, message: str, *, severity: str = "info", details: Optional[dict] = None) -> None:
    _GLOBAL_BUS.publish(source, message, severity=severity, details=details)
