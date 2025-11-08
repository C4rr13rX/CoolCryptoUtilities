from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class HeartbeatFile:
    """
    Lightweight heartbeat writer so external monitors can confirm the
    production manager is alive even when it runs inside the CLI harness.
    """

    def __init__(self, *, label: str, path: Optional[Path] = None) -> None:
        name = label.replace(" ", "_").lower()
        default_path = Path("logs") / f"{name}_heartbeat.json"
        self.path = (path or default_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._label = label

    def update(self, status: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "label": self._label,
            "status": status,
            "timestamp": int(time.time()),
        }
        if metadata:
            payload["meta"] = metadata
        data = json.dumps(payload, indent=2)
        with self._lock:
            self.path.write_text(data, encoding="utf-8")

    def clear(self) -> None:
        with self._lock:
            self.path.unlink(missing_ok=True)
