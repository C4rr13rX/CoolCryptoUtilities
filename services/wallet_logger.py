from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict

LOG_PATH = Path("logs/waallet_lig.log")
_LOCK = threading.Lock()


def _serialize(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    return repr(value)


def wallet_log(event: str, **details: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        "details": _serialize(details),
    }
    line = json.dumps(entry, ensure_ascii=False)
    with _LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
