from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional


def _fallback_log(source: str, message: str, severity: str, details: Optional[Dict[str, Any]]) -> None:
    """Lightweight stdout logger used when the async bus is unavailable."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    payload = f"[{stamp}] [{severity.upper()}] {source}: {message}"
    if details:
        payload += f" -> {details}"
    print(payload, file=sys.stderr if severity.lower() in {"error", "warning"} else sys.stdout, flush=True)


try:  # pragma: no cover - defensive: bus may not import early in boot.
    from services.logging_bus import log_message as _bus_log_message
except Exception:  # pragma: no cover
    _bus_log_message = None  # type: ignore[assignment]


def log_message(source: str, message: str, *, severity: str = "info", details: Optional[Dict[str, Any]] = None) -> None:
    """
    Proxy that prefers the shared logging bus but gracefully falls back to a
    synchronous printer so callers never raise NameError/ImportError.
    """
    if _bus_log_message is not None:
        try:
            _bus_log_message(source, message, severity=severity, details=details)
            return
        except Exception:
            # fall back to prevent log storms during shutdown/corruption
            pass
    _fallback_log(source, message, severity, details)
