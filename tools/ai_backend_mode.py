"""Process-wide AI backend exclusivity controls."""
from __future__ import annotations

import os
import threading


FREELOADER_ALIASES = frozenset({
    "freeloader", "agentthefreeloader", "agent_the_freeloader", "free_models",
})

_ATF_ACTIVE = threading.Event()


def configured_backend(default: str = "") -> str:
    """Read the global C0d3rV2 backend from env or the encrypted vault."""
    raw = os.getenv("C0D3R_BACKEND") or os.getenv("BRANDDOZER_SESSION_PROVIDER") or ""
    if not raw:
        try:
            from tools.secret_vault import get_secret

            raw = get_secret("C0D3R_BACKEND") or ""
        except Exception:
            raw = ""
    value = str(raw or default).strip().lower().replace("-", "_").replace(" ", "_")
    return value


def activate_freeloader_mode() -> None:
    """Prevent any subsequent Wizard connection in this process."""
    _ATF_ACTIVE.set()


def set_freeloader_mode(active: bool) -> None:
    """Switch process-wide backend exclusivity after an explicit control change."""
    if active:
        _ATF_ACTIVE.set()
    else:
        _ATF_ACTIVE.clear()


def freeloader_mode_active() -> bool:
    explicit = os.getenv("AGENT_FREELOADER_EXCLUSIVE", "").strip().lower()
    if explicit in {"0", "false", "no", "off"}:
        return False
    configured = configured_backend()
    return _ATF_ACTIVE.is_set() or configured in FREELOADER_ALIASES


def deactivate_freeloader_mode_for_tests() -> None:
    """Test-only reset; production switches should restart workers."""
    _ATF_ACTIVE.clear()
