"""
Shared utilities for c0d3r V2.

Context building, memory management, and orchestration live in their own
modules.  This file holds only cross-cutting utility functions.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Import-path bootstrapping
# ------------------------------------------------------------------

V2_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = V2_ROOT.parent.parent
WEB_ROOT = PROJECT_ROOT / "web"

for _p in (str(PROJECT_ROOT), str(V2_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if WEB_ROOT.exists() and str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_DOCUMENT_EXTENSIONS = {
    ".pdf", ".csv", ".doc", ".docx",
    ".xls", ".xlsx", ".html", ".txt", ".md",
}

_HEARTBEAT_SESSION = None
_HEARTBEAT_MODEL_ID: str = ""
_HEARTBEAT_INTERVAL_S: float = 30.0 * 60.0

# ------------------------------------------------------------------
# Environment helpers
# ------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


# ------------------------------------------------------------------
# Runtime paths
# ------------------------------------------------------------------


def _runtime_root() -> Path:
    override = os.getenv("C0D3R_RUNTIME_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (PROJECT_ROOT / "runtime" / "c0d3r").resolve()


def _runtime_path(*parts: str) -> Path:
    return _runtime_root().joinpath(*parts)


# ------------------------------------------------------------------
# Django
# ------------------------------------------------------------------


def _ensure_django_ready() -> bool:
    try:
        import os as _os
        if not _os.getenv("DJANGO_SETTINGS_MODULE"):
            _os.environ["DJANGO_SETTINGS_MODULE"] = (
                "coolcrypto_dashboard.settings"
            )
        import django
        django.setup()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------
# File utilities
# ------------------------------------------------------------------


def _is_supported_document(path: Path) -> bool:
    try:
        return path.suffix.lower() in _DOCUMENT_EXTENSIONS
    except Exception:
        return False


def _resolve_image_paths(
    paths: list[str] | None, workdir: Path,
) -> list[str]:
    if not paths:
        return []
    resolved: list[str] = []
    for raw in paths:
        if not raw:
            continue
        try:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = (workdir / candidate).resolve()
            if candidate.exists() and candidate.is_file():
                resolved.append(str(candidate))
        except Exception:
            continue
    return resolved


def _resolve_document_paths(
    paths: list[str] | None, workdir: Path,
) -> list[str]:
    if not paths:
        return []
    resolved: list[str] = []
    for raw in paths:
        if not raw:
            continue
        try:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = (workdir / candidate).resolve()
            if (
                candidate.exists()
                and candidate.is_file()
                and _is_supported_document(candidate)
            ):
                resolved.append(str(candidate))
        except Exception:
            continue
    return resolved


# ------------------------------------------------------------------
# Text utilities
# ------------------------------------------------------------------


def _strip_context_block(prompt: str) -> str:
    if not prompt:
        return ""
    for marker in ("User request:", "Latest user question:"):
        if marker in prompt:
            return prompt.split(marker, 1)[-1].strip()
    if "User:\n" in prompt:
        return prompt.split("User:\n", 1)[-1].strip()
    return prompt.strip()


# ------------------------------------------------------------------
# Session heartbeat
# ------------------------------------------------------------------


def _init_heartbeat(session) -> None:
    global _HEARTBEAT_SESSION, _HEARTBEAT_MODEL_ID, _HEARTBEAT_INTERVAL_S
    try:
        from tools.c0d3r_session import ROLE_FALLBACK_MODEL
    except Exception:
        ROLE_FALLBACK_MODEL = {}
    _HEARTBEAT_SESSION = session
    fallback = ""
    try:
        fallback = ROLE_FALLBACK_MODEL.get("worker", "")
    except Exception:
        fallback = ""
    _HEARTBEAT_MODEL_ID = (
        os.getenv("C0D3R_HEARTBEAT_MODEL")
        or fallback
        or getattr(session, "get_model_id", lambda: "")()
    )
    _HEARTBEAT_INTERVAL_S = _env_float("C0D3R_HEARTBEAT_MINUTES", 30.0) * 60.0


# ------------------------------------------------------------------
# Workdir utilities
# ------------------------------------------------------------------


def _maybe_retarget_project(prompt: str, workdir: Path) -> Path | None:
    """
    If prompt references a project name and we're not already in it,
    retarget to that folder under the Projects directory.
    """
    lower = (prompt or "").lower()
    if "project" not in lower:
        return None
    if not any(
        phrase in lower
        for phrase in (
            "update project", "work on", "open project",
            "continue project", "go to project", "switch to project",
        )
    ):
        return None
    projects_root = Path("C:\\Users\\Adam\\Projects")
    if not projects_root.exists():
        return None
    try:
        for child in projects_root.iterdir():
            if not child.is_dir():
                continue
            if child.name.lower() in lower:
                if not str(workdir).lower().startswith(str(child).lower()):
                    return child.resolve()
    except Exception:
        return None
    return None
