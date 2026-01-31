from __future__ import annotations

import os
from typing import Any, Dict, Type

from tools.codex_session import CodexSession, codex_default_settings, codex_settings_for_role
from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings, c0d3r_settings_for_role


def session_provider_from_context(ctx: Dict[str, Any] | None = None) -> str:
    ctx = ctx or {}
    raw = ctx.get("session_provider") or ctx.get("provider") or os.getenv("BRANDDOZER_SESSION_PROVIDER") or "codex"
    provider = str(raw).strip().lower()
    return provider or "codex"


def get_session_class(provider: str) -> Type:
    provider = (provider or "codex").strip().lower()
    if provider in {"c0d3r", "coder", "bedrock"}:
        return C0d3rSession
    return CodexSession


def default_settings(provider: str) -> Dict[str, Any]:
    provider = (provider or "codex").strip().lower()
    if provider in {"c0d3r", "coder", "bedrock"}:
        return c0d3r_default_settings()
    return codex_default_settings()


def settings_for_role(provider: str, role: str | None = None) -> Dict[str, Any]:
    provider = (provider or "codex").strip().lower()
    if provider in {"c0d3r", "coder", "bedrock"}:
        return c0d3r_settings_for_role(role)
    return codex_settings_for_role(role)


__all__ = [
    "session_provider_from_context",
    "get_session_class",
    "default_settings",
    "settings_for_role",
]
