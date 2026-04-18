"""tools/ai_session.py — Central AI session factory.

Provider routing for all AI sessions across the site.

Supported providers (set via BRANDDOZER_SESSION_PROVIDER env or
context["session_provider"]):
  wizard   — W1z4rD Vision Node at localhost:8090 (DEFAULT)
  bedrock  — AWS Bedrock (Anthropic Claude via boto3)
  c0d3r    — alias for bedrock
  coder    — alias for bedrock
  openai   — OpenAI API (uses C0d3rSession with openai-compatible settings)

Codex CLI is no longer supported.  Any legacy "codex" references are
silently mapped to "wizard".
"""
from __future__ import annotations

import os
from typing import Any, Dict, Type

from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings, c0d3r_settings_for_role
from tools.wizard_session import WizardSession


_BEDROCK_ALIASES = {"c0d3r", "coder", "bedrock"}
_WIZARD_ALIASES  = {"wizard", "w1z4rd", "wizard_node", "local"}
_OPENAI_ALIASES  = {"openai", "gpt", "openai_api"}

# Legacy aliases: map old "codex" to wizard
_LEGACY_MAP: Dict[str, str] = {
    "codex": "wizard",
    "openai_codex": "wizard",
}


def _normalise(raw: str) -> str:
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    return _LEGACY_MAP.get(key, key)


def session_provider_from_context(ctx: Dict[str, Any] | None = None) -> str:
    ctx = ctx or {}
    raw = (
        ctx.get("session_provider")
        or ctx.get("provider")
        or os.getenv("BRANDDOZER_SESSION_PROVIDER")
        or "wizard"
    )
    return _normalise(str(raw))


def get_session_class(provider: str) -> Type:
    norm = _normalise(provider or "wizard")
    if norm in _BEDROCK_ALIASES or norm in _OPENAI_ALIASES:
        return C0d3rSession
    # wizard is default; also catches unknown values
    return WizardSession


def default_settings(provider: str) -> Dict[str, Any]:
    norm = _normalise(provider or "wizard")
    if norm in _BEDROCK_ALIASES or norm in _OPENAI_ALIASES:
        return c0d3r_default_settings()
    # WizardSession accepts no special settings object — return empty dict
    return {}


def settings_for_role(provider: str, role: str | None = None) -> Dict[str, Any]:
    norm = _normalise(provider or "wizard")
    if norm in _BEDROCK_ALIASES or norm in _OPENAI_ALIASES:
        return c0d3r_settings_for_role(role)
    # Wizard roles all share the same session (node handles all requests)
    return {}


def make_session(
    provider: str | None = None,
    *,
    session_name: str = "agent",
    role: str | None = None,
    model: str | None = None,
    workdir: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Convenience factory: create a session for the given provider.

    provider    "wizard" (default), "bedrock", "openai"
    session_name  Logical name for transcripts / logs.
    role        Optional role for model selection.
    model       Override model ID (Bedrock only).
    workdir     Working directory hint.
    """
    norm = _normalise(provider or "wizard")
    cls = get_session_class(norm)
    settings = settings_for_role(norm, role)
    if model and norm in _BEDROCK_ALIASES:
        settings["model"] = model
    settings.update({k: v for k, v in kwargs.items() if v is not None})
    return cls(session_name=session_name, workdir=workdir or "", **settings)


__all__ = [
    "session_provider_from_context",
    "get_session_class",
    "default_settings",
    "settings_for_role",
    "make_session",
]
