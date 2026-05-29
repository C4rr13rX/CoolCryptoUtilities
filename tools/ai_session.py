"""tools/ai_session.py — Central AI session factory.

Provider routing for all AI sessions across the site.

Supported providers (set via BRANDDOZER_SESSION_PROVIDER env or
context["session_provider"]):
  wizard   — W1z4rDV1510n merged main node at localhost:8090 routed
             through /brain/chat (the Phase A–E substrate is mounted
             under /brain/* on the main node).  DEFAULT, preferred.
             Override URL via WIZARD_BRAIN_URL / WIZARD_BRAIN_CHAT_URL.
             The standalone w1z4rd_brain_server (port :8095) is still
             accepted for backward compat; set WIZARD_BRAIN_CHAT_URL
             accordingly if you're running it.  Legacy port-8090
             /neuro/ask is still reachable via LEGACY_NEURO_API=1.
  bedrock  — AWS Bedrock (Anthropic Claude via boto3 — C0d3rSession is
             the wrapper class; this is JUST the LLM backend, not the
             C0d3rV2 agent).
  c0d3r    — The C0d3rV2 AGENT.  Not an LLM backend — an agent that
             orchestrates work.  Defaults its OWN backend chain
             starting at wizard, so requesting provider="c0d3r"
             resolves to the wizard-first cascade.
  coder    — Alias for c0d3r (same agent reference).
  claude   — Anthropic API direct (ClaudeSession).
  openai   — OpenAI API (OpenAISession) — last resort.

Preferred fallback order when none is specified or when `auto` is
requested: wizard → bedrock → claude → openai.

Codex CLI is no longer supported.  Any legacy "codex" references are
silently mapped to "wizard".
"""
from __future__ import annotations

import os
from typing import Any, Dict, Type

from tools.wizard_session import WizardSession
from tools.openai_session import OpenAISession
from tools.claude_session import ClaudeSession

# C0d3rSession is imported lazily because its module pulls in
# `services.web_search`, which is only resolvable when Django (and the
# project's `services/` package) is on sys.path.  CLI consumers of
# ai_session shouldn't be forced to bootstrap Django just to route
# through wizard/openai/claude.
def _import_c0d3r():
    from tools.c0d3r_session import (  # type: ignore
        C0d3rSession, c0d3r_default_settings, c0d3r_settings_for_role,
    )
    return C0d3rSession, c0d3r_default_settings, c0d3r_settings_for_role


_BEDROCK_ALIASES = {"bedrock"}
# C0d3rV2 is the AGENT, not an LLM backend.  When the user requests
# "c0d3r" or "coder" as a provider, they mean "use the C0d3rV2 agent,"
# which itself runs its own backend cascade starting from wizard.  So
# at the session-factory level we resolve c0d3r/coder to the wizard
# backend — the agent's preferred LLM — rather than to the Bedrock
# wrapper class.  Agent-mode routing in wizard_chat/agent_runner.py is
# what actually invokes the C0d3rV2 orchestrator; this factory just
# answers "which LLM session class do we construct for that provider
# string?"
_WIZARD_ALIASES  = {"wizard", "w1z4rd", "wizard_node", "local",
                    "c0d3r", "coder"}
_CLAUDE_ALIASES  = {"claude", "anthropic", "claude_api"}
_OPENAI_ALIASES  = {"openai", "gpt", "openai_api"}

# Cascading preference order — wizard first, OpenAI last by policy.
_FALLBACK_ORDER  = ("wizard", "bedrock", "claude", "openai")

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
    if norm in _OPENAI_ALIASES:
        return OpenAISession
    if norm in _CLAUDE_ALIASES:
        return ClaudeSession
    if norm in _BEDROCK_ALIASES:
        C0d3rSession, _, _ = _import_c0d3r()
        return C0d3rSession
    # wizard is default; also catches unknown values
    return WizardSession


def default_settings(provider: str) -> Dict[str, Any]:
    norm = _normalise(provider or "wizard")
    if norm in _OPENAI_ALIASES or norm in _CLAUDE_ALIASES:
        return {}
    if norm in _BEDROCK_ALIASES:
        _, c0d3r_default_settings, _ = _import_c0d3r()
        return c0d3r_default_settings()
    return {}


def settings_for_role(provider: str, role: str | None = None) -> Dict[str, Any]:
    norm = _normalise(provider or "wizard")
    if norm in _OPENAI_ALIASES or norm in _CLAUDE_ALIASES:
        return {}
    if norm in _BEDROCK_ALIASES:
        _, _, c0d3r_settings_for_role = _import_c0d3r()
        return c0d3r_settings_for_role(role)
    return {}


def _probe(provider: str) -> bool:
    """Cheap availability check.  True means we believe the backend
    will accept a real request right now."""
    norm = _normalise(provider)
    try:
        if norm in _WIZARD_ALIASES or norm == "wizard":
            return bool(WizardSession.probe().get("online"))
        if norm in _BEDROCK_ALIASES:
            # C0d3rSession doesn't expose a cheap probe; treat the
            # presence of AWS creds as the gate.
            return bool(os.getenv("AWS_ACCESS_KEY_ID")
                          or os.getenv("AWS_PROFILE")
                          or os.getenv("AWS_DEFAULT_PROFILE"))
        if norm in _CLAUDE_ALIASES:
            # Don't burn a real Anthropic API call here — just check
            # whether a key is resolvable.  Cheaper than probe().
            try:
                from tools.secret_vault import get_secret
                if get_secret("ANTHROPIC_API_KEY"):
                    return True
            except Exception:
                pass
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        if norm in _OPENAI_ALIASES:
            try:
                from tools.secret_vault import get_secret
                if get_secret("OPENAI_API_KEY"):
                    return True
            except Exception:
                pass
            return bool(os.getenv("OPENAI_API_KEY"))
    except Exception:
        return False
    return False


def resolve_with_fallback(preferred: str | None = None) -> str:
    """Return the first available provider, honouring the global
    preference order: wizard → bedrock → claude → openai.

    If `preferred` is supplied and available, it wins.  Otherwise we
    walk the fallback chain and return the first one that probes
    online.  Falls back to "wizard" as a last resort even if nothing
    probes — the caller will then surface a clear error.
    """
    if preferred:
        norm = _normalise(preferred)
        if _probe(norm):
            return norm
    for candidate in _FALLBACK_ORDER:
        if _probe(candidate):
            return candidate
    return "wizard"


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
