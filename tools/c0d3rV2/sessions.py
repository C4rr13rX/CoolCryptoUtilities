"""tools/c0d3rV2/sessions.py — Session factory for C0d3r V2.

Creates an AI session for the V2 orchestration engine.  The session
must expose a send(prompt, stream=False, system='') → str interface.

C0d3rV2 is the AGENT — it orchestrates tool use, planning, and
multi-step task execution.  It is NOT a backend; it consumes one.
The backend is the LLM that actually generates each step.

Backend priority (CLI --backend flag or env):
  1. wizard   — W1z4rDV1510n merged main node at localhost:8090,
                routed through /brain/chat (the Phase A-E substrate
                mounted alongside the legacy Web3 + cluster + wallet
                stack).  DEFAULT.  Set WIZARD_BRAIN_URL or
                WIZARD_BRAIN_CHAT_URL to override.  The standalone
                w1z4rd_brain_server binary at :8095 still works for
                backward compat — point WIZARD_BRAIN_CHAT_URL there
                if you prefer the isolated brain process.  The brain
                may be a §18 cluster head; it transparently pools
                state across N hosts.  Legacy port-8090 /neuro/ask is
                still reachable via LEGACY_NEURO_API=1.
  2. bedrock  — AWS Bedrock via C0d3rSession (fallback when wizard
                offline or for tasks needing transformer-style generation).
  3. claude   — Anthropic API direct (third fallback).
  4. openai   — OpenAI API (last resort).

The wizard backend is the strategic choice: as the brain trains on
engineering/research corpora, it replaces external APIs.  Bedrock /
claude / openai are the operational fallbacks while the brain catches
up.  This file is the session-factory layer; the cascading
availability check lives in `tools.ai_session.resolve_with_fallback`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _default_backend() -> str:
    raw = os.getenv("C0D3R_BACKEND", "wizard").strip().lower()
    legacy = {"codex": "wizard", "openai_codex": "wizard"}
    return legacy.get(raw, raw) or "wizard"


class SessionManager:
    """
    Factory wrapper that creates the right AI session for the V2 pipeline
    and exposes a uniform send() interface.
    """

    def __init__(
        self,
        *,
        backend: str | None = None,
        # Wizard-specific
        wizard_endpoint: str | None = None,
        # Bedrock-specific
        model: str | None = None,
        region: str | None = None,
        profile: str | None = None,
        reasoning_effort: str | None = None,
        # Shared
        transcript_dir: Path | None = None,
        workdir: str = "",
    ) -> None:
        self._backend = (backend or _default_backend()).lower().strip()
        self._session = self._create_session(
            backend=self._backend,
            wizard_endpoint=wizard_endpoint,
            model=model,
            region=region,
            profile=profile,
            reasoning_effort=reasoning_effort,
            transcript_dir=transcript_dir,
            workdir=workdir,
        )

    @staticmethod
    def _create_session(
        backend: str,
        *,
        wizard_endpoint: str | None,
        model: str | None,
        region: str | None,
        profile: str | None,
        reasoning_effort: str | None,
        transcript_dir: Path | None,
        workdir: str,
    ) -> Any:
        _BEDROCK = {"bedrock", "c0d3r", "coder"}
        _WIZARD  = {"wizard", "w1z4rd", "wizard_node", "local"}

        if backend in _WIZARD:
            from tools.wizard_session import WizardSession
            probe = WizardSession.probe(wizard_endpoint)
            if probe["online"]:
                return WizardSession(
                    session_name="c0d3rv2",
                    endpoint=wizard_endpoint,
                    transcript_dir=transcript_dir,
                    workdir=workdir,
                )
            # Node offline: fall back to Bedrock if configured.
            print(
                f"[c0d3rv2] W1z4rD node offline ({probe['error']}). "
                "Falling back to Bedrock.",
                flush=True,
            )
            backend = "bedrock"

        if backend in _BEDROCK or backend in {"openai"}:
            from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings
            settings = c0d3r_default_settings()
            if model:
                settings["model"] = model
            if region:
                settings["region"] = region
            if profile:
                settings["profile"] = profile
            if reasoning_effort:
                settings["reasoning_effort"] = reasoning_effort
            return C0d3rSession(
                session_name="c0d3rv2",
                transcript_dir=str(transcript_dir) if transcript_dir else None,
                workdir=workdir,
                **settings,
            )

        raise ValueError(
            f"Unknown C0d3rV2 backend: {backend!r}. "
            "Use 'wizard', 'bedrock', or 'openai'."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session(self) -> Any:
        return self._session

    @property
    def session_id(self) -> str:
        sid = getattr(self._session, "session_id", "")
        return str(sid) if sid else ""

    @property
    def model_id(self) -> str:
        get_id = getattr(self._session, "get_model_id", None)
        if callable(get_id):
            return get_id()
        return getattr(self._session, "MODEL_ID", "unknown")

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def send(self, prompt: str, *, stream: bool = False, system: str = "") -> str:
        return self._session.send(prompt=prompt, stream=stream, system=system) or ""
