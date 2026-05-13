"""tools/openai_session.py — OpenAI Chat Completions AI session.

Drop-in session that talks to the OpenAI API via the official SDK.
Same shape as WizardSession / C0d3rSession so it's swappable
everywhere a session is constructed:

    session = OpenAISession(session_name="...", transcript_dir=Path(...))
    answer  = session.send("Hello", system="You are helpful")

Requires the `openai` package (already in requirements.txt) and the
OPENAI_API_KEY env var.  Model + temperature configurable via
constructor args or env vars (OPENAI_MODEL, OPENAI_TEMPERATURE).

Originally the C0d3rV2 stack listed "openai" as a backend but routed
it back to Bedrock as a "future" placeholder.  This module is what
makes that backend real.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence


DEFAULT_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS  = int(os.getenv("OPENAI_MAX_TOKENS", "2048"))
DEFAULT_TIMEOUT_S   = float(os.getenv("OPENAI_TIMEOUT_S", "60"))


class OpenAISession:
    """
    Conforms to the duck-type interface used by C0d3rV2's
    ProcessFlow / Orchestrator:

        .send(prompt, *, stream=False, system="", **kwargs) -> str
        .session_name: str
        .transcript_dir: Optional[Path]

    Other downstream consumers (Branddozer, etc.) use the same shape.
    """

    def __init__(
        self,
        session_name: str = "openai",
        transcript_dir: Optional[Path] = None,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        stream_default: bool = False,
        transcript_enabled: bool = False,
        # Optional system prompt baked in at session-level.  Per-call
        # `system=` argument is concatenated on top.
        system_prompt: str = "",
        **_ignored: Any,
    ) -> None:
        # Lazy import so module import never fails when openai is
        # missing.  send() will raise clearly if it's actually called.
        self._OpenAI = None
        self._client = None
        self.session_name      = session_name
        self.transcript_dir    = transcript_dir
        self.model             = model
        self.temperature       = temperature
        self.max_tokens        = max_tokens
        self.timeout_s         = timeout_s
        self.stream_default    = stream_default
        self.transcript_enabled = transcript_enabled
        self.system_prompt     = system_prompt
        self._call_count       = 0
        self._last_usage: dict[str, Any] = {}

    # ── Setup helpers ──────────────────────────────────────────────────────

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        api_key = _resolve_openai_key()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in the secure vault and env var "
                "is unset.  Store it via the securevault UI or set it in "
                "the shell before using OpenAISession."
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed.  `pip install openai>=1.0`"
            ) from exc
        self._OpenAI = OpenAI
        self._client = OpenAI(api_key=api_key, timeout=self.timeout_s)

    def _maybe_log_transcript(self, prompt: str, system: str, reply: str) -> None:
        if not self.transcript_enabled or self.transcript_dir is None:
            return
        try:
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self.transcript_dir / f"{self.session_name}_{ts}_{uuid.uuid4().hex[:8]}.json"
            path.write_text(json.dumps({
                "session": self.session_name,
                "model":   self.model,
                "system":  system,
                "prompt":  prompt,
                "reply":   reply,
                "usage":   self._last_usage,
            }, indent=2), encoding="utf-8")
        except Exception:
            pass  # never let logging failure break the chat

    # ── Public API ─────────────────────────────────────────────────────────

    def send(
        self,
        prompt: str,
        *,
        stream: bool = False,
        system: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Send `prompt` to the configured OpenAI Chat Completions model.
        Returns the assistant's reply as a plain string.

        The `system` argument is concatenated with any constructor-level
        system_prompt so callers can layer global system context with
        per-call directives.
        """
        self._ensure_client()
        self._call_count += 1

        full_system = "\n\n".join(s for s in (self.system_prompt, system) if s.strip())
        messages: list[dict[str, str]] = []
        if full_system:
            messages.append({"role": "system", "content": full_system})
        messages.append({"role": "user", "content": prompt})

        if stream or self.stream_default:
            return self._send_streaming(messages)
        return self._send_blocking(messages)

    def _send_blocking(self, messages: list[dict[str, str]]) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc
        reply = (resp.choices[0].message.content or "").strip()
        try:
            self._last_usage = {
                "prompt_tokens":     resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens":      resp.usage.total_tokens,
            }
        except Exception:
            self._last_usage = {}
        self._maybe_log_transcript(messages[-1]["content"],
                                     messages[0]["content"] if messages[0]["role"] == "system" else "",
                                     reply)
        return reply

    def _send_streaming(self, messages: list[dict[str, str]]) -> str:
        chunks: list[str] = []
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for ev in stream:
                delta = ev.choices[0].delta.content
                if delta:
                    chunks.append(delta)
        except Exception as exc:
            raise RuntimeError(f"OpenAI streaming call failed: {exc}") from exc
        reply = "".join(chunks).strip()
        self._maybe_log_transcript(messages[-1]["content"],
                                     messages[0]["content"] if messages[0]["role"] == "system" else "",
                                     reply)
        return reply

    # ── Health probe ──────────────────────────────────────────────────────

    @classmethod
    def probe(cls) -> dict:
        """Quick health probe so callers can decide whether to fall
        back to another backend.  Returns dict with online / model /
        error keys, matching WizardSession.probe()."""
        api_key = _resolve_openai_key()
        if not api_key:
            return {"online": False, "error": "OPENAI_API_KEY not in vault or env",
                    "model":  DEFAULT_MODEL}
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key, timeout=5)
            # Cheap call: list one model.  If the SDK / key works, this
            # responds quickly.  Don't hit /completions just to probe.
            _ = next(iter(client.models.list().data), None)
            return {"online": True, "model": DEFAULT_MODEL,
                    "endpoint": "https://api.openai.com"}
        except Exception as exc:
            return {"online": False, "error": str(exc),
                    "model":  DEFAULT_MODEL}


def _resolve_openai_key() -> str:
    """Vault first, env fallback.  Lazy import so module load doesn't
    require Django to be importable in tooling contexts."""
    try:
        from tools.secret_vault import get_secret  # type: ignore
        val = get_secret("OPENAI_API_KEY")
        if val:
            return val
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")
