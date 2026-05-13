"""tools/claude_session.py — Anthropic Claude API session.

Direct Claude session via the official `anthropic` SDK.  Same shape as
WizardSession / C0d3rSession / OpenAISession so it's swappable wherever
a session is constructed:

    session = ClaudeSession(session_name="...", transcript_dir=Path(...))
    answer  = session.send("Hello", system="You are helpful")

Requires the `anthropic` package and the ANTHROPIC_API_KEY env var (or
a value supplied by the vault layer).  Model + temperature configurable
via constructor args or env vars (ANTHROPIC_MODEL, ANTHROPIC_TEMPERATURE).

Distinct from C0d3rSession (which routes Claude through AWS Bedrock).
Use this when you want to call api.anthropic.com directly without an
AWS account or Bedrock IAM setup.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional


DEFAULT_MODEL       = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
DEFAULT_TEMPERATURE = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS  = int(os.getenv("ANTHROPIC_MAX_TOKENS", "2048"))
DEFAULT_TIMEOUT_S   = float(os.getenv("ANTHROPIC_TIMEOUT_S", "60"))


class ClaudeSession:
    """Conforms to the duck-type interface used everywhere:

        .send(prompt, *, stream=False, system="", **kwargs) -> str
        .session_name: str
        .transcript_dir: Optional[Path]
    """

    def __init__(
        self,
        session_name: str = "claude",
        transcript_dir: Optional[Path] = None,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        stream_default: bool = False,
        transcript_enabled: bool = False,
        system_prompt: str = "",
        **_ignored: Any,
    ) -> None:
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

    # ── Setup ─────────────────────────────────────────────────────────────

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        api_key = _resolve_anthropic_key()
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY env var is not set (and vault lookup found "
                "nothing).  Set it before using ClaudeSession."
            )
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package not installed.  `pip install anthropic`"
            ) from exc
        self._client = Anthropic(api_key=api_key, timeout=self.timeout_s)

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
            pass

    # ── Public API ────────────────────────────────────────────────────────

    def send(
        self,
        prompt: str,
        *,
        stream: bool = False,
        system: str = "",
        **kwargs: Any,
    ) -> str:
        self._ensure_client()
        self._call_count += 1

        full_system = "\n\n".join(s for s in (self.system_prompt, system) if s.strip())

        if stream or self.stream_default:
            return self._send_streaming(prompt, full_system)
        return self._send_blocking(prompt, full_system)

    def _send_blocking(self, prompt: str, system: str) -> str:
        try:
            kwargs = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"Anthropic API call failed: {exc}") from exc
        parts = []
        for block in resp.content or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        reply = "".join(parts).strip()
        try:
            self._last_usage = {
                "input_tokens":  resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        except Exception:
            self._last_usage = {}
        self._maybe_log_transcript(prompt, system, reply)
        return reply

    def _send_streaming(self, prompt: str, system: str) -> str:
        chunks: list[str] = []
        try:
            kwargs = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs["system"] = system
            with self._client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    chunks.append(text)
        except Exception as exc:
            raise RuntimeError(f"Anthropic streaming call failed: {exc}") from exc
        reply = "".join(chunks).strip()
        self._maybe_log_transcript(prompt, system, reply)
        return reply

    # ── Health probe ──────────────────────────────────────────────────────

    @classmethod
    def probe(cls) -> dict:
        """Match the shape of the other session probes."""
        api_key = _resolve_anthropic_key()
        if not api_key:
            return {"online": False, "error": "ANTHROPIC_API_KEY not set",
                    "model": DEFAULT_MODEL}
        try:
            from anthropic import Anthropic  # type: ignore
            client = Anthropic(api_key=api_key, timeout=5)
            # Cheap probe: 1-token completion.
            client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "."}],
            )
            return {"online": True, "model": DEFAULT_MODEL,
                    "endpoint": "https://api.anthropic.com"}
        except Exception as exc:
            return {"online": False, "error": str(exc),
                    "model": DEFAULT_MODEL}


def _resolve_anthropic_key() -> str:
    """Resolve the Anthropic API key — vault first, env fallback.
    Lazy import of the vault layer so this module loads even when the
    Django app isn't wired in (e.g. CLI use)."""
    try:
        from tools.secret_vault import get_secret  # type: ignore
        val = get_secret("ANTHROPIC_API_KEY")
        if val:
            return val
    except Exception:
        pass
    return os.getenv("ANTHROPIC_API_KEY", "")
