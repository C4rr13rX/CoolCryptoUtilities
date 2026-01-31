from __future__ import annotations

from typing import Optional

from tools.ai_session import get_session_class, default_settings, session_provider_from_context


def send_codex_update(message: str, *, session_name: str = "u53rxr080t") -> str:
    """
    Forward a message to the Codex CLI for UX automation loops. This borrows the
    same harness used by BrandDozer to stream transcripts into runtime/branddozer/transcripts.
    """
    provider = session_provider_from_context({})
    SessionClass = get_session_class(provider)
    session = SessionClass(
        session_name=session_name,
        transcript_dir="runtime/u53rxr080t/transcripts",
        read_timeout_s=None,
        **default_settings(provider),
    )
    return session.send(message, stream=True)
