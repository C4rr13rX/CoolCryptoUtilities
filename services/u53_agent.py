from __future__ import annotations

from typing import Optional

from tools.codex_session import CodexSession, codex_default_settings


def send_codex_update(message: str, *, session_name: str = "u53rxr080t") -> str:
    """
    Forward a message to the Codex CLI for UX automation loops. This borrows the
    same harness used by BrandDozer to stream transcripts into runtime/branddozer/transcripts.
    """
    session = CodexSession(
        session_name=session_name,
        transcript_dir="runtime/u53rxr080t/transcripts",
        read_timeout_s=None,
        **codex_default_settings(),
    )
    return session.send(message, stream=True)
