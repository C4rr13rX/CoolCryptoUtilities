from __future__ import annotations

from typing import Optional

from tools.codex_session import CodexSession


def send_codex_update(message: str, *, session_name: str = "u53rxr080t") -> str:
    """
    Forward a message to the Codex CLI for UX automation loops. This borrows the
    same harness used by BrandDozer to stream transcripts into runtime/branddozer/transcripts.
    """
    session = CodexSession(
        session_name=session_name,
        transcript_dir="runtime/u53rxr080t/transcripts",
        sandbox_mode="danger-full-access",
        approval_policy="never",
        model="gpt-5.1-codex-max",
        reasoning_effort="xhigh",
        read_timeout_s=None,
    )
    return session.send(message, stream=True)
