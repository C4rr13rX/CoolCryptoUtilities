from __future__ import annotations

from typing import Optional

from tools.ai_session import get_session_class, default_settings, session_provider_from_context


def send_agent_update(message: str, *, session_name: str = "u53rxr080t") -> str:
    """
    Forward a message to the AI agent (wizard/bedrock) for UX automation loops.
    Streams transcripts into runtime/u53rxr080t/transcripts.
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
