from __future__ import annotations

from pathlib import Path
from typing import Any


class SessionManager:
    """
    Thin factory/wrapper around C0d3rSession that gives the rest of the V2
    system a consistent interface without importing C0d3rSession directly.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        region: str | None = None,
        profile: str | None = None,
        reasoning_effort: str | None = None,
        transcript_dir: Path | None = None,
        workdir: str = "",
    ) -> None:
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

        self._session = C0d3rSession(
            session_name="c0d3rv2",
            transcript_dir=str(transcript_dir) if transcript_dir else None,
            workdir=workdir,
            **settings,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session(self) -> Any:
        return self._session

    @property
    def session_id(self) -> str:
        return str(getattr(self._session, "session_id", ""))

    @property
    def model_id(self) -> str:
        return self._session.get_model_id()

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def send(self, prompt: str, *, stream: bool = False, system: str = "") -> str:
        return self._session.send(prompt=prompt, stream=stream, system=system) or ""
