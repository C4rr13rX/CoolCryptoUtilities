from __future__ import annotations


class UsageTracker:
    """
    Tracks token usage and session status.

    This class is purely a data store — it does NOT drive animations or UI.
    TerminalUI polls self.status and self._status_anim_kind to decide what
    to animate on its own render loop.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.input_chars: int = 0
        self.output_chars: int = 0
        self.input_tokens: float = 0.0
        self.output_tokens: float = 0.0
        self.status: str = "idle"
        self.last_action: str = ""
        self._status_anim_kind: str = ""

    def add_input(self, text: str) -> None:
        self.input_chars += len(text or "")
        self.input_tokens = self.input_chars / 4.0

    def add_output(self, text: str) -> None:
        self.output_chars += len(text or "")
        self.output_tokens = self.output_chars / 4.0

    def set_status(self, status: str, action: str = "") -> None:
        """Update current status and last action label."""
        self.status = status
        if action:
            self.last_action = action
        # Record animation kind so TerminalUI can pick it up on its render loop.
        self._status_anim_kind = status if status in {"research", "matrix"} else ""
