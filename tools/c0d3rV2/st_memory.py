"""
Short-Term Memory — session-scoped rolling summary, key points, and transcript.

This is the main ST memory that powers context injection.  Every turn:
  1. The user message and system response are recorded.
  2. The rolling summary is updated via a model call (previous summary +
     new turn → updated summary + 10 most important points).
  3. When context is needed, the builder pulls:
     a. The rolling summary (compressed history).
     b. The 10 key points (highlights to keep track of).
     c. As many recent user/system messages as fit in the remaining
        character budget, newest first, with smart truncation that
        preserves the most informative parts.

Design:
  - Modular OOP class that ProcessFlow delegates to.
  - Owns its own persistence (JSON file for summary, in-memory ring
    buffer for transcript).
  - Smart transcript budgeting: user messages get more space (they
    contain instructions), system messages are trimmed to essentials.
  - Priority weighting: turns with errors, tool calls, or user
    instructions get more space than routine successful turns.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TranscriptEntry:
    """One turn of conversation."""

    timestamp: str
    user: str
    system: str
    has_error: bool = False
    has_tool_calls: bool = False
    has_user_instruction: bool = False
    importance: float = 0.5  # 0-1, computed on insert

    def compact(self, max_user: int = 0, max_system: int = 0) -> str:
        """Format for injection with optional truncation."""
        u = self.user
        s = self.system
        if max_user and len(u) > max_user:
            u = u[:max_user] + "..."
        if max_system and len(s) > max_system:
            s = s[:max_system] + "..."
        return f"User: {u}\nAssistant: {s}"


class STMemory:
    """
    Session-scoped short-term memory.

    Manages the rolling summary, key points, and transcript ring buffer.
    ProcessFlow calls `record_turn()` after each turn, and ContextBuilder
    calls `build_memory_section()` and `build_transcript_section()` when
    assembling context.
    """

    DEFAULT_TRANSCRIPT_BUDGET: int = 8000
    DEFAULT_SUMMARY_MAX_WORDS: int = 300
    DEFAULT_MAX_KEY_POINTS: int = 10
    # Keep this many turns in the ring buffer.
    DEFAULT_RING_SIZE: int = 50

    # Instruction-like keywords that indicate user intent.
    _INSTRUCTION_KEYWORDS: set[str] = {
        "please", "make", "create", "update", "fix", "change", "add",
        "remove", "delete", "rename", "move", "install", "run", "build",
        "test", "deploy", "implement", "refactor", "use", "don't", "never",
        "always", "should", "must", "ensure", "configure", "set up",
        "write", "modify", "edit", "replace", "search", "find",
    }

    def __init__(
        self,
        session: Any,
        *,
        session_id: str = "",
        runtime_root: Path | None = None,
        transcript_budget: int = DEFAULT_TRANSCRIPT_BUDGET,
        ring_size: int = DEFAULT_RING_SIZE,
    ) -> None:
        self.session = session
        self.session_id = session_id
        self._runtime_root = runtime_root or Path("runtime/c0d3r")
        self._transcript_budget = transcript_budget
        self._ring: deque[TranscriptEntry] = deque(maxlen=ring_size)
        self._summary: str = ""
        self._key_points: list[str] = []
        self._turn_count: int = 0

        # Load persisted summary if exists.
        self._load()

    # ------------------------------------------------------------------
    # Public: record a turn
    # ------------------------------------------------------------------

    def record_turn(
        self,
        user_input: str,
        system_output: str,
        *,
        has_error: bool = False,
        has_tool_calls: bool = False,
    ) -> None:
        """
        Record a conversation turn and update the rolling summary.

        Called by ProcessFlow after each turn completes.
        """
        self._turn_count += 1

        # Compute importance.
        has_instruction = self._detect_instruction(user_input)
        importance = self._compute_importance(
            user_input, system_output,
            has_error=has_error,
            has_tool_calls=has_tool_calls,
            has_instruction=has_instruction,
        )

        entry = TranscriptEntry(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            user=user_input,
            system=system_output,
            has_error=has_error,
            has_tool_calls=has_tool_calls,
            has_user_instruction=has_instruction,
            importance=importance,
        )
        self._ring.append(entry)

        # Update rolling summary via model.
        self._update_summary(user_input, system_output)

    # ------------------------------------------------------------------
    # Public: build context sections for ContextBuilder
    # ------------------------------------------------------------------

    def build_memory_section(self) -> str:
        """
        Build the memory section: rolling summary + key points.
        Called by ContextBuilder._memory_section().
        """
        parts: list[str] = []

        if self._summary:
            parts.append(f"[Rolling Summary]\n{self._summary}")

        if self._key_points:
            pts = "\n".join(f"- {p}" for p in self._key_points[:10])
            parts.append(f"[Key Points]\n{pts}")

        return "\n\n".join(parts)

    def build_transcript_section(self, budget: int = 0) -> str:
        """
        Build the transcript section with smart budgeting.

        Fills the character budget with recent turns, giving more space
        to high-importance turns (errors, instructions, tool calls) and
        less to routine turns.  User messages get proportionally more
        space than system messages because they contain instructions.
        """
        budget = budget or self._transcript_budget
        if not self._ring:
            return ""

        # Sort ring entries: most recent first, but weight by importance.
        entries = list(self._ring)

        # Allocate budget across entries.
        allocated = self._allocate_budget(entries, budget)

        lines: list[str] = ["[Recent Transcript]"]
        char_count = 0

        # Walk from most recent to oldest.
        for entry, (user_chars, sys_chars) in zip(
            reversed(entries), reversed(allocated),
        ):
            block = entry.compact(
                max_user=user_chars,
                max_system=sys_chars,
            )
            if char_count + len(block) > budget:
                break
            lines.append(block)
            char_count += len(block)

        return "\n".join(lines) if len(lines) > 1 else ""

    # ------------------------------------------------------------------
    # Public: access summary data directly
    # ------------------------------------------------------------------

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def key_points(self) -> list[str]:
        return list(self._key_points)

    @property
    def summary_bundle(self) -> dict:
        """Backward-compatible dict format."""
        return {
            "summary": self._summary,
            "key_points": self._key_points,
        }

    # ------------------------------------------------------------------
    # Rolling summary update
    # ------------------------------------------------------------------

    def _update_summary(self, user_input: str, output: str) -> None:
        """
        Update the rolling summary with the new turn.

        The model receives the previous summary + new turn and produces
        an updated summary + key points.  The prompt tells the model
        specifically WHAT to track.
        """
        prompt = (
            f"Previous rolling summary:\n{self._summary[:2000]}\n\n"
            f"Previous key points:\n"
            + "\n".join(f"- {p}" for p in self._key_points[:10])
            + f"\n\nNew user message:\n{user_input[:1500]}\n\n"
            f"New system response:\n{output[:2500]}\n\n"
            "Update the rolling summary and key points.  The summary should:\n"
            "- Capture WHAT the user is working on and their current goal.\n"
            "- Track decisions made, approaches chosen, and approaches rejected.\n"
            "- Note files created, modified, or referenced.\n"
            "- Record errors encountered and how they were resolved.\n"
            "- Preserve any standing user instructions (e.g. 'always use X').\n"
            f"- Be at most {self.DEFAULT_SUMMARY_MAX_WORDS} words.\n\n"
            "The key points should be the 10 most important things to "
            "keep track of for the NEXT turn — prioritize:\n"
            "1. Active user instructions and preferences.\n"
            "2. Current task and sub-task status.\n"
            "3. Errors or blockers not yet resolved.\n"
            "4. Important file paths and tool outputs.\n"
            "5. Decisions that constrain future actions.\n\n"
            'Return JSON only: {"summary": str, "key_points": [str]}'
        )
        try:
            raw = self.session.send(prompt=prompt, stream=False)
            m = re.search(r"\{[\s\S]*\}", raw or "")
            if m:
                payload = json.loads(m.group(0))
                self._summary = str(payload.get("summary", ""))[:2000]
                self._key_points = [
                    str(p)
                    for p in (payload.get("key_points") or [])[:10]
                ]
                self._save()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Importance scoring
    # ------------------------------------------------------------------

    def _compute_importance(
        self,
        user_input: str,
        system_output: str,
        *,
        has_error: bool,
        has_tool_calls: bool,
        has_instruction: bool,
    ) -> float:
        """
        Score 0-1 how important this turn is for context retention.

        Higher importance = more character budget when building transcript.
        """
        score = 0.3  # Base importance.

        if has_error:
            score += 0.3  # Errors are critical context.
        if has_tool_calls:
            score += 0.1  # Tool calls produce outputs worth tracking.
        if has_instruction:
            score += 0.2  # User instructions must be preserved.

        # Long user messages tend to contain more detail.
        if len(user_input) > 500:
            score += 0.1

        return min(score, 1.0)

    def _detect_instruction(self, text: str) -> bool:
        """Check if user text contains instruction-like language."""
        lower = text.lower()
        return any(kw in lower for kw in self._INSTRUCTION_KEYWORDS)

    # ------------------------------------------------------------------
    # Budget allocation
    # ------------------------------------------------------------------

    def _allocate_budget(
        self,
        entries: list[TranscriptEntry],
        total_budget: int,
    ) -> list[tuple[int, int]]:
        """
        Allocate character budgets to each transcript entry.

        Higher-importance entries get more space.  User messages get 60%
        of each entry's budget, system messages get 40%.
        """
        if not entries:
            return []

        # Weight each entry by importance, with recency bonus.
        weights: list[float] = []
        n = len(entries)
        for i, entry in enumerate(entries):
            recency_bonus = (i + 1) / n  # 0→1, most recent = 1.
            weight = entry.importance * 0.6 + recency_bonus * 0.4
            weights.append(weight)

        total_weight = sum(weights) or 1.0

        allocations: list[tuple[int, int]] = []
        for weight in weights:
            entry_budget = int((weight / total_weight) * total_budget)
            # User gets 60%, system gets 40%.
            user_budget = int(entry_budget * 0.6)
            sys_budget = entry_budget - user_budget
            # Minimum 100 chars each to be useful.
            user_budget = max(user_budget, 100)
            sys_budget = max(sys_budget, 100)
            allocations.append((user_budget, sys_budget))

        return allocations

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _summary_path(self) -> Path:
        if self.session_id:
            return self._runtime_root / f"st_memory_{self.session_id}.json"
        return self._runtime_root / "st_memory.json"

    def _save(self) -> None:
        path = self._summary_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "summary": self._summary,
            "key_points": self._key_points,
            "turn_count": self._turn_count,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8",
            )
        except Exception:
            pass

    def _load(self) -> None:
        path = self._summary_path()
        if not path.exists():
            return
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8", errors="ignore")
            )
            stored_id = str(payload.get("session_id", "")).strip()
            if self.session_id and stored_id and stored_id != self.session_id:
                return  # Different session — start fresh.
            self._summary = str(payload.get("summary", ""))[:2000]
            self._key_points = [
                str(p).strip()
                for p in (payload.get("key_points") or [])[:10]
                if str(p).strip()
            ]
            self._turn_count = int(payload.get("turn_count", 0))
        except Exception:
            pass
