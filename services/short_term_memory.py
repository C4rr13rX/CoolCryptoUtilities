from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from services.conversation_memory import ConversationMemory, MemoryEntry


class ShortTermMemory:
    def __init__(self, *, runtime_root: Path, session_id: str | None) -> None:
        self.session_id = session_id or ""
        path = runtime_root / f"conversation_{self.session_id}.jsonl" if self.session_id else runtime_root / "conversation.jsonl"
        self._memory = ConversationMemory(path)

    def append(self, *args, **kwargs) -> None:
        return self._memory.append(*args, **kwargs)

    def load(self, limit: int = 40, *, session_id: str | None = None) -> List[MemoryEntry]:
        return self._memory.load(limit=limit, session_id=self.session_id)

    def load_all(self) -> List[MemoryEntry]:
        return self._memory.load_all()

    def build_context(
        self,
        summary,
        *,
        max_chars: int = 12000,
        context_limit: int = 1400,
        session_id: str | None = None,
    ) -> str:
        return self._memory.build_context(
            summary,
            max_chars=max_chars,
            context_limit=context_limit,
            session_id=self.session_id,
        )

    def last_user(self, *, session_id: str | None = None) -> Optional[MemoryEntry]:
        return self._memory.last_user(session_id=self.session_id)

    def last_exchange(self, *, session_id: str | None = None) -> List[MemoryEntry]:
        return self._memory.last_exchange(session_id=self.session_id)

    def search(self, query: str, limit: int = 5) -> List[str]:
        return self._memory.search(query, limit=limit)

    def search_long_term(self, query: str, *, limit: int = 5) -> List[str]:
        return self._memory.search_long_term(query, limit=limit)

    def search_if_referenced(self, prompt: str, limit: int = 5) -> List[str]:
        return self._memory.search_if_referenced(prompt, limit=limit)

    def update_summary(self, summary, user_text: str, assistant_text: str, session) -> dict:
        return self._memory.update_summary(summary, user_text, assistant_text, session)


__all__ = ["ShortTermMemory"]
