from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from services.conversation_memory import ConversationMemory, MemoryEntry


class LongTermMemory:
    def __init__(self, *, runtime_root: Path) -> None:
        path = runtime_root / "conversation_global.jsonl"
        self._memory = ConversationMemory(path)

    def append(self, *args, **kwargs) -> None:
        return self._memory.append(*args, **kwargs)

    def load(self, limit: int = 40) -> List[MemoryEntry]:
        return self._memory.load(limit=limit)

    def load_all(self) -> List[MemoryEntry]:
        return self._memory.load_all()

    def last_user(self) -> Optional[MemoryEntry]:
        return self._memory.last_user()

    def last_exchange(self) -> List[MemoryEntry]:
        return self._memory.last_exchange()

    def search(self, query: str, limit: int = 5) -> List[str]:
        return self._memory.search(query, limit=limit)

    def search_long_term(self, query: str, *, limit: int = 5) -> List[str]:
        return self._memory.search_long_term(query, limit=limit)

    def search_if_referenced(self, prompt: str, limit: int = 5) -> List[str]:
        return self._memory.search_if_referenced(prompt, limit=limit)


__all__ = ["LongTermMemory"]
