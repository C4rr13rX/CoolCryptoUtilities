from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class MemoryEntry:
    role: str
    content: str


class ConversationMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, user_text: str, assistant_text: str) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"role": "user", "content": user_text}) + "\n")
            fh.write(json.dumps({"role": "assistant", "content": assistant_text}) + "\n")

    def load(self, limit: int = 40) -> List[MemoryEntry]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8", errors="ignore").splitlines()
        entries: List[MemoryEntry] = []
        for line in lines[-limit:]:
            try:
                payload = json.loads(line)
                entries.append(MemoryEntry(role=payload.get("role", ""), content=payload.get("content", "")))
            except Exception:
                continue
        return entries

    def build_context(self, summary: str) -> str:
        parts = []
        if summary.strip():
            parts.append("[rolling_summary]\n" + summary.strip())
        history = self.load(limit=20)
        if history:
            parts.append("[recent_history]")
            for entry in history:
                parts.append(f"{entry.role}: {entry.content[:400]}")
        return "\n".join(parts)

    def search(self, query: str, limit: int = 5) -> List[str]:
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        scored: List[tuple[float, str]] = []
        for i, line in enumerate(lines):
            if query.lower() in line.lower():
                score = 1.0
            else:
                score = SequenceMatcher(None, query.lower(), line.lower()).ratio()
            if score < 0.2:
                continue
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            snippet = "\n".join(lines[start:end])
            scored.append((score, snippet))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:limit]]

    def search_if_referenced(self, prompt: str, limit: int = 5) -> List[str]:
        lower = (prompt or "").lower()
        triggers = ("remember", "recall", "do you remember", "when we were", "that time")
        if not any(t in lower for t in triggers):
            return []
        # extract a rough query from the prompt
        query = prompt.strip()
        if len(query) > 200:
            query = query[-200:]
        return self.search(query, limit=limit)

    def update_summary(self, summary: str, user_text: str, assistant_text: str, session) -> str:
        prompt = (
            "Update the rolling summary in <= 10 bullets. "
            "Keep the most recent and most important points. "
            "Summary must be concise and project-management focused.\n\n"
            f"Current summary:\n{summary}\n\n"
            f"New exchange:\nUser: {user_text}\nAssistant: {assistant_text}\n"
        )
        try:
            response = session.send(prompt, stream=False)
            return response.strip()
        except Exception:
            return summary
