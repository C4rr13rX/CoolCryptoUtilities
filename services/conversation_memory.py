from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from typing import List


@dataclass
class MemoryEntry:
    role: str
    content: str
    ts: str = ""
    context: str = ""
    workdir: str = ""
    model_id: str = ""
    session_id: str = ""


class ConversationMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        user_text: str,
        assistant_text: str,
        *,
        context: str | None = None,
        workdir: str | None = None,
        model_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "role": "user",
                        "content": user_text,
                        "ts": ts,
                        "context": context or "",
                        "workdir": workdir or "",
                        "model_id": model_id or "",
                        "session_id": session_id or "",
                    }
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {
                        "role": "assistant",
                        "content": assistant_text,
                        "ts": ts,
                        "workdir": workdir or "",
                        "model_id": model_id or "",
                        "session_id": session_id or "",
                    }
                )
                + "\n"
            )

    def load(self, limit: int = 40, *, session_id: str | None = None) -> List[MemoryEntry]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8", errors="ignore").splitlines()
        entries: List[MemoryEntry] = []
        for line in lines[-limit:]:
            try:
                payload = json.loads(line)
                entry = MemoryEntry(
                    role=payload.get("role", ""),
                    content=payload.get("content", ""),
                    ts=payload.get("ts", ""),
                    context=payload.get("context", ""),
                    workdir=payload.get("workdir", ""),
                    model_id=payload.get("model_id", ""),
                    session_id=payload.get("session_id", ""),
                )
                if session_id and entry.session_id != session_id:
                    continue
                entries.append(
                    MemoryEntry(
                        role=entry.role,
                        content=entry.content,
                        ts=entry.ts,
                        context=entry.context,
                        workdir=entry.workdir,
                        model_id=entry.model_id,
                        session_id=entry.session_id,
                    )
                )
            except Exception:
                continue
        return entries

    def load_all(self) -> List[MemoryEntry]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8", errors="ignore").splitlines()
        entries: List[MemoryEntry] = []
        for line in lines:
            try:
                payload = json.loads(line)
                entries.append(
                    MemoryEntry(
                        role=payload.get("role", ""),
                        content=payload.get("content", ""),
                        ts=payload.get("ts", ""),
                        context=payload.get("context", ""),
                        workdir=payload.get("workdir", ""),
                        model_id=payload.get("model_id", ""),
                        session_id=payload.get("session_id", ""),
                    )
                )
            except Exception:
                continue
        return entries

    def _unpack_summary(self, summary) -> tuple[str, List[str]]:
        if isinstance(summary, dict):
            summary_text = str(summary.get("summary") or "")
            points = summary.get("key_points") or []
            if isinstance(points, list):
                key_points = [str(p) for p in points if str(p).strip()]
            else:
                key_points = []
            return summary_text, key_points
        return str(summary or ""), []

    def build_context(
        self,
        summary,
        *,
        max_chars: int = 12000,
        context_limit: int = 1400,
        session_id: str | None = None,
    ) -> str:
        summary_text, key_points = self._unpack_summary(summary)
        parts: List[str] = []
        if summary_text.strip():
            parts.append("[rolling_summary]\n" + summary_text.strip())
        if key_points:
            points = key_points[:10]
            parts.append("[key_points]\n" + "\n".join(f"- {p}" for p in points))

        budget = max(2000, int(max_chars))
        used = len("\n\n".join(parts))
        if used >= budget:
            return "\n\n".join(parts)[:budget]

        history = self.load(limit=200, session_id=session_id)
        if not history:
            return "\n\n".join(parts)

        blocks: List[str] = []
        for entry in history:
            header = f"[{entry.ts or 'unknown'}] {entry.role}:"
            lines = [header]
            if entry.role == "user" and entry.context:
                ctx = entry.context.strip()
                if context_limit and len(ctx) > context_limit:
                    ctx = ctx[:context_limit].rstrip() + "..."
                lines.append("[context]")
                lines.append(ctx)
            content = (entry.content or "").strip()
            if content:
                lines.append(content)
            blocks.append("\n".join(lines))

        transcript_lines: List[str] = []
        transcript_lines.append("[recent_transcript]")
        transcript_body: List[str] = []
        size = len("\n".join(transcript_lines))
        for block in reversed(blocks):
            block_size = len(block) + 2
            if used + size + block_size > budget:
                break
            transcript_body.append(block)
            size += block_size
        if transcript_body:
            transcript_body.reverse()
            transcript_lines.append("\n\n".join(transcript_body))
            parts.append("\n".join(transcript_lines))
        return "\n\n".join(parts)

    def search(self, query: str, limit: int = 5) -> List[str]:
        return self.search_long_term(query, limit=limit)

    def search_long_term(self, query: str, *, limit: int = 5, window: int = 2) -> List[str]:
        if not self.path.exists():
            return []
        entries = self.load_all()
        if not entries:
            return []
        query_l = (query or "").lower().strip()
        if not query_l:
            return []

        scored: List[tuple[float, int]] = []
        for idx, entry in enumerate(entries):
            blob = f"{entry.content}\n{entry.context}".lower()
            if query_l in blob:
                score = 1.0
            else:
                score = SequenceMatcher(None, query_l, blob).ratio()
            if score < 0.2:
                continue
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: List[str] = []
        used_indices: set[int] = set()
        for score, idx in scored[:limit * 2]:
            if idx in used_indices:
                continue
            start = max(0, idx - window)
            end = min(len(entries), idx + window + 1)
            snippet_lines: List[str] = [f"[match score={score:.2f}]"]
            for j in range(start, end):
                used_indices.add(j)
                e = entries[j]
                header = f"[{e.ts or 'unknown'}] {e.role}:"
                snippet_lines.append(header)
                if e.role == "user" and e.context:
                    snippet_lines.append("[context]")
                    snippet_lines.append((e.context or "").strip())
                snippet_lines.append((e.content or "").strip())
            results.append("\n".join([ln for ln in snippet_lines if ln]))
            if len(results) >= limit:
                break
        return results

    def search_if_referenced(self, prompt: str, limit: int = 5) -> List[str]:
        lower = (prompt or "").lower()
        triggers = ("remember", "recall", "do you remember", "when we were", "that time", "earlier", "previous", "last time")
        if not any(t in lower for t in triggers):
            return []
        # extract a rough query from the prompt
        query = prompt.strip()
        if len(query) > 200:
            query = query[-200:]
        return self.search_long_term(query, limit=limit)

    def update_summary(self, summary, user_text: str, assistant_text: str, session) -> dict:
        summary_text, key_points = self._unpack_summary(summary)
        system = (
            "Return ONLY JSON with keys: summary (string, <=200 words), "
            "key_points (list of 10 short strings). "
            "Focus on the most important and most recent conversation facts."
        )
        prompt = (
            f"Current summary (<=200 words):\n{summary_text}\n\n"
            f"Current key points:\n{key_points}\n\n"
            f"New exchange:\nUser: {user_text}\nAssistant: {assistant_text}\n"
        )
        try:
            response = session.send(prompt, stream=False, system=system)
        except Exception:
            return {"summary": summary_text, "key_points": key_points}

        payload = {}
        try:
            payload = json.loads(response)
        except Exception:
            try:
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end > start:
                    payload = json.loads(response[start : end + 1])
            except Exception:
                payload = {}

        new_summary = str(payload.get("summary") or summary_text).strip()
        words = new_summary.split()
        if len(words) > 200:
            new_summary = " ".join(words[:200])
        new_points = payload.get("key_points") or key_points
        if not isinstance(new_points, list):
            new_points = key_points
        new_points = [str(p).strip() for p in new_points if str(p).strip()]
        if len(new_points) > 10:
            new_points = new_points[:10]
        return {"summary": new_summary, "key_points": new_points}
