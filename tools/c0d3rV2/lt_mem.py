from __future__ import annotations

import json
import time
from pathlib import Path


class LongTermMemory:
    """
    Main LT Memory Module: stores all session user requests and model responses,
    full transcripts, and code, organised by date and searchable by context.

    Every turn is appended to a JSONL file.  Retrieval is keyword-based
    (simple for now; future: semantic search via embeddings or Kuzu).
    """

    MAX_ENTRIES: int = 10_000

    def __init__(self, runtime_root: Path) -> None:
        self._path = runtime_root / "lt_memory.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def append(
        self,
        user_input: str,
        model_output: str,
        *,
        context: str = "",
        workdir: str = "",
        model_id: str = "",
        session_id: str = "",
    ) -> None:
        """Append one conversation turn to the long-term store."""
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "workdir": workdir,
            "model_id": model_id,
            "user": user_input[:8000],
            "model": model_output[:8000],
            "context_snippet": context[:1000],
        }
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def search(self, query: str, *, limit: int = 20) -> list[dict]:
        """
        Keyword search over stored entries.
        Returns up to `limit` matching records, newest first.
        """
        if not query or not self._path.exists():
            return []

        tokens = query.lower().split()
        matches: list[dict] = []

        for line in self._tail_lines():
            try:
                record = json.loads(line)
            except Exception:
                continue
            blob = (record.get("user", "") + " " + record.get("model", "")).lower()
            if all(t in blob for t in tokens):
                matches.append(record)
            if len(matches) >= limit:
                break

        return matches

    def recent(self, *, limit: int = 10, session_id: str = "") -> list[dict]:
        """Return the most recent `limit` entries, optionally filtered by session."""
        if not self._path.exists():
            return []
        result: list[dict] = []
        # Use tail-reading to avoid loading the entire file.
        for line in self._tail_lines():
            try:
                record = json.loads(line)
            except Exception:
                continue
            if session_id and record.get("session_id") != session_id:
                continue
            result.append(record)
            if len(result) >= limit:
                break
        return result

    # ------------------------------------------------------------------
    # Efficient tail reading
    # ------------------------------------------------------------------

    def _tail_lines(self, max_lines: int = 200) -> list[str]:
        """
        Read the last `max_lines` lines from the JSONL file without loading
        the entire file into memory.  Returns lines in reverse order
        (newest first).
        """
        if not self._path.exists():
            return []
        try:
            size = self._path.stat().st_size
            if size == 0:
                return []
            # Read at most 2 MB from the tail — enough for ~200 entries.
            read_size = min(size, 2 * 1024 * 1024)
            with self._path.open("rb") as fh:
                fh.seek(max(0, size - read_size))
                chunk = fh.read().decode("utf-8", errors="ignore")
            lines = chunk.splitlines()
            # The first line may be partial if we seeked mid-line; drop it
            # unless we read from the start.
            if size > read_size and lines:
                lines = lines[1:]
            # Return newest first, capped at max_lines.
            lines.reverse()
            return lines[:max_lines]
        except Exception:
            return []
