from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import difflib
import hashlib
import json
import re
import time


@dataclass
class SideLoadedEntry:
    id: str
    scope: str
    ts: float
    query: str
    purpose: str
    paths: List[str]
    cwd: str
    project_root: str
    host: str
    os_name: str
    session_id: str
    hits: int
    last_confirmed: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "scope": self.scope,
            "ts": self.ts,
            "query": self.query,
            "purpose": self.purpose,
            "paths": self.paths,
            "cwd": self.cwd,
            "project_root": self.project_root,
            "host": self.host,
            "os": self.os_name,
            "session_id": self.session_id,
            "hits": self.hits,
            "last_confirmed": self.last_confirmed,
        }


def extract_search_terms(command: str) -> str:
    cmd = command.strip()
    m = re.search(r"rg\b.*?\s+(['\"]?)([^'\"\\s]+)\\1", cmd)
    if m:
        return m.group(2)
    m = re.search(r"grep\b.*?\s+(['\"]?)([^'\"\\s]+)\\1", cmd)
    if m:
        return m.group(2)
    m = re.search(r"find\\b.*-name\\s+(['\"]?)([^'\"\\s]+)\\1", cmd)
    if m:
        return m.group(2)
    m = re.search(r"-Filter\\s+(['\"]?)([^'\"\\s]+)\\1", cmd, re.I)
    if m:
        return m.group(2)
    return cmd[:80]


def extract_paths_from_output(output: str) -> List[str]:
    if not output:
        return []
    paths: List[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        if "[c0d3r]" in line or "targets:" in line or "finished:" in line:
            continue
        rg_match = re.match(r"^([A-Za-z]:\\\\[^:]+|/[^:]+):", line)
        if rg_match:
            paths.append(rg_match.group(1))
            continue
        candidates = re.findall(r"([A-Za-z]:\\\\[^\\s]+|/[^\\s]+)", line)
        for candidate in candidates:
            paths.append(candidate)
    deduped: List[str] = []
    seen = set()
    for path in paths:
        key = path.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
        if len(deduped) >= 12:
            break
    return deduped


def _normalize_queries(queries: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for query in queries:
        query = (query or "").strip()
        if not query:
            continue
        if query not in cleaned:
            cleaned.append(query)
    return cleaned


def _keyword_tokens(text: str) -> List[str]:
    if not text:
        return []
    tokens = [t for t in re.findall(r"[a-z0-9_\\-]{3,}", text.lower())]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "where",
        "path",
        "file",
        "folder",
        "find",
        "search",
        "location",
        "cli",
    }
    return [t for t in tokens if t not in stop]


class SideLoadedMemory:
    def __init__(self, path: Path, *, scope: str) -> None:
        self.path = path
        self.scope = scope
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: SideLoadedEntry) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def load(self, limit: int = 300) -> List[SideLoadedEntry]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8", errors="ignore").splitlines()
        entries: List[SideLoadedEntry] = []
        for line in lines[-limit:]:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            try:
                entries.append(
                    SideLoadedEntry(
                        id=str(payload.get("id") or ""),
                        scope=str(payload.get("scope") or self.scope),
                        ts=float(payload.get("ts") or 0.0),
                        query=str(payload.get("query") or ""),
                        purpose=str(payload.get("purpose") or ""),
                        paths=list(payload.get("paths") or []),
                        cwd=str(payload.get("cwd") or ""),
                        project_root=str(payload.get("project_root") or payload.get("cwd") or ""),
                        host=str(payload.get("host") or ""),
                        os_name=str(payload.get("os") or ""),
                        session_id=str(payload.get("session_id") or ""),
                        hits=int(payload.get("hits") or 0),
                        last_confirmed=float(payload.get("last_confirmed") or payload.get("ts") or 0.0),
                    )
                )
            except Exception:
                continue
        return entries

    def touch(self, record_id: str) -> None:
        if not self.path.exists():
            return
        entries = self.load(limit=400)
        updated = False
        for entry in entries:
            if entry.id == record_id:
                entry.hits += 1
                entry.last_confirmed = time.time()
                updated = True
                break
        if not updated:
            return
        with self.path.open("w", encoding="utf-8") as fh:
            for entry in entries[-300:]:
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def search(
        self,
        queries: Iterable[str],
        *,
        project_root: str | None = None,
        host: str | None = None,
        os_name: str | None = None,
        limit: int = 5,
    ) -> List[Tuple[float, SideLoadedEntry]]:
        entries = self.load(limit=400)
        if not entries:
            return []
        normalized = _normalize_queries(queries)
        if not normalized:
            return []

        scored: List[Tuple[float, SideLoadedEntry]] = []
        for entry in entries:
            text = f"{entry.purpose} {entry.query} {' '.join(entry.paths)}".strip().lower()
            best = 0.0
            for query in normalized:
                ratio = difflib.SequenceMatcher(None, query.lower(), text).ratio()
                best = max(best, ratio)
                if query.lower() in text:
                    best = max(best, 0.9)
            if best < 0.4:
                continue
            score = best
            if project_root and entry.project_root:
                try:
                    if Path(entry.project_root) == Path(project_root):
                        score += 0.15
                    elif str(entry.project_root) and str(entry.project_root) in project_root:
                        score += 0.1
                except Exception:
                    pass
            if host and entry.host and host == entry.host:
                score += 0.05
            if os_name and entry.os_name and os_name == entry.os_name:
                score += 0.03
            if entry.hits > 0:
                score += min(0.1, entry.hits * 0.02)
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:limit]


class SideLoadedMemoryIndex:
    def __init__(
        self,
        runtime_root: Path,
        *,
        session_id: str | None,
        workdir: Path,
        env: Dict[str, str] | None = None,
    ) -> None:
        self.runtime_root = runtime_root
        self.session_id = session_id or ""
        self.workdir = workdir
        self.env = env or {}
        session_path = runtime_root / f"search_memory_{self.session_id}.jsonl" if self.session_id else runtime_root / "search_memory.jsonl"
        self.short_term = SideLoadedMemory(session_path, scope="session")
        self.long_term = SideLoadedMemory(runtime_root / "search_memory.jsonl", scope="global")

    def _entry_id(self, scope: str, purpose: str, command: str, cwd: str) -> str:
        raw = f"{scope}|{purpose}|{command}|{cwd}"
        return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]

    def record_paths(
        self,
        *,
        command: str,
        purpose: str,
        cwd: str,
        paths: List[str],
        query: str,
        project_root: str | None = None,
    ) -> None:
        if not paths:
            return
        timestamp = time.time()
        env_host = str(self.env.get("host") or "")
        env_os = str(self.env.get("os") or "")
        project_root = str(project_root or self.workdir)
        for scope, store in (("session", self.short_term), ("global", self.long_term)):
            entry = SideLoadedEntry(
                id=self._entry_id(scope, purpose, command, cwd),
                scope=scope,
                ts=timestamp,
                query=query,
                purpose=purpose,
                paths=paths,
                cwd=cwd,
                project_root=project_root,
                host=env_host,
                os_name=env_os,
                session_id=self.session_id,
                hits=1,
                last_confirmed=timestamp,
            )
            store.append(entry)

    def lookup(
        self,
        *,
        command: str,
        purpose: str,
        queries: List[str],
    ) -> Optional[Dict[str, object]]:
        query_terms = _normalize_queries(queries)
        if not query_terms:
            query_terms = _normalize_queries([extract_search_terms(command)])
        host = self.env.get("host")
        os_name = self.env.get("os")
        project_root = str(self.workdir)
        candidates: List[Tuple[float, SideLoadedEntry, str]] = []
        for store, source in ((self.short_term, str(self.short_term.path)), (self.long_term, str(self.long_term.path))):
            for score, entry in store.search(query_terms, project_root=project_root, host=host, os_name=os_name, limit=6):
                candidates.append((score + (0.05 if entry.scope == "session" else 0.0), entry, source))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        top_score, entry, source = candidates[0]
        if top_score < 0.45:
            return None
        return {
            "id": entry.id,
            "paths": entry.paths,
            "purpose": entry.purpose,
            "query": entry.query,
            "scope": entry.scope,
            "source_path": source,
            "project_root": entry.project_root,
        }

    def touch(self, record_id: str, source_path: str | None = None) -> None:
        if source_path:
            target = SideLoadedMemory(Path(source_path), scope="session" if self.session_id else "global")
            target.touch(record_id)
            return
        self.short_term.touch(record_id)
        self.long_term.touch(record_id)

    def hint(self, prompt: str, *, limit: int = 2) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return ""
        tokens = _keyword_tokens(prompt)
        queries = [prompt]
        if tokens:
            queries.extend(tokens[:3])
        host = self.env.get("host")
        os_name = self.env.get("os")
        project_root = str(self.workdir)
        candidates: List[Tuple[float, SideLoadedEntry]] = []
        for store in (self.short_term, self.long_term):
            candidates.extend(store.search(queries, project_root=project_root, host=host, os_name=os_name, limit=6))
        if not candidates:
            return ""
        candidates.sort(key=lambda item: item[0], reverse=True)
        lines = ["Side-loaded memory locations:"]
        for score, entry in candidates[:limit]:
            paths = ", ".join(entry.paths[:4])
            tag = "session" if entry.scope == "session" else "global"
            lines.append(f"- {entry.purpose or entry.query} ({tag}, score={score:.2f}): {paths}")
        return "\n".join(lines)


__all__ = [
    "SideLoadedEntry",
    "SideLoadedMemory",
    "SideLoadedMemoryIndex",
    "extract_search_terms",
    "extract_paths_from_output",
]
