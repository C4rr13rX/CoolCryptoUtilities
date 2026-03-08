"""
Short-term side-loaded memory — session-scoped file-location index using Hazy Hash.

Within a single session, the AI model discovers file locations as it works.
ST side-loaded memory captures these discoveries so subsequent steps can
re-reference them instantly without the model scanning the file system again.

Compared to LT side-loaded memory:
  - Scoped to the current session only.
  - Context is richer and more specific (session history, recent cwd changes,
    tools used this session).
  - Faster access because the graph is smaller.
  - At session end, high-value entries are promoted to LT via
    LTSideLoadedMemory.absorb_from_session().

Architecture:
  - Owns a HazyHash instance scoped to the session_id.
  - Builds context from the current session state (cwd, project root,
    recent queries, accumulated tool outputs).
  - Lookup returns candidates ranked by session-specific relevance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hazy_hash import HazyHash, HazyHashContext


class STSideLoadedMemory:
    """
    Session-scoped file-location memory backed by the Hazy Hash graph.

    Every file path discovered during the session is recorded with full
    context.  Subsequent lookups in the same session benefit from this
    accumulated knowledge — the model doesn't have to rediscover files
    it already found.
    """

    def __init__(self, session_id: str, runtime_root: Path) -> None:
        self.session_id = session_id
        db_path = runtime_root / f"hazy_hash_st_{session_id}_db"
        self._hash = HazyHash(db_path, scope=session_id)
        # In-memory quick-access cache for this session's exact lookups.
        self._cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_paths(
        self,
        query: str,
        paths: list[str],
        *,
        cwd: str = "",
        project_root: str = "",
    ) -> None:
        """Record discovered file paths for a query in session context."""
        ctx = HazyHashContext.from_env(
            cwd=cwd,
            project_root=project_root,
            session_id=self.session_id,
            query=query,
        )
        for path in paths:
            self._hash.record(path, ctx)
        # Also cache for instant re-lookup within the same session.
        cached = self._cache.setdefault(query.lower().strip(), [])
        for p in paths:
            if p not in cached:
                cached.append(p)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, query: str, *, cwd: str = "") -> list[str]:
        """
        Return ranked candidate paths for ``query`` within this session.

        First checks the in-memory cache (instant, exact matches from
        earlier in this session).  Then falls back to the Hazy Hash graph
        for contextual approximation.
        """
        # Fast path: exact cache hit from this session
        key = query.lower().strip()
        if key in self._cache and self._cache[key]:
            return self._cache[key]

        # Contextual graph lookup
        ctx = HazyHashContext.from_env(
            cwd=cwd,
            session_id=self.session_id,
            query=query,
        )
        results = self._hash.lookup(ctx, limit=15)
        return [r["path"] for r in results if r.get("path")]

    def lookup_detailed(
        self, query: str, *, cwd: str = "",
    ) -> list[dict]:
        """
        Like lookup() but returns full candidate details (path, score,
        level, label, reason) for the AI model to reason about.
        """
        ctx = HazyHashContext.from_env(
            cwd=cwd,
            session_id=self.session_id,
            query=query,
        )
        return self._hash.lookup(ctx, limit=15)

    # ------------------------------------------------------------------
    # Promotion to LT
    # ------------------------------------------------------------------

    @property
    def hazy_hash(self) -> HazyHash:
        """Expose the underlying HazyHash for promotion at session end."""
        return self._hash
