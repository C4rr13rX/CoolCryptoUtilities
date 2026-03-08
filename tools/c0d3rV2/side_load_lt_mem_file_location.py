"""
Long-term side-loaded memory — cross-session file-location index using Hazy Hash.

This is the system that answers "where is that project?" across sessions.
When a user says "Go into the No Man's Land project," and LT Memory has
no exact record, the LT side-loaded memory uses the Hazy Hash graph to
approximate: "Based on this machine, this user, and past sessions, projects
tend to be in C:/Users/adam/Projects.  There's something called N0M4n5L4nD
there — that's probably it."

The AI model makes the final disambiguation.  This module narrows the
search space so the model doesn't have to scan the entire file system.

Architecture:
  - Owns a HazyHash instance scoped to "global" (persists across sessions).
  - On record: decomposes paths into the hierarchical graph with full
    context (machine, user, cwd, project root, session id).
  - On lookup: builds a context vector from what's known and traverses
    the graph to return ranked candidates.
  - At session end: ST side-loaded memory promotes its high-value entries
    here via HazyHash.promote_to_scope().
"""
from __future__ import annotations

from pathlib import Path

from hazy_hash import HazyHash, HazyHashContext


class LTSideLoadedMemory:
    """
    Cross-session file-location memory backed by the Hazy Hash graph.

    The Kuzu graph accumulates location patterns across all sessions:
    which machines, which users, which project directories, which files
    were accessed and from what context.  Over time, the graph builds a
    rich map of "where things tend to be" for this user on this machine.
    """

    def __init__(self, runtime_root: Path) -> None:
        db_path = runtime_root / "hazy_hash_lt_db"
        self._hash = HazyHash(db_path, scope="global")
        self._runtime_root = runtime_root

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        query: str,
        path: str,
        *,
        cwd: str = "",
        project_root: str = "",
        session_id: str = "",
        user: str = "",
    ) -> None:
        """Record that ``path`` was found for ``query`` in the given context."""
        ctx = HazyHashContext.from_env(
            cwd=cwd,
            project_root=project_root,
            session_id=session_id,
            query=query,
        )
        if user:
            ctx.username = user
        self._hash.record(path, ctx)

    def record_paths(
        self,
        query: str,
        paths: list[str],
        *,
        cwd: str = "",
        project_root: str = "",
        session_id: str = "",
        user: str = "",
    ) -> None:
        """Record multiple paths for the same query."""
        for path in paths:
            self.record(
                query, path,
                cwd=cwd,
                project_root=project_root,
                session_id=session_id,
                user=user,
            )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        query: str,
        *,
        cwd: str = "",
        project_root: str = "",
        user: str = "",
    ) -> list[str]:
        """
        Return ranked candidate paths for ``query``.

        Traverses the Hazy Hash graph using context (machine, user,
        cwd, query tokens) to narrow the search space.  Returns paths
        ordered by relevance score — the AI model picks the final match.
        """
        ctx = HazyHashContext.from_env(
            cwd=cwd,
            project_root=project_root,
            query=query,
        )
        if user:
            ctx.username = user
        results = self._hash.lookup(ctx, limit=20)
        return [r["path"] for r in results if r.get("path")]

    def lookup_detailed(
        self,
        query: str,
        *,
        cwd: str = "",
        project_root: str = "",
        user: str = "",
    ) -> list[dict]:
        """
        Like lookup() but returns full candidate details (path, score,
        level, label, reason) so the AI model can reason about why each
        candidate was suggested.
        """
        ctx = HazyHashContext.from_env(
            cwd=cwd,
            project_root=project_root,
            query=query,
        )
        if user:
            ctx.username = user
        return self._hash.lookup(ctx, limit=20)

    # ------------------------------------------------------------------
    # Absorption (ST → LT promotion)
    # ------------------------------------------------------------------

    def absorb_from_session(self, session_scope: str) -> int:
        """
        Promote high-value entries from a session-scoped HazyHash into
        the global LT graph.  Called at session end.

        Returns the number of nodes promoted.
        """
        return self._hash.promote_to_scope("global")
