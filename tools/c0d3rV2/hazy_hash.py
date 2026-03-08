"""
Hazy Hash — context-anchored file-location memory backed by Kuzu.

Inspired by how human brains approximate locations rather than computing
exact hashes.  The entire file system is conceptually a tree, but we
don't need the whole thing — we need an *approximation* of where a file
is based on context.  The approximation gets sharper as more context is
provided (machine → user → common dirs → project areas → specific paths).

Architecture:
  The Kuzu graph stores a hierarchy of **LocationNode** entries at varying
  abstraction levels:

    Machine ──→ User ──→ Zone (e.g. "Projects", "Desktop") ──→ Project ──→ Path

  Each node carries contextual tags (hostname, username, os, cwd at time
  of recording) and a recency timestamp.  Edges connect parents to children
  and record co-occurrence strength (how often a child was accessed in
  the context of a parent).

  On **lookup**, the system:
    1. Builds a context vector from what is known (machine, user, cwd,
       query tokens).
    2. Traverses the graph top-down, scoring each node by how well it
       matches the context vector.
    3. Returns ranked candidate paths — the AI model makes the final
       disambiguation (e.g. recognising "N0M4n5L4nD" as "No Man's Land").

  Kuzu accelerates the traversal; if Kuzu is unavailable, a JSON fallback
  is used so the system degrades gracefully.

Public classes:
  HazyHash           — core engine: record paths, lookup with context.
  HazyHashContext     — context vector passed into lookups.
"""
from __future__ import annotations

import json
import os
import platform
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# ------------------------------------------------------------------
# Context vector
# ------------------------------------------------------------------

@dataclass
class HazyHashContext:
    """Everything we know right now that can narrow a location search."""

    hostname: str = ""
    username: str = ""
    os_name: str = ""          # e.g. "nt", "posix"
    platform_tag: str = ""     # e.g. "win32", "linux"
    cwd: str = ""
    project_root: str = ""
    session_id: str = ""
    query: str = ""
    # Extra tags the caller can inject (model id, prior tool outputs, etc.)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        *,
        cwd: str = "",
        project_root: str = "",
        session_id: str = "",
        query: str = "",
        extra: dict | None = None,
    ) -> HazyHashContext:
        """Build a context from the current environment."""
        hostname = ""
        try:
            hostname = platform.node()
        except Exception:
            pass
        username = ""
        try:
            username = os.getlogin()
        except Exception:
            try:
                username = os.environ.get("USER") or os.environ.get("USERNAME") or ""
            except Exception:
                pass
        return cls(
            hostname=hostname,
            username=username,
            os_name=os.name,
            platform_tag=__import__("sys").platform,
            cwd=cwd or os.getcwd(),
            project_root=project_root,
            session_id=session_id,
            query=query,
            extra=extra or {},
        )


# ------------------------------------------------------------------
# Graph-backed hazy hash engine
# ------------------------------------------------------------------

# Abstraction levels (higher = more abstract)
LEVEL_MACHINE = 0
LEVEL_USER = 1
LEVEL_ZONE = 2      # e.g. "Projects", "Desktop", "Documents"
LEVEL_PROJECT = 3
LEVEL_PATH = 4


class HazyHash:
    """
    Kuzu-backed location memory with hierarchical abstraction.

    Records file locations as paths through an abstraction hierarchy.
    Lookups traverse the graph using context to narrow candidates, then
    score by fuzzy match + recency + co-occurrence strength.
    """

    def __init__(self, db_path: Path, *, scope: str = "global") -> None:
        self._db_path = db_path
        self._scope = scope
        self._conn: Any | None = None
        self._db: Any | None = None
        self._kuzu_ok: bool = False
        # JSON fallback index for when Kuzu is unavailable.
        self._fallback_path = db_path.parent / f"hazy_hash_{scope}.json"
        self._fallback: dict[str, list[dict]] = {}
        self._init_backend()

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        """Try Kuzu first; fall back to JSON."""
        try:
            import kuzu  # type: ignore
            self._db_path.mkdir(parents=True, exist_ok=True)
            self._db = kuzu.Database(str(self._db_path))
            self._conn = kuzu.Connection(self._db)
            self._ensure_schema()
            self._kuzu_ok = True
        except Exception:
            self._kuzu_ok = False
            self._fallback = self._load_fallback()

    def _ensure_schema(self) -> None:
        """Create the Kuzu node and relationship tables."""
        stmts = [
            (
                "CREATE NODE TABLE IF NOT EXISTS LocationNode("
                "nid STRING, "           # unique id
                "label STRING, "         # human-readable name
                "level INT64, "          # abstraction level (0=machine .. 4=path)
                "abs_path STRING, "      # full normalised path (empty for abstract nodes)
                "hostname STRING, "      # machine context
                "username STRING, "      # user context
                "os_name STRING, "       # os.name
                "scope STRING, "         # "global" or session_id
                "access_count INT64, "   # total times recorded
                "last_accessed DOUBLE, " # epoch timestamp
                "created_at DOUBLE, "    # epoch timestamp
                "PRIMARY KEY (nid))"
            ),
            (
                "CREATE REL TABLE IF NOT EXISTS LocationEdge("
                "FROM LocationNode TO LocationNode, "
                "weight DOUBLE, "        # co-occurrence strength
                "last_traversed DOUBLE)"
            ),
        ]
        for stmt in stmts:
            try:
                self._conn.execute(stmt)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, path: str, ctx: HazyHashContext) -> None:
        """
        Record that ``path`` was accessed in the given context.

        Decomposes the path into the abstraction hierarchy and upserts
        nodes + edges in the graph.
        """
        if not path:
            return
        norm = _normalise_path(path)
        chain = self._decompose(norm, ctx)
        if self._kuzu_ok:
            self._record_kuzu(chain, ctx)
        else:
            self._record_fallback(chain, ctx)

    def _decompose(self, norm_path: str, ctx: HazyHashContext) -> list[dict]:
        """
        Break a normalised path into hierarchy nodes.

        Returns a list from most abstract to most specific:
          [machine, user, zone, project, full_path]
        """
        parts: list[dict] = []

        # Level 0 — Machine
        machine_label = ctx.hostname or ctx.platform_tag or "unknown"
        parts.append({
            "level": LEVEL_MACHINE,
            "label": machine_label,
            "abs_path": "",
            "nid": f"machine:{machine_label}",
        })

        # Level 1 — User
        user_label = ctx.username or "default"
        parts.append({
            "level": LEVEL_USER,
            "label": user_label,
            "abs_path": "",
            "nid": f"user:{machine_label}/{user_label}",
        })

        # Level 2 — Zone (top-level directory category)
        zone = _detect_zone(norm_path, ctx)
        if zone:
            parts.append({
                "level": LEVEL_ZONE,
                "label": zone["label"],
                "abs_path": zone.get("abs_path", ""),
                "nid": f"zone:{machine_label}/{user_label}/{zone['label']}",
            })

        # Level 3 — Project (directory containing the file)
        project = _detect_project(norm_path, ctx)
        if project:
            parts.append({
                "level": LEVEL_PROJECT,
                "label": project["label"],
                "abs_path": project.get("abs_path", ""),
                "nid": f"project:{project['abs_path']}",
            })

        # Level 4 — Full path
        parts.append({
            "level": LEVEL_PATH,
            "label": Path(norm_path).name,
            "abs_path": norm_path,
            "nid": f"path:{norm_path}",
        })

        return parts

    def _record_kuzu(self, chain: list[dict], ctx: HazyHashContext) -> None:
        """Upsert nodes and edges in Kuzu."""
        now = time.time()
        prev_nid: str | None = None
        for node in chain:
            nid = node["nid"]
            # Upsert node
            try:
                result = self._conn.execute(
                    f'MATCH (n:LocationNode) WHERE n.nid = "{_esc(nid)}" '
                    f"RETURN n.access_count"
                )
                if result.has_next():
                    # Update existing
                    self._conn.execute(
                        f'MATCH (n:LocationNode) WHERE n.nid = "{_esc(nid)}" '
                        f"SET n.access_count = n.access_count + 1, "
                        f"n.last_accessed = {now}"
                    )
                else:
                    # Create new
                    self._conn.execute(
                        f"CREATE (n:LocationNode {{"
                        f'nid: "{_esc(nid)}", '
                        f'label: "{_esc(node["label"])}", '
                        f"level: {node['level']}, "
                        f'abs_path: "{_esc(node.get("abs_path", ""))}", '
                        f'hostname: "{_esc(ctx.hostname)}", '
                        f'username: "{_esc(ctx.username)}", '
                        f'os_name: "{_esc(ctx.os_name)}", '
                        f'scope: "{_esc(self._scope)}", '
                        f"access_count: 1, "
                        f"last_accessed: {now}, "
                        f"created_at: {now}}})"
                    )
            except Exception:
                pass

            # Create/update edge from parent
            if prev_nid:
                try:
                    result = self._conn.execute(
                        f'MATCH (a:LocationNode)-[e:LocationEdge]->(b:LocationNode) '
                        f'WHERE a.nid = "{_esc(prev_nid)}" AND b.nid = "{_esc(nid)}" '
                        f"RETURN e.weight"
                    )
                    if result.has_next():
                        self._conn.execute(
                            f'MATCH (a:LocationNode)-[e:LocationEdge]->(b:LocationNode) '
                            f'WHERE a.nid = "{_esc(prev_nid)}" AND b.nid = "{_esc(nid)}" '
                            f"SET e.weight = e.weight + 1.0, e.last_traversed = {now}"
                        )
                    else:
                        self._conn.execute(
                            f'MATCH (a:LocationNode), (b:LocationNode) '
                            f'WHERE a.nid = "{_esc(prev_nid)}" AND b.nid = "{_esc(nid)}" '
                            f"CREATE (a)-[:LocationEdge {{weight: 1.0, last_traversed: {now}}}]->(b)"
                        )
                except Exception:
                    pass

            prev_nid = nid

    def _record_fallback(self, chain: list[dict], ctx: HazyHashContext) -> None:
        """JSON fallback when Kuzu is unavailable."""
        path_node = chain[-1] if chain else None
        if not path_node:
            return
        key = ctx.query or path_node.get("abs_path", "")
        entry = {
            "abs_path": path_node.get("abs_path", ""),
            "chain": [n["nid"] for n in chain],
            "hostname": ctx.hostname,
            "username": ctx.username,
            "os_name": ctx.os_name,
            "cwd": ctx.cwd,
            "project_root": ctx.project_root,
            "ts": time.time(),
        }
        bucket = self._fallback.setdefault(key, [])
        bucket.append(entry)
        self._fallback[key] = bucket[-200:]
        self._save_fallback()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, ctx: HazyHashContext, *, limit: int = 20) -> list[dict]:
        """
        Return ranked candidate locations for the given context.

        Each result: {path, score, level, label, reason}

        The AI model receives these candidates and makes the final
        decision — including fuzzy name matching (leet speak, typos, etc.)
        that only the model can judge.
        """
        if self._kuzu_ok:
            return self._lookup_kuzu(ctx, limit=limit)
        return self._lookup_fallback(ctx, limit=limit)

    def _lookup_kuzu(self, ctx: HazyHashContext, *, limit: int = 20) -> list[dict]:
        """Traverse the Kuzu graph, scoring by context match."""
        candidates: list[dict] = []

        # Strategy 1: direct query-token match on labels and paths
        tokens = _tokenise(ctx.query)
        if tokens:
            for token in tokens[:5]:
                try:
                    result = self._conn.execute(
                        f'MATCH (n:LocationNode) '
                        f'WHERE n.label CONTAINS "{_esc(token)}" '
                        f'   OR n.abs_path CONTAINS "{_esc(token)}" '
                        f'RETURN n.nid, n.label, n.level, n.abs_path, '
                        f'       n.hostname, n.username, n.access_count, '
                        f'       n.last_accessed '
                        f'ORDER BY n.last_accessed DESC LIMIT 50'
                    )
                    while result.has_next():
                        row = result.get_next()
                        candidates.append(self._row_to_candidate(row, ctx, token))
                except Exception:
                    pass

        # Strategy 2: context-narrowed traversal (machine → user → zone → paths)
        try:
            machine_label = ctx.hostname or ctx.platform_tag or "unknown"
            user_label = ctx.username or "default"
            result = self._conn.execute(
                f'MATCH (m:LocationNode)-[e1:LocationEdge]->(u:LocationNode)'
                f'-[e2:LocationEdge*1..3]->(target:LocationNode) '
                f'WHERE m.nid = "machine:{_esc(machine_label)}" '
                f'  AND u.nid = "user:{_esc(machine_label)}/{_esc(user_label)}" '
                f'  AND target.level >= {LEVEL_ZONE} '
                f'RETURN target.nid, target.label, target.level, target.abs_path, '
                f'       target.hostname, target.username, target.access_count, '
                f'       target.last_accessed '
                f'ORDER BY target.last_accessed DESC LIMIT 100'
            )
            while result.has_next():
                row = result.get_next()
                candidates.append(self._row_to_candidate(row, ctx, "context_traversal"))
        except Exception:
            pass

        return self._rank_and_dedupe(candidates, ctx, limit=limit)

    @staticmethod
    def _row_to_candidate(row: list, ctx: HazyHashContext, match_reason: str) -> dict:
        """Convert a Kuzu result row to a candidate dict."""
        return {
            "nid": row[0],
            "label": row[1],
            "level": int(row[2]),
            "path": row[3] or "",
            "hostname": row[4] or "",
            "username": row[5] or "",
            "access_count": int(row[6] or 0),
            "last_accessed": float(row[7] or 0),
            "match_reason": match_reason,
        }

    def _lookup_fallback(self, ctx: HazyHashContext, *, limit: int = 20) -> list[dict]:
        """JSON fallback lookup with scoring."""
        candidates: list[dict] = []
        tokens = _tokenise(ctx.query)

        for key, entries in self._fallback.items():
            for entry in entries:
                abs_path = entry.get("abs_path", "")
                if not abs_path:
                    continue
                # Check if any query token appears in the key or path
                key_lower = key.lower()
                path_lower = abs_path.lower()
                matched_token = ""
                for t in tokens:
                    if t in key_lower or t in path_lower:
                        matched_token = t
                        break
                # Also include if context matches (same machine + user)
                context_match = (
                    entry.get("hostname") == ctx.hostname
                    and entry.get("username") == ctx.username
                )
                if not matched_token and not context_match:
                    continue
                candidates.append({
                    "nid": f"fallback:{abs_path}",
                    "label": Path(abs_path).name,
                    "level": LEVEL_PATH,
                    "path": abs_path,
                    "hostname": entry.get("hostname", ""),
                    "username": entry.get("username", ""),
                    "access_count": 1,
                    "last_accessed": entry.get("ts", 0),
                    "match_reason": matched_token or "context",
                })

        return self._rank_and_dedupe(candidates, ctx, limit=limit)

    def _rank_and_dedupe(
        self, candidates: list[dict], ctx: HazyHashContext, *, limit: int = 20,
    ) -> list[dict]:
        """Score, deduplicate, and rank candidates."""
        scored: list[tuple[float, dict]] = []
        seen_paths: set[str] = set()

        for c in candidates:
            path = c.get("path", "")
            nid = c.get("nid", "")
            key = path or nid
            if key in seen_paths:
                continue
            seen_paths.add(key)

            score = _score_candidate(c, ctx)
            c["score"] = round(score, 3)
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict] = []
        for _, c in scored[:limit]:
            results.append({
                "path": c.get("path", ""),
                "score": c["score"],
                "level": c.get("level", -1),
                "label": c.get("label", ""),
                "reason": c.get("match_reason", ""),
            })
        return results

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def promote_to_scope(self, target_scope: str) -> int:
        """
        Copy high-value nodes from this scope into a target scope.

        Used to promote ST (session) entries to LT (global) at session end.
        Returns the number of nodes promoted.
        """
        if not self._kuzu_ok:
            return 0
        count = 0
        try:
            result = self._conn.execute(
                f'MATCH (n:LocationNode) '
                f'WHERE n.scope = "{_esc(self._scope)}" '
                f'  AND n.access_count >= 2 '
                f'RETURN n.nid, n.label, n.level, n.abs_path, '
                f'       n.hostname, n.username, n.os_name, '
                f'       n.access_count, n.last_accessed, n.created_at'
            )
            while result.has_next():
                row = result.get_next()
                nid = str(row[0]).replace(f":{self._scope}", f":{target_scope}")
                try:
                    self._conn.execute(
                        f'MERGE (n:LocationNode {{nid: "{_esc(nid)}"}}) '
                        f'SET n.label = "{_esc(row[1])}", '
                        f"n.level = {row[2]}, "
                        f'n.abs_path = "{_esc(row[3])}", '
                        f'n.hostname = "{_esc(row[4])}", '
                        f'n.username = "{_esc(row[5])}", '
                        f'n.os_name = "{_esc(row[6])}", '
                        f'n.scope = "{_esc(target_scope)}", '
                        f"n.access_count = n.access_count + {row[7]}, "
                        f"n.last_accessed = {row[8]}, "
                        f"n.created_at = CASE WHEN n.created_at = 0 "
                        f"THEN {row[9]} ELSE n.created_at END"
                    )
                    count += 1
                except Exception:
                    pass
        except Exception:
            pass
        return count

    # ------------------------------------------------------------------
    # Fallback persistence
    # ------------------------------------------------------------------

    def _load_fallback(self) -> dict:
        if self._fallback_path.exists():
            try:
                return json.loads(
                    self._fallback_path.read_text(encoding="utf-8", errors="ignore")
                )
            except Exception:
                pass
        return {}

    def _save_fallback(self) -> None:
        try:
            self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
            self._fallback_path.write_text(
                json.dumps(self._fallback, indent=2), encoding="utf-8",
            )
        except Exception:
            pass


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def _score_candidate(c: dict, ctx: HazyHashContext) -> float:
    """
    Score a candidate by how well it matches the context.

    Factors: hostname match, username match, cwd proximity, query
    fuzzy match, access frequency, recency.
    """
    score = 0.0

    # Hostname match
    if ctx.hostname and c.get("hostname") == ctx.hostname:
        score += 3.0

    # Username match
    if ctx.username and c.get("username") == ctx.username:
        score += 2.0

    # CWD proximity — if the candidate path is under or near the cwd
    path = c.get("path", "")
    if path and ctx.cwd:
        cwd_norm = _normalise_path(ctx.cwd)
        path_norm = _normalise_path(path)
        if path_norm.startswith(cwd_norm):
            score += 4.0
        elif cwd_norm.startswith(path_norm):
            score += 2.0
        else:
            # Shared prefix depth bonus
            common = os.path.commonpath([cwd_norm, path_norm]) if cwd_norm and path_norm else ""
            if common and common != "/":
                depth = common.count("/")
                score += min(depth * 0.5, 2.0)

    # Query fuzzy match against label
    if ctx.query and c.get("label"):
        ratio = _fuzzy_ratio(ctx.query.lower(), c["label"].lower())
        score += ratio * 5.0  # up to 5.0 for perfect match

    # Query token match against path
    if ctx.query and path:
        tokens = _tokenise(ctx.query)
        path_lower = path.lower()
        for t in tokens:
            if t in path_lower:
                score += 1.5

    # Access frequency (logarithmic)
    access_count = c.get("access_count", 0)
    if access_count > 0:
        import math
        score += math.log2(access_count + 1) * 0.5

    # Recency (decay over 30 days for LT, 24h for ST)
    last_accessed = c.get("last_accessed", 0)
    if last_accessed > 0:
        age_days = (time.time() - last_accessed) / 86400.0
        if age_days < 1:
            score += 2.0
        elif age_days < 7:
            score += 1.0
        elif age_days < 30:
            score += 0.5

    # Level bonus — more specific = slightly higher
    level = c.get("level", 0)
    score += level * 0.3

    return score


# ------------------------------------------------------------------
# Path decomposition helpers
# ------------------------------------------------------------------

_ZONE_PATTERNS = {
    "Projects": re.compile(r"[/\\](projects?|repos?|src|dev|code|workspace)[/\\]", re.I),
    "Desktop": re.compile(r"[/\\]desktop[/\\]", re.I),
    "Documents": re.compile(r"[/\\]documents[/\\]", re.I),
    "Downloads": re.compile(r"[/\\]downloads[/\\]", re.I),
    "AppData": re.compile(r"[/\\](appdata|\.config|\.local)[/\\]", re.I),
    "System": re.compile(r"^(/usr|/etc|/var|/opt|C:\\Windows)", re.I),
    "Home": re.compile(r"[/\\](home|Users)[/\\]", re.I),
}


def _detect_zone(norm_path: str, ctx: HazyHashContext) -> dict | None:
    """Identify which zone a path belongs to."""
    for zone_name, pattern in _ZONE_PATTERNS.items():
        m = pattern.search(norm_path)
        if m:
            zone_path = norm_path[: m.end()].rstrip("/\\")
            return {"label": zone_name, "abs_path": zone_path}
    return None


def _detect_project(norm_path: str, ctx: HazyHashContext) -> dict | None:
    """
    Identify the project directory.

    Heuristic: the directory containing a known project marker (.git,
    package.json, pyproject.toml, etc.), or the directory just below
    a zone directory.
    """
    p = Path(norm_path)
    markers = {".git", "package.json", "pyproject.toml", "Cargo.toml",
               "go.mod", "pom.xml", ".sln", "manage.py", "setup.py"}
    # Walk up from the file looking for a project root
    for parent in p.parents:
        if any((parent / m).exists() for m in markers):
            return {"label": parent.name, "abs_path": str(parent)}
        # Stop at user home or drive root
        if parent == parent.parent:
            break
    # Fallback: use the project_root from context
    if ctx.project_root:
        pr = Path(ctx.project_root)
        return {"label": pr.name, "abs_path": str(pr)}
    return None


# ------------------------------------------------------------------
# Text utilities
# ------------------------------------------------------------------

def _normalise_path(path: str) -> str:
    """Normalise a path to forward slashes, resolved."""
    try:
        return str(Path(path).resolve()).replace("\\", "/")
    except Exception:
        return path.replace("\\", "/")


def _tokenise(text: str) -> list[str]:
    """Split text into searchable tokens (3+ chars, lowercased)."""
    stop = {"the", "and", "for", "from", "into", "with", "that", "this"}
    return [
        t for t in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
        if t not in stop
    ]


def _fuzzy_ratio(a: str, b: str) -> float:
    """Quick fuzzy match ratio (0.0 to 1.0)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _esc(text: str) -> str:
    """Escape a string for Kuzu Cypher literals."""
    return str(text).replace("\\", "\\\\").replace('"', '\\"')
