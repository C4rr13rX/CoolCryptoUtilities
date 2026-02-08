from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


try:
    import kuzu  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    kuzu = None  # type: ignore


def _graph_dir() -> Path:
    override = os.getenv("C0D3R_GRAPH_DB_DIR") or os.getenv("GRAPH_DB_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path("storage/graph/kuzu").resolve()


def graph_enabled() -> bool:
    vendor = (os.getenv("C0D3R_GRAPH_DB_VENDOR") or os.getenv("GRAPH_DB_VENDOR") or "kuzu").lower()
    return vendor == "kuzu" and kuzu is not None


class GraphStore:
    def __init__(self, path: Path | None = None) -> None:
        if kuzu is None:
            raise RuntimeError("kuzu not installed")
        self.path = path or _graph_dir()
        self.path.mkdir(parents=True, exist_ok=True)
        self.db = kuzu.Database(str(self.path))
        self.conn = kuzu.Connection(self.db)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        statements = [
            "CREATE NODE TABLE IF NOT EXISTS Equation("
            "eq_id STRING, text STRING, latex STRING, disciplines STRING, tool_used STRING, "
            "captured_at STRING, citations STRING, PRIMARY KEY (eq_id))",
            "CREATE REL TABLE IF NOT EXISTS EquationLink(FROM Equation TO Equation, relation_type STRING, notes STRING, created_at STRING)",
        ]
        for stmt in statements:
            try:
                self.conn.execute(stmt)
            except Exception:
                continue

    def rebuild_from_django(self) -> bool:
        try:
            from django.utils import timezone
            from core.models import Equation, EquationLink
            import pandas as pd
        except Exception:
            return False
        try:
            # Drop and recreate for a clean rebuild.
            try:
                self.conn.execute("DROP TABLE EquationLink")
            except Exception:
                pass
            try:
                self.conn.execute("DROP TABLE Equation")
            except Exception:
                pass
            self._ensure_schema()

            eq_rows = []
            for eq in Equation.objects.all():
                citations = eq.citations or []
                if not citations and eq.source and eq.source.citation:
                    citations = [eq.source.citation]
                captured_at = eq.captured_at or eq.created_at or timezone.now()
                eq_rows.append(
                    {
                        "eq_id": str(eq.id),
                        "text": eq.text or "",
                        "latex": eq.latex or "",
                        "disciplines": ",".join(eq.disciplines or eq.domains or []),
                        "tool_used": eq.tool_used or "",
                        "captured_at": captured_at.isoformat() if captured_at else "",
                        "citations": json.dumps(citations),
                    }
                )
            if eq_rows:
                df = pd.DataFrame(eq_rows)
                self.conn.execute("COPY Equation FROM df")

            link_rows = []
            for link in EquationLink.objects.all():
                link_rows.append(
                    {
                        "FROM": str(link.from_equation_id),
                        "TO": str(link.to_equation_id),
                        "relation_type": link.relation_type or "bridges",
                        "notes": link.notes or "",
                        "created_at": link.created_at.isoformat() if link.created_at else "",
                    }
                )
            if link_rows:
                df_rel = pd.DataFrame(link_rows)
                self.conn.execute("COPY EquationLink FROM df_rel")
            return True
        except Exception:
            return False

    def search_equations(self, query: str, limit: int = 12) -> list[dict]:
        if not query:
            return []
        try:
            escaped = query.replace("\\", "\\\\").replace('"', "\\\"")
            q = (
                "MATCH (e:Equation) "
                f"WHERE e.text CONTAINS \"{escaped}\" OR e.latex CONTAINS \"{escaped}\" "
                f"RETURN e.eq_id, e.text, e.disciplines LIMIT {int(limit)}"
            )
            result = self.conn.execute(q)
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append(
                    {
                        "eq_id": row[0],
                        "equation": row[1],
                        "domain": row[2] or "",
                        "summary": "",
                    }
                )
            return rows
        except Exception:
            return []


def get_graph_store() -> GraphStore | None:
    if not graph_enabled():
        return None
    try:
        return GraphStore()
    except Exception:
        return None


def sync_graph_from_django() -> bool:
    store = get_graph_store()
    if not store:
        return False
    return store.rebuild_from_django()


def search_graph_equations(query: str, limit: int = 12) -> list[dict]:
    store = get_graph_store()
    if not store:
        return []
    return store.search_equations(query, limit=limit)
