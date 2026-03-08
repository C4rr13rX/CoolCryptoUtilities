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
            # Equation nodes with metadata for search.
            "CREATE NODE TABLE IF NOT EXISTS Equation("
            "eq_id STRING, text STRING, latex STRING, disciplines STRING, "
            "variables STRING, label STRING, confidence DOUBLE, "
            "constraints STRING, assumptions STRING, "
            "tool_used STRING, captured_at STRING, citations STRING, "
            "PRIMARY KEY (eq_id))",
            # Discipline nodes for traversal.
            "CREATE NODE TABLE IF NOT EXISTS Discipline("
            "name STRING, PRIMARY KEY (name))",
            # Variable nodes for cross-equation linking.
            "CREATE NODE TABLE IF NOT EXISTS Variable("
            "symbol STRING, units STRING, description STRING, "
            "PRIMARY KEY (symbol))",
            # Equation-to-equation links.
            "CREATE REL TABLE IF NOT EXISTS EquationLink("
            "FROM Equation TO Equation, "
            "relation_type STRING, notes STRING, created_at STRING)",
            # Equation-to-discipline membership.
            "CREATE REL TABLE IF NOT EXISTS BelongsToDiscipline("
            "FROM Equation TO Discipline)",
            # Equation-to-variable usage.
            "CREATE REL TABLE IF NOT EXISTS UsesVariable("
            "FROM Equation TO Variable)",
        ]
        for stmt in statements:
            try:
                self.conn.execute(stmt)
            except Exception:
                continue

    def rebuild_from_django(self) -> bool:
        try:
            from django.utils import timezone
            from core.models import Equation, EquationLink, EquationVariable
            import pandas as pd
        except Exception:
            return False
        try:
            # Drop and recreate for a clean rebuild.
            for table in (
                "EquationLink", "BelongsToDiscipline", "UsesVariable",
                "Equation", "Discipline", "Variable",
            ):
                try:
                    self.conn.execute(f"DROP TABLE {table}")
                except Exception:
                    pass
            self._ensure_schema()

            # --- Equations ---
            eq_rows = []
            disc_set: set[str] = set()
            for eq in Equation.objects.all():
                citations = eq.citations or []
                if not citations and eq.source and eq.source.citation:
                    citations = [eq.source.citation]
                captured_at = eq.captured_at or eq.created_at or timezone.now()
                disciplines = eq.disciplines or eq.domains or []
                disc_set.update(disciplines)
                label = ""
                if eq.constraints:
                    label = str(eq.constraints[0]) if eq.constraints else ""
                eq_rows.append(
                    {
                        "eq_id": str(eq.id),
                        "text": eq.text or "",
                        "latex": eq.latex or "",
                        "disciplines": ",".join(disciplines),
                        "variables": ",".join(eq.variables or []),
                        "label": label,
                        "confidence": float(eq.confidence) if hasattr(eq, "confidence") else 0.5,
                        "constraints": json.dumps(eq.constraints or []),
                        "assumptions": json.dumps(eq.assumptions or []),
                        "tool_used": eq.tool_used or "",
                        "captured_at": captured_at.isoformat() if captured_at else "",
                        "citations": json.dumps(citations),
                    }
                )
            if eq_rows:
                df = pd.DataFrame(eq_rows)
                self.conn.execute("COPY Equation FROM df")

            # --- Disciplines ---
            if disc_set:
                disc_rows = [{"name": d} for d in disc_set]
                df_disc = pd.DataFrame(disc_rows)
                self.conn.execute("COPY Discipline FROM df_disc")

            # --- Variables ---
            var_rows = []
            for v in EquationVariable.objects.all():
                var_rows.append({
                    "symbol": v.symbol,
                    "units": v.units or "",
                    "description": v.description or v.name or "",
                })
            if var_rows:
                df_var = pd.DataFrame(var_rows)
                self.conn.execute("COPY Variable FROM df_var")

            # --- Equation links ---
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

            # --- BelongsToDiscipline edges ---
            belongs_rows = []
            for eq in Equation.objects.all():
                for disc in (eq.disciplines or eq.domains or []):
                    if disc in disc_set:
                        belongs_rows.append({"FROM": str(eq.id), "TO": disc})
            if belongs_rows:
                df_belongs = pd.DataFrame(belongs_rows)
                self.conn.execute("COPY BelongsToDiscipline FROM df_belongs")

            # --- UsesVariable edges ---
            uses_rows = []
            known_vars = {v.symbol for v in EquationVariable.objects.all()}
            for eq in Equation.objects.all():
                for var in (eq.variables or []):
                    if var in known_vars:
                        uses_rows.append({"FROM": str(eq.id), "TO": var})
            if uses_rows:
                df_uses = pd.DataFrame(uses_rows)
                self.conn.execute("COPY UsesVariable FROM df_uses")

            return True
        except Exception:
            return False

    def search_equations(self, query: str, limit: int = 12) -> list[dict]:
        """Search equations by text, latex, label, or variable content."""
        if not query:
            return []
        try:
            escaped = query.replace("\\", "\\\\").replace('"', '\\"')
            q = (
                "MATCH (e:Equation) "
                f'WHERE e.text CONTAINS "{escaped}" '
                f'OR e.latex CONTAINS "{escaped}" '
                f'OR e.label CONTAINS "{escaped}" '
                f'OR e.variables CONTAINS "{escaped}" '
                f"RETURN e.eq_id, e.text, e.disciplines, e.label, "
                f"e.variables, e.confidence "
                f"LIMIT {int(limit)}"
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
                        "label": row[3] or "",
                        "variables": (row[4] or "").split(","),
                        "confidence": row[5] if row[5] is not None else 0.5,
                        "summary": "",
                    }
                )
            return rows
        except Exception:
            return []

    def search_by_discipline(self, discipline: str, limit: int = 20) -> list[dict]:
        """Find all equations in a discipline via graph traversal."""
        if not discipline:
            return []
        try:
            escaped = discipline.replace("\\", "\\\\").replace('"', '\\"')
            q = (
                "MATCH (e:Equation)-[:BelongsToDiscipline]->(d:Discipline) "
                f'WHERE d.name = "{escaped}" '
                f"RETURN e.eq_id, e.text, e.label, e.variables, e.confidence "
                f"LIMIT {int(limit)}"
            )
            result = self.conn.execute(q)
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append({
                    "eq_id": row[0],
                    "equation": row[1],
                    "label": row[2] or "",
                    "variables": (row[3] or "").split(","),
                    "confidence": row[4] if row[4] is not None else 0.5,
                    "domain": discipline,
                })
            return rows
        except Exception:
            return []

    def search_by_variable(self, symbol: str, limit: int = 20) -> list[dict]:
        """Find all equations that use a specific variable."""
        if not symbol:
            return []
        try:
            escaped = symbol.replace("\\", "\\\\").replace('"', '\\"')
            q = (
                "MATCH (e:Equation)-[:UsesVariable]->(v:Variable) "
                f'WHERE v.symbol = "{escaped}" '
                f"RETURN e.eq_id, e.text, e.disciplines, e.label "
                f"LIMIT {int(limit)}"
            )
            result = self.conn.execute(q)
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append({
                    "eq_id": row[0],
                    "equation": row[1],
                    "domain": row[2] or "",
                    "label": row[3] or "",
                    "variable": symbol,
                })
            return rows
        except Exception:
            return []

    def find_gaps(self, discipline_a: str, discipline_b: str) -> list[dict]:
        """
        Find equations in two disciplines that share variables but have
        no direct EquationLink — these are potential bridging gaps.
        """
        if not discipline_a or not discipline_b:
            return []
        try:
            esc_a = discipline_a.replace("\\", "\\\\").replace('"', '\\"')
            esc_b = discipline_b.replace("\\", "\\\\").replace('"', '\\"')
            # Find equations sharing a variable across disciplines
            # but not directly linked.
            q = (
                "MATCH (ea:Equation)-[:UsesVariable]->(v:Variable)"
                "<-[:UsesVariable]-(eb:Equation) "
                f'WHERE ea.disciplines CONTAINS "{esc_a}" '
                f'AND eb.disciplines CONTAINS "{esc_b}" '
                "AND NOT EXISTS { "
                "  MATCH (ea)-[:EquationLink]-(eb) "
                "} "
                "RETURN ea.eq_id, ea.text, eb.eq_id, eb.text, v.symbol "
                "LIMIT 50"
            )
            result = self.conn.execute(q)
            gaps = []
            while result.has_next():
                row = result.get_next()
                gaps.append({
                    "eq_a_id": row[0],
                    "eq_a": row[1],
                    "eq_b_id": row[2],
                    "eq_b": row[3],
                    "shared_variable": row[4],
                    "discipline_a": discipline_a,
                    "discipline_b": discipline_b,
                })
            return gaps
        except Exception:
            return []

    def find_linked(self, eq_id: str) -> list[dict]:
        """Get all equations linked to a given equation."""
        if not eq_id:
            return []
        try:
            escaped = str(eq_id).replace("\\", "\\\\").replace('"', '\\"')
            q = (
                "MATCH (e1:Equation)-[r:EquationLink]-(e2:Equation) "
                f'WHERE e1.eq_id = "{escaped}" '
                "RETURN e2.eq_id, e2.text, e2.disciplines, r.relation_type, r.notes"
            )
            result = self.conn.execute(q)
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append({
                    "eq_id": row[0],
                    "equation": row[1],
                    "domain": row[2] or "",
                    "relation": row[3] or "",
                    "notes": row[4] or "",
                })
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
