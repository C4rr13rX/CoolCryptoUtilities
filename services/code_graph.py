from __future__ import annotations

import ast
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "runtime",
    "storage",
    "logs",
    "data",
    "dist",
    "build",
    "lib",
    "bin",
    "include",
}


@dataclass
class GraphNode:
    id: str
    label: str
    kind: str
    file: str
    status: str = "ok"
    line: Optional[int] = None
    column: Optional[int] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "file": self.file,
            "status": self.status,
            "line": self.line,
            "column": self.column,
            "meta": self.meta,
        }


@dataclass
class GraphEdge:
    source: str
    target: str
    kind: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "id": f"{self.source}->{self.target}:{self.kind}",
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
        }


class FileAnalyzer(ast.NodeVisitor):
    def __init__(self, rel_path: str, file_node: GraphNode) -> None:
        self.rel_path = rel_path
        self.file_node = file_node
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []
        self.calls: List[Tuple[str, str]] = []
        self.scope: List[Tuple[str, str, str]] = []
        self.name_index: Dict[str, List[str]] = defaultdict(list)

    # ------------------------------------------------------------------ helpers
    def _register(self, node: GraphNode, parent_id: Optional[str] = None) -> None:
        self.nodes.append(node)
        parent = parent_id or self.file_node.id
        self.edges.append(GraphEdge(parent, node.id, "contains"))
        self.name_index[node.label].append(node.id)

    def _current_container(self) -> Optional[str]:
        return self.scope[-1][0] if self.scope else None

    def _qual_name(self, name: str) -> str:
        if not self.scope:
            return name
        chain = [entry[1] for entry in self.scope] + [name]
        return ".".join(chain)

    def _extract_call_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    # ------------------------------------------------------------------ visitors
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qual_name = self._qual_name(node.name)
        graph_node = GraphNode(
            id=f"{self.rel_path}::{qual_name}",
            label=node.name,
            kind="class",
            file=self.rel_path,
            line=getattr(node, "lineno", None),
            column=getattr(node, "col_offset", None),
        )
        parent_id = self._current_container() or self.file_node.id
        self._register(graph_node, parent_id=parent_id)
        self.scope.append((graph_node.id, node.name, "class"))
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    def _handle_function(self, node: ast.AST) -> None:
        func_name = getattr(node, "name", "function")
        qual_name = self._qual_name(func_name)
        container = self._current_container()
        kind = "function"
        if self.scope:
            # if the immediate scope is a class, treat as method
            parent_kind = self.scope[-1][2]
            if parent_kind == "class":
                kind = "method"
        graph_node = GraphNode(
            id=f"{self.rel_path}::{qual_name}",
            label=func_name,
            kind=kind,
            file=self.rel_path,
            line=getattr(node, "lineno", None),
            column=getattr(node, "col_offset", None),
        )
        parent_id = container or self.file_node.id
        self._register(graph_node, parent_id=parent_id)
        self.scope.append((graph_node.id, func_name, kind))
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        callee = self._extract_call_name(node.func)
        caller = self._current_container()
        if callee and caller:
            self.calls.append((caller, callee))
        self.generic_visit(node)


class ProjectGraph:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.call_records: List[Tuple[str, str]] = []
        self.name_index: Dict[str, List[str]] = defaultdict(list)
        self.reference_counts: Dict[str, int] = defaultdict(int)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    # ------------------------------------------------------------------ public
    def build(self) -> Dict[str, object]:
        files = list(self._iter_source_files())
        for file_path in files:
            self._process_file(file_path)
        self._resolve_references()
        self._mark_unused_nodes()
        return {
            "nodes": [node.as_dict() for node in self.nodes.values()],
            "edges": [edge.as_dict() for edge in self.edges],
            "warnings": self.warnings,
            "errors": self.errors,
            "generated_at": time.time(),
            "summary": {
                "files": len([n for n in self.nodes.values() if n.kind == "file"]),
                "classes": len([n for n in self.nodes.values() if n.kind == "class"]),
                "functions": len([n for n in self.nodes.values() if n.kind in {"function", "method"}]),
            },
        }

    # ------------------------------------------------------------------ helpers
    def _iter_source_files(self) -> Iterable[Path]:
        for path in self.root.rglob("*.py"):
            rel = path.relative_to(self.root)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            yield path

    def _process_file(self, path: Path) -> None:
        rel_path = str(path.relative_to(self.root))
        file_node = GraphNode(
            id=f"file::{rel_path}",
            label=rel_path,
            kind="file",
            file=rel_path,
        )
        self.nodes[file_node.id] = file_node
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel_path)
        except Exception as exc:
            file_node.status = "broken"
            file_node.meta["error"] = str(exc)
            self.errors.append(file_node.id)
            return

        analyzer = FileAnalyzer(rel_path, file_node)
        analyzer.visit(tree)
        for node in analyzer.nodes:
            self.nodes[node.id] = node
            if node.status == "broken":
                self.errors.append(node.id)
        for edge in analyzer.edges:
            self.edges.append(edge)
        for name, ids in analyzer.name_index.items():
            self.name_index[name].extend(ids)
        self.call_records.extend(analyzer.calls)

    def _resolve_references(self) -> None:
        seen_pairs = set()
        for caller_id, callee_name in self.call_records:
            targets = self.name_index.get(callee_name, [])
            for target_id in targets:
                self.reference_counts[target_id] += 1
                pair_key = (caller_id, target_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                self.edges.append(GraphEdge(caller_id, target_id, "calls"))

    def _mark_unused_nodes(self) -> None:
        for node in self.nodes.values():
            if node.kind not in {"function", "method", "class"}:
                continue
            if node.label.startswith("__"):
                continue
            if self.reference_counts.get(node.id):
                continue
            node.status = "unused"
            self.warnings.append(node.id)


def build_code_graph() -> Dict[str, object]:
    graph = ProjectGraph(PROJECT_ROOT)
    return graph.build()
