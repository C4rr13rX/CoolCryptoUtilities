"""Repository-agnostic, cancellable source-code graph indexing.

HTTP callers never scan a repository. They read a small catalog/cache and may
start a background job. This keeps repository switching responsive even while
another (possibly remote) source is being indexed.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "runtime" / "code_graph"
CATALOG_PATH = RUNTIME_ROOT / "repositories.json"
CACHE_ROOT = RUNTIME_ROOT / "repositories"
CHECKOUT_ROOT = RUNTIME_ROOT / "checkouts"

EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode",
    "runtime", "storage", "logs", "data", "dist", "build", "target", "vendor",
    "coverage", ".next", ".nuxt", "bin", "obj",
}
SUPPORTED_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".vue",
    ".rs", ".go", ".java", ".kt", ".kts", ".cs", ".c", ".h", ".cc",
    ".cpp", ".cxx", ".hpp", ".hh", ".rb", ".php", ".swift", ".scala",
    ".sh", ".ps1", ".sql",
}
MAX_FILES = max(100, int(os.getenv("CODE_GRAPH_MAX_FILES", "5000")))
MAX_FILE_BYTES = max(32_768, int(os.getenv("CODE_GRAPH_MAX_FILE_BYTES", "1500000")))

_CATALOG_LOCK = threading.RLock()
_JOBS_LOCK = threading.RLock()
_JOBS: Dict[str, "BuildJob"] = {}


@dataclass
class GraphNode:
    id: str
    label: str
    kind: str
    file: str = ""
    status: str = "ok"
    line: Optional[int] = None
    column: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "kind": self.kind,
            "file": self.file, "status": self.status, "line": self.line,
            "column": self.column, "meta": self.meta,
        }


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    kind: str
    meta: Tuple[Tuple[str, Any], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        suffix = hashlib.sha1(f"{self.source}|{self.target}|{self.kind}".encode()).hexdigest()[:10]
        return {
            "id": f"edge::{suffix}", "source": self.source, "target": self.target,
            "kind": self.kind, "meta": dict(self.meta),
        }


@dataclass
class Definition:
    node: GraphNode
    calls: List[str] = field(default_factory=list)


class CancelledBuild(RuntimeError):
    pass


class BuildJob:
    def __init__(self, repository_id: str, *, refresh_remote: bool = False) -> None:
        self.repository_id = repository_id
        self.refresh_remote = refresh_remote
        self.cancel_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name=f"codegraph-{repository_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> None:
        self.cancel_event.set()

    def _run(self) -> None:
        try:
            _update_repository(self.repository_id, status="preparing", progress=0.01, error="")
            repo = get_repository(self.repository_id)
            if not repo:
                return
            root = _materialize_repository(repo, self.cancel_event, self.refresh_remote)
            def save_partial(partial: Dict[str, Any]) -> None:
                partial["repository"] = _public_repository(get_repository(self.repository_id) or repo)
                _save_graph_cache(self.repository_id, partial)

            graph = ProjectGraph(root, cancel_event=self.cancel_event, repository=repo).build(
                progress=lambda value, message: _update_repository(
                    self.repository_id, status="indexing", progress=value, message=message,
                ),
                partial=save_partial,
            )
            if self.cancel_event.is_set():
                raise CancelledBuild()
            graph["repository"] = _public_repository(get_repository(self.repository_id) or repo)
            _save_graph_cache(self.repository_id, graph)
            _update_repository(
                self.repository_id,
                status="ready",
                progress=1.0,
                message="Index ready",
                error="",
                summary=graph.get("summary") or {},
                generated_at=graph.get("generated_at"),
            )
        except CancelledBuild:
            _update_repository(self.repository_id, status="cancelled", message="Indexing cancelled")
        except Exception as exc:
            _update_repository(
                self.repository_id, status="error", message="Indexing failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            with _JOBS_LOCK:
                if _JOBS.get(self.repository_id) is self:
                    _JOBS.pop(self.repository_id, None)


class PythonAnalyzer(ast.NodeVisitor):
    def __init__(self, rel_path: str, file_id: str) -> None:
        self.rel_path = rel_path
        self.file_id = file_id
        self.scope: List[Tuple[str, str, str]] = []
        self.definitions: List[Definition] = []
        self.edges: List[GraphEdge] = []
        self.imports: List[str] = []

    def _container(self) -> str:
        return self.scope[-1][0] if self.scope else self.file_id

    def _qualify(self, name: str) -> str:
        return ".".join([item[1] for item in self.scope] + [name])

    @staticmethod
    def _annotation(node: Optional[ast.AST]) -> str:
        if node is None:
            return ""
        try:
            return ast.unparse(node)
        except Exception:
            return ""

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qual = self._qualify(node.name)
        graph_node = GraphNode(
            id=f"symbol::{self.rel_path}::{qual}", label=node.name, kind="class",
            file=self.rel_path, line=node.lineno, column=node.col_offset,
            meta={
                "qualified_name": qual,
                "bases": [self._annotation(base) for base in node.bases],
                "inputs": [], "outputs": [],
            },
        )
        self.edges.append(GraphEdge(self._container(), graph_node.id, "contains"))
        self.definitions.append(Definition(graph_node))
        parent = self._container()
        self.scope.append((graph_node.id, node.name, "class"))
        self.generic_visit(node)
        self.scope.pop()
        for base in graph_node.meta["bases"]:
            if base:
                graph_node.meta.setdefault("references", []).append(base)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qual = self._qualify(node.name)
        kind = "method" if self.scope and self.scope[-1][2] == "class" else "function"
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        inputs = [
            {"name": arg.arg, "type": self._annotation(arg.annotation)} for arg in arguments
            if arg.arg not in {"self", "cls"}
        ]
        if node.args.vararg:
            inputs.append({"name": f"*{node.args.vararg.arg}", "type": self._annotation(node.args.vararg.annotation)})
        if node.args.kwarg:
            inputs.append({"name": f"**{node.args.kwarg.arg}", "type": self._annotation(node.args.kwarg.annotation)})
        output = self._annotation(node.returns)
        graph_node = GraphNode(
            id=f"symbol::{self.rel_path}::{qual}", label=node.name, kind=kind,
            file=self.rel_path, line=node.lineno, column=node.col_offset,
            meta={
                "qualified_name": qual, "inputs": inputs,
                "outputs": [output] if output else [],
                "async": isinstance(node, ast.AsyncFunctionDef),
                "signature": f"{node.name}({', '.join(item['name'] for item in inputs)})",
            },
        )
        self.edges.append(GraphEdge(self._container(), graph_node.id, "contains"))
        definition = Definition(graph_node)
        self.definitions.append(definition)
        self.scope.append((graph_node.id, node.name, kind))
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name = self._call_name(node.func)
        if name and self.scope:
            current_id = self.scope[-1][0]
            for definition in reversed(self.definitions):
                if definition.node.id == current_id:
                    definition.calls.append(name)
                    break
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend(alias.name for alias in node.names if alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append(node.module)


GENERIC_CLASS_PATTERNS = [
    re.compile(r"^\s*(?:export\s+)?(?:public\s+|private\s+|protected\s+|internal\s+|abstract\s+|final\s+|sealed\s+|open\s+)*(?:class|struct|interface|trait|enum)\s+([A-Za-z_$][\w$]*)", re.M),
    re.compile(r"^\s*(?:module|namespace)\s+([A-Za-z_$][\w$.:]*)", re.M),
]
GENERIC_FUNCTION_PATTERNS = [
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*(?:->|:)?\s*([^\n{=>]*)", re.M),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>", re.M),
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][\w]*)\s*\(([^)]*)\)\s*(?:->\s*([^\n{]+))?", re.M),
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)\s*\(([^)]*)\)\s*([^\n{]*)", re.M),
    re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|internal\s+|static\s+|virtual\s+|override\s+|final\s+|synchronized\s+|abstract\s+|inline\s+|constexpr\s+|template\s*<[^>]+>\s*)+(?:[\w:<>,?\[\].*&]+\s+)+([A-Za-z_$][\w$]*)\s*\(([^;{}]*)\)\s*(?:const\s*)?(?:\{|=>)", re.M),
    re.compile(r"^\s*def\s+([A-Za-z_][\w!?=]*)\s*(?:\(([^)]*)\))?", re.M),
]
GENERIC_IMPORT_PATTERNS = [
    re.compile(r"(?:from\s+['\"]([^'\"]+)['\"]|require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)|import\s+(?:[^;\n]+?\s+from\s+)?['\"]([^'\"]+)['\"])", re.M),
    re.compile(r"^\s*(?:use|mod|package|import|using)\s+([A-Za-z_$][\w$.:/\\-]*)", re.M),
    re.compile(r"^\s*#include\s*[<\"]([^>\"]+)[>\"]", re.M),
]


def _generic_definitions(rel_path: str, file_id: str, source: str) -> Tuple[List[Definition], List[GraphEdge], List[str]]:
    definitions: List[Definition] = []
    edges: List[GraphEdge] = []
    occupied: set[Tuple[str, int]] = set()
    for pattern in GENERIC_CLASS_PATTERNS:
        for match in pattern.finditer(source):
            label = match.group(1)
            line = source.count("\n", 0, match.start()) + 1
            key = (label, line)
            if key in occupied:
                continue
            occupied.add(key)
            node = GraphNode(
                id=f"symbol::{rel_path}::{label}@{line}", label=label, kind="class",
                file=rel_path, line=line,
                meta={"qualified_name": label, "inputs": [], "outputs": []},
            )
            definitions.append(Definition(node))
            edges.append(GraphEdge(file_id, node.id, "contains"))
    for pattern in GENERIC_FUNCTION_PATTERNS:
        for match in pattern.finditer(source):
            label = match.group(1)
            line = source.count("\n", 0, match.start()) + 1
            key = (label, line)
            if key in occupied or label in {"if", "for", "while", "switch", "catch"}:
                continue
            occupied.add(key)
            args = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
            output = match.group(3).strip() if match.lastindex and match.lastindex >= 3 and match.group(3) else ""
            inputs = [{"name": item.strip(), "type": ""} for item in (args or "").split(",") if item.strip()]
            node = GraphNode(
                id=f"symbol::{rel_path}::{label}@{line}", label=label, kind="function",
                file=rel_path, line=line,
                meta={
                    "qualified_name": label, "inputs": inputs,
                    "outputs": [output] if output else [],
                    "signature": f"{label}({', '.join(item['name'] for item in inputs)})",
                },
            )
            # A conservative lexical call set is sufficient for wiring while avoiding
            # the old all-to-all match for duplicate function names.
            body_window = source[match.end():match.end() + 12_000]
            calls = re.findall(r"\b([A-Za-z_$][\w$]*)\s*\(", body_window)
            definitions.append(Definition(node, calls=calls[:500]))
            edges.append(GraphEdge(file_id, node.id, "contains"))
    imports: List[str] = []
    for pattern in GENERIC_IMPORT_PATTERNS:
        for match in pattern.finditer(source):
            value = next((group for group in match.groups() if group), "")
            if value:
                imports.append(value)
    return definitions, edges, sorted(set(imports))


class ProjectGraph:
    def __init__(self, root: Path, *, cancel_event: threading.Event, repository: Dict[str, Any]) -> None:
        self.root = root.resolve()
        self.cancel_event = cancel_event
        self.repository = repository
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: set[GraphEdge] = set()
        self.definitions: List[Definition] = []
        self.file_imports: Dict[str, List[str]] = {}
        self.file_by_module: Dict[str, str] = {}
        self.file_id_by_path: Dict[str, str] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.truncated = False

    def build(self, *, progress, partial=None) -> Dict[str, Any]:
        files = list(self._iter_source_files())
        total = max(1, len(files))
        repo_id = f"repository::{self.repository['id']}"
        self.nodes[repo_id] = GraphNode(
            id=repo_id, label=self.repository.get("name") or self.root.name,
            kind="repository", meta={"path": str(self.root), "inputs": [], "outputs": []},
        )
        self._create_structure_nodes(files, repo_id)
        if partial:
            partial(self._payload(files, building=True))
        last_partial = time.monotonic()
        for index, path in enumerate(files):
            self._check_cancel()
            rel = path.relative_to(self.root).as_posix()
            progress(0.08 + 0.80 * ((index + 1) / total), f"Indexing {rel}")
            self._process_file(path, rel)
            if partial and (index == len(files) - 1 or index % 50 == 49 or time.monotonic() - last_partial >= 1.2):
                partial(self._payload(files, building=True))
                last_partial = time.monotonic()
        progress(0.90, "Resolving calls and imports")
        self._resolve_relationships()
        self._mark_unreferenced()
        progress(0.98, "Writing graph cache")
        return self._payload(files, building=False)

    def _payload(self, files: List[Path], *, building: bool) -> Dict[str, Any]:
        summary = {
            "files": sum(node.kind == "file" for node in self.nodes.values()),
            "modules": sum(node.kind == "module" for node in self.nodes.values()),
            "classes": sum(node.kind == "class" for node in self.nodes.values()),
            "functions": sum(node.kind in {"function", "method"} for node in self.nodes.values()),
            "relationships": len(self.edges),
            "truncated": self.truncated,
        }
        return {
            "nodes": [node.as_dict() for node in self.nodes.values()],
            "edges": [edge.as_dict() for edge in sorted(self.edges, key=lambda e: (e.source, e.target, e.kind))],
            "file_links": self._file_links(),
            "warnings": self.warnings,
            "errors": self.errors,
            "summary": summary,
            "generated_at": time.time(),
            "files": [{"path": path.relative_to(self.root).as_posix()} for path in files],
            "entry_points": [node.id for node in self.nodes.values() if node.meta.get("entry_point")],
            "cached": False,
            "building": building,
            "partial": building,
        }

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledBuild()

    def _iter_source_files(self) -> Iterable[Path]:
        count = 0
        for current, dirs, names in os.walk(self.root):
            self._check_cancel()
            dirs[:] = sorted(directory for directory in dirs if directory not in EXCLUDE_DIRS and not directory.startswith("."))
            for name in sorted(names):
                path = Path(current) / name
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                try:
                    if path.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                yield path
                count += 1
                if count >= MAX_FILES:
                    self.truncated = True
                    return

    def _create_structure_nodes(self, files: List[Path], repo_id: str) -> None:
        modules: set[str] = set()
        for path in files:
            rel = path.relative_to(self.root).as_posix()
            parts = Path(rel).parts
            module = parts[0] if len(parts) > 1 else "root"
            modules.add(module)
            module_id = f"module::{module}"
            file_id = f"file::{rel}"
            self.file_id_by_path[rel] = file_id
            self.file_by_module[_module_name(rel)] = file_id
            self.nodes[file_id] = GraphNode(
                id=file_id, label=Path(rel).name, kind="file", file=rel,
                meta={"path": rel, "extension": path.suffix.lower(), "module": _module_name(rel), "inputs": [], "outputs": []},
            )
            self.edges.add(GraphEdge(module_id, file_id, "contains"))
        for module in sorted(modules):
            module_id = f"module::{module}"
            self.nodes[module_id] = GraphNode(
                id=module_id, label=module, kind="module",
                meta={"path": module, "inputs": [], "outputs": []},
            )
            self.edges.add(GraphEdge(repo_id, module_id, "contains"))

    def _process_file(self, path: Path, rel: str) -> None:
        file_id = self.file_id_by_path[rel]
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            file_name = path.name.lower()
            conventional_entry = file_name in {
                "main.py", "app.py", "manage.py", "main.ts", "main.tsx", "main.js", "main.jsx",
                "index.ts", "index.tsx", "index.js", "index.jsx", "main.rs", "main.go",
                "program.cs", "application.java", "main.java",
            }
            explicit_entry = "if __name__" in source and "__main__" in source
            self.nodes[file_id].meta["entry_point"] = bool(conventional_entry or explicit_entry)
            depth = max(0, len(Path(rel).parts) - 1)
            if rel.lower() in {"main.py", "main.ts", "main.tsx", "main.js", "main.rs", "main.go", "program.cs"}:
                entry_priority = 100
            elif conventional_entry:
                entry_priority = max(45, 80 - depth * 8)
            elif explicit_entry:
                entry_priority = max(10, 35 - depth * 4)
            else:
                entry_priority = 0
            self.nodes[file_id].meta["entry_priority"] = entry_priority
            if path.suffix.lower() in {".py", ".pyi"}:
                tree = ast.parse(source, filename=rel)
                analyzer = PythonAnalyzer(rel, file_id)
                analyzer.visit(tree)
                definitions, edges, imports = analyzer.definitions, analyzer.edges, analyzer.imports
            else:
                definitions, edges, imports = _generic_definitions(rel, file_id, source)
        except Exception as exc:
            self.nodes[file_id].status = "broken"
            self.nodes[file_id].meta["error"] = str(exc)
            self.errors.append(file_id)
            return
        for definition in definitions:
            if definition.node.label.lower() in {"main", "run", "start", "bootstrap"} and self.nodes[file_id].meta.get("entry_point"):
                definition.node.meta["entry_point"] = True
            self.nodes[definition.node.id] = definition.node
            self.definitions.append(definition)
        self.edges.update(edges)
        self.file_imports[file_id] = imports
        self.nodes[file_id].meta["imports"] = imports
        self.nodes[file_id].meta["definitions"] = [item.node.label for item in definitions]

    def _resolve_relationships(self) -> None:
        by_name: Dict[str, List[Definition]] = defaultdict(list)
        by_file_name: Dict[Tuple[str, str], Definition] = {}
        for definition in self.definitions:
            by_name[definition.node.label].append(definition)
            by_file_name[(definition.node.file, definition.node.label)] = definition
        referenced: set[str] = set()
        for definition in self.definitions:
            for call in sorted(set(definition.calls)):
                target = by_file_name.get((definition.node.file, call))
                if target is None and len(by_name.get(call, [])) == 1:
                    target = by_name[call][0]
                if target and target.node.id != definition.node.id:
                    self.edges.add(GraphEdge(definition.node.id, target.node.id, "calls"))
                    referenced.add(target.node.id)
        for source_id, imports in self.file_imports.items():
            for imported in imports:
                target_id = self._resolve_import(imported)
                if target_id and target_id != source_id:
                    self.edges.add(GraphEdge(source_id, target_id, "imports"))
                    referenced.add(target_id)
        for node_id in referenced:
            node = self.nodes.get(node_id)
            if node:
                node.meta["referenced"] = True

    def _resolve_import(self, imported: str) -> Optional[str]:
        cleaned = imported.replace("\\", "/").lstrip("./").replace("/", ".")
        candidates = [cleaned, cleaned.rsplit(".", 1)[0] if "." in cleaned else cleaned]
        for candidate in candidates:
            if candidate in self.file_by_module:
                return self.file_by_module[candidate]
        for module, file_id in self.file_by_module.items():
            if module.endswith(cleaned) or cleaned.endswith(module):
                return file_id
        return None

    def _mark_unreferenced(self) -> None:
        inbound = {edge.target for edge in self.edges if edge.kind in {"calls", "imports", "inherits"}}
        for definition in self.definitions:
            if definition.node.id not in inbound and not definition.node.label.startswith("__"):
                definition.node.status = "unused"
                self.warnings.append(definition.node.id)

    def _file_links(self) -> List[Dict[str, Any]]:
        counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"imports": 0, "calls": 0})
        node_file = {node.id: self.file_id_by_path.get(node.file) for node in self.nodes.values() if node.file}
        for edge in self.edges:
            if edge.kind not in {"imports", "calls"}:
                continue
            source = edge.source if edge.source.startswith("file::") else node_file.get(edge.source)
            target = edge.target if edge.target.startswith("file::") else node_file.get(edge.target)
            if not source or not target or source == target:
                continue
            counts[(source, target)][edge.kind] += 1
        return [
            {"source": source, "target": target, **kinds, "weight": kinds["imports"] + kinds["calls"]}
            for (source, target), kinds in sorted(counts.items())
        ]


def _module_name(rel_path: str) -> str:
    path = Path(rel_path)
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] in {"__init__", "index"}:
        parts = parts[:-1]
    return ".".join(parts)


def _default_catalog() -> Dict[str, Any]:
    repo_id = hashlib.sha1(str(PROJECT_ROOT).lower().encode()).hexdigest()[:12]
    now = time.time()
    return {
        "active_id": repo_id,
        "repositories": [{
            "id": repo_id, "name": PROJECT_ROOT.name, "source_type": "local",
            "location": str(PROJECT_ROOT), "branch": "", "status": "idle",
            "progress": 0.0, "message": "Ready to index", "error": "",
            "created_at": now, "updated_at": now, "summary": {},
        }],
    }


def _load_catalog() -> Dict[str, Any]:
    with _CATALOG_LOCK:
        if not CATALOG_PATH.exists():
            catalog = _default_catalog()
            _save_catalog(catalog)
            return catalog
        try:
            catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            catalog = _default_catalog()
        if not isinstance(catalog.get("repositories"), list) or not catalog["repositories"]:
            catalog = _default_catalog()
        active = catalog.get("active_id")
        ids = {repo.get("id") for repo in catalog["repositories"]}
        if active not in ids:
            catalog["active_id"] = catalog["repositories"][0]["id"]
        for repo in catalog["repositories"]:
            if repo.get("status") in {"preparing", "indexing", "cloning"} and not _job_running(str(repo.get("id"))):
                repo["status"] = "idle"
                repo["message"] = "Indexing was interrupted; refresh to resume"
        return catalog


def _save_catalog(catalog: Dict[str, Any]) -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    temp = CATALOG_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    temp.replace(CATALOG_PATH)


def _public_repository(repo: Dict[str, Any], *, active_id: Optional[str] = None) -> Dict[str, Any]:
    payload = {key: value for key, value in repo.items() if key != "checkout_path"}
    if active_id is not None:
        payload["active"] = repo.get("id") == active_id
    payload["building"] = _job_running(str(repo.get("id")))
    return payload


def list_repositories() -> Dict[str, Any]:
    catalog = _load_catalog()
    active_id = catalog.get("active_id")
    return {
        "active_id": active_id,
        "repositories": [_public_repository(repo, active_id=active_id) for repo in catalog["repositories"]],
    }


def get_repository(repository_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    catalog = _load_catalog()
    target = repository_id or catalog.get("active_id")
    return next((dict(repo) for repo in catalog["repositories"] if repo.get("id") == target), None)


def create_repository(*, name: str, source_type: str, location: str, branch: str = "", activate: bool = True) -> Dict[str, Any]:
    source_type = source_type.strip().lower()
    location = location.strip()
    if source_type not in {"local", "github"}:
        raise ValueError("source_type must be local or github")
    if source_type == "local":
        root = Path(location).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("Local repository directory does not exist")
        location = str(root)
    else:
        _validate_github_url(location)
    catalog = _load_catalog()
    identity = f"{source_type}:{location.lower()}:{branch.lower()}"
    existing = next((repo for repo in catalog["repositories"] if repo.get("identity") == identity), None)
    if existing:
        if activate:
            activate_repository(str(existing["id"]))
        return _public_repository(existing, active_id=str(existing["id"]) if activate else catalog.get("active_id"))
    repo_id = uuid.uuid4().hex[:12]
    now = time.time()
    repo = {
        "id": repo_id,
        "identity": identity,
        "name": (name or _repository_name(location)).strip(),
        "source_type": source_type,
        "location": location,
        "branch": branch.strip(),
        "checkout_path": str(CHECKOUT_ROOT / repo_id) if source_type == "github" else location,
        "status": "idle", "progress": 0.0, "message": "Ready to index", "error": "",
        "created_at": now, "updated_at": now, "summary": {},
    }
    with _CATALOG_LOCK:
        catalog = _load_catalog()
        catalog["repositories"].append(repo)
        if activate:
            catalog["active_id"] = repo_id
        _save_catalog(catalog)
    if activate:
        _cancel_other_jobs(repo_id)
        request_code_graph_refresh(repo_id, refresh_remote=False)
    return _public_repository(repo, active_id=catalog.get("active_id"))


def activate_repository(repository_id: str) -> Dict[str, Any]:
    with _CATALOG_LOCK:
        catalog = _load_catalog()
        repo = next((item for item in catalog["repositories"] if item.get("id") == repository_id), None)
        if not repo:
            raise KeyError("Repository not found")
        catalog["active_id"] = repository_id
        repo["updated_at"] = time.time()
        _save_catalog(catalog)
    _cancel_other_jobs(repository_id)
    if not _cache_path(repository_id).exists():
        request_code_graph_refresh(repository_id)
    return _public_repository(repo, active_id=repository_id)


def delete_repository(repository_id: str) -> None:
    catalog = _load_catalog()
    if len(catalog["repositories"]) <= 1:
        raise ValueError("At least one repository must remain")
    repo = next((item for item in catalog["repositories"] if item.get("id") == repository_id), None)
    if not repo:
        raise KeyError("Repository not found")
    with _JOBS_LOCK:
        job = _JOBS.get(repository_id)
        if job:
            job.cancel()
    with _CATALOG_LOCK:
        catalog = _load_catalog()
        catalog["repositories"] = [item for item in catalog["repositories"] if item.get("id") != repository_id]
        if catalog.get("active_id") == repository_id:
            catalog["active_id"] = catalog["repositories"][0]["id"]
        _save_catalog(catalog)
    _cache_path(repository_id).unlink(missing_ok=True)
    if repo.get("source_type") == "github":
        checkout = Path(str(repo.get("checkout_path") or ""))
        if checkout.is_dir() and CHECKOUT_ROOT in checkout.parents:
            shutil.rmtree(checkout, ignore_errors=True)


def get_code_graph(
    force_refresh: bool = False,
    repository_id: Optional[str] = None,
    parent_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    repo = get_repository(repository_id)
    if not repo:
        payload = _empty_payload()
        payload["error"] = "Repository not found"
        return payload
    if force_refresh:
        request_code_graph_refresh(str(repo["id"]), refresh_remote=True)
        repo = get_repository(str(repo["id"])) or repo
    cached = _load_graph_cache(str(repo["id"]))
    payload = _filter_graph_payload(cached, parent_ids) if cached else _empty_payload()
    public = _public_repository(repo)
    payload["repository"] = public
    payload["building"] = public["building"]
    payload["cached"] = bool(cached)
    payload["status"] = {
        "state": repo.get("status", "idle"), "progress": repo.get("progress", 0.0),
        "message": repo.get("message", ""), "error": repo.get("error", ""),
    }
    if not cached and not public["building"] and repo.get("status") != "error":
        request_code_graph_refresh(str(repo["id"]))
        payload["building"] = True
        payload["status"] = {"state": "preparing", "progress": 0.0, "message": "Starting index", "error": ""}
    return payload


def _filter_graph_payload(payload: Dict[str, Any], parent_ids: Optional[List[str]]) -> Dict[str, Any]:
    """Return the light structure or requested file-detail chunks from a full cache."""
    all_nodes = list(payload.get("nodes") or [])
    all_edges = list(payload.get("edges") or [])
    if parent_ids:
        requested = {value for value in parent_ids if value.startswith("file::")}
        allowed = set(requested)
        changed = True
        while changed:
            changed = False
            for edge in all_edges:
                if edge.get("kind") == "contains" and edge.get("source") in allowed and edge.get("target") not in allowed:
                    allowed.add(str(edge.get("target")))
                    changed = True
        selected_nodes = [node for node in all_nodes if node.get("id") in allowed]
        selected_edges = [
            edge for edge in all_edges
            if edge.get("source") in allowed and edge.get("target") in allowed
        ]
        return {
            "nodes": selected_nodes,
            "edges": selected_edges,
            "summary": payload.get("summary") or {},
            "generated_at": payload.get("generated_at"),
            "chunk": True,
            "parents": sorted(requested),
            "partial": bool(payload.get("partial")),
        }
    structural = {"repository", "module", "file"}
    selected_nodes = [node for node in all_nodes if node.get("kind") in structural]
    allowed = {str(node.get("id")) for node in selected_nodes}
    selected_edges = [
        edge for edge in all_edges
        if edge.get("source") in allowed and edge.get("target") in allowed
        and edge.get("kind") in {"contains", "imports"}
    ]
    result = dict(payload)
    result["nodes"] = selected_nodes
    result["edges"] = selected_edges
    result["structure_only"] = True
    result["detail_nodes_available"] = max(0, len(all_nodes) - len(selected_nodes))
    return result


def request_code_graph_refresh(repository_id: Optional[str] = None, *, refresh_remote: bool = False) -> bool:
    repo = get_repository(repository_id)
    if not repo:
        return False
    repo_id = str(repo["id"])
    with _JOBS_LOCK:
        current = _JOBS.get(repo_id)
        if current and current.thread.is_alive():
            if not refresh_remote:
                return False
            current.cancel()
        job = BuildJob(repo_id, refresh_remote=refresh_remote)
        _JOBS[repo_id] = job
        job.start()
    return True


def list_tracked_files(repository_id: Optional[str] = None) -> List[str]:
    repo = get_repository(repository_id)
    if not repo:
        return []
    cached = _load_graph_cache(str(repo["id"])) or {}
    return [str(item.get("path")) for item in cached.get("files", []) if item.get("path")]


def read_repository_source(repository_id: str, relative_path: str) -> Dict[str, Any]:
    """Read one indexed source file without permitting traversal outside its repo."""
    repo = get_repository(repository_id)
    if not repo:
        raise KeyError("Repository not found")
    root = Path(str(repo.get("checkout_path") or repo.get("location"))).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Repository has not been materialized yet")
    normalized = str(relative_path or "").replace("\\", "/").lstrip("/")
    path = (root / normalized).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Source path escapes the repository")
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise FileNotFoundError("Indexed source file was not found")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Source file is too large to display")
    content = path.read_text(encoding="utf-8", errors="replace")
    return {
        "repository_id": repository_id,
        "path": normalized,
        "language": path.suffix.lower().lstrip("."),
        "content": content,
        "lines": content.count("\n") + 1,
    }


def _cache_path(repository_id: str) -> Path:
    return CACHE_ROOT / f"{repository_id}.json"


def _load_graph_cache(repository_id: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(repository_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _save_graph_cache(repository_id: str, payload: Dict[str, Any]) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = _cache_path(repository_id)
    temp = path.with_suffix(".tmp")
    serializable = {key: value for key, value in payload.items() if key not in {"building"}}
    temp.write_text(json.dumps(serializable, separators=(",", ":")), encoding="utf-8")
    temp.replace(path)


def _empty_payload() -> Dict[str, Any]:
    return {
        "nodes": [], "edges": [], "file_links": [], "warnings": [], "errors": [],
        "summary": {"files": 0, "modules": 0, "classes": 0, "functions": 0, "relationships": 0},
        "generated_at": None, "files": [], "cached": False, "building": False,
    }


def _update_repository(repository_id: str, **changes: Any) -> None:
    with _CATALOG_LOCK:
        catalog = _load_catalog()
        repo = next((item for item in catalog["repositories"] if item.get("id") == repository_id), None)
        if not repo:
            return
        repo.update(changes)
        repo["updated_at"] = time.time()
        _save_catalog(catalog)


def _job_running(repository_id: str) -> bool:
    with _JOBS_LOCK:
        job = _JOBS.get(repository_id)
        return bool(job and job.thread.is_alive())


def _cancel_other_jobs(active_id: str) -> None:
    with _JOBS_LOCK:
        for repository_id, job in list(_JOBS.items()):
            if repository_id != active_id and job.thread.is_alive():
                job.cancel()


def _validate_github_url(value: str) -> None:
    if value.startswith("git@github.com:"):
        if not re.fullmatch(r"git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", value):
            raise ValueError("Invalid GitHub SSH repository URL")
        return
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname not in {"github.com", "www.github.com"}:
        raise ValueError("Use an https://github.com/owner/repository URL")
    if not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?/?", parsed.path):
        raise ValueError("GitHub URL must identify one repository")


def _repository_name(location: str) -> str:
    cleaned = location.rstrip("/\\")
    name = cleaned.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    return name[:-4] if name.endswith(".git") else name


def _materialize_repository(repo: Dict[str, Any], cancel_event: threading.Event, refresh_remote: bool) -> Path:
    if repo.get("source_type") == "local":
        root = Path(str(repo.get("location"))).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Repository directory is unavailable: {root}")
        return root
    repo_id = str(repo["id"])
    checkout = Path(str(repo.get("checkout_path") or CHECKOUT_ROOT / repo_id)).resolve()
    CHECKOUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    branch = str(repo.get("branch") or "").strip()
    if not (checkout / ".git").exists():
        if checkout.exists():
            if checkout != CHECKOUT_ROOT and CHECKOUT_ROOT.resolve() not in checkout.parents:
                raise ValueError("Refusing to replace a checkout outside the CodeGraph workspace")
            shutil.rmtree(checkout, ignore_errors=True)
        _update_repository(repo_id, status="cloning", progress=0.03, message="Cloning GitHub repository")
        command = ["git", "clone", "--depth", "1"]
        if branch:
            command += ["--branch", branch]
        command += [str(repo["location"]), str(checkout)]
        _run_cancellable(command, cwd=CHECKOUT_ROOT, env=env, cancel_event=cancel_event)
    elif refresh_remote:
        _update_repository(repo_id, status="cloning", progress=0.03, message="Updating GitHub repository")
        if branch:
            _run_cancellable(["git", "checkout", branch], cwd=checkout, env=env, cancel_event=cancel_event)
            _run_cancellable(["git", "pull", "--ff-only", "origin", branch], cwd=checkout, env=env, cancel_event=cancel_event)
        else:
            _run_cancellable(["git", "pull", "--ff-only"], cwd=checkout, env=env, cancel_event=cancel_event)
    return checkout


def _run_cancellable(command: List[str], *, cwd: Path, env: Dict[str, str], cancel_event: threading.Event) -> None:
    process = subprocess.Popen(
        command, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    deadline = time.time() + 600
    while process.poll() is None:
        if cancel_event.wait(0.15):
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
            raise CancelledBuild()
        if time.time() > deadline:
            process.kill()
            raise TimeoutError("Git operation exceeded 10 minutes")
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        detail = (stderr or stdout or "Git operation failed").strip()[-1000:]
        raise RuntimeError(detail)
