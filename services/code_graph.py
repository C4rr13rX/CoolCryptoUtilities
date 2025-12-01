from __future__ import annotations

import ast
import hashlib
import json
import logging
import multiprocessing
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import OperationalError, ProgrammingError

try:
    from core.models import CodeGraphCache
except Exception:  # pragma: no cover - model unavailable before migrations
    CodeGraphCache = None

logger = logging.getLogger(__name__)

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
TARGET_DIRS = {
    "web",
    "services",
    "monitoring_guardian",
    "trading",
    "tools",
    "apps",
    "orchestration",
    "walletpanel",
    "guardianpanel",
    "opsconsole",
}
TARGET_FILES = {
    "main.py",
    "production.py",
    "router_wallet.py",
    "balances.py",
    "wallet_cli.py",
    "db.py",
    "services.py",
}
CACHE_DIR = Path("runtime/code_graph")
CACHE_PATH = CACHE_DIR / "graph_cache.json"
CACHE_TTL = int(os.getenv("CODE_GRAPH_CACHE_TTL", "0"))
CACHE_KEY = "codegraph:default"

_BUILD_LOCK = threading.Lock()
_BUILD_PROCESS: Optional[multiprocessing.Process] = None


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
    def __init__(self, rel_path: str, module_name: str, file_node: GraphNode) -> None:
        self.rel_path = rel_path
        self.module_name = module_name
        self.module_parts = module_name.split(".") if module_name else []
        self.file_node = file_node
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []
        self.calls: List[Tuple[str, str]] = []
        self.scope: List[Tuple[str, str, str]] = []
        self.name_index: Dict[str, List[str]] = defaultdict(list)
        self.imports: List[str] = []
        self.definitions: List[str] = []

    # ------------------------------------------------------------------ helpers
    def _register(self, node: GraphNode, parent_id: Optional[str] = None) -> None:
        self.nodes.append(node)
        parent = parent_id or self.file_node.id
        self.edges.append(GraphEdge(parent, node.id, "contains"))
        self.name_index[node.label].append(node.id)
        if node.kind in {"class", "function", "method"}:
            self.definitions.append(node.label)

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

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = getattr(alias, "name", "")
            if name:
                self.imports.append(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = self._resolve_import_from_module(node.module, node.level or 0)
        if not module and not node.names:
            return
        if module:
            self.imports.append(module)
        for alias in node.names:
            if alias.name == "*":
                continue
            target = f"{module}.{alias.name}" if module else alias.name
            if target:
                self.imports.append(target)
        self.generic_visit(node)

    def _resolve_import_from_module(self, module: Optional[str], level: int) -> str:
        if not level:
            return module or ""
        base = self.module_parts[:-1] if self.module_parts else []
        if level > 0:
            base = base[: max(0, len(base) - level)]
        if module:
            base += module.split(".")
        return ".".join(part for part in base if part)


class ProjectGraph:
    def __init__(self, root: Path, file_list: Optional[List[str]] = None) -> None:
        self.root = root
        self.selected_files = file_list
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.call_records: List[Tuple[str, str]] = []
        self.name_index: Dict[str, List[str]] = defaultdict(list)
        self.reference_counts: Dict[str, int] = defaultdict(int)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.file_imports: Dict[str, set[str]] = defaultdict(set)
        self.file_nodes_by_path: Dict[str, str] = {}
        self.module_index: Dict[str, str] = {}
        self.file_links: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"imports": 0, "calls": 0})

    # ------------------------------------------------------------------ public
    def build(self) -> Dict[str, object]:
        files = list(self._iter_source_files())
        for file_path in files:
            self._process_file(file_path)
        self._resolve_references()
        self._mark_unused_nodes()
        self._build_file_links()
        summary = {
            "files": len([node for node in self.nodes.values() if node.kind == "file"]),
            "classes": len([node for node in self.nodes.values() if node.kind == "class"]),
            "functions": len([node for node in self.nodes.values() if node.kind in {"function", "method"}]),
        }
        return {
            "nodes": [node.as_dict() for node in self.nodes.values()],
            "edges": [edge.as_dict() for edge in self.edges],
            "warnings": self.warnings,
            "errors": self.errors,
            "summary": summary,
            "generated_at": time.time(),
            "file_links": self._serialize_file_links(),
        }

    # ------------------------------------------------------------------ helpers
    def _iter_source_files(self) -> Iterable[Path]:
        if self.selected_files is not None:
            for rel_path in self.selected_files:
                path = (self.root / rel_path).resolve()
                if path.exists():
                    yield path
            return
        for path in self.root.rglob("*.py"):
            rel = path.relative_to(self.root)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            top = rel.parts[0] if rel.parts else ""
            if top not in TARGET_DIRS and rel.as_posix() not in TARGET_FILES:
                continue
            yield path

    def _process_file(self, path: Path) -> None:
        rel_path = str(path.relative_to(self.root))
        file_node = GraphNode(id=f"file::{rel_path}", label=rel_path, kind="file", file=rel_path)
        self.nodes[file_node.id] = file_node
        self.file_nodes_by_path[rel_path] = file_node.id
        module_name = self._module_name_from_path(rel_path)
        if module_name:
            file_node.meta["module"] = module_name
            self.module_index[module_name] = file_node.id
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel_path)
        except Exception as exc:
            file_node.status = "broken"
            file_node.meta["error"] = str(exc)
            self.errors.append(file_node.id)
            return

        analyzer = FileAnalyzer(rel_path, module_name, file_node)
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
        if analyzer.imports:
            file_node.meta["imports"] = sorted(set(analyzer.imports))
            self.file_imports[file_node.id].update(analyzer.imports)
        if analyzer.definitions:
            file_node.meta["definitions"] = sorted(set(analyzer.definitions))

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

    def _build_file_links(self) -> None:
        for file_id, imports in self.file_imports.items():
            for module in imports:
                target = self.module_index.get(module)
                if target:
                    self._record_file_link(file_id, target, "imports")
        for edge in self.edges:
            if edge.kind != "calls":
                continue
            source_node = self.nodes.get(edge.source)
            target_node = self.nodes.get(edge.target)
            if not source_node or not target_node:
                continue
            source_file = self.file_nodes_by_path.get(source_node.file)
            target_file = self.file_nodes_by_path.get(target_node.file)
            if not source_file or not target_file:
                continue
            self._record_file_link(source_file, target_file, "calls")

    def _record_file_link(self, source_file: str, target_file: str, reason: str) -> None:
        if source_file == target_file:
            return
        bucket = self.file_links[(source_file, target_file)]
        bucket[reason] = bucket.get(reason, 0) + 1

    def _serialize_file_links(self) -> List[Dict[str, object]]:
        payload: List[Dict[str, object]] = []
        for (source, target), counts in self.file_links.items():
            weight = counts.get("imports", 0) + counts.get("calls", 0)
            if weight <= 0:
                continue
            payload.append(
                {
                    "source": source,
                    "target": target,
                    "imports": counts.get("imports", 0),
                    "calls": counts.get("calls", 0),
                    "weight": weight,
                }
            )
        return payload

    @staticmethod
    def _module_name_from_path(rel_path: str) -> str:
        path = Path(rel_path)
        parts = list(path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)


def build_code_graph(file_list: Optional[List[str]] = None) -> Dict[str, object]:
    graph = ProjectGraph(PROJECT_ROOT, file_list=file_list)
    return graph.build()


def get_code_graph(force_refresh: bool = False) -> Dict[str, object]:
    file_paths = _list_source_files()
    cached = _load_cached_graph()
    snapshot: Optional[List[Dict[str, object]]] = None

    if cached and not force_refresh:
        payload = dict(cached)
        payload["cached"] = True
        payload["building"] = _is_building()
        return payload

    if cached:
        snapshot = _collect_file_snapshot(file_paths)
        if not force_refresh and not _files_changed(cached.get("files") or [], snapshot):
            payload = dict(cached)
            payload["cached"] = True
            payload["building"] = _is_building()
            return payload
    elif force_refresh:
        snapshot = _collect_file_snapshot(file_paths)

    _kickoff_background_build(file_paths, snapshot)

    placeholder = dict(cached) if cached else _empty_payload()
    if not placeholder.get("nodes"):
        placeholder["nodes"] = []
        placeholder["edges"] = []
        placeholder["warnings"] = []
        placeholder["errors"] = []
        placeholder["summary"] = placeholder.get("summary") or {"files": 0, "classes": 0, "functions": 0}

    if placeholder.get("files"):
        pass
    elif snapshot:
        placeholder["files"] = snapshot
    else:
        placeholder["files"] = [{"path": path} for path in file_paths]

    placeholder["cached"] = bool(cached)
    placeholder["building"] = True
    return placeholder


def request_code_graph_refresh() -> None:
    file_paths = _list_source_files()
    _kickoff_background_build(file_paths)


def list_tracked_files() -> List[str]:
    cached = _load_cached_graph()
    if cached and cached.get("files"):
        return [entry["path"] for entry in cached["files"]]  # type: ignore[index]
    snapshot = _collect_file_snapshot(_list_source_files())
    return [entry["path"] for entry in snapshot]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _list_source_files() -> List[str]:
    files: List[str] = []
    for path in PROJECT_ROOT.rglob("*.py"):
        rel = path.relative_to(PROJECT_ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        top = rel.parts[0] if rel.parts else ""
        rel_posix = rel.as_posix()
        if top not in TARGET_DIRS and rel_posix not in TARGET_FILES:
            continue
        files.append(rel_posix)
    files.sort()
    return files


def _collect_file_snapshot(file_paths: List[str]) -> List[Dict[str, object]]:
    snapshot: List[Dict[str, object]] = []
    for rel_path in file_paths:
        full_path = (PROJECT_ROOT / rel_path).resolve()
        try:
            data = full_path.read_bytes()
        except FileNotFoundError:
            continue
        stats = full_path.stat()
        snapshot.append(
            {
                "path": rel_path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": stats.st_size,
                "mtime": stats.st_mtime,
            }
        )
    return snapshot


def _files_changed(old: List[Dict[str, object]], new: List[Dict[str, object]]) -> bool:
    if len(old) != len(new):
        return True
    old_map = {entry["path"]: entry for entry in old}
    for entry in new:
        prev = old_map.get(entry["path"])
        if not prev:
            return True
        if prev.get("sha256") != entry.get("sha256"):
            return True
    return False


def _empty_payload() -> Dict[str, object]:
    return {
        "nodes": [],
        "edges": [],
        "warnings": [],
        "errors": [],
        "summary": {"files": 0, "classes": 0, "functions": 0},
        "generated_at": None,
        "files": [],
        "cached": False,
        "building": False,
    }


def _kickoff_background_build(file_paths: List[str], snapshot: Optional[List[Dict[str, object]]] = None) -> None:
    global _BUILD_PROCESS
    with _BUILD_LOCK:
        if _is_building():
            return
        process = multiprocessing.Process(
            target=_worker_process,
            args=(file_paths, snapshot),
            name="code-graph-build",
            daemon=True,
        )
        process.start()
        _BUILD_PROCESS = process


def _load_cached_graph() -> Optional[Dict[str, object]]:
    entry = _load_cached_graph_from_db()
    if entry:
        return entry
    entry = _load_cached_graph_from_disk()
    return entry


def _save_cached_graph(payload: Dict[str, object]) -> None:
    payload["generated_at"] = payload.get("generated_at") or time.time()
    _save_cached_graph_to_db(payload)
    _save_cached_graph_to_disk(payload)


def _load_cached_graph_from_db() -> Optional[Dict[str, object]]:
    if not _db_cache_enabled():
        return None
    try:
        cache_entry = CodeGraphCache.objects.filter(cache_key=CACHE_KEY).first()
    except (OperationalError, ProgrammingError, ImproperlyConfigured):
        return None
    if not cache_entry:
        return None
    graph = dict(cache_entry.graph or {})
    graph["files"] = cache_entry.files or []
    if not graph["files"]:
        graph["files"] = _safe_snapshot()
        try:
            cache_entry.files = graph["files"]
            cache_entry.save(update_fields=["files"])
        except Exception:
            pass
    graph["cached"] = True
    graph["building"] = False
    return graph


def _save_cached_graph_to_db(payload: Dict[str, object]) -> None:
    if not _db_cache_enabled():
        return
    try:
        CodeGraphCache.objects.update_or_create(
            cache_key=CACHE_KEY,
            defaults={
                "graph": {
                    key: value
                    for key, value in payload.items()
                    if key not in {"files", "cached", "building"}
                },
                "files": payload.get("files", []),
            },
        )
    except (OperationalError, ProgrammingError, ImproperlyConfigured):
        logger.debug("Skipping DB cache save (database unavailable)", exc_info=True)


def _load_cached_graph_from_disk() -> Optional[Dict[str, object]]:
    if not CACHE_PATH.exists():
        return None
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    ts = payload.get("generated_at")
    if not isinstance(ts, (int, float)):
        return None
    if CACHE_TTL > 0 and (time.time() - ts) > CACHE_TTL:
        return None
    payload["cached"] = True
    files = payload.get("files")
    if not files:
        payload["files"] = _safe_snapshot()
        try:
            _save_cached_graph_to_disk(payload)
        except Exception:
            pass
    else:
        payload["files"] = files
    payload["building"] = False
    return payload


def _save_cached_graph_to_disk(payload: Dict[str, object]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        serializable = {
            key: value for key, value in payload.items() if key not in {"cached", "building"}
        }
        CACHE_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    except Exception:
        logger.debug("Unable to persist graph cache to disk", exc_info=True)


def _worker_process(file_paths: List[str], snapshot: Optional[List[Dict[str, object]]]) -> None:
    try:
        try:
            import django  # type: ignore

            if hasattr(django, "setup"):
                django.setup()
        except Exception:
            pass
        paths = file_paths
        meta = snapshot or _collect_file_snapshot(paths)
        payload = build_code_graph(paths)
        payload["files"] = meta
        payload["cached"] = False
        payload["building"] = False
        _save_cached_graph(payload)
    except Exception:
        logger.exception("Code graph build failed")


def _is_building() -> bool:
    global _BUILD_PROCESS
    process = _BUILD_PROCESS
    if not process:
        return False
    if process.is_alive():
        return True
    process.join(timeout=0.1)
    _BUILD_PROCESS = None
    return False


def _db_cache_enabled() -> bool:
    if CodeGraphCache is None:
        return False
    try:
        engine = settings.DATABASES["default"]["ENGINE"]
    except Exception:
        return False
    return "postgresql" in engine


def _safe_snapshot() -> List[Dict[str, object]]:
    try:
        return _collect_file_snapshot(_list_source_files())
    except Exception:
        logger.debug("Unable to compute file snapshot for cache", exc_info=True)
        return []
