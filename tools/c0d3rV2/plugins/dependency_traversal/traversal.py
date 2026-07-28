from __future__ import annotations

import ast
import hashlib
import json
import posixpath
import re
import time
from collections import deque
from pathlib import Path
from typing import Any


_IGNORED = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    "dist", "build", "coverage", ".idea", ".vscode",
}
_SOURCE_SUFFIXES = {
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".rs", ".go", ".java", ".kt", ".kts", ".c", ".cc", ".cpp", ".h",
    ".hpp", ".php", ".pl", ".pm", ".rb", ".cs", ".vue", ".svelte",
    ".json", ".toml", ".yaml", ".yml",
}


class DependencyTraversal:
    """Build and traverse a deterministic, workspace-bounded dependency graph."""

    VERSION = 1

    def __init__(self, workdir: str | Path) -> None:
        self.workdir = Path(workdir).resolve()
        self.state_path = self.workdir / ".c0d3r" / "dependency-traversal.json"

    def scan(self, *, max_files: int = 4000, force: bool = False) -> dict[str, Any]:
        fingerprint = self._fingerprint(max_files=max_files)
        current = self._load()
        if not force and current.get("fingerprint") == fingerprint:
            return current
        files = self._files(max_files=max_files)
        known = {item.relative_to(self.workdir).as_posix(): item for item in files}
        basename: dict[str, list[str]] = {}
        for relative in known:
            basename.setdefault(Path(relative).name.lower(), []).append(relative)
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, str]] = []
        for relative, path in known.items():
            text = self._read(path)
            imports, symbols = self._extract(path, text)
            nodes[relative] = {
                "path": relative,
                "kind": self._kind(relative),
                "symbols": symbols[:120],
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
            }
            for raw in imports:
                target = self._resolve(relative, raw, known, basename)
                if target and target != relative:
                    edges.append({"source": relative, "target": target, "kind": "imports"})
        state = {
            "version": self.VERSION,
            "root": str(self.workdir),
            "fingerprint": fingerprint,
            "created_at": time.time(),
            "nodes": nodes,
            "edges": self._dedupe_edges(edges),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
        return state

    def traverse(
        self, query: str, *, paths: list[str] | None = None, depth: int = 3,
        max_nodes: int = 64, force_scan: bool = False,
    ) -> dict[str, Any]:
        state = self.scan(force=force_scan)
        nodes = state.get("nodes") or {}
        anchors = self._anchors(query, paths or [], nodes)
        outgoing: dict[str, list[str]] = {}
        incoming: dict[str, list[str]] = {}
        for edge in state.get("edges") or []:
            source, target = edge["source"], edge["target"]
            outgoing.setdefault(source, []).append(target)
            incoming.setdefault(target, []).append(source)
        upstream = self._walk(anchors, outgoing, depth, max_nodes)
        downstream = self._walk(anchors, incoming, depth, max_nodes)
        selected = list(dict.fromkeys(anchors + upstream + downstream))[:max_nodes]
        tests = [path for path in selected if (nodes.get(path) or {}).get("kind") == "test"]
        configs = [path for path in selected if (nodes.get(path) or {}).get("kind") == "config"]
        return {
            "schema": "c0d3r.dependency-traversal/v1",
            "query": query,
            "anchors": anchors,
            "upstream_dependencies": upstream,
            "downstream_consumers": downstream,
            "regression_tests": tests,
            "configuration": configs,
            "nodes": [nodes[path] for path in selected if path in nodes],
            "edges": [
                edge for edge in state.get("edges") or []
                if edge["source"] in selected and edge["target"] in selected
            ],
            "bounded": {"depth": min(8, max(0, depth)), "max_nodes": max_nodes},
        }

    def injection_packet(
        self, query: str, *, paths: list[str] | None = None, depth: int = 3,
        max_nodes: int = 48, memory: list[Any] | None = None,
        hazy_hints: list[str] | None = None, failures: list[Any] | None = None,
    ) -> dict[str, Any]:
        traversal = self.traverse(query, paths=paths, depth=depth, max_nodes=max_nodes)
        evidence_paths = list(dict.fromkeys([
            *traversal["anchors"], *traversal["upstream_dependencies"],
            *traversal["downstream_consumers"], *traversal["regression_tests"],
            *traversal["configuration"],
        ]))
        evidence_files = []
        remaining = 12000
        for relative in evidence_paths[:20]:
            if remaining <= 0:
                break
            try:
                path = (self.workdir / relative).resolve()
                path.relative_to(self.workdir)
                content = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue
            excerpt = content[: min(3000, remaining)]
            evidence_files.append({
                "path": relative,
                "sha256": hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest(),
                "excerpt": excerpt,
                "truncated": len(excerpt) < len(content),
            })
            remaining -= len(excerpt)
        downstream = traversal["downstream_consumers"]
        composition_roots = [
            path for path in downstream
            if Path(path).stem.lower() in {"main", "index", "app", "bootstrap", "entrypoint"}
            and path not in traversal["regression_tests"]
        ]
        ordinary_consumers = [
            path for path in downstream
            if path not in composition_roots and path not in traversal["regression_tests"]
        ]
        regression_route = [
            *({"phase": "definition_or_contract", "path": path} for path in traversal["upstream_dependencies"]),
            *({"phase": "change_surface", "path": path} for path in traversal["anchors"]),
            *(
                {
                    "phase": "regression_test" if path in traversal["regression_tests"]
                    else "composition_root" if path in composition_roots else "consumer",
                    "path": path,
                }
                for path in downstream
            ),
            *({"phase": "configuration", "path": path} for path in traversal["configuration"]),
        ]
        return {
            "schema": "c0d3r.regression-injection/v1",
            "scope": {"root": str(self.workdir), "escape": "forbidden"},
            "request": query,
            "change_surface": traversal["anchors"],
            "definitions_and_dependencies": traversal["upstream_dependencies"],
            "consumers_and_regression_surface": traversal["downstream_consumers"],
            "tests": traversal["regression_tests"],
            "configuration": traversal["configuration"],
            "edges": traversal["edges"],
            "evidence_files": evidence_files,
            "regression_route": regression_route,
            "memory": (memory or [])[:10],
            "hazy_hash_candidates": self._scoped_hints(hazy_hints or [])[:20],
            "validator_failures": (failures or [])[:12],
            "instruction": (
                "Follow regression_route in phase order. Inspect the hashed evidence excerpt for the "
                "current phase, make one cohesive mutation, validate, and advance only from fresh "
                "validator evidence. Preserve upstream contracts and downstream consumers. Do not "
                "inspect unrelated package/config files when the route already contains the failing "
                "definition, composition root, consumer, and test. Expand one graph level only when "
                "fresh evidence names an unresolved caller."
            ),
        }

    def status(self) -> dict[str, Any]:
        state = self._load()
        return {
            "indexed": bool(state), "path": str(self.state_path),
            "files": len(state.get("nodes") or {}), "edges": len(state.get("edges") or []),
            "fingerprint": state.get("fingerprint", ""),
        }

    def _files(self, *, max_files: int) -> list[Path]:
        result: list[Path] = []
        if not self.workdir.exists():
            return result
        for path in self.workdir.rglob("*"):
            if len(result) >= max(1, min(20_000, max_files)):
                break
            if not path.is_file() or any(part.lower() in _IGNORED for part in path.parts):
                continue
            if path.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            try:
                if path.stat().st_size <= 1_000_000:
                    result.append(path)
            except OSError:
                continue
        return sorted(result, key=lambda item: item.relative_to(self.workdir).as_posix())

    def _fingerprint(self, *, max_files: int) -> str:
        facts = []
        for path in self._files(max_files=max_files):
            stat = path.stat()
            facts.append((path.relative_to(self.workdir).as_posix(), stat.st_size, stat.st_mtime_ns))
        return hashlib.sha256(json.dumps(facts, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _extract(path: Path, text: str) -> tuple[list[str], list[str]]:
        imports: list[str] = []
        symbols: list[str] = []
        if path.suffix.lower() in {".py", ".pyi"}:
            try:
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        imports.append("." * node.level + (node.module or ""))
                    elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(node.name)
                return list(dict.fromkeys(imports)), list(dict.fromkeys(symbols))
            except SyntaxError:
                pass
        patterns = (
            r"(?:import|export)\s+(?:type\s+)?(?:[^'\"]+?\s+from\s+)?['\"]([^'\"]+)['\"]",
            r"require\(\s*['\"]([^'\"]+)['\"]\s*\)",
            r"#include\s*[<\"]([^>\"]+)[>\"]",
            r"\b(?:use|mod)\s+([A-Za-z_][\w:]*)",
            r"\bimport\s+([A-Za-z_][\w.]*)\s*;",
            r"\b(?:include|require)(?:_once)?\s*\(?\s*['\"]([^'\"]+)['\"]",
        )
        for pattern in patterns:
            imports.extend(re.findall(pattern, text))
        symbols.extend(re.findall(
            r"\b(?:export\s+)?(?:class|interface|type|enum|function|struct|trait|fn)\s+([A-Za-z_]\w*)",
            text,
        ))
        return list(dict.fromkeys(imports)), list(dict.fromkeys(symbols))

    @staticmethod
    def _resolve(source: str, raw: str, known: dict[str, Path], basename: dict[str, list[str]]) -> str:
        raw = raw.strip().replace("\\", "/").replace("::", "/")
        base = Path(source).parent
        if raw.startswith(("./", "../")):
            candidate = posixpath.normpath(f"{base.as_posix()}/{raw}")
        elif raw.startswith("."):
            level = len(raw) - len(raw.lstrip("."))
            parent = base.as_posix()
            for _ in range(max(0, level - 1)):
                parent = posixpath.dirname(parent)
            candidate = posixpath.normpath(
                f"{parent}/{raw[level:].replace('.', '/')}"
            )
        elif "/" in raw:
            candidate = raw
        else:
            candidate = raw.replace(".", "/")
        variants = [candidate]
        for suffix in _SOURCE_SUFFIXES:
            variants.extend([candidate + suffix, candidate + "/index" + suffix])
        normalized_known = {path.lower(): path for path in known}
        for variant in variants:
            value = str(Path(variant)).replace("\\", "/").lower()
            if value in normalized_known:
                return normalized_known[value]
        name = Path(raw).name.lower()
        for suffix in _SOURCE_SUFFIXES:
            matches = basename.get(name + suffix, [])
            if len(matches) == 1:
                return matches[0]
        return ""

    @staticmethod
    def _kind(path: str) -> str:
        lowered = path.lower()
        if "/test" in f"/{lowered}" or re.search(r"(?:^|/)[^/]+\.(?:test|spec)\.", lowered):
            return "test"
        if Path(lowered).name in {"package.json", "tsconfig.json", "pyproject.toml", "cargo.toml", "go.mod"} or Path(lowered).suffix in {".yaml", ".yml", ".toml"}:
            return "config"
        return "source"

    @staticmethod
    def _anchors(query: str, requested: list[str], nodes: dict[str, dict]) -> list[str]:
        anchors: list[str] = []
        for raw in requested:
            normalized = str(raw).replace("\\", "/").lower()
            matches = [path for path in nodes if path.lower() == normalized or path.lower().endswith("/" + normalized)]
            anchors.extend(matches[:1])
        tokens = {token for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", query.lower()) if token not in {"the", "and", "for", "with", "from", "this", "that"}}
        scored = []
        for path, node in nodes.items():
            haystack = " ".join([path.lower(), *(str(item).lower() for item in node.get("symbols") or [])])
            score = sum(3 if token in Path(path).name.lower() else 1 for token in tokens if token in haystack)
            if score:
                scored.append((score, path))
        scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
        anchors.extend(path for _score, path in scored[:12])
        return list(dict.fromkeys(anchors))[:12]

    @staticmethod
    def _walk(starts: list[str], adjacency: dict[str, list[str]], depth: int, limit: int) -> list[str]:
        seen = set(starts)
        found: list[str] = []
        queue = deque((item, 0) for item in starts)
        while queue and len(found) < limit:
            node, level = queue.popleft()
            if level >= min(8, max(0, depth)):
                continue
            for neighbor in sorted(adjacency.get(node, [])):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                found.append(neighbor)
                queue.append((neighbor, level + 1))
                if len(found) >= limit:
                    break
        return found

    def _scoped_hints(self, hints: list[str]) -> list[str]:
        scoped = []
        for raw in hints:
            try:
                path = Path(raw).resolve()
                path.relative_to(self.workdir)
                scoped.append(str(path))
            except (OSError, ValueError):
                continue
        return list(dict.fromkeys(scoped))

    @staticmethod
    def _dedupe_edges(edges: list[dict[str, str]]) -> list[dict[str, str]]:
        seen = set()
        result = []
        for edge in edges:
            key = (edge["source"], edge["target"], edge["kind"])
            if key not in seen:
                seen.add(key)
                result.append(edge)
        return result

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
