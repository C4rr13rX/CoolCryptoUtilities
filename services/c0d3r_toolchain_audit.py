"""Independent, repeatable audit of the C0d3rV2 delivery toolchain."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


def _shape(value: Any) -> dict[str, Any]:
    return {"type": type(value).__name__, "keys": sorted(value) if isinstance(value, dict) else [],
            "nonempty": bool(value)}


def _valid_result(name: str, result: Any, root: Path) -> tuple[bool, str]:
    if not isinstance(result, dict):
        return False, "result is not an object"
    if result.get("error"):
        return False, str(result["error"])
    validators = {
        "file_read": lambda r: r.get("content") == "alpha\nbeta\n",
        "file_write": lambda r: Path(str(r.get("path", ""))).is_file(),
        "directory_ensure": lambda r: all((root / p).is_dir() for p in ("created/a", "created/b")),
        "workspace_scaffold": lambda r: (root / "audit-framework" / "hello.txt").is_file(),
        "executor": lambda r: r.get("return_code") == 0 and "executor-ok" in str(r.get("stdout", "")),
        "product_artifact_materializer": lambda r: bool(r.get("files")) and all(Path(p).is_file() for p in r["files"]),
        "project_work_mapper": lambda r: bool(r.get("tasks")),
        "memory_search": lambda r: bool(r.get("results")) and "audit-memory-token-7319" in json.dumps(r).lower(),
        "equation_matrix": lambda r: bool(r.get("hits")) and any("kinetic" in json.dumps(hit).lower() for hit in r["hits"]),
        "file_locate": lambda r: any(Path(str(c.get("path", c) if isinstance(c, dict) else c)).name == "fixture.txt" for c in r.get("candidates", [])),
    }
    validator = validators.get(name)
    if validator is None:
        return bool(result), "empty result" if not result else ""
    try:
        valid = bool(validator(result))
    except Exception as exc:
        return False, f"validator raised {exc!r}"
    return valid, "functional postcondition failed" if not valid else ""


def run_audit(root: Path, *, include_expensive: bool = True) -> dict[str, Any]:
    from tools.c0d3rV2.delivery_runner import _build_delivery_flow
    root = root.resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    flow = _build_delivery_flow("independent-tool-audit", root, backend="freeloader")
    registry = flow.tools
    (root / "fixture.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    if flow.lt_memory:
        flow.lt_memory.append(
            "audit-memory-token-7319", "memory retrieval is operational",
            workdir=str(root), session_id="independent-tool-audit",
        )
    fixtures: dict[str, dict[str, Any]] = {
        "file_read": {"path": "fixture.txt"},
        "file_write": {"path": "written/result.txt", "content": "verified write\n"},
        "directory_ensure": {"paths": ["created/a", "created/b"]},
        "workspace_scaffold": {"root_readme": "# Audit", "frameworks": [{"name": "audit-framework", "files": {"hello.txt": "hello"}}]},
        "environment_bootstrap": {"preset": "python_fastapi", "timeout_s": 90},
        "executor": {"command": "python -c \"print('executor-ok')\""},
        "web_search": {"query": "site:openstax.org free fall gravitational acceleration equation"},
        "scientific_method": {"question": "In the Monty Hall problem, should the player switch?", "domain": "probability", "max_sources": 2},
        "branddozer_product_cycle": {"root_path": str(root / "branddozer-cycle"), "cycles": 1},
        "product_artifact_materializer": {"kind": "document", "spec": {"name": "Audit Document", "description": "Functional materializer audit", "sections": ["Purpose", "Result"]}},
        "project_work_mapper": {"action": "map", "request": "Create a small verified text artifact in the audit workspace"},
        "class_refinement_benchmark": {"count": 1, "attempts": 1},
        "memory_search": {"query": "audit-memory-token-7319"},
        "equation_matrix": {"action": "search", "query": "kinetic energy", "limit": 5},
        "file_locate": {"query": "fixture.txt", "cwd": str(root), "project_root": str(root), "detailed": True},
    }
    expensive = {"environment_bootstrap", "web_search", "scientific_method", "branddozer_product_cycle", "class_refinement_benchmark"}
    rows = []
    descriptions = {item["name"]: item for item in registry.tool_descriptions()}
    for name in registry.tool_names():
        row: dict[str, Any] = {"tool": name, "registered": name in descriptions,
                               "schema_nonempty": bool((descriptions.get(name) or {}).get("params")),
                               "status": "pending"}
        if name in expensive and not include_expensive:
            row["status"] = "deferred_expensive"
            rows.append(row); continue
        started = time.perf_counter()
        try:
            result = registry.dispatch(name, fixtures[name])
            valid, reason = _valid_result(name, result, root)
            row.update({"status": "passed" if valid else "failed",
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "result_shape": _shape(result), "result": result})
            if reason:
                row["error"] = reason
            if name == "branddozer_product_cycle":
                # The tool persists a resumable BrandDozer project. An audit
                # must not leave that project enabled for the global keeper.
                try:
                    from services.branddozer_runner import branddozer_manager
                    from services.branddozer_state import delete_project, list_projects
                    audit_root = str((root / "branddozer-cycle").resolve()).lower()
                    for project in list_projects():
                        if str(project.get("root_path", "")).lower() == audit_root:
                            branddozer_manager.stop(project["id"])
                            delete_project(project["id"])
                except Exception as cleanup_exc:
                    row["cleanup_error"] = repr(cleanup_exc)
        except Exception as exc:
            row.update({"status": "exception", "elapsed_seconds": round(time.perf_counter() - started, 3), "error": repr(exc)})
        rows.append(row)
    # Stateful mapper chain: map -> next -> complete -> status.
    chain = []
    try:
        nxt = registry.dispatch("project_work_mapper", {"action": "next"}); chain.append({"action": "next", "result": nxt})
        task_id = str((nxt.get("task") or {}).get("id") or nxt.get("task_id") or "")
        if task_id:
            chain.append({"action": "complete", "result": registry.dispatch("project_work_mapper", {"action": "complete", "task_id": task_id, "evidence": {"audit": True}})})
        chain.append({"action": "status", "result": registry.dispatch("project_work_mapper", {"action": "status"})})
    except Exception as exc:
        chain.append({"action": "exception", "error": repr(exc)})
    expected_handoffs = {
        "file_locate": ["file_read"],
        "file_read": ["file_write"],
        "file_write": ["executor"],
        "directory_ensure": ["workspace_scaffold"],
        "workspace_scaffold": ["environment_bootstrap"],
        "web_search": ["scientific_method", "equation_matrix"],
        "scientific_method": ["equation_matrix"],
        "project_work_mapper": ["file_locate", "file_read", "file_write", "executor"],
        "branddozer_product_cycle": ["product_artifact_materializer", "project_work_mapper"],
        "memory_search": ["file_locate"],
    }
    handoffs = [
        {"source": source, "target": target, "status": "passed" if target in descriptions else "failed"}
        for source, targets in expected_handoffs.items() for target in targets
    ]
    report = {"schema_version": 1, "workspace": str(root), "include_expensive": include_expensive,
              "tool_count": len(rows), "passed": sum(row["status"] == "passed" for row in rows),
              "failed": sum(row["status"] in {"failed", "exception"} for row in rows),
              "rows": rows, "branch_chains": {"project_work_mapper": chain},
              "declared_handoffs": handoffs}
    target = Path(__file__).resolve().parents[1] / "runtime" / "benchmarks" / "c0d3r_toolchain_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    print(json.dumps(run_audit(project / "runtime" / "tool_audit_sandbox"), indent=2, default=str))
