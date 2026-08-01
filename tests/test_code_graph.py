from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from services import code_graph


def _repository(root: Path, repository_id: str = "repo-one"):
    return {
        "id": repository_id,
        "name": root.name,
        "source_type": "local",
        "location": str(root),
        "status": "idle",
    }


def test_project_graph_maps_cross_language_symbols_inputs_outputs_and_calls(tmp_path: Path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "service.py").write_text(
        """
class Calculator:
    def double(self, value: int) -> int:
        return helper(value)

def helper(value: int) -> int:
    return value * 2
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "client.ts").write_text(
        """
export class ApiClient {}
export function loadUser(userId: string): Promise<User> {
  return request(userId)
}
""".strip(),
        encoding="utf-8",
    )

    payload = code_graph.ProjectGraph(
        tmp_path,
        cancel_event=threading.Event(),
        repository=_repository(tmp_path),
    ).build(progress=lambda *_: None)

    nodes = {node["label"]: node for node in payload["nodes"]}
    assert nodes["Calculator"]["kind"] == "class"
    assert nodes["double"]["meta"]["inputs"] == [{"name": "value", "type": "int"}]
    assert nodes["double"]["meta"]["outputs"] == ["int"]
    assert nodes["ApiClient"]["kind"] == "class"
    assert nodes["loadUser"]["kind"] == "function"
    assert any(
        edge["kind"] == "calls" and edge["source"].endswith("Calculator.double") and edge["target"].endswith("helper")
        for edge in payload["edges"]
    )
    assert payload["summary"]["files"] == 2
    assert payload["summary"]["modules"] == 2


def test_repository_catalog_persists_and_switches_without_scanning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    (project / "one.py").write_text("def one():\n    return 1\n", encoding="utf-8")
    (other / "two.rs").write_text("pub fn two() -> i32 { 2 }\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(code_graph, "PROJECT_ROOT", project)
    monkeypatch.setattr(code_graph, "RUNTIME_ROOT", runtime)
    monkeypatch.setattr(code_graph, "CATALOG_PATH", runtime / "repositories.json")
    monkeypatch.setattr(code_graph, "CACHE_ROOT", runtime / "repositories")
    monkeypatch.setattr(code_graph, "CHECKOUT_ROOT", runtime / "checkouts")
    monkeypatch.setattr(code_graph, "_JOBS", {})

    initial = code_graph.list_repositories()
    added = code_graph.create_repository(
        name="Other",
        source_type="local",
        location=str(other),
        activate=False,
    )
    activated = code_graph.activate_repository(added["id"])

    assert initial["repositories"][0]["location"] == str(project)
    assert activated["active"] is True
    assert code_graph.list_repositories()["active_id"] == added["id"]
    # A status response must not wait for the repository build to finish.
    started = time.perf_counter()
    payload = code_graph.get_code_graph(repository_id=added["id"])
    assert time.perf_counter() - started < 0.5
    assert payload["repository"]["id"] == added["id"]

    for job in list(code_graph._JOBS.values()):
        job.cancel()
        job.thread.join(timeout=3)


def test_source_reader_blocks_traversal_and_returns_full_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = tmp_path / "project"
    project.mkdir()
    source = "class Visible:\n    pass\n"
    (project / "visible.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(code_graph, "get_repository", lambda repository_id=None: _repository(project))

    payload = code_graph.read_repository_source("repo-one", "visible.py")
    assert payload["content"] == source
    assert payload["lines"] == 3
    with pytest.raises(ValueError, match="escapes"):
        code_graph.read_repository_source("repo-one", "../secret.py")


def test_cancelled_graph_build_stops_between_files(tmp_path: Path):
    for index in range(20):
        (tmp_path / f"file_{index}.py").write_text(f"def fn_{index}():\n    return {index}\n", encoding="utf-8")
    cancelled = threading.Event()
    cancelled.set()
    graph = code_graph.ProjectGraph(tmp_path, cancel_event=cancelled, repository=_repository(tmp_path))
    with pytest.raises(code_graph.CancelledBuild):
        graph.build(progress=lambda *_: None)


def test_structure_stream_starts_at_primary_entry_and_chunks_its_children(tmp_path: Path):
    (tmp_path / "main.py").write_text(
        "def main(value: str) -> int:\n    return len(value)\n\nif __name__ == '__main__':\n    main('go')\n",
        encoding="utf-8",
    )
    payload = code_graph.ProjectGraph(
        tmp_path,
        cancel_event=threading.Event(),
        repository=_repository(tmp_path),
    ).build(progress=lambda *_: None)
    structure = code_graph._filter_graph_payload(payload, None)
    entry = next(node for node in structure["nodes"] if node["id"] == "file::main.py")
    chunk = code_graph._filter_graph_payload(payload, ["file::main.py"])

    assert entry["meta"]["entry_point"] is True
    assert entry["meta"]["entry_priority"] == 100
    assert {node["kind"] for node in structure["nodes"]} <= {"repository", "module", "file"}
    assert any(node["label"] == "main" and node["kind"] == "function" for node in chunk["nodes"])
