from pathlib import Path

from tools.c0d3rV2.plugins.dependency_traversal import DependencyTraversal
from tools.c0d3rV2.tool_registry import DependencyTraversalTool
from services.c0d3r_jeopardy_ctf_benchmark import CHALLENGES, create_arena
from tools.c0d3rV2.delivery_runner import _read_only_evidence_delivery


def write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def jeopardy_arena(root: Path) -> None:
    write(root, "src/domain/answers.ts", "export const hiddenAnswer = 'photosynthesis';\n")
    write(
        root, "src/game/question.ts",
        "import { hiddenAnswer } from '../domain/answers';\n"
        "export const questionFor = () => hiddenAnswer === 'photosynthesis' ? "
        "'What process converts light energy into chemical energy?' : 'unknown';\n",
    )
    write(root, "src/app.ts", "import { questionFor } from './game/question';\nexport const play = questionFor;\n")
    write(root, "tests/game.spec.ts", "import { play } from '../src/app';\nexpect(play()).toContain('light energy');\n")
    write(root, "src/decoys/answer.ts", "export const hiddenAnswer = 'combustion';\n")
    write(root, "package.json", '{"scripts":{"test":"vitest run"}}')


def test_jeopardy_answer_to_question_regression_traversal(tmp_path):
    jeopardy_arena(tmp_path)
    graph = DependencyTraversal(tmp_path)

    result = graph.traverse("photosynthesis hiddenAnswer", paths=["src/domain/answers.ts"], depth=4)

    assert result["anchors"][0] == "src/domain/answers.ts"
    assert "src/game/question.ts" in result["downstream_consumers"]
    assert "src/app.ts" in result["downstream_consumers"]
    assert "tests/game.spec.ts" in result["downstream_consumers"]
    assert result["regression_tests"] == ["tests/game.spec.ts"]
    assert "src/decoys/answer.ts" not in result["downstream_consumers"]


def test_cycle_is_bounded_and_deterministic(tmp_path):
    write(tmp_path, "src/a.ts", "import { b } from './b'; export const a = b;")
    write(tmp_path, "src/b.ts", "import { a } from './a'; export const b = a;")
    graph = DependencyTraversal(tmp_path)

    first = graph.traverse("a", paths=["src/a.ts"], depth=8, max_nodes=4)
    second = graph.traverse("a", paths=["src/a.ts"], depth=8, max_nodes=4)

    assert first["upstream_dependencies"] == ["src/b.ts"]
    assert first["edges"] == second["edges"]
    assert len(first["nodes"]) == 2


def test_python_relative_import_chain(tmp_path):
    write(tmp_path, "pkg/value.py", "ANSWER = 42\n")
    write(tmp_path, "pkg/service.py", "from .value import ANSWER\ndef result(): return ANSWER\n")
    write(tmp_path, "tests/test_service.py", "from pkg.service import result\ndef test_it(): assert result() == 42\n")

    result = DependencyTraversal(tmp_path).traverse("ANSWER", paths=["pkg/value.py"], depth=3)

    assert "pkg/service.py" in result["downstream_consumers"]
    assert "tests/test_service.py" in result["downstream_consumers"]


def test_injection_combines_memory_hazy_hash_and_validator_scope(tmp_path):
    jeopardy_arena(tmp_path)
    outside = tmp_path.parent / "secret.txt"
    packet = DependencyTraversal(tmp_path).injection_packet(
        "photosynthesis", paths=["src/domain/answers.ts"],
        memory=[{"summary": "Prior decision: preserve the question API."}],
        hazy_hints=[str(tmp_path / "src/game/question.ts"), str(outside)],
        failures=[{"message": "question regression failed"}],
    )

    assert packet["schema"] == "c0d3r.regression-injection/v1"
    assert packet["memory"][0]["summary"].startswith("Prior decision")
    assert packet["hazy_hash_candidates"] == [str(tmp_path / "src/game/question.ts")]
    assert packet["validator_failures"][0]["message"] == "question regression failed"
    assert any(item["path"] == "src/domain/answers.ts" for item in packet["evidence_files"])
    assert any("photosynthesis" in item["excerpt"] for item in packet["evidence_files"])
    assert packet["regression_route"] == [
        {"phase": "change_surface", "path": "src/domain/answers.ts"},
        {"phase": "consumer", "path": "src/game/question.ts"},
        {"phase": "composition_root", "path": "src/app.ts"},
        {"phase": "regression_test", "path": "tests/game.spec.ts"},
    ]


def test_tool_queries_memory_and_hazy_hash(tmp_path):
    jeopardy_arena(tmp_path)

    class Memory:
        def execute(self, params):
            return {"results": ["remembered capture flag contract"]}

    class Locator:
        def execute(self, params):
            return {"paths": [str(tmp_path / "src/domain/answers.ts")]}

    tool = DependencyTraversalTool(tmp_path, Memory(), Locator())
    result = tool.execute({"action": "inject", "query": "photosynthesis"})

    assert result["memory"] == ["remembered capture flag contract"]
    assert result["hazy_hash_candidates"] == [str(tmp_path / "src/domain/answers.ts")]


def test_full_jeopardy_arena_each_answer_reaches_question_round_and_test(tmp_path):
    arena = create_arena(tmp_path / "arena")
    graph = DependencyTraversal(arena)

    for challenge in CHALLENGES:
        category = challenge["id"]
        result = graph.traverse(
            challenge["clue"], paths=[f"vault/{category}/answer.ts"], depth=4,
        )
        assert f"questions/{category}/question.ts" in result["downstream_consumers"]
        assert f"rounds/{category}.ts" in result["downstream_consumers"]
        assert f"tests/{category}.spec.ts" in result["regression_tests"]
        assert all("decoys/" not in path for path in result["downstream_consumers"])


def test_read_only_evidence_path_uses_one_answer_call_without_planning():
    class Session:
        def __init__(self):
            self.calls = 0

        def send(self, prompt, **kwargs):
            self.calls += 1
            return '```json\n{"answer":"ribosome","flag":"FLAG{RIBOSOME_TRANSLATION}"}\n```'

    class Flow:
        session = Session()

    flow = Flow()
    output = _read_only_evidence_delivery(
        "Read-only CTF. Do not write files. Injection packet: evidence_files=[...]", flow,
    )

    assert "FLAG{RIBOSOME_TRANSLATION}" in output
    assert "```" not in output
    assert flow.session.calls == 1
