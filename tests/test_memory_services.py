from services.short_term_memory import ShortTermMemory
from services.long_term_memory import LongTermMemory


def test_short_term_memory_roundtrip(tmp_path):
    memory = ShortTermMemory(runtime_root=tmp_path, session_id="session-a")
    memory.append(
        "user message",
        "assistant response",
        context="context",
        workdir=str(tmp_path),
        model_id="test",
        session_id="session-a",
    )
    entries = memory.load(limit=4)
    assert len(entries) >= 2
    assert entries[-2].role == "user"
    assert entries[-1].role == "assistant"


def test_long_term_memory_roundtrip(tmp_path):
    memory = LongTermMemory(runtime_root=tmp_path)
    memory.append(
        "user message",
        "assistant response",
        context="context",
        workdir=str(tmp_path),
        model_id="test",
        session_id="session-b",
    )
    entries = memory.load(limit=4)
    assert len(entries) >= 2
    assert entries[-2].content == "user message"
