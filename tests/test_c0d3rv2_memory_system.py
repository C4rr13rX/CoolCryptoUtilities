from __future__ import annotations

import tempfile
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"/"c0d3rV2"))

from tools.c0d3rV2.lt_mem import LongTermMemory
from tools.c0d3rV2.st_memory import STMemory
from tools.c0d3rV2.side_load_st_mem_file_location import STSideLoadedMemory
from tools.c0d3rV2.side_load_lt_mem_file_location import LTSideLoadedMemory
from tools.c0d3rV2.process_flow import ProcessFlow


class NoModel:
    def send(self, **_kwargs):
        raise RuntimeError("offline")


def test_long_term_sqlite_recall_uses_partial_project_terms():
    with tempfile.TemporaryDirectory() as tmp:
        memory=LongTermMemory(Path(tmp))
        for index in range(250):
            memory.append(f"routine turn {index}","ok",session_id="other")
        memory.append("We built the lunar telemetry decoder","Stored it under D:/Projects/LunarDecoder",workdir="D:/Projects/LunarDecoder",session_id="project")
        found=memory.search("Do you remember when we worked on lunar decoder?",limit=5)
        assert found and found[0]["session_id"]=="project"
        assert (Path(tmp)/"lt_memory.sqlite3").exists()


def test_short_term_transcript_and_summary_survive_restart():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);memory=STMemory(NoModel(),session_id="alpha",runtime_root=root)
        memory.record_turn("Always use the blue vault","Created plan at alpha/notes/plan.txt",update_summary_model=False)
        restored=STMemory(NoModel(),session_id="alpha",runtime_root=root)
        assert "blue vault" in restored.summary
        assert "plan.txt" in restored.build_transcript_section()


def test_session_hazy_hash_promotes_into_separate_global_database():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); target=root/"arena"/"hidden-flag.txt";target.parent.mkdir();target.write_text("flag")
        short=STSideLoadedMemory("alpha",root);long=LTSideLoadedMemory(root)
        short.record_paths("hidden flag",[str(target)],cwd=str(root),project_root=str(root))
        assert long.absorb_from_session(short.hazy_hash)>=1
        assert str(target.resolve()) in [str(Path(item).resolve()) for item in long.lookup("hidden flag",cwd=str(root),project_root=str(root))]


def test_recall_trigger_recognizes_dates_and_project_continuation():
    assert ProcessFlow._memory_recall_trigger("Do you remember when we built that?")
    assert ProcessFlow._memory_recall_trigger("Continue working on the radio project")
    assert ProcessFlow._memory_recall_trigger("What happened on 2026-07-12?")
    assert not ProcessFlow._memory_recall_trigger("Hello")
    for text in ("2026-07-12", "7/12/2026", "July 12, 2026", "last week"):
        start,end=LongTermMemory._date_range(text)
        assert start is not None and end is not None and end>start
