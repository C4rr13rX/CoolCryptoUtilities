#!/usr/bin/env python3
"""
c0d3r V2 — entry point.

Parses CLI arguments, wires up all dependencies, and delegates to
ProcessFlow for the three-step pipeline (input → context → orchestration).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the V2 package and project root are importable.
_V2_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _V2_ROOT.parent.parent
for _p in (str(_PROJECT_ROOT), str(_V2_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_WEB_ROOT = _PROJECT_ROOT / "web"
if _WEB_ROOT.exists() and str(_WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(_WEB_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="c0d3r",
        description="c0d3r V2 CLI — modular AI engineering assistant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("prompt", nargs="*", help="Prompt text.")
    p.add_argument(
        "-C", "--cd", dest="workdir",
        help="Working directory (defaults to cwd).",
    )
    p.add_argument("--model", help="Override Bedrock model id.")
    p.add_argument(
        "--reasoning", default=None,
        help="Reasoning effort (low, medium, high, extra_high).",
    )
    p.add_argument("--profile", help="AWS profile.")
    p.add_argument("--region", help="AWS region.")
    p.add_argument(
        "--matrix-query", dest="matrix_query",
        help="Query the equation matrix and exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # UTF-8 stdout
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    workdir = Path(args.workdir or os.getcwd()).resolve()

    # ------------------------------------------------------------------
    # Standalone matrix query
    # ------------------------------------------------------------------
    if args.matrix_query:
        from helpers import _ensure_django_ready
        from matrix_helpers import _matrix_search, _seed_base_matrix_django

        if not _ensure_django_ready():
            print("Matrix query requires Django + database configured")
            return 1
        _seed_base_matrix_django()
        print(json.dumps(_matrix_search(args.matrix_query, limit=20), indent=2))
        return 0

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    # ------------------------------------------------------------------
    # Wire up the V2 pipeline
    # ------------------------------------------------------------------
    from sessions import SessionManager
    from executor import Executor
    from usage_tracker import UsageTracker
    from header_renderer import HeaderRenderer
    from lt_mem import LongTermMemory
    from side_load_st_mem_file_location import STSideLoadedMemory
    from side_load_lt_mem_file_location import LTSideLoadedMemory
    from web_search import WebSearch
    from vm_playground import VMPlayground
    from terminal_ui import TerminalUI
    from tool_registry import (
        ToolRegistry,
        ExecutorTool,
        WebSearchTool,
        MemorySearchTool,
        FileLocateTool,
        MatrixSearchTool,
        VMPlaygroundTool,
    )
    from process_flow import ProcessFlow
    from helpers import _runtime_root, _init_heartbeat

    # Session
    sm = SessionManager(
        model=args.model,
        region=args.region,
        profile=args.profile,
        reasoning_effort=args.reasoning,
        workdir=str(workdir),
    )
    session = sm.session
    session_id = sm.session_id

    _init_heartbeat(session)

    try:
        os.environ["C0D3R_SESSION_ID"] = str(session_id)
    except Exception:
        pass

    # Executor
    executor = Executor(workdir)

    # Memory systems
    runtime_root = _runtime_root()
    lt_memory = LongTermMemory(runtime_root)
    st_memory = STSideLoadedMemory(session_id, runtime_root)
    lt_side_memory = LTSideLoadedMemory(runtime_root)

    # Tool registry
    tools = ToolRegistry()
    tools.register(ExecutorTool(executor))
    tools.register(WebSearchTool(WebSearch(session)))
    tools.register(MemorySearchTool(lt_memory))
    tools.register(FileLocateTool(st_memory, lt_side_memory))
    tools.register(MatrixSearchTool())
    tools.register(VMPlaygroundTool(VMPlayground(session, executor)))

    # Usage tracking + header
    usage = UsageTracker(model_id=sm.model_id)
    header = HeaderRenderer(usage)

    # TUI (optional)
    tui = None
    use_tui = (
        os.getenv("C0D3R_TUI", "1").strip().lower()
        not in {"0", "false", "no", "off"}
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )
    if use_tui:
        try:
            tui = TerminalUI(header, workdir)
            tui.start()
            header.ui_manager = tui
        except Exception:
            tui = None

    header.render()

    # ProcessFlow — the main coordinator
    flow = ProcessFlow(
        session=session,
        workdir=workdir,
        tools=tools,
        session_id=session_id,
        lt_memory=lt_memory,
        st_memory=st_memory,
        lt_side_memory=lt_side_memory,
        usage_tracker=usage,
        header=header,
        tui=tui,
    )

    return flow.run(initial_prompt=prompt or None)


if __name__ == "__main__":
    raise SystemExit(main())
