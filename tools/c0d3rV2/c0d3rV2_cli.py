#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import re
import os
import getpass
import sys
import time
import datetime
import threading
import queue
import subprocess
import urllib.request
import hashlib
import shutil
import platform
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = PROJECT_ROOT / "web"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if WEB_ROOT.exists() and str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="c0d3r",
        description="c0d3r CLI (Bedrock-backed) - run prompts against a working directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text. If omitted, reads from stdin.")
    parser.add_argument("-C", "--cd", dest="workdir", help="Working directory (defaults to current directory).")
    parser.add_argument("--model", help="Override Bedrock model id or inference profile.")
    parser.add_argument("--reasoning", default=None, help="Reasoning effort (low, medium, high, extra_high).")
    parser.add_argument("--profile", help="AWS profile to use (defaults to auto-detect).")
    parser.add_argument("--region", help="AWS region (default: from env or us-east-1).")
    parser.add_argument("--research", action="store_true", help="Enable web research + synthesis.")
    parser.add_argument("--image", action="append", help="Image path(s) for multimodal review.")
    parser.add_argument("--doc", "--document", dest="documents", action="append", help="Document path(s) for Bedrock document analysis.")
    parser.add_argument("--context", action="append", help="Extra context text to inject before the prompt.")
    parser.add_argument("--context-file", action="append", help="Path to a text/markdown file to inject as context.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output.")
    parser.add_argument("--tool-loop", action="store_true", help="Enable local command execution loop.")
    parser.add_argument("--no-tools", action="store_true", help="Disable local command execution loop.")
    parser.add_argument("--no-context", action="store_true", help="Disable automatic repo context summary.")
    parser.add_argument("--scientific", action="store_true", help="Enable scientific-method analysis mode.")
    parser.add_argument("--no-scientific", action="store_true", help="Disable scientific-method analysis mode.")
    parser.add_argument("--rigorous", action="store_true", help="Enable rigorous verification mode.")
    parser.add_argument("--no-rigorous", action="store_true", help="Disable rigorous verification mode.")
    parser.add_argument("--matrix-query", dest="matrix_query", help="Query the equation matrix and exit.")
    parser.add_argument("--scripted", dest="scripted", help="Path to a newline-delimited scripted prompt file.")
    return parser

def _emit_live(message: str) -> None:
    if not _live_log_enabled():
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    if _UI_MANAGER:
        _UI_MANAGER.write_line(line)
    try:
        if not _UI_MANAGER:
            print(line, flush=True)
    except Exception:
        pass
    try:
        log_path = _runtime_path("live.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _emit_status_line(message: str) -> None:
    if _UI_MANAGER:
        _UI_MANAGER.set_status(message)
        return
    try:
        sys.stdout.write("\r" + message + " " * 10)
        sys.stdout.flush()
    except Exception:
        pass



def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    _emit_live("boot: start")
    prompt = " ".join(args.prompt).strip()
    scripted_prompts: list[str] | None = None
    if args.scripted:
        try:
            path = Path(args.scripted).expanduser()
            raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            scripted_prompts = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]
        except Exception:
            scripted_prompts = []
    if not prompt and not scripted_prompts:
        prompt = sys.stdin.read().strip()
    _emit_live(f"boot: prompt loaded (len={len(prompt)})")
    # Silence noisy startup warnings for CLI usage.
    os.environ.setdefault("C0D3R_QUIET_STARTUP", "1")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    # Default to full verbosity unless user disables it.
    os.environ.setdefault("C0D3R_VERBOSE_MODEL_OUTPUT", "1")
    # Defaults tuned for "executor_v2": model decides terminal commands; python file_ops are disabled.
    os.environ.setdefault("C0D3R_EXECUTOR_V2", "1")
    os.environ.setdefault("C0D3R_COMMANDS_ONLY", "1")
    os.environ["C0D3R_ENABLE_FILE_OPS"] = "0"
    os.environ.setdefault("C0D3R_BEDROCK_LIVE", "1")
    os.environ.setdefault("C0D3R_BEDROCK_STREAM", "1")
    os.environ.setdefault("C0D3R_AUTO_CONTEXT_COMMANDS", "1")
    if os.getenv("C0D3R_DB_USER") or os.getenv("C0D3R_DB_USER_ID"):
        os.environ.setdefault("C0D3R_DB_SYNC", "1")
    from services.env_loader import EnvLoader
    from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings
    from services.agent_workspace import run_command
    from services.system_probe import system_probe_context

    _emit_live("boot: env load")
    EnvLoader.load()

    _emit_live("boot: settings build")
    settings = c0d3r_default_settings()
    if args.model:
        settings["model"] = args.model
    if args.reasoning:
        settings["reasoning_effort"] = args.reasoning
    if args.profile:
        settings["profile"] = args.profile
    if args.region:
        settings["region"] = args.region
    if args.research:
        settings["research"] = True

    if args.matrix_query:
        query = args.matrix_query.strip()
        if not query:
            print("matrix query is empty")
            return 1
        if not _ensure_django_ready():
            print("matrix query requires Django + database configured")
            return 1
        _seed_base_matrix_django()
        result = _matrix_search(query, limit=20)
        print(json.dumps(result, indent=2))
        return 0

    _emit_live("boot: resolve workdir")
    workdir = Path(args.workdir or os.getcwd()).resolve()
    os.environ.setdefault("C0D3R_ROOT_CWD", str(workdir))
    resolved_images = _resolve_image_paths(args.image, workdir)
    if args.image and not resolved_images:
        _emit_live("boot: image paths provided but none resolved")
    elif resolved_images:
        _emit_live(f"boot: images loaded ({len(resolved_images)})")
    resolved_documents = _resolve_document_paths(args.documents, workdir)
    if args.documents and not resolved_documents:
        _emit_live("boot: document paths provided but none resolved")
    elif resolved_documents:
        _emit_live(f"boot: documents loaded ({len(resolved_documents)})")
    rigorous = bool(settings.get("rigorous_mode", False))
    if args.rigorous:
        rigorous = True
    if args.no_rigorous:
        rigorous = False
    if not args.rigorous and not args.no_rigorous and not rigorous:
        if _rigorous_auto_enabled() and _requires_rigorous_constraints(prompt):
            rigorous = True
            _emit_live("rigorous auto: enabled")
        elif _requires_rigorous_constraints(prompt):
            _emit_live("rigorous auto: suggested; set --rigorous or C0D3R_RIGOROUS_AUTO=1")
    settings["rigorous_mode"] = rigorous
    scientific = args.scientific or (os.getenv("C0D3R_SCIENTIFIC_MODE", "1").strip().lower() not in {"0", "false", "no", "off"})
    if args.no_scientific:
        scientific = False
    _emit_live("boot: build context")
    context_block = ""
    if not args.no_context:
        context_block = _build_context_block(workdir, run_command)
    tool_loop = args.tool_loop or (os.getenv("C0D3R_TOOL_LOOP", "1").strip().lower() not in {"0", "false", "no", "off"})
    if args.no_tools:
        tool_loop = False

    _emit_live("boot: probe context")
    probe_block = system_probe_context(workdir)
    if context_block:
        context_block = f"{probe_block}\n{context_block}"
    else:
        context_block = probe_block
    context_files = list(args.context_file or [])
    env_context_files = os.getenv("C0D3R_CONTEXT_FILE")
    if env_context_files:
        context_files.extend([p.strip() for p in env_context_files.split(",") if p.strip()])
    extra_context = _collect_context_injections(args.context, context_files, workdir)
    if extra_context:
        extra_block = "\n\n".join(extra_context)
        if context_block:
            context_block = f"{context_block}\n\nAdditional context:\n{extra_block}"
        else:
            context_block = f"Additional context:\n{extra_block}"
    if context_block and prompt:
        prompt = f"{context_block}\n\nUser request:\n{prompt}"
    base_request = _strip_context_block(prompt)
    settings = dict(settings)
    if "stream_default" in settings:
        settings["stream_default"] = settings.get("stream_default") and not args.no_stream
    if os.getenv("C0D3R_ONESHOT", "").strip().lower() in {"1", "true", "yes", "on"}:
        if "update project" in base_request.lower():
            retarget = _maybe_retarget_project(base_request, workdir)
            if retarget:
                _emit_live(f"oneshot: retargeting workdir -> {retarget}")
                workdir = retarget
                os.chdir(workdir)
            else:
                _emit_live("oneshot: target project not found; skipping local fallback")
                return 0
    _emit_live("boot: session init")
    session = C0d3rSession(
        session_name="c0d3r-cli",
        transcript_dir=_runtime_path("transcripts"),
        workdir=str(workdir),
        **settings,
    )
    _init_heartbeat(session)
    try:
        os.environ["C0D3R_SESSION_ID"] = str(session.session_id)
    except Exception:
        pass
    usage = UsageTracker(model_id=session.get_model_id())
    header = HeaderRenderer(usage)
    # Initialize terminal UI if available.
    global _UI_MANAGER
    use_tui = os.getenv("C0D3R_TUI", "1").strip().lower() not in {"0", "false", "no", "off"}
    # Disable TUI when not attached to a real terminal (e.g., redirected output).
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        use_tui = False
    if use_tui:
        try:
            _UI_MANAGER = TerminalUI(header, workdir)
            _UI_MANAGER.start()
        except Exception:
            _UI_MANAGER = None
    header.render()
    _emit_live("boot: header rendered")
    os.environ.setdefault("C0D3R_SKIP_PRICING", "1")
    _emit_live("pricing: skipped by default (commands-only mode)")
    # Commands-only execution: no planner/screener/matrix or math grounding.
    plan = {
        "mode": "tool_loop",
        "do_math": False,
        "do_research": False,
        "do_tool_loop": True,
        "model_override": "",
    }
    do_research = False
    tech_matrix = None
    _emit_live("boot: enter repl")
    initial_prompt = prompt or None
    if scripted_prompts:
        if initial_prompt:
            scripted_prompts.insert(0, initial_prompt)
        initial_prompt = None
    return _run_repl(
        session,
        usage,
        workdir,
        run_command,
        scientific=scientific if plan.get("mode") != "tool_loop" else False,
        tool_loop=tool_loop if plan.get("do_tool_loop", True) else False,
        initial_prompt=initial_prompt,
        scripted_prompts=scripted_prompts,
        images=resolved_images,
        documents=resolved_documents,
        header=header,
        pre_research_enabled=do_research,
        tech_matrix=tech_matrix,
        plan=plan,
    )