#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output.")
    parser.add_argument("--tool-loop", action="store_true", help="Enable local command execution loop.")
    parser.add_argument("--no-tools", action="store_true", help="Disable local command execution loop.")
    parser.add_argument("--no-context", action="store_true", help="Disable automatic repo context summary.")
    parser.add_argument("--scientific", action="store_true", help="Enable scientific-method analysis mode.")
    parser.add_argument("--no-scientific", action="store_true", help="Disable scientific-method analysis mode.")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        parser.error("Prompt is required (args or stdin).")
    from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings
    from services.agent_workspace import run_command

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

    workdir = Path(args.workdir or os.getcwd()).resolve()
    scientific = args.scientific or (os.getenv("C0D3R_SCIENTIFIC_MODE", "1").strip().lower() not in {"0", "false", "no", "off"})
    if args.no_scientific:
        scientific = False
    context_block = ""
    if not args.no_context:
        context_block = _build_context_block(workdir, run_command)
    tool_loop = args.tool_loop or (os.getenv("C0D3R_TOOL_LOOP", "1").strip().lower() not in {"0", "false", "no", "off"})
    if args.no_tools:
        tool_loop = False

    if context_block:
        prompt = f"{context_block}\n\nUser request:\n{prompt}"
    settings = dict(settings)
    if "stream_default" in settings:
        settings["stream_default"] = settings.get("stream_default") and not args.no_stream
    session = C0d3rSession(
        session_name="c0d3r-cli",
        transcript_dir=Path("runtime/c0d3r/transcripts"),
        read_timeout_s=None,
        workdir=str(workdir),
        **settings,
    )
    if scientific:
        output = _run_scientific_loop(session, prompt, workdir, run_command, images=args.image, stream=not args.no_stream)
    elif tool_loop:
        output = _run_tool_loop(session, prompt, workdir, run_command, images=args.image, stream=not args.no_stream)
    else:
        output = session.send(prompt, stream=not args.no_stream, images=args.image)
    if args.no_stream:
        print(output)
    return 0


def _build_context_block(workdir: Path, run_command) -> str:
    lines = [
        "[context]",
        f"- cwd: {workdir}",
        f"- os: {os.name}",
    ]
    code, stdout, stderr = run_command("git status -sb", cwd=workdir)
    if stdout.strip():
        lines.append("git status -sb:")
        lines.append(stdout.strip()[:2000])
    if stderr.strip():
        lines.append("git status stderr:")
        lines.append(stderr.strip()[:500])
    code, stdout, stderr = run_command("git rev-parse --show-toplevel", cwd=workdir)
    if stdout.strip():
        lines.append(f"repo root: {stdout.strip()}")
    code, stdout, stderr = run_command("Get-ChildItem -Name", cwd=workdir) if os.name == "nt" else run_command("ls -1", cwd=workdir)
    if stdout.strip():
        lines.append("top-level files:")
        lines.append("\n".join(stdout.strip().splitlines()[:80]))
    return "\n".join(lines)


def _run_tool_loop(
    session: C0d3rSession,
    prompt: str,
    workdir: Path,
    run_command,
    *,
    images: List[str] | None,
    stream: bool,
) -> str:
    max_steps = int(os.getenv("C0D3R_TOOL_STEPS", "3"))
    history: List[str] = []
    for step in range(max_steps):
        tool_prompt = (
            "You can run local shell commands to inspect the repo and validate work. "
            "Return ONLY JSON with keys: commands (list of strings) and final (string or empty). "
            "If you are done, set final to your response and commands to []. "
            "Be concise and execution-focused.\n"
            f"Step {step + 1}/{max_steps}.\n"
            f"Request:\n{prompt}\n"
            + ("\n\nRecent outputs:\n" + "\n".join(history[-6:]) if history else "")
        )
        response = session.send(tool_prompt, stream=stream, images=images)
        commands, final = _extract_commands(response)
        if final:
            return final
        if not commands:
            return response
        for cmd in commands[:6]:
            code, stdout, stderr = run_command(cmd, cwd=workdir)
            snippet = f"$ {cmd}\n(exit {code})\n{stdout.strip()}\n{stderr.strip()}".strip()
            history.append(snippet[:4000])
    return history[-1] if history else "No output."


def _run_scientific_loop(
    session: C0d3rSession,
    prompt: str,
    workdir: Path,
    run_command,
    *,
    images: List[str] | None,
    stream: bool,
) -> str:
    # Observe: fixed inspection bundle (empirical evidence)
    bundle = []
    for cmd in [
        "git status -sb",
        "git log -1 --oneline",
        "rg -n \"TODO|FIXME|HACK\" -S",
        "Get-ChildItem -Name" if os.name == "nt" else "ls -1",
    ]:
        code, stdout, stderr = run_command(cmd, cwd=workdir)
        bundle.append(
            f"$ {cmd}\n(exit {code})\n{(stdout or '').strip()}\n{(stderr or '').strip()}"
        )
    evidence = "\n\n".join(bundle)

    analysis_prompt = (
        "Use the scientific method. Do not speculate. "
        "Use ONLY the provided empirical evidence. "
        "Return ONLY JSON with keys:\n"
        "observations (list of strings),\n"
        "hypotheses (list of strings),\n"
        "tests (list of strings),\n"
        "findings (list of {severity, claim, evidence_cmd, evidence_excerpt, files}),\n"
        "conclusion (string),\n"
        "next_steps (list of strings).\n"
        "Every finding must cite evidence_cmd and evidence_excerpt from the bundle.\n\n"
        f"User request:\n{prompt}\n\n"
        f"Evidence bundle:\n{evidence}\n"
    )
    response = session.send(analysis_prompt, stream=stream, images=images)
    commands, final = _extract_commands(response)
    if final:
        return final
    return response


def _extract_commands(text: str) -> Tuple[List[str], str]:
    try:
        import json

        payload = json.loads(_extract_json(text))
        commands = payload.get("commands") or []
        if isinstance(commands, list):
            commands = [str(c) for c in commands if str(c).strip()]
        else:
            commands = []
        final = str(payload.get("final") or "").strip()
        return commands, final
    except Exception:
        try:
            import re

            commands: List[str] = []
            cmd_match = re.search(r'"commands"\s*:\s*\[(.*?)\]', text, re.S)
            if cmd_match:
                inner = cmd_match.group(1)
                commands = [c.strip().strip('"') for c in inner.split(",") if c.strip()]
            final_match = re.search(r'"final"\s*:\s*"(.+)"', text, re.S)
            if final_match:
                final = final_match.group(1).strip()
                return commands, final
        except Exception:
            pass
        return [], ""


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "{}"
    return text[start : end + 1]


if __name__ == "__main__":
    raise SystemExit(main())
