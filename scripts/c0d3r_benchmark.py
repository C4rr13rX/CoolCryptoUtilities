#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PROMPTS = [
    "Write a Python function that retries a request with exponential backoff and jitter.",
    "Given a list of dicts with keys name and score, return the top 3 sorted by score desc.",
]


def _run_codex(prompt: str, images: List[str] | None = None) -> str | None:
    try:
        import shutil
        if shutil.which("codex") is None:
            return None
        from tools.codex_session import CodexSession, codex_default_settings

        session = CodexSession(
            session_name="c0d3r-benchmark-codex",
            transcript_dir=Path("runtime/benchmarks/transcripts/codex"),
            read_timeout_s=120,
            **codex_default_settings(),
        )
        return session.send(prompt, stream=False, images=images)
    except Exception:
        return None


def _run_c0d3r(prompt: str, images: List[str] | None = None) -> str:
    from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings

    session = C0d3rSession(
        session_name="c0d3r-benchmark",
        transcript_dir=Path("runtime/benchmarks/transcripts/c0d3r"),
        read_timeout_s=120,
        **c0d3r_default_settings(),
    )
    return session.send(prompt, stream=False, images=images)


def _toy_app_prompt() -> str:
    return (
        "Create a single-file Kivy toy app for electronics engineering calculations. "
        "Requirements:\n"
        "- Provide two pure functions with exact names:\n"
        "  - ohms_law_voltage(current_a, resistance_ohm) -> voltage_v\n"
        "  - rc_time_constant(resistance_ohm, capacitance_f) -> tau_s\n"
        "- Include a minimal Kivy UI with two tabs: Ohm's Law and RC.\n"
        "- Inputs: current (A) + resistance (Ohm) for Ohm's; resistance + capacitance for RC.\n"
        "- Output label must show computed value with units.\n"
        "- Avoid external dependencies beyond Kivy and stdlib.\n"
        "- The module must import without running the UI (only run App if __name__ == '__main__').\n"
        "Return only the code in a Python fenced block."
    )


def _extract_code(text: str) -> str:
    if "```" not in text:
        return ""
    parts = text.split("```")
    for i in range(len(parts) - 1):
        header = parts[i].strip().lower()
        body = parts[i + 1]
        if header.endswith("python") or header.endswith("py") or header.endswith("python3"):
            return body.strip()
    return parts[1].strip() if len(parts) > 1 else ""


def _score_toy_app(code: str) -> tuple[int, List[str]]:
    issues: List[str] = []
    if "class" not in code or "App" not in code:
        issues.append("Missing Kivy App class.")
    if "TabbedPanel" not in code and "TabbedPanelItem" not in code:
        issues.append("Missing tabbed UI.")
    if "ohms_law_voltage" not in code:
        issues.append("Missing ohms_law_voltage function.")
    if "rc_time_constant" not in code:
        issues.append("Missing rc_time_constant function.")
    if "__main__" not in code:
        issues.append("Missing __main__ guard.")
    score = max(0, 10 - len(issues) * 2)
    return score, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick benchmark: compare c0d3r output vs Codex when available.")
    parser.add_argument("--prompt", action="append", help="Prompt to run (repeatable).")
    parser.add_argument("--image", action="append", help="Optional image paths for UX review prompts.")
    parser.add_argument("--smoke-cmd", help="Optional local smoke-test command to run before prompts.")
    parser.add_argument("--kivy-toy", action="store_true", help="Run the Kivy toy app benchmark.")
    args = parser.parse_args()

    prompts: List[str] = args.prompt or DEFAULT_PROMPTS
    images = [p for p in (args.image or []) if p]
    if args.kivy_toy:
        prompts.append(_toy_app_prompt())
    if images:
        prompts.append(
            "You are a UX reviewer. Review the attached screenshots and return ONLY JSON with keys: "
            '{"status":"pass|fail","issues":[{"title":"","detail":"","severity":"info|warn|error"}],"notes":""}.'
        )
    out_dir = Path("runtime/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report = out_dir / f"benchmark_{ts}.md"

    lines = [f"# c0d3r benchmark {ts}", ""]
    if args.smoke_cmd:
        lines.append("## Smoke test")
        lines.append(f"Command: `{args.smoke_cmd}`")
        from services.agent_workspace import run_command

        code, stdout, stderr = run_command(args.smoke_cmd, cwd=PROJECT_ROOT)
        lines.append(f"Exit code: {code}")
        if stdout:
            lines.append("```\n" + stdout.strip() + "\n```")
        if stderr:
            lines.append("```\n" + stderr.strip() + "\n```")
        lines.append("")
    for idx, prompt in enumerate(prompts, start=1):
        lines.append(f"## Prompt {idx}")
        lines.append(prompt)
        lines.append("")
        c0d3r_output = _run_c0d3r(prompt, images=images if images else None)
        lines.append("### c0d3r")
        lines.append("```")
        lines.append(c0d3r_output.strip())
        lines.append("```")
        codex_output = _run_codex(prompt, images=images if images else None)
        if codex_output is None:
            lines.append("### Codex")
            lines.append("_skipped (codex CLI not available)_")
        else:
            lines.append("### Codex")
            lines.append("```")
            lines.append(codex_output.strip())
            lines.append("```")
        if args.kivy_toy and "Kivy toy app" in prompt:
            code = _extract_code(c0d3r_output)
            score, issues = _score_toy_app(code)
            lines.append("### Toy app score (c0d3r)")
            lines.append(f"Score: {score}/10")
            if issues:
                lines.append("Issues:")
                lines.extend([f"- {item}" for item in issues])
            else:
                lines.append("Issues: none")
            if score < 8:
                lines.append("Recommendation: Upgrade feedback loop or adjust model selection.")
            else:
                lines.append("Recommendation: OK (no upgrade needed).")
        lines.append("")

    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote benchmark report to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
