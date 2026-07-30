"""tools/claude_code_session.py — Claude Code CLI agent session.

Headless wrapper around the `claude` CLI (Claude Code), shaped like
CodexSession so BrandDozer can pick it as an *agent* rather than a raw
LLM backend:

    session = ClaudeCodeSession(session_name="...", transcript_dir=Path(...))
    answer  = session.send("Do the thing", system="You are ...")

Distinct from tools/claude_session.py (ClaudeSession), which calls
api.anthropic.com directly for a single completion.  This class drives
the Claude Code *agent*, which plans, reads/edits files, and runs
commands inside `workdir` on its own.  Because it is an agent, the model
choice belongs to Claude Code's own supported set, not the site's
Bedrock/ATF catalogs.

Invocation contract
-------------------
Uses non-interactive print mode:
    claude -p --output-format text --model <id> [--permission-mode ...]
with the prompt written to stdin.  A system prompt is passed through
--append-system-prompt so the caller's role framing survives.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional


DEFAULT_MODEL = os.getenv("CLAUDE_CODE_MODEL", "claude-opus-5")

# Models the Claude Code CLI accepts for --model.  Aliases ("opus",
# "sonnet", "haiku") are resolved by the CLI itself against the account's
# entitlements; the pinned IDs are what the site advertises.
SUPPORTED_MODELS = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
    "claude-fable-5",
)

ROLE_MODEL_ENV = {
    "planner": "CLAUDE_CODE_MODEL_PLANNER",
    "manager": "CLAUDE_CODE_MODEL_MANAGER",
    "auditor": "CLAUDE_CODE_MODEL_AUDITOR",
    "qa": "CLAUDE_CODE_MODEL_QA",
    "worker": "CLAUDE_CODE_MODEL_WORKER",
}

# Claude Code has no numeric "reasoning effort" knob like Codex.  Map the
# site's shared reasoning vocabulary onto thinking-budget token hints so
# the same UI control stays meaningful across agents.
_THINKING_BUDGET = {
    "low": 0,
    "medium": 4000,
    "high": 10000,
    "extra_high": 31999,
    "xhigh": 31999,
}

DEFAULT_PERMISSION_MODE = os.getenv(
    "CLAUDE_CODE_PERMISSION_MODE", "bypassPermissions"
)


def _normalize_reasoning(value: str | None) -> str:
    key = (value or "medium").strip().lower().replace("-", "_").replace(" ", "_")
    return key if key in _THINKING_BUDGET else "medium"


def claude_code_default_settings() -> dict[str, Any]:
    """Resolve Claude Code CLI settings from environment defaults."""
    return {
        "model": os.getenv("CLAUDE_CODE_MODEL", DEFAULT_MODEL),
        "reasoning_effort": _normalize_reasoning(
            os.getenv("CLAUDE_CODE_REASONING_EFFORT", "medium")
        ),
        # BrandDozer runs unattended; the agent needs to act without prompts.
        "permission_mode": DEFAULT_PERMISSION_MODE,
    }


def claude_code_settings_for_role(role: str | None = None) -> dict[str, Any]:
    """Return Claude Code settings tuned by role, with env overrides."""
    base = claude_code_default_settings()
    if not role:
        return base
    role_key = role.lower().strip()
    if not role_key:
        return base
    override = os.getenv(ROLE_MODEL_ENV.get(role_key, ""), "")
    if override.strip():
        base["model"] = override.strip()
    base["meta_role"] = role_key
    return base


def claude_code_available() -> bool:
    """True when the `claude` CLI is resolvable on PATH."""
    return shutil.which(os.getenv("CLAUDE_CODE_EXECUTABLE", "claude")) is not None


class ClaudeCodeSession:
    """Conforms to the duck-type interface used across the site:

        .send(prompt, *, stream=False, system="", **kwargs) -> str
        .session_name: str
        .transcript_dir: Optional[Path]
    """

    def __init__(
        self,
        session_name: str = "claude-code",
        transcript_dir: str | Path | None = None,
        *,
        executable: str | None = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        permission_mode: Optional[str] = None,
        read_timeout_s: float | None = None,
        stream_default: bool = True,
        verbose_default: bool = False,
        workdir: str | Path | None = None,
        meta_role: Optional[str] = None,
        system_prompt: str = "",
        **_ignored: Any,
    ) -> None:
        defaults = claude_code_default_settings()
        self.session_name = session_name
        self.executable = executable or os.getenv("CLAUDE_CODE_EXECUTABLE", "claude")
        self.model = model or defaults["model"]
        self.reasoning_effort = _normalize_reasoning(
            reasoning_effort if reasoning_effort is not None else defaults["reasoning_effort"]
        )
        self.permission_mode = permission_mode or defaults["permission_mode"]
        self.read_timeout_s = None if read_timeout_s is None else float(read_timeout_s)
        self.stream_default = stream_default
        self.verbose_default = verbose_default
        self.meta_role = meta_role
        self.system_prompt = system_prompt
        self.workdir = Path(workdir).resolve() if workdir else None

        self.transcript_dir = Path(transcript_dir) if transcript_dir else None
        if self.transcript_dir is not None:
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
            self.transcript_path = self.transcript_dir / f"{session_name}.log"
        else:
            self.transcript_path = None

        self._resolved_executable = shutil.which(self.executable) or self.executable
        self._available = shutil.which(self.executable) is not None
        self._stream_callback: Optional[Callable[[str], None]] = None

    # ===== Public API ======================================================

    @classmethod
    def probe(cls) -> dict[str, Any]:
        """Cheap availability check mirroring WizardSession.probe()."""
        exe = os.getenv("CLAUDE_CODE_EXECUTABLE", "claude")
        path = shutil.which(exe)
        if not path:
            return {"online": False, "detail": f"`{exe}` not found on PATH"}
        return {"online": True, "detail": path}

    def send(
        self,
        prompt: str,
        *,
        stream: Optional[bool] = None,
        verbose: Optional[bool] = None,
        system: str = "",
        stream_callback: Optional[Callable[[str], None]] = None,
        **_ignored: Any,
    ) -> str:
        stream = self.stream_default if stream is None else stream
        verbose = self.verbose_default if verbose is None else verbose
        prev_callback = self._stream_callback
        if stream_callback is not None:
            self._stream_callback = stream_callback

        if not self._available:
            resp = (
                "[claude-code missing] Install the Claude Code CLI and ensure "
                "`claude` is on PATH (npm i -g @anthropic-ai/claude-code)."
            )
            self._append_transcript(prompt, resp)
            if stream:
                self._print(resp + "\n")
            self._stream_callback = prev_callback
            return resp

        cmd = self._build_cmd(system=system or self.system_prompt)
        if verbose:
            self._print(f"[claude-code] {' '.join(cmd)}\n")

        try:
            out, rc, err = self._run(cmd, prompt, stream=stream)
        finally:
            self._stream_callback = prev_callback

        text = (out or "").strip()
        if rc != 0 and not text:
            text = f"[claude-code error rc={rc}] {(err or '').strip()}"
        self._append_transcript(prompt, text)
        return text

    # ===== Internals =======================================================

    def _build_cmd(self, *, system: str) -> list[str]:
        cmd = [
            self._resolved_executable,
            "-p",
            "--output-format",
            "text",
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.permission_mode:
            cmd.extend(["--permission-mode", self.permission_mode])
        if system.strip():
            cmd.extend(["--append-system-prompt", system.strip()])
        return cmd

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        budget = _THINKING_BUDGET.get(self.reasoning_effort, 4000)
        if budget > 0:
            env["MAX_THINKING_TOKENS"] = str(budget)
        else:
            env.pop("MAX_THINKING_TOKENS", None)
        # Keep the CLI non-interactive and machine-friendly.
        env.setdefault("CI", "1")
        env.setdefault("TERM", "dumb")
        return env

    def _run(
        self, cmd: list[str], prompt: str, *, stream: bool
    ) -> tuple[str, int, str]:
        """Run the CLI with *prompt* on stdin, returning (stdout, rc, stderr)."""
        timeout = None if (self.read_timeout_s or 0) <= 0 else self.read_timeout_s
        try:
            proc = subprocess.run(
                self._resolve_windows_cmd(cmd),
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                cwd=str(self.workdir) if self.workdir else None,
                env=self._env(),
                timeout=timeout,
            )
        except FileNotFoundError:
            return "", 127, "claude executable not found"
        except subprocess.TimeoutExpired:
            return "", 1, f"claude-code timed out after {timeout}s"
        except Exception as exc:  # pragma: no cover - defensive
            return "", 1, f"claude-code spawn error: {exc!r}"

        if stream and proc.stdout:
            self._print(proc.stdout)
        return (proc.stdout or ""), proc.returncode, (proc.stderr or "")

    @staticmethod
    def _resolve_windows_cmd(cmd: list[str]) -> list[str]:
        """On Windows, npm shims are .cmd files that need the shell resolver."""
        if os.name != "nt" or not cmd:
            return cmd
        exe = cmd[0]
        if exe.lower().endswith((".cmd", ".bat")):
            return ["cmd.exe", "/c", *cmd]
        resolved = shutil.which(exe)
        if resolved and resolved.lower().endswith((".cmd", ".bat")):
            return ["cmd.exe", "/c", resolved, *cmd[1:]]
        return cmd

    def _print(self, text: str) -> None:
        if self._stream_callback is not None:
            try:
                self._stream_callback(text)
                return
            except Exception:
                pass
        print(text, end="", flush=True)

    def _append_transcript(self, prompt: str, response: str) -> None:
        if self.transcript_path is None:
            return
        try:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with self.transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n=== {stamp} model={self.model} "
                    f"effort={self.reasoning_effort} ===\n"
                    f"PROMPT\n{prompt}\n\nRESPONSE\n{response}\n"
                )
        except Exception:
            pass


__all__ = [
    "ClaudeCodeSession",
    "claude_code_available",
    "claude_code_default_settings",
    "claude_code_settings_for_role",
    "SUPPORTED_MODELS",
]
