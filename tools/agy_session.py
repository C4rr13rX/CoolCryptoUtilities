"""tools/agy_session.py — Google Antigravity (`agy`) CLI agent session.

Headless wrapper around the `agy` CLI (Google Antigravity), shaped like
ClaudeCodeSession and CodexSession so the platform, TradingAgent, and the
EdgeForge GA Research Agent can invoke `agy` as a first-class agent.

    session = AgySession(session_name="...", agent="research")
    answer  = session.send("Analyze GA lineage stagnation...", system="...")

Invocation contract
-------------------
Uses non-interactive print mode:
    agy -p --dangerously-skip-permissions --output-format text [--agent <id>] [--model <id>] [--effort <level>]
with the prompt written to stdin (or passed via positional arg when short).
A system prompt is passed through `--append-system-prompt` so the caller's
role framing (such as the Research Agent guiding the GA) survives intact.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from services.token_limiter import TokenLimiter, is_token_or_rate_limit


# Process-wide single-agent lock: ensures only ONE agent call runs at a time,
# switching its own model dynamically across cost tiers rather than spawning
# concurrent agents.
SINGLE_AGENT_LOCK = threading.RLock()

DEFAULT_MODEL = os.getenv("AGY_MODEL", "auto")
DEFAULT_AGENT = os.getenv("AGY_AGENT", "")

# Ordered from least expensive to most expensive
SUPPORTED_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.8-flash-low",
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-high",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-pro",
    "gemini-4-argon-low",
    "gemini-4-argon-medium",
    "gemini-4-argon-high",
    "claude-sonnet-4-5",
    "claude-sonnet-5-5-medium",
    "claude-sonnet-5-5-high",
    "claude-opus-4-5",
    "claude-opus-5-5-high",
)

# Map logical model names to the live AGY CLI model IDs accepted by `agy --model`
_CLI_MODEL_MAP: dict[tuple[str, str], str] = {
    ("gemini-2.5-flash-lite", "low"): "gemini-3.8-flash-low",
    ("gemini-2.5-flash-lite", "medium"): "gemini-3.8-flash-low",
    ("gemini-2.5-flash", "low"): "gemini-3.8-flash-low",
    ("gemini-2.5-flash", "medium"): "gemini-3.8-flash-medium",
    ("gemini-2.5-flash", "high"): "gemini-3.8-flash-high",
    ("gemini-2.5-pro", "low"): "gemini-4-argon-low",
    ("gemini-2.5-pro", "medium"): "gemini-4-argon-medium",
    ("gemini-2.5-pro", "high"): "gemini-4-argon-high",
    ("claude-sonnet-4-5", "low"): "claude-sonnet-5-5-low",
    ("claude-sonnet-4-5", "medium"): "claude-sonnet-5-5-medium",
    ("claude-sonnet-4-5", "high"): "claude-sonnet-5-5-high",
    ("claude-opus-4-5", "low"): "claude-opus-5-5-low",
    ("claude-opus-4-5", "medium"): "claude-opus-5-5-medium",
    ("claude-opus-4-5", "high"): "claude-opus-5-5-high",
}

# Relative cost units per 1M tokens (used for least-expensive-first routing)
MODEL_COST_TIERS: dict[str, int] = {
    "gemini-2.5-flash-lite": 1,
    "gemini-3.8-flash-low": 1,
    "gemini-2.5-flash": 2,
    "gemini-3.8-flash-medium": 2,
    "gemini-3.8-flash-high": 2,
    "claude-haiku-4-5-20251001": 3,
    "gemini-2.5-pro": 4,
    "gemini-4-argon-low": 3,
    "gemini-4-argon-medium": 4,
    "gemini-4-argon-high": 4,
    "claude-sonnet-4-5": 5,
    "claude-sonnet-5-5-medium": 5,
    "claude-sonnet-5-5-high": 5,
    "claude-opus-4-5": 6,
    "claude-opus-5-5-high": 6,
}

ROLE_MODEL_ENV = {
    "planner": "AGY_MODEL_PLANNER",
    "manager": "AGY_MODEL_MANAGER",
    "auditor": "AGY_MODEL_AUDITOR",
    "qa": "AGY_MODEL_QA",
    "worker": "AGY_MODEL_WORKER",
    "research": "AGY_MODEL_RESEARCH",
    "trading": "AGY_MODEL_TRADING",
    "fast": "AGY_MODEL_FAST",
}

_EFFORT_MAP = {
    "low": "low",
    "l": "low",
    "medium": "medium",
    "med": "medium",
    "m": "medium",
    "high": "high",
    "h": "high",
    "extra_high": "max",
    "xhigh": "max",
    "max": "max",
}

DEFAULT_PERMISSION_MODE = os.getenv("AGY_PERMISSION_MODE", "bypassPermissions")


def classify_job_for_agy(
    prompt: str,
    system: str = "",
    role: str | None = None,
) -> dict[str, Any]:
    """Classify the task into a job type and return the least expensive AGY
    model + effort + escalation ladder that can reliably complete the job."""
    text = f"{role or ''}\n{system}\n{prompt}".lower()
    role_key = (role or "").strip().lower()

    # Check explicit role override first
    if role_key in {"fast", "reformulation", "classify", "route", "summary", "memory"}:
        job_type = "fast_utility"
    elif role_key in {"trading", "trading_decision", "ga_research", "research", "researcher", "scientist"}:
        job_type = "quant_research"
    elif role_key in {"coding", "worker", "fix", "debug"}:
        job_type = "coding_and_tools"
    elif role_key in {"planner", "manager", "auditor"}:
        job_type = "planning_and_audit"
    else:
        # Content-based classification
        is_fast = any(
            m in text
            for m in (
                "reformulate", "summarize", "summary", "classify", "route",
                "short-term memory", "compact", "quick", "fast",
            )
        ) and len(prompt) < 2500
        is_structured_simple = (
            ("return only json" in text or "json object" in text or "extract" in text)
            and len(prompt) < 3000
            and not any(k in text for k in ("theorem", "game theory", "genetic", "implement", "refactor", "pytest"))
        )
        is_quant_or_trading = any(
            m in text
            for m in (
                "edgeforge", "genetic algorithm", "lineage", "holdout", "r3v3n!r reading agent",
                "find trades that make money", "new_constraints", "new_theorems", "oos_win_rate",
                "sharpe_ratio", "round-trip fee",
            )
        )
        is_coding_or_tools = any(
            m in text
            for m in (
                "tool_calls", "available tools", "function calling", "file_write",
                "pytest", "refactor", "traceback", "syntaxerror", "implement", "class ", "def ",
            )
        )
        is_deep_architecture = any(
            m in text
            for m in (
                "deep architectural", "formal proof", "complex multi-file", "critical security audit",
            )
        )

        if is_fast:
            job_type = "fast_utility"
        elif is_structured_simple:
            job_type = "structured_extraction"
        elif is_quant_or_trading:
            job_type = "quant_research"
        elif is_deep_architecture:
            job_type = "deep_architecture"
        elif is_coding_or_tools:
            job_type = "coding_and_tools"
        else:
            job_type = "general_reasoning"

    # Map each job type to its least-expensive working model ladder:
    # (model_id, effort, agent_name)
    ladders: dict[str, list[tuple[str, str, str]]] = {
        "fast_utility": [
            ("gemini-2.5-flash", "low", ""),
            ("gemini-2.5-pro", "low", ""),
        ],
        "structured_extraction": [
            ("gemini-2.5-flash", "low", ""),
            ("gemini-2.5-pro", "medium", ""),
            ("claude-sonnet-4-5", "medium", ""),
        ],
        "general_reasoning": [
            ("gemini-2.5-flash", "medium", ""),
            ("gemini-2.5-pro", "medium", ""),
            ("claude-sonnet-4-5", "medium", ""),
        ],
        "quant_research": [
            # Start with flash (medium) for fast JSON decisions; auto-escalate to pro (high) if needed
            ("gemini-2.5-flash", "medium", "research"),
            ("gemini-2.5-pro", "high", "research"),
            ("claude-sonnet-4-5", "high", "research"),
        ],
        "planning_and_audit": [
            ("gemini-2.5-flash", "medium", "research"),
            ("gemini-2.5-pro", "high", "research"),
            ("claude-sonnet-4-5", "high", "research"),
        ],
        "coding_and_tools": [
            ("gemini-2.5-flash", "medium", ""),
            ("gemini-2.5-pro", "high", ""),
            ("claude-sonnet-4-5", "high", ""),
            ("claude-opus-4-5", "high", ""),
        ],
        "deep_architecture": [
            ("gemini-2.5-pro", "high", "research"),
            ("claude-sonnet-4-5", "high", "research"),
            ("claude-opus-4-5", "high", "research"),
        ],
    }
    ladder = ladders.get(job_type, ladders["general_reasoning"])
    primary_model, primary_effort, primary_agent = ladder[0]
    return {
        "job_type": job_type,
        "model": primary_model,
        "effort": primary_effort,
        "agent": primary_agent,
        "ladder": ladder,
        "cost_tier": MODEL_COST_TIERS.get(primary_model, 2),
    }


def _normalize_effort(value: str | None) -> str:
    key = (value or "high").strip().lower().replace("-", "_").replace(" ", "_")
    return _EFFORT_MAP.get(key, "high")


def resolve_agy_executable(executable: str | None = None) -> Optional[str]:
    """Locate the `agy` CLI binary on PATH or standard Windows install locations."""
    candidates = []
    if executable:
        candidates.append(executable)
    env_exe = os.getenv("AGY_EXECUTABLE")
    if env_exe:
        candidates.append(env_exe)
    candidates.extend(["agy", "agy.exe", "agy.cmd"])

    for cand in candidates:
        found = shutil.which(cand)
        if found:
            return found
        cand_path = Path(cand)
        if cand_path.is_file():
            return str(cand_path)

    # Standard Windows AppData installation path for Antigravity CLI
    local_app_data = os.getenv("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    fallback_paths = [
        Path(local_app_data) / "agy" / "bin" / "agy.EXE",
        Path(local_app_data) / "agy" / "bin" / "agy.exe",
        Path(local_app_data) / "Programs" / "Antigravity" / "bin" / "agy.exe",
    ]
    for path in fallback_paths:
        if path.is_file():
            return str(path)
    return None


def agy_available() -> bool:
    """True when the `agy` CLI is resolvable on PATH or standard install path."""
    return resolve_agy_executable() is not None


def agy_default_settings() -> dict[str, Any]:
    """Resolve AGY CLI settings from environment defaults."""
    return {
        "model": os.getenv("AGY_MODEL", ""),
        "reasoning_effort": _normalize_effort(os.getenv("AGY_REASONING_EFFORT", "high")),
        "permission_mode": DEFAULT_PERMISSION_MODE,
        "agent": os.getenv("AGY_AGENT", DEFAULT_AGENT),
    }


def agy_settings_for_role(role: str | None = None) -> dict[str, Any]:
    """Return AGY settings tuned by role, with env overrides."""
    base = agy_default_settings()
    if not role:
        return base
    role_key = role.lower().strip()
    if not role_key:
        return base
    override = os.getenv(ROLE_MODEL_ENV.get(role_key, ""), "")
    if override.strip():
        base["model"] = override.strip()
    if role_key in {"research", "researcher", "scientist", "ga_research"}:
        base["agent"] = "research"
    base["meta_role"] = role_key
    return base


class AgySession:
    """Conforms to the duck-type interface used across the site:

        .send(prompt, *, stream=False, system="", **kwargs) -> str
        .session_name: str
        .transcript_dir: Optional[Path]
    """

    def __init__(
        self,
        session_name: str = "agy",
        transcript_dir: str | Path | None = None,
        *,
        executable: str | None = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        permission_mode: Optional[str] = None,
        agent: Optional[str] = None,
        agent_name: Optional[str] = None,
        read_timeout_s: float | None = None,
        timeout_s: float | None = None,
        stream_default: bool = False,
        verbose_default: bool = False,
        workdir: str | Path | None = None,
        meta_role: Optional[str] = None,
        system_prompt: str = "",
        auto_wait_quota: bool = True,
        **_ignored: Any,
    ) -> None:
        defaults = agy_default_settings()
        self.session_name = session_name
        self.auto_wait_quota = auto_wait_quota
        self.executable = executable or os.getenv("AGY_EXECUTABLE", "agy")
        self.model = model if model is not None else defaults["model"]
        self.reasoning_effort = _normalize_effort(
            reasoning_effort if reasoning_effort is not None else defaults["reasoning_effort"]
        )
        self.permission_mode = permission_mode or defaults["permission_mode"]
        eff_agent = agent if agent is not None else agent_name
        self.agent = eff_agent if eff_agent is not None else defaults.get("agent", "")
        eff_timeout = read_timeout_s if read_timeout_s is not None else (timeout_s if timeout_s is not None else 120.0)
        self.read_timeout_s = float(eff_timeout) if eff_timeout is not None else 120.0
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

        resolved = resolve_agy_executable(self.executable)
        self._resolved_executable = resolved or self.executable
        self._available = resolved is not None
        self._stream_callback: Optional[Callable[[str], None]] = None
        self.last_model_id: str = self.model or "gemini-2.5-flash"
        self.last_job_type: str = "general_reasoning"
        self.last_escalation_trace: list[dict[str, Any]] = []

    # ===== Public API ======================================================

    @classmethod
    def probe(cls) -> dict[str, Any]:
        """Cheap availability check mirroring WizardSession.probe()."""
        path = resolve_agy_executable()
        if not path:
            return {"online": False, "detail": "`agy` not found on PATH"}
        return {"online": True, "detail": path}

    def get_model_id(self) -> str:
        return self.last_model_id or self.model or "agy"

    def send(
        self,
        prompt: str,
        *,
        stream: Optional[bool] = None,
        verbose: Optional[bool] = None,
        system: str = "",
        role: Optional[str] = None,
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
                "[agy missing] Install the Google Antigravity CLI and ensure "
                "`agy` is on PATH."
            )
            self._append_transcript(prompt, resp)
            if stream:
                self._print(resp + "\n")
            self._stream_callback = prev_callback
            return resp

        effective_system = system or self.system_prompt
        # Wait if currently in token limit cooldown
        if self.auto_wait_quota:
            TokenLimiter.wait_until_available("agy", callback=self._print if verbose else None)

        job_plan = classify_job_for_agy(prompt, effective_system, role=role or self.meta_role)
        self.last_job_type = job_plan["job_type"]

        # If caller explicitly pinned a model (not "" or "auto"), respect it as
        # the primary step, followed by any higher-tier models on the ladder.
        pinned_model = (self.model or "").strip()
        if pinned_model and pinned_model.lower() != "auto":
            ladder: list[tuple[str, str, str]] = [
                (pinned_model, self.reasoning_effort or "medium", self.agent or "")
            ]
            for m, eff, ag in job_plan["ladder"]:
                if m != pinned_model:
                    ladder.append((m, eff, self.agent or ag))
        else:
            ladder = [
                (m, eff, self.agent or ag)
                for m, eff, ag in job_plan["ladder"]
            ]

        requires_json = (
            "json object" in f"{effective_system}\n{prompt}".lower()
            or "return only json" in f"{effective_system}\n{prompt}".lower()
        )

        trace: list[dict[str, Any]] = []
        text = ""
        # Enforce SINGLE_AGENT_LOCK: only 1 agent working at a time across the
        # system; it switches its own model upward along the cost ladder only
        # when a cheaper model fails or returns unusable output.
        with SINGLE_AGENT_LOCK:
            try:
                for step_idx, (cand_model, cand_effort, cand_agent) in enumerate(ladder):
                    cmd = self._build_cmd(
                        system=effective_system,
                        model_override=cand_model,
                        effort_override=cand_effort,
                        agent_override=cand_agent,
                    )
                    if verbose:
                        self._print(
                            f"[agy single-agent tier={step_idx + 1} job={job_plan['job_type']} "
                            f"model={cand_model} effort={cand_effort}]\n"
                        )
                    full_prompt = (
                        f"[SYSTEM INSTRUCTIONS]\n{effective_system.strip()}\n\n[USER REQUEST]\n{prompt}"
                        if effective_system.strip()
                        else prompt
                    )
                    out, rc, err = self._run(cmd, full_prompt, stream=stream)
                    candidate_text = (out or "").strip()
                    combined_lower = f"{candidate_text}\n{err or ''}".lower()
                    is_quota = is_token_or_rate_limit(combined_lower)
                    quota_or_err = rc != 0 or is_quota
                    if is_quota:
                        TokenLimiter.record_rate_limit("agy", combined_lower)

                    json_invalid = False
                    if requires_json and candidate_text and not quota_or_err:
                        s_idx, e_idx = candidate_text.find("{"), candidate_text.rfind("}")
                        if s_idx == -1 or e_idx <= s_idx:
                            json_invalid = True

                    if candidate_text and not quota_or_err and not json_invalid:
                        self.last_model_id = cand_model
                        TokenLimiter.record_success("agy")
                        trace.append({
                            "model": cand_model,
                            "effort": cand_effort,
                            "agent": cand_agent,
                            "cost_tier": MODEL_COST_TIERS.get(cand_model, 2),
                            "outcome": "selected",
                        })
                        text = candidate_text
                        break

                    trace.append({
                        "model": cand_model,
                        "effort": cand_effort,
                        "agent": cand_agent,
                        "cost_tier": MODEL_COST_TIERS.get(cand_model, 2),
                        "outcome": "escalated",
                        "rc": rc,
                        "reason": "invalid_json" if json_invalid else (err or "empty_or_error")[:200],
                    })
                    if candidate_text and not text:
                        text = candidate_text
                    elif not text and rc != 0:
                        text = f"[agy error rc={rc}] {(err or '').strip()}"

                # If all ladder steps failed due to token limits, wait until replenished and auto-restart primary model
                if (not text or is_token_or_rate_limit(text)) and self.auto_wait_quota and ladder:
                    prim_model, prim_eff, prim_ag = ladder[0]
                    delay_msg = f"[agy token limit] All models hit token limit. Waiting for replenishment then auto-restarting primary model ({prim_model})...\n"
                    if verbose:
                        self._print(delay_msg)
                    if TokenLimiter.wait_until_available("agy", callback=self._print if verbose else None):
                        cmd = self._build_cmd(system=effective_system, model_override=prim_model, effort_override=prim_eff, agent_override=prim_ag)
                        out, rc, err = self._run(cmd, full_prompt, stream=stream)
                        cand = (out or "").strip()
                        if cand and rc == 0 and not is_token_or_rate_limit(cand):
                            TokenLimiter.record_success("agy")
                            text = cand
                            self.last_model_id = prim_model
            finally:
                self._stream_callback = prev_callback

        self.last_escalation_trace = trace
        self._append_transcript(prompt, text)
        return text

    # ===== Internals =======================================================

    def _build_cmd(
        self,
        *,
        system: str,
        model_override: Optional[str] = None,
        effort_override: Optional[str] = None,
        agent_override: Optional[str] = None,
    ) -> list[str]:
        cmd = [
            self._resolved_executable,
            "--dangerously-skip-permissions",
            "--disable-slash-commands",
            "--output-format",
            "text",
        ]
        chosen_agent = agent_override if agent_override is not None else self.agent
        chosen_model = (model_override if model_override is not None else self.model) or ""
        chosen_effort = _normalize_effort(
            effort_override if effort_override is not None else self.reasoning_effort
        )

        if chosen_agent and chosen_agent.strip():
            cmd.extend(["--agent", chosen_agent.strip()])
        if chosen_model.strip() and chosen_model.strip().lower() != "auto":
            cli_model = _CLI_MODEL_MAP.get(
                (chosen_model.strip(), chosen_effort),
                _CLI_MODEL_MAP.get((chosen_model.strip(), "medium"), chosen_model.strip()),
            )
            cmd.extend(["--model", cli_model])
        if chosen_effort:
            cmd.extend(["--effort", chosen_effort])
        cmd.append("-p")
        return cmd

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("CI", "1")
        env.setdefault("TERM", "dumb")
        return env

    def _run(
        self, cmd: list[str], prompt: str, *, stream: bool
    ) -> tuple[str, int, str]:
        """Run the `agy` CLI with `-p <prompt>`, returning (stdout, rc, stderr)."""
        timeout = None if (self.read_timeout_s or 0) <= 0 else self.read_timeout_s
        # Windows CreateProcess command-line limit is 32,767 chars; truncate middle if huge
        safe_prompt = prompt if len(prompt) <= 24000 else (prompt[:12000] + "\n...[truncated]...\n" + prompt[-11000:])
        # Strip built-in virtual agent names ('research', 'self') before invoking agy.EXE
        exec_cmd: list[str] = []
        skip_next = False
        for idx, token in enumerate(cmd):
            if skip_next:
                skip_next = False
                continue
            if token == "--agent" and idx + 1 < len(cmd) and cmd[idx + 1].lower() in {"research", "default", "self"}:
                skip_next = True
                continue
            exec_cmd.append(token)
        if not exec_cmd or exec_cmd[-1] != "-p":
            exec_cmd.append("-p")
        full_cmd = [*exec_cmd, safe_prompt]
        try:
            proc = subprocess.run(
                self._resolve_windows_cmd(full_cmd),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                cwd=str(self.workdir) if self.workdir else None,
                env=self._env(),
                timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except FileNotFoundError:
            return "", 127, "agy executable not found"
        except subprocess.TimeoutExpired:
            return "", 1, f"agy timed out after {timeout}s"
        except Exception as exc:  # pragma: no cover - defensive
            return "", 1, f"agy spawn error: {exc!r}"

        if stream and proc.stdout:
            self._print(proc.stdout)
        return (proc.stdout or ""), proc.returncode, (proc.stderr or "")

    @staticmethod
    def _resolve_windows_cmd(cmd: list[str]) -> list[str]:
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
                    f"\n=== {stamp} agent={self.agent} model={self.model} "
                    f"effort={self.reasoning_effort} ===\n"
                    f"PROMPT\n{prompt}\n\nRESPONSE\n{response}\n"
                )
        except Exception:
            pass


__all__ = [
    "AgySession",
    "SINGLE_AGENT_LOCK",
    "MODEL_COST_TIERS",
    "classify_job_for_agy",
    "agy_available",
    "agy_default_settings",
    "agy_settings_for_role",
    "resolve_agy_executable",
    "SUPPORTED_MODELS",
]
