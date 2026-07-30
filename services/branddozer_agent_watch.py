"""services/branddozer_agent_watch.py — auto-resume runs blocked on agent cooldown.

Why this exists
---------------
When Claude Code or Codex hits a usage limit, the delivery loop pauses the
run (``status="blocked"`` with ``context["ai_paused"]``; see
``_pause_run_for_quota`` in branddozer_delivery.py). Nothing then restarts
it: the job completes, and the run sits blocked until someone notices.
That is fine for "out of credits" (a human must act) but wrong for a
*cooldown*, which clears on its own after a stated window.

This module distinguishes the two, records when the agent said it would be
available again, and requeues the run once that moment passes and a live
probe confirms the CLI actually answers.

Both CLIs report limits in their output rather than through an API, so the
reset time is recovered from the message text. When no explicit time is
given we fall back to a bounded retry schedule instead of hammering.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any, Optional

# Cooldowns clear on their own; these are safe to retry automatically.
COOLDOWN_MARKERS = (
    "usage limit reached",
    "rate limit",
    "rate_limit",
    "too many requests",
    "429",
    "please try again later",
    "try again in",
    "temporarily unavailable",
    "overloaded",
    "capacity",
    "server_overloaded",
    "cooldown",
    "resets at",
    "quota will reset",
)

# Hard stops: a human has to add credits or change plan. Never auto-retry,
# because retrying cannot succeed and just burns the run's error budget.
HARD_STOP_MARKERS = (
    "out of credits",
    "not enough credits",
    "insufficient balance",
    "credit balance is insufficient",
    "billing hard limit",
    "exceeded your current quota",
    "payment required",
    "subscription",
    "upgrade your plan",
)

# Retry backoff (seconds) when the agent gives no explicit reset time.
FALLBACK_BACKOFF = (300, 900, 1800, 3600, 7200)
MAX_AUTO_RESUMES = 12

_CLI_FOR_PROVIDER = {
    "claude_code": "claude",
    "claudecode": "claude",
    "cc": "claude",
    "codex": "codex",
    "codex_cli": "codex",
    "openai_codex": "codex",
}


def _now() -> datetime:
    return datetime.now(dt_timezone.utc)


def classify_block(text: str) -> str:
    """Return "cooldown", "hard_stop", or "unknown" for a pause reason."""
    lowered = (text or "").lower()
    # Hard stops win: "quota exceeded, upgrade your plan" must not be
    # treated as a self-clearing cooldown.
    for marker in HARD_STOP_MARKERS:
        if marker in lowered:
            return "hard_stop"
    for marker in COOLDOWN_MARKERS:
        if marker in lowered:
            return "cooldown"
    return "unknown"


def parse_reset_at(text: str, *, now: Optional[datetime] = None) -> Optional[datetime]:
    """Recover the moment an agent says it will be available again.

    Handles the shapes both CLIs emit, e.g.:
      "usage limit reached; resets at 2026-07-30T18:00:00Z"
      "try again in 42 minutes" / "retry after 90s"
      "Please try again at 3pm"  -> not parsed (ambiguous TZ), falls back
    """
    if not text:
        return None
    now = now or _now()
    lowered = text.lower()

    # ISO-8601 timestamp.
    iso = re.search(
        r"(20\d{2}-\d{2}-\d{2}[t ]\d{2}:\d{2}(?::\d{2})?(?:z|[+-]\d{2}:?\d{2})?)",
        lowered,
    )
    if iso:
        raw = iso.group(1).replace(" ", "T")
        raw = raw.replace("z", "+00:00") if raw.endswith("z") else raw
        try:
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt_timezone.utc)
            return parsed
        except ValueError:
            pass

    # Unix epoch seconds, as some rate-limit headers surface.
    epoch = re.search(r"resets?[_ ]?at[\"']?\s*[:=]\s*(\d{10})\b", lowered)
    if epoch:
        try:
            return datetime.fromtimestamp(int(epoch.group(1)), dt_timezone.utc)
        except (ValueError, OSError):
            pass

    # Relative durations: "try again in 5 minutes", "retry after 30s".
    rel = re.search(
        r"(?:try again in|retry after|available again in|wait)\s+"
        # Longest alternatives first: "min" would otherwise match inside
        # "minutes" and leave a trailing "utes" that fails the \b anchor.
        r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|s|m|h)\b",
        lowered,
    )
    if rel:
        amount = float(rel.group(1))
        unit = rel.group(2)
        factor = 1 if unit.startswith("s") else 60 if unit.startswith("m") else 3600
        return now + timedelta(seconds=amount * factor)

    return None


def probe_agent(provider: str, *, timeout_s: float = 60.0) -> dict[str, Any]:
    """Cheap liveness check: is this agent CLI answering right now?

    Runs a trivial prompt rather than trusting the clock, because a reset
    time can be wrong and a resumed run that immediately re-blocks is worse
    than waiting another cycle.
    """
    cli = _CLI_FOR_PROVIDER.get((provider or "").strip().lower())
    if cli is None:
        return {"online": False, "detail": f"not a CLI agent: {provider}"}
    resolved = shutil.which(cli)
    if not resolved:
        return {"online": False, "detail": f"`{cli}` not on PATH"}

    if cli == "claude":
        cmd = [resolved, "-p", "--output-format", "text"]
        stdin = "Reply with the single word: ok"
    else:
        cmd = [resolved, "exec", "--color", "never", "-"]
        stdin = "Reply with the single word: ok"
    if resolved.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd.exe", "/c", *cmd]

    try:
        proc = subprocess.run(
            cmd,
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"online": False, "detail": f"{cli} probe timed out"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"online": False, "detail": f"{cli} probe error: {exc!r}"}

    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    kind = classify_block(combined)
    if kind in {"cooldown", "hard_stop"}:
        return {"online": False, "detail": combined.strip()[:400], "block_kind": kind}
    if proc.returncode != 0:
        return {
            "online": False,
            "detail": (proc.stderr or "non-zero exit").strip()[:400],
        }
    return {"online": True, "detail": (proc.stdout or "").strip()[:200]}


def next_retry_delay(attempt: int) -> int:
    """Bounded backoff for cooldowns with no stated reset time."""
    idx = min(max(attempt, 0), len(FALLBACK_BACKOFF) - 1)
    return FALLBACK_BACKOFF[idx]


def plan_resume(run_context: dict[str, Any], *, now: Optional[datetime] = None) -> dict[str, Any]:
    """Decide whether a blocked run should be resumed, and when.

    Returns a dict with:
      action  — "resume" | "wait" | "give_up" | "ignore"
      reason  — human-readable explanation
      due_at  — ISO timestamp when the next attempt is allowed (for "wait")
    """
    now = now or _now()
    paused = (run_context or {}).get("ai_paused") or {}
    if not paused:
        return {"action": "ignore", "reason": "run is not paused on an agent limit"}

    provider = str(
        (run_context or {}).get("session_provider")
        or (run_context or {}).get("agent_provider")
        or ""
    ).lower()
    if provider not in _CLI_FOR_PROVIDER:
        return {
            "action": "ignore",
            "reason": f"auto-resume only handles CLI agents, not {provider or 'unset'}",
        }

    reason_text = str(paused.get("reason") or "")
    # Prefer the classification captured at pause time: it saw the full
    # provider message, while `reason` here is truncated to 400 chars.
    kind = str(paused.get("block_kind") or "") or classify_block(reason_text)
    if kind == "hard_stop":
        return {
            "action": "give_up",
            "reason": "provider reported a billing/quota stop that cannot self-clear",
        }

    attempts = int((run_context or {}).get("auto_resume_attempts") or 0)
    if attempts >= MAX_AUTO_RESUMES:
        return {
            "action": "give_up",
            "reason": f"exhausted {MAX_AUTO_RESUMES} automatic resume attempts",
        }

    # An explicit due time set by a previous cycle wins over re-parsing.
    stored_due = (run_context or {}).get("auto_resume_due_at")
    due_at: Optional[datetime] = None
    if stored_due:
        try:
            due_at = datetime.fromisoformat(str(stored_due))
            if due_at.tzinfo is None:
                due_at = due_at.replace(tzinfo=dt_timezone.utc)
        except ValueError:
            due_at = None
    if due_at is None and paused.get("reset_at"):
        try:
            due_at = datetime.fromisoformat(str(paused["reset_at"]))
            if due_at.tzinfo is None:
                due_at = due_at.replace(tzinfo=dt_timezone.utc)
        except ValueError:
            due_at = None
    if due_at is None:
        due_at = parse_reset_at(reason_text, now=now)
    if due_at is None:
        paused_ts = str(paused.get("ts") or "")
        base = now
        try:
            base = datetime.strptime(paused_ts, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=dt_timezone.utc
            )
        except ValueError:
            pass
        due_at = base + timedelta(seconds=next_retry_delay(attempts))

    if now < due_at:
        return {
            "action": "wait",
            "reason": f"agent cooldown until {due_at.isoformat()}",
            "due_at": due_at.isoformat(),
            "provider": provider,
        }
    return {
        "action": "resume",
        "reason": "cooldown window elapsed; probing agent",
        "due_at": due_at.isoformat(),
        "provider": provider,
    }


__all__ = [
    "classify_block",
    "parse_reset_at",
    "probe_agent",
    "plan_resume",
    "next_retry_delay",
    "COOLDOWN_MARKERS",
    "HARD_STOP_MARKERS",
    "MAX_AUTO_RESUMES",
]
