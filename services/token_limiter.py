"""services/token_limiter.py — Unified Token Limit & Quota Awareness with Auto-Resume.

Tracks token limits, rate limits (TPM/RPM), and quota exhaustion across external
agents: Google Antigravity (`agy`), OpenAI Codex (`codex`), and Anthropic Claude (`claude`/`claude_code`).
Automatically detects limits, extracts replenishment reset delays, pauses cleanly,
and restarts the task automatically when tokens become available again.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger("services.token_limiter")

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "runtime"
TOKEN_LIMIT_FILE = RUNTIME_DIR / "agent_token_limits.json"

T = TypeVar("T")

# Common rate limit & token exhaustion markers across AGY, Codex, and Claude
TOKEN_LIMIT_MARKERS = (
    "rate_limit_error",
    "rate_limit_exceeded",
    "rate limit exceeded",
    "rate limit",
    "ratelimit",
    "too many requests",
    "429",
    "resource_exhausted",
    "resourceexhausted",
    "quota exceeded",
    "quota exhausted",
    "insufficient_quota",
    "tokens per minute",
    "tpm limit",
    "requests per minute",
    "rpm limit",
    "token limit reached",
    "max token limit",
    "token balance depleted",
    "upgrade to increase your usage limit",
    "resets at",
    "resets_at",
    "resetsat",
    "retry after",
    "retry-after",
    "retry_after",
    "try again in",
    "please try again in",
    "overloaded_error",
    "server is temporarily overloaded",
)


def is_token_or_rate_limit(text: str, exc: Optional[Exception] = None) -> bool:
    """True if text or exception message indicates token limit, rate limit, or quota exhaustion."""
    combined = f"{text or ''}\n{str(exc or '')}\n{type(exc).__name__ if exc else ''}".lower()
    return any(marker in combined for marker in TOKEN_LIMIT_MARKERS)


def parse_retry_delay(text: str, default_backoff: float = 30.0) -> float:
    """Extract wait delay in seconds until tokens are available again.

    Parses:
    - Seconds: 'retry after 25s', 'try again in 14.5 seconds', 'resets in 45s'
    - Minutes: 'retry after 2 minutes', 'try again in 1.5m'
    - Unix timestamp: 'resets_at: 1730398000' or 'resets at: 1730398000000'
    - HTTP headers: 'retry-after: 30'
    Returns seconds clamped between 2.0s and 3600.0s.
    """
    if not text:
        return default_backoff

    clean = text.replace("\\", "")

    # 1. Unix timestamp reset: resets_at / resetsAt: 1730398000
    ts_match = re.search(r'"?resets?[_ ]?at"?\s*[:=]?\s*"?(\d{10,13})', clean, re.IGNORECASE)
    if ts_match:
        try:
            val = float(ts_match.group(1))
            if val > 1e12:
                val /= 1000.0
            now = time.time()
            if now < val < now + 86400:
                diff = val - now
                return max(0.5, min(3600.0, diff))
        except Exception:
            pass

    # 2. Standard regex for 'try again in X seconds' or 'resets in X seconds'
    delay_match = re.search(
        r'(?:retry[-_ ]after|try again in|resets? in|available in|wait)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(s|sec|seconds?|m|min|minutes?|h|hours?)?',
        clean,
        re.IGNORECASE,
    )
    if delay_match:
        try:
            qty = float(delay_match.group(1))
            unit = (delay_match.group(2) or "s").lower()
            if unit.startswith("m"):
                qty *= 60.0
            elif unit.startswith("h"):
                qty *= 3600.0
            return max(0.5, min(3600.0, qty))
        except Exception:
            pass

    # 3. HTTP Retry-After header format (digits only)
    header_match = re.search(r'retry-after\s*:\s*(\d+)', clean, re.IGNORECASE)
    if header_match:
        try:
            return max(0.5, min(3600.0, float(header_match.group(1))))
        except Exception:
            pass

    return default_backoff


@dataclass
class AgentLimitState:
    agent: str
    status: str = "available"  # "available", "cooling_down", "exhausted"
    available_at: float = 0.0
    consecutive_hits: int = 0
    total_rate_limits: int = 0
    last_hit_at: float = 0.0
    last_error_message: str = ""
    last_wait_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        return time.time() >= self.available_at

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, round(self.available_at - time.time(), 1))


class TokenLimiter:
    """Master Token Limit & Quota Registry.

    Coordinates rate limits across AGY, Codex, and Claude.
    Automatically manages cooldowns and resumes tasks when tokens are replenished.
    """

    _lock = threading.RLock()
    _states: Dict[str, AgentLimitState] = {}
    _loaded = False

    @classmethod
    def _normalize_agent(cls, agent: str) -> str:
        key = (agent or "agy").strip().lower().replace("-", "_")
        if key in {"agy", "antigravity", "agy_cli", "agy_research", "google_agy"}:
            return "agy"
        if key in {"codex", "codex_cli", "openai_codex"}:
            return "codex"
        if key in {"claude", "claude_code", "claudecode", "anthropic"}:
            return "claude"
        return key

    @classmethod
    def _load(cls) -> None:
        if cls._loaded:
            return
        cls.RUNTIME_DIR = RUNTIME_DIR
        try:
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            if TOKEN_LIMIT_FILE.exists():
                data = json.loads(TOKEN_LIMIT_FILE.read_text(encoding="utf-8"))
                for k, v in data.items():
                    cls._states[k] = AgentLimitState(**v)
        except Exception as exc:
            logger.info("Could not load token limits: %s", exc)
        cls._loaded = True

    @classmethod
    def _save(cls) -> None:
        try:
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            serializable = {k: asdict(v) for k, v in cls._states.items()}
            TOKEN_LIMIT_FILE.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.info("Could not save token limits: %s", exc)

    @classmethod
    def get_state(cls, agent: str) -> AgentLimitState:
        with cls._lock:
            cls._load()
            norm = cls._normalize_agent(agent)
            if norm not in cls._states:
                cls._states[norm] = AgentLimitState(agent=norm)
            state = cls._states[norm]
            if not state.is_available:
                state.status = "cooling_down"
            else:
                state.status = "available"
            return state

    @classmethod
    def record_rate_limit(
        cls,
        agent: str,
        error_message: str,
        delay_seconds: Optional[float] = None,
    ) -> float:
        """Record that an agent hit a token or rate limit, setting its available_at timestamp."""
        with cls._lock:
            cls._load()
            norm = cls._normalize_agent(agent)
            state = cls.get_state(norm)

            state.consecutive_hits += 1
            state.total_rate_limits += 1
            state.last_hit_at = time.time()
            state.last_error_message = (error_message or "Rate limit / token quota exceeded")[:300]

            # Determine delay: parse explicitly, or apply exponential backoff (15s, 30s, 60s, 120s...)
            if delay_seconds is not None and delay_seconds > 0:
                wait_time = delay_seconds
            else:
                base_backoff = 15.0 * (2 ** min(state.consecutive_hits - 1, 4))
                wait_time = parse_retry_delay(error_message, default_backoff=base_backoff)

            state.last_wait_seconds = wait_time
            state.available_at = time.time() + wait_time
            state.status = "cooling_down"
            cls._save()

            available_str = time.strftime("%H:%M:%S", time.localtime(state.available_at))
            logger.warning(
                "[TokenLimiter] Agent '%s' hit token/rate limit (Hit #%d). "
                "Pausing for %.1fs until tokens replenish at %s.",
                norm,
                state.consecutive_hits,
                wait_time,
                available_str,
            )
            return wait_time

    @classmethod
    def record_success(cls, agent: str) -> None:
        """Reset consecutive hits and cooldown after a successful request or reset."""
        with cls._lock:
            cls._load()
            norm = cls._normalize_agent(agent)
            state = cls.get_state(norm)
            state.consecutive_hits = 0
            state.available_at = 0.0
            state.status = "available"
            cls._save()

    @classmethod
    def is_available(cls, agent: str) -> bool:
        """True if the agent is not in cooldown and has tokens available."""
        with cls._lock:
            state = cls.get_state(agent)
            return state.is_available

    @classmethod
    def wait_until_available(
        cls,
        agent: str,
        max_wait_s: float = 300.0,
        callback: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Wait until tokens are replenished for the agent.

        Returns True when available, or False if exceeded max_wait_s.
        """
        norm = cls._normalize_agent(agent)
        state = cls.get_state(norm)
        if state.is_available:
            return True

        remaining = state.seconds_remaining
        if remaining > max_wait_s:
            msg = f"[TokenLimiter] Cooldown for {norm} ({remaining:.1f}s) exceeds max wait ({max_wait_s:.1f}s)."
            if callback:
                callback(msg)
            return False

        available_str = time.strftime("%H:%M:%S", time.localtime(state.available_at))
        msg = f"[TokenLimiter] Token limit active for {norm.upper()}. Pausing {remaining:.1f}s until tokens replenish at {available_str}..."
        logger.info(msg)
        if callback:
            callback(msg)

        # Sleep with 1-second ticks so interrupts/cancellations respond promptly
        deadline = state.available_at
        while time.time() < deadline:
            sleep_chunk = min(1.0, deadline - time.time())
            if sleep_chunk > 0:
                time.sleep(sleep_chunk)

        with cls._lock:
            state.status = "available"
            cls._save()

        ready_msg = f"[TokenLimiter] Tokens replenished for {norm.upper()}. Automatically resuming execution now."
        logger.info(ready_msg)
        if callback:
            callback(ready_msg)
        return True

    @classmethod
    def auto_restart_call(
        cls,
        agent: str,
        call_fn: Callable[[], T],
        max_retries: int = 4,
        callback: Optional[Callable[[str], None]] = None,
    ) -> T:
        """Executes call_fn with automatic token awareness.
        If a token/rate limit occurs, waits for token replenishment and restarts automatically!
        """
        norm = cls._normalize_agent(agent)
        attempts = 0

        while attempts < max_retries:
            attempts += 1
            # 1. Wait if currently in cooldown
            cls.wait_until_available(norm, callback=callback)

            try:
                result = call_fn()
                # 2. Check string results for embedded rate limits (some CLIs exit 0 with error text)
                if isinstance(result, str) and is_token_or_rate_limit(result):
                    delay = cls.record_rate_limit(norm, result)
                    if attempts < max_retries:
                        msg = f"[{norm.upper()} Rate Limit] Output indicates quota/token limit. Pausing {delay:.1f}s, will auto-restart (attempt {attempts}/{max_retries})..."
                        logger.warning(msg)
                        if callback:
                            callback(msg)
                        cls.wait_until_available(norm, callback=callback)
                        continue
                # Success
                cls.record_success(norm)
                return result
            except Exception as exc:
                if is_token_or_rate_limit("", exc=exc) and attempts < max_retries:
                    delay = cls.record_rate_limit(norm, str(exc))
                    msg = f"[{norm.upper()} Rate Limit] Exception: {exc}. Pausing {delay:.1f}s, will auto-restart (attempt {attempts}/{max_retries})..."
                    logger.warning(msg)
                    if callback:
                        callback(msg)
                    cls.wait_until_available(norm, callback=callback)
                    continue
                # Non-rate-limit error or retries exhausted
                raise

        raise RuntimeError(f"Agent '{norm}' exhausted max auto-restart attempts ({max_retries}) due to token limits.")

    @classmethod
    def get_all_limits_summary(cls) -> Dict[str, Any]:
        """Provides full token limit status across all external agents (AGY, Codex, Claude)."""
        with cls._lock:
            cls._load()
            summary = {}
            for ag in ("agy", "codex", "claude"):
                state = cls.get_state(ag)
                summary[ag] = {
                    "agent": ag,
                    "status": "available" if state.is_available else "cooling_down",
                    "is_available": state.is_available,
                    "seconds_remaining": state.seconds_remaining,
                    "available_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(state.available_at)) if state.available_at > 0 else "Ready",
                    "consecutive_hits": state.consecutive_hits,
                    "total_rate_limits": state.total_rate_limits,
                    "last_wait_seconds": state.last_wait_seconds,
                    "last_error": state.last_error_message,
                }
            return summary
