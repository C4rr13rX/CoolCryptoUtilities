#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import psutil

# --- Robust import that works in both package and script contexts ----------------
try:
    # When run as a module from project root: python -m monitoring_guardian.guardian
    from ..tools.codex_session import CodexSession  # type: ignore[relative-beyond-top-level]
except (ImportError, ValueError):
    try:
        # When PYTHONPATH includes project root: python monitoring_guardian/guardian.py
        from tools.codex_session import CodexSession  # type: ignore[no-redef]
    except ImportError:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from tools.codex_session import CodexSession  # type: ignore[no-redef]
# -------------------------------------------------------------------------------

from monitoring_guardian.prompt_text import DEFAULT_GUARDIAN_PROMPT

try:  # Optional dependency when running inside Django
    from services.guardian_state import consume_one_time_prompt, get_guardian_settings
    from services.guardian_status import (
        enqueue_slot,
        mark_slot_finished,
        mark_slot_running,
    )
except Exception:  # pragma: no cover - fallback for standalone CLI
    consume_one_time_prompt = None  # type: ignore
    get_guardian_settings = None  # type: ignore
    enqueue_slot = mark_slot_finished = mark_slot_running = None  # type: ignore

try:  # Optional but preferred for multi-process coordination
    from services.guardian_lock import GuardianLease
except Exception:  # pragma: no cover - fallback when services import fails
    GuardianLease = None  # type: ignore

TRANSCRIPT_DIR = Path("runtime/guardian/transcripts")
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

session = CodexSession("guardian-session", transcript_dir=TRANSCRIPT_DIR)

DEFAULT_CONFIG: Dict[str, object] = {
    "log_files": ["logs/system.log"],
    "report_interval_minutes": 120,
    "sample_tail_lines": 200,
}


class LogFollower:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.position = 0
        self.buffer: deque[str] = deque(maxlen=1000)
        self._lock = threading.Lock()

    def read_new_lines(self) -> List[str]:
        if not self.path.exists():
            return []
        with self._lock:
            lines: List[str] = []
            try:
                with self.path.open("r", encoding="utf-8", errors="ignore") as stream:
                    stream.seek(self.position)
                    for line in stream:
                        cleaned = line.rstrip("\n")
                        lines.append(cleaned)
                        self.buffer.append(cleaned)
                    self.position = stream.tell()
            except Exception as exc:  # filesystem errors are non-fatal
                print(f"[guardian] unable to read {self.path}: {exc}", file=sys.stderr)
            return lines

    def tail(self, lines: int) -> List[str]:
        with self._lock:
            return list(self.buffer)[-lines:]


def detect_main_process() -> Optional[psutil.Process]:
    for proc in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if not cmdline:
                continue
            if "main.py" in cmdline[-1] or any("main.py" in part for part in cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous monitor for CoolCryptoUtilities.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="Path to configuration JSON (defaults to monitoring_guardian/config.json if present, else example file).",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return {**DEFAULT_CONFIG, **json.load(handle)}
    example = Path(__file__).with_name("config.example.json")
    if example.exists():
        with example.open("r", encoding="utf-8") as handle:
            return {**DEFAULT_CONFIG, **json.load(handle)}
    return DEFAULT_CONFIG.copy()


class Guardian:
    def __init__(
        self,
        config: Dict[str, object],
        *,
        prompt_provider: Optional[Callable[[], str]] = None,
        status_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        self.followers: List[LogFollower] = [
            LogFollower((root / Path(str(path))).resolve())
            for path in config.get("log_files", DEFAULT_CONFIG["log_files"])  # type: ignore[index]
        ]
        self.report_interval = int(config.get("report_interval_minutes", 60)) * 60
        self.sample_tail = int(config.get("sample_tail_lines", 200))
        self.last_report_ts = 0.0
        self.shutdown = threading.Event()
        self._force_report = threading.Event()
        self.recent_events: deque[Tuple[float, str, str, str]] = deque(maxlen=4096)
        self.window_seconds = 30 * 60
        self._log_regex = re.compile(
            r"\[(?P<ts>[^]]+)\]\s+\[(?P<level>[A-Z]+)\]\s+(?P<src>[^:]+):\s+(?P<msg>.*)"
        )
        self.prompt_provider = prompt_provider
        self.status_hook = status_hook

    def run(self) -> None:
        self.last_report_ts = time.time()
        self._poll_logs()
        self._check_process_health()
        self._emit_report()
        try:
            while not self.shutdown.is_set():
                self._poll_logs()
                self._check_process_health()
                now = time.time()
                if self._force_report.is_set() or now - self.last_report_ts >= self.report_interval:
                    self._emit_report()
                    self.last_report_ts = now
                    self._force_report.clear()
                time.sleep(10)
        except KeyboardInterrupt:
            print("[guardian] stopping...")
        finally:
            self.shutdown.set()

    def _poll_logs(self) -> None:
        aggregated: List[str] = []
        for follower in self.followers:
            aggregated.extend(follower.read_new_lines())
        if aggregated:
            self._extract_findings(aggregated)

    def _extract_findings(self, lines: Iterable[str]) -> None:
        for line in lines:
            parsed = self._parse_log_line(line)
            if not parsed:
                continue
            ts, level, msg = parsed
            self.recent_events.append((ts, line, self._normalize_message(msg), level))
        self._trim_events()

    def _trim_events(self) -> None:
        cutoff = time.time() - self.window_seconds
        while self.recent_events and self.recent_events[0][0] < cutoff:
            self.recent_events.popleft()

    def _parse_log_line(self, line: str) -> Optional[Tuple[float, str, str]]:
        match = self._log_regex.match(line)
        if not match:
            return None
        ts_str = match.group("ts")
        level = match.group("level").upper()
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            ts = dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None
        msg = match.group("msg")
        return ts, level, msg

    def _normalize_message(self, message: str) -> str:
        normalized = re.sub(r"0x[0-9a-fA-F]+", "<hex>", message)
        normalized = re.sub(r"\d+", "<num>", normalized)
        return normalized

    def _check_process_health(self) -> None:
        proc = detect_main_process()
        if proc is None:
            self.recent_events.append(
                (time.time(), "Main process not detected.", "main_missing", "WARNING")
            )
        else:
            try:
                _ = proc.cpu_percent(interval=0.01)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.recent_events.append(
                    (
                        time.time(),
                        "Unable to read main.py process metrics.",
                        "main_metrics_err",
                        "WARNING",
                    )
                )

    def _emit_report(self) -> None:
        unique_findings = self._gather_unique_findings()

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        report = (
            f"{'='*80}\n"
            f"HEALTH REPORT @ {timestamp}\n"
            f"{'-'*80}\n"
            f"{'-'*80}\n"
            f"{self._build_cli_prompt(unique_findings)}\n"
            f"{'='*80}"
        )
        response: str
        owner_id = f"guardian@{os.getpid()}:{threading.get_ident()}"
        ticket_id = enqueue_slot("codex", owner_id, {"findings": len(unique_findings)}) if enqueue_slot else None
        try:
            if GuardianLease is not None:
                lease = GuardianLease("guardian-codex", timeout=900, poll_interval=2.0)
                if not lease.acquire(cancel_event=self.shutdown):
                    if mark_slot_finished and ticket_id:
                        mark_slot_finished("codex", ticket_id, outcome="skipped", message="lease busy")
                    print("[guardian] codex session currently busy; skipping report this cycle.")
                    return
                try:
                    if mark_slot_running and ticket_id:
                        mark_slot_running("codex", ticket_id)
                    response = session.send(report)
                finally:
                    lease.release()
            else:
                response = session.send(report)
            if mark_slot_finished and ticket_id:
                mark_slot_finished("codex", ticket_id, outcome="success")
            print(response)
            if self.status_hook:
                try:
                    self.status_hook(
                        {
                            "timestamp": time.time(),
                            "findings": unique_findings,
                            "response": response,
                        }
                    )
                except Exception:
                    pass
        except Exception as exc:
            if mark_slot_finished and ticket_id:
                mark_slot_finished("codex", ticket_id, outcome="error", message=str(exc))
            raise

    def _gather_unique_findings(self, limit: int = 30) -> List[str]:
        cutoff = time.time() - self.window_seconds
        seen: set[str] = set()
        results: List[str] = []
        for ts, line, norm, level in reversed(self.recent_events):
            if ts < cutoff:
                continue
            if level not in {"WARNING", "ERROR"}:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            results.append(line)
            if len(results) >= limit:
                break
        return results

    def _suggest_improvements(self) -> List[str]:
        return [
            "Tighten REST poll backoff by caching endpoint latencies.",
            "Increase dataset diversity by scheduling extra news scrapes for low-signal tokens.",
            "Re-run ghost evaluation with a lowered focus window to gather fresh positives.",
            "Profile TensorFlow CPU usage; adjust batch size dynamically under load.",
            "Add additional free news feeds (e.g., Binance Blog, GitHub releases) to the collector.",
            "Validate scheduler queue lengths and ensure backpressure is recorded to DB.",
            "Expand idle-work batches to include stable pairs for synthetic labeling.",
            "Cross-check fallback prices with on-chain listeners before accepting them.",
            "Run regression suite (pytest + lab regression) after each fix cycle.",
            "Review CoinCap DNS suppression timers; consider alternative price APIs.",
        ]

    def _build_cli_prompt(self, findings: List[str]) -> str:
        findings_snippet = "\n".join(f"- {line}" for line in findings) or "No warnings captured in this window."
        if self.prompt_provider:
            try:
                instructions = self.prompt_provider() or DEFAULT_GUARDIAN_PROMPT
            except Exception:
                instructions = DEFAULT_GUARDIAN_PROMPT
        else:
            instructions = DEFAULT_GUARDIAN_PROMPT
        return f"{findings_snippet}\n{instructions}"

    def request_report(self) -> None:
        self._force_report.set()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lease = None
    if GuardianLease is not None:
        lease = GuardianLease("guardian-process", poll_interval=2.0)
        if not lease.acquire():
            print("[guardian] unable to obtain guardian-process lease; exiting.")
            return
    try:
        guardian = Guardian(config, prompt_provider=_default_prompt_provider)
        guardian.run()
    finally:
        if lease:
            lease.release()


def _default_prompt_provider() -> str:
    if get_guardian_settings:
        try:
            settings = get_guardian_settings()
            one_time = consume_one_time_prompt() if consume_one_time_prompt else None
            if one_time:
                return one_time
            return settings.get("default_prompt") or DEFAULT_GUARDIAN_PROMPT
        except Exception:
            return DEFAULT_GUARDIAN_PROMPT
    return DEFAULT_GUARDIAN_PROMPT


if __name__ == "__main__":
    main()
