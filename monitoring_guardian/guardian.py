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
from typing import Dict, Iterable, List, Optional, Tuple

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

session = CodexSession("guardian-session")

DEFAULT_CONFIG: Dict[str, object] = {
    "log_files": ["logs/system.log"],
    "report_interval_minutes": 60,
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
    def __init__(self, config: Dict[str, object]) -> None:
        root = Path(__file__).resolve().parents[1]
        self.followers: List[LogFollower] = [
            LogFollower((root / Path(str(path))).resolve())
            for path in config.get("log_files", DEFAULT_CONFIG["log_files"])  # type: ignore[index]
        ]
        self.report_interval = int(config.get("report_interval_minutes", 60)) * 60
        self.sample_tail = int(config.get("sample_tail_lines", 200))
        self.last_report_ts = 0.0
        self.shutdown = threading.Event()
        self.recent_events: deque[Tuple[float, str, str, str]] = deque(maxlen=4096)
        self.window_seconds = 30 * 60
        self._log_regex = re.compile(
            r"\[(?P<ts>[^]]+)\]\s+\[(?P<level>[A-Z]+)\]\s+(?P<src>[^:]+):\s+(?P<msg>.*)"
        )

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
                if now - self.last_report_ts >= self.report_interval:
                    self._emit_report()
                    self.last_report_ts = now
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
        response = session.send(report)
        print(response)

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
        prompt = (
            f"{findings_snippet}\n"
            "Instructions:\n"
            "1. Fix any errors found in the logs by making meaningful upgrades while following a fix/test/fix/test loop.\n"
            "2. Answer the following question:\n"
            '"What are 15 improvements to make for this system to (A) enhance it toward equilibrium in producing the most accurate crypto price and volume predictions for windows from 5 minutes to 6 months—from the current data stream onward—while finding buy-low/sell-high opportunities and scheduling them in ghost and live trading with 15% of each profitable swap flowing into stablecoin savings once Nash equilibrium is reached; (B) operate efficiently on an i5 CPU with 32GB RAM without downgrading functionality or accuracy—only more efficient engineering; (C) draw on additional ethical, free news sources at arbitrary dates, scrape/store them per design, and inject them into model training; (D) Try to get ghost trading to start trading accurately sooner than later and let it become more accurate as the system evolves. To such a point that live trading kicks in on what it is doing. We want to get live trading to start as soon as possible, but it does need some pretty decent degree of accuracy. Where can you be accurate most of the time with what changes? Make sure to validate your solution by testing predictions in confusion matrices if you use classification such as predicing an upward climb or a downard fall, and try to integrate that with other metrics to find a margin, for example, but you could also test it on historical ohlcv data we already have if you are using regression, and refine as needed before implementing. Try to expand your horizons all across the ML domain keeping it to methods that will run on a pc with a core i5 cpu and 32 GB of ram and no dedicated GPU with CUDA (E) how can we improve the Django website/command center to show more of what is happening in the trading bot system and implement tools that will help us visualize new outcomes and experiment with refining accuracy and collecting meaningful data for the system to injest. How can we demonstrate all of this better within the organism 3D area to show the processes and data and where they are in snapshots and cluster them better in the visualization to push it to make it more energetic and biology-like pushing our ability to run the snapshots in a sequence one after the other and show a vast network of processes working to predict crypto prices and volume and trade accurately. (F) how can we upgrade the swarm? (G) how can we coordinate the transition of swaps in ghost and live trading for higher gains and lower risks? fix errors as you go using a fix/test loop running all relevant unit tests for changed components)."\n'
            "3. Implement those improvements.\n"
            "4. After each fix, run the relevant tests (unit tests, lab regression, etc.) and report the results.\n"
            "5. Stop only after all items are addressed or when manual approval is required.\n"
            "6. Make sure to update the .gitignore if you add anything that creates many downloaded files like node_modules, insecure information to release to the public, or anything else a .gitignore entry would be qualified for.\n"
            "7. Start main.py and option 7 again. make sure the program is running before you stop, because it has to start using the upgrades as soon as possible to gather data on how it is working. Don't shut the program back down before you stop working. Let it run.\n"
            "8. Make sure not to touch anything in the monitoring_guardian folder or tools/codex_session.py as you are working.\n"

        )
        return prompt


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    guardian = Guardian(config)
    guardian.run()


if __name__ == "__main__":
    main()
