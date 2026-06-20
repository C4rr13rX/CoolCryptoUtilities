"""log_signal — extract the signal from prod_manager / console logs.

Strips out the dashboard-polling INFOs, the duplicate "production.console"
echoes, the gas-rebalance verbose blobs, and the per-tick stream-status
spam, leaving:

  * WARN / ERROR / CRITICAL lines
  * Python tracebacks
  * "shutdown:" sequences (the production manager dying)
  * One-time startup signals (env hydrated, bootstrap complete)
  * gas_swap_executed / gas_swap_failed / gas_swap_unaffordable summaries
  * trade entries / exits (when they exist)
  * brain HTTP failures
  * unhandled exception markers from main.py

Use this to figure out why prod_manager keeps dying without grepping
through dashboard noise. Reads a log file (default console.log) and
prints the filtered stream.

Usage:
  python scripts/log_signal.py                # default: tail console.log
  python scripts/log_signal.py prod_direct.log
  python scripts/log_signal.py --tail 500     # last 500 lines only
  python scripts/log_signal.py --since "2026-06-20 04:00"   # filter by ts
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"

# Patterns we DROP — known noise.
DROP_PATTERNS = [
    re.compile(r"api: GET /api/(console/(status|logs)|guardian/logs|telemetry/(advisories|trades|feedback|metrics|dashboard)|streams/latest)/"),
    re.compile(r"production\.console: "),     # duplicate of production: stream
    re.compile(r"INFO.*market-stream: stream configured"),
    re.compile(r"INFO.*market-stream: using \w+ endpoint"),
    re.compile(r"INFO.*market-stream: \w+(-\w+)+ flow samples"),
    re.compile(r"INFO.*cex-fallback: \w+: \d+ candles for"),
    re.compile(r"DEBUG bootstrap: skipping"),
    re.compile(r"INFO wallet-bootstrap: (generated|persisted|watchlists|bootstrap complete) "),
    re.compile(r"INFO production: wallet bootstrap complete"),
    re.compile(r"INFO production: orchestrator configured"),
    re.compile(r"INFO production: delegation client started"),
    re.compile(r"INFO production: secure settings loaded"),
    re.compile(r"INFO internal-cron: internal cron active"),
    re.compile(r"INFO internal-cron: cron lease busy"),
    re.compile(r"INFO production: cycle \d+ status"),
    re.compile(r"INFO tf-runtime: SKIP_TF_CONFIGURE.*TF disabled"),
    re.compile(r"INFO production: starting auto-bootstrap"),
    re.compile(r"\[feedback\] \[OK\] training:horizon_bias_tuned"),
    re.compile(r"\[feedback\] \[OK\] trading:gas_rebalanced_alert.*'strategy':"),
    re.compile(r"\[feedback\] \[OK\] trading:gas_swap_executed.*'amount':"),
    re.compile(r"INFO ghost-supervisor: live readiness snapshot"),
    re.compile(r"oneDNN custom operations are on"),
    re.compile(r"WARNING: All log messages before absl::InitializeLog"),
    re.compile(r"^\s*cron-bootstrap: started\s*$"),
    re.compile(r"^\s*guardian-bootstrap: started\s*$"),
    re.compile(r"^\s*delegation client started\s*$"),
    re.compile(r"^\s*production-bootstrap: started "),
    re.compile(r"^\s*\[production\] env hydrated"),
]

# Patterns we ALWAYS KEEP — high-signal markers.
KEEP_PATTERNS = [
    re.compile(r"\[(ERROR|CRITICAL|WARNING|WARN)\]"),
    re.compile(r"\bTraceback\b"),
    re.compile(r"\bunhandled exception\b", re.I),
    re.compile(r"\bImportError\b"),
    re.compile(r"\bRuntimeError\b"),
    re.compile(r"\bAttributeError\b"),
    re.compile(r"\bKeyError\b"),
    re.compile(r"\bNameError\b"),
    re.compile(r"\bConnectionError\b"),
    re.compile(r"\bMemoryError\b"),
    re.compile(r"\bOSError\b"),
    re.compile(r"\bDLL\b"),
    re.compile(r"shutdown: "),
    re.compile(r"callback error"),
    re.compile(r"action failed:"),
    re.compile(r"gas_swap_failed"),
    re.compile(r"gas_swap_unaffordable"),
    re.compile(r"gas_swap_stranded"),
    re.compile(r"gas_swap_unsat"),
    re.compile(r"live_promotion"),
    re.compile(r"circuit_breaker"),
    re.compile(r"ghost.*enter|live.*enter|ghost.*exit|live.*exit", re.I),
    re.compile(r"brain unreachable"),
    re.compile(r"BadStatusLine|RemoteDisconnected|TimeoutError"),
]

TS_PATTERN = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def should_keep(line: str) -> bool:
    # KEEP wins over DROP.
    for kp in KEEP_PATTERNS:
        if kp.search(line):
            return True
    for dp in DROP_PATTERNS:
        if dp.search(line):
            return False
    return True  # default: keep — we don't want to silently drop unknowns


def filter_stream(lines: Iterable[str], since: str = "") -> List[str]:
    out: List[str] = []
    for line in lines:
        if since:
            m = TS_PATTERN.match(line)
            if m and m.group(1) < since:
                continue
        if should_keep(line):
            out.append(line.rstrip("\n"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default="logs/console.log",
                    help="Log file (relative to project root)")
    ap.add_argument("--tail", type=int, default=0,
                    help="Only process the last N lines")
    ap.add_argument("--since", default="",
                    help="Drop lines with timestamps before this (YYYY-MM-DD HH:MM:SS)")
    ap.add_argument("--count", action="store_true",
                    help="Print a count of dropped vs kept instead of the lines")
    args = ap.parse_args()

    path = (ROOT / args.file).resolve()
    if not path.exists():
        # also try absolute
        path = Path(args.file).resolve()
    if not path.exists():
        print(f"file not found: {args.file}", file=sys.stderr)
        return 1

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    if args.tail > 0:
        lines = lines[-args.tail:]

    kept = filter_stream(lines, since=args.since)
    if args.count:
        print(f"file:    {path}")
        print(f"total:   {len(lines)}")
        print(f"kept:    {len(kept)}")
        print(f"dropped: {len(lines) - len(kept)}  ({100*(len(lines)-len(kept))/max(len(lines),1):.1f}%)")
    else:
        for ln in kept:
            print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
