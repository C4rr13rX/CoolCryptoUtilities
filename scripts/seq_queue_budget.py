"""How many seconds per 10 minutes does each task CATEGORY hold the sequential queue?

The question this answers is not "which task is slow" but "which task makes the
FEED wait". `SequentialScheduler._execute` runs the task on a worker thread and
then `join(timeout=task.timeout_sec)` **on the scheduler's own thread**, so
every second a task spends before it returns or times out is a second the tasks
behind it -- data_ingest among them -- do not get.

That makes one log line an exact measurement rather than an estimate: a
``sequential task_timeout`` event means the scheduler sat in that join for the
task's full ``timeout_sec`` and then gave up. Summing those per category, over a
window, is seconds-of-queue-held per category with no modelling in between.

What this CANNOT see, and the reason every number here is a FLOOR: a task that
finishes inside its timeout logs nothing, so its (smaller, but real) hold is
invisible from the log. Use ``--live`` against a running scheduler's
``status()`` for the complete number; this reads history, which is the only
thing available after the fact.

Usage:
    python -X utf8 scripts/seq_queue_budget.py                 # whole log
    python -X utf8 scripts/seq_queue_budget.py --hours 6       # last 6 hours
    python -X utf8 scripts/seq_queue_budget.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Category of each registered task, from production._build_sequential_scheduler.
# Kept here rather than imported so the script runs without booting production.
TASK_CATEGORY = {
    "atf_static_strategy": "trade",
    "scheduler_refresh": "trade",
    "ghost_metrics": "trade",
    "data_ingest": "feed",
    "news_enrichment": "feed",
    "dataset_warmup": "model",
    "candidate_training": "model",
    "telemetry_flush": "housekeeping",
    "background_refresh": "housekeeping",
}

_LINE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*sequential (?P<event>task_\w+): (?P<payload>\{.*\})\s*$"
)


def _parse(path: str, since: Optional[datetime]) -> List[Tuple[datetime, str, dict]]:
    events: List[Tuple[datetime, str, dict]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "sequential task_" not in line:
                continue
            m = _LINE.match(line.rstrip("\n"))
            if not m:
                continue
            try:
                ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
                payload = ast.literal_eval(m.group("payload"))
            except Exception:
                continue
            if since is not None and ts < since:
                continue
            events.append((ts, m.group("event"), payload))
    return events


def budget(path: str, hours: Optional[float] = None) -> Dict[str, object]:
    since = None
    if hours:
        # Anchor on the log's own last event, not wall clock: production may
        # have stopped, and "the last 6 hours of nothing" is not a measurement.
        tail = _parse(path, None)
        if tail:
            since = tail[-1][0] - timedelta(hours=hours)
        events = [e for e in tail if since is None or e[0] >= since]
    else:
        events = _parse(path, None)
    if not events:
        return {"events": 0}

    first, last = events[0][0], events[-1][0]
    span_sec = max(1.0, (last - first).total_seconds())

    held_by_task: Dict[str, float] = defaultdict(float)
    timeouts_by_task: Dict[str, int] = defaultdict(int)
    overruns_by_task: Dict[str, int] = defaultdict(int)
    errors_by_task: Dict[str, int] = defaultdict(int)
    max_running_for: Dict[str, float] = defaultdict(float)

    for _ts, event, payload in events:
        name = str(payload.get("task", "?"))
        if event == "task_timeout":
            # EXACT: the scheduler thread was blocked in join() for this long.
            held_by_task[name] += float(payload.get("timeout", 0.0) or 0.0)
            timeouts_by_task[name] += 1
        elif event == "task_overrun":
            # The scheduler did NOT wait -- it declined to start a second copy.
            # Zero queue cost, but it means this task is not running at all.
            overruns_by_task[name] += 1
            max_running_for[name] = max(
                max_running_for[name], float(payload.get("running_for", 0.0) or 0.0)
            )
        elif event == "task_error":
            errors_by_task[name] += 1

    per_cat: Dict[str, float] = defaultdict(float)
    for name, secs in held_by_task.items():
        per_cat[TASK_CATEGORY.get(name, "unknown")] += secs

    windows = span_sec / 600.0
    return {
        "log": path,
        "events": len(events),
        "window_start": first.isoformat(sep=" "),
        "window_end": last.isoformat(sep=" "),
        "window_sec": round(span_sec, 1),
        "ten_min_windows": round(windows, 2),
        "held_sec_per_10min_by_category": {
            cat: round(secs / windows, 2) for cat, secs in sorted(per_cat.items())
        },
        "held_sec_total_by_category": {cat: round(s, 1) for cat, s in sorted(per_cat.items())},
        "held_sec_per_10min_by_task": {
            name: round(secs / windows, 2)
            for name, secs in sorted(held_by_task.items(), key=lambda kv: -kv[1])
        },
        "timeouts_by_task": dict(sorted(timeouts_by_task.items(), key=lambda kv: -kv[1])),
        "overruns_by_task": dict(sorted(overruns_by_task.items(), key=lambda kv: -kv[1])),
        "errors_by_task": dict(sorted(errors_by_task.items(), key=lambda kv: -kv[1])),
        "max_abandoned_run_sec_by_task": {
            k: round(v, 1) for k, v in sorted(max_running_for.items(), key=lambda kv: -kv[1])
        },
        "floor_only": True,
        "floor_reason": "successful runs log nothing, so real held-seconds are >= these",
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=os.path.join("logs", "system.log"))
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.exists(args.log):
        print("no such log: %s" % args.log)
        return 2
    report = budget(args.log, args.hours)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    if not report.get("events"):
        print("no sequential scheduler events in window")
        return 0

    print("SEQUENTIAL QUEUE BUDGET  %s -> %s  (%.2f ten-minute windows)" % (
        report["window_start"], report["window_end"], report["ten_min_windows"]))
    print("  seconds the SCHEDULER THREAD was blocked, per 10 minutes, by category")
    print("  (floor: only timeouts are logged, so completed runs are invisible)")
    print()
    for cat, secs in report["held_sec_per_10min_by_category"].items():
        print("    %-14s %8.2f s/10min   (%.1f%% of the queue)" % (cat, secs, 100.0 * secs / 600.0))
    print()
    print("  by task:")
    for name, secs in report["held_sec_per_10min_by_task"].items():
        print("    %-20s %8.2f s/10min   timeouts=%d" % (
            name, secs, report["timeouts_by_task"].get(name, 0)))
    if report["overruns_by_task"]:
        print()
        print("  ABANDONED AND STILL RUNNING (scheduler declined to restart; 0 queue cost,")
        print("  but the task itself is NOT RUNNING for that whole time):")
        for name, n in report["overruns_by_task"].items():
            print("    %-20s overruns=%-6d longest abandoned run %.0fs" % (
                name, n, report["max_abandoned_run_sec_by_task"].get(name, 0.0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
