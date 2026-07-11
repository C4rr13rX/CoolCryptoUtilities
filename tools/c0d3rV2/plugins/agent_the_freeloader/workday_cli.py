from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .feedback import ModelFeedbackStore
from .workday import WorkdayConfig, WorkdayStore, WorkdaySupervisor


def _json(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Durable C0d3rV2 + AgentTheFreeloader workday queue")
    parser.add_argument("--db", help="SQLite queue path")
    commands = parser.add_subparsers(dest="command", required=True)

    enqueue = commands.add_parser("enqueue", help="enqueue one atomic job")
    enqueue.add_argument("prompt", nargs="?")
    enqueue.add_argument("--prompt-file")
    enqueue.add_argument("--workdir", default=".")
    enqueue.add_argument("--validate", default="")
    enqueue.add_argument("--priority", type=int, default=0)
    enqueue.add_argument("--max-attempts", type=int, default=3)
    enqueue.add_argument("--timeout", type=int, default=1800)
    enqueue.add_argument("--tag", action="append", default=[])

    run = commands.add_parser("run", help="run the supervisor")
    run.add_argument("--until-empty", action="store_true")
    run.add_argument("--hours", type=float)
    run.add_argument("--concurrency", type=int)

    listing = commands.add_parser("list", help="list queued and historical jobs")
    listing.add_argument("--status")
    listing.add_argument("--limit", type=int, default=100)

    show = commands.add_parser("show", help="show one job")
    show.add_argument("job_id")

    cancel = commands.add_parser("cancel", help="cancel a job")
    cancel.add_argument("job_id")

    retry = commands.add_parser("retry", help="resume a failed job with its checkpoint")
    retry.add_argument("job_id")
    retry.add_argument("--extra-attempts", type=int, default=1)

    report = commands.add_parser("report", help="write a report for recent jobs")
    report.add_argument("--hours", type=float, default=24.0)

    commands.add_parser("status", help="show queue, budgets, and model feedback")

    args = parser.parse_args(argv)
    config = WorkdayConfig.from_env(db_path=args.db)
    store = WorkdayStore(config.db_path)

    if args.command == "enqueue":
        prompt = args.prompt or ""
        if args.prompt_file:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        if not prompt.strip():
            parser.error("enqueue requires prompt text or --prompt-file")
        job_id = store.enqueue(
            prompt,
            workdir=args.workdir,
            validation_command=args.validate,
            priority=args.priority,
            max_attempts=args.max_attempts,
            timeout_seconds=args.timeout,
            tags=args.tag,
        )
        _json(store.get(job_id))
        return 0

    if args.command == "run":
        if args.concurrency:
            config = WorkdayConfig(**{**config.__dict__, "concurrency": max(1, args.concurrency)})
        supervisor = WorkdaySupervisor(config)
        seconds = args.hours * 3600.0 if args.hours is not None else None
        _json(supervisor.run(until_empty=args.until_empty, max_runtime_seconds=seconds))
        return 0

    if args.command == "list":
        _json(store.list(status=args.status, limit=args.limit))
        return 0
    if args.command == "show":
        job = store.get(args.job_id)
        if not job:
            print("job not found", file=sys.stderr)
            return 2
        _json(job)
        return 0
    if args.command == "cancel":
        _json({"job_id": args.job_id, "cancel_requested": store.cancel(args.job_id)})
        return 0
    if args.command == "retry":
        resumed = store.requeue(args.job_id, extra_attempts=max(1, args.extra_attempts))
        _json({"job_id": args.job_id, "requeued": resumed, "job": store.get(args.job_id)})
        return 0 if resumed else 2
    if args.command == "report":
        supervisor = WorkdaySupervisor(config)
        supervisor.started_at = time.time() - max(0.0, args.hours) * 3600.0
        supervisor.stop_reason = "manual report"
        _json(supervisor.write_report())
        return 0
    if args.command == "status":
        supervisor = WorkdaySupervisor(config)
        _json({
            "database": str(config.db_path),
            "queue": store.stats(),
            "budget": supervisor.budget_status(),
            "running": store.list(status="running", limit=100),
            "feedback": ModelFeedbackStore().snapshot(),
        })
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
