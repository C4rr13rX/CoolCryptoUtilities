from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import load_catalog, merge_pool_limits
from .quota import QuotaLedger
from .notifications import WorkdayNotifier


TERMINAL_STATES = {"completed", "failed", "cancelled"}
RUNNABLE_STATES = {"queued", "retry"}


def default_workday_db() -> Path:
    configured = os.getenv("AGENT_FREELOADER_WORKDAY_DB", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[4] / "runtime" / "agent_the_freeloader" / "workday.sqlite3"


@dataclass(frozen=True)
class WorkdayConfig:
    db_path: Path = default_workday_db()
    concurrency: int = 1
    lease_seconds: int = 90
    heartbeat_seconds: int = 15
    poll_seconds: float = 2.0
    job_timeout_seconds: int = 1800
    retry_base_seconds: int = 60
    quota_retry_seconds: int = 300
    max_requests_per_day: int = 200
    max_tokens_per_day: int = 2_000_000
    shift_hours: float = 8.0
    report_dir: Path | None = None
    notifications_enabled: bool = True
    notification_log: Path | None = None

    @classmethod
    def from_env(cls, *, db_path: str | Path | None = None) -> WorkdayConfig:
        root = Path(__file__).resolve().parents[4]
        return cls(
            db_path=Path(db_path) if db_path else default_workday_db(),
            concurrency=max(1, int(os.getenv("ATF_WORKDAY_CONCURRENCY", "1"))),
            lease_seconds=max(15, int(os.getenv("ATF_WORKDAY_LEASE_SECONDS", "90"))),
            heartbeat_seconds=max(5, int(os.getenv("ATF_WORKDAY_HEARTBEAT_SECONDS", "15"))),
            poll_seconds=max(0.1, float(os.getenv("ATF_WORKDAY_POLL_SECONDS", "2"))),
            job_timeout_seconds=max(30, int(os.getenv("ATF_WORKDAY_JOB_TIMEOUT_SECONDS", "1800"))),
            retry_base_seconds=max(1, int(os.getenv("ATF_WORKDAY_RETRY_SECONDS", "60"))),
            quota_retry_seconds=max(1, int(os.getenv("ATF_WORKDAY_QUOTA_RETRY_SECONDS", "300"))),
            max_requests_per_day=max(0, int(os.getenv("ATF_WORKDAY_MAX_REQUESTS", "200"))),
            max_tokens_per_day=max(0, int(os.getenv("ATF_WORKDAY_MAX_TOKENS", "2000000"))),
            shift_hours=max(0.0, float(os.getenv("ATF_WORKDAY_SHIFT_HOURS", "8"))),
            report_dir=root / "runtime" / "agent_the_freeloader" / "reports",
            notifications_enabled=os.getenv("ATF_WORKDAY_NOTIFICATIONS", "1").strip().lower() not in {"0", "false", "no", "off"},
            notification_log=Path(os.getenv(
                "ATF_WORKDAY_NOTIFICATION_LOG",
                str(root / "runtime" / "agent_the_freeloader" / "notifications.jsonl"),
            )),
        )


class WorkdayStore:
    """SQLite-backed durable queue with atomic claims and expiring leases."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_workday_db()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    workdir TEXT NOT NULL,
                    validation_command TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    available_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    timeout_seconds INTEGER NOT NULL DEFAULT 1800,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    heartbeat_at REAL,
                    worker_pid INTEGER,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    checkpoint_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS jobs_runnable_idx
                    ON jobs(status, available_at, priority, created_at);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    ts REAL NOT NULL,
                    kind TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT ''
                );
                """
            )

    def enqueue(
        self,
        prompt: str,
        *,
        workdir: str | Path,
        validation_command: str = "",
        priority: int = 0,
        max_attempts: int = 3,
        timeout_seconds: int = 1800,
        tags: list[str] | None = None,
    ) -> str:
        if not prompt.strip():
            raise ValueError("prompt is required")
        resolved = Path(workdir).expanduser().resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise ValueError(f"workdir is not a directory: {resolved}")
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, prompt, workdir, validation_command, status, priority,
                    created_at, updated_at, available_at, max_attempts,
                    timeout_seconds, tags_json
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id, prompt.strip(), str(resolved), validation_command.strip(),
                    int(priority), now, now, now, max(1, int(max_attempts)),
                    max(30, int(timeout_seconds)), json.dumps(tags or []),
                ),
            )
            self._event(connection, job_id, "queued", "job enqueued")
        return job_id

    def claim(self, owner: str, lease_seconds: int) -> dict | None:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('queued', 'retry')
                  AND cancel_requested = 0
                  AND available_at <= ?
                  AND attempts < max_attempts
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                """
                UPDATE jobs SET status='running', attempts=attempts+1,
                    lease_owner=?, lease_expires_at=?, heartbeat_at=?,
                    started_at=COALESCE(started_at, ?), updated_at=?, error=''
                WHERE id=?
                """,
                (owner, now + lease_seconds, now, now, now, row["id"]),
            )
            self._event(connection, row["id"], "claimed", owner)
            connection.commit()
        return self.get(str(row["id"]))

    def get(self, job_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, *, status: str | None = None, limit: int = 100) -> list[dict]:
        with self._connect() as connection:
            if status:
                rows = connection.execute(
                    "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._row(row) for row in rows]

    def set_worker_pid(self, job_id: str, owner: str, pid: int) -> None:
        self._owned_update(job_id, owner, "worker_pid=?, updated_at=?", (int(pid), time.time()))

    def heartbeat(self, job_id: str, owner: str, lease_seconds: int) -> bool:
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET heartbeat_at=?, lease_expires_at=?, updated_at=?
                WHERE id=? AND lease_owner=? AND status='running'
                """,
                (now, now + lease_seconds, now, job_id, owner),
            )
            return cursor.rowcount == 1

    def checkpoint(self, job_id: str, owner: str, checkpoint: dict) -> None:
        self._owned_update(
            job_id, owner, "checkpoint_json=?, updated_at=?",
            (json.dumps(checkpoint, default=str), time.time()),
        )

    def complete(self, job_id: str, owner: str, result: dict) -> None:
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET status='completed', result_json=?, completed_at=?,
                    updated_at=?, lease_owner=NULL, lease_expires_at=NULL,
                    worker_pid=NULL, error=''
                WHERE id=? AND lease_owner=? AND status='running'
                """,
                (json.dumps(result, default=str), now, now, job_id, owner),
            )
            if cursor.rowcount:
                self._event(connection, job_id, "completed", "validation passed")

    def retry(self, job_id: str, owner: str, error: str, delay_seconds: float, checkpoint: dict) -> None:
        now = time.time()
        job = self.get(job_id)
        exhausted = bool(job and int(job["attempts"]) >= int(job["max_attempts"]))
        status = "failed" if exhausted else "retry"
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET status=?, error=?, checkpoint_json=?, available_at=?,
                    completed_at=?, updated_at=?, lease_owner=NULL,
                    lease_expires_at=NULL, worker_pid=NULL
                WHERE id=? AND lease_owner=? AND status='running'
                """,
                (
                    status, error[:4000], json.dumps(checkpoint, default=str),
                    now + max(0.0, delay_seconds), now if exhausted else None,
                    now, job_id, owner,
                ),
            )
            if cursor.rowcount:
                self._event(connection, job_id, status, error[:1000])

    def defer_for_capacity(
        self, job_id: str, owner: str, error: str, delay_seconds: float, checkpoint: dict,
    ) -> None:
        """Return a job to the queue without spending an attempt when no model can run."""
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET status='retry', attempts=MAX(0, attempts-1),
                    error=?, checkpoint_json=?, available_at=?, updated_at=?,
                    lease_owner=NULL, lease_expires_at=NULL, worker_pid=NULL
                WHERE id=? AND lease_owner=? AND status='running'
                """,
                (
                    error[:4000], json.dumps(checkpoint, default=str),
                    now + max(1.0, delay_seconds), now, job_id, owner,
                ),
            )
            if cursor.rowcount:
                self._event(connection, job_id, "capacity_wait", error[:1000])

    def cancel(self, job_id: str) -> bool:
        now = time.time()
        with self._connect() as connection:
            row = connection.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row[0] in TERMINAL_STATES:
                return False
            if row[0] == "running":
                connection.execute(
                    "UPDATE jobs SET cancel_requested=1, updated_at=? WHERE id=?",
                    (now, job_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE jobs SET status='cancelled', cancel_requested=1,
                        completed_at=?, updated_at=? WHERE id=?
                    """,
                    (now, now, job_id),
                )
            self._event(connection, job_id, "cancel_requested", "user requested cancellation")
        return True

    def requeue(self, job_id: str, *, extra_attempts: int = 1, delay_seconds: float = 0) -> bool:
        """Resume a failed job with its checkpoint so ATF can correct it."""
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET status='retry', max_attempts=max_attempts+?,
                    available_at=?, completed_at=NULL, updated_at=?, error=''
                WHERE id=? AND status='failed' AND cancel_requested=0
                """,
                (max(1, int(extra_attempts)), now + max(0.0, delay_seconds), now, job_id),
            )
            if cursor.rowcount:
                self._event(connection, job_id, "requeued", f"added {max(1, int(extra_attempts))} attempt(s)")
            return cursor.rowcount == 1

    def reclaim_expired(self) -> list[dict]:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT * FROM jobs WHERE status='running'
                  AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                """,
                (now,),
            ).fetchall()
            for row in rows:
                status = "failed" if int(row["attempts"]) >= int(row["max_attempts"]) else "retry"
                connection.execute(
                    """
                    UPDATE jobs SET status=?, available_at=?, error='worker lease expired',
                        lease_owner=NULL, lease_expires_at=NULL, worker_pid=NULL,
                        completed_at=?, updated_at=? WHERE id=?
                    """,
                    (status, now, now if status == "failed" else None, now, row["id"]),
                )
                self._event(connection, row["id"], "lease_expired", status)
            connection.commit()
        return [self._row(row) for row in rows]

    def stats(self, *, since: float | None = None) -> dict:
        where = " WHERE updated_at >= ?" if since is not None else ""
        args = (since,) if since is not None else ()
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT status, COUNT(*) FROM jobs{where} GROUP BY status", args
            ).fetchall()
        counts = {str(row[0]): int(row[1]) for row in rows}
        return {"total": sum(counts.values()), "by_status": counts}

    def _owned_update(self, job_id: str, owner: str, assignment: str, values: tuple) -> None:
        with self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignment} WHERE id=? AND lease_owner=? AND status='running'",
                (*values, job_id, owner),
            )

    @staticmethod
    def _event(connection: sqlite3.Connection, job_id: str, kind: str, detail: str) -> None:
        connection.execute(
            "INSERT INTO events(job_id, ts, kind, detail) VALUES (?, ?, ?, ?)",
            (job_id, time.time(), kind, detail),
        )

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        result = dict(row)
        for key in ("checkpoint_json", "result_json", "tags_json"):
            try:
                result[key.removesuffix("_json")] = json.loads(result.pop(key) or "{}")
            except Exception:
                result[key.removesuffix("_json")] = {} if key != "tags_json" else []
        result["cancel_requested"] = bool(result.get("cancel_requested"))
        return result


class WorkdaySupervisor:
    """Runs queued ATF jobs in isolated, cancellable C0d3rV2 workers."""

    def __init__(self, config: WorkdayConfig | None = None) -> None:
        self.config = config or WorkdayConfig.from_env()
        self.store = WorkdayStore(self.config.db_path)
        self.owner = f"supervisor-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.children: dict[str, dict[str, Any]] = {}
        self.started_at = time.time()
        self.stop_reason = ""
        log_path = self.config.notification_log or self.config.db_path.parent / "notifications.jsonl"
        self.notifier = WorkdayNotifier(log_path, enabled=self.config.notifications_enabled)

    def run(self, *, until_empty: bool = False, max_runtime_seconds: float | None = None) -> dict:
        from services.guardian_lock import GuardianLease

        lease = GuardianLease("agent-the-freeloader-workday", timeout=0.2, poll_interval=0.05)
        if not lease.acquire():
            self.stop_reason = "another ATF workday supervisor holds the lease"
            self.notifier.send("ATF supervisor not started", self.stop_reason, level="warning")
            return self.write_report()
        max_runtime = max_runtime_seconds
        if max_runtime is None and self.config.shift_hours:
            max_runtime = self.config.shift_hours * 3600.0
        try:
            self._recover_orphans()
            pending_budget_stop = ""
            while True:
                self._poll_children()
                if pending_budget_stop and not self.children:
                    self.stop_reason = pending_budget_stop
                    self.notifier.send("ATF paused for budget", self.stop_reason, level="warning")
                    break
                if max_runtime and time.time() - self.started_at >= max_runtime:
                    self.stop_reason = "shift duration reached"
                    break
                budget = self.budget_status()
                if budget["exhausted"]:
                    pending_budget_stop = budget["reason"]
                    if self.children:
                        # Reservations made by an in-flight atomic job can push
                        # the rolling budget over its threshold. Killing that
                        # child discards its route, validation, and checkpoint
                        # after already consuming provider quota. Drain it, but
                        # do not claim another job.
                        time.sleep(self.config.poll_seconds)
                        continue
                    self.stop_reason = pending_budget_stop
                    self.notifier.send("ATF paused for budget", self.stop_reason, level="warning")
                    break
                while not pending_budget_stop and len(self.children) < self.config.concurrency:
                    job = self.store.claim(self.owner, self.config.lease_seconds)
                    if not job:
                        break
                    self._launch(job)
                if until_empty and not self.children and not self._has_runnable_jobs():
                    self.stop_reason = "queue drained"
                    self.notifier.send("ATF queue complete", "All queued jobs reached a terminal state.")
                    break
                time.sleep(self.config.poll_seconds)
        except KeyboardInterrupt:
            self.stop_reason = "interrupted"
        finally:
            self._shutdown_children()
            lease.release()
        return self.write_report()

    def budget_status(self) -> dict:
        ledger = self._quota_ledger()
        usage = ledger.usage_since(86_400.0)
        reasons: list[str] = []
        if self.config.max_requests_per_day and usage["requests"] >= self.config.max_requests_per_day:
            reasons.append("daily request budget reached")
        if self.config.max_tokens_per_day and usage["tokens"] >= self.config.max_tokens_per_day:
            reasons.append("daily token budget reached")
        return {"usage": usage, "exhausted": bool(reasons), "reason": "; ".join(reasons)}

    def write_report(self) -> dict:
        report_dir = self.config.report_dir or self.config.db_path.parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        jobs = self.store.list(limit=1000)
        payload = {
            "started_at": self.started_at,
            "finished_at": time.time(),
            "stop_reason": self.stop_reason or "stopped",
            "budget": self.budget_status(),
            "stats": self.store.stats(since=self.started_at),
            "jobs": [
                {
                    "id": job["id"], "status": job["status"],
                    "attempts": job["attempts"], "error": job["error"],
                    "workdir": job["workdir"],
                }
                for job in jobs if float(job["updated_at"]) >= self.started_at
            ],
        }
        json_path = report_dir / f"workday_{stamp}.json"
        md_path = report_dir / f"workday_{stamp}.md"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        lines = [
            "# AgentTheFreeloader Workday Report", "",
            f"- Stop reason: {payload['stop_reason']}",
            f"- Jobs: {payload['stats']['total']}",
            f"- Requests (rolling 24h): {payload['budget']['usage']['requests']}",
            f"- Tokens (rolling 24h): {payload['budget']['usage']['tokens']}", "",
            "## Jobs", "",
        ]
        lines.extend(
            f"- `{job['id']}` — {job['status']} — attempts {job['attempts']}"
            + (f" — {job['error']}" if job["error"] else "")
            for job in payload["jobs"]
        )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        payload["json_report"] = str(json_path)
        payload["markdown_report"] = str(md_path)
        return payload

    def _launch(self, job: dict) -> None:
        command = [
            sys.executable, "-m",
            "tools.c0d3rV2.plugins.agent_the_freeloader.workday_worker",
            "--db", str(self.config.db_path), "--job", job["id"],
            "--owner", self.owner, "--lease-seconds", str(self.config.lease_seconds),
            "--heartbeat-seconds", str(self.config.heartbeat_seconds),
            "--retry-seconds", str(self.config.retry_base_seconds),
            "--quota-retry-seconds", str(self.config.quota_retry_seconds),
        ]
        kwargs: dict[str, Any] = {
            "cwd": str(Path(__file__).resolve().parents[4]),
            "env": {**os.environ, "C0D3R_BACKEND": "freeloader"},
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "start_new_session": os.name != "nt",
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        process = subprocess.Popen(command, **kwargs)
        self.store.set_worker_pid(job["id"], self.owner, process.pid)
        self.children[job["id"]] = {
            "process": process,
            "started_at": time.time(),
            "timeout": int(job.get("timeout_seconds") or self.config.job_timeout_seconds),
        }
        label = Path(str(job.get("workdir") or "job")).name
        self.notifier.send(
            "ATF job started", f"{label} — attempt {job.get('attempts')}/{job.get('max_attempts')}",
            job_id=str(job["id"]),
        )

    def _poll_children(self) -> None:
        for job_id, child in list(self.children.items()):
            process: subprocess.Popen = child["process"]
            job = self.store.get(job_id)
            if not job:
                self._terminate(process)
                self.children.pop(job_id, None)
                continue
            if job["cancel_requested"]:
                self._terminate(process)
                self._mark_cancelled(job_id)
                self.children.pop(job_id, None)
                continue
            if time.time() - child["started_at"] > child["timeout"]:
                self._terminate(process)
                self.store.retry(
                    job_id, self.owner, "job timeout exceeded",
                    self.config.retry_base_seconds * max(1, int(job["attempts"])),
                    job.get("checkpoint") or {},
                )
                self.children.pop(job_id, None)
                continue
            if process.poll() is not None:
                fresh = self.store.get(job_id)
                if fresh and fresh["status"] == "running":
                    self.store.retry(
                        job_id, self.owner,
                        f"worker exited with code {process.returncode}",
                        self.config.retry_base_seconds * max(1, int(fresh["attempts"])),
                        fresh.get("checkpoint") or {},
                    )
                    fresh = self.store.get(job_id)
                if fresh:
                    label = Path(str(fresh.get("workdir") or "job")).name
                    status = str(fresh.get("status") or "unknown")
                    error = str(fresh.get("error") or "").replace("\n", " ")[:180]
                    if status == "completed":
                        self.notifier.send("ATF milestone passed", label, job_id=job_id)
                    elif status == "failed":
                        self.notifier.send(
                            "ATF review needed", f"{label} failed after {fresh.get('attempts')} attempts. {error}",
                            level="error", job_id=job_id,
                        )
                    elif status == "retry":
                        capacity = "no eligible model" in error.lower() or "quota" in error.lower()
                        self.notifier.send(
                            "ATF capacity cooldown" if capacity else "ATF retry scheduled",
                            f"{label} — {error or 'validation failed'}",
                            level="warning", job_id=job_id,
                        )
                self.children.pop(job_id, None)

    def _recover_orphans(self) -> None:
        for job in self.store.reclaim_expired():
            pid = job.get("worker_pid")
            if pid:
                _terminate_owned_pid(int(pid), str(job["id"]))

    def _shutdown_children(self) -> None:
        for job_id, child in list(self.children.items()):
            self._terminate(child["process"])
            job = self.store.get(job_id)
            if job and job["status"] == "running" and job["lease_owner"] == self.owner:
                self.store.retry(
                    job_id, self.owner, self.stop_reason or "supervisor stopped", 0,
                    job.get("checkpoint") or {},
                )
        self.children.clear()

    def _has_runnable_jobs(self) -> bool:
        return any(job["status"] in RUNNABLE_STATES for job in self.store.list(limit=1000))

    def _mark_cancelled(self, job_id: str) -> None:
        now = time.time()
        with self.store._connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status='cancelled', completed_at=?, updated_at=?,
                    lease_owner=NULL, lease_expires_at=NULL, worker_pid=NULL
                WHERE id=?
                """,
                (now, now, job_id),
            )
            self.store._event(connection, job_id, "cancelled", "worker terminated")

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True, timeout=10,
                )
            else:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    @staticmethod
    def _quota_ledger() -> QuotaLedger:
        specs = load_catalog()
        limits = merge_pool_limits(specs)
        configured = os.getenv("AGENT_FREELOADER_STATE_PATH", "").strip()
        path = Path(configured).expanduser() if configured else default_workday_db().parent / "quota.json"
        return QuotaLedger(limits, state_path=path)


def _terminate_owned_pid(pid: int, job_id: str) -> bool:
    """Kill an orphan only when its command line proves it is our worker."""
    try:
        import psutil  # type: ignore
        process = psutil.Process(pid)
        command = " ".join(process.cmdline())
        if "workday_worker" not in command or job_id not in command:
            return False
        for child in process.children(recursive=True):
            child.kill()
        process.kill()
        return True
    except Exception:
        return False
