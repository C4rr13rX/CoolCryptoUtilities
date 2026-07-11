from __future__ import annotations

import json
import os
import argparse
import sqlite3
import subprocess
import time
from pathlib import Path


class WorkdayNotifier:
    """Best-effort desktop landmarks with an append-only audit fallback."""

    def __init__(self, log_path: str | Path, *, enabled: bool = True) -> None:
        self.log_path = Path(log_path)
        self.enabled = enabled

    def send(self, title: str, message: str, *, level: str = "info", job_id: str = "") -> bool:
        event = {
            "timestamp": time.time(), "title": title, "message": message,
            "level": level, "job_id": job_id,
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass
        if not self.enabled or os.name != "nt":
            return False
        safe_title = title.replace("'", "''")[:80]
        safe_message = message.replace("'", "''").replace("\r", " ").replace("\n", " ")[:240]
        icon = "Warning" if level in {"warning", "error"} else "Info"
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "Add-Type -AssemblyName System.Drawing;"
            "$n=New-Object System.Windows.Forms.NotifyIcon;"
            "$n.Icon=[System.Drawing.SystemIcons]::Information;"
            f"$n.BalloonTipIcon=[System.Windows.Forms.ToolTipIcon]::{icon};"
            f"$n.BalloonTipTitle='{safe_title}';"
            f"$n.BalloonTipText='{safe_message}';"
            "$n.Visible=$true;$n.ShowBalloonTip(8000);Start-Sleep -Seconds 9;$n.Dispose()"
        )
        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return True
        except OSError:
            return False


def monitor_database(
    db_path: str | Path, *, log_path: str | Path, poll_seconds: float = 5.0,
    hours: float = 24.0, enabled: bool = True, progress_seconds: float = 600.0,
) -> None:
    """Watch an active workday database without taking the supervisor lease."""
    db_path = Path(db_path)
    notifier = WorkdayNotifier(log_path, enabled=enabled)
    seen: dict[str, tuple[str, int, str]] = {}
    started = time.time()
    drained_sent = False
    first_scan = True
    progress_sent: dict[str, float] = {}
    while time.time() - started < max(0.01, hours) * 3600:
        try:
            with sqlite3.connect(db_path, timeout=5) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    "SELECT id,status,attempts,max_attempts,error,workdir FROM jobs ORDER BY created_at"
                ).fetchall()
        except sqlite3.Error:
            time.sleep(max(1.0, poll_seconds))
            continue
        active = False
        for row in rows:
            job_id = str(row["id"])
            status = str(row["status"])
            attempts = int(row["attempts"])
            error = str(row["error"] or "")
            fingerprint = (status, attempts, error[:300])
            if status in {"queued", "retry", "running"}:
                active = True
            if seen.get(job_id) == fingerprint:
                if status == "running" and time.time() - progress_sent.get(job_id, 0.0) >= progress_seconds:
                    workdir = Path(str(row["workdir"] or ""))
                    try:
                        artifact_count = sum(1 for path in workdir.rglob("*") if path.is_file())
                    except OSError:
                        artifact_count = 0
                    notifier.send(
                        "ATF still working" if artifact_count else "ATF stalled: no artifacts yet",
                        f"{workdir.name or 'job'} — attempt {attempts}/{row['max_attempts']}; {artifact_count} artifact files",
                        level="info" if artifact_count else "warning", job_id=job_id,
                    )
                    progress_sent[job_id] = time.time()
                continue
            seen[job_id] = fingerprint
            label = Path(str(row["workdir"] or "job")).name
            if first_scan and status in {"completed", "failed", "cancelled"}:
                continue
            if status == "running":
                notifier.send("ATF job running", f"{label} — attempt {attempts}/{row['max_attempts']}", job_id=job_id)
                progress_sent[job_id] = time.time()
            elif status == "completed":
                notifier.send("ATF milestone passed", label, job_id=job_id)
            elif status == "failed":
                notifier.send("ATF review needed", f"{label} — {error[:180]}", level="error", job_id=job_id)
            elif status == "retry" and attempts:
                capacity = "quota" in error.lower() or "no eligible model" in error.lower()
                notifier.send(
                    "ATF capacity cooldown" if capacity else "ATF retry scheduled",
                    f"{label} — {error[:180] or 'validation failed'}", level="warning", job_id=job_id,
                )
        if rows and not active and not drained_sent:
            notifier.send("ATF queue complete", "All monitored jobs reached a terminal state.")
            drained_sent = True
        elif active:
            drained_sent = False
        first_scan = False
        time.sleep(max(1.0, poll_seconds))


def main() -> int:
    parser = argparse.ArgumentParser(description="Desktop notifications for an ATF workday database")
    parser.add_argument("--db", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--progress", type=float, default=600.0)
    parser.add_argument("--no-desktop", action="store_true")
    args = parser.parse_args()
    monitor_database(
        args.db, log_path=args.log, poll_seconds=args.poll,
        hours=args.hours, enabled=not args.no_desktop, progress_seconds=max(60.0, args.progress),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
