from __future__ import annotations

import os
import sqlite3
import time
import json
from pathlib import Path


def default_feedback_path() -> Path:
    configured = os.getenv("AGENT_FREELOADER_FEEDBACK_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[4] / "runtime" / "agent_the_freeloader" / "feedback.sqlite3"


class ModelFeedbackStore:
    """Persistent semantic-quality feedback shared by every ATF process."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_feedback_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15.0)
        connection.execute("PRAGMA busy_timeout=15000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS model_feedback (
                    identity TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    successes INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    last_reason TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS correction_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    session_name TEXT NOT NULL DEFAULT '',
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    is_hallucination INTEGER NOT NULL DEFAULT 0,
                    trigger TEXT NOT NULL DEFAULT '',
                    failed_output TEXT NOT NULL DEFAULT '',
                    correction TEXT NOT NULL DEFAULT '',
                    resolved INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )

    def record(self, provider: str, model_id: str, *, success: bool, reason: str = "") -> None:
        identity = f"{provider}:{model_id}"
        successes = 1 if success else 0
        failures = 0 if success else 1
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_feedback
                    (identity, provider, model_id, successes, failures, last_reason, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(identity) DO UPDATE SET
                    successes = successes + excluded.successes,
                    failures = failures + excluded.failures,
                    last_reason = excluded.last_reason,
                    updated_at = excluded.updated_at
                """,
                (identity, provider, model_id, successes, failures, reason[:1000], time.time()),
            )

    def factor(self, identity: str) -> float:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT successes, failures FROM model_feedback WHERE identity = ?",
                (identity,),
            ).fetchone()
        if not row:
            return 1.0
        successes, failures = int(row[0]), int(row[1])
        evidence = successes + failures
        # A small prior tolerates one noisy observation, while repeated empty,
        # invalid, or failed responses quickly stop consuming benchmark time.
        return max(0.15, min(1.35, 1.0 + (successes - failures) / (evidence + 2)))

    def record_correction(
        self,
        provider: str,
        model_id: str,
        *,
        session_name: str = "",
        classification: str,
        is_hallucination: bool,
        trigger: str,
        failed_output: str = "",
        correction: str = "",
        resolved: bool = False,
        metadata: dict | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO correction_events
                    (created_at, session_name, provider, model_id, classification,
                     is_hallucination, trigger, failed_output, correction, resolved,
                     metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(), session_name[:200], provider[:200], model_id[:300],
                    classification[:100], 1 if is_hallucination else 0,
                    trigger[:4000], failed_output[:8000], correction[:8000],
                    1 if resolved else 0,
                    json.dumps(metadata or {}, default=str)[:12000],
                ),
            )
            return int(cursor.lastrowid)

    def correction_snapshot(self, limit: int = 500) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, session_name, provider, model_id,
                       classification, is_hallucination, trigger, failed_output,
                       correction, resolved, metadata_json
                FROM correction_events ORDER BY id DESC LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [
            {
                "id": row[0], "created_at": row[1], "session": row[2],
                "provider": row[3], "model": row[4], "classification": row[5],
                "is_hallucination": bool(row[6]), "trigger": row[7],
                "failed_output": row[8], "correction": row[9],
                "resolved": bool(row[10]), "metadata": _json_dict(row[11]),
            }
            for row in rows
        ]

    def resolve_correction(self, event_id: int, correction: str) -> bool:
        """Mark a previously recorded correction as resolved after validation passes."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE correction_events
                SET resolved=1, correction=?
                WHERE id=?
                """,
                (correction[:8000], int(event_id)),
            )
            return cursor.rowcount == 1

    def snapshot(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT identity, provider, model_id, successes, failures,
                       last_reason, updated_at
                FROM model_feedback ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            {
                "identity": row[0],
                "provider": row[1],
                "model": row[2],
                "successes": row[3],
                "failures": row[4],
                "last_reason": row[5],
                "updated_at": row[6],
                "factor": self.factor(row[0]),
            }
            for row in rows
        ]


def _json_dict(raw: str) -> dict:
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}
