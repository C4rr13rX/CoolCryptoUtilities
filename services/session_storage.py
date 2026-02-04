from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


@dataclass
class SessionConfig:
    bucket: str
    region: str = "us-east-1"
    prefix: str = "sessions"
    mode: str = "auto"  # auto | local | s3


class SessionStore:
    def __init__(self, config: SessionConfig, *, profile: Optional[str] = None) -> None:
        self.config = config
        self.profile = profile
        self._s3 = None
        self._s3_ready = False
        base_root = os.getenv("C0D3R_STORAGE_ROOT")
        if base_root:
            self._base_dir = Path(base_root).expanduser().resolve()
        else:
            project_root = Path(__file__).resolve().parents[1]
            self._base_dir = (project_root / "storage" / "c0d3r_sessions").resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._local_next = self._base_dir / "next.txt"

    def ensure_backend(self) -> None:
        if self.config.mode == "local":
            return
        if self._s3_ready:
            return
        try:
            session = boto3.Session(profile_name=self.profile, region_name=self.config.region) if self.profile else boto3.Session(region_name=self.config.region)
            self._s3 = session.client("s3")
            bucket = self.config.bucket
            if not bucket:
                return
            self._ensure_bucket(bucket)
            self._s3_ready = True
            self._sync_local_pending()
        except (NoCredentialsError, ClientError, Exception):
            self._s3_ready = False

    def _ensure_bucket(self, bucket: str) -> None:
        try:
            self._s3.head_bucket(Bucket=bucket)
            return
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code") or "")
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                return
        if self.config.region == "us-east-1":
            self._s3.create_bucket(Bucket=bucket)
        else:
            self._s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": self.config.region},
            )

    def next_session_id(self) -> int:
        self.ensure_backend()
        if self._s3_ready:
            try:
                key = self._s3_key("next.txt")
                obj = self._s3.get_object(Bucket=self.config.bucket, Key=key)
                value = obj["Body"].read().decode("utf-8", errors="ignore").strip()
                return int(value) if value else 1
            except Exception:
                pass
        return self._read_local_next()

    def update_next_session_id(self, next_id: int) -> None:
        self._write_local_next(next_id)
        self.ensure_backend()
        if self._s3_ready:
            try:
                key = self._s3_key("next.txt")
                self._s3.put_object(Bucket=self.config.bucket, Key=key, Body=str(next_id).encode("utf-8"))
            except Exception:
                pass

    def append_event(self, session_id: int, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        local_path = self._local_session_path(session_id)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self.ensure_backend()
        if self._s3_ready:
            try:
                key = self._s3_key(f"{session_id}.jsonl")
                self._s3.put_object(Bucket=self.config.bucket, Key=key, Body=local_path.read_bytes())
                marker = local_path.with_suffix(".uploaded")
                marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            except Exception:
                pass

    def _sync_local_pending(self) -> None:
        if not self._s3_ready:
            return
        for path in self._base_dir.glob("*.jsonl"):
            marker = path.with_suffix(".uploaded")
            if marker.exists():
                continue
            try:
                key = self._s3_key(path.name)
                self._s3.put_object(Bucket=self.config.bucket, Key=key, Body=path.read_bytes())
                marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            except Exception:
                continue

    def _local_session_path(self, session_id: int) -> Path:
        return self._base_dir / f"{session_id}.jsonl"

    def _s3_key(self, name: str) -> str:
        prefix = (self.config.prefix or "sessions").strip("/")
        return f"{prefix}/{name}"

    def _read_local_next(self) -> int:
        if not self._local_next.exists():
            self._write_local_next(1)
            return 1
        try:
            value = self._local_next.read_text(encoding="utf-8").strip()
            return int(value) if value else 1
        except Exception:
            self._write_local_next(1)
            return 1

    def _write_local_next(self, next_id: int) -> None:
        self._local_next.write_text(str(next_id), encoding="utf-8")


def default_session_config() -> SessionConfig:
    raw_bucket = os.getenv("C0D3R_SESSION_BUCKET", "c0dersessions")
    bucket = _sanitize_bucket(raw_bucket)
    region = os.getenv("C0D3R_SESSION_REGION", "us-east-1")
    prefix = os.getenv("C0D3R_SESSION_PREFIX", "sessions")
    mode = os.getenv("C0D3R_SESSION_MODE", "auto")
    return SessionConfig(bucket=bucket, region=region, prefix=prefix, mode=mode)


def _sanitize_bucket(name: str) -> str:
    cleaned = (name or "").strip().lower()
    cleaned = cleaned.replace("_", "-")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch in "-.")
    cleaned = cleaned.strip(".-")
    return cleaned or "c0dersessions"
