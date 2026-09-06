"""SmartStorageService — market data that is local when it is needed and remote when it is not.

WHY THIS EXISTS
---------------
Measured 2026-09-06, this box ran out of disk: D: hit **0 bytes free of 894 GB**
and the wizard brain's checkpoint began failing every 30 minutes, leaving a
179-byte truncated stub where a 15 GB brain should have been. The node then
restarted into an empty fabric and its prediction confidence fell from 0.278
to 0.000057.

Nothing about that was a market-data problem, and market data caused it. The
project holds 102 GB across ``storage/`` and ``data/`` — 28 GB of it news and
OHLCV that grows ~51 MB every day and is never deleted.

THE DESIGN QUESTION, ANSWERED IN THE ORDER IT WAS ASKED
-------------------------------------------------------
**Q1: what must be local for ZERO latency, at the smallest possible footprint?**

Not a guess, and not an LRU. Market data has a property ordinary caches do not:
*a historical bar is immutable*. The 14:00 bar for 2024-03-01 will never change
again. So the only data that must be resident is the data that will be read
NEXT, and for this system that quantity is deterministic rather than
probabilistic — the trading loop reads a bounded window per active symbol on a
fixed cadence:

    active symbols (24h)     26
    BOT_WINDOW_SIZE          60 bars
    bytes per bar (measured) 217

    HOT SET = 26 x 60 x 217 = **0.34 MB**

Against 2914 MB of OHLCV currently on disk, the hot set is **0.012%** of it —
a factor of about 8500. That is the whole insight this module is built on:
zero-latency operation does not require the archive, it requires the tail.

**Q2: given Q1, what is the lowest transfer cost?**

The trap is per-request pricing, not bandwidth. Reading bar-by-bar from S3
would be 449,280 GETs/day ($5.39/mo); reading one object per symbol per day is
780 GETs/mo ($0.0003) — **17,000x cheaper for identical data**. So objects are
written as SEGMENTS (one per symbol per period), never per record.

But Q1 makes the stronger point: the hot set is local, so the steady-state
read cost is *zero requests*. S3 is touched only to sync new data up (S3 never
charges for ingress) and to fetch history on demand for backfill or replay.

    one-time upload of 28 GB   $0.00 ingress + $0.03 in PUTs
    steady storage             $0.65/mo Standard -> $0.11/mo after tiering
    daily sync of 51 MB        $0.00

**Q3: given Q1 and Q2, what is the smallest local footprint that is still
safe?** A cache that can evict is a cache that can lose the only copy of
something. Eviction here is therefore permitted ONLY for a key that is
confirmed durable in S3 (``_is_durable``), so the worst case of an over-eager
cache is a slow read, never a missing one. In local-only mode nothing is ever
evicted, because there is nowhere for it to have gone.

CONFIGURATION
-------------
Two independent switches, both default OFF so that a fresh clone and every
existing deployment keep behaving exactly as they do today:

    MARKET_DATA_S3_ENABLED=0|1     sync market data + news to S3
    ACCOUNT_DATA_S3_ENABLED=0|1    sync account/wallet data (leave 0)
    MARKET_DATA_S3_BUCKET=...      bucket name
    MARKET_DATA_S3_PROFILE=...     boto3 profile (per-developer, never committed)
    MARKET_DATA_S3_REGION=...      default us-east-1
    SMART_STORAGE_LOCAL_BUDGET_MB  local ceiling before eviction considers running

The profile name is read from the environment precisely so that no developer's
AWS identity ends up in the repository. A second developer points
MARKET_DATA_S3_PROFILE at their own profile and their own bucket, and the same
code serves them.

FAILURE POSTURE
---------------
Every remote path fails OPEN to local. If boto3 is missing, credentials are
absent, the bucket is unreachable or a key does not exist, the caller gets the
local answer or None — never an exception that stops a trading loop. Losing a
cache is recoverable; stalling the money path is not.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from services.logging_service import log_message
except Exception:  # noqa: BLE001 - logging must never block storage
    def log_message(*_args, **_kwargs):  # type: ignore[misc]
        return None

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Default local roots. Overridable so a deployment can move them wholesale.
DEFAULT_ROOTS: Dict[str, str] = {
    "market": "data/historical_ohlcv",
    "news": "data/news",
    "news_cache": "storage/news_cache",
    "intermediate": "data/intermediate",
}


def _flag(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in {
        "1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


@dataclass
class StorageConfig:
    """Everything that decides where a byte lives.

    Constructed from the environment so that the same code runs local-only on
    a laptop, S3-backed on this box, and fully remote in a Lambda, with no code
    change and no credentials in the repository.
    """

    market_s3_enabled: bool = field(
        default_factory=lambda: _flag("MARKET_DATA_S3_ENABLED"))
    account_s3_enabled: bool = field(
        default_factory=lambda: _flag("ACCOUNT_DATA_S3_ENABLED"))
    bucket: str = field(
        default_factory=lambda: os.getenv("MARKET_DATA_S3_BUCKET", "").strip())
    profile: str = field(
        default_factory=lambda: os.getenv("MARKET_DATA_S3_PROFILE", "").strip())
    region: str = field(
        default_factory=lambda: os.getenv("MARKET_DATA_S3_REGION", "us-east-1"))
    #: Local ceiling in MB. Eviction is only ever CONSIDERED above this, and
    #: only ever ACTED on for keys confirmed durable in S3.
    local_budget_mb: int = field(
        default_factory=lambda: _int_env("SMART_STORAGE_LOCAL_BUDGET_MB", 4096))
    #: Bars per symbol that must stay resident. This is the Q1 quantity: it is
    #: what the trading loop reads, so it is what "zero latency" costs.
    hot_window_bars: int = field(
        default_factory=lambda: _int_env("BOT_WINDOW_SIZE", 60))

    @property
    def remote_enabled(self) -> bool:
        """Remote is on only when it is switched on AND actually configured.

        A half-configured remote (flag on, no bucket) must behave exactly like
        local-only rather than raising on the first read.
        """
        return bool(self.market_s3_enabled and self.bucket)


class SmartStorageService:
    """Reads and writes market data without the caller knowing where it lives.

    The contract is deliberately small -- ``get``, ``put``, ``exists``,
    ``sync_up`` -- because every call site that currently touches a path has
    to be able to adopt it without restructuring.
    """

    def __init__(self, config: Optional[StorageConfig] = None,
                 roots: Optional[Dict[str, str]] = None) -> None:
        self.config = config or StorageConfig()
        self.roots = dict(DEFAULT_ROOTS)
        if roots:
            self.roots.update(roots)
        self._lock = threading.RLock()
        self._client = None
        self._client_failed = False
        #: Keys confirmed present in S3. Eviction consults this and nothing
        #: else, so a key can never be evicted on an assumption.
        self._durable: Dict[str, float] = {}
        self._stats: Dict[str, int] = {
            "local_hits": 0, "remote_hits": 0, "misses": 0,
            "uploads": 0, "downloads": 0, "evictions": 0, "errors": 0,
        }

    # -- placement -----------------------------------------------------------
    def local_path(self, domain: str, key: str) -> Path:
        """Where ``key`` lives on this machine.

        Keys are relative POSIX-ish paths under a domain root. Traversal is
        rejected rather than normalised: a key that escapes its root is a bug
        at the call site, and silently repairing it would hide the bug.
        """
        root = self.roots.get(domain)
        if not root:
            raise ValueError(f"unknown storage domain: {domain!r}")
        cleaned = str(key).replace("\\", "/").strip("/")
        if not cleaned or ".." in cleaned.split("/"):
            raise ValueError(f"unsafe storage key: {key!r}")
        base = (REPO_ROOT / root).resolve()
        target = (base / cleaned).resolve()
        if base != target and base not in target.parents:
            raise ValueError(f"key escapes its domain root: {key!r}")
        return target

    def remote_key(self, domain: str, key: str) -> str:
        """The S3 key for the same logical object.

        Prefixed by domain so the bucket lifecycle rules (which tier ``market/``
        and ``news/`` to IA at 30d and Glacier-IR at 120d) apply cleanly.
        """
        cleaned = str(key).replace("\\", "/").strip("/")
        return f"{domain}/{cleaned}"

    # -- remote client -------------------------------------------------------
    def _s3(self):
        """A boto3 client, or None.

        None is a normal outcome, not an error: it means "run local-only",
        which is the correct behaviour when boto3 is absent, the profile does
        not exist, or credentials have not been configured. The failure is
        recorded once so a misconfiguration is visible without spamming a
        trading loop that calls this thousands of times.
        """
        if not self.config.remote_enabled or self._client_failed:
            return None
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                import boto3  # noqa: PLC0415 - optional dependency by design
                if self.config.profile:
                    session = boto3.Session(profile_name=self.config.profile,
                                            region_name=self.config.region)
                else:
                    session = boto3.Session(region_name=self.config.region)
                self._client = session.client("s3")
                return self._client
            except Exception as exc:  # noqa: BLE001
                self._client_failed = True
                self._stats["errors"] += 1
                log_message(
                    "storage",
                    f"S3 unavailable, continuing local-only: "
                    f"{type(exc).__name__}: {exc}",
                    severity="warning")
                return None

    # -- reads ---------------------------------------------------------------
    def get(self, domain: str, key: str) -> Optional[bytes]:
        """Bytes for ``key``, local first, then remote, else None.

        Local first is not merely an optimisation -- it is the Q1 guarantee.
        The hot set is resident, so the steady-state read path never leaves
        this machine and never costs a request.
        """
        try:
            path = self.local_path(domain, key)
        except ValueError:
            self._stats["errors"] += 1
            return None

        if path.exists():
            try:
                self._stats["local_hits"] += 1
                return path.read_bytes()
            except OSError:
                self._stats["errors"] += 1  # fall through to remote

        client = self._s3()
        if client is None:
            self._stats["misses"] += 1
            return None

        try:
            obj = client.get_object(Bucket=self.config.bucket,
                                    Key=self.remote_key(domain, key))
            body = obj["Body"].read()
        except Exception:  # noqa: BLE001 - a missing key is not an error here
            self._stats["misses"] += 1
            return None

        self._stats["remote_hits"] += 1
        self._stats["downloads"] += 1
        self._mark_durable(domain, key)
        # Re-hydrate locally: a key we just paid to fetch is by definition one
        # this machine wanted, so the next read should be free.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        except OSError:
            pass  # a full disk must not turn a successful read into a failure
        return body

    def exists(self, domain: str, key: str) -> bool:
        """True if the object is readable from anywhere."""
        try:
            if self.local_path(domain, key).exists():
                return True
        except ValueError:
            return False
        client = self._s3()
        if client is None:
            return False
        try:
            client.head_object(Bucket=self.config.bucket,
                               Key=self.remote_key(domain, key))
            self._mark_durable(domain, key)
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- writes --------------------------------------------------------------
    def put(self, domain: str, key: str, data: bytes,
            *, sync: Optional[bool] = None) -> bool:
        """Write locally, and to S3 when remote sync is on.

        The local write is authoritative and happens first: if the upload
        fails, the data still exists and the next ``sync_up`` will carry it.
        The reverse order would risk acknowledging a write that is nowhere.
        """
        try:
            path = self.local_path(domain, key)
        except ValueError:
            self._stats["errors"] += 1
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write through a temp file in the same directory so a crash
            # mid-write leaves the PREVIOUS version intact rather than a
            # truncated stub -- the exact failure that cost this project its
            # brain checkpoint on 2026-09-06.
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except OSError as exc:
            self._stats["errors"] += 1
            log_message("storage", f"local write failed for {key}: {exc}",
                        severity="error")
            return False

        should_sync = self.config.remote_enabled if sync is None else sync
        if should_sync:
            self._upload(domain, key, data)
        return True

    def _upload(self, domain: str, key: str, data: bytes) -> bool:
        client = self._s3()
        if client is None:
            return False
        try:
            client.put_object(Bucket=self.config.bucket,
                              Key=self.remote_key(domain, key),
                              Body=data)
            self._stats["uploads"] += 1
            self._mark_durable(domain, key)
            return True
        except Exception as exc:  # noqa: BLE001
            self._stats["errors"] += 1
            log_message("storage", f"upload failed for {key}: {exc}",
                        severity="warning")
            return False

    # -- durability tracking -------------------------------------------------
    def _mark_durable(self, domain: str, key: str) -> None:
        with self._lock:
            self._durable[f"{domain}/{key}"] = time.time()

    def _is_durable(self, domain: str, key: str) -> bool:
        """Has this exact key been confirmed to exist remotely?

        Eviction depends entirely on this answer, so it is deliberately
        conservative: a key is durable only if THIS process uploaded it or
        read it back from S3. An unverified assumption here would delete the
        only copy of something.
        """
        with self._lock:
            if f"{domain}/{key}" in self._durable:
                return True
        client = self._s3()
        if client is None:
            return False
        try:
            client.head_object(Bucket=self.config.bucket,
                               Key=self.remote_key(domain, key))
            self._mark_durable(domain, key)
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- footprint management ------------------------------------------------
    def local_footprint_mb(self, domain: Optional[str] = None) -> float:
        domains = [domain] if domain else list(self.roots)
        total = 0
        for name in domains:
            root = REPO_ROOT / self.roots.get(name, "")
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    try:
                        total += path.stat().st_size
                    except OSError:
                        continue
        return total / 1e6

    def reclaim(self, domain: str, *, dry_run: bool = True) -> Dict[str, Any]:
        """Evict cold local copies that are safely in S3.

        Oldest-first by modification time, and ONLY for keys confirmed durable.
        Nothing is evicted in local-only mode: with no remote, every local file
        is the only copy, and a cache that deletes the last copy of something
        is not a cache.

        Defaults to ``dry_run`` because the caller should see what would go
        before anything goes.
        """
        result: Dict[str, Any] = {
            "domain": domain, "dry_run": dry_run, "candidates": 0,
            "freed_mb": 0.0, "skipped_not_durable": 0, "evicted": [],
        }
        if not self.config.remote_enabled:
            result["reason"] = "remote disabled: local copies are the only copies"
            return result

        budget_mb = self.config.local_budget_mb
        current = self.local_footprint_mb(domain)
        if current <= budget_mb:
            result["reason"] = f"{current:.1f} MB is within the {budget_mb} MB budget"
            return result

        root = REPO_ROOT / self.roots.get(domain, "")
        files = sorted(
            (p for p in root.rglob("*") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        to_free = (current - budget_mb) * 1e6
        freed = 0.0
        for path in files:
            if freed >= to_free:
                break
            key = path.relative_to(root).as_posix()
            result["candidates"] += 1
            if not self._is_durable(domain, key):
                result["skipped_not_durable"] += 1
                continue
            size = path.stat().st_size
            if not dry_run:
                try:
                    path.unlink()
                    self._stats["evictions"] += 1
                except OSError:
                    continue
            freed += size
            result["evicted"].append(key)
        result["freed_mb"] = round(freed / 1e6, 2)
        return result

    # -- bulk sync -----------------------------------------------------------
    def sync_up(self, domain: str, *, limit: Optional[int] = None,
                dry_run: bool = False) -> Dict[str, Any]:
        """Upload local objects that are not yet in S3.

        Skips anything already confirmed durable, so re-running is cheap and
        an interrupted sync resumes rather than restarting.
        """
        out: Dict[str, Any] = {
            "domain": domain, "uploaded": 0, "skipped": 0,
            "bytes": 0, "errors": 0, "dry_run": dry_run,
        }
        if not self.config.remote_enabled:
            out["reason"] = "remote disabled"
            return out
        root = REPO_ROOT / self.roots.get(domain, "")
        if not root.exists():
            out["reason"] = f"no local root at {root}"
            return out

        for path in root.rglob("*"):
            if not path.is_file() or path.suffix == ".tmp":
                continue
            if limit and out["uploaded"] >= limit:
                break
            key = path.relative_to(root).as_posix()
            if self._is_durable(domain, key):
                out["skipped"] += 1
                continue
            if dry_run:
                out["uploaded"] += 1
                out["bytes"] += path.stat().st_size
                continue
            try:
                data = path.read_bytes()
            except OSError:
                out["errors"] += 1
                continue
            if self._upload(domain, key, data):
                out["uploaded"] += 1
                out["bytes"] += len(data)
            else:
                out["errors"] += 1
        return out

    def stats(self) -> Dict[str, Any]:
        """Counters plus the configuration that produced them."""
        with self._lock:
            data = dict(self._stats)
        data.update({
            "remote_enabled": self.config.remote_enabled,
            "bucket": self.config.bucket or None,
            "region": self.config.region,
            "durable_keys_known": len(self._durable),
            "hot_window_bars": self.config.hot_window_bars,
            "local_budget_mb": self.config.local_budget_mb,
        })
        return data


_default: Optional[SmartStorageService] = None
_default_lock = threading.Lock()


def get_storage() -> SmartStorageService:
    """Process-wide service.

    A single instance so the durable-key map and the boto3 client are shared;
    building one per call would re-probe S3 on every read.
    """
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = SmartStorageService()
    return _default


def reset_storage() -> None:
    """Drop the singleton. Tests use this after changing the environment."""
    global _default
    with _default_lock:
        _default = None
