"""
S3-as-database storage engine — the server half of the hybrid model.

This is the Python port of C4rr13rX's ``aws/private-api`` layer.  It is
deliberately **stateless and cache-free**: the local, queryable tier lives in
the browser (AllezORM over IndexedDB), not here.

    Browser: AllezORM / IndexedDB     <- the local tier: real SQL, offline,
       |                                 durable across sessions and devices
       |  REST over API Gateway
       v
    Lambda: this module (stateless)
       |
       v
    S3: database/tables/<table>/<id>.json   <- durable source of truth

Why nothing is cached server-side
---------------------------------
An earlier draft kept a SQLite mirror in ``/tmp``.  That is the wrong shape: a
Lambda sandbox is per-invocation and disposable, so such a cache dies on every
cold start, splits into N inconsistent copies under concurrency, and buys
nothing the browser tier does not already do better.  It also inverts the
point of the architecture -- the client is supposed to own the fast path so the
server can stay thin and cheap.

So the only state this process keeps is a per-invocation memo (``_memo``),
which exists purely to avoid re-reading the same object twice while serving
one request.  It is dropped between invocations by design.

Cost
----
This is the cheapest durable option on AWS.  RDS bills per hour whether or not
traffic arrives (~$15-30/month minimum) for a dataset of ~1,400 rows.  S3
bills per request and per GB: the same workload costs cents, scales to zero,
and needs no VPC, no NAT gateway, and no RDS Proxy.

Consistency
-----------
S3 is strongly consistent per object, but this layer has **no cross-object
transactions**.  Concurrent writers can observe each other mid-update.  That
is acceptable for the configuration/admin tables this serves, and is the same
trade C4rr13rX makes.  What is guaranteed:

* a mutation reaches S3 before it is acknowledged, so an accepted write
  survives the sandbox dying;
* ``total.txt`` allocates ids under a compare-and-set retry loop, so two
  concurrent inserts cannot land on the same id;
* ``change.txt`` carries a monotonic sequence so the browser tier can tell
  whether its IndexedDB snapshot is stale without refetching every row.

A table needing multi-row atomicity does not belong here.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("serverless.hybrid.smart_storage")

# Mirrors C4rr13rX's BASE_PREFIX so one bucket can host both projects.
BASE_PREFIX = os.getenv("HYBRID_BASE_PREFIX", "database/tables")


@dataclass
class StorageStats:
    """Per-invocation counters, surfaced by the health endpoint."""

    s3_gets: int = 0
    s3_puts: int = 0
    memo_hits: int = 0
    bytes_down: int = 0
    bytes_up: int = 0

    def as_dict(self) -> dict:
        return {
            "s3_gets": self.s3_gets,
            "s3_puts": self.s3_puts,
            "memo_hits": self.memo_hits,
            "bytes_down": self.bytes_down,
            "bytes_up": self.bytes_up,
        }


class SmartStorage:
    """
    Stateless S3-backed record store.

    One instance per process is fine -- it holds no durable state. Call
    ``reset_memo()`` at the start of each invocation so a warm sandbox never
    serves a value written by a different request.
    """

    def __init__(
        self,
        bucket: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        self.bucket = bucket or os.getenv("HYBRID_BUCKET", "")
        self.endpoint_url = endpoint_url or os.getenv("AWS_S3_ENDPOINT_URL") or None
        self.stats = StorageStats()
        self._memo: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._client = None

    # -- lifecycle --------------------------------------------------------
    def reset_memo(self) -> None:
        """
        Drop the per-invocation memo.

        Must run at the top of every handler. A warm sandbox otherwise
        returns a row cached during a previous request, which would surface as
        a stale read that only ever reproduces under load.
        """
        with self._lock:
            self._memo.clear()
            self.stats = StorageStats()

    # -- S3 plumbing ------------------------------------------------------
    @property
    def client(self):
        """Lazy so importing this module never requires credentials."""
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                # Fail fast: a hung S3 call otherwise burns the whole API
                # Gateway budget and returns 504 instead of a useful error.
                config=Config(
                    retries={"max_attempts": 3, "mode": "standard"},
                    connect_timeout=3,
                    read_timeout=5,
                ),
            )
        return self._client

    def key_for(self, table: str, ident: Any) -> str:
        return f"{BASE_PREFIX}/{table}/{ident}.json"

    def get_json(self, key: str) -> Any | None:
        """Read one object; None when absent (absence is not an error)."""
        with self._lock:
            if key in self._memo:
                self.stats.memo_hits += 1
                return self._memo[key]
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
            raw = resp["Body"].read()
        except Exception as exc:  # noqa: BLE001
            if _is_missing(exc):
                return None
            logger.warning("s3 get failed key=%s: %s", key, exc)
            raise
        self.stats.s3_gets += 1
        self.stats.bytes_down += len(raw)
        value = json.loads(raw.decode("utf-8"))
        with self._lock:
            self._memo[key] = value
        return value

    def put_json(self, key: str, value: Any) -> None:
        body = json.dumps(value, default=str, separators=(",", ":")).encode()
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=body,
            ContentType="application/json",
        )
        self.stats.s3_puts += 1
        self.stats.bytes_up += len(body)
        with self._lock:
            self._memo[key] = value

    def get_text(self, key: str) -> str | None:
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            if _is_missing(exc):
                return None
            raise
        self.stats.s3_gets += 1
        return resp["Body"].read().decode("utf-8").strip()

    def put_text(self, key: str, value: str) -> None:
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=value.encode(),
            ContentType="text/plain",
        )
        self.stats.s3_puts += 1

    # -- id allocation ----------------------------------------------------
    def next_id(self, table: str) -> int:
        """
        Allocate the next id, mirroring C4rr13rX's ``total.txt``.

        A bare read-then-write races: two Lambdas both read N and both write
        N+1, and one row silently overwrites the other. The loop below re-reads
        on conflict so each caller gets a distinct id.
        """
        key = f"{BASE_PREFIX}/{table}/total.txt"
        for attempt in range(8):
            raw = self.get_text(key)
            current = int(raw) if (raw or "").isdigit() else 0
            nxt = current + 1
            try:
                # IfNoneMatch makes the very first write conditional, so two
                # cold starts cannot both initialise the counter to 1.
                extra = {"IfNoneMatch": "*"} if current == 0 else {}
                self.client.put_object(
                    Bucket=self.bucket, Key=key, Body=str(nxt).encode(),
                    ContentType="text/plain", **extra,
                )
                self.stats.s3_puts += 1
                return nxt
            except Exception as exc:  # noqa: BLE001
                if attempt == 7:
                    raise
                time.sleep(0.02 * (2 ** attempt))
                logger.debug("next_id contention on %s: %s", table, exc)
        raise RuntimeError(f"could not allocate id for {table}")

    # -- change feed ------------------------------------------------------
    def change_seq(self, table: str) -> int:
        """Latest change sequence; the browser compares this to its snapshot."""
        raw = self.get_text(f"{BASE_PREFIX}/{table}/change.txt")
        return int(raw) if (raw or "").isdigit() else 0

    def record_change(self, table: str, entry: dict) -> None:
        seq = int(time.time() * 1000)
        base = f"{BASE_PREFIX}/{table}"
        self.put_json(f"{base}/changes/{seq}.json", {"seq": seq, **entry})
        self.put_text(f"{base}/change.txt", str(seq))

    # -- table IO ---------------------------------------------------------
    def keys_manifest(self, table: str) -> list | None:
        """
        Explicit key list for tables whose primary key is not a 1..N integer.

        Several Django models here use 32-char UUID strings as their pk. For
        those, ``total.txt`` cannot describe the key space at all, so the
        migration writes ``_keys.json`` instead and this is what enumerates
        them. Returns None when the table uses integer ids.
        """
        return self.get_json(f"{BASE_PREFIX}/{table}/_keys.json")

    def set_keys_manifest(self, table: str, keys: list) -> None:
        self.put_json(f"{BASE_PREFIX}/{table}/_keys.json", keys)

    def list_table(self, table: str, limit: int = 0, offset: int = 0) -> list[dict]:
        """
        Read live rows of *table*.

        Reads by key rather than LIST: S3's LIST is eventually consistent for
        just-written keys, so a row inserted moments ago could be missing from
        the response. The key space comes from ``_keys.json`` when the table
        has UUID primary keys, otherwise from the ``total.txt`` id range.
        Gaps left by deleted rows simply read as None.
        """
        manifest = self.keys_manifest(table)
        if manifest is not None:
            idents: list = list(manifest)
        else:
            raw = self.get_text(f"{BASE_PREFIX}/{table}/total.txt")
            total = int(raw) if (raw or "").isdigit() else 0
            idents = list(range(1, total + 1))

        rows: list[dict] = []
        seen = 0
        for ident in idents:
            rec = self.get_json(self.key_for(table, ident))
            if not rec or rec.get("_deleted"):
                continue
            if seen < offset:
                seen += 1
                continue
            rows.append(rec)
            if limit and len(rows) >= limit:
                break
        return rows

    def get_row(self, table: str, ident: Any) -> dict | None:
        rec = self.get_json(self.key_for(table, ident))
        if not rec or rec.get("_deleted"):
            return None
        return rec

    def put_row(self, table: str, ident: Any, row: dict) -> None:
        self.put_json(self.key_for(table, ident), row)

    def insert_row(self, table: str, row: dict) -> dict:
        """Allocate an id, write the row, and record the change."""
        ident = self.next_id(table)
        record = {**row, "id": ident}
        self.put_row(table, ident, record)
        self.record_change(table, {"op": "insert", "id": ident})
        return record

    def delete_row(self, table: str, ident: Any) -> None:
        """Soft delete, as C4rr13rX does -- keeps ids stable and auditable."""
        rec = self.get_json(self.key_for(table, ident)) or {}
        rec["_deleted"] = True
        rec["_deleted_at"] = time.time()
        self.put_json(self.key_for(table, ident), rec)
        self.record_change(table, {"op": "delete", "id": ident})

    # -- secondary lookup -------------------------------------------------
    def find_by(self, table: str, field_name: str, value: Any) -> dict | None:
        """
        Look a row up by a non-primary field.

        Backed by an explicit index object rather than a table scan: login
        resolves an email on every attempt, and scanning would make that O(n)
        S3 GETs -- slow and, at S3's per-request price, the main cost driver.
        """
        idx = self.get_json(f"{BASE_PREFIX}/{table}/_index/{field_name}.json") or {}
        ident = idx.get(str(value))
        return self.get_row(table, ident) if ident is not None else None

    def set_index(self, table: str, field_name: str, value: Any, ident: Any) -> None:
        key = f"{BASE_PREFIX}/{table}/_index/{field_name}.json"
        idx = self.get_json(key) or {}
        idx[str(value)] = ident
        self.put_json(key, idx)


def _is_missing(exc: Exception) -> bool:
    """True for the several ways S3/botocore spell 'not found'."""
    code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
    if code in {"NoSuchKey", "404", "NotFound"}:
        return True
    return type(exc).__name__ in {"NoSuchKey", "NoSuchBucket"}


_INSTANCE: SmartStorage | None = None


def get_storage() -> SmartStorage:
    """Process-wide singleton. Holds no durable state; safe to reuse warm."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = SmartStorage()
    return _INSTANCE
