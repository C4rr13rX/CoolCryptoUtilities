"""
Shared market-data store: Parquet partitions in S3, searched locally.

Market data is *shared* -- one copy serves every account, because the price of
WETH at a given second is not per-user. That is why it lives under its own
prefix with no owner column, and why the browser can cache it aggressively:
there is nothing account-specific to leak between sessions.

Why this is a different shape from ``smart_storage``
----------------------------------------------------
The per-row ``<table>/<id>.json`` layout is right for ~1,400 rows of config
and admin data. It is catastrophically wrong for 3.4M rows of time series:

* 3.4M PUTs to write (~$17) and, far worse, 3.4M GETs on every sync;
* per-object overhead dwarfs a 142-byte metrics row;
* no way to fetch "last 7 days" without reading everything.

So market data is written as **Parquet, partitioned by month**:

    database/market/<table>/<YYYY-MM>.parquet
    database/market/<table>/_manifest.json

That turns 3.4M objects into ~40. Parquet is columnar, so a chart that needs
``ts`` and ``price`` never transfers ``raw``; it is compressed (~7x here); and
``hyparquet`` reads it directly in the browser, where AllezORM does the actual
searching. The manifest lists partitions with row counts and time ranges so a
client can decide what to fetch without a LIST call.

The one exception: ``organism_snapshots``
------------------------------------------
54k rows averaging ~430 KB of JSON each -- 22.3 GB raw, ~3 GB compressed. Its
payloads are opaque blobs, so columnar storage buys nothing. It is stored as
individually gzipped objects plus a lightweight timestamp index. The browser
mirrors only the index (~54k rows, a few MB) and fetches a payload on demand.
Mirroring the bodies would blow past every browser's IndexedDB quota.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

logger = logging.getLogger("serverless.hybrid.market_store")

MARKET_PREFIX = os.getenv("HYBRID_MARKET_PREFIX", "database/market")

# Tables that are shared across all accounts and partitioned by month.
# `ts_column` is the epoch-seconds column the partitioning keys on.
PARTITIONED_TABLES: dict[str, str] = {
    "market_stream": "ts",
    "metrics": "ts",
    "feedback_events": "ts",
    "trade_fills": "ts",
    "trading_ops": "ts",
    "prices": "ts",
    "trade_outcomes": "ts",
    "advisories": "ts",
}

# Stored one-object-per-row, gzipped, with a separate index. See the docstring.
BLOB_TABLES: dict[str, str] = {
    "organism_snapshots": "ts",
}


@dataclass
class Partition:
    """One month of one table."""

    key: str          # e.g. "2026-07"
    rows: int
    min_ts: float
    max_ts: float
    bytes: int

    def as_dict(self) -> dict:
        return {
            "key": self.key, "rows": self.rows,
            "min_ts": self.min_ts, "max_ts": self.max_ts, "bytes": self.bytes,
        }


def month_of(ts: float) -> str:
    """Partition key for an epoch-seconds timestamp."""
    import datetime as dt

    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m")


class MarketStore:
    """Reader/writer for the shared market-data prefix."""

    def __init__(self, storage) -> None:
        # Reuses SmartStorage purely for its S3 client and bucket; the key
        # layout here is entirely different.
        self.storage = storage

    # -- keys -------------------------------------------------------------
    def partition_key(self, table: str, month: str) -> str:
        return f"{MARKET_PREFIX}/{table}/{month}.parquet"

    def manifest_key(self, table: str) -> str:
        return f"{MARKET_PREFIX}/{table}/_manifest.json"

    def blob_key(self, table: str, ident: Any) -> str:
        return f"{MARKET_PREFIX}/{table}/blobs/{ident}.json.gz"

    def blob_index_key(self, table: str) -> str:
        return f"{MARKET_PREFIX}/{table}/_index.parquet"

    # -- manifest ---------------------------------------------------------
    def read_manifest(self, table: str) -> dict:
        return self.storage.get_json(self.manifest_key(table)) or {
            "table": table, "partitions": [], "rows": 0,
        }

    def write_manifest(self, table: str, partitions: list[Partition]) -> dict:
        manifest = {
            "table": table,
            "partitions": [p.as_dict() for p in sorted(partitions, key=lambda x: x.key)],
            "rows": sum(p.rows for p in partitions),
            "bytes": sum(p.bytes for p in partitions),
            # Shared data: state this explicitly so a client never attaches an
            # account filter to it and silently gets nothing back.
            "shared": True,
        }
        self.storage.put_json(self.manifest_key(table), manifest)
        return manifest

    # -- parquet ----------------------------------------------------------
    def write_partition(self, table: str, month: str, rows: list[dict]) -> Partition:
        """Write one month as a Parquet object and return its descriptor."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        ts_col = PARTITIONED_TABLES.get(table, "ts")
        buf = io.BytesIO()
        arrow = pa.Table.from_pylist(rows)
        pq.write_table(
            arrow, buf,
            # zstd beats snappy on this data and hyparquet reads both; the
            # transfer size is what the browser actually pays for.
            compression="zstd", compression_level=6,
        )
        body = buf.getvalue()

        self.storage.client.put_object(
            Bucket=self.storage.bucket,
            Key=self.partition_key(table, month),
            Body=body,
            ContentType="application/vnd.apache.parquet",
        )
        self.storage.stats.s3_puts += 1
        self.storage.stats.bytes_up += len(body)

        stamps = [float(r.get(ts_col) or 0) for r in rows]
        return Partition(
            key=month, rows=len(rows), bytes=len(body),
            min_ts=min(stamps) if stamps else 0.0,
            max_ts=max(stamps) if stamps else 0.0,
        )

    def read_partition(self, table: str, month: str) -> list[dict]:
        """
        Read one month back as row dicts.

        Migration/verification only. The Lambda bundle ships without pyarrow
        (136 MB, and the request path does not need it) -- the handler
        presigns partition URLs and the browser parses them with hyparquet.
        """
        import pyarrow.parquet as pq

        try:
            resp = self.storage.client.get_object(
                Bucket=self.storage.bucket, Key=self.partition_key(table, month)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("partition missing %s/%s: %s", table, month, exc)
            return []
        raw = resp["Body"].read()
        self.storage.stats.s3_gets += 1
        self.storage.stats.bytes_down += len(raw)
        return pq.read_table(io.BytesIO(raw)).to_pylist()

    # -- blobs ------------------------------------------------------------
    def write_blob(self, table: str, ident: Any, payload: str | bytes) -> int:
        """Store one large opaque payload, gzipped. Returns bytes written."""
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        body = gzip.compress(data, 6)
        self.storage.client.put_object(
            Bucket=self.storage.bucket,
            Key=self.blob_key(table, ident),
            Body=body,
            ContentType="application/json",
            ContentEncoding="gzip",
        )
        self.storage.stats.s3_puts += 1
        self.storage.stats.bytes_up += len(body)
        return len(body)

    def read_blob(self, table: str, ident: Any) -> Any | None:
        try:
            resp = self.storage.client.get_object(
                Bucket=self.storage.bucket, Key=self.blob_key(table, ident)
            )
        except Exception:  # noqa: BLE001
            return None
        raw = resp["Body"].read()
        self.storage.stats.s3_gets += 1
        # boto3 does not transparently gunzip a stored ContentEncoding, and a
        # caller that assumed it would get bytes back instead of JSON.
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))

    def write_blob_index(self, table: str, entries: list[dict]) -> int:
        """
        Write the lightweight index the browser mirrors.

        Holds only the key and timestamp of each blob -- the whole point is
        that it is small enough for IndexedDB while the 22 GB of payloads stay
        in S3, fetched on demand.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        buf = io.BytesIO()
        pq.write_table(
            pa.Table.from_pylist(entries), buf,
            compression="zstd", compression_level=6,
        )
        body = buf.getvalue()
        self.storage.client.put_object(
            Bucket=self.storage.bucket,
            Key=self.blob_index_key(table),
            Body=body,
            ContentType="application/vnd.apache.parquet",
        )
        self.storage.stats.s3_puts += 1
        return len(body)


def batched(rows: Iterable[dict], size: int) -> Iterator[list[dict]]:
    """Yield fixed-size chunks so a migration never holds a whole table."""
    chunk: list[dict] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
