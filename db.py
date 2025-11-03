from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class TradingDatabase:
    """
    Lightweight SQLite wrapper used for caching, experiment tracking, and trading logs.
    Designed to be thread-safe (within CPython) and shared across the application.
    """

    _DEFAULT_PATH = os.getenv(
        "TRADING_DB_PATH",
        str(Path(os.getenv("PORTFOLIO_CACHE_DIR", "~/.cache/mchain")).expanduser() / "trading_cache.db"),
    )

    def __init__(self, path: Optional[str] = None) -> None:
        candidate = Path(path or self._DEFAULT_PATH).expanduser()
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(candidate)
        except Exception:
            fallback = Path("trading_cache.db").absolute()
            fallback.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(fallback)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS balances (
                    wallet TEXT NOT NULL,
                    chain  TEXT NOT NULL,
                    token  TEXT NOT NULL,
                    balance_hex TEXT,
                    asof_block INTEGER,
                    ts REAL,
                    decimals INTEGER,
                    quantity TEXT,
                    usd_amount TEXT,
                    symbol TEXT,
                    name TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (wallet, chain, token)
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transfers (
                    wallet TEXT NOT NULL,
                    chain  TEXT NOT NULL,
                    id     TEXT PRIMARY KEY,
                    hash   TEXT,
                    log_index INTEGER,
                    block INTEGER,
                    ts TEXT,
                    from_addr TEXT,
                    to_addr TEXT,
                    token TEXT,
                    value TEXT,
                    inserted_at REAL DEFAULT (strftime('%s','now'))
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prices (
                    chain TEXT NOT NULL,
                    token TEXT NOT NULL,
                    usd   TEXT,
                    source TEXT,
                    ts REAL,
                    PRIMARY KEY (chain, token)
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_ops (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    wallet TEXT,
                    chain TEXT,
                    symbol TEXT,
                    action TEXT,
                    status TEXT,
                    details TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    status TEXT,
                    params TEXT,
                    results TEXT,
                    created REAL,
                    updated REAL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT,
                    created REAL,
                    metrics TEXT,
                    path TEXT,
                    is_active INTEGER DEFAULT 0
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_stream (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    chain TEXT,
                    symbol TEXT,
                    price REAL,
                    volume REAL,
                    raw TEXT
                );
                """
            )

    @contextmanager
    def _cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    # ------------------------------------------------------------------
    # KV utilities (for model state)
    # ------------------------------------------------------------------

    def load_state(self) -> Dict[str, Any]:
        with self._cursor() as cur:
            cur.execute("SELECT value FROM kv_store WHERE key = ?", ("state",))
            row = cur.fetchone()
            if not row or row["value"] is None:
                return {}
            try:
                return json.loads(row["value"])
            except Exception:
                return {}

    def save_state(self, state: Dict[str, Any]) -> None:
        payload = json.dumps(state or {})
        with self._conn:
            self._conn.execute(
                "INSERT INTO kv_store(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("state", payload),
            )

    # ------------------------------------------------------------------
    # Balance operations
    # ------------------------------------------------------------------

    def upsert_balances(self, entries: Sequence[Dict[str, Any]]) -> None:
        if not entries:
            return
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO balances
                    (wallet, chain, token, balance_hex, asof_block, ts,
                     decimals, quantity, usd_amount, symbol, name, updated_at)
                VALUES
                    (:wallet, :chain, :token, :balance_hex, :asof_block, :ts,
                     :decimals, :quantity, :usd_amount, :symbol, :name, :updated_at)
                ON CONFLICT(wallet, chain, token) DO UPDATE SET
                    balance_hex=excluded.balance_hex,
                    asof_block=excluded.asof_block,
                    ts=excluded.ts,
                    decimals=excluded.decimals,
                    quantity=excluded.quantity,
                    usd_amount=excluded.usd_amount,
                    symbol=CASE WHEN excluded.symbol IS NOT NULL AND excluded.symbol!=''
                                THEN excluded.symbol ELSE balances.symbol END,
                    name=CASE WHEN excluded.name IS NOT NULL AND excluded.name!=''
                              THEN excluded.name ELSE balances.name END,
                    updated_at=excluded.updated_at;
                """,
                entries,
            )

    def fetch_balances(self, wallet: str, chain: str) -> List[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM balances WHERE wallet=? AND chain=? ORDER BY token",
                (_lower(wallet), _lower(chain)),
            )
            return cur.fetchall()

    def fetch_balances_flat(
        self,
        *,
        wallet: Optional[str] = None,
        chains: Optional[Iterable[str]] = None,
        include_zero: bool = False,
    ) -> List[sqlite3.Row]:
        where: List[str] = []
        params: List[Any] = []
        if wallet:
            where.append("wallet=?")
            params.append(_lower(wallet))
        if chains:
            chains_l = [_lower(ch) for ch in chains]
            placeholders = ",".join("?" for _ in chains_l)
            where.append(f"chain IN ({placeholders})")
            params.extend(chains_l)
        if not include_zero:
            where.append("(quantity IS NOT NULL AND quantity != '0')")
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        query = f"""
            SELECT wallet, chain, token, quantity, usd_amount, symbol, updated_at
            FROM balances
            {clause}
            ORDER BY wallet, chain, token;
        """
        with self._cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def delete_balances(self, wallet: str, chain: str, tokens: Sequence[str]) -> None:
        if not tokens:
            return
        params = [_lower(wallet), _lower(chain)] + [_lower(t) for t in tokens]
        placeholders = ",".join("?" for _ in tokens)
        with self._conn:
            self._conn.execute(
                f"DELETE FROM balances WHERE wallet=? AND chain=? AND token IN ({placeholders})",
                params,
            )

    # ------------------------------------------------------------------
    # Transfer operations
    # ------------------------------------------------------------------

    def fetch_transfers(self, wallet: str, chain: str) -> Tuple[List[sqlite3.Row], int, Optional[str]]:
        wallet_l = _lower(wallet)
        chain_l = _lower(chain)
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM transfers WHERE wallet=? AND chain=? ORDER BY block, log_index",
                (wallet_l, chain_l),
            )
            rows = cur.fetchall()
            cur.execute(
                "SELECT MAX(block) AS last_block, MAX(ts) AS last_ts FROM transfers WHERE wallet=? AND chain=?",
                (wallet_l, chain_l),
            )
            meta = cur.fetchone() or {"last_block": 0, "last_ts": None}
        return rows, int(meta["last_block"] or 0), meta["last_ts"]

    def fetch_transfers_since(self, wallet: str, chain: str, since_block: int) -> List[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT token FROM transfers WHERE wallet=? AND chain=? AND block>?;",
                (_lower(wallet), _lower(chain), int(since_block)),
            )
            return cur.fetchall()

    def merge_transfers(self, wallet: str, chain: str, new_items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not new_items:
            rows, last_block, last_ts = self.fetch_transfers(wallet, chain)
            return {
                "wallet": _lower(wallet),
                "chain": _lower(chain),
                "last_block": last_block,
                "last_ts": last_ts,
                "items": [dict(r) for r in rows],
            }
        wallet_l = _lower(wallet)
        chain_l = _lower(chain)
        transformed: List[Dict[str, Any]] = []
        for item in new_items:
            h = item.get("hash") or item.get("transactionHash") or ""
            log_idx = item.get("logIndex")
            if log_idx is None:
                uid = item.get("uniqueId")
                if isinstance(uid, str) and ":" in uid:
                    try:
                        log_idx = int(uid.rsplit(":", 1)[-1])
                    except Exception:
                        log_idx = None
            try:
                log_idx = int(log_idx)
            except Exception:
                log_idx = None
            uid = item.get("uniqueId")
            if isinstance(uid, str) and uid:
                dedupe_id = uid
            elif h and log_idx is not None:
                dedupe_id = f"{h}:{log_idx}"
            else:
                rc = ((item.get("rawContract") or {}).get("address")) or item.get("erc20Contract") or ""
                ts = ((item.get("metadata") or {}).get("blockTimestamp")) or item.get("timestamp") or ""
                dedupe_id = f"{h}:{rc}:{ts}"
            blk = item.get("blockNum") or item.get("blockNumber") or (item.get("metadata") or {}).get("block") or 0
            block_i = _hex_to_int(blk)
            ts_val = (item.get("metadata") or {}).get("blockTimestamp") or item.get("timestamp")
            token_addr = (
                (item.get("rawContract") or {}).get("address")
                or item.get("erc20Contract")
                or item.get("contract")
                or ""
            )
            transformed.append(
                {
                    "wallet": wallet_l,
                    "chain": chain_l,
                    "id": dedupe_id,
                    "hash": h,
                    "log_index": log_idx,
                    "block": block_i,
                    "ts": ts_val,
                    "from_addr": _lower(item.get("from") or item.get("fromAddress")),
                    "to_addr": _lower(item.get("to") or item.get("toAddress")),
                    "token": _lower(token_addr),
                    "value": item.get("value") if "value" in item else item.get("amount"),
                }
            )
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO transfers
                    (wallet, chain, id, hash, log_index, block, ts, from_addr, to_addr, token, value)
                VALUES
                    (:wallet, :chain, :id, :hash, :log_index, :block, :ts, :from_addr, :to_addr, :token, :value)
                ON CONFLICT(id) DO UPDATE SET
                    block=excluded.block,
                    ts=excluded.ts,
                    from_addr=excluded.from_addr,
                    to_addr=excluded.to_addr,
                    token=excluded.token,
                    value=excluded.value;
                """,
                transformed,
            )
        rows, last_block, last_ts = self.fetch_transfers(wallet_l, chain_l)
        return {
            "wallet": wallet_l,
            "chain": chain_l,
            "last_block": last_block,
            "last_ts": last_ts,
            "items": [dict(r) for r in rows],
        }

    # ------------------------------------------------------------------
    # Price operations
    # ------------------------------------------------------------------

    def fetch_price(self, chain: str, token: str) -> Optional[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM prices WHERE chain=? AND token=?", (_lower(chain), _lower(token)))
            return cur.fetchone()

    def fetch_prices(self, chain: str, tokens: Sequence[str]) -> List[sqlite3.Row]:
        if not tokens:
            return []
        placeholders = ",".join("?" for _ in tokens)
        params = [_lower(chain)] + [_lower(t) for t in tokens]
        query = f"SELECT * FROM prices WHERE chain=? AND token IN ({placeholders})"
        with self._cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def upsert_prices(self, entries: Sequence[Tuple[str, str, str, str, float]]) -> None:
        if not entries:
            return
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO prices (chain, token, usd, source, ts)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chain, token) DO UPDATE SET
                    usd=excluded.usd,
                    source=excluded.source,
                    ts=excluded.ts;
                """,
                [( _lower(chain), _lower(token), usd, source, ts) for chain, token, usd, source, ts in entries],
            )

    # ------------------------------------------------------------------
    # Trading / experiment utilities
    # ------------------------------------------------------------------

    def log_trade(self, *, wallet: str, chain: str, symbol: str, action: str, status: str, details: Dict[str, Any]) -> int:
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO trading_ops(ts, wallet, chain, symbol, action, status, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (time.time(), wallet, chain, symbol, action, status, json.dumps(details or {})),
            )
            return cur.lastrowid

    def record_experiment(self, name: str, status: str, params: Dict[str, Any], results: Optional[Dict[str, Any]] = None) -> int:
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO experiments(name, status, params, results, created, updated)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, status, json.dumps(params or {}), json.dumps(results or {}), time.time(), time.time()),
            )
            return cur.lastrowid

    def update_experiment(self, exp_id: int, *, status: Optional[str] = None, results: Optional[Dict[str, Any]] = None) -> None:
        assignments: List[str] = []
        params: List[Any] = []
        if status is not None:
            assignments.append("status=?")
            params.append(status)
        if results is not None:
            assignments.append("results=?")
            params.append(json.dumps(results))
        assignments.append("updated=?")
        params.append(time.time())
        params.append(exp_id)
        with self._conn:
            self._conn.execute(
                f"UPDATE experiments SET {', '.join(assignments)} WHERE id=?",
                params,
            )

    def register_model_version(self, version: str, metrics: Dict[str, Any], path: str, *, activate: bool = False) -> int:
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO model_versions(version, created, metrics, path, is_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (version, time.time(), json.dumps(metrics or {}), path, 1 if activate else 0),
            )
            if activate:
                self._conn.execute(
                    "UPDATE model_versions SET is_active=0 WHERE id != ?",
                    (cur.lastrowid,),
                )
            return cur.lastrowid

    def set_active_model(self, model_id: int) -> None:
        with self._conn:
            self._conn.execute("UPDATE model_versions SET is_active=0")
            self._conn.execute("UPDATE model_versions SET is_active=1 WHERE id=?", (model_id,))

    def insert_market_sample(self, chain: str, symbol: str, price: float, volume: float, raw: Dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO market_stream(ts, chain, symbol, price, volume, raw)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (time.time(), chain, symbol, price, volume, json.dumps(raw or {})),
            )

    def fetch_market_samples(self, limit: int = 512) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT ts, chain, symbol, price, volume, raw FROM market_stream ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            # attempt to decode raw JSON lazily for convenience
            decoded = []
            for row in rows:
                raw = row["raw"]
                parsed = raw
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        parsed = raw
                decoded.append(
                    {
                        "ts": row["ts"],
                        "chain": row["chain"],
                        "symbol": row["symbol"],
                        "price": row["price"],
                        "volume": row["volume"],
                        "raw": parsed,
                    }
                )
            return decoded


# ----------------------------------------------------------------------
# Singleton helpers
# ----------------------------------------------------------------------

_DB_SINGLETON: Optional[TradingDatabase] = None
_DB_LOCK = threading.Lock()


def get_db() -> TradingDatabase:
    global _DB_SINGLETON
    if _DB_SINGLETON is None:
        with _DB_LOCK:
            if _DB_SINGLETON is None:
                _DB_SINGLETON = TradingDatabase()
    return _DB_SINGLETON


def load_db() -> Dict[str, Any]:
    return get_db().load_state()


def save_db(state: Dict[str, Any]) -> None:
    get_db().save_state(state)


# ----------------------------------------------------------------------
# Local helpers (reuse from cache)
# ----------------------------------------------------------------------

def _lower(val: Optional[str]) -> str:
    return (val or "").lower()
