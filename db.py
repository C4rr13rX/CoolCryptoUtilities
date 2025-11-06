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
        str((Path.cwd() / "storage" / "trading_cache.db").resolve()),
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
                CREATE TABLE IF NOT EXISTS control_flags (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated REAL DEFAULT (strftime('%s','now'))
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
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    chain TEXT,
                    symbol TEXT,
                    expected_amount REAL,
                    executed_amount REAL,
                    expected_price REAL,
                    executed_price REAL,
                    details TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    stage TEXT,
                    category TEXT,
                    name TEXT,
                    value REAL,
                    meta TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metrics_stage_ts
                ON metrics(stage, ts);
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    source TEXT,
                    severity TEXT,
                    label TEXT,
                    details TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_source_ts
                ON feedback_events(source, ts);
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS advisories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    scope TEXT,
                    topic TEXT,
                    severity TEXT,
                    message TEXT,
                    recommendation TEXT,
                    meta TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolved_ts REAL
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_advisories_resolved_ts
                ON advisories(resolved, ts);
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS organism_snapshots (
                    ts REAL PRIMARY KEY,
                    payload TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pair_suppression (
                    symbol TEXT PRIMARY KEY,
                    reason TEXT,
                    strikes INTEGER DEFAULT 1,
                    last_failure REAL,
                    release_ts REAL,
                    metadata TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pair_adjustments (
                    symbol TEXT PRIMARY KEY,
                    priority INTEGER DEFAULT 0,
                    enter_offset REAL DEFAULT 0.0,
                    exit_offset REAL DEFAULT 0.0,
                    size_multiplier REAL DEFAULT 1.0,
                    margin_offset REAL DEFAULT 0.0,
                    allocation_multiplier REAL DEFAULT 1.0,
                    label_scale REAL DEFAULT 1.0,
                    updated REAL DEFAULT (strftime('%s','now')),
                    details TEXT
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

    def set_json(self, key: str, payload: Any) -> None:
        data = json.dumps(payload or {})
        with self._conn:
            self._conn.execute(
                "INSERT INTO kv_store(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(key), data),
            )

    def get_json(self, key: str) -> Optional[Any]:
        with self._cursor() as cur:
            cur.execute("SELECT value FROM kv_store WHERE key=?", (str(key),))
            row = cur.fetchone()
        if not row or row["value"] is None:
            return None
        try:
            return json.loads(row["value"])
        except Exception:
            return None

    def set_control_flag(self, key: str, value: Any) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO control_flags(key, value, updated)
                VALUES(?, ?, strftime('%s','now'))
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated=excluded.updated
                """,
                (str(key), "" if value is None else str(value)),
            )

    def get_control_flag(self, key: str) -> Optional[str]:
        with self._cursor() as cur:
            cur.execute("SELECT value FROM control_flags WHERE key=?", (str(key),))
            row = cur.fetchone()
        if not row:
            return None
        return row["value"]

    def clear_control_flag(self, key: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM control_flags WHERE key=?", (str(key),))

    # ------------------------------------------------------------------
    # Fills / execution feedback
    # ------------------------------------------------------------------

    def record_trade_fill(
        self,
        *,
        chain: str,
        symbol: str,
        expected_amount: float,
        executed_amount: float,
        expected_price: float,
        executed_price: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO trade_fills(ts, chain, symbol, expected_amount, executed_amount, expected_price, executed_price, details)
                VALUES(strftime('%s','now'), ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chain,
                    symbol,
                    float(expected_amount),
                    float(executed_amount),
                    float(expected_price),
                    float(executed_price),
                    json.dumps(details or {}),
                ),
            )

    def fetch_trade_fills(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT ts, chain, symbol, expected_amount, executed_amount, expected_price, executed_price, details FROM trade_fills ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
            rows = cur.fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            entry = dict(row)
            try:
                entry["details"] = json.loads(entry.get("details") or "{}")
            except Exception:
                entry["details"] = {}
            results.append(entry)
        return results

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

    def fetch_market_samples_for(self, symbol: str, limit: int = 512) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT ts, chain, symbol, price, volume, raw
                FROM market_stream
                WHERE symbol = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (symbol, limit),
            )
            rows = cur.fetchall()
        decoded: List[Dict[str, Any]] = []
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

    def fetch_recent_pairs(self, *, horizon_sec: int = 24 * 3600, limit: int = 200) -> List[Dict[str, Any]]:
        cutoff = time.time() - max(0, float(horizon_sec))
        query = """
            SELECT chain, symbol, MAX(ts) AS last_ts
            FROM market_stream
            WHERE ts >= ?
            GROUP BY chain, symbol
            ORDER BY last_ts DESC
            LIMIT ?
        """
        with self._cursor() as cur:
            cur.execute(query, (cutoff, int(limit)))
            rows = cur.fetchall()
        return [{"chain": row["chain"], "symbol": row["symbol"], "last_ts": row["last_ts"]} for row in rows]

    # ------------------------------------------------------------------
    # Metrics & feedback
    # ------------------------------------------------------------------

    def record_metrics(
        self,
        *,
        stage: str,
        metrics: Dict[str, Any],
        category: str = "general",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not metrics:
            return
        ts = time.time()
        payload = json.dumps(meta or {})
        rows: List[Tuple[float, str, str, str, float, str]] = []
        for name, value in metrics.items():
            try:
                numeric_value = float(value)
            except Exception:
                continue
            rows.append((ts, stage, category, str(name), numeric_value, payload))
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO metrics(ts, stage, category, name, value, meta)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def fetch_metrics(
        self,
        *,
        stage: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        where: List[str] = []
        params: List[Any] = []
        if stage:
            where.append("stage=?")
            params.append(stage)
        if category:
            where.append("category=?")
            params.append(category)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        query = f"""
            SELECT ts, stage, category, name, value, meta
            FROM metrics
            {clause}
            ORDER BY ts DESC
            LIMIT ?
        """
        params.append(int(limit))
        with self._cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            meta_raw = row["meta"]
            try:
                meta = json.loads(meta_raw) if meta_raw else {}
            except Exception:
                meta = {}
            out.append(
                {
                    "ts": row["ts"],
                    "stage": row["stage"],
                    "category": row["category"],
                    "name": row["name"],
                    "value": row["value"],
                    "meta": meta,
                }
            )
        return out

    def record_feedback_event(
        self,
        *,
        source: str,
        severity: str,
        label: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = json.dumps(details or {})
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO feedback_events(ts, source, severity, label, details)
                VALUES (?, ?, ?, ?, ?)
                """,
                (time.time(), source, severity, label, payload),
            )

    def fetch_feedback_events(
        self,
        *,
        sources: Optional[Sequence[str]] = None,
        severity: Optional[Sequence[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        where: List[str] = []
        params: List[Any] = []
        if sources:
            placeholders = ",".join("?" for _ in sources)
            where.append(f"source IN ({placeholders})")
            params.extend(sources)
        if severity:
            placeholders = ",".join("?" for _ in severity)
            where.append(f"severity IN ({placeholders})")
            params.extend(severity)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        query = f"""
            SELECT ts, source, severity, label, details
            FROM feedback_events
            {clause}
            ORDER BY ts DESC
            LIMIT ?
        """
        params.append(int(limit))
        with self._cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        entries: List[Dict[str, Any]] = []
        for row in rows:
            details_raw = row["details"]
            try:
                details = json.loads(details_raw) if details_raw else {}
            except Exception:
                details = {}
            entries.append(
                {
                    "ts": row["ts"],
                    "source": row["source"],
                    "severity": row["severity"],
                    "label": row["label"],
                    "details": details,
                }
            )
        return entries

    def fetch_trades(
        self,
        *,
        limit: int = 200,
        statuses: Optional[Sequence[str]] = None,
        wallets: Optional[Sequence[str]] = None,
        symbol: Optional[str] = None,
        since_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        where: List[str] = []
        params: List[Any] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            where.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if wallets:
            placeholders = ",".join("?" for _ in wallets)
            where.append(f"wallet IN ({placeholders})")
            params.extend(wallets)
        if symbol:
            where.append("symbol=?")
            params.append(symbol)
        if since_ts is not None:
            where.append("ts>=?")
            params.append(float(since_ts))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        query = f"""
            SELECT ts, wallet, chain, symbol, action, status, details
            FROM trading_ops
            {clause}
            ORDER BY ts DESC
            LIMIT ?
        """
        params.append(int(limit))
        with self._cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        trades: List[Dict[str, Any]] = []
        for row in rows:
            details_raw = row["details"]
            try:
                details = json.loads(details_raw) if details_raw else {}
            except Exception:
                details = {}
            trades.append(
                {
                    "ts": row["ts"],
                    "wallet": row["wallet"],
                    "chain": row["chain"],
                    "symbol": row["symbol"],
                    "action": row["action"],
                    "status": row["status"],
                    "details": details,
                }
            )
        return trades

    # ------------------------------------------------------------------
    # Advisory / recommendation helpers
    # ------------------------------------------------------------------

    def record_advisory(
        self,
        *,
        topic: str,
        message: str,
        severity: str = "info",
        scope: Optional[str] = None,
        recommendation: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        payload = json.dumps(meta or {})
        ts = time.time()
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO advisories(ts, scope, topic, severity, message, recommendation, meta, resolved)
                VALUES(?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (ts, scope or "", topic, severity, message, recommendation or "", payload),
            )
            return int(cur.lastrowid)

    def fetch_advisories(
        self,
        *,
        limit: int = 200,
        include_resolved: bool = False,
        severity: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        where: List[str] = []
        params: List[Any] = []
        if not include_resolved:
            where.append("resolved=0")
        if severity:
            placeholders = ",".join("?" for _ in severity)
            where.append(f"severity IN ({placeholders})")
            params.extend(severity)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        query = f"""
            SELECT id, ts, scope, topic, severity, message, recommendation, meta, resolved, resolved_ts
            FROM advisories
            {clause}
            ORDER BY resolved ASC, ts DESC
            LIMIT ?
        """
        params.append(int(limit))
        with self._cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        advisories: List[Dict[str, Any]] = []
        for row in rows:
            try:
                meta = json.loads(row["meta"] or "{}")
            except Exception:
                meta = {}
            advisories.append(
                {
                    "id": row["id"],
                    "ts": row["ts"],
                    "scope": row["scope"],
                    "topic": row["topic"],
                    "severity": row["severity"],
                    "message": row["message"],
                    "recommendation": row["recommendation"],
                    "meta": meta,
                    "resolved": bool(row["resolved"]),
                    "resolved_ts": row["resolved_ts"],
                }
            )
        return advisories

    def resolve_advisory(self, advisory_id: int) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE advisories
                SET resolved=1,
                    resolved_ts=strftime('%s','now')
                WHERE id=?;
                """,
                (int(advisory_id),),
            )

    # ------------------------------------------------------------------
    # Discovery helpers (Django-managed tables)
    # ------------------------------------------------------------------

    def discovery_status_counts(self) -> Dict[str, int]:
        try:
            with self._cursor() as cur:
                cur.execute(
                    "SELECT status, COUNT(*) as total FROM discovery_discoveredtoken GROUP BY status"
                )
                rows = cur.fetchall()
        except Exception:
            return {}
        return {row["status"]: int(row["total"]) for row in rows}

    def discovery_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            with self._cursor() as cur:
                cur.execute(
                    """
                    SELECT created_at, symbol, chain, source, bull_score, bear_score, liquidity_usd, volume_24h, price_change_24h
                    FROM discovery_discoveryevent
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                )
                rows = cur.fetchall()
        except Exception:
            return []
        return [dict(row) for row in rows]

    def discovery_recent_honeypots(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            with self._cursor() as cur:
                cur.execute(
                    """
                    SELECT created_at, symbol, chain, verdict, confidence
                    FROM discovery_honeypotcheck
                    WHERE verdict='honeypot'
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                )
                rows = cur.fetchall()
        except Exception:
            return []
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Organism snapshots
    # ------------------------------------------------------------------

    def record_organism_snapshot(self, snapshot: Dict[str, Any]) -> None:
        ts = float(snapshot.get("timestamp") or time.time())
        payload = json.dumps(snapshot)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO organism_snapshots(ts, payload)
                VALUES(?, ?)
                ON CONFLICT(ts) DO UPDATE SET payload=excluded.payload
                """,
                (ts, payload),
            )

    def fetch_latest_organism_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT payload FROM organism_snapshots ORDER BY ts DESC LIMIT 1")
            row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row["payload"])
        except Exception:
            return None

    def fetch_organism_snapshot_at(self, ts: float) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT payload FROM organism_snapshots WHERE ts<=? ORDER BY ts DESC LIMIT 1",
                (float(ts),),
            )
            row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row["payload"])
        except Exception:
            return None

    def fetch_organism_history(
        self,
        *,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if start_ts is not None:
            clauses.append("ts >= ?")
            params.append(float(start_ts))
        if end_ts is not None:
            clauses.append("ts <= ?")
            params.append(float(end_ts))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT ts, payload FROM organism_snapshots {where} ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))
        with self._cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        history: List[Dict[str, Any]] = []
        for row in rows:
            try:
                history.append(json.loads(row["payload"]))
            except Exception:
                continue
        return history

    # ------------------------------------------------------------------
    # Pair suppression (for unreliable market data)
    # ------------------------------------------------------------------

    def record_pair_suppression(
        self,
        symbol: str,
        reason: str,
        *,
        ttl_seconds: float = 6 * 3600,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        release_ts = now + max(float(ttl_seconds), 0.0)
        payload = json.dumps(metadata or {})
        symbol_u = symbol.upper()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO pair_suppression (symbol, reason, strikes, last_failure, release_ts, metadata)
                VALUES(?, ?, 1, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    reason=excluded.reason,
                    strikes=pair_suppression.strikes + 1,
                    last_failure=excluded.last_failure,
                    release_ts=excluded.release_ts,
                    metadata=excluded.metadata;
                """,
                (symbol_u, reason, now, release_ts, payload),
            )

    def get_pair_suppression(self, symbol: str) -> Optional[Dict[str, Any]]:
        symbol_u = symbol.upper()
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT symbol, reason, strikes, last_failure, release_ts, metadata
                FROM pair_suppression
                WHERE symbol=?
                """,
                (symbol_u,),
            )
            row = cur.fetchone()
        if not row:
            return None
        metadata = {}
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except Exception:
            metadata = {}
        return {
            "symbol": row["symbol"],
            "reason": row["reason"],
            "strikes": int(row["strikes"] or 0),
            "last_failure": float(row["last_failure"] or 0.0),
            "release_ts": float(row["release_ts"] or 0.0),
            "metadata": metadata,
        }

    def is_pair_suppressed(self, symbol: str) -> bool:
        record = self.get_pair_suppression(symbol)
        if not record:
            return False
        release = float(record.get("release_ts") or 0.0)
        if release > time.time():
            return True
        self.clear_pair_suppression(symbol)
        return False

    def clear_pair_suppression(self, symbol: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM pair_suppression WHERE symbol=?",
                (symbol.upper(),),
            )

    # ------------------------------------------------------------------
    # Pair adjustments (trade/scheduler knobs)
    # ------------------------------------------------------------------

    def get_pair_adjustment(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT symbol, priority, enter_offset, exit_offset, size_multiplier,
                       margin_offset, allocation_multiplier, updated, details
                FROM pair_adjustments
                WHERE symbol=?
                """,
                (symbol.upper(),),
            )
            row = cur.fetchone()
        if not row:
            return None
        details = {}
        try:
            details = json.loads(row["details"] or "{}")
        except Exception:
            details = {}
        return {
            "symbol": row["symbol"],
            "priority": int(row["priority"] or 0),
            "enter_offset": float(row["enter_offset"] or 0.0),
            "exit_offset": float(row["exit_offset"] or 0.0),
           "size_multiplier": float(row["size_multiplier"] or 1.0),
           "margin_offset": float(row["margin_offset"] or 0.0),
           "allocation_multiplier": float(row["allocation_multiplier"] or 1.0),
            "label_scale": float(row["label_scale"] or 1.0),
           "updated": float(row["updated"] or 0.0),
           "details": details,
       }

    def upsert_pair_adjustment(
        self,
        symbol: str,
        *,
        enter_offset: Optional[float] = None,
        exit_offset: Optional[float] = None,
        size_multiplier: Optional[float] = None,
        margin_offset: Optional[float] = None,
        allocation_multiplier: Optional[float] = None,
        label_scale: Optional[float] = None,
        priority: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        current = self.get_pair_adjustment(symbol) or {}
        current_details = current.get("details", {})
        if not isinstance(current_details, dict):
            current_details = {}
        payload_details = details if details is not None else current_details
        payload = {
            "symbol": symbol.upper(),
            "enter_offset": enter_offset if enter_offset is not None else current.get("enter_offset", 0.0),
            "exit_offset": exit_offset if exit_offset is not None else current.get("exit_offset", 0.0),
            "size_multiplier": size_multiplier if size_multiplier is not None else current.get("size_multiplier", 1.0),
            "margin_offset": margin_offset if margin_offset is not None else current.get("margin_offset", 0.0),
            "allocation_multiplier": allocation_multiplier if allocation_multiplier is not None else current.get(
                "allocation_multiplier", 1.0
            ),
            "label_scale": label_scale if label_scale is not None else current.get("label_scale", 1.0),
            "priority": priority if priority is not None else current.get("priority", 0),
            "details": json.dumps(payload_details),
        }
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO pair_adjustments(symbol, priority, enter_offset, exit_offset,
                                             size_multiplier, margin_offset, allocation_multiplier,
                                             label_scale, details, updated)
                VALUES(:symbol, :priority, :enter_offset, :exit_offset,
                       :size_multiplier, :margin_offset, :allocation_multiplier,
                       :label_scale, :details, strftime('%s','now'))
                ON CONFLICT(symbol) DO UPDATE SET
                    priority=excluded.priority,
                    enter_offset=excluded.enter_offset,
                    exit_offset=excluded.exit_offset,
                    size_multiplier=excluded.size_multiplier,
                    margin_offset=excluded.margin_offset,
                    allocation_multiplier=excluded.allocation_multiplier,
                    label_scale=excluded.label_scale,
                    details=excluded.details,
                    updated=strftime('%s','now');
                """,
                payload,
            )

    def adjust_pair_allocation(self, symbol: str, delta: float, *, floor: float = 0.05, ceiling: float = 2.0) -> None:
        record = self.get_pair_adjustment(symbol) or {}
        multiplier = float(record.get("allocation_multiplier", 1.0))
        multiplier = max(floor, min(ceiling, multiplier + delta))
        self.upsert_pair_adjustment(symbol, allocation_multiplier=multiplier)

    def adjust_pair_size(self, symbol: str, delta: float, *, floor: float = 0.1, ceiling: float = 3.0) -> None:
        record = self.get_pair_adjustment(symbol) or {}
        multiplier = float(record.get("size_multiplier", 1.0))
        multiplier = max(floor, min(ceiling, multiplier + delta))
        self.upsert_pair_adjustment(symbol, size_multiplier=multiplier)

    def get_label_scale(self, symbol: str = "__GLOBAL__") -> float:
        record = self.get_pair_adjustment(symbol)
        if not record:
            return 1.0
        try:
            return float(record.get("label_scale", 1.0))
        except Exception:
            return 1.0

    def set_label_scale(self, scale: float, symbol: str = "__GLOBAL__") -> None:
        safe = max(0.5, min(7.0, float(scale)))
        self.upsert_pair_adjustment(symbol, label_scale=safe)


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
def _hex_to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if s.startswith("0x"):
        try:
            return int(s, 16)
        except Exception:
            return 0
    try:
        return int(s)
    except Exception:
        return 0
