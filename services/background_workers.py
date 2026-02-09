from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

from db import TradingDatabase, get_db
from services.public_api_clients import aggregate_market_data
from services.discovery.coordinator import DiscoveryCoordinator
from services.env_loader import resolve_python_bin
from services.logging_utils import log_message

_PAIR_INDEX_ATTEMPTED: set[str] = set()


def _maybe_generate_pair_index(chain: str) -> bool:
    chain = chain.lower()
    index_path = DATA_ROOT / f"pair_index_{chain}.json"
    if index_path.exists():
        return True
    if chain in _PAIR_INDEX_ATTEMPTED:
        return False
    _PAIR_INDEX_ATTEMPTED.add(chain)
    script = REPO_ROOT / "make2000index.py"
    if not script.exists():
        log_message("pair-index", f"generator script missing: {script}", severity="warning")
        return False
    env = os.environ.copy()
    env.setdefault("CHAIN_NAME", chain)
    env.setdefault("PAIR_INDEX_OUTPUT_PATH", str(index_path))
    try:
        result = subprocess.run([resolve_python_bin(), str(script)], env=env, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            log_message(
                "pair-index",
                f"generation failed for {chain}: rc={result.returncode} stderr={result.stderr.strip()}",
                severity="warning",
            )
            return False
    except Exception as exc:
        log_message("pair-index", f"generation error for {chain}: {exc}", severity="warning")
        return False
    return index_path.exists()


def _load_assignment(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data.setdefault("pairs", {})
        return data
    except Exception as exc:
        log_message("download-worker", f"failed to read {path}: {exc}", severity="warning")
        return None


def _ensure_assignment_template(chain: str, assignment_path: Path) -> Dict[str, Any]:
    data = _load_assignment(assignment_path)
    if data is not None:
        data.setdefault("chain", chain)
        return data
    index_path = DATA_ROOT / f"pair_index_{chain}.json"
    if not index_path.exists():
        if not _maybe_generate_pair_index(chain):
            raise FileNotFoundError(f"pair index not found for {chain}: {index_path}")
    with index_path.open("r", encoding="utf-8") as fh:
        index = json.load(fh)
    pairs = {}
    for addr, meta in index.items():
        symbol = str(meta.get("symbol") or "").upper()
        if not symbol:
            continue
        pairs[addr] = {
            "symbol": symbol,
            "index": int(meta.get("index", len(pairs))),
            "completed": False,
        }
    assignment = {"chain": chain, "pairs": pairs}
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    with assignment_path.open("w", encoding="utf-8") as fh:
        json.dump(assignment, fh, indent=2)
    return assignment


def _update_assignment(path: Path, assignment: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(assignment, fh, indent=2)


def _run_download(chain: str, assignment_path: Path) -> None:
    assignment = _load_assignment(assignment_path)
    if assignment is None:
        assignment = _ensure_assignment_template(chain, assignment_path)
    incomplete = [
        addr for addr, meta in assignment.get("pairs", {}).items() if not meta.get("completed")
    ]
    max_pairs = int(os.getenv("DOWNLOAD_MAX_PAIRS", "256"))
    if max_pairs > 0:
        incomplete = incomplete[:max_pairs]
    if not incomplete:
        return
    max_parallel = max(1, int(os.getenv("DOWNLOAD_MAX_PARALLEL", "1")))
    env = os.environ.copy()
    env["CHAIN_NAME"] = chain
    env["PAIR_ASSIGNMENT_FILE"] = str(assignment_path)
    env.setdefault("OUTPUT_DIR", str(DATA_ROOT / "historical_ohlcv" / chain))
    env.setdefault("INTERMEDIATE_DIR", str(DATA_ROOT / "intermediate" / chain))
    script = Path(__file__).resolve().parents[1] / "download2000.py"
    try:
        for _ in range(max_parallel):
            proc = subprocess.Popen([resolve_python_bin(), str(script)], env=env)
            proc.wait()
    except Exception as exc:
        log_message("download-worker", f"error running download2000 for {chain}: {exc}", severity="error")


class DownloadWorker:
    def __init__(self, chain: str, interval_sec: int = 4 * 3600, assignment_path: Optional[Path] = None) -> None:
        self.chain = chain
        self.assignment_path = assignment_path or DATA_ROOT / f"{chain}_pair_provider_assignment.json"
        self.interval = interval_sec
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def run_once(self) -> None:
        try:
            _run_download(self.chain, self.assignment_path)
        except Exception as exc:
            log_message("download-worker", f"run_once error for {self.chain}: {exc}", severity="error")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self.run_once()
            if self._stop_event.wait(self.interval):
                break


class DynamicDownloadWorker:
    def __init__(
        self,
        db: Optional[TradingDatabase] = None,
        interval_sec: int = 8 * 3600,
        horizon_sec: int = 3 * 24 * 3600,
    ) -> None:
        self.db = db or get_db()
        self.interval = interval_sec
        self.horizon = horizon_sec
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def run_once(self) -> None:
        self._run_cycle()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_cycle()
            except Exception as exc:
                log_message("download-worker", f"dynamic cycle error: {exc}", severity="error")
            if self._stop_event.wait(self.interval):
                break

    def _run_cycle(self) -> None:
        rows = self.db.fetch_recent_pairs(horizon_sec=self.horizon, limit=500)
        chains: Dict[str, Set[str]] = defaultdict(set)
        for row in rows:
            chain = str(row.get("chain") or "").lower()
            symbol = str(row.get("symbol") or "").upper()
            if not chain or not symbol:
                continue
            chains[chain].add(symbol)
        for chain, symbols in chains.items():
            if not symbols:
                continue
            assignment_path = Path("data") / f"{chain}_pair_provider_assignment.json"
            try:
                self._ensure_pairs(chain, symbols, assignment_path)
                _run_download(chain, assignment_path)
            except FileNotFoundError:
                # skip chains without pair index (not yet supported)
                continue

    def _ensure_pairs(self, chain: str, symbols: Iterable[str], assignment_path: Path) -> None:
        assignment = _ensure_assignment_template(chain, assignment_path)
        pairs = assignment.setdefault("pairs", {})
        pair_index_path = DATA_ROOT / f"pair_index_{chain}.json"
        if not pair_index_path.exists():
            if not _maybe_generate_pair_index(chain):
                raise FileNotFoundError(pair_index_path)
        with pair_index_path.open("r", encoding="utf-8") as fh:
            index = json.load(fh)
        existing_symbols = {meta.get("symbol", "").upper() for meta in pairs.values()}
        added = False
        for addr, meta in index.items():
            symbol = str(meta.get("symbol") or "").upper()
            if not symbol or symbol not in symbols or symbol in existing_symbols:
                continue
            pairs[addr] = {
                "symbol": symbol,
                "index": int(meta.get("index", len(pairs))),
                "completed": False,
            }
            added = True
        if added:
            _update_assignment(assignment_path, assignment)


class MarketDataWorker:
    def __init__(
        self,
        db: Optional[TradingDatabase] = None,
        interval_sec: int = 1200,
        symbols: Optional[Sequence[str]] = None,
    ) -> None:
        self.db = db or get_db()
        self.interval = max(60, interval_sec)
        env_symbols = os.getenv("MARKET_DATA_SYMBOLS")
        if env_symbols:
            symbols = [sym.strip().upper() for sym in env_symbols.split(",") if sym.strip()]
        self.symbols = symbols
        self.gecko_ids = [gid.strip() for gid in os.getenv("MARKET_DATA_COINGECKO_IDS", "bitcoin,ethereum").split(",") if gid.strip()]
        self.output_path = Path(os.getenv("MARKET_DATA_SNAPSHOT_PATH", "data/market_snapshots.json"))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def run_once(self) -> None:
        snapshots = aggregate_market_data(
            symbols=self.symbols,
            coingecko_ids=self.gecko_ids,
            top_n=int(os.getenv("MARKET_DATA_TOP_N", "25")),
        )
        if not snapshots:
            return
        rows = self._build_price_rows(snapshots)
        if rows:
            try:
                self.db.upsert_prices(rows)
            except Exception as exc:
                log_message("market-data", f"failed to persist prices: {exc}", severity="error")
        try:
            from services.public_api_clients import save_snapshots

            save_snapshots(snapshots, self.output_path)
        except Exception as exc:
            log_message("market-data", f"failed to write snapshots: {exc}", severity="error")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                log_message("market-data", f"cycle error: {exc}", severity="error")
            if self._stop_event.wait(self.interval):
                break

    def _build_price_rows(self, snapshots) -> List[Tuple[str, str, float, str, float]]:
        rows: Dict[Tuple[str, str], Tuple[str, str, float, str, float]] = {}
        now = time.time()
        for snap in snapshots:
            symbol = str(getattr(snap, "symbol", "") or "").upper()
            if not symbol:
                continue
            try:
                price = float(getattr(snap, "price_usd", 0.0))
            except Exception:
                continue
            if not math.isfinite(price) or price <= 0:
                continue
            source = str(getattr(snap, "source", "unknown") or "unknown").lower()
            ts = float(getattr(snap, "ts", now) or now)
            key = (symbol, source)
            candidate = ("global", symbol, price, source, ts)
            existing = rows.get(key)
            if existing is None or candidate[-1] >= existing[-1]:
                rows[key] = candidate
        return list(rows.values())


class DiscoveryWorker:
    def __init__(
        self,
        *,
        interval_sec: int = int(os.getenv("DISCOVERY_INTERVAL_SEC", "1800")),
        chains: Optional[Sequence[str]] = None,
    ) -> None:
        self.interval = max(300, interval_sec)
        self.coordinator = DiscoveryCoordinator(chains=chains)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def run_once(self) -> None:
        results = self.coordinator.run()
        if results:
            log_message("discovery", f"processed {len(results)} tokens")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                log_message("discovery", f"cycle error: {exc}", severity="error")
            if self._stop_event.wait(self.interval):
                break


class TokenDownloadSupervisor:
    def __init__(self, db: Optional[TradingDatabase] = None) -> None:
        chains_env = os.getenv("DOWNLOAD_WORKER_CHAINS", "base,ethereum,arbitrum,optimism,polygon")
        chains = [c.strip().lower() for c in chains_env.split(",") if c.strip()]
        self.static_workers: list[DownloadWorker] = []
        for chain in chains:
            index_path = DATA_ROOT / f"pair_index_{chain}.json"
            if not index_path.exists():
                if _maybe_generate_pair_index(chain):
                    log_message("download-supervisor", f"auto-generated {index_path} for {chain}", severity="info")
                else:
                    log_message(
                        "download-supervisor",
                        f"skipping download worker for {chain}: missing {index_path}",
                        severity="warning",
                    )
                    continue
            self.static_workers.append(
                DownloadWorker(chain, interval_sec=int(os.getenv("BASE_DOWNLOAD_INTERVAL", "10800")))
            )
        self.dynamic_worker = DynamicDownloadWorker(
            db=db,
            interval_sec=int(os.getenv("DYNAMIC_DOWNLOAD_INTERVAL", "21600")),
            horizon_sec=int(os.getenv("DYNAMIC_DOWNLOAD_HORIZON", str(3 * 24 * 3600))),
        )
        self.market_worker = MarketDataWorker(
            db=db,
            interval_sec=int(os.getenv("MARKET_DATA_REFRESH_SEC", "1200")),
        )
        discovery_chains = os.getenv("DISCOVERY_CHAINS")
        chains = [c.strip() for c in discovery_chains.split(",") if c.strip()] if discovery_chains else None
        self.discovery_worker = DiscoveryWorker(
            interval_sec=int(os.getenv("DISCOVERY_INTERVAL_SEC", "1800")),
            chains=chains,
        )

    def start(self) -> None:
        for worker in self.static_workers:
            worker.start()
        self.dynamic_worker.start()
        self.market_worker.start()
        self.discovery_worker.start()

    def stop(self) -> None:
        for worker in self.static_workers:
            worker.stop()
        self.dynamic_worker.stop()
        self.market_worker.stop()
        self.discovery_worker.stop()

    def run_cycle(self) -> None:
        for worker in self.static_workers:
            try:
                worker.run_once()
            except Exception as exc:
                log_message("download-supervisor", f"{worker.chain} cycle error: {exc}", severity="error")
        try:
            self.dynamic_worker.run_once()
        except Exception as exc:
            log_message("download-supervisor", f"dynamic cycle error: {exc}", severity="error")
        try:
            self.market_worker.run_once()
        except Exception as exc:
            log_message("download-supervisor", f"market cycle error: {exc}", severity="error")
        try:
            self.discovery_worker.run_once()
        except Exception as exc:
            log_message("download-supervisor", f"discovery cycle error: {exc}", severity="error")
