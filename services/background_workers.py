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
    tmp = path.with_suffix(".tmp")
    for attempt in range(3):
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(assignment, fh, indent=2)
            tmp.replace(path)
            return
        except OSError:
            if attempt < 2:
                time.sleep(0.3 * (attempt + 1))
            else:
                raise


def _collect_completed_symbols(assignment_path: Path) -> List[str]:
    """Extract individual token symbols from completed (non-skipped) pairs."""
    import re
    data = _load_assignment(assignment_path)
    if not data:
        return []
    symbols: List[str] = []
    seen: set = set()
    for _addr, meta in (data.get("pairs") or {}).items():
        if not meta.get("completed") or meta.get("skipped"):
            continue
        raw = str(meta.get("symbol") or "")
        for part in re.split(r"[-_/]", raw.upper()):
            part = part.strip()
            if not part or len(part) < 2 or part in seen:
                continue
            if part.startswith("W") and len(part) > 2:
                base = part[1:]
                if base not in seen:
                    seen.add(base)
                    symbols.append(base)
            seen.add(part)
            symbols.append(part)
    return symbols


def _trigger_news_for_symbols(symbols: List[str], lookback_hours: int = 72) -> None:
    """Collect news for the given crypto symbols after a download cycle."""
    if not symbols:
        return
    max_tokens = int(os.getenv("NEWS_POST_DOWNLOAD_MAX_TOKENS", "12"))
    tokens = symbols[:max_tokens]
    try:
        from datetime import datetime, timedelta, timezone
        from services.news_lab import collect_news_for_terms

        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)
        result = collect_news_for_terms(tokens=tokens, start=start, end=end)
        count = len(result.get("items", []))
        log_message("download-worker", f"post-download news: {count} articles for {tokens[:5]}...")
    except Exception as exc:
        log_message("download-worker", f"post-download news error: {exc}", severity="warning")


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

    # If all on-chain pairs are completed/skipped or CEX fallback is enabled,
    # use the CEX OHLCV fallback to download from Binance/CoinGecko.
    cex_fallback = os.getenv("OHLCV_CEX_FALLBACK", "0").lower() in {"1", "true", "yes", "on"}
    if not incomplete or cex_fallback:
        _try_cex_fallback(chain)
        if not incomplete:
            # Even if all pairs are done, trigger news for completed symbols
            _trigger_news_for_symbols(_collect_completed_symbols(assignment_path))
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
            # Check if system has room for another subprocess
            try:
                from services.resource_governor import governor, Priority
                governor.wait_if_pressured(label="download_spawn", max_wait=60.0, priority=Priority.NORMAL)
                if not governor.can_spawn_subprocess():
                    log_message("download-worker", "skipping download spawn — system under resource pressure")
                    break
            except Exception:
                pass
            proc = subprocess.Popen([resolve_python_bin(), str(script)], env=env)
            proc.wait()
    except Exception as exc:
        log_message("download-worker", f"error running download2000 for {chain}: {exc}", severity="error")

    # After downloads complete, trigger news collection for downloaded symbols
    news_enabled = os.getenv("NEWS_AFTER_DOWNLOAD", "1").lower() not in {"0", "false", "no"}
    if news_enabled:
        _trigger_news_for_symbols(_collect_completed_symbols(assignment_path))


def _try_cex_fallback(chain: str) -> None:
    """Run CEX OHLCV fallback if the historical data directory is empty or sparse."""
    ohlcv_dir = DATA_ROOT / "historical_ohlcv" / chain
    existing_count = len(list(ohlcv_dir.glob("*.json"))) if ohlcv_dir.exists() else 0
    min_threshold = int(os.getenv("CEX_FALLBACK_MIN_FILES", "5"))
    if existing_count >= min_threshold:
        return
    try:
        from services.cex_ohlcv_fallback import run_cex_fallback_cycle
        days = int(os.getenv("CEX_FALLBACK_DAYS", "90"))
        max_pairs = int(os.getenv("CEX_FALLBACK_MAX_PAIRS", "20"))
        log_message("download-worker", f"CEX fallback: {chain} has {existing_count} files, bootstrapping...")
        run_cex_fallback_cycle(chain=chain, days_back=days, max_pairs=max_pairs)
    except Exception as exc:
        log_message("download-worker", f"CEX fallback error for {chain}: {exc}", severity="error")


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
