from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from db import TradingDatabase, get_db


def _load_assignment(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data.setdefault("pairs", {})
        return data
    except Exception as exc:
        print(f"[download-worker] failed to read {path}: {exc}")
        return None


def _ensure_assignment_template(chain: str, assignment_path: Path) -> Dict[str, Any]:
    data = _load_assignment(assignment_path)
    if data is not None:
        data.setdefault("chain", chain)
        return data
    index_path = Path("data") / f"pair_index_{chain}.json"
    if not index_path.exists():
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
    if not incomplete:
        return
    env = os.environ.copy()
    env["CHAIN_NAME"] = chain
    env["PAIR_ASSIGNMENT_FILE"] = str(assignment_path)
    env.setdefault("OUTPUT_DIR", str(Path("data") / "historical_ohlcv" / chain))
    env.setdefault("INTERMEDIATE_DIR", str(Path("data") / "intermediate" / chain))
    script = Path(__file__).resolve().parents[1] / "download2000.py"
    try:
        subprocess.run([sys.executable, str(script)], env=env, check=False)
    except Exception as exc:
        print(f"[download-worker] error running download2000 for {chain}: {exc}")


class DownloadWorker:
    def __init__(self, chain: str, interval_sec: int = 4 * 3600, assignment_path: Optional[Path] = None) -> None:
        self.chain = chain
        self.assignment_path = assignment_path or Path("data") / f"{chain}_pair_provider_assignment.json"
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
            print(f"[download-worker] run_once error for {self.chain}: {exc}")

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

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_cycle()
            except Exception as exc:
                print(f"[download-worker] dynamic cycle error: {exc}")
            if self._stop_event.wait(self.interval):
                break

    def _run_cycle(self) -> None:
        rows = self.db.fetch_recent_pairs(horizon_sec=self.horizon, limit=500)
        chains: Dict[str, Set[str]] = defaultdict(set)
        for row in rows:
            chain = str(row.get("chain") or "").lower()
            symbol = str(row.get("symbol") or "").upper()
            if not chain or not symbol or chain == "base":
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
        pair_index_path = Path("data") / f"pair_index_{chain}.json"
        if not pair_index_path.exists():
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


class TokenDownloadSupervisor:
    def __init__(self, db: Optional[TradingDatabase] = None) -> None:
        self.base_worker = DownloadWorker("base", interval_sec=int(os.getenv("BASE_DOWNLOAD_INTERVAL", "10800")))
        self.dynamic_worker = DynamicDownloadWorker(
            db=db,
            interval_sec=int(os.getenv("DYNAMIC_DOWNLOAD_INTERVAL", "21600")),
            horizon_sec=int(os.getenv("DYNAMIC_DOWNLOAD_HORIZON", str(3 * 24 * 3600))),
        )

    def start(self) -> None:
        self.base_worker.start()
        self.dynamic_worker.start()

    def stop(self) -> None:
        self.base_worker.stop()
        self.dynamic_worker.stop()
