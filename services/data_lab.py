from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from datetime import datetime

from services.news_lab import collect_news_for_terms

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
HIST_ROOT = DATA_ROOT / "historical_ohlcv"


@dataclass
class DatasetEntry:
    rank: int
    category: str
    path: str
    chain: str
    symbol: str
    size_bytes: int
    modified_ts: float

    @property
    def modified_iso(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.modified_ts))

    @property
    def size_human(self) -> str:
        size = float(self.size_bytes)
        units = ["B", "KB", "MB", "GB", "TB"]
        idx = 0
        while size >= 1024 and idx < len(units) - 1:
            size /= 1024
            idx += 1
        return f"{size:.2f} {units[idx]}"


def _iter_historical(chain: Optional[str] = None) -> Iterable[Tuple[str, str, Path]]:
    if not HIST_ROOT.exists():
        return
    if chain:
        chains = [chain.lower()]
    else:
        chains = [p.name for p in HIST_ROOT.iterdir() if p.is_dir()]
    for ch in chains:
        dir_path = HIST_ROOT / ch
        if not dir_path.is_dir():
            continue
        for file_path in dir_path.glob("*.json"):
            symbol = file_path.stem.split("_", 1)[-1].upper()
            yield "historical", ch, file_path


def _iter_pair_indexes() -> Iterable[Tuple[str, str, Path]]:
    if not DATA_ROOT.exists():
        return
    for file_path in DATA_ROOT.glob("pair_index_*.json"):
        chain = file_path.stem.replace("pair_index_", "").lower()
        yield "pair_index", chain, file_path


def _iter_assignments() -> Iterable[Tuple[str, str, Path]]:
    if not DATA_ROOT.exists():
        return
    for file_path in DATA_ROOT.glob("*_pair_provider_assignment.json"):
        chain = file_path.stem.replace("_pair_provider_assignment", "").lower()
        yield "assignment", chain, file_path


def list_datasets(
    *,
    chain: Optional[str] = None,
    category: Optional[str] = None,
    sort_key: str = "modified",
    order: str = "desc",
) -> List[Dict[str, Any]]:
    entries: List[Tuple[str, str, Path]] = []
    if category in (None, "historical"):
        entries.extend(_iter_historical(chain))
    if category in (None, "pair_index"):
        entries.extend(_iter_pair_indexes())
    if category in (None, "assignment"):
        entries.extend(_iter_assignments())

    dataset_rows: List[DatasetEntry] = []
    for cat, ch, path in entries:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        symbol = path.stem.split("_", 1)[-1].upper() if cat == "historical" else path.stem
        dataset_rows.append(
            DatasetEntry(
                rank=0,
                category=cat,
                path=str(path.relative_to(REPO_ROOT)),
                chain=ch,
                symbol=symbol,
                size_bytes=stat.st_size,
                modified_ts=stat.st_mtime,
            )
        )

    reverse = order.lower() != "asc"
    if sort_key == "size":
        dataset_rows.sort(key=lambda row: row.size_bytes, reverse=reverse)
    elif sort_key == "symbol":
        dataset_rows.sort(key=lambda row: (row.symbol, row.chain), reverse=reverse)
    else:
        dataset_rows.sort(key=lambda row: row.modified_ts, reverse=reverse)

    for idx, row in enumerate(dataset_rows, start=1):
        row.rank = idx

    return [
        {
            "rank": row.rank,
            "category": row.category,
            "path": row.path,
            "chain": row.chain,
            "symbol": row.symbol,
            "size_bytes": row.size_bytes,
            "size_human": row.size_human,
            "modified": row.modified_ts,
            "modified_iso": row.modified_iso,
        }
        for row in dataset_rows
    ]


class DataLabRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._status: Dict[str, Any] = {
            "running": False,
            "job_type": None,
            "options": {},
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "message": "idle",
            "log": [],
            "history": [],
        }
        self._history: List[Dict[str, Any]] = []

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def start(self, job_type: str, options: Dict[str, Any]) -> None:
        with self._lock:
            if self._status.get("running"):
                raise RuntimeError("Data lab job already running.")
            self._status.update(
                {
                    "running": True,
                    "job_type": job_type,
                    "options": options,
                    "started_at": time.time(),
                    "finished_at": None,
                    "returncode": None,
                    "message": "initialising",
                    "log": [],
                    "history": list(self._history),
                }
            )
        self._append_log(f"Queued job `{job_type}` with options: {json.dumps(options, sort_keys=True)}")
        thread = threading.Thread(target=self._run_job, args=(job_type, options), daemon=True)
        self._thread = thread
        thread.start()

    def _run_job(self, job_type: str, options: Dict[str, Any]) -> None:
        env = self._prepare_env(job_type, options)
        command = self._build_command(job_type, options)
        if not command:
            self._append_log(f"Unknown job type requested: {job_type}")
            self._set_status(False, message=f"Unknown job type: {job_type}")
            return
        self._append_log(f"Starting job `{job_type}` with command: {' '.join(command)}")
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self._append_log(f"Failed to start job: {exc}")
            self._set_status(False, message=f"Failed to start job: {exc}")
            return

        assert proc.stdout is not None
        for line in proc.stdout:
            self._append_log(line.rstrip())
        proc.wait()
        success = proc.returncode == 0
        if success:
            self._append_log("Job completed successfully.")
        else:
            self._append_log(f"Job failed with exit code {proc.returncode}.")
        message = "completed successfully" if success else f"job failed with code {proc.returncode}"
        self._set_status(success, message=message, returncode=proc.returncode)

    def _append_log(self, line: str) -> None:
        with self._lock:
            log = self._status.get("log", [])
            log.append(line)
            if len(log) > 400:
                log = log[-400:]
            self._status["log"] = log
            self._status["history"] = list(self._history)

    def _record_history(self, success: bool, message: str, returncode: Optional[int]) -> None:
        with self._lock:
            entry = {
                "job_type": self._status.get("job_type"),
                "options": self._status.get("options") or {},
                "started_at": self._status.get("started_at"),
                "finished_at": self._status.get("finished_at"),
                "status": "success" if success else "failure",
                "message": message,
                "returncode": returncode,
                "log": list(self._status.get("log", [])),
            }
            self._history.append(entry)
            self._history = self._history[-50:]
            self._status["history"] = list(self._history)

    def _set_status(self, success: bool, *, message: str, returncode: Optional[int] = None) -> None:
        with self._lock:
            self._status.update(
                {
                    "running": False,
                    "message": message,
                    "returncode": returncode,
                    "finished_at": time.time(),
                }
            )
            self._status["history"] = list(self._history)
        self._record_history(success, message, returncode)

    def _prepare_env(self, job_type: str, options: Dict[str, Any]) -> Dict[str, str]:
        env = os.environ.copy()
        chain = (options.get("chain") or "base").strip().lower()
        env["CHAIN_NAME"] = chain
        years_back = int(options.get("years_back") or 3)
        granularity = int(options.get("granularity_seconds") or 300)
        pair_index = options.get("pair_index_file") or f"data/pair_index_{chain}.json"
        assignment_file = options.get("assignment_file") or f"data/{chain}_pair_provider_assignment.json"
        output_dir = options.get("output_dir") or str(HIST_ROOT / chain)
        env["PAIR_INDEX_FILE"] = pair_index
        env["PAIR_ASSIGNMENT_FILE"] = assignment_file
        env["OUTPUT_DIR"] = output_dir
        env["YEARS_BACK"] = str(years_back)
        env["GRANULARITY_SECONDS"] = str(granularity)

        if options.get("intermediate_dir"):
            env["INTERMEDIATE_DIR"] = str(options["intermediate_dir"])

        if options.get("graph_url"):
            env["THEGRAPH_SUBGRAPH_URL"] = str(options["graph_url"])
        if options.get("thegraph_api_key"):
            env["THEGRAPH_API_KEY"] = str(options["thegraph_api_key"])
        if options.get("subgraph_id"):
            env["UNISWAP_V2_SUBGRAPH_ID"] = str(options["subgraph_id"])
        if options.get("max_workers"):
            env["MAX_WORKERS"] = str(int(options["max_workers"]))
        if options.get("ankr_api_key"):
            env["ANKR_API_KEY"] = str(options["ankr_api_key"])
        if options.get("rpc_url"):
            env["ANKR_RPC_URL"] = str(options["rpc_url"])
        return env

    def _build_command(self, job_type: str, options: Dict[str, Any]) -> Optional[List[str]]:
        python = options.get("python_bin") or "python3"
        if job_type == "make2000index":
            return [python, "make2000index.py"]
        if job_type in {"make_assignments", "assignment"}:
            return [python, "makeServiceAssignment.py"]
        if job_type == "download2000":
            return [python, "download2000.py"]
        return None


_RUNNER_SINGLETON: Optional[DataLabRunner] = None
_RUNNER_LOCK = threading.Lock()


def get_runner() -> DataLabRunner:
    global _RUNNER_SINGLETON
    with _RUNNER_LOCK:
        if _RUNNER_SINGLETON is None:
            _RUNNER_SINGLETON = DataLabRunner()
    return _RUNNER_SINGLETON


def fetch_news(
    *,
    tokens: Sequence[str],
    start: datetime,
    end: datetime,
    query: Optional[str] = None,
    max_pages: Optional[int] = None,
) -> Dict[str, Any]:
    result = collect_news_for_terms(tokens=tokens, start=start, end=end, query=query, max_pages=max_pages)
    return result
