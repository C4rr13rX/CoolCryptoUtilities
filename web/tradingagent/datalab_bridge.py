"""Let the agent fetch the data it says it is missing, and choose what to watch.

Two failures this exists to prevent.

The agent reasons about whatever it happens to be shown. If a token is not on
the watchlist it is invisible, so a decision to "stay on top of bulls and
bears" is only as good as the list -- and the list was static. Opportunities
were missed because nothing was looking.

The opposite failure is worse: an agent that can start any job whenever it
likes will start them constantly, and this box already had a download flood
saturate its connection budget. So requests are rate-limited, the job list is
an allowlist rather than a passthrough, and a request that is already running
is refused rather than queued.

Watchlist changes are ADDITIVE by default. Replacing a watchlist would let one
bad pass discard symbols other strategies are mid-position on, and the cost of
watching a symbol too long is a little bandwidth against losing a position we
cannot see.
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Jobs the agent may start. An allowlist, not a passthrough: an agent that can
#: name any job type can eventually name one that costs money or mutates state.
ALLOWED_JOBS = {
    "download2000": "Fetch OHLCV history for the configured pairs.",
    "make2000index": "Rebuild the pair index from what has been downloaded.",
    "make_assignments": "Reassign which provider serves which pair.",
}

#: A job may not be restarted more often than this. The runner is a singleton
#: with one slot, and hammering it starves the streams that share the box.
MIN_JOB_INTERVAL_SEC = 900.0

#: How many symbols the agent may add in one pass. Unbounded, it will happily
#: watch everything and thin the feed for every symbol.
MAX_ADDS_PER_PASS = 8

_last_job_at: Dict[str, float] = {}


def _job_already_running(job_type: str) -> Optional[int]:
    """PID of a job of this type already running, or None.

    Reads the process table rather than any in-process bookkeeping, because
    the thing that must not happen -- two copies competing for the same
    upstream endpoints -- is a property of the machine, not of this worker.

    Returns None when the table cannot be read: an unreadable process list is
    not evidence that nothing is running, but refusing every job because a
    query failed would be worse than the duplicate it is guarding against.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=45).stdout or ""
    except Exception:  # noqa: BLE001
        return None
    if not out.strip():
        return None

    try:
        rows = json.loads(out)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(rows, dict):
        rows = [rows]

    needle = job_type.lower()
    for row in rows:
        if not isinstance(row, dict):
            continue
        cmd = str(row.get("CommandLine") or "").lower()
        if needle and needle in cmd:
            try:
                return int(row.get("ProcessId") or 0) or None
            except (TypeError, ValueError):
                return None
    return None


def available_jobs() -> Dict[str, str]:
    return dict(ALLOWED_JOBS)


def job_status() -> Dict[str, Any]:
    try:
        from services.data_lab import get_runner

        return get_runner().status() or {}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def request_job(job_type: str, options: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Any]:
    """Start a Data Lab job on the agent's behalf, if it is allowed to.

    Returns a dict describing what happened -- never raises, because a refused
    data request must not end the pass.
    """
    job_type = str(job_type or "").strip()
    if job_type not in ALLOWED_JOBS:
        return {"ok": False, "reason": "job_not_allowed",
                "detail": f"{job_type!r} is not one of {sorted(ALLOWED_JOBS)}"}

    now = time.time()
    since = now - _last_job_at.get(job_type, 0.0)
    if since < MIN_JOB_INTERVAL_SEC:
        return {"ok": False, "reason": "rate_limited",
                "detail": f"{job_type} ran {int(since)}s ago; "
                          f"minimum interval is {int(MIN_JOB_INTERVAL_SEC)}s"}

    # IS ONE ALREADY RUNNING ON THE MACHINE?
    #
    # _last_job_at lives in this process's memory, and the runner's own
    # status() only knows about jobs IT started. Neither survives a worker
    # restart, and neither sees a job launched by anything else -- so the
    # interval was enforced against a clock that resets and a registry that
    # is not authoritative.
    #
    # Measured 2026-09-05: TWO download2000 processes were running at once,
    # and the agent worker held 156 TCP connections (75 established to
    # Cloudflare/AWS) for a task that runs once every 15 minutes. Those hit
    # the SAME endpoints the live price feed polls, and the feed collapsed
    # from 130 ticks per 10 minutes to zero, logging "network outage
    # detected; pausing live connections" while all three endpoints were in
    # fact reachable. No feed means no entries, no exits, and no live trades.
    #
    # The process table is the one source that is actually true.
    running = _job_already_running(job_type)
    if running:
        return {"ok": False, "reason": "already_running",
                "detail": f"{job_type} is already running as PID {running}; "
                          f"a second copy would compete with the live price "
                          f"feed for the same endpoints"}

    try:
        from services.data_lab import get_runner

        runner = get_runner()
        current = runner.status() or {}
        if current.get("running"):
            # Refuse rather than queue: a queued job runs at a moment nobody
            # chose, against data nobody looked at.
            return {"ok": False, "reason": "busy",
                    "detail": f"{current.get('job_type')} is already running"}

        runner.start(job_type, options or {})
        _last_job_at[job_type] = now
        return {"ok": True, "job_type": job_type, "status": runner.status()}
    except RuntimeError as exc:
        return {"ok": False, "reason": "conflict", "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": "error",
                "detail": f"{type(exc).__name__}: {exc}"}


def current_watchlist() -> Dict[str, List[str]]:
    try:
        from services.watchlists import load_watchlists

        data = load_watchlists() or {}
        return {k: list(v or []) for k, v in data.items() if isinstance(v, list)}
    except Exception:
        return {}


def add_symbols(symbols: List[str], *, target: str = "stream") -> Dict[str, Any]:
    """Watch these symbols too. Additive, and capped.

    Additive on purpose: replacing a watchlist would let one pass discard
    symbols another strategy is holding a position on, and a symbol watched
    too long costs bandwidth while a symbol dropped too early costs a position
    we can no longer see.
    """
    cleaned: List[str] = []
    for raw in (symbols or []):
        symbol = str(raw or "").strip().upper()
        # Shape check, not a truth claim: the contract guard decides whether a
        # token is real. This only rejects things that are not symbols at all.
        if not symbol or len(symbol) > 32 or " " in symbol:
            continue
        if symbol not in cleaned:
            cleaned.append(symbol)
        if len(cleaned) >= MAX_ADDS_PER_PASS:
            break

    if not cleaned:
        return {"ok": False, "reason": "nothing_valid", "added": []}

    try:
        from services.watchlists import mutate_watchlist

        updated = mutate_watchlist(target, add=cleaned)
        return {"ok": True, "added": cleaned, "watchlists": updated}
    except ValueError as exc:
        return {"ok": False, "reason": "bad_target", "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": "error",
                "detail": f"{type(exc).__name__}: {exc}"}


def candidate_tokens(limit: int = 20) -> List[Dict[str, Any]]:
    """Tokens worth considering, ranked by how much they are actually moving.

    Deliberately reads the same stream the executor trades on. A candidate the
    executor cannot price is not an opportunity, it is a way to open a position
    that can never be closed -- which is exactly how capital got stranded here
    before.
    """
    out: List[Dict[str, Any]] = []
    try:
        import sqlite3

        conn = sqlite3.connect(
            f"file:{ROOT / 'storage' / 'trading_cache.db'}?mode=ro", uri=True)
        now = time.time()
        rows = list(conn.execute(
            "SELECT symbol, COUNT(*) n, MIN(price) lo, MAX(price) hi "
            "FROM market_stream WHERE ts > ? GROUP BY symbol "
            "HAVING n >= 5 ORDER BY n DESC LIMIT ?", (now - 3600, limit * 3)))

        watched = set()
        for symbols in current_watchlist().values():
            watched.update(s.upper() for s in symbols)

        for symbol, n, lo, hi in rows:
            try:
                lo, hi = float(lo), float(hi)
            except (TypeError, ValueError):
                continue
            if lo <= 0:
                continue
            volatility = ((hi / lo) - 1.0) * 100.0
            out.append({
                "symbol": symbol,
                "ticks_1h": int(n),
                "volatility_1h_pct": round(volatility, 4),
                "watched": symbol.upper() in watched,
            })

        # Movement first: a symbol that does not move cannot be traded
        # profitably against a round-trip fee, however well covered it is.
        out.sort(key=lambda r: r["volatility_1h_pct"], reverse=True)
        return out[:limit]
    except Exception:
        return out
