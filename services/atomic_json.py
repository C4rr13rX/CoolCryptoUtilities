"""Cross-process safe read-modify-write for the small JSON files that gate money.

Two files decide whether a strategy is allowed to spend real funds:
``data/strategy_ledger.json`` (the rolling promotion window) and
``data/strategy_registry.json`` (the append-only lifetime record). Both are
written by several processes at once -- the trading bot holds one handle from
startup, services/atf_static_strategy.py builds a fresh one per outcome, and
the web workers open their own -- and both were written with the same three
defects:

  * a ``threading.Lock`` guarding a read-modify-write, which orders writers
    inside ONE interpreter and nothing at all between processes;
  * a temp file named the same for every writer, so two concurrent saves wrote
    over each other's half-finished file and raced to rename it;
  * ``os.replace`` and ``read_text`` called once, with the exception swallowed
    -- and on Windows both fail with PermissionError whenever another handle
    holds the file open, which under concurrency is most of the time.

Measured 2026-09-02: forty outcomes recorded through separate ledger instances
left TWO in the file. That is a 95% loss on the evidence that gates graduation,
and it is why money_button was absent from the ledger entirely while the
database held its trade, and why twelve of rsi_reversal@5h's thirteen losses
were missing. With the lock alone, 5 of 40 were still lost; with the retries as
well, 320 of 320 survived.

The lock is an O_EXCL lock file rather than fcntl/msvcrt because it behaves
identically on both platforms this runs on.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

#: Retries for the two operations Windows fails while another handle is open.
LOAD_ATTEMPTS = 4
SAVE_ATTEMPTS = 6


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@contextmanager
def file_lock(path: Path, *, timeout: Optional[float] = None, stale_after: Optional[float] = None):
    """Hold an exclusive cross-process lock for ``path`` while the body runs."""
    lock_path = path.with_name(path.name + ".lock")
    timeout = _env_float("JSON_STORE_LOCK_TIMEOUT_SEC", 10.0) if timeout is None else timeout
    stale_after = (
        _env_float("JSON_STORE_LOCK_STALE_SEC", 30.0) if stale_after is None else stale_after
    )
    deadline = time.time() + timeout
    acquired = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("ascii", "replace"))
            finally:
                os.close(fd)
            acquired = True
            break
        except OSError:
            # FileExistsError means somebody holds it. On Windows an os.open
            # racing another process's unlink of the same name raises
            # PermissionError instead, and treating that as "unlockable"
            # rather than "try again" silently dropped the lock and put the
            # write straight back into the race it was meant to leave.
            try:
                if time.time() - lock_path.stat().st_mtime > stale_after:
                    # A process killed mid-write must not deadlock promotion.
                    lock_path.unlink()
                    continue
            except OSError:
                pass
            if time.time() >= deadline:
                break
            time.sleep(0.005)
    try:
        yield acquired
    finally:
        if acquired:
            try:
                lock_path.unlink()
            except OSError:
                pass


def read_json(path: Path, default: Any = None) -> tuple[Any, bool]:
    """``(data, ok)``. ``ok`` is False when the file exists but could not be read.

    A failed read must never be reported as empty data. Callers write back what
    they read, so "unreadable" turning into "{}" persists an empty file over
    every record in it -- losing not one outcome but the entire history.
    """
    for attempt in range(LOAD_ATTEMPTS):
        try:
            if not path.exists():
                return (default, True)
            raw = json.loads(path.read_text(encoding="utf-8"))
            return (raw, True)
        except OSError:
            # Windows refuses the read while another writer is mid-replace.
            time.sleep(0.005 * (attempt + 1))
        except ValueError:
            return (default, False)      # truncated / not JSON
    return (default, False)


def write_json(path: Path, data: Any) -> bool:
    """Atomically replace ``path``. Returns False if the write did not land."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    # A temp name unique to this writer. A shared one meant two concurrent
    # saves wrote over each other's half-written file before renaming it.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
    for attempt in range(SAVE_ATTEMPTS):
        try:
            tmp.replace(path)
            return True
        except OSError:
            # os.replace fails with PermissionError while ANY other handle has
            # the destination open, and these files are read without the write
            # lock by every construction. Measured under 60 concurrent
            # outcomes: 8 lost, every one of them this call, silently ignored.
            time.sleep(0.005 * (attempt + 1))
    try:
        tmp.unlink()
    except OSError:
        pass
    return False
