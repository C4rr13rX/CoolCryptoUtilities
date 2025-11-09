from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - platform-specific
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore

try:  # pragma: no cover - platform-specific
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None  # type: ignore


class GuardianLease:
    """
    Cross-process advisory lock backed by a lock file. Used to ensure the guardian
    (and Codex sessions it launches) never overlap, even if multiple Django
    processes spin up simultaneously.
    """

    def __init__(
        self,
        name: str,
        *,
        timeout: Optional[float] = None,
        poll_interval: float = 1.0,
    ) -> None:
        self.name = name
        self.timeout = timeout
        self.poll_interval = poll_interval
        lock_root = Path("runtime/guardian")
        lock_root.mkdir(parents=True, exist_ok=True)
        self.lock_path = lock_root / f"{name}.lock"
        self._handle: Optional[object] = None
        self._acquired = False

    # ------------------------------------------------------------------ helpers
    def _try_lock(self) -> bool:
        if fcntl:
            return self._try_lock_posix()
        if msvcrt:
            return self._try_lock_windows()
        return self._try_lock_portable()

    def _try_lock_posix(self) -> bool:  # pragma: no cover - requires POSIX
        assert fcntl is not None
        self._handle = self.lock_path.open("w+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._handle.close()
            self._handle = None
            return False
        self._handle.write(str(os.getpid()))
        self._handle.flush()
        self._handle.truncate()
        return True

    def _try_lock_windows(self) -> bool:  # pragma: no cover - requires Windows
        assert msvcrt is not None
        self._handle = self.lock_path.open("w+")
        try:
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self._handle.close()
            self._handle = None
            return False
        self._handle.write(str(os.getpid()))
        self._handle.flush()
        self._handle.truncate()
        return True

    def _try_lock_portable(self) -> bool:
        # best-effort fallback using exclusive file creation
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            return False
        self._handle = os.fdopen(fd, "w+")
        self._handle.write(str(os.getpid()))
        self._handle.flush()
        return True

    def _unlock(self) -> None:
        handle = self._handle
        self._handle = None
        if not handle:
            return
        try:
            if fcntl:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
            elif msvcrt:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # pragma: no cover - Windows
        finally:
            try:
                handle.close()
            finally:
                if not fcntl:
                    # portable fallback uses exclusive file creation, so remove on release
                    try:
                        self.lock_path.unlink()
                    except FileNotFoundError:
                        pass

    # ------------------------------------------------------------------ API
    def acquire(self, *, cancel_event: Optional[threading.Event] = None) -> bool:
        deadline = time.time() + self.timeout if self.timeout is not None else None
        while True:
            if self._try_lock():
                self._acquired = True
                return True
            if cancel_event and cancel_event.is_set():
                return False
            if deadline is not None and time.time() >= deadline:
                return False
            time.sleep(self.poll_interval)

    def release(self) -> None:
        if not self._acquired:
            return
        self._acquired = False
        self._unlock()

    # Context manager support ---------------------------------------------------
    def __enter__(self):
        acquired = self.acquire()
        if not acquired:
            raise TimeoutError(f"guardian lease '{self.name}' unavailable")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
