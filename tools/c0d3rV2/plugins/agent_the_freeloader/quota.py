from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .models import PoolLimit


@dataclass
class Reservation:
    reservation_id: str
    pool_ids: tuple[str, ...]


class QuotaLedger:
    """Thread-safe, persistent multi-pool quota accounting.

    A model can belong to more than one pool.  A request is eligible only
    when every pool has capacity.  Shared provider quotas therefore block the
    entire provider family while model-specific quotas block only one model.
    """

    def __init__(
        self,
        limits: Mapping[str, PoolLimit],
        *,
        state_path: str | Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.limits = dict(limits)
        self.state_path = Path(state_path) if state_path else None
        self.clock = clock
        self._lock = threading.RLock()
        self._events: dict[str, list[dict]] = {}
        self._blocked_until: dict[str, float] = {}
        self._external_headroom: dict[str, float] = {}
        self._load()

    def available(self, pool_ids: tuple[str, ...], request_tokens: int) -> bool:
        with self._guard():
            now = self.clock()
            self._prune(now)
            return all(
                self._pool_headroom(pool_id, request_tokens, now) > 0.0
                for pool_id in pool_ids
            )

    def headroom(self, pool_ids: tuple[str, ...], request_tokens: int) -> float:
        with self._guard():
            now = self.clock()
            self._prune(now)
            values = [self._pool_headroom(pool_id, request_tokens, now) for pool_id in pool_ids]
            return min(values) if values else 1.0

    def reserve(self, pool_ids: tuple[str, ...], request_tokens: int) -> Reservation:
        with self._guard():
            now = self.clock()
            self._prune(now)
            if not all(
                self._pool_headroom(pool_id, request_tokens, now) > 0.0
                for pool_id in pool_ids
            ):
                raise RuntimeError("quota pool has no remaining local capacity")
            reservation_id = uuid.uuid4().hex
            for pool_id in pool_ids:
                event = {
                    "id": reservation_id,
                    "ts": now,
                    "requests": 1,
                    "tokens": max(0, int(request_tokens)),
                }
                self._events.setdefault(pool_id, []).append(event)
            self._save()
            return Reservation(reservation_id=reservation_id, pool_ids=pool_ids)

    def reconcile(self, reservation: Reservation, actual_tokens: int | None) -> None:
        if actual_tokens is None:
            return
        with self._guard():
            for pool_id in reservation.pool_ids:
                for event in self._events.get(pool_id, []):
                    if event.get("id") == reservation.reservation_id:
                        event["tokens"] = max(0, int(actual_tokens))
            self._save()

    def release(self, reservation: Reservation) -> None:
        """Release a reservation when a request failed before inference."""
        with self._guard():
            for pool_id in reservation.pool_ids:
                self._events[pool_id] = [
                    event for event in self._events.get(pool_id, [])
                    if event.get("id") != reservation.reservation_id
                ]
            self._save()

    def block(self, pool_ids: tuple[str, ...], retry_after_s: float = 60.0) -> None:
        with self._guard():
            until = self.clock() + max(1.0, retry_after_s)
            for pool_id in pool_ids:
                self._blocked_until[pool_id] = max(until, self._blocked_until.get(pool_id, 0.0))
            self._save()

    def observe_headers(self, pool_ids: tuple[str, ...], headers: Mapping[str, str]) -> None:
        normalized = {str(key).lower(): str(value) for key, value in headers.items()}
        ratios: list[float] = []
        for kind in ("requests", "tokens"):
            limit = _first_number(normalized, (f"x-ratelimit-limit-{kind}", f"x-ratelimit-limit-{kind}-day"))
            remaining = _first_number(normalized, (f"x-ratelimit-remaining-{kind}", f"x-ratelimit-remaining-{kind}-day"))
            if limit and remaining is not None:
                ratios.append(max(0.0, min(1.0, remaining / limit)))
        if not ratios:
            return
        observed = min(ratios)
        with self._guard():
            for pool_id in pool_ids:
                self._external_headroom[pool_id] = observed

    def snapshot(self) -> dict:
        with self._guard():
            now = self.clock()
            self._prune(now)
            return {
                "pools": {
                    pool_id: {
                        "events": len(events),
                        "blocked_until": self._blocked_until.get(pool_id, 0.0),
                        "headroom": self._pool_headroom(pool_id, 0, now),
                    }
                    for pool_id, events in self._events.items()
                }
            }

    def usage_since(self, seconds: float = 86_400.0) -> dict[str, int]:
        """Return de-duplicated successful reservations in a time window."""
        with self._guard():
            now = self.clock()
            self._prune(now)
            reservations: dict[str, dict[str, int]] = {}
            for pool_id, events in self._events.items():
                for index, event in enumerate(events):
                    if float(event.get("ts", 0.0)) <= now - seconds:
                        continue
                    key = str(event.get("id") or f"{pool_id}:{index}:{event.get('ts')}")
                    current = reservations.setdefault(key, {"requests": 0, "tokens": 0})
                    # One reservation is duplicated across attached pools; max
                    # preserves one logical request instead of double counting.
                    current["requests"] = max(current["requests"], int(event.get("requests", 0)))
                    current["tokens"] = max(current["tokens"], int(event.get("tokens", 0)))
            return {
                "requests": sum(item["requests"] for item in reservations.values()),
                "tokens": sum(item["tokens"] for item in reservations.values()),
            }

    def _pool_headroom(self, pool_id: str, request_tokens: int, now: float) -> float:
        if self._blocked_until.get(pool_id, 0.0) > now:
            return 0.0
        limit = self.limits.get(pool_id, PoolLimit())
        events = self._events.get(pool_id, [])
        ratios = [self._external_headroom.get(pool_id, 1.0)]
        checks = (
            (limit.requests_per_minute, 60.0, "requests", 1),
            (limit.requests_per_day, 86_400.0, "requests", 1),
            (limit.requests_per_month, 30 * 86_400.0, "requests", 1),
            (limit.tokens_per_minute, 60.0, "tokens", request_tokens),
            (limit.tokens_per_day, 86_400.0, "tokens", request_tokens),
        )
        for maximum, window, field, requested in checks:
            if maximum is None:
                continue
            used = sum(int(event.get(field, 0)) for event in events if event["ts"] > now - window)
            remaining = maximum - used
            if remaining < requested:
                return 0.0
            ratios.append(max(0.0, min(1.0, remaining / maximum)))
        return min(ratios)

    def _prune(self, now: float) -> None:
        cutoff = now - 31 * 86_400.0
        for pool_id, events in list(self._events.items()):
            self._events[pool_id] = [event for event in events if float(event.get("ts", 0)) > cutoff]
        self._blocked_until = {
            pool_id: until for pool_id, until in self._blocked_until.items() if until > now
        }

    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._events = payload.get("events", {})
            self._blocked_until = payload.get("blocked_until", {})
        except Exception:
            self._events = {}
            self._blocked_until = {}

    @contextmanager
    def _guard(self):
        """Serialize quota decisions across threads and worker processes."""
        with self._lock:
            if not self.state_path:
                yield
                return
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
            with lock_path.open("a+b") as lock_file:
                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"0")
                    lock_file.flush()
                lock_file.seek(0)
                _lock_file(lock_file)
                try:
                    self._load()
                    yield
                finally:
                    _unlock_file(lock_file)

    def _save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        payload = {"events": self._events, "blocked_until": self._blocked_until}
        temp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temp.replace(self.state_path)


def _first_number(headers: Mapping[str, str], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = headers.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _lock_file(handle) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
