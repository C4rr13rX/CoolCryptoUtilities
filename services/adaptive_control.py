from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Optional

import os

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


class ResourceMonitor:
    def sample(self) -> tuple[float, float]:
        """
        Returns (cpu_percent, memory_utilization) as floats from 0-100 (cpu) and 0-1 (memory).
        """
        cpu = 0.0
        mem = 0.0
        if psutil:
            try:
                cpu = float(psutil.cpu_percent(interval=0.05))
                mem = float(psutil.virtual_memory().percent) / 100.0
            except Exception:
                cpu = 0.0
                mem = 0.0
        return cpu, mem


class AdaptiveLimiter:
    """
    Adjusts task speed according to CPU and RAM headroom.
    """

    def __init__(
        self,
        *,
        cpu_soft: float = 60.0,
        cpu_hard: float = 75.0,
        mem_soft: float = 0.70,
        mem_hard: float = 0.85,
        cool_down: float = 3.0,
    ) -> None:
        self.cpu_soft = cpu_soft
        self.cpu_hard = cpu_hard
        self.mem_soft = mem_soft
        self.mem_hard = mem_hard
        self.cool_down = cool_down
        self.monitor = ResourceMonitor()
        self._last_adjust = 0.0
        self._throttle_level = 0.0

    @classmethod
    def from_env(
        cls,
        *,
        cpu_soft: float = 60.0,
        cpu_hard: float = 75.0,
        mem_soft: float = 0.70,
        mem_hard: float = 0.85,
        cool_down: float = 3.0,
    ) -> "AdaptiveLimiter":
        def _get(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        return cls(
            cpu_soft=_get("ADAPTIVE_CPU_SOFT", cpu_soft),
            cpu_hard=_get("ADAPTIVE_CPU_HARD", cpu_hard),
            mem_soft=_get("ADAPTIVE_MEM_SOFT", mem_soft),
            mem_hard=_get("ADAPTIVE_MEM_HARD", mem_hard),
            cool_down=_get("ADAPTIVE_COOLDOWN_SEC", cool_down),
        )

    def before_task(self, component: str) -> None:
        cpu, mem = self.monitor.sample()
        now = time.time()
        if cpu >= self.cpu_hard or mem >= self.mem_hard:
            self._throttle_level = min(4.0, self._throttle_level + 1.0)
            self._last_adjust = now
        elif cpu <= self.cpu_soft and mem <= self.mem_soft and (now - self._last_adjust) > self.cool_down:
            self._throttle_level = max(0.0, self._throttle_level - 0.5)
            self._last_adjust = now
        delay = 0.0
        if self._throttle_level > 0.0:
            delay = min(6.0, 1.0 * self._throttle_level)
        if delay > 0:
            time.sleep(delay)


class APIRateLimiter:
    """
    Simple token bucket limiter per API key (usually per hostname) with
    lightweight penalty/healing controls so misbehaving hosts back off
    automatically (e.g., when remote APIs respond with 429/451).
    """

    def __init__(self, default_capacity: float = 5.0, default_refill_rate: float = 1.0) -> None:
        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate
        self._lock = threading.Lock()
        self._buckets: Dict[str, Dict[str, float]] = defaultdict(self._bucket_factory)

    def _bucket_factory(self) -> Dict[str, float]:
        now = time.time()
        return {
            "tokens": self.default_capacity,
            "capacity": self.default_capacity,
            "rate": self.default_refill_rate,
            "base_capacity": self.default_capacity,
            "base_rate": self.default_refill_rate,
            "timestamp": now,
            "penalty_until": 0.0,
        }

    def configure(self, key: str, *, capacity: float, refill_rate: float) -> None:
        with self._lock:
            bucket = self._buckets[key]
            bucket["capacity"] = capacity
            bucket["rate"] = refill_rate
            bucket["base_capacity"] = capacity
            bucket["base_rate"] = refill_rate
            bucket["tokens"] = min(bucket["tokens"], capacity)

    def acquire(self, key: str, tokens: float = 1.0, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while True:
            with self._lock:
                bucket = self._buckets[key]
                now = time.time()
                self._refill(bucket, now)
                penalty_until = bucket.get("penalty_until", 0.0)
                if penalty_until and now < penalty_until:
                    wait_time = penalty_until - now
                elif bucket["tokens"] >= tokens:
                    bucket["tokens"] -= tokens
                    return
                else:
                    rate = max(bucket["rate"], 1e-6)
                    wait_time = max(0.05, (tokens - bucket["tokens"]) / rate)
            if time.time() + wait_time > deadline:
                raise TimeoutError(f"Rate limit exceeded for {key}")
            time.sleep(wait_time)

    def penalize(self, key: str, *, cooldown: float = 5.0, drain: float = 0.5) -> None:
        """
        Temporarily throttle a host after hard failures (e.g., HTTP 429).
        """
        if cooldown <= 0 and drain <= 0:
            return
        with self._lock:
            bucket = self._buckets[key]
            bucket["tokens"] = max(0.0, bucket["tokens"] - max(0.0, drain))
            penalty_until = max(bucket.get("penalty_until", 0.0), time.time() + max(0.0, cooldown))
            bucket["penalty_until"] = penalty_until

    def heal(self, key: str, *, boost: float = 0.25) -> None:
        """
        Clears any pending penalty and optionally tops up the bucket when an
        endpoint starts succeeding again.
        """
        with self._lock:
            bucket = self._buckets[key]
            bucket["penalty_until"] = 0.0
            if boost > 0:
                bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + boost)

    def _refill(self, bucket: Dict[str, float], now: float) -> None:
        elapsed = max(0.0, now - bucket["timestamp"])
        bucket["timestamp"] = now
        bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + elapsed * bucket["rate"])
        penalty_until = bucket.get("penalty_until", 0.0)
        if penalty_until and now >= penalty_until:
            bucket["penalty_until"] = 0.0
