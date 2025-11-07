from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Optional

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
        cpu_hard: float = 85.0,
        mem_soft: float = 0.70,
        mem_hard: float = 0.90,
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

    def before_task(self, component: str) -> None:
        cpu, mem = self.monitor.sample()
        now = time.time()
        if cpu >= self.cpu_hard or mem >= self.mem_hard:
            self._throttle_level = min(3.0, self._throttle_level + 1.0)
            self._last_adjust = now
        elif cpu <= self.cpu_soft and mem <= self.mem_soft and (now - self._last_adjust) > self.cool_down:
            self._throttle_level = max(0.0, self._throttle_level - 0.5)
            self._last_adjust = now
        delay = 0.0
        if self._throttle_level > 0.0:
            delay = min(5.0, 0.5 * self._throttle_level)
        if delay > 0:
            time.sleep(delay)


class APIRateLimiter:
    """
    Simple token bucket limiter per API key (usually per hostname).
    """

    def __init__(self, default_capacity: float = 5.0, default_refill_rate: float = 1.0) -> None:
        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate
        self._lock = threading.Lock()
        self._buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tokens": default_capacity, "capacity": default_capacity, "rate": default_refill_rate, "timestamp": time.time()}
        )

    def configure(self, key: str, *, capacity: float, refill_rate: float) -> None:
        with self._lock:
            bucket = self._buckets[key]
            bucket["capacity"] = capacity
            bucket["rate"] = refill_rate
            bucket["tokens"] = min(bucket["tokens"], capacity)

    def acquire(self, key: str, tokens: float = 1.0, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while True:
            with self._lock:
                bucket = self._buckets[key]
                self._refill(bucket)
                if bucket["tokens"] >= tokens:
                    bucket["tokens"] -= tokens
                    return
                wait_time = max(0.05, (tokens - bucket["tokens"]) / bucket["rate"])
            if time.time() + wait_time > deadline:
                raise TimeoutError(f"Rate limit exceeded for {key}")
            time.sleep(wait_time)

    def _refill(self, bucket: Dict[str, float]) -> None:
        now = time.time()
        elapsed = max(0.0, now - bucket["timestamp"])
        bucket["timestamp"] = now
        bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + elapsed * bucket["rate"])
