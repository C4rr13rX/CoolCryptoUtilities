"""
Dynamic resource governor — system-wide singleton that adapts thread pools,
worker counts, and task pacing to real-time CPU/memory availability.

Design goals:
  - Leave headroom for the user's other programs (Chrome, WSL, GIMP, etc.).
  - When the system is idle, allow the application to use most resources.
  - When pressure rises, smoothly back off — no cliff edges.
  - Trigger Python GC under memory pressure.
  - Provide a single source of truth for "how many workers can I use right now?"
  - Never let the system reach the point of black-screen / OOM crash.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------

# Fraction of system resources to *always* keep free for other programs.
# 0.25 means "always leave at least 25 % CPU and 25 % memory free."
_RESERVE_CPU = float(os.getenv("GOVERNOR_RESERVE_CPU", "0.25"))
_RESERVE_MEM = float(os.getenv("GOVERNOR_RESERVE_MEM", "0.30"))

# Hard ceiling: never exceed this fraction no matter what.
_HARD_CPU = float(os.getenv("GOVERNOR_HARD_CPU", "0.80"))
_HARD_MEM = float(os.getenv("GOVERNOR_HARD_MEM", "0.82"))

# Critical threshold — trigger emergency GC and pause new work.
_CRITICAL_MEM = float(os.getenv("GOVERNOR_CRITICAL_MEM", "0.88"))

# Absolute emergency — start killing our own child processes.
_EMERGENCY_MEM = float(os.getenv("GOVERNOR_EMERGENCY_MEM", "0.93"))

# How often the background sampler runs (seconds).
_SAMPLE_INTERVAL = float(os.getenv("GOVERNOR_SAMPLE_INTERVAL", "5"))

# Absolute bounds on the dynamic worker count.
_MIN_WORKERS = int(os.getenv("GOVERNOR_MIN_WORKERS", "1"))
_MAX_WORKERS = int(os.getenv("GOVERNOR_MAX_WORKERS", "0"))  # 0 = auto


class ResourceGovernor:
    """
    Periodically samples system CPU and memory, then exposes dynamic limits
    that any component can query to decide how many threads/workers to use.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cpu: float = 0.0          # 0-100
        self._mem: float = 0.0          # 0-1
        self._mem_available_mb: float = 0.0
        self._total_mem_mb: float = 0.0
        self._cpu_count: int = max(1, os.cpu_count() or 1)
        self._max_workers_ceil = _MAX_WORKERS if _MAX_WORKERS > 0 else max(4, self._cpu_count)
        self._gc_triggered_at: float = 0.0
        self._emergency_triggered_at: float = 0.0
        self._started = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Smoothed values (exponential moving average) to avoid jitter.
        self._ema_cpu: float = 0.0
        self._ema_mem: float = 0.0
        self._alpha = 0.3  # EMA smoothing factor (higher = more responsive)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sampling thread (idempotent)."""
        if self._started:
            return
        self._stop.clear()
        self._sample()  # immediate first sample
        self._thread = threading.Thread(target=self._run, daemon=True, name="resource-governor")
        self._thread.start()
        self._started = True
        logger.info(
            "resource governor started: cpus=%d reserve_cpu=%.0f%% reserve_mem=%.0f%% "
            "hard_mem=%.0f%% critical=%.0f%% emergency=%.0f%% max_workers=%d",
            self._cpu_count, _RESERVE_CPU * 100, _RESERVE_MEM * 100,
            _HARD_MEM * 100, _CRITICAL_MEM * 100, _EMERGENCY_MEM * 100,
            self._max_workers_ceil,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._started = False

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(_SAMPLE_INTERVAL)
            if self._stop.is_set():
                break
            self._sample()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(self) -> None:
        cpu = 0.0
        mem = 0.0
        avail_mb = 0.0
        total_mb = 0.0
        if psutil:
            try:
                cpu = float(psutil.cpu_percent(interval=0.1))
                vm = psutil.virtual_memory()
                mem = vm.percent / 100.0
                avail_mb = vm.available / (1024 * 1024)
                total_mb = vm.total / (1024 * 1024)
            except Exception:
                pass
        with self._lock:
            self._cpu = cpu
            self._mem = mem
            self._mem_available_mb = avail_mb
            self._total_mem_mb = total_mb
            # Update EMA
            self._ema_cpu = self._alpha * cpu + (1 - self._alpha) * self._ema_cpu
            self._ema_mem = self._alpha * mem + (1 - self._alpha) * self._ema_mem

        # Emergency GC under critical memory pressure.
        if mem >= _CRITICAL_MEM and (time.time() - self._gc_triggered_at) > 15:
            self._gc_triggered_at = time.time()
            collected = gc.collect()
            logger.warning(
                "CRITICAL memory pressure (%.1f%%) — forced GC collected %d objects",
                mem * 100, collected,
            )

        # Absolute emergency — try to free memory by clearing Python caches.
        if mem >= _EMERGENCY_MEM and (time.time() - self._emergency_triggered_at) > 30:
            self._emergency_triggered_at = time.time()
            self._emergency_shed_load()

    def _emergency_shed_load(self) -> None:
        """Last resort: aggressively free memory to prevent OS-level OOM."""
        logger.critical(
            "EMERGENCY memory shed (%.1f%%) — clearing caches and killing workers",
            self._mem * 100,
        )
        # Force all generations of GC
        for gen in range(3):
            gc.collect(gen)

        # Try to trim linecache, functools caches, etc.
        try:
            import linecache
            linecache.clearcache()
        except Exception:
            pass
        try:
            import importlib
            importlib.invalidate_caches()
        except Exception:
            pass

        # Kill our own child processes if memory is still critical
        if psutil and self._mem >= _EMERGENCY_MEM:
            try:
                current = psutil.Process()
                children = current.children(recursive=True)
                # Kill the heaviest child first
                children.sort(
                    key=lambda p: p.memory_info().rss if p.is_running() else 0,
                    reverse=True,
                )
                for child in children[:2]:  # at most 2
                    try:
                        mem_mb = child.memory_info().rss / (1024 * 1024)
                        if mem_mb > 200:  # only kill if using > 200MB
                            logger.critical(
                                "EMERGENCY: killing child pid=%d (%s) using %.0fMB",
                                child.pid, child.name(), mem_mb,
                            )
                            child.kill()
                    except Exception:
                        pass
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers (no lock — caller reads snapshot values)
    # ------------------------------------------------------------------

    def _read_ema(self):
        """Read smoothed values under lock, return (cpu_frac, mem_frac)."""
        with self._lock:
            return self._ema_cpu / 100.0, self._ema_mem

    # ------------------------------------------------------------------
    # Public queries — these are the API that components use
    # ------------------------------------------------------------------

    @property
    def cpu_percent(self) -> float:
        """Smoothed CPU usage (0-100)."""
        with self._lock:
            return self._ema_cpu

    @property
    def memory_percent(self) -> float:
        """Smoothed memory usage (0-1)."""
        with self._lock:
            return self._ema_mem

    @property
    def memory_available_mb(self) -> float:
        with self._lock:
            return self._mem_available_mb

    def max_workers(self, *, base: int = 8, floor: int = 0) -> int:
        """
        Dynamic worker count scaled to current system availability.

        *base* is the ideal number of workers when the system is idle.
        *floor* overrides the minimum (defaults to _MIN_WORKERS).
        Returns a value between floor and min(base, _max_workers_ceil).
        """
        actual_floor = max(floor, _MIN_WORKERS) if floor else _MIN_WORKERS
        ceiling = min(base, self._max_workers_ceil)

        cpu, mem = self._read_ema()

        # How much headroom do we have above our reserve?
        cpu_headroom = max(0.0, (1.0 - _RESERVE_CPU) - cpu)
        mem_headroom = max(0.0, (1.0 - _RESERVE_MEM) - mem)

        # Scale factor: 1.0 when plenty of room, 0.0 when at the reserve boundary.
        cpu_scale = min(1.0, cpu_headroom / max(0.01, 1.0 - _RESERVE_CPU))
        mem_scale = min(1.0, mem_headroom / max(0.01, 1.0 - _RESERVE_MEM))

        # Use the tighter constraint.
        scale = min(cpu_scale, mem_scale)

        workers = max(actual_floor, int(ceiling * scale))

        # Hard override: if we're above the hard ceiling, clamp to floor.
        if cpu >= _HARD_CPU or mem >= _HARD_MEM:
            workers = actual_floor

        return workers

    def should_pause(self) -> bool:
        """True if the system is under critical pressure — new work should wait."""
        cpu, mem = self._read_ema()
        return mem >= _CRITICAL_MEM or cpu >= _HARD_CPU

    def should_throttle(self) -> bool:
        """True if moderate pressure — work should slow down."""
        cpu, mem = self._read_ema()
        return cpu >= (1.0 - _RESERVE_CPU) or mem >= (1.0 - _RESERVE_MEM)

    def wait_if_pressured(self, label: str = "", max_wait: float = 60.0) -> float:
        """
        Block until system pressure drops below the hard ceiling.
        Returns the total seconds waited (0 if no wait was needed).
        """
        waited = 0.0
        while self.should_pause() and waited < max_wait:
            if waited == 0.0:
                logger.info("governor: pausing %s — system under pressure (cpu=%.0f%% mem=%.0f%%)",
                            label, self.cpu_percent, self.memory_percent * 100)
            time.sleep(2.0)
            waited += 2.0
            # Re-sample if the background thread isn't running
            if not self._started:
                self._sample()
        return waited

    def snapshot(self) -> dict:
        """Return current state for diagnostics / API endpoints."""
        with self._lock:
            cpu = self._ema_cpu
            mem = self._ema_mem
            avail = self._mem_available_mb
            total = self._total_mem_mb
        # Call methods OUTSIDE the lock to avoid deadlock
        return {
            "cpu_percent": round(cpu, 1),
            "memory_percent": round(mem * 100, 1),
            "memory_available_mb": round(avail, 0),
            "total_memory_mb": round(total, 0),
            "max_workers": self.max_workers(),
            "should_throttle": self.should_throttle(),
            "should_pause": self.should_pause(),
            "reserve_cpu": _RESERVE_CPU,
            "reserve_mem": _RESERVE_MEM,
            "hard_mem": _HARD_MEM,
            "critical_mem": _CRITICAL_MEM,
            "emergency_mem": _EMERGENCY_MEM,
        }


# ---------------------------------------------------------------------------
# Module-level singleton — import and use directly.
# ---------------------------------------------------------------------------

governor = ResourceGovernor()
