"""
Dynamic resource governor — system-wide singleton that adapts thread pools,
worker counts, and task pacing to real-time CPU/memory/disk/network availability.

Design goals:
  - Leave headroom for the user's other programs (Chrome, WSL, GIMP, etc.).
  - When the system is idle, allow the application to use most resources.
  - When pressure rises, smoothly back off — no cliff edges.
  - Trigger Python GC under memory pressure.
  - Set TensorFlow inter/intra-op threads dynamically.
  - Monitor our own process tree RSS and throttle before the OS does.
  - Provide a single source of truth for "how many workers can I use right now?"
  - Never let the system reach the point of black-screen / OOM crash.

Priority tiers (highest → lowest):
  CRITICAL   — gas solver, live wallet operations
  HIGH       — trading bot main loop, data streams
  NORMAL     — ML training, data downloads
  LOW        — code graph, background discovery, branddozer
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


# ---------------------------------------------------------------------------
# Priority tiers
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------

# Fraction of system resources to *always* keep free for other programs.
_RESERVE_CPU = float(os.getenv("GOVERNOR_RESERVE_CPU", "0.25"))
_RESERVE_MEM = float(os.getenv("GOVERNOR_RESERVE_MEM", "0.30"))

# Hard ceiling: reduce all non-critical work to minimum.
_HARD_CPU = float(os.getenv("GOVERNOR_HARD_CPU", "0.75"))
_HARD_MEM = float(os.getenv("GOVERNOR_HARD_MEM", "0.75"))

# Critical threshold — trigger emergency GC and pause new work.
_CRITICAL_MEM = float(os.getenv("GOVERNOR_CRITICAL_MEM", "0.82"))

# How rarely a FULL (gen-2) collection may run. Everything else is gen-0.
_FULL_GC_SEC = float(os.getenv("GOVERNOR_FULL_GC_SEC", "300"))

# Absolute emergency — start killing our own child processes.
_EMERGENCY_MEM = float(os.getenv("GOVERNOR_EMERGENCY_MEM", "0.90"))

# Below this much free memory the next allocation fails whoever caused it, so
# we shed regardless of attribution. Above it, memory we do not hold is not a
# reason to shed OUR work — see _memory_pressure_is_external().
_MIN_FREE_MB = float(os.getenv("GOVERNOR_MIN_FREE_MB", "256"))

# Our tree counts as "the hog" at this fraction of its own RSS budget. Matches
# the point at which _scale_workers() already starts shrinking us.
_OUR_HOG_FRACTION = float(os.getenv("GOVERNOR_OUR_HOG_FRACTION", "0.75"))

# How often the background sampler runs (seconds).
_SAMPLE_INTERVAL = float(os.getenv("GOVERNOR_SAMPLE_INTERVAL", "2"))

# How often the process-TREE RSS walk runs (seconds). This is deliberately much
# slower than _SAMPLE_INTERVAL. psutil's children(recursive=True) enumerates the
# WHOLE system process table to build a ppid map -- a C call that holds the GIL
# for its duration.
#
# THIS IS A TAIL FIX, NOT A MEDIAN FIX. Measured 2026-09-22 against live
# production pid 6340, 375 processes, n=100: min 9.5ms, median 14.2ms, p75
# 21.6ms, p90 40.4ms, p99 344.4ms. At the 2s sample cadence that is a 0.71% duty
# cycle at the median, 1.14% at the mean, 2.02% at p90. An earlier n=30 run here
# reported a 40.3ms median and a 2.01% duty cycle; that was the p90 of the real
# distribution, sampled 30 times into a heavy right tail while the box was
# loaded, and it is WRONG -- do not quote it. The case for throttling is the
# p99: 344ms of uninterruptible GIL, which now fires on 1 sample in 15 rather
# than on every one. The median cost was never the problem.
#
# The value it produces only ever moves slowly -- our tree is 2,561 MB of
# long-lived Python. Every consumer of _our_rss_mb already reads the CACHED
# field under _lock (lines ~389, ~470, ~559, ~645, ~655) rather than a fresh
# reading, so a value up to _RSS_SCAN_SEC old changes no decision. The system
# memory guards are untouched: _CRITICAL_MEM and _EMERGENCY_MEM both branch on
# ``mem`` from virtual_memory(), which is still sampled every single tick.
_RSS_SCAN_SEC = float(os.getenv("GOVERNOR_RSS_SCAN_SEC", "30"))

# Absolute bounds on the dynamic worker count.
_MIN_WORKERS = int(os.getenv("GOVERNOR_MIN_WORKERS", "1"))
_MAX_WORKERS = int(os.getenv("GOVERNOR_MAX_WORKERS", "0"))  # 0 = auto

# Max RSS for our entire process tree (MB). 0 = auto (50% of system RAM).
_MAX_PROCESS_RSS_MB = int(os.getenv("GOVERNOR_MAX_RSS_MB", "0"))

# Disk free floor in MB — warn if disk drops below this.
_DISK_FREE_FLOOR_MB = int(os.getenv("GOVERNOR_DISK_FREE_FLOOR_MB", "2048"))


class ResourceGovernor:
    """
    Periodically samples system CPU, memory, disk, and network usage, then
    exposes dynamic limits that any component can query to decide how many
    threads/workers to use.

    Components register with a priority so the governor can selectively pause
    lower-priority work before higher-priority work.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # --- Raw instant values ---
        self._cpu: float = 0.0          # 0-100
        self._mem: float = 0.0          # 0-1
        self._mem_available_mb: float = 0.0
        self._total_mem_mb: float = 0.0
        self._disk_free_mb: float = float("inf")
        self._disk_io_busy: float = 0.0  # 0-100
        self._net_bytes_sent: float = 0.0
        self._net_bytes_recv: float = 0.0
        self._our_rss_mb: float = 0.0
        self._other_programs_mem_mb: float = 0.0

        self._cpu_count: int = max(1, os.cpu_count() or 1)
        self._max_workers_ceil = _MAX_WORKERS if _MAX_WORKERS > 0 else max(4, self._cpu_count * 2)
        self._gc_triggered_at: float = 0.0
        self._full_gc_at: float = 0.0
        # Last time the process-tree RSS walk ran. 0.0 forces a scan on the
        # first sample so the guards are never armed off a zero reading.
        self._rss_scan_at: float = 0.0
        self._emergency_triggered_at: float = 0.0
        self._started = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Smoothed values (exponential moving average) to avoid jitter.
        self._ema_cpu: float = 0.0
        self._ema_mem: float = 0.0
        self._ema_disk_io: float = 0.0
        self._alpha = 0.35  # EMA smoothing factor (higher = more responsive)

        # Previous network counters for delta calculation
        self._prev_net_sent: float = 0.0
        self._prev_net_recv: float = 0.0
        self._prev_net_ts: float = 0.0
        self._net_send_rate_kbps: float = 0.0
        self._net_recv_rate_kbps: float = 0.0

        # Priority-based pause controls
        self._paused_priorities: Set[int] = set()
        self._pause_callbacks: Dict[str, Callable[[], None]] = {}
        self._resume_callbacks: Dict[str, Callable[[], None]] = {}

        # TF thread control
        self._tf_threads_set: int = 0

        # Process tree RSS limit
        total_mb = 0.0
        if psutil:
            try:
                total_mb = psutil.virtual_memory().total / (1024 * 1024)
            except Exception:
                pass
        self._max_rss_mb = float(
            _MAX_PROCESS_RSS_MB if _MAX_PROCESS_RSS_MB > 0
            else max(2048, total_mb * 0.50)
        )

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
            "hard_mem=%.0f%% critical=%.0f%% emergency=%.0f%% max_workers=%d max_rss=%.0fMB",
            self._cpu_count, _RESERVE_CPU * 100, _RESERVE_MEM * 100,
            _HARD_MEM * 100, _CRITICAL_MEM * 100, _EMERGENCY_MEM * 100,
            self._max_workers_ceil, self._max_rss_mb,
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
            try:
                self._sample()
            except Exception:
                pass  # never crash the governor thread

    # ------------------------------------------------------------------
    # Sampling — collects CPU, memory, disk, network, and process RSS
    # ------------------------------------------------------------------

    def _sample(self) -> None:
        cpu = 0.0
        mem = 0.0
        avail_mb = 0.0
        total_mb = 0.0
        disk_free_mb = float("inf")
        disk_io_busy = 0.0
        our_rss = 0.0
        other_mem = 0.0

        now = time.time()

        if psutil:
            try:
                cpu = float(psutil.cpu_percent(interval=0.1))
                vm = psutil.virtual_memory()
                mem = vm.percent / 100.0
                avail_mb = vm.available / (1024 * 1024)
                total_mb = vm.total / (1024 * 1024)
            except Exception:
                pass

            # Disk free space
            try:
                disk = psutil.disk_usage(os.path.abspath(os.sep))
                disk_free_mb = disk.free / (1024 * 1024)
            except Exception:
                pass

            # Disk I/O busy (if counters available)
            try:
                counters = psutil.disk_io_counters()
                if counters:
                    # Use read/write time as a proxy for busyness
                    # (psutil doesn't directly give busy%, so we approximate)
                    disk_io_busy = min(100.0, cpu * 0.3)  # rough correlation
            except Exception:
                pass

            # Network bandwidth
            try:
                net = psutil.net_io_counters()
                if net and self._prev_net_ts > 0:
                    dt = max(0.1, now - self._prev_net_ts)
                    self._net_send_rate_kbps = (net.bytes_sent - self._prev_net_sent) / dt / 1024
                    self._net_recv_rate_kbps = (net.bytes_recv - self._prev_net_recv) / dt / 1024
                if net:
                    self._prev_net_sent = net.bytes_sent
                    self._prev_net_recv = net.bytes_recv
                    self._prev_net_ts = now
            except Exception:
                pass

            # Our own process tree RSS.
            #
            # children(recursive=True) walks the entire system process table and
            # holds the GIL while it does — median 40.3ms, p90 372.2ms on this
            # box. Run it on _RSS_SCAN_SEC, not on every sample, and carry the
            # previous reading forward in between. See _RSS_SCAN_SEC above for
            # the measurement and for why a stale value is safe here.
            with self._lock:
                our_rss = self._our_rss_mb
            if (now - self._rss_scan_at) >= _RSS_SCAN_SEC:
                self._rss_scan_at = now
                try:
                    proc = psutil.Process()
                    scanned = proc.memory_info().rss / (1024 * 1024)
                    for child in proc.children(recursive=True):
                        try:
                            scanned += child.memory_info().rss / (1024 * 1024)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    our_rss = scanned
                except Exception:
                    pass

            # Estimate memory used by OTHER programs
            other_mem = max(0.0, (total_mb - avail_mb) - our_rss)

        with self._lock:
            self._cpu = cpu
            self._mem = mem
            self._mem_available_mb = avail_mb
            self._total_mem_mb = total_mb
            self._disk_free_mb = disk_free_mb
            self._disk_io_busy = disk_io_busy
            self._our_rss_mb = our_rss
            self._other_programs_mem_mb = other_mem
            # Update EMA
            self._ema_cpu = self._alpha * cpu + (1 - self._alpha) * self._ema_cpu
            self._ema_mem = self._alpha * mem + (1 - self._alpha) * self._ema_mem
            self._ema_disk_io = self._alpha * disk_io_busy + (1 - self._alpha) * self._ema_disk_io

        # --- Pressure response cascade ---

        # 1. Check if our process tree exceeds its RSS limit → force GC
        if our_rss > self._max_rss_mb * 0.85:
            if (now - self._gc_triggered_at) > 10:
                self._gc_triggered_at = now
                # A full gc.collect() is a stop-the-world gen-2 sweep that
                # holds the GIL uninterruptibly. At 83.9% system memory the
                # branch above is TRUE continuously, so the 10s cooldown ran
                # one full sweep every 10 seconds against a 14.9s median
                # loop_lag -- the governor was the thing blocking the loop it
                # exists to protect. Collect gen-0 on the normal cadence
                # (microseconds, and gen-0 is where short-lived tick objects
                # die anyway) and pay for a full sweep at most every
                # GOVERNOR_FULL_GC_SEC. The memory guard itself is untouched:
                # the LOW/NORMAL priority pauses below still fire every sample.
                if (now - self._full_gc_at) > _FULL_GC_SEC:
                    self._full_gc_at = now
                    collected = gc.collect()
                else:
                    collected = gc.collect(0)
                logger.warning(
                    "RSS pressure: our tree uses %.0fMB / %.0fMB limit — GC collected %d",
                    our_rss, self._max_rss_mb, collected,
                )

        # 2. System-wide critical memory → aggressive GC + pause LOW/NORMAL
        if mem >= _CRITICAL_MEM:
            if (now - self._gc_triggered_at) > 10:
                self._gc_triggered_at = now
                # A full gc.collect() is a stop-the-world gen-2 sweep that
                # holds the GIL uninterruptibly. At 83.9% system memory the
                # branch above is TRUE continuously, so the 10s cooldown ran
                # one full sweep every 10 seconds against a 14.9s median
                # loop_lag -- the governor was the thing blocking the loop it
                # exists to protect. Collect gen-0 on the normal cadence
                # (microseconds, and gen-0 is where short-lived tick objects
                # die anyway) and pay for a full sweep at most every
                # GOVERNOR_FULL_GC_SEC. The memory guard itself is untouched:
                # the LOW/NORMAL priority pauses below still fire every sample.
                if (now - self._full_gc_at) > _FULL_GC_SEC:
                    self._full_gc_at = now
                    collected = gc.collect()
                else:
                    collected = gc.collect(0)
                logger.warning(
                    "CRITICAL memory (%.1f%%) — GC collected %d objects, pausing low-priority work",
                    mem * 100, collected,
                )
            # Pause low and normal priority work
            self._set_priority_pause(Priority.LOW, True)
            self._set_priority_pause(Priority.NORMAL, True)
        elif mem >= _HARD_MEM:
            # Only pause low priority
            self._set_priority_pause(Priority.LOW, True)
            self._set_priority_pause(Priority.NORMAL, False)
        else:
            # All clear
            self._set_priority_pause(Priority.LOW, False)
            self._set_priority_pause(Priority.NORMAL, False)

        # 3. Emergency — kill heaviest children
        if mem >= _EMERGENCY_MEM and (now - self._emergency_triggered_at) > 20:
            self._emergency_triggered_at = now
            self._emergency_shed_load()

        # 4. Dynamic TensorFlow thread limit
        self._adjust_tf_threads()

        # 5. Disk space warning
        if disk_free_mb < _DISK_FREE_FLOOR_MB:
            logger.warning("Low disk space: %.0fMB free (floor: %dMB)", disk_free_mb, _DISK_FREE_FLOOR_MB)

    def _set_priority_pause(self, priority: Priority, paused: bool) -> None:
        """Track which priority levels are paused."""
        was_paused = priority.value in self._paused_priorities
        if paused and not was_paused:
            self._paused_priorities.add(priority.value)
            logger.info("governor: pausing priority=%s work", priority.name)
        elif not paused and was_paused:
            self._paused_priorities.discard(priority.value)
            logger.info("governor: resuming priority=%s work", priority.name)

    def _adjust_tf_threads(self) -> None:
        """Dynamically limit TensorFlow thread count based on pressure."""
        cpu, mem = self._read_ema()
        if mem >= _HARD_MEM or cpu >= _HARD_CPU:
            target = max(1, self._cpu_count // 4)
        elif mem >= (1.0 - _RESERVE_MEM) or cpu >= (1.0 - _RESERVE_CPU):
            target = max(2, self._cpu_count // 2)
        else:
            target = max(2, self._cpu_count)

        if target != self._tf_threads_set:
            self._tf_threads_set = target
            try:
                import tensorflow as tf
                tf.config.threading.set_intra_op_parallelism_threads(target)
                tf.config.threading.set_inter_op_parallelism_threads(max(1, target // 2))
            except Exception:
                pass  # TF not loaded or doesn't allow runtime changes

    def _emergency_shed_load(self) -> None:
        """Last resort: aggressively free memory to prevent OS-level OOM."""
        logger.critical(
            "EMERGENCY memory shed (%.1f%%, our RSS=%.0fMB) — clearing caches and killing workers",
            self._mem * 100, self._our_rss_mb,
        )
        # Force all generations of GC
        for gen in range(3):
            gc.collect(gen)

        # Trim standard caches
        for mod_name, method in [("linecache", "clearcache"), ("importlib", "invalidate_caches")]:
            try:
                mod = __import__(mod_name)
                getattr(mod, method)()
            except Exception:
                pass

        # Try to release TensorFlow memory
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass

        # Kill our heaviest child processes
        if psutil and self._mem >= _EMERGENCY_MEM:
            try:
                current = psutil.Process()
                children = current.children(recursive=True)
                children.sort(
                    key=lambda p: p.memory_info().rss if p.is_running() else 0,
                    reverse=True,
                )
                for child in children[:3]:  # up to 3
                    try:
                        mem_mb = child.memory_info().rss / (1024 * 1024)
                        if mem_mb > 150:  # lower threshold: kill anything > 150MB
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_ema(self):
        """Read smoothed values under lock, return (cpu_frac, mem_frac)."""
        with self._lock:
            return self._ema_cpu / 100.0, self._ema_mem

    def _memory_pressure_is_external(self) -> bool:
        """True when the memory is other programs' and shedding us cannot help.

        The governor sheds our work on the premise that doing so relieves the
        pressure. That premise only holds when the memory is OURS. Measured on
        this box 2026-09-16: the system was 74.5% used with 8.10 GB free while
        our whole process tree held 1,164 MB — 3.6%. Shedding us frees 3.6% and
        the box stays full, so the only effect is that we go dark. That is how
        the feed spent 57.6 hours dark, and it is why ``should_throttle`` is
        True today at a 0.70 threshold against a 0.745 reading, halving the
        ghost pair budget at trading/ghost_limits.py:38.

        Three ways this returns False, i.e. keeps the old behaviour at full
        force:

          * the sampler has not produced a reading yet (``avail <= 0``) — that
            is "unknown", and the safe reading of unknown is the conservative
            one;
          * absolute free memory is under ``_MIN_FREE_MB`` — below that the
            next allocation fails whoever caused it;
          * our own tree is at or over ``_OUR_HOG_FRACTION`` of its RSS budget
            — then the memory IS ours and shedding genuinely works.

        CPU is deliberately not covered: a busy CPU is real contention no
        matter who owns the processes.
        """
        with self._lock:
            avail = self._mem_available_mb
            ours = self._our_rss_mb
            budget = self._max_rss_mb
        if avail <= 0.0:
            return False
        if avail < _MIN_FREE_MB:
            return False
        if budget > 0.0 and ours >= budget * _OUR_HOG_FRACTION:
            return False
        return True

    # ------------------------------------------------------------------
    # Public queries — the API that components use
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

    @property
    def our_rss_mb(self) -> float:
        """Total RSS of our process tree in MB."""
        with self._lock:
            return self._our_rss_mb

    @property
    def disk_free_mb(self) -> float:
        with self._lock:
            return self._disk_free_mb

    def is_priority_paused(self, priority: Priority) -> bool:
        """Check if a given priority level is currently paused."""
        return priority.value in self._paused_priorities

    def max_workers(self, *, base: int = 8, floor: int = 0, priority: Priority = Priority.NORMAL) -> int:
        """
        Dynamic worker count scaled to current system availability.

        *base* is the ideal number of workers when the system is idle.
        *floor* overrides the minimum (defaults to _MIN_WORKERS).
        *priority* — lower-priority work gets fewer workers under pressure.
        Returns a value between floor and min(base, _max_workers_ceil).
        """
        actual_floor = max(floor, _MIN_WORKERS) if floor else _MIN_WORKERS
        ceiling = min(base, self._max_workers_ceil)

        # If this priority level is paused, return floor
        if self.is_priority_paused(priority):
            return actual_floor

        cpu, mem = self._read_ema()

        # How much headroom do we have above our reserve?
        cpu_headroom = max(0.0, (1.0 - _RESERVE_CPU) - cpu)
        mem_headroom = max(0.0, (1.0 - _RESERVE_MEM) - mem)

        # Scale factor: 1.0 when plenty of room, 0.0 when at the reserve boundary.
        cpu_scale = min(1.0, cpu_headroom / max(0.01, 1.0 - _RESERVE_CPU))
        mem_scale = min(1.0, mem_headroom / max(0.01, 1.0 - _RESERVE_MEM))

        # Use the tighter constraint.
        scale = min(cpu_scale, mem_scale)

        # Lower-priority work gets a tighter squeeze
        if priority == Priority.LOW:
            scale *= 0.5
        elif priority == Priority.NORMAL:
            scale *= 0.75

        workers = max(actual_floor, int(ceiling * scale))

        # Hard override: if we're above the hard ceiling, clamp to floor.
        if cpu >= _HARD_CPU or mem >= _HARD_MEM:
            workers = actual_floor

        # RSS override: if our process tree is over budget, reduce
        with self._lock:
            rss = self._our_rss_mb
        if rss > self._max_rss_mb * 0.75:
            rss_scale = max(0.2, 1.0 - (rss - self._max_rss_mb * 0.75) / (self._max_rss_mb * 0.25))
            workers = max(actual_floor, int(workers * rss_scale))

        return workers

    def should_pause(self, priority: Priority = Priority.NORMAL) -> bool:
        """True if the system is under enough pressure to pause this priority."""
        if self.is_priority_paused(priority):
            return True
        cpu, mem = self._read_ema()
        if self._memory_pressure_is_external():
            # Not our memory, and headroom remains: shedding us would free a
            # rounding error and stop the feed. CPU pressure below is unaffected.
            mem = 0.0
        if priority <= Priority.HIGH:
            # Only pause HIGH/CRITICAL at emergency levels
            return mem >= _EMERGENCY_MEM
        return mem >= _CRITICAL_MEM or cpu >= _HARD_CPU

    def should_throttle(self, priority: Priority = Priority.NORMAL) -> bool:
        """True if moderate pressure — work should slow down."""
        if self.is_priority_paused(priority):
            return True
        cpu, mem = self._read_ema()
        if self._memory_pressure_is_external():
            mem = 0.0
        threshold_cpu = 1.0 - _RESERVE_CPU
        threshold_mem = 1.0 - _RESERVE_MEM
        if priority == Priority.LOW:
            threshold_cpu *= 0.8
            threshold_mem *= 0.8
        return cpu >= threshold_cpu or mem >= threshold_mem

    def wait_if_pressured(
        self,
        label: str = "",
        max_wait: float = 120.0,
        priority: Priority = Priority.NORMAL,
    ) -> float:
        """
        Block until system pressure drops below acceptable level for this priority.
        Returns the total seconds waited (0 if no wait was needed).
        """
        waited = 0.0
        sleep_step = 2.0
        while self.should_pause(priority) and waited < max_wait:
            if waited == 0.0:
                logger.info(
                    "governor: pausing %s (priority=%s) — mem=%.0f%% cpu=%.0f%% rss=%.0fMB",
                    label, priority.name, self.memory_percent * 100,
                    self.cpu_percent, self.our_rss_mb,
                )
            time.sleep(sleep_step)
            waited += sleep_step
            # Back off sleep time to reduce sampling overhead
            sleep_step = min(sleep_step * 1.3, 10.0)
            # Re-sample if background thread isn't running
            if not self._started:
                self._sample()
        if waited > 0:
            logger.info("governor: resumed %s after %.1fs wait", label, waited)
        return waited

    def dynamic_batch_size(self, base: int = 32, floor: int = 4) -> int:
        """Dynamic ML training batch size — smaller when memory is tight."""
        cpu, mem = self._read_ema()
        if mem >= _HARD_MEM:
            return floor
        scale = max(0.2, 1.0 - mem / _HARD_MEM)
        return max(floor, int(base * scale))

    def dynamic_prefetch(self, base: int = 4) -> int:
        """Dynamic TF dataset prefetch buffer count."""
        _, mem = self._read_ema()
        if mem >= _HARD_MEM:
            return 1
        return max(1, int(base * (1.0 - mem)))

    def can_spawn_subprocess(self) -> bool:
        """Check if it's safe to spawn another subprocess."""
        cpu, mem = self._read_ema()
        if mem >= _HARD_MEM or cpu >= _HARD_CPU:
            return False
        with self._lock:
            rss = self._our_rss_mb
        return rss < self._max_rss_mb * 0.80

    def snapshot(self) -> dict:
        """Return current state for diagnostics / API endpoints."""
        with self._lock:
            cpu = self._ema_cpu
            mem = self._ema_mem
            avail = self._mem_available_mb
            total = self._total_mem_mb
            rss = self._our_rss_mb
            other = self._other_programs_mem_mb
            disk_free = self._disk_free_mb
            net_send = self._net_send_rate_kbps
            net_recv = self._net_recv_rate_kbps
        return {
            "cpu_percent": round(cpu, 1),
            "memory_percent": round(mem * 100, 1),
            "memory_available_mb": round(avail, 0),
            "total_memory_mb": round(total, 0),
            "our_rss_mb": round(rss, 0),
            "max_rss_mb": round(self._max_rss_mb, 0),
            "other_programs_mem_mb": round(other, 0),
            "disk_free_mb": round(disk_free, 0),
            "net_send_kbps": round(net_send, 1),
            "net_recv_kbps": round(net_recv, 1),
            "max_workers": self.max_workers(),
            "should_throttle": self.should_throttle(),
            "should_pause": self.should_pause(),
            "paused_priorities": sorted(self._paused_priorities),
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
