"""Lightweight resource monitor for delegation hosts — works even without psutil."""
from __future__ import annotations

import os
import platform
import threading
import time
from typing import Dict, Optional


class ResourceMonitor:
    """Periodically samples CPU/memory and exposes a snapshot for the API."""

    def __init__(self, sample_interval: float = 3.0) -> None:
        self._interval = sample_interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu_percent: float = 0.0
        self._mem_percent: float = 0.0
        self._mem_available_mb: int = 0
        self._mem_total_mb: int = 0
        self._disk_free_mb: int = 0
        self._our_rss_mb: float = 0.0
        self._has_psutil = False
        try:
            import psutil  # noqa: F401
            self._has_psutil = True
        except ImportError:
            pass

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True, name="res-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._sample()
            except Exception:
                pass

    def _sample(self) -> None:
        if self._has_psutil:
            self._sample_psutil()
        else:
            self._sample_fallback()

    def _sample_psutil(self) -> None:
        import psutil
        cpu = psutil.cpu_percent(interval=0)
        vm = psutil.virtual_memory()
        try:
            disk = psutil.disk_usage("/")
            disk_free = int(disk.free / (1024 * 1024))
        except Exception:
            disk_free = 0
        rss = 0.0
        try:
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            pass
        with self._lock:
            self._cpu_percent = cpu
            self._mem_percent = vm.percent
            self._mem_available_mb = int(vm.available / (1024 * 1024))
            self._mem_total_mb = int(vm.total / (1024 * 1024))
            self._disk_free_mb = disk_free
            self._our_rss_mb = rss

    def _sample_fallback(self) -> None:
        """Fallback for systems without psutil (Userland Ubuntu, etc.)."""
        cpu = 0.0
        mem_percent = 0.0
        mem_avail = 0
        mem_total = 0
        disk_free = 0

        # /proc/stat CPU (Linux)
        try:
            with open("/proc/stat") as f:
                line = f.readline()
            parts = line.split()
            idle = int(parts[4])
            total = sum(int(p) for p in parts[1:])
            # Rough estimate from current snapshot
            cpu = max(0.0, min(100.0, 100.0 * (1.0 - idle / max(total, 1))))
        except Exception:
            pass

        # /proc/meminfo
        try:
            meminfo: Dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(":")] = int(parts[1])
            mem_total = meminfo.get("MemTotal", 0) // 1024
            mem_avail = meminfo.get("MemAvailable", meminfo.get("MemFree", 0)) // 1024
            if mem_total > 0:
                mem_percent = 100.0 * (1.0 - mem_avail / mem_total)
        except Exception:
            pass

        # Disk
        try:
            st = os.statvfs("/")
            disk_free = int(st.f_bavail * st.f_frsize / (1024 * 1024))
        except Exception:
            pass

        # Windows fallback
        if platform.system() == "Windows":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                mem_total = int(stat.ullTotalPhys / (1024 * 1024))
                mem_avail = int(stat.ullAvailPhys / (1024 * 1024))
                mem_percent = float(stat.dwMemoryLoad)
            except Exception:
                pass
            try:
                import shutil
                usage = shutil.disk_usage("/")
                disk_free = int(usage.free / (1024 * 1024))
            except Exception:
                pass

        with self._lock:
            self._cpu_percent = cpu
            self._mem_percent = mem_percent
            self._mem_available_mb = mem_avail
            self._mem_total_mb = mem_total
            self._disk_free_mb = disk_free

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "cpu_percent": round(self._cpu_percent, 1),
                "memory_percent": round(self._mem_percent, 1),
                "memory_available_mb": self._mem_available_mb,
                "total_memory_mb": self._mem_total_mb,
                "disk_free_mb": self._disk_free_mb,
                "our_rss_mb": round(self._our_rss_mb, 1),
            }

    @property
    def should_throttle(self) -> bool:
        with self._lock:
            return self._cpu_percent > 85 or self._mem_percent > 80

    @property
    def should_pause(self) -> bool:
        with self._lock:
            return self._cpu_percent > 95 or self._mem_percent > 90
