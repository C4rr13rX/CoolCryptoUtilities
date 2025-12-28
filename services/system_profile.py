from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SystemProfile:
    cpu_count: int
    total_memory_gb: float
    max_threads: int
    is_low_power: bool
    memory_pressure: bool


def _read_meminfo() -> Optional[float]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    kb = float(parts[1])
                    return kb / (1024 * 1024)
    except Exception:
        return None
    return None


def _sysconf_mem() -> Optional[float]:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")  # type: ignore[attr-defined]
        page_size = os.sysconf("SC_PAGE_SIZE")  # type: ignore[attr-defined]
        if pages and page_size:
            bytes_total = float(pages) * float(page_size)
            return bytes_total / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        return None
    return None


def detect_system_profile(
    *,
    target_threads: Optional[int] = None,
    min_threads: int = 2,
    max_threads: int = 8,
) -> SystemProfile:
    cpu_count = max(1, os.cpu_count() or 1)
    mem_gb: Optional[float] = None
    mem_override = os.getenv("SYSTEM_MEMORY_GB")
    if mem_override:
        try:
            mem_gb = float(mem_override)
        except ValueError:
            mem_gb = None
    if mem_gb is None:
        mem_gb = _sysconf_mem() or _read_meminfo()
    if mem_gb is None or mem_gb <= 0:
        mem_gb = 8.0
    threads = target_threads or max(min_threads, min(cpu_count, max_threads))
    low_memory_floor = 16.0
    high_memory_floor = 32.0
    is_low_power = cpu_count <= 2 or mem_gb < low_memory_floor or (cpu_count <= 4 and mem_gb < high_memory_floor)
    memory_pressure = mem_gb < 24
    return SystemProfile(
        cpu_count=cpu_count,
        total_memory_gb=float(mem_gb),
        max_threads=int(max(min_threads, min(max_threads, threads))),
        is_low_power=is_low_power,
        memory_pressure=memory_pressure,
    )
