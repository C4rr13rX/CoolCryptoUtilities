"""Detect device capabilities and platform quirks."""
from __future__ import annotations

import os
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def detect_device() -> Dict:
    """Return a profile of this device's capabilities."""
    info: Dict = {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "pointer_size": struct.calcsize("P") * 8,
        "hostname": platform.node(),
        "device_type": "desktop",  # default
        "cpu_count": os.cpu_count() or 1,
        "total_memory_mb": 0,
        "is_userland": False,
        "is_termux": False,
        "is_mobile": False,
        "capabilities": [],
    }

    # Detect Userland Ubuntu / Termux on Android
    if os.path.exists("/data/data/tech.ula") or os.getenv("USERLAND"):
        info["is_userland"] = True
        info["is_mobile"] = True
        info["device_type"] = "mobile_userland"
    elif os.path.exists("/data/data/com.termux") or os.getenv("TERMUX_VERSION"):
        info["is_termux"] = True
        info["is_mobile"] = True
        info["device_type"] = "mobile_termux"
    elif _is_android():
        info["is_mobile"] = True
        info["device_type"] = "mobile"
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        # Could be M-series Mac or iPad
        info["device_type"] = "desktop"  # treat as desktop either way

    # Memory detection
    info["total_memory_mb"] = _get_total_memory_mb()

    # Capabilities
    info["capabilities"] = _detect_capabilities(info)

    return info


def _is_android() -> bool:
    try:
        uname = platform.uname()
        if "android" in uname.release.lower() or "android" in uname.version.lower():
            return True
    except Exception:
        pass
    return os.path.exists("/system/build.prop")


def _get_total_memory_mb() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024 * 1024))
    except ImportError:
        pass
    # Fallback for Userland/Termux where psutil may not install
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    # Windows fallback
    if platform.system() == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulong),
                    ("ullAvailPhys", c_ulong),
                    ("ullTotalPageFile", c_ulong),
                    ("ullAvailPageFile", c_ulong),
                    ("ullTotalVirtual", c_ulong),
                    ("ullAvailVirtual", c_ulong),
                    ("ullAvailExtendedVirtual", c_ulong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys / (1024 * 1024))
        except Exception:
            pass
    return 2048  # safe default


def _detect_capabilities(info: Dict) -> List[str]:
    """Determine which task types this device can handle."""
    caps: List[str] = []
    mem_mb = info["total_memory_mb"]
    cpus = info["cpu_count"]
    is_mobile = info["is_mobile"]

    # All devices can download data
    caps.append("data_ingest")
    caps.append("news_enrichment")
    caps.append("background_refresh")

    # Dataset warmup needs ~500MB
    if mem_mb >= 512:
        caps.append("dataset_warmup")

    # Training needs real CPU + memory
    if mem_mb >= 2048 and cpus >= 2 and not is_mobile:
        caps.append("candidate_training")

    # Ghost trading/metrics need moderate resources
    if mem_mb >= 1024:
        caps.append("ghost_metrics")
        caps.append("ghost_trading")

    # Live monitoring is lightweight
    caps.append("live_monitoring")

    # Check for specific libraries
    for mod, cap in [("numpy", None), ("pandas", None), ("sklearn", None)]:
        try:
            __import__(mod)
        except ImportError:
            # If numpy/pandas missing, remove training capability
            if "candidate_training" in caps:
                caps.remove("candidate_training")
            break

    return caps


def recommended_max_workers(info: Dict) -> int:
    """Suggest a max concurrent task count for this device."""
    mem_mb = info["total_memory_mb"]
    cpus = info["cpu_count"]
    is_mobile = info["is_mobile"]

    if is_mobile:
        # Mobile: conservative — 1-2 tasks
        if mem_mb < 2048:
            return 1
        return min(2, cpus)

    # Desktop/server: scale with resources
    base = min(cpus, max(1, mem_mb // 2048))
    return max(1, min(base, 8))
