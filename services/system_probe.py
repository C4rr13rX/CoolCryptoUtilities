from __future__ import annotations

import os
import platform
import socket
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SystemProbe:
    os_name: str
    os_release: str
    os_version: str
    machine: str
    processor: str
    python: str
    cwd: str
    is_admin: bool
    cpu_count: Optional[int]
    total_memory_gb: Optional[float]
    hostname: str
    network_available: bool

    def to_context_block(self) -> str:
        lines = ["System probe:"]
        for key, value in asdict(self).items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)


def _is_admin_windows() -> bool:
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _is_admin_posix() -> bool:
    try:
        return os.geteuid() == 0
    except Exception:
        return False


def _total_memory_gb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        return None


def _network_available() -> bool:
    try:
        socket.getaddrinfo("example.com", 80)
        return True
    except Exception:
        return False


def collect_system_probe(*, cwd: Path | str | None = None) -> SystemProbe:
    cwd_path = Path(cwd or os.getcwd()).resolve()
    is_admin = _is_admin_windows() if os.name == "nt" else _is_admin_posix()
    cpu_count = os.cpu_count()
    total_mem = _total_memory_gb()
    return SystemProbe(
        os_name=platform.system(),
        os_release=platform.release(),
        os_version=platform.version(),
        machine=platform.machine(),
        processor=platform.processor(),
        python=platform.python_version(),
        cwd=str(cwd_path),
        is_admin=is_admin,
        cpu_count=cpu_count,
        total_memory_gb=total_mem,
        hostname=platform.node(),
        network_available=_network_available(),
    )


def system_probe_context(cwd: Path | str | None = None) -> str:
    return collect_system_probe(cwd=cwd).to_context_block()


__all__ = ["SystemProbe", "collect_system_probe", "system_probe_context"]
