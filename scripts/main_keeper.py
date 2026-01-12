from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import psutil

try:
    from services.guardian_lock import GuardianLease
except Exception:  # pragma: no cover - fallback if services import fails
    GuardianLease = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = REPO_ROOT / "main.py"
DEFAULT_INTERVAL = 60.0
LOG_PATH = REPO_ROOT / "runtime" / "main_keeper.log"
MAIN_LOG_PATH = REPO_ROOT / "runtime" / "main_autostart.log"


def _log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}] {message}\n"
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        pass
    print(line, end="")


def _cmdline_matches(cmdline: Iterable[str]) -> bool:
    for part in cmdline:
        if "main.py" in str(part):
            return True
    return False


def main_process_running() -> bool:
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
            if not cmd:
                continue
            if _cmdline_matches(cmd):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def launch_main() -> Optional[subprocess.Popen[str]]:
    if not MAIN_PATH.exists():
        _log("main.py not found; cannot start main process.")
        return None
    cmd = [sys.executable, str(MAIN_PATH), "--action", "start_production", "--stay-alive"]
    env = os.environ.copy()
    env.setdefault("ALLOW_SQLITE_FALLBACK", "1")
    MAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with MAIN_LOG_PATH.open("a", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
                env=env,
            )
        _log(f"launched main process pid={proc.pid} (logs -> {MAIN_LOG_PATH})")
        return proc
    except Exception as exc:
        _log(f"failed to launch main process: {exc}")
        return None


def run_loop(interval: float) -> None:
    lease: Optional[GuardianLease] = None
    if GuardianLease is not None:
        lease = GuardianLease("main-keeper", poll_interval=1.0)
        if not lease.acquire():
            _log("another main_keeper instance holds the lease; exiting.")
            return
    try:
        while True:
            if not main_process_running():
                launch_main()
            time.sleep(interval)
    finally:
        if lease:
            try:
                lease.release()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure main.py stays alive (option 7 equivalent).")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Seconds between checks (default 60).")
    parser.add_argument("--once", action="store_true", help="Run a single check/start and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.once:
        if main_process_running():
            _log("main process already running; nothing to do.")
        else:
            launch_main()
    else:
        run_loop(max(5.0, float(args.interval or DEFAULT_INTERVAL)))
