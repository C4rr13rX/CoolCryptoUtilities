"""tools/run_programming_brain.py

Manages the local Senior Software Engineer Brain (72.2M terminals) daemon on port 18096.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIZARD_ROOT = Path("D:/Projects/W1z4rDV1510n")
DEFAULT_PORT = 18096
DEFAULT_BRAIN_DIR = WIZARD_ROOT / "runtime" / "brains" / "programming-integrated-20260713" / "brain"
DEFAULT_IDENTITY = WIZARD_ROOT / "brains" / "coding_debug.identity.toml"
RUN_DIR = WIZARD_ROOT / "runtime" / "programming-brain-local"
PID_FILE = RUN_DIR / "brain_server.pid"


def find_binary() -> Path:
    candidates = [
        WIZARD_ROOT / "target" / "release" / "w1z4rd_brain_server.exe",
        WIZARD_ROOT / "bin" / "w1z4rd_brain_server.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("w1z4rd_brain_server.exe not found in target/release or bin")


def is_running(port: int = DEFAULT_PORT) -> bool:
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1.5)
        return r.status_code == 200 and r.text.strip().lower() in {"ok", '{"status":"ok"}'}
    except Exception:
        return False


def start_brain(port: int = DEFAULT_PORT) -> int:
    if is_running(port):
        print(f"Senior software engineer brain already healthy on port {port}.")
        return 0

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    binary = find_binary()

    env = os.environ.copy()
    env["W1Z4RD_BRAIN_PORT"] = str(port)
    env["W1Z4RD_NODE_BRAIN_DIR"] = str(DEFAULT_BRAIN_DIR)
    env["W1Z4RD_BRAIN_IDENTITY"] = str(DEFAULT_IDENTITY)
    env["W1Z4RD_BRAIN_AUTOCHECKPOINT_SECS"] = "0"

    stdout_path = RUN_DIR / "brain_server.stdout.log"
    stderr_path = RUN_DIR / "brain_server.stderr.log"

    stdout_file = open(stdout_path, "a", encoding="utf-8")
    stderr_file = open(stderr_path, "a", encoding="utf-8")

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    proc = subprocess.Popen(
        [str(binary)],
        env=env,
        stdout=stdout_file,
        stderr=stderr_file,
        creationflags=creationflags,
        close_fds=True,
    )

    PID_FILE.write_text(str(proc.pid), encoding="ascii")
    print(f"Started w1z4rd_brain_server PID={proc.pid} on port {port}, verifying health...")

    healthy = False
    for attempt in range(25):
        time.sleep(0.4)
        if proc.poll() is not None:
            print(f"Process died with code {proc.returncode}. See {stderr_path}")
            return 1
        if is_running(port):
            healthy = True
            break

    if healthy:
        print(f"Senior Software Engineer brain healthy on http://127.0.0.1:{port} (PID={proc.pid})")
        return 0
    else:
        print(f"Timed out waiting for senior brain health check on port {port}.")
        return 1


def stop_brain(port: int = DEFAULT_PORT) -> int:
    stopped = False
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text(encoding="ascii").strip())
            os.kill(pid, signal.SIGTERM)
            stopped = True
            print(f"Sent SIGTERM to senior brain PID={pid}")
        except Exception:
            pass
        PID_FILE.unlink(missing_ok=True)

    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "w1z4rd_brain_server.exe"], capture_output=True)
            stopped = True
        except Exception:
            pass

    print(f"Senior software engineer brain stopped on port {port}.")
    return 0


def status_brain(port: int = DEFAULT_PORT) -> int:
    if is_running(port):
        try:
            s = requests.get(f"http://127.0.0.1:{port}/stats", timeout=3).json()
            print(f"ONLINE on port {port}:")
            print(f"  Neurons:   {s.get('total_neurons'):,}")
            print(f"  Terminals: {s.get('total_terminals'):,}")
            print(f"  Pools:     {s.get('pool_count')}")
            print(f"  Tick:      {s.get('tick'):,}")
            return 0
        except Exception as exc:
            print(f"ONLINE on port {port} (stats error: {exc})")
            return 0
    else:
        print(f"OFFLINE on port {port}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status", "restart"], default="start", nargs="?")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    if args.action == "start":
        sys.exit(start_brain(args.port))
    elif args.action == "stop":
        sys.exit(stop_brain(args.port))
    elif args.action == "status":
        sys.exit(status_brain(args.port))
    elif args.action == "restart":
        stop_brain(args.port)
        time.sleep(1)
        sys.exit(start_brain(args.port))
