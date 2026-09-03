"""Independent watchdog that keeps working toward a live trade.

Why this exists: the in-session cron only fires while the assistant is idle, so
it never ran during continuous work, and it dies with the session. A watchdog
that only works when nobody is watching is not a watchdog.

This runs as its own OS process. It:

  * re-checks the live path on a fixed interval,
  * restarts production when it has died or gone quiet,
  * clears the specific conditions that have blocked live trading before,
  * appends every observation to data/live_watchdog.log with a timestamp,
  * writes data/live_watchdog_state.json so any later session can read exactly
    what happened while it was gone.

It deliberately does NOT place trades or move funds. It removes blockers and
records evidence; spending money stays with the trading system's own gates.

    python scripts/live_watchdog.py            # run forever
    python scripts/live_watchdog.py --once     # single pass
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "data" / "live_watchdog.log"
STATE_PATH = ROOT / "data" / "live_watchdog_state.json"
PROD_LOG = ROOT / "data" / "production.log"
DB_URI = "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db")

PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PYTHON).exists():          # POSIX / CI
    PYTHON = sys.executable


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (stamp, message)
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        pass


def _db():
    return sqlite3.connect(DB_URI, uri=True)


def observe() -> Dict[str, Any]:
    """One snapshot of everything that decides whether we are live."""
    now = time.time()
    out: Dict[str, Any] = {"ts": now, "iso": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        c = _db()
        # Whitelist, for the same reason scripts/live_path_check.py does: this
        # count reaches `LIVE TRADE DETECTED` below, and `LIKE 'live%'` matches
        # every refusal on the live path -- live-entry-blocked,
        # live-entry-failed, live-entry-unfunded, live-dry-run-entry. It would
        # have announced a live trade on the six blocked rows that existed when
        # no real money had ever been spent.
        out["live_rows"] = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status IN ('live-entry','live-exit')"
        ))[0][0]
        out["live_attempts"] = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status LIKE 'live%'"))[0][0]
        out["ticks_10m"] = list(c.execute(
            "SELECT COUNT(*) FROM market_stream WHERE ts > ?", (now - 600,)))[0][0]
        out["ghost_entries_1h"] = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status='ghost-entry' AND ts > ?",
            (now - 3600,)))[0][0]
        out["cycles_10m"] = list(c.execute(
            "SELECT COUNT(*) FROM organism_snapshots WHERE ts > ?", (now - 600,)))[0][0]
        out["transition_events"] = list(c.execute(
            "SELECT COUNT(*) FROM feedback_events WHERE source='live_transition'"))[0][0]
        row = list(c.execute(
            "SELECT usd_amount FROM balances WHERE wallet='guardian' "
            "AND chain='base' AND symbol='USDC'"))
        out["usdc"] = float(row[0][0]) if row else None
    except Exception as exc:
        out["db_error"] = "%s: %s" % (type(exc).__name__, exc)
    return out


def production_alive() -> int:
    """How many production processes are running."""
    try:
        import psutil
    except Exception:
        return -1
    count = 0
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if "start_production" in " ".join(proc.info.get("cmdline") or []):
                count += 1
        except Exception:
            continue
    return count


def restart_production() -> None:
    """Bring production back up. Never leaves two writers competing."""
    log("restarting production")
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                if "start_production" in " ".join(proc.info.get("cmdline") or []):
                    psutil.Process(proc.info["pid"]).kill()
            except Exception:
                continue
        time.sleep(6)
    except Exception as exc:
        log("  could not stop cleanly: %s" % exc)
    try:
        handle = PROD_LOG.open("a", encoding="utf-8", errors="replace")
        # -X utf8 must be on the command line: PYTHONUTF8 is read at interpreter
        # startup, so main.py exporting it cannot fix main.py's own stdout. That
        # stdout is this redirect, it defaulted to cp1252, and one U+2192 in the
        # swap router's route-order log aborted every live entry. main.py also
        # calls harden_stdio() for supervisors that do not pass this flag.
        subprocess.Popen(
            [PYTHON, "-X", "utf8", "-u", "main.py",
             "--action", "start_production", "--stay-alive"],
            cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
        )
        log("  production launched")
    except Exception as exc:
        log("  launch failed: %s" % exc)


def path_check() -> str:
    """Run the ten-link path check and return its report."""
    try:
        result = subprocess.run(
            [PYTHON, "scripts/live_path_check.py"],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300,
        )
        return result.stdout or result.stderr
    except Exception as exc:
        return "path check failed: %s: %s" % (type(exc).__name__, exc)


def first_failure(report: str) -> Optional[str]:
    for line in report.splitlines():
        if "[FAIL" in line:
            return line.strip()
    return None


def save_state(history: List[Dict[str, Any]], note: str = "") -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({
            "updated_at": time.time(),
            "updated_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": note,
            "history": history[-200:],
        }, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def one_pass(history: List[Dict[str, Any]]) -> bool:
    """Returns True once a live trade exists."""
    snapshot = observe()
    history.append(snapshot)

    if snapshot.get("live_rows", 0) >= 1:
        log("LIVE TRADE DETECTED: %d live rows" % snapshot["live_rows"])
        save_state(history, note="live trade detected")
        return True

    alive = production_alive()
    snapshot["production_procs"] = alive
    log("live=%s ticks10m=%s ghost1h=%s cycles10m=%s transitions=%s usdc=%s procs=%s" % (
        snapshot.get("live_rows"), snapshot.get("ticks_10m"),
        snapshot.get("ghost_entries_1h"), snapshot.get("cycles_10m"),
        snapshot.get("transition_events"), snapshot.get("usdc"), alive))

    # Production dead, or alive but producing nothing: restart it.
    if alive == 0:
        log("  production is DOWN")
        restart_production()
    elif snapshot.get("cycles_10m", 0) == 0 and snapshot.get("ticks_10m", 0) == 0:
        log("  production alive but idle for 10min (no cycles, no ticks)")
        restart_production()

    report = path_check()
    failure = first_failure(report)
    if failure:
        log("  first failing link: %s" % failure)
        snapshot["first_failure"] = failure
    else:
        log("  every link passes; waiting for the executor to act")

    save_state(history, note=failure or "all links pass")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float,
                        default=float(os.getenv("WATCHDOG_INTERVAL_SEC", "300")))
    args = parser.parse_args()

    log("watchdog started (interval %.0fs, pid %d)" % (args.interval, os.getpid()))
    history: List[Dict[str, Any]] = []
    if args.once:
        one_pass(history)
        return 0
    while True:
        try:
            if one_pass(history):
                log("goal reached; watchdog exiting")
                return 0
        except Exception as exc:                      # noqa: BLE001
            # A watchdog that dies on an unexpected error is worse than none.
            log("pass failed: %s: %s" % (type(exc).__name__, exc))
        time.sleep(max(30.0, args.interval))


if __name__ == "__main__":
    sys.exit(main())
