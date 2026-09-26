"""[e60e1a20] Does a pytest sweep starve the live price feed, or does the feed
stall on its own?

Jet filed the item at pass 107 off a single observation: feed age 526s and
rising, 3 ticks in 10 minutes, WHILE a full pytest sweep ran. That is one
sample of the treatment and zero samples of the control. Sage measured at pass
720 that the tape stops for 13.3% of wall time with NO sweep running, so the
base rate of a zero-tick window is not zero and an uncontrolled sweep
measurement cannot tell contention from the spontaneous stall.

This probe samples the SAME quantities through a quiet phase and a sweep phase
so the two are comparable:

  * rows written to market_stream in the trailing window
  * age of the newest market_stream row
  * per-process CPU and disk IO for the production processes and for the sweep
  * system-wide CPU

Read-only against the store. It never writes to trading_cache.db.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a hard requirement here
    print("psutil is required for the per-process half of this probe")
    raise

DB_PATH = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
PRODUCTION_MARKER = "--action start_production"
SWEEP_MARKER = "pytest"


def _production_pids() -> List[int]:
    pids = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(proc.info.get("cmdline") or [])
        if PRODUCTION_MARKER in cmd:
            pids.append(proc.info["pid"])
    return sorted(pids)


def _sweep_pids() -> List[int]:
    pids = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(proc.info.get("cmdline") or [])
        if SWEEP_MARKER in cmd and "feed_contention_probe" not in cmd:
            pids.append(proc.info["pid"])
    return sorted(pids)


def _feed_state(conn: sqlite3.Connection, window: float) -> Dict[str, float]:
    now = time.time()
    newest = conn.execute("SELECT MAX(ts) FROM market_stream").fetchone()[0]
    rows = conn.execute(
        "SELECT COUNT(*) FROM market_stream WHERE ts > ?", (now - window,)
    ).fetchone()[0]
    return {
        "wall": now,
        "rows_in_window": int(rows),
        "rows_per_min": rows / (window / 60.0),
        "newest_age_sec": (now - float(newest)) if newest else float("nan"),
    }


class _ProcTracker:
    """CPU percent and cumulative IO deltas for a set of pids."""

    def __init__(self) -> None:
        self._procs: Dict[int, psutil.Process] = {}
        self._io: Dict[int, tuple] = {}

    def sample(self, pids: List[int]) -> Dict[str, float]:
        cpu = 0.0
        read_b = 0.0
        write_b = 0.0
        alive = 0
        for pid in pids:
            proc = self._procs.get(pid)
            if proc is None:
                try:
                    proc = psutil.Process(pid)
                    proc.cpu_percent(None)  # prime the counter
                except psutil.Error:
                    continue
                self._procs[pid] = proc
            try:
                cpu += proc.cpu_percent(None)
                io = proc.io_counters()
                prev = self._io.get(pid)
                if prev is not None:
                    read_b += max(0, io.read_bytes - prev[0])
                    write_b += max(0, io.write_bytes - prev[1])
                self._io[pid] = (io.read_bytes, io.write_bytes)
                alive += 1
            except psutil.Error:
                self._procs.pop(pid, None)
                self._io.pop(pid, None)
        return {
            "procs": alive,
            "cpu_pct": round(cpu, 1),
            "read_mb": round(read_b / 1e6, 2),
            "write_mb": round(write_b / 1e6, 2),
        }


def run_phase(
    label: str,
    seconds: float,
    interval: float,
    window: float,
    out: List[dict],
) -> None:
    conn = sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True, timeout=20)
    prod = _ProcTracker()
    sweep = _ProcTracker()
    psutil.cpu_percent(None)
    deadline = time.time() + seconds
    while time.time() < deadline:
        started = time.time()
        try:
            feed = _feed_state(conn, window)
        except sqlite3.Error as exc:  # the store is shared; do not die on a lock
            feed = {"wall": started, "rows_in_window": -1, "rows_per_min": -1.0,
                    "newest_age_sec": float("nan"), "db_error": str(exc)}
        sample = {
            "phase": label,
            "t": round(started, 3),
            "system_cpu_pct": psutil.cpu_percent(None),
            "production": prod.sample(_production_pids()),
            "sweep": sweep.sample(_sweep_pids()),
        }
        sample.update(feed)
        out.append(sample)
        print(
            "[%s] age=%6.1fs  rows/%ds=%3d (%5.1f/min)  syscpu=%5.1f%%  "
            "prod cpu=%5.1f%% w=%5.2fMB  sweep n=%d cpu=%6.1f%% r=%5.2fMB"
            % (
                label,
                sample["newest_age_sec"],
                int(window),
                sample["rows_in_window"],
                sample["rows_per_min"],
                sample["system_cpu_pct"],
                sample["production"]["cpu_pct"],
                sample["production"]["write_mb"],
                sample["sweep"]["procs"],
                sample["sweep"]["cpu_pct"],
                sample["sweep"]["read_mb"],
            ),
            flush=True,
        )
        slack = interval - (time.time() - started)
        if slack > 0:
            time.sleep(slack)
    conn.close()


def summarise(samples: List[dict], label: str) -> Optional[dict]:
    rows = [s for s in samples if s["phase"] == label and s["rows_in_window"] >= 0]
    if not rows:
        return None
    rates = [s["rows_per_min"] for s in rows]
    ages = [s["newest_age_sec"] for s in rows]
    zero = [s for s in rows if s["rows_in_window"] == 0]
    return {
        "phase": label,
        "samples": len(rows),
        "rows_per_min_mean": round(sum(rates) / len(rates), 2),
        "rows_per_min_min": round(min(rates), 2),
        "rows_per_min_max": round(max(rates), 2),
        "zero_tick_samples": len(zero),
        "zero_tick_fraction": round(len(zero) / len(rows), 4),
        "age_max_sec": round(max(ages), 1),
        "age_median_sec": round(sorted(ages)[len(ages) // 2], 1),
        "system_cpu_mean": round(
            sum(s["system_cpu_pct"] for s in rows) / len(rows), 1
        ),
        "sweep_cpu_mean": round(
            sum(s["sweep"]["cpu_pct"] for s in rows) / len(rows), 1
        ),
        "production_cpu_mean": round(
            sum(s["production"]["cpu_pct"] for s in rows) / len(rows), 1
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="quiet", help="label for this phase")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--window", type=float, default=120.0,
                    help="trailing window, in seconds, for the tick count")
    ap.add_argument("--out", default="data/feed_contention_probe.jsonl")
    args = ap.parse_args()

    print("db=%s  production pids=%s  sweep pids=%s"
          % (DB_PATH, _production_pids(), _sweep_pids()), flush=True)

    samples: List[dict] = []
    run_phase(args.phase, args.seconds, args.interval, args.window, samples)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample) + "\n")

    summary = summarise(samples, args.phase)
    print("\nSUMMARY " + json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
