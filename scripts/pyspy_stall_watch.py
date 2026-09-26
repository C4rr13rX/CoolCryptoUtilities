"""[20e2f185 evidence] Sample the GIL holder of live production against the
feed's newest-row age, so the holder can be compared BETWEEN stalled and
healthy samples instead of looked at during whichever window happened to be
convenient.

A single dump during a healthy window shows what production does normally; it
cannot show what it does during a stall. This logs both and reports the split.

Read-only against the store. Attaches to production by pid via py-spy, which
does not require a restart or a code change, but does pause the target for a
few milliseconds per sample -- keep the sample count modest.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from typing import List, Tuple

PY_SPY = os.path.abspath(".venv/Scripts/py-spy.exe")
DB_PATH = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
LOOP_THREADS = ("asyncio", "Thread-10", "_run_supervisor_loop")


def _pids() -> List[int]:
    import psutil

    out = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(proc.info.get("cmdline") or [])
        if "--action start_production" in cmd:
            out.append(proc.info["pid"])
    return sorted(out)


def _holder(pid: int) -> Tuple[str, str, int]:
    """(thread name, top frame, total threads) for the GIL-holding thread."""
    try:
        res = subprocess.run(
            [PY_SPY, "dump", "--pid", str(pid)],
            capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired:
        return ("<dump timeout>", "", -1)
    blocks = re.split(r"(?m)^Thread ", res.stdout)
    total = max(0, len(blocks) - 1)
    for block in blocks[1:]:
        head = block.split("\n")[0]
        if "active+gil" not in head:
            continue
        name = head.split('"')[1] if '"' in head else head
        frames = [l.strip() for l in block.split("\n")[1:] if l.strip()]
        return (name, frames[0] if frames else "?", total)
    return ("<no gil holder>", "", total)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--stall-sec", type=float, default=60.0,
                    help="newest-row age above which a sample counts as stalled")
    args = ap.parse_args()

    pids = _pids()
    if not pids:
        print("no production process found -- nothing to sample")
        return 1
    pid = min(pids)  # the child is the bot; the other is its supervisor
    print("sampling pid %d of %s  stall threshold %.0fs" % (pid, pids, args.stall_sec),
          flush=True)

    conn = sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True, timeout=20)
    rows: List[Tuple[float, str, str]] = []
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        started = time.time()
        newest = conn.execute("SELECT MAX(ts) FROM market_stream").fetchone()[0]
        age = started - float(newest) if newest else float("nan")
        name, frame, total = _holder(pid)
        rows.append((age, name, frame))
        print("age=%7.1fs %-7s threads=%3d  GIL=%-28s %s"
              % (age, "STALLED" if age >= args.stall_sec else "ok", total, name, frame),
              flush=True)
        slack = args.interval - (time.time() - started)
        if slack > 0:
            time.sleep(slack)
    conn.close()

    stalled = [r for r in rows if r[0] >= args.stall_sec]
    healthy = [r for r in rows if r[0] < args.stall_sec]
    print("\nsamples=%d  stalled=%d  healthy=%d" % (len(rows), len(stalled), len(healthy)))
    for label, subset in (("STALLED", stalled), ("HEALTHY", healthy)):
        if not subset:
            print("%s: NO SAMPLES -- the comparison this script exists for was "
                  "NOT made" % label)
            continue
        loop = [r for r in subset if any(t in r[1] for t in LOOP_THREADS)]
        print("%s n=%d  loop-thread holds GIL %d/%d = %.1f%%"
              % (label, len(subset), len(loop), len(subset),
                 100.0 * len(loop) / len(subset)))
        for frame, count in Counter(r[2] for r in subset).most_common(6):
            print("    %3d  %s" % (count, frame))
    return 0


if __name__ == "__main__":
    sys.exit(main())
