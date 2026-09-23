"""Count production INSTANCES, not PIDs.

Why this exists. p679 measured four `main.py --action start_production` PIDs and
could not say whether that was two instances or one, because the answer is in
ParentProcessId and Windows does not keep it for a dead process (4688 auditing
is off on this box). By the time anyone asked, the processes were gone.

The routine reading is TWO PIDs, not one: the live instance runs as a
parent/child pair with an identical command line (measured 2026-09-23,
pid 3584 -> pid 20452, same second, same argv, launched by run_waitress.py).
So "two processes" is not evidence of anything. The number that matters is the
count of distinct ROOTS -- a production process whose parent is not itself a
production process.

    .venv/Scripts/python.exe -X utf8 scripts/production_instances.py

Exit code is the number of instances, so a supervisor can branch on it:
0 = production is down, 1 = healthy, 2+ = duplicates and every zero this repo
measures over that window is unfalsifiable.

Since commit 551343c, main.py refuses to start a second instance, so a reading
of 2+ here means either a process that predates that commit (production serves
its BOOT-TIME code -- check the start times printed below against the commit) or
a defect in the guard.
"""
from __future__ import annotations

import datetime as _dt
import sys


def _is_production(cmdline) -> bool:
    joined = " ".join(str(part) for part in (cmdline or []))
    return "main.py" in joined and "start_production" in joined


def main() -> int:
    try:
        import psutil
    except ImportError:
        print("psutil is not installed; cannot enumerate processes.")
        return 255

    procs = {}
    for proc in psutil.process_iter(["pid", "ppid", "cmdline", "create_time"]):
        try:
            if _is_production(proc.info.get("cmdline")):
                procs[proc.info["pid"]] = proc.info
        except Exception:  # noqa: BLE001 - a vanished process is not an error here
            continue

    if not procs:
        print("PRODUCTION INSTANCES: 0   (production is DOWN)")
        return 0

    roots = [info for pid, info in procs.items() if info.get("ppid") not in procs]

    print(f"PRODUCTION INSTANCES: {len(roots)}   ({len(procs)} pids)")
    for root in sorted(roots, key=lambda i: i.get("create_time") or 0):
        started = _dt.datetime.fromtimestamp(root["create_time"]).isoformat(timespec="seconds")
        kids = [p for p, i in procs.items() if i.get("ppid") == root["pid"]]
        parent = "?"
        try:
            parent = psutil.Process(root["ppid"]).name()
            parent = f"{root['ppid']} ({parent})"
        except Exception:  # noqa: BLE001
            parent = str(root.get("ppid"))
        print(f"  root pid {root['pid']}  started {started}  launched-by {parent}  children {kids}")

    if len(roots) > 1:
        print("")
        print("  DUPLICATE INSTANCES. Every count measured over this window --")
        print("  ghost entries, refusals, tick rates -- is pooled across two")
        print("  processes and cannot be read as one system's behaviour.")
    return len(roots)


if __name__ == "__main__":
    sys.exit(main())
