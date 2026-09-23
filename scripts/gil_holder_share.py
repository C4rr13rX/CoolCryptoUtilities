#!/usr/bin/env python
"""Who actually holds the GIL in production, measured over an HOUR, not a burst.

Criterion 1 of backlog item [7e9ba214]. Bay, pass 672.

WHY THIS EXISTS. Four different threads were named as "the GIL holder" in
twenty-five minutes, each from a burst of py-spy dumps, and each died on its
next sample:

    Echo p671   5 dumps  mem 83.9%   governor gc.collect()      4/5
    Bay  b1     4 dumps  mem 79.3%   pair-select_0              3/3
    Bay  b2    10 dumps  mem 79.3%   news_enrichment-runner     7/10   (pair-select 0/10)
    Bay  b3     8 dumps  mem 80.0%   genome-signal-feed         7/8    (news_enrich 1/8)

A burst of dumps measures who happened to be running in that three-minute
window. It does not measure who blocks the loop. This process cycles its hot
thread faster than a burst can resolve, and the answer demonstrably changes
when system memory crosses the 0.82 GOVERNOR_CRITICAL_MEM threshold.

So: sample SPARSELY over a long window, record memory beside every dump, and
report each thread's share of active+gil against the chance rate (1/nthreads),
bucketed by memory. Nothing here edits anything; it is read-only.

    python scripts/gil_holder_share.py --pid 6340
    python scripts/gil_holder_share.py --pid 6340 --minutes 60 --every 70
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time

_THREAD_RE = re.compile(r'Thread (\d+) \(([^)]*)\): "([^"]+)"')

# py-spy ships in the venv on this box; fall back to PATH.
_DEFAULT_PYSPY = os.path.join(".venv", "Scripts", "py-spy.exe")


def _pyspy_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.path.exists(_DEFAULT_PYSPY):
        return _DEFAULT_PYSPY
    return "py-spy"


def _memory_percent() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().percent)
    except Exception:
        return float("nan")


def _dump(pyspy: str, pid: int, timeout: float) -> str:
    try:
        out = subprocess.run(
            [pyspy, "dump", "--pid", str(pid)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return out.stdout or ""
    except Exception as exc:  # a timed-out dump is data too: report, do not crash
        print(f"  dump failed: {exc}", file=sys.stderr)
        return ""


def _holder(dump_text: str) -> tuple[str | None, int]:
    """Return (name of the active+gil thread, total thread count)."""
    threads = _THREAD_RE.findall(dump_text)
    holder = None
    for _tid, state, name in threads:
        if "gil" in state:
            holder = name
            break
    return holder, len(threads)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True, help="production pid (main.py --action start_production)")
    ap.add_argument("--minutes", type=float, default=60.0, help="total window; default 60")
    ap.add_argument("--every", type=float, default=70.0, help="seconds between dumps; default 70")
    ap.add_argument("--pyspy", default=None)
    ap.add_argument("--dump-timeout", type=float, default=90.0)
    ap.add_argument("--json-out", default=None, help="write the raw per-dump rows here")
    args = ap.parse_args()

    pyspy = _pyspy_path(args.pyspy)
    deadline = time.time() + args.minutes * 60.0
    rows: list[dict] = []

    print(f"pid={args.pid}  window={args.minutes:.0f}min  every={args.every:.0f}s  pyspy={pyspy}")
    print("NOTE: sparse-by-design. A burst gives a different answer every time -- that is the finding.")

    i = 0
    while time.time() < deadline:
        i += 1
        mem = _memory_percent()
        text = _dump(pyspy, args.pid, args.dump_timeout)
        holder, nthreads = _holder(text)
        rows.append({"i": i, "ts": time.time(), "mem_pct": mem, "holder": holder, "nthreads": nthreads})
        print(f"  dump {i:3d}  mem={mem:5.1f}%  threads={nthreads:3d}  gil={holder}")
        sleep_for = args.every - 0  # dumps are slow; do not compound the wait
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(max(0.0, min(sleep_for, remaining)))

    if not rows:
        print("no dumps captured")
        return 1

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print(f"raw rows -> {args.json_out}")

    _report(rows, "ALL DUMPS")
    lo = [r for r in rows if r["mem_pct"] == r["mem_pct"] and r["mem_pct"] < 82.0]
    hi = [r for r in rows if r["mem_pct"] == r["mem_pct"] and r["mem_pct"] >= 82.0]
    # The whole point: the answer changed when memory crossed the governor's
    # 0.82 GOVERNOR_CRITICAL_MEM branch. A pooled share hides that.
    if lo:
        _report(lo, "MEMORY UNDER 82% (governor critical branch NOT firing)")
    if hi:
        _report(hi, "MEMORY AT OR ABOVE 82% (governor critical branch firing)")
    if not lo or not hi:
        print("\nONLY ONE MEMORY BUCKET WAS OBSERVED. Do not generalise across the")
        print("threshold from this run -- say which bucket the number came from.")
    return 0


def _report(rows: list[dict], label: str) -> None:
    n = len(rows)
    held = [r for r in rows if r["holder"]]
    counts = collections.Counter(r["holder"] for r in held)
    threads = [r["nthreads"] for r in rows if r["nthreads"]]
    chance = (100.0 / (sum(threads) / len(threads))) if threads else float("nan")
    print(f"\n=== {label} ===")
    print(f"dumps={n}  with a gil holder={len(held)}  mean threads/dump={sum(threads)/len(threads):.1f}"
          f"  chance rate={chance:.1f}%")
    if n < 30:
        print(f"WARNING: n={n} is a BURST, not a window. This is the sample size that produced")
        print("four contradictory answers in twenty-five minutes. Do not quote a holder off it.")
    for name, c in counts.most_common(10):
        print(f"  {name:32s} {c:3d}/{n:<3d} = {100*c/n:5.1f}%   ({100*c/n/chance:.1f}x chance)")
    print("\nDUTY CYCLE IS NOT SHARE. Before changing anything, quote how long one call")
    print("takes and how often it is scheduled. Known: news_enrichment is 25s on a 900s")
    print("interval = 2.8% MAX (production.py:594-598); the governor gc branch fires on")
    print("~21% of samples. 74.4% of polls blow the 10s budget, so no single candidate")
    print("measured so far can account for it -- expect partial wins, not a fix.")


if __name__ == "__main__":
    raise SystemExit(main())
