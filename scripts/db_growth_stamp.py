"""Stamp the trading store's size and the volume's free space, and report the
growth rate over the interval between the first and last stamp.

Backlog item [1706fe9e]: the volume holding storage/trading_cache.db is ~99%
full. The growth RATE is what decides that item's priority, and a rate cannot be
measured inside one pass -- so this appends a stamp and computes the rate across
however many passes have run it.

    python -X utf8 scripts/db_growth_stamp.py

Read-only: it stats files and the volume. It never opens the database, so it
cannot disturb the live writer. Safe to run against production at any time.

See also scripts/wal_frame_census.py, which parses the WAL's frame headers to
show how much of that file is LIVE rather than dead high-water mark (pass 481:
0.0557% live).
"""
import json
import os
import shutil
import sys
import time

STAMPS = "data/db_growth_stamps.json"
TARGETS = (
    "storage/trading_cache.db",
    "storage/trading_cache.db-wal",
    "storage/trading_cache.db-shm",
)
MAIN = TARGETS[0]


def _sizes():
    out = {}
    for path in TARGETS:
        try:
            out[path] = os.path.getsize(path)
        except OSError as exc:
            out[path] = f"ERR {exc}"
    return out


def main():
    note = sys.argv[1] if len(sys.argv) > 1 else ""
    volume = os.path.abspath(MAIN)[:3]
    usage = shutil.disk_usage(volume)

    stamp = {
        "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stamped_epoch": int(time.time()),
        "note": note,
        "sizes_bytes": _sizes(),
        "volume": {
            "path": volume,
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
            "pct_used": round(100.0 * usage.used / usage.total, 2),
        },
    }

    history = []
    if os.path.exists(STAMPS):
        try:
            with open(STAMPS, encoding="utf-8") as fh:
                history = json.load(fh)
        except Exception:
            history = []
    history.append(stamp)
    with open(STAMPS, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    db = stamp["sizes_bytes"][MAIN]
    free = stamp["volume"]["free"]
    print(f"stamped {stamp['stamped_utc']}   (stamp #{len(history)})")
    print(f"  {MAIN:34s} {db:>16,} bytes = {db / 1024 ** 3:7.2f} GiB")
    print(f"  volume {volume} free{'':22s} {free:>16,} bytes = {free / 1024 ** 3:7.2f} GiB"
          f"  ({stamp['volume']['pct_used']}% used)")

    if len(history) < 2:
        print("\nONE stamp only -- the growth RATE is UNMEASURED.")
        print("Run this again on a later pass (an hour apart is plenty) to get it.")
        return

    first, last = history[0], history[-1]
    seconds = last["stamped_epoch"] - first["stamped_epoch"]
    if seconds <= 0:
        print("\nstamps are not separated in time; cannot compute a rate.")
        return

    grew = last["sizes_bytes"][MAIN] - first["sizes_bytes"][MAIN]
    per_day = grew / seconds * 86400.0
    print(f"\nover {seconds / 3600:.2f} h across {len(history)} stamps:")
    print(f"  db growth      {grew:>16,} bytes = {per_day / 1e9:.3f} GB/day"
          f" ({per_day / 1024 ** 2 / 24:.1f} MiB/hour)")
    if per_day > 0:
        days = free / per_day
        print(f"  HEADROOM       {days:.1f} days at this rate")
        if days < 30:
            print("  *** UNDER 30 DAYS -- [1706fe9e] is priority 1. Tell the operator. ***")
    else:
        print("  db is not growing over this interval.")
    if seconds < 3600:
        print("  CAVEAT: interval under an hour. A burst inside it overstates the rate;")
        print("  re-run later for a longer baseline before treating this as a date.")


if __name__ == "__main__":
    main()
