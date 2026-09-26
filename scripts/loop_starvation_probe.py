"""Sample the production ingest rate on a schedule, not once.

[20e2f185] acceptance: "ONE OBSERVATION OF A TIME SERIES IS NOT ITS LEVEL."
Echo's p713 headline ("the feed is dead") came from a single point reading and
was wrong; p714 re-read the same process and found a healthy tape. So this
probe never reports a level -- it reports a series and a duty cycle.

Read-only against the store. It opens the db with mode=ro so it cannot itself
add write pressure to the thing it is measuring.

    python scripts/loop_starvation_probe.py --seconds 600 --every 15
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import statistics
import sys
import time

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")

# The two independent witnesses of "is the ingest draining".
#   data_stream metrics  -- written by the sampler that feeds the strategies
#   trading_ops          -- written by the decision loop on the same event loop
PROBES = (
    ("data_stream", "select count(*) from metrics where stage='data_stream' and ts>?"),
    ("live_trading", "select count(*) from metrics where stage='live_trading' and ts>?"),
    ("trading_ops", "select count(*) from trading_ops where ts>?"),
)

AGE_PROBES = (
    ("data_stream", "select max(ts) from metrics where stage='data_stream'"),
    ("trading_ops", "select max(ts) from trading_ops"),
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def sample(conn: sqlite3.Connection, window: float) -> dict:
    now = time.time()
    row = {"t": now, "utc": dt.datetime.fromtimestamp(now, dt.UTC).strftime("%H:%M:%S")}
    for label, sql in PROBES:
        (n,) = conn.execute(sql, (now - window,)).fetchone()
        row[f"{label}_per_min"] = round(n * 60.0 / window, 2)
    for label, sql in AGE_PROBES:
        (mx,) = conn.execute(sql).fetchone()
        row[f"{label}_age_sec"] = round(now - mx, 1) if mx else None
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--every", type=float, default=15.0)
    ap.add_argument("--window", type=float, default=60.0, help="rate window per sample")
    ap.add_argument("--out", default="data/loop_starvation_probe.jsonl")
    args = ap.parse_args(argv)

    conn = _conn()
    deadline = time.time() + args.seconds
    rows: list[dict] = []
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        while time.time() < deadline:
            row = sample(conn, args.window)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"{row['utc']}  data_stream {row['data_stream_per_min']:7.2f}/min"
                f"  age {row['data_stream_age_sec']}s"
                f"  | live_trading {row['live_trading_per_min']:7.2f}/min"
                f"  | trading_ops {row['trading_ops_per_min']:6.2f}/min"
                f"  age {row['trading_ops_age_sec']}s",
                flush=True,
            )
            time.sleep(max(1.0, args.every))

    if not rows:
        print("no samples")
        return 1

    # Duty cycle: the acceptance criterion asks what FRACTION of wall time is
    # starved, which is a property of the series and not of any one sample.
    series = [r["data_stream_per_min"] for r in rows]
    starved = [r for r in rows if r["data_stream_per_min"] <= 0.0]
    print()
    print(f"samples {len(rows)} over {args.seconds:.0f}s at {args.every:.0f}s spacing")
    print(f"data_stream rows/min: min {min(series):.2f}  median "
          f"{statistics.median(series):.2f}  max {max(series):.2f}")
    print(f"STARVED (0 rows/min) in {len(starved)} of {len(rows)} samples "
          f"= duty cycle {100.0 * len(starved) / len(rows):.1f}%")
    ages = [r["data_stream_age_sec"] for r in rows if r["data_stream_age_sec"] is not None]
    if ages:
        print(f"newest-row age sec: min {min(ages):.1f}  median "
              f"{statistics.median(ages):.1f}  max {max(ages):.1f}")
    print(f"series written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
