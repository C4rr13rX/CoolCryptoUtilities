"""Print the loop_lag_sec population from the only source the code writes it to.

Pass 672 (Sage), item 307d969d. Criterion 4 of that item says to quote the
post-restart median against "Echo p671 post-restart baseline: 6h median 14.9s
p90 59.7s max 585.3s n=9643". That baseline is not reproducible: the single
writer of loop_lag_sec is trading/data_stream.py:3685, it emits the value in the
``details`` JSON of a severity=warning message, and that lands in
``system_logs.details`` in the trading store. The whole population of that column
at the time of writing was 29 rows -- 332x fewer than the 9643 the baseline
quotes -- and the value appears in no log file on disk (grep over the last 30MB
of logs/prod_direct.err and 20MB of logs/system.log returns nothing).

So this script exists to make the number checkable rather than asserted. It
prints n, median, p90, max and the share over the 10s budget for each window,
and it prints the size of the whole population first so a small n is impossible
to quote by accident.

    python scripts/loop_lag_baseline.py
    python scripts/loop_lag_baseline.py --hours 6 --hours 24

``created_at`` in this table is written as a naive UTC ISO string, so the
windows are computed in UTC. A restart does not truncate the table, so rows from
before the current boot are included unless --since-boot is passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import statistics
import sys


DEFAULT_DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
BUDGET_SEC = 10.0


def _quantile(sorted_values: list[float], frac: float) -> float:
    return sorted_values[int(frac * (len(sorted_values) - 1))]


def _lags(rows: list[tuple[str, str]], cutoff: str | None) -> list[float]:
    out: list[float] = []
    for created_at, details in rows:
        if cutoff is not None and str(created_at) < cutoff:
            continue
        try:
            value = json.loads(details).get("loop_lag_sec")
        except (TypeError, ValueError):
            continue
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _report(label: str, lags: list[float]) -> None:
    if not lags:
        print(f"  {label:<24} n=0  -- nothing to quote")
        return
    ordered = sorted(lags)
    over = sum(1 for x in ordered if x > BUDGET_SEC)
    print(
        f"  {label:<24} n={len(ordered):<6} "
        f"median={statistics.median(ordered):7.2f}s "
        f"p90={_quantile(ordered, 0.90):8.2f}s "
        f"max={ordered[-1]:9.2f}s "
        f"over{BUDGET_SEC:.0f}s={100 * over / len(ordered):5.1f}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument(
        "--hours",
        type=float,
        action="append",
        help="window in hours; repeatable (default: 1, 6, 24, and all time)",
    )
    args = parser.parse_args()
    windows = args.hours or [1.0, 6.0, 24.0]

    if not os.path.exists(args.db):
        print(f"no such db: {args.db}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")

    total, newest, oldest = conn.execute(
        "select count(*), max(created_at), min(created_at) from system_logs "
        "where details like '%loop_lag_sec%'"
    ).fetchone()

    print(f"db: {args.db}")
    print("source: system_logs.details <- trading/data_stream.py:3685 (REST timeout branch)")
    print(f"WHOLE POPULATION: n={total}  oldest={oldest}  newest={newest}")
    if not total:
        print("nothing written -- either no REST timeout has been charged, or the writer changed")
        return 1
    if total < 200:
        print(
            f"WARNING: n={total} is too small to carry a verdict. "
            "Do not quote a median off this without saying n."
        )

    rows = conn.execute(
        "select created_at, details from system_logs "
        "where details like '%loop_lag_sec%' order by id desc"
    ).fetchall()

    print(f"budget: {BUDGET_SEC:.0f}s")
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    for hours in windows:
        cutoff = (now - dt.timedelta(hours=hours)).isoformat(sep=" ")
        _report(f"last {hours:g}h", _lags(rows, cutoff))
    _report("all time", _lags(rows, None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
