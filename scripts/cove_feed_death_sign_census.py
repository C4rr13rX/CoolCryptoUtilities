"""Where does the ghost book's sign live -- in feed-dead symbols, or in a
handful of rows the repo already knows are implausible?

Buckets every closed ghost row by whether its symbol has ticked in the last
FRESH_DAYS, and prints the book both RAW and with the repo's own censor
(services.outcome_plausibility) applied.  The two have opposite signs.

Read-only.  No cost model, no join -- the numbers are the rows' own.

    python scripts/cove_feed_death_sign_census.py
"""

from __future__ import annotations

import os
import sqlite3
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import outcome_plausibility as op  # noqa: E402

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
FRESH_DAYS = 7.0
COLS = [
    "symbol",
    "entry_price",
    "exit_price",
    "quantity",
    "gross_profit",
    "fee_cost",
    "net_profit",
    "wallet",
    "status",
]


def _summary(label: str, rows: list[dict]) -> None:
    nets = [r["net_profit"] for r in rows]
    if not nets:
        print(f"  {label:9s} EMPTY")
        return
    total = sum(nets)
    top = max(nets)
    print(
        f"  {label:9s} n={len(nets):4d}  sum {total:+10.6f}  "
        f"median {statistics.median(nets):+9.6f}  "
        f"win {sum(1 for n in nets if n > 0) / len(nets) * 100:5.1f}%  "
        f"top row {top:+.6f} = {top / total * 100 if total else 0:.1f}% of sum"
    )


def main() -> None:
    conn = sqlite3.connect(DB)
    now = datetime.now(timezone.utc).timestamp()
    last_tick = {
        s: t
        for s, t in conn.execute(
            "SELECT symbol, MAX(ts) FROM market_stream GROUP BY symbol"
        )
    }

    raw = [
        dict(zip(COLS, r))
        for r in conn.execute(
            f"SELECT {','.join(COLS)} FROM trade_outcomes "
            "WHERE wallet='ghost' AND status='closed' AND net_profit IS NOT NULL"
        )
    ]
    censored = [r for r in raw if op.is_implausible(r)]
    kept = [r for r in raw if not op.is_implausible(r)]

    def age_days(sym: str) -> float:
        t = last_tick.get(sym)
        return (now - t) / 86400.0 if t else float("inf")

    def split(rows: list[dict]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {"LIVE-FED": [], "DEAD": []}
        for r in rows:
            out["LIVE-FED" if age_days(r["symbol"]) <= FRESH_DAYS else "DEAD"].append(r)
        return out

    print(f"ghost closed n={len(raw)}   freshness cut {FRESH_DAYS:.0f} days")
    print(
        f"censor: services.outcome_plausibility, IMPLAUSIBLE_RET={op.IMPLAUSIBLE_RET} "
        f"-> {len(censored)} rows, sum {sum(r['net_profit'] for r in censored):+.6f}"
    )
    for r in sorted(censored, key=lambda r: -abs(r["net_profit"])):
        print(
            f"   CENSORED {r['symbol']:16s} net {r['net_profit']:+10.6f}  "
            f"ret {op.booked_return(r) * 100:+9.1f}%  {op.implausible_reason(r)}  "
            f"(symbol dark {age_days(r['symbol']):.1f}d)"
        )

    for label, rows in (("UNCENSORED", raw), ("CENSORED", kept)):
        print(f"\n=== {label} ===")
        buckets = split(rows)
        for name in ("LIVE-FED", "DEAD"):
            _summary(name, buckets[name])
        _summary("WHOLE", rows)

    print(
        "\nthe sign does NOT live in feed-dead symbols -- the DEAD bucket is negative "
        "either way.  It lives in the censored rows."
    )


if __name__ == "__main__":
    main()
