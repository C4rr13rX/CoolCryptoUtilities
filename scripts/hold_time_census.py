"""Does holding shorter help? Bucket closed ghost round trips by hold duration
and by EXIT REASON, gross and net.

Pass 719 (Bay). Written to test the standing instruction's route (c), "SHORTEN
THE HOLD -- the tradeable book is POSITIVE inside 15 minutes and negative after".
On the join below it is the other way round: the 0-15m bucket is the second
WORST, and the only positive bucket is 4h-24h.

Hold duration is not stored on an outcome. It is recovered by joining
trade_outcomes.trade_id to the trading_ops 'ghost-entry' row that carries
details.entry_ts. Status is matched EXACTLY, never with LIKE -- 'quarantined-
ghost-entry' matches LIKE '%ghost-entry%' and would pull rejected rows back in.

    python scripts/hold_time_census.py
"""

from __future__ import annotations

import collections
import json
import os
import sqlite3
import statistics
import sys

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
BUCKETS = [(15, "0-15m"), (60, "15-60m"), (240, "60-240m"), (1440, "4h-24h"), (float("inf"), ">24h")]


def _bucket(minutes):
    for hi, label in BUCKETS:
        if minutes < hi:
            return label
    return BUCKETS[-1][1]


def load_rows(conn):
    """(hold_minutes, gross, net, fee, exit_reason) per closed round trip."""
    cur = conn.cursor()
    entries = {}
    cur.execute("select details from trading_ops where status = 'ghost-entry'")
    for (raw,) in cur.fetchall():
        d = json.loads(raw) if raw else {}
        trade_id = d.get("trade_id")
        entry_ts = d.get("entry_ts") or d.get("timestamp")
        if trade_id and entry_ts:
            entries.setdefault(trade_id, float(entry_ts))

    rows, unmatched = [], 0
    cur.execute("select trade_id, ts, gross_profit, net_profit, fee_cost, details from trade_outcomes")
    for trade_id, ts, gross, net, fee, raw in cur.fetchall():
        entry_ts = entries.get(trade_id)
        if entry_ts is None:
            unmatched += 1
            continue
        d = json.loads(raw) if raw else {}
        reason = (d.get("reason") or "?").split(":")[0]
        rows.append(((ts - entry_ts) / 60.0, gross or 0.0, net or 0.0, fee or 0.0, reason))
    return rows, unmatched, len(entries)


def _table(title, groups, order=None):
    print(f"\n{title}")
    print("  %-20s %5s %9s %8s %10s %9s %10s" % ("", "n", "grossWin", "netWin", "GROSS", "fees", "NET"))
    keys = order or [k for k, _ in sorted(groups.items(), key=lambda kv: -kv[1][0])]
    for key in keys:
        if key not in groups:
            continue
        n, gw, nw, gross, fee, net = groups[key]
        print("  %-20s %5d %8.1f%% %7.1f%% %+10.4f %9.4f %+10.4f"
              % (key, n, 100 * gw / n, 100 * nw / n, gross, fee, net))


def _group(rows, keyfn):
    out = collections.defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0.0])
    for minutes, gross, net, fee, reason in rows:
        acc = out[keyfn(minutes, reason)]
        acc[0] += 1
        acc[1] += gross > 0
        acc[2] += net > 0
        acc[3] += gross
        acc[4] += fee
        acc[5] += net
    # reorder to (n, grossWin, netWin, gross, fee, net)
    return {k: (v[0], v[1], v[2], v[3], v[4], v[5]) for k, v in out.items()}


def main():
    if not os.path.exists(DB):
        print(f"no trading db at {DB}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows, unmatched, n_entries = load_rows(conn)
    if not rows:
        print("no closed round trips could be joined to a ghost-entry", file=sys.stderr)
        return 1

    holds = [r[0] for r in rows]
    print(f"db {DB}")
    print(f"ghost-entry rows carrying trade_id+entry_ts : {n_entries}")
    print(f"closed round trips joined                   : {len(rows)}  (unjoinable: {unmatched})")
    print("hold minutes  min %.2f  median %.2f  max %.2f" % (min(holds), statistics.median(holds), max(holds)))

    _table("BY HOLD BUCKET", _group(rows, lambda m, r: _bucket(m)), [b for _, b in BUCKETS])
    _table("BY EXIT REASON", _group(rows, lambda m, r: r))
    _table("SUB-15-MINUTE BUCKET, BY EXIT REASON  (the kill test: is the short bucket "
           "positive only where the exit is conditioned on having already won?)",
           _group([r for r in rows if r[0] < 15], lambda m, r: r))
    _table("WHOLE BOOK", _group(rows, lambda m, r: "ALL"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
