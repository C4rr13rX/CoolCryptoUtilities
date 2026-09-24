#!/usr/bin/env python
"""What the chaos gate ACTUALLY did on every entry it let through.

`entry-refused-lattice` rows are easy to count and they tell you only half the
story: they are the trades the gate STOPPED.  The other half -- the trades it
did not stop -- is the half that becomes ghost P/L, and until pass 692 nobody
had counted it.

The gate is `TradingBot._lattice_refusal` (trading/bot.py:2261).  It returns a
refusal string, or `None` meaning PERMIT.  It returns `None` from seven
different places, and only ONE of them means "I judged this and it is fine":

    disabled                        the leg is switched off
    not_an_entry                    exits are not forecasts
    expected_return_not_positive    nothing to judge
    unreadable_horizon:<label>      horizon_seconds() returned <= 0
    short_window:<n>                fewer than 64 prices in the ring buffer
    exception:<Type>                the leg threw and failed open
    answered_pass                   <-- the only one that is a judgement

The caller stamps the exit onto the entry row as `details.lattice_exit`, so the
distinction is recoverable from the store.  This script counts it.

MEASURED BY SAGE p692 over 7 days, `action='enter' AND status='ghost-entry'`,
n=72 of which 44 carry a stamped exit (stamping begins 2026-09-18; the 28 with
no stamp are older rows and are reported separately rather than pooled in):

    short_window:20/21/60      29   65.9%   failed open -- not enough tape
    unreadable_horizon:atf     11   25.0%   failed open -- label unparseable
    answered_pass               4    9.1%   ACTUALLY JUDGED AND PERMITTED
    answered_refuse             0    0.0%

40 of 44 entries got in because the gate could not answer, not because it said
yes.  The gate REFUSES where the tape is thick (4866 chaos refusals in 7d, on
symbols carrying 459-1059 lattice rows each) and ADMITS where it is thin.

`--buffer` breaks out the short_window mode, which is the actionable one.  The
threshold at bot.py:2386 is 64 and the observed lengths are BIMODAL: 26 rows at
20-21 (a hard floor, a genuinely starved buffer) and 19 rows at 60-63, which
missed being judged by one to four ticks.  `self._buffer` is an in-process ring
that starts EMPTY at boot, so this is a WARM-UP fact, not a feed fact and not a
query fact -- the two the census was first written to distinguish.

Read-only.  Nothing here writes to the store.

    python scripts/lattice_exit_census.py            # 7 days
    python scripts/lattice_exit_census.py 30         # 30 days
    python scripts/lattice_exit_census.py 7 --buffer # + the len(prices) mode
    python scripts/lattice_exit_census.py 7 --strategy rsi_reversal@12h
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sqlite3
import sys

DEFAULT_DB = os.getenv("TRADING_DB_PATH", "storage/trading_cache.db")

# The one exit that is a judgement rather than a shrug.  Everything else on
# this list reaches `return None` without measuring anything.
JUDGED = "answered_pass"


def _open(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        sys.exit(f"no store at {path} -- set TRADING_DB_PATH")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _rows(conn: sqlite3.Connection, days: float, strategy: str | None):
    sql = (
        "SELECT ts, symbol, details FROM trading_ops "
        "WHERE action='enter' AND status='ghost-entry' AND ts >= ? ORDER BY ts"
    )
    cut = _dt.datetime.now().timestamp() - days * 86400
    for ts, symbol, blob in conn.execute(sql, (cut,)):
        try:
            detail = json.loads(blob) or {}
        except Exception:
            continue
        if strategy:
            plan = detail.get("bus_plan") or {}
            if plan.get("strategy_id") != strategy:
                continue
        yield ts, symbol, detail


def _fmt(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def _floor(conn: sqlite3.Connection, symbol: str) -> int:
    """Where does chaos_profile FIRST answer?  Call it; do not read it.

    Pass 692 read `len(rets) < 32` at chaos.py:78/148/224, inferred a 33-price
    floor, and concluded the 64 at bot.py:2386 was arbitrary and roughly twice
    what the estimator needs.  That was wrong, and one call showed it: the
    `< 32` checks are not the binding constraint and the profile first answers
    at 64 exactly.  This mode exists so the next agent does not repeat the
    inference.
    """
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "web"))
    try:
        from tradingagent.chaos import chaos_profile
    except Exception as exc:  # noqa: BLE001
        print(f"cannot import chaos_profile: {exc}")
        return 1

    prices = [float(p) for (p,) in conn.execute(
        "SELECT price FROM market_stream WHERE symbol=? AND price>0 "
        "ORDER BY ts DESC LIMIT 400", (symbol,))][::-1]
    print(f"{symbol}: {len(prices)} real prices from market_stream")
    if len(prices) < 64:
        print("  fewer than 64 available -- pick a denser symbol")
        return 0

    print("  the guard at bot.py:2386 permits below 64. Does the estimator"
          " actually need 64?")
    first = None
    for n in (20, 21, 33, 60, 61, 62, 63, 64, 100, 200, 400):
        if n > len(prices):
            continue
        usable = chaos_profile(symbol, prices[-n:], bar_sec=30.0
                               ).get("usable_horizon_sec")
        if usable is not None and first is None:
            first = n
        print("    n=%-4d usable_horizon_sec = %s"
              % (n, "None" if usable is None else f"{float(usable):.2f}"))
    print(f"  first answers at n={first}; the guard's literal is 64")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("days", nargs="?", type=float, default=7.0)
    ap.add_argument("--buffer", action="store_true",
                    help="break out len(prices) on the short_window rows")
    ap.add_argument("--strategy", default=None,
                    help="restrict to one bus_plan.strategy_id")
    ap.add_argument("--floor", metavar="SYMBOL", nargs="?", const="ETH-USDC",
                    help="probe chaos_profile at increasing n to find the "
                         "sample count it FIRST answers at")
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args()

    conn = _open(args.db)
    if args.floor:
        return _floor(conn, args.floor)
    rows = list(_rows(conn, args.days, args.strategy))
    if not rows:
        print(f"no ghost-entry rows in {args.days:g}d"
              + (f" for {args.strategy}" if args.strategy else ""))
        return 0

    exits = collections.Counter()
    unstamped = 0
    short: list[tuple[float, str, int]] = []
    for ts, symbol, detail in rows:
        exit_name = detail.get("lattice_exit")
        if not exit_name:
            unstamped += 1
            continue
        exits[exit_name] += 1
        if str(exit_name).startswith("short_window:"):
            try:
                short.append((ts, symbol, int(str(exit_name).split(":", 1)[1])))
            except ValueError:
                pass

    stamped = sum(exits.values())
    head = f"ghost entries, last {args.days:g}d"
    if args.strategy:
        head += f", strategy={args.strategy}"
    print(head)
    print(f"  total rows           {len(rows)}")
    print(f"  carry lattice_exit   {stamped}")
    print(f"  no stamp (older)     {unstamped}   <- reported, NOT pooled")
    if not stamped:
        print("\nnothing stamped in this window -- widen --days")
        return 0

    print(f"\n  what the gate did, over the {stamped} readable rows:")
    failed_open = 0
    for name, count in exits.most_common():
        judged = str(name) == JUDGED
        if not judged:
            failed_open += count
        print("    %-30s %4d  %5.1f%%  %s"
              % (name, count, 100.0 * count / stamped,
                 "JUDGED" if judged else "failed open"))
    print("    %-30s %4d  %5.1f%%"
          % ("-- failed open, total", failed_open,
             100.0 * failed_open / stamped))
    print("    %-30s %4d  %5.1f%%"
          % ("-- actually judged", stamped - failed_open,
             100.0 * (stamped - failed_open) / stamped))

    if args.buffer and short:
        print(f"\n  short_window: len(prices) against the >=64 at bot.py:2386")
        for length, count in sorted(collections.Counter(
                n for _, _, n in short).items()):
            print("    %3d  %-24s %d" % (length, "#" * min(count, 24), count))
        near = [n for _, _, n in short if 64 - n <= 4]
        print("    %d of %d are within 4 samples of the threshold -- but see"
              " --floor: chaos_profile" % (len(near), len(short)))
        print("    returns None at 60-63 too, so these are NOT a cheap win")
        days = {_fmt(ts)[:5] for ts, _, _ in short}
        print("    spread %s .. %s over %d distinct day(s)"
              % (_fmt(short[0][0]), _fmt(short[-1][0]), len(days)))
        by_symbol = collections.Counter(sym for _, sym, _ in short)
        print("    symbols: " + ", ".join(
            f"{s}({c})" for s, c in by_symbol.most_common(8)))
    elif args.buffer:
        print("\n  no short_window rows in this window")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
