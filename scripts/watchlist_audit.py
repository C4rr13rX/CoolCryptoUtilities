#!/usr/bin/env python3
"""Audit ``services.watchlists`` against the tape that market_stream actually wrote.

[00e3666a], Echo p698.

Three questions, and the third is the one that produced the finding:

  1. Which watchlist entries have a tape, and which never did? THIN IS NOT DEAD.
     An entry with 28 ticks that stopped two days ago is a DROPPED feed; an entry
     with zero ticks all-time never resolved to a pair in the first place. They
     are different faults with different fixes and pooling them hides both, so
     this script never reports a single "dead" count.

  2. Which entries are MALFORMED? A watchlist entry is a pair symbol, BASE-QUOTE.
     Anything with three or more tokens ("TRUE-WETH-USDC", "CATALYST-USDC-USDC")
     cannot resolve: trading/selector.py:929 splits on "-" and builds a
     PairCandidate with one token per part, so a 3-token entry asks the feed for
     a pair that does not exist. Measured p698: 13 of 13 such entries had ticked
     zero times all-time, 52% of the whole never-ticked population.

  3. Does the tape contain actively-ticking symbols that are NOT on the stream
     list? That is the expensive direction -- a dead entry is dead weight, but a
     live symbol nobody subscribes to is forgone evidence.

WHERE A MALFORMED ENTRY COMES FROM. Two writers merge into watchlists.stream and
only one is guarded:

    services/discovery/coordinator.py:92-97   GUARDED -- requires exactly
        BASE-QUOTE, requires the quote in {USDC,USDT,DAI}, and promotes the
        symbol VERBATIM. Carries a comment describing this exact bug.

    services/discovery/stream_bridge.py:144   UNGUARDED -- one f-string,
        ``f"{entry['symbol']}-USDC"``, appended to whatever the discovery feed
        called the token. A feed that returns a pair yields BASE-QUOTE-USDC.

A fix lands on stream_bridge.py:144 and is a copy of coordinator.py:92-97. This
script does NOT edit the watchlist: editing it changes what production streams,
which is a separate change with its own back-to-back measurement.

    python scripts/watchlist_audit.py
    python scripts/watchlist_audit.py --dark-hours 24 --show-absent 20
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: A watchlist entry is BASE-QUOTE. More parts than this and it cannot resolve.
_PAIR_TOKEN_COUNT = 2

#: Quotes a well-formed entry may carry, mirroring coordinator.py:96.
_VALID_QUOTES = {"USDC", "USDT", "DAI"}


def _db_path() -> str:
    return os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return _dt.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def _malformed_reason(symbol: str) -> str | None:
    """Why this entry can never resolve to a pair, or None if it is well-formed."""
    parts = [p for p in symbol.split("-") if p]
    if len(parts) > _PAIR_TOKEN_COUNT:
        return f"{len(parts)} tokens (needs {_PAIR_TOKEN_COUNT}): {'+'.join(parts)}"
    if len(parts) < _PAIR_TOKEN_COUNT:
        return "no quote token"
    if parts[1].upper() not in _VALID_QUOTES:
        return f"quote {parts[1].upper()} not in {sorted(_VALID_QUOTES)}"
    return None


def _tape(cur: sqlite3.Cursor, symbols: Sequence[str]) -> Dict[str, Tuple[int, float | None, float | None]]:
    """all-time (count, first_ts, last_ts) per symbol, one indexed query each."""
    out: Dict[str, Tuple[int, float | None, float | None]] = {}
    for sym in symbols:
        cur.execute(
            "SELECT COUNT(*), MIN(ts), MAX(ts) FROM market_stream WHERE symbol = ?",
            (sym,),
        )
        n, first, last = cur.fetchone()
        out[sym] = (int(n or 0), first, last)
    return out


def _ticking_symbols(cur: sqlite3.Cursor, since: float, min_ticks: int) -> List[Tuple[str, int, float]]:
    cur.execute(
        "SELECT symbol, COUNT(*) AS n, MAX(ts) FROM market_stream "
        "WHERE ts >= ? GROUP BY symbol HAVING n >= ? ORDER BY n DESC",
        (since, min_ticks),
    )
    return [(r[0], int(r[1]), float(r[2])) for r in cur.fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dark-hours", type=float, default=24.0,
                    help="an entry whose last tick is older than this counts as DROPPED (default 24)")
    ap.add_argument("--window-hours", type=float, default=24.0,
                    help="window for the 'ticking but absent from stream' sweep (default 24)")
    ap.add_argument("--min-ticks", type=int, default=5,
                    help="ticks required in the window before a symbol counts as actively ticking")
    ap.add_argument("--show-absent", type=int, default=25, help="cap rows in the absent table")
    args = ap.parse_args()

    try:
        from services.watchlists import load_watchlists
    except Exception as exc:  # noqa: BLE001 - the import IS the finding if it fails
        print(f"could not import services.watchlists: {exc}")
        return 2

    watchlists = load_watchlists() or {}
    now = time.time()
    dark_cut = args.dark_hours * 3600.0

    conn = sqlite3.connect(_db_path())
    cur = conn.cursor()

    print("=" * 78)
    print("WATCHLIST AUDIT -- entries vs the tape market_stream actually wrote")
    print("=" * 78)
    print(f"  db      : {_db_path()}")
    print(f"  as of   : {_fmt_ts(now)}")
    for name in ("stream", "ghost", "live"):
        print(f"  {name:<8}: {len(watchlists.get(name) or [])} entries")
    print()
    print("  NOTE: the discovery bridge writes this list WHILE production runs.")
    print("        Every count below is a snapshot, not a stable population.")
    print()

    for name in ("stream", "ghost", "live"):
        entries = list(watchlists.get(name) or [])
        if not entries:
            continue
        tape = _tape(cur, entries)

        malformed: List[Tuple[str, str, int]] = []
        never: List[Tuple[str, int]] = []
        dropped: List[Tuple[str, int, float | None, float]] = []
        ticking: List[Tuple[str, int, float]] = []

        for sym in entries:
            n, first, last = tape[sym]
            reason = _malformed_reason(sym)
            if reason:
                malformed.append((sym, reason, n))
            if n == 0:
                never.append((sym, n))
            elif last is not None and (now - last) >= dark_cut:
                dropped.append((sym, n, first, last))
            else:
                ticking.append((sym, n, last or 0.0))

        print("-" * 78)
        print(f"{name.upper()}  --  {len(entries)} entries")
        print("-" * 78)
        print(f"  NEVER TICKED (0 rows all time) : {len(never)}")
        print(f"  DROPPED (had a tape, lost it)  : {len(dropped)}")
        print(f"  TICKING (< {args.dark_hours:g}h)             : {len(ticking)}")
        print(f"  MALFORMED (cannot resolve)     : {len(malformed)}")

        if malformed:
            silent = sum(1 for _, _, n in malformed if n == 0)
            print()
            print(f"  MALFORMED -- {silent} of {len(malformed)} have ticked zero times all-time.")
            print("  Write site: services/discovery/stream_bridge.py:144 (unguarded f-string).")
            print("  The guard already exists at services/discovery/coordinator.py:92-97.")
            for sym, reason, n in sorted(malformed):
                print(f"    {sym:<26} n={n:<8} {reason}")

        if never:
            print()
            print("  NEVER TICKED -- never resolved to a pair, so no feed was ever requested:")
            bad = {s for s, _, _ in malformed}
            for sym, _ in sorted(never):
                tag = "  <- malformed" if sym in bad else ""
                print(f"    {sym}{tag}")

        if dropped:
            print()
            print("  DROPPED -- these DID have a tape. A different fault from NEVER TICKED:")
            for sym, n, first, last in sorted(dropped, key=lambda r: -(r[3] or 0)):
                print(f"    {sym:<26} n={n:<8} first={_fmt_ts(first)}  LAST={_fmt_ts(last)}  "
                      f"dark {(now - (last or now)) / 86400:.1f}d")

        if ticking:
            print()
            print("  TICKING:")
            for sym, n, last in sorted(ticking, key=lambda r: -r[1]):
                print(f"    {sym:<26} n={n:<8} last={_fmt_ts(last)}  age {(now - last) / 3600:.1f}h")
        print()

    # The expensive direction: a live symbol nobody subscribes to is forgone
    # evidence, where a dead entry is only dead weight.
    since = now - args.window_hours * 3600.0
    live_tape = _ticking_symbols(cur, since, args.min_ticks)
    on_stream = {s.upper() for s in (watchlists.get("stream") or [])}
    absent = [(s, n, last) for s, n, last in live_tape if s.upper() not in on_stream]

    print("=" * 78)
    print(f"ACTIVELY TICKING BUT ABSENT FROM watchlists.stream "
          f"({args.window_hours:g}h window, >= {args.min_ticks} ticks)")
    print("=" * 78)
    print(f"  ticking symbols in window : {len(live_tape)}")
    print(f"  of those, absent from stream: {len(absent)}")
    if absent:
        print()
        for sym, n, last in absent[: args.show_absent]:
            print(f"    {sym:<26} n={n:<8} last={_fmt_ts(last)}  age {(now - last) / 3600:.1f}h")
        if len(absent) > args.show_absent:
            print(f"    ... and {len(absent) - args.show_absent} more")
    print()
    print("CONSUMERS -- who actually reads each list (grepped, p698):")
    print("  stream : trading/selector.py:923 (UNIONED with live), "
          "services/atf_static_strategy.py:1838")
    print("  ghost  : trading/pipeline.py:7058")
    print("  live   : trading/selector.py:923 ONLY, and only unioned into pair")
    print("           selection. NOTHING reads 'live' alone -- it does NOT gate the")
    print("           live lane and it is NOT a live-approved universe.")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
