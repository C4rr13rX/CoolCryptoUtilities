#!/usr/bin/env python3
"""
How much of the ghost book has a MEASURED round-trip cost, and how much is a
guess wearing a measurement's clothes?

WHY THIS EXISTS. Every per-symbol re-pricing in this repo charges a handful of
symbols their measured pool cost and silently falls everything else through to
a DEFAULT round-trip rate. That default is not a measurement of anything: it is
a constant someone chose. When two thirds of the rows take the default, a
"re-priced book" is mostly a restatement of that constant, and this repo has
already paid for exactly that mistake once -- a round trip that measured its
own default.

So before anyone re-prices the book per symbol, this answers the prior
question: WHICH ROWS CAN HONESTLY BE RE-PRICED AT ALL. It splits the ghost book
into the symbols with a measured cost and the symbols without, reports rows and
net for each side, and ranks the unpriced symbols by how much book they carry.

WHAT THIS IS NOT. It prices nothing and quotes nothing. It is a coverage audit
of someone else's cost table. The measured set below is the set this repo has
actually quoted on chain (Gale's routing census, 879ea6e, plus the symbols
carried in scripts/money_button_dip_expectancy.py); if you measure more
symbols, add them here or this will understate coverage.

THE TRAP IT IS BUILT AGAINST. A coverage number computed over DISTINCT SYMBOLS
flatters itself -- 5 of 49 symbols sounds like a rounding error, but those 5
carry a third of the rows. Coverage is reported by ROWS and by NET, because
rows are what a re-pricing charges and net is what a conclusion rests on.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Symbols this repo has actually quoted on chain, USDC in and back.
#: Sources: scripts/routing_arbitrage_census.py output recorded in 879ea6e
#: (WETH, CBBTC, SOL, AERO, KTA) and the TOKEN_ROUND_TRIP table in
#: scripts/money_button_dip_expectancy.py (adds ETH, CBETH, USDC).
#: Everything NOT in this set is charged an assumed default wherever it is
#: re-priced, which is the whole point of this script.
MEASURED = {"WETH", "ETH", "CBETH", "USDC", "AERO", "KTA", "CBBTC", "SOL"}


def _db_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    return os.environ.get("TRADING_DB_PATH", str(ROOT / "storage" / "trading_cache.db"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", default=None, help="trading store (default TRADING_DB_PATH)")
    ap.add_argument("--top", type=int, default=15,
                    help="how many unpriced symbols to list")
    args = ap.parse_args()

    path = _db_path(args.db)
    if not Path(path).exists():
        print("coverage census: no trading store at %s" % path)
        return 2

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "select symbol, count(*), sum(net_profit) from trade_outcomes "
        "where symbol is not null group by symbol"
    )
    rows = cur.fetchall()
    if not rows:
        print("coverage census: trade_outcomes is empty at %s" % path)
        return 2

    total_n = sum(r[1] for r in rows)
    total_net = sum((r[2] or 0.0) for r in rows)

    meas_n = meas_net = 0.0
    unmeas_n = unmeas_net = 0.0
    unpriced: list[tuple[int, str, float]] = []
    for symbol, n, net in rows:
        base = str(symbol).split("-")[0].upper()
        net = net or 0.0
        if base in MEASURED:
            meas_n += n
            meas_net += net
        else:
            unmeas_n += n
            unmeas_net += net
            unpriced.append((n, base, net))

    pct = lambda x: 100.0 * x / total_n if total_n else 0.0  # noqa: E731

    print("=" * 74)
    print("ROUND-TRIP COST COVERAGE OF THE GHOST BOOK")
    print("=" * 74)
    print("store: %s" % path)
    print("ghost book: %d rows across %d distinct symbols, net %+.4f"
          % (total_n, len(rows), total_net))
    print()
    print("  MEASURED cost symbols : %4d rows (%5.1f%%)   net %+.4f"
          % (meas_n, pct(meas_n), meas_net))
    print("  ASSUMED default cost  : %4d rows (%5.1f%%)   net %+.4f"
          % (unmeas_n, pct(unmeas_n), unmeas_net))
    print()
    print("  distinct symbols measured: %d of %d"
          % (len(rows) - len(unpriced), len(rows)))
    if total_net:
        print("  share of book NET sitting on an assumed cost: %.1f%%"
              % (100.0 * abs(unmeas_net) / (abs(meas_net) + abs(unmeas_net))))
    print()
    print("UNPRICED SYMBOLS BY ROWS CARRIED -- these set the book's sign and")
    print("no on-chain quote has ever been taken for any of them:")
    print()
    print("    %-16s %6s   %10s" % ("symbol", "rows", "net"))
    for n, base, net in sorted(unpriced, reverse=True)[: args.top]:
        print("    %-16s %6d   %+10.4f" % (base, n, net))
    print()
    print("A re-pricing run today charges the default to %d of %d rows (%.1f%%)."
          % (unmeas_n, total_n, pct(unmeas_n)))
    print("Those rows are not re-priced. They are restated at a constant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
