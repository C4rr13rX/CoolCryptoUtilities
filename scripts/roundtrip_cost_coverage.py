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

THE SECOND TRAP, FIXED 2026-09-22 ([a9bbe626]). ``trade_outcomes`` is a
PRE-GUARD source: ``trading/strategies/ledger.py::record`` refuses an outcome
whose shape says it did not happen, but the table keeps the refused row anyway.
Until this pass every number above was read straight off that table, so this
script measured a DIRTIER book than graduation does. It now partitions through
``services.outcome_plausibility.partition`` and prints BOTH columns with the
gap between them, because the gap IS the finding: hiding the uncensored figure
would remove the only evidence that the contamination exists.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.outcome_plausibility import (  # noqa: E402
    booked_return, partition, summarise,
)


def _strategy_id(row: dict) -> str:
    """The row's strategy, dug out of the JSON ``details`` blob.

    The dollar arm of the plausibility test judges a row against ITS OWN
    strategy's scale, and trade_outcomes carries no strategy_id column -- it is
    inside ``details``. Without this the dollar arm never runs and only the
    ratio arm censors, which is a quieter filter than the ledger's.
    """
    import json
    blob = row.get("details")
    if not blob:
        return ""
    try:
        d = json.loads(blob) if isinstance(blob, str) else blob
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(d, dict):
        return ""
    return str(d.get("strategy_id") or d.get("strategy") or "").strip()

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
    conn.row_factory = sqlite3.Row
    all_rows = [dict(r) for r in conn.execute(
        "select symbol, entry_price, exit_price, quantity, gross_profit, "
        "fee_cost, net_profit, status, details, ts from trade_outcomes "
        "where symbol is not null")]
    conn.close()
    if not all_rows:
        print("coverage census: trade_outcomes is empty at %s" % path)
        return 2

    for r in all_rows:
        r["strategy_id"] = _strategy_id(r)

    keep, dropped = partition(all_rows)
    summary = summarise(all_rows)

    def census(rows: list[dict]) -> dict:
        per: dict[str, list[float]] = {}
        for r in rows:
            base = str(r.get("symbol") or "").split("-")[0].upper()
            per.setdefault(base, []).append(float(r.get("net_profit") or 0.0))
        meas_n = meas_net = 0.0
        unmeas_n = unmeas_net = 0.0
        unpriced: list[tuple[int, str, float]] = []
        for base, nets in per.items():
            n, net = len(nets), sum(nets)
            if base in MEASURED:
                meas_n += n
                meas_net += net
            else:
                unmeas_n += n
                unmeas_net += net
                unpriced.append((n, base, net))
        return {
            "symbols": len(per), "total_n": len(rows),
            "total_net": sum(float(r.get("net_profit") or 0.0) for r in rows),
            "meas_n": meas_n, "meas_net": meas_net,
            "unmeas_n": unmeas_n, "unmeas_net": unmeas_net,
            "unpriced": unpriced,
        }

    raw = census(all_rows)
    cen = census(keep)

    print("=" * 74)
    print("ROUND-TRIP COST COVERAGE OF THE GHOST BOOK")
    print("=" * 74)
    print("store: %s" % path)
    print()
    print("CENSORING -- services.outcome_plausibility.partition, the ledger's own")
    print("refusal rule applied on the READ side. trade_outcomes is PRE-GUARD: it")
    print("keeps every row ledger.record() refused, so an uncensored census here")
    print("measures a dirtier book than graduation reads.")
    print("  rows read %d   countable %d   REFUSED %d   reasons %s"
          % (len(all_rows), summary["kept"], summary["dropped"],
             summary["reasons"] or "{}"))
    print("  the refused rows carry gross %+.4f  net %+.4f"
          % (summary["gross_dropped"], summary["net_dropped"]))
    print()

    hdr = "%-38s %10s %10s %10s" % ("", "UNCENSORED", "CENSORED", "GAP")
    print(hdr)
    print("-" * len(hdr))

    def line(label: str, a, b, fmt: str = "%10.4f") -> None:
        print(("%-38s " + fmt + " " + fmt + " " + fmt)
              % (label, a, b, b - a))

    line("rows", raw["total_n"], cen["total_n"], "%10.0f")
    line("distinct symbols", raw["symbols"], cen["symbols"], "%10.0f")
    line("book net", raw["total_net"], cen["total_net"])
    line("MEASURED-cost rows", raw["meas_n"], cen["meas_n"], "%10.0f")
    line("MEASURED-cost net", raw["meas_net"], cen["meas_net"])
    line("ASSUMED-default rows", raw["unmeas_n"], cen["unmeas_n"], "%10.0f")
    line("ASSUMED-default net", raw["unmeas_net"], cen["unmeas_net"])
    print()

    for label, c in (("UNCENSORED", raw), ("CENSORED", cen)):
        tn = c["total_n"] or 1
        share = (100.0 * abs(c["unmeas_net"])
                 / ((abs(c["meas_net"]) + abs(c["unmeas_net"])) or 1.0))
        print("  %-10s  coverage %5.1f%% of rows measured; %5.1f%% of |net| "
              "sits on an assumed cost; sign of book net %s"
              % (label, 100.0 * c["meas_n"] / tn, share,
                 "POSITIVE" if c["total_net"] > 0 else "NEGATIVE"))
    if (raw["total_net"] > 0) != (cen["total_net"] > 0):
        print("  !! THE SIGN OF THE BOOK FLIPS UNDER CENSORING. Every conclusion")
        print("  !! drawn from the uncensored column is drawing it from the")
        print("  !! refused rows.")
    print()

    print("UNPRICED SYMBOLS BY ROWS CARRIED (CENSORED book) -- these set the")
    print("book's sign and no on-chain quote has ever been taken for any of them:")
    print()
    raw_by_sym = {b: (n, net) for n, b, net in raw["unpriced"]}
    print("    %-16s %6s   %10s   %10s" % ("symbol", "rows", "net", "uncens.net"))
    for n, base, net in sorted(cen["unpriced"], reverse=True)[: args.top]:
        rn = raw_by_sym.get(base, (n, net))[1]
        flag = "  <- censoring changes this symbol" if abs(rn - net) > 1e-9 else ""
        print("    %-16s %6d   %+10.4f   %+10.4f%s" % (base, n, net, rn, flag))
    print()
    print("A re-pricing run today charges the default to %d of %d CENSORED rows "
          "(%.1f%%)." % (cen["unmeas_n"], cen["total_n"],
                         100.0 * cen["unmeas_n"] / (cen["total_n"] or 1)))
    print("Those rows are not re-priced. They are restated at a constant.")
    if dropped:
        print()
        print("THE %d REFUSED ROWS, largest |gross| first -- printed, never hidden:"
              % len(dropped))
        for r in sorted(dropped, key=lambda r: -abs(float(r.get("gross_profit") or 0.0)))[:10]:
            ret = booked_return(r)
            print("    %-14s gross %+9.4f  net %+9.4f  booked return %s"
                  % (str(r.get("symbol")), float(r.get("gross_profit") or 0.0),
                     float(r.get("net_profit") or 0.0),
                     "%+.2f%%" % (100.0 * ret) if ret is not None else "(unpriced leg)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
