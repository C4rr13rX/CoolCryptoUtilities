#!/usr/bin/env python3
"""Which symbol the live lane CAN trade actually pays for its round trip?

The wall as of 2026-09-10 is EVIDENCE-AND-QUALITY on the live-tradeable book:
over the whole 38-strategy ledger, 410 pooled ghost round trips reduce to 23
the live lane could have placed, and those 23 win 17% and net -0.8940 against a
bar of 20 trades / 55% / P-L above 0. The pooled book's +6.8134 is earned on
symbols the live lane refuses. Feeding the funnel harder does not graduate a
17% book, so the question stops being "how do we get more evidence" and becomes
"is there ANY symbol we can actually spend on where the edge survives the fee".

This answers that from ``trade_outcomes`` -- the money table, not the
append-only ``trading_ops`` log, which keeps pre-fix artifacts forever and has
produced a -0.25 where the books said +0.14.

Two things it is careful about, because both have shipped as bugs here:

  * ``fee_cost`` in the book is NOT trustworthy as a per-row constant. 105 of
    196 rows once carried EXACTLY 0.650000% with zero variance -- the default
    written back into the book, measuring itself. So net-of-cost is recomputed
    from ``services.round_trip_cost`` as well as read from the row, and BOTH
    are printed. Where they disagree, the recomputed one is the honest number.
  * ``annulled`` rows are excluded. They are bookkeeping reversals, not fills.

Run:  python -X utf8 scripts/tradeable_symbol_edge.py
      python -X utf8 scripts/tradeable_symbol_edge.py --json
      python -X utf8 scripts/tradeable_symbol_edge.py --min-trades 3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB = ROOT / "storage" / "trading_cache.db"


def _measured_cost_fraction() -> float:
    """The round-trip cost as a FRACTION of notional, from receipts.

    Falls back to the documented constant only if the measurement module is
    unavailable -- and says so, rather than silently reading a literal that
    once measured its own default.
    """
    try:
        from services.round_trip_cost import round_trip_cost  # type: ignore

        return float(round_trip_cost())
    except Exception:  # noqa: BLE001
        try:
            from services.round_trip_cost import fallback_cost  # type: ignore

            return float(fallback_cost())
        except Exception:  # noqa: BLE001
            return 0.0065


def collect(min_trades: int = 1) -> dict:
    from trading.strategies.ledger import _live_tradeable

    cost_frac = _measured_cost_fraction()

    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    rows = list(
        con.execute(
            "SELECT symbol, gross_profit, net_profit, fee_cost, entry_price, "
            "quantity FROM trade_outcomes WHERE status = 'closed'"
        )
    )
    con.close()

    per: dict = {}
    for r in rows:
        sym = str(r["symbol"] or "").strip()
        if not sym:
            continue
        d = per.setdefault(
            sym,
            {
                "symbol": sym,
                "tradeable": bool(_live_tradeable(sym)),
                "trades": 0,
                "wins_net": 0,
                "gross": 0.0,
                "net_booked": 0.0,
                "fee_booked": 0.0,
                "notional": 0.0,
            },
        )
        gross = float(r["gross_profit"] or 0.0)
        net = float(r["net_profit"] or 0.0)
        fee = float(r["fee_cost"] or 0.0)
        notional = abs(float(r["entry_price"] or 0.0) * float(r["quantity"] or 0.0))

        d["trades"] += 1
        d["gross"] += gross
        d["net_booked"] += net
        d["fee_booked"] += fee
        d["notional"] += notional
        if net > 0:
            d["wins_net"] += 1

    out = []
    for d in per.values():
        n = d["trades"]
        # Net recomputed against the MEASURED cost rather than the booked fee.
        d["net_recomputed"] = d["gross"] - cost_frac * d["notional"]
        d["win_rate_net"] = d["wins_net"] / n if n else 0.0
        d["net_per_trade"] = d["net_booked"] / n if n else 0.0
        d["recomputed_per_trade"] = d["net_recomputed"] / n if n else 0.0
        if n >= min_trades:
            out.append(d)

    out.sort(key=lambda d: (-d["recomputed_per_trade"], -d["trades"]))

    trad = [d for d in out if d["tradeable"]]
    refu = [d for d in out if not d["tradeable"]]

    def _tot(group):
        n = sum(d["trades"] for d in group)
        return {
            "symbols": len(group),
            "trades": n,
            "wins": sum(d["wins_net"] for d in group),
            "win_rate": (sum(d["wins_net"] for d in group) / n) if n else 0.0,
            "gross": sum(d["gross"] for d in group),
            "net_booked": sum(d["net_booked"] for d in group),
            "net_recomputed": sum(d["net_recomputed"] for d in group),
        }

    return {
        "cost_fraction": cost_frac,
        "min_trades": min_trades,
        "symbols": out,
        "totals": {"tradeable": _tot(trad), "refused": _tot(refu)},
    }


def render(rep: dict) -> str:
    a = []
    A = a.append
    A("=" * 78)
    A("  WHICH TRADEABLE SYMBOL PAYS FOR ITS ROUND TRIP")
    A("=" * 78)
    A("")
    A("  measured round-trip cost: %.4f%% of notional" % (rep["cost_fraction"] * 100))
    A("  'net' below is BOOKED net_profit; 'recomp' re-charges the measured")
    A("  cost against gross, because the booked fee has previously been a")
    A("  constant writing itself back into the book.")
    A("")
    for label, key in (("LIVE-TRADEABLE", True), ("REFUSED BY THE LIVE LANE", False)):
        A("  %s" % label)
        A("  %-24s%6s%7s%11s%11s%11s" % ("symbol", "n", "win%", "gross", "net", "recomp"))
        A("  " + "-" * 70)
        group = [d for d in rep["symbols"] if d["tradeable"] is key]
        if not group:
            A("    (none)")
        for d in group:
            A("  %-24s%6d%6.0f%%%+11.4f%+11.4f%+11.4f" % (
                d["symbol"][:24], d["trades"], d["win_rate_net"] * 100,
                d["gross"], d["net_booked"], d["net_recomputed"],
            ))
        t = rep["totals"]["tradeable" if key else "refused"]
        A("  %-24s%6d%6.0f%%%+11.4f%+11.4f%+11.4f   <- TOTAL" % (
            "", t["trades"], t["win_rate"] * 100,
            t["gross"], t["net_booked"], t["net_recomputed"],
        ))
        A("")

    tt = rep["totals"]["tradeable"]
    rt = rep["totals"]["refused"]
    A("  VERDICT")
    A("    tradeable : %d trades, %.0f%% win, booked %+.4f, recomputed %+.4f"
      % (tt["trades"], tt["win_rate"] * 100, tt["net_booked"], tt["net_recomputed"]))
    A("    refused   : %d trades, %.0f%% win, booked %+.4f, recomputed %+.4f"
      % (rt["trades"], rt["win_rate"] * 100, rt["net_booked"], rt["net_recomputed"]))
    winners = [d for d in rep["symbols"]
               if d["tradeable"] and d["net_recomputed"] > 0 and d["trades"] >= 3]
    if winners:
        A("    symbols we CAN spend on that pay after the measured cost, n>=3:")
        for d in winners:
            A("      %-22s %d trades, %.0f%% win, %+.4f recomputed (%+.4f/trade)"
              % (d["symbol"], d["trades"], d["win_rate_net"] * 100,
                 d["net_recomputed"], d["recomputed_per_trade"]))
    else:
        A("    NO live-tradeable symbol with 3+ closed round trips is net")
        A("    positive after the measured cost. Building a strategy for a")
        A("    specific symbol is not available from this book; the cost, the")
        A("    barrier width or the entry has to change first.")
    A("")
    return "\n".join(a)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true")
    p.add_argument("--min-trades", type=int, default=1)
    args = p.parse_args()
    rep = collect(min_trades=args.min_trades)
    print(json.dumps(rep, indent=2) if args.json else render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
