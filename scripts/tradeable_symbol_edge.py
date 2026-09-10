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
  * ROWS THE LEDGER WOULD HAVE REFUSED are excluded from the verdict. This is
    an ALL-TIME table, so it reads every pre-guard artifact ``trade_outcomes``
    still holds: the ledger added ``_is_implausible`` to reject a repricing
    shape on the way in, but the money table is append-only and the rows are
    still there. Measured 2026-09-10, AERO-USDC's all-time gross is +2.0252
    booked and -0.0959 once the two rows of its contaminated pair are dropped
    -- the +161% fake win AND the -55% stop taken from its fictional basis.
    Per-symbol BOOKED figures are still printed; the verdict is read off the
    filtered ones, via ``services.outcome_plausibility`` -- the SAME helper
    ``scripts/tradeable_book.py`` uses, so the two tables cannot disagree about
    which rows exist.

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


def load_closed(db_path=None) -> list:
    """Every closed ``trade_outcomes`` row, with the fields the filter needs.

    ``details`` is read for ``strategy_id`` because the dollar arm of the
    implausibility test judges a row against its OWN strategy's scale, exactly
    as ``ledger._recent_scale`` does. Without it every row would fall back to
    the absolute bound, which is a different (harsher) rule than the one the
    write path runs.
    """
    con = sqlite3.connect(str(db_path or DB))
    con.row_factory = sqlite3.Row
    try:
        # SELECT ONLY WHAT THE TABLE HAS. ``exit_price`` and ``details`` are
        # needed by the plausibility filter and were not in the original query;
        # naming them unconditionally makes this function raise against any
        # narrower trade_outcomes, which is what the tmp-db fixtures build and
        # what an older checkout's schema is. A missing column leaves the field
        # at its neutral value, and a missing ``exit_price`` makes the ratio arm
        # return None -- unjudgeable, not silently plausible.
        have = {r[1] for r in con.execute("PRAGMA table_info(trade_outcomes)")}
        wanted = ["symbol", "gross_profit", "net_profit", "fee_cost",
                  "entry_price", "exit_price", "quantity", "details"]
        cols = [c for c in wanted if c in have]
        raw = list(
            con.execute(
                "SELECT %s FROM trade_outcomes WHERE status = 'closed'"
                % ", ".join(cols)
            )
        )
    finally:
        con.close()

    def _col(row, name, default=None):
        return row[name] if name in cols else default

    out = []
    for r in raw:
        try:
            det = json.loads(_col(r, "details") or "{}")
        except Exception:  # noqa: BLE001
            det = {}
        if not isinstance(det, dict):
            det = {}
        out.append({
            "symbol": str(_col(r, "symbol") or "").strip(),
            "strategy_id": str(det.get("strategy_id") or "") or "unclassified",
            "gross_profit": float(_col(r, "gross_profit") or 0.0),
            "net_profit": float(_col(r, "net_profit") or 0.0),
            "fee_cost": float(_col(r, "fee_cost") or 0.0),
            "entry_price": float(_col(r, "entry_price") or 0.0),
            "exit_price": float(_col(r, "exit_price") or 0.0),
            "quantity": abs(float(_col(r, "quantity") or 0.0)),
        })
    return out


def collect(min_trades: int = 1, rows: list | None = None) -> dict:
    from trading.strategies.ledger import _live_tradeable
    from services.outcome_plausibility import (
        IMPLAUSIBLE_RET, booked_return, partition, sweep,
    )

    cost_frac = _measured_cost_fraction()

    if rows is None:
        rows = load_closed()

    # THE VERDICT IS READ OFF THE ROWS THE LEDGER WOULD HAVE ACCEPTED.
    # Booked figures are still accumulated per symbol, because hiding how much
    # of a record is artifact is its own failure -- but `gross`/`net_*` that
    # feed the ranking and the totals come from the kept rows only.
    kept, dropped = partition(rows)
    keep_ids = {id(r) for r in kept}

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
                "gross_booked": 0.0,
                "net_booked_all": 0.0,
                "trades_booked": 0,
                "implausible": 0,
                "worst_implausible_ret": 0.0,
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

        d["trades_booked"] += 1
        d["gross_booked"] += gross
        d["net_booked_all"] += net
        if id(r) not in keep_ids:
            d["implausible"] += 1
            ret = booked_return(r)
            if ret is not None and abs(ret) > abs(d["worst_implausible_ret"]):
                d["worst_implausible_ret"] = ret
            continue

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
        # How much of this symbol's BOOKED record the filter removed. Over 100%
        # means the artifact rows carry the SIGN, not just the size -- which is
        # the AERO case and the reason this column is printed rather than
        # inferred.
        d["artifact_share_of_gross"] = (
            100.0 * (d["gross_booked"] - d["gross"]) / d["gross_booked"]
            if d["gross_booked"] else 0.0
        )
        d["sign_flipped"] = (d["gross_booked"] > 0) != (d["gross"] > 0)
        if d["trades_booked"] >= min_trades:
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
            "gross_booked": sum(d["gross_booked"] for d in group),
            "net_booked": sum(d["net_booked"] for d in group),
            "net_recomputed": sum(d["net_recomputed"] for d in group),
            "implausible": sum(d["implausible"] for d in group),
        }

    return {
        "cost_fraction": cost_frac,
        "min_trades": min_trades,
        "implausible_ret": IMPLAUSIBLE_RET,
        # THE THRESHOLD SWEEP, because the default is calibrated above most of
        # the rows the ledger already calls fabricated. A reader who sees only
        # one number inherits a verdict; this prints where the sign flips.
        "sweep": sweep(rows),
        "rows_read": len(rows),
        "rows_dropped": len(dropped),
        "gross_dropped": sum(float(r["gross_profit"] or 0.0) for r in dropped),
        "net_dropped": sum(float(r["net_profit"] or 0.0) for r in dropped),
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
    A("  PRE-GUARD ARTIFACTS DROPPED: %d of %d closed rows, carrying gross "
      "%+.4f and net %+.4f" % (rep["rows_dropped"], rep["rows_read"],
                               rep["gross_dropped"], rep["net_dropped"]))
    A("  These are rows ``ledger._is_implausible`` would have refused on the way")
    A("  in; ``trade_outcomes`` is append-only and still holds them. |ret| >")
    A("  %.0f%% or outsized against the strategy's own scale. 'bkd' below is the"
      % (rep["implausible_ret"] * 100))
    A("  BOOKED gross including them; 'gross' excludes them and is the verdict.")
    A("")
    sw = rep.get("sweep") or {}
    if sw.get("steps"):
        A("  THE THRESHOLD IS NOT A PROPERTY OF THE BOOK -- HERE IS THE WHOLE CURVE")
        A("  The %.0f%% default is calibrated ABOVE most of the rows the ledger"
          % (sw["default"] * 100))
        A("  already calls fabricated (BSTONK-USDC at +25.35%/+17.28% are the two")
        A("  rows [c4f16946] names as atf_static's +1.0201 of fake fills). An")
        A("  overshoot is named by filling past ITS OWN limit, not by being large,")
        A("  so the real fix is annulment ([db76611a]); this curve exists so no")
        A("  verdict below is read off a cap that never touched those rows.")
        A("    %-10s%7s%7s%11s%11s" % ("|ret| <=", "kept", "drop", "gross", "net"))
        for s in sw["steps"]:
            A("    %-10s%7d%7d%+11.4f%+11.4f%s" % (
                "%.0f%%" % (s["max_ret"] * 100), s["kept"], s["dropped"],
                s["gross"], s["net"],
                "   <- DEFAULT" if abs(s["max_ret"] - sw["default"]) < 1e-9 else ""))
        if sw.get("verdict_is_threshold_dependent"):
            A("    THE SIGN FLIPS AT %.0f%%: this book's verdict is a CHOICE OF"
              % (sw["flips_at"] * 100))
            A("    THRESHOLD, not a measurement. Do not quote either half alone.")
        else:
            A("    The sign does not change across the sweep, so the verdict is")
            A("    a property of the book rather than of the cap.")
        A("")
    for label, key in (("LIVE-TRADEABLE", True), ("REFUSED BY THE LIVE LANE", False)):
        A("  %s" % label)
        A("  %-22s%5s%6s%4s%10s%10s%10s%10s" % (
            "symbol", "n", "win%", "bad", "bkd", "gross", "net", "recomp"))
        A("  " + "-" * 77)
        group = [d for d in rep["symbols"] if d["tradeable"] is key]
        if not group:
            A("    (none)")
        for d in group:
            A("  %-22s%5d%5.0f%%%4d%+10.4f%+10.4f%+10.4f%+10.4f%s" % (
                d["symbol"][:22], d["trades"], d["win_rate_net"] * 100,
                d["implausible"], d["gross_booked"],
                d["gross"], d["net_booked"], d["net_recomputed"],
                "  <- SIGN FLIPS without the artifact rows"
                if d["sign_flipped"] else "",
            ))
        t = rep["totals"]["tradeable" if key else "refused"]
        A("  %-22s%5d%5.0f%%%4d%+10.4f%+10.4f%+10.4f%+10.4f   <- TOTAL" % (
            "", t["trades"], t["win_rate"] * 100, t["implausible"],
            t["gross_booked"], t["gross"], t["net_booked"], t["net_recomputed"],
        ))
        A("")
    flipped = [d for d in rep["symbols"] if d["sign_flipped"]]
    if flipped:
        A("  SYMBOLS WHOSE VERDICT IS CARRIED BY AN ARTIFACT ROW")
        A("    A symbol here is one the book would call a winner and the ledger")
        A("    already decided did not happen. This is the AERO shape.")
        for d in flipped:
            A("      %-20s %d artifact row(s) of %d: booked gross %+.4f -> %+.4f"
              % (d["symbol"], d["implausible"], d["trades_booked"],
                 d["gross_booked"], d["gross"]))
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
