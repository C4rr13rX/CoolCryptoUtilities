"""Re-price the existing ghost book charging every symbol its OWN measured
round-trip cost, instead of the one flat rate the book was built with.

WHY THIS EXISTS
---------------
``services/roundtrip_cost.py`` publishes ONE cost curve that every ghost P/L in
this repo was computed from:

    cost_usd = 0.004047 + 0.003187 * notional

It takes no symbol. WETH and KTA are charged the same rate across a ~700x
spread in real pool depth. Gale's routing census (879ea6e, 2026-09-11) quoted
read-only round trips on base -- USDC in, token out, token back to USDC -- and
found the spread is not a venue fee, it is pool DEPTH:

    WETH   0.1074% / 0.0369% / 0.0553%   at $2 / $10 / $50
    CBBTC  0.0728% / 0.0617% / 1.4659%
    SOL    0.4386% / 0.4390% / 2.0217%
    AERO   0.6395% / 4.3502% / 0.1330%
    KTA   13.5379% / 2.0274% / 74.9849%

THE QUESTION THIS ANSWERS. The tradeable ghost book is negative. Is that
because the strategies call DIRECTION badly, or because they are called on
symbols whose round trip costs more than any edge this repo has ever measured?
Those are different findings and they have different fixes. Re-pricing the
existing book answers it without placing a single trade.

WHAT IT DOES NOT DO
-------------------
It bans nothing and changes no gate. It is a measurement.

THREE CAVEATS THAT MUST TRAVEL WITH EVERY NUMBER BELOW, and they are printed in
the report itself, not just here:

  1. THE COSTS ARE QUOTES, NOT FILLS. A read-only quote is what the router says
     it would give you. Slippage against a moving book and MEV are NOT modelled.
     A re-priced book is therefore a COST FLOOR -- the best case -- and not a
     realisable P/L.

  2. ONLY SIX SYMBOLS HAVE A MEASURED COST. Every other symbol in the book is
     reported in a separate UNMEASURED bucket and is never silently charged a
     default. A default rate applied to an unquoted pool is an assumption
     wearing a measurement's clothes, and this repo has shipped that bug before
     (round_trip_cost measured its own default, 2026-09-04).

  3. THE GHOST BOOK RECORDS NO REALISED SLIPPAGE AT ALL. From
     data/roundtrip_cost_census.json: of 183 ghost round trips,
     fully_measured_round_trips = 0, buy_slippage_bps_median = 0.0 and
     sell_slippage_bps_median = -0.0. The fee_cost column on a ghost row is
     entirely the synthetic model above. So "before" is not an observation
     either -- both columns are models, and this script compares two models.
     The LIVE lane, by contrast, measured 21.73 bps on the buy leg and 6.04 bps
     on the sell leg (n=12), i.e. ~27.8 bps of real slippage that NO ghost row
     carries.

Run:  python -X utf8 scripts/deep_pool_reprice.py
      python -X utf8 scripts/deep_pool_reprice.py --json out.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Gale's routing census, 879ea6e, 2026-09-11. Read-only quotes on base:
# USDC -> token -> USDC, cost as a POSITIVE percentage of notional, by clip size.
# Keys are the base symbol of a <BASE>-USDC pair. Nothing else is measured.
MEASURED_ROUND_TRIP_PCT: Dict[str, List[Tuple[float, float]]] = {
    "WETH": [(2.0, 0.1074), (10.0, 0.0369), (50.0, 0.0553)],
    "ETH": [(2.0, 0.1074), (10.0, 0.0369), (50.0, 0.0553)],
    "CBBTC": [(2.0, 0.0728), (10.0, 0.0617), (50.0, 1.4659)],
    "SOL": [(2.0, 0.4386), (10.0, 0.4390), (50.0, 2.0217)],
    "AERO": [(2.0, 0.6395), (10.0, 4.3502), (50.0, 0.1330)],
    "KTA": [(2.0, 13.5379), (10.0, 2.0274), (50.0, 74.9849)],
}

# The depth bar. A symbol is DEEP if its round trip at the size the book
# actually trades stays under this. It is stated as a parameter, not hidden:
# 0.50% is the two-leg cost the flat model already assumes at a $0.75 clip
# (0.004047 + 0.003187*0.75 = $0.00644 on $0.75 = 0.859%), rounded DOWN so the
# bar is stricter than the model the book was built with.
DEFAULT_DEPTH_BAR_PCT = 0.50


def _base_symbol(sym: str) -> str:
    s = (sym or "").upper().strip()
    return s.split("-", 1)[0] if "-" in s else s


def round_trip_pct(symbol: str, notional_usd: float) -> Optional[float]:
    """Measured round-trip cost for this symbol at the nearest quoted clip.

    Returns None when the symbol was never quoted. It NEVER falls back to a
    default -- that is the whole discipline of this script.
    """
    table = MEASURED_ROUND_TRIP_PCT.get(_base_symbol(symbol))
    if not table:
        return None
    n = abs(float(notional_usd or 0.0))
    if n <= 0:
        n = 2.0
    return min(table, key=lambda kv: abs(kv[0] - n))[1]


def _move_pct(row: Dict[str, Any]) -> Optional[float]:
    """Realised move of the round trip, in percent, from the row's own prices."""
    try:
        ep = float(row.get("entry_price") or 0.0)
        xp = float(row.get("exit_price") or 0.0)
    except (TypeError, ValueError):
        return None
    if ep <= 0 or xp <= 0:
        return None
    return (xp / ep - 1.0) * 100.0


def _notional(row: Dict[str, Any]) -> Optional[float]:
    """Notional of the round trip, in USD, from the row's own fields."""
    try:
        qty = float(row.get("quantity") or 0.0)
        px = float(row.get("entry_price") or 0.0)
    except (TypeError, ValueError):
        return None
    n = abs(qty * px)
    return n if n > 0 else None


def load_rows(db_path: str) -> List[Dict[str, Any]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            "SELECT symbol, quantity, entry_price, exit_price, gross_profit, "
            "fee_cost, net_profit, status, ts FROM trade_outcomes"
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def reprice(rows: List[Dict[str, Any]], depth_bar: float,
            max_move_pct: Optional[float] = None) -> Dict[str, Any]:
    per = collections.defaultdict(
        lambda: {
            "n": 0,
            "wins_before": 0,
            "wins_after": 0,
            "gross": 0.0,
            "fee_charged": 0.0,
            "net_before": 0.0,
            "net_after": 0.0,
            "fee_measured": 0.0,
            "notional": 0.0,
            "n_notional": 0,
            "rate_pct": None,
            "measured": False,
            "gross_rows": [],
            "censored": 0,
        }
    )

    censored_rows = []
    for r in rows:
        sym = (r.get("symbol") or "?").upper()
        a = per[sym]
        mv = _move_pct(r)
        if max_move_pct is not None and mv is not None and abs(mv) > max_move_pct:
            # Implausible round trip. Ghost rows in this repo have filled past
            # their own take-profit limit before (98% of gross was 7 rows), so
            # a row that moves further than any stop could survive is an
            # artifact, not evidence. Counted and reported, never averaged in.
            a["censored"] += 1
            censored_rows.append({
                "symbol": sym, "move_pct": mv,
                "gross": float(r.get("gross_profit") or 0.0),
                "notional": _notional(r),
            })
            continue
        a["n"] += 1
        gross = float(r.get("gross_profit") or 0.0)
        fee = float(r.get("fee_cost") or 0.0)
        net_before = float(r.get("net_profit") or 0.0)
        a["gross"] += gross
        a["gross_rows"].append(gross)
        a["fee_charged"] += fee
        a["net_before"] += net_before
        if net_before > 0:
            a["wins_before"] += 1

        notional = _notional(r)
        rate = round_trip_pct(sym, notional if notional is not None else 0.0)
        if rate is None:
            # Unmeasured pool. Carry the row so n reconciles, but its
            # re-priced net is NOT computed and never enters a total.
            a["measured"] = False
            continue
        a["measured"] = True
        a["rate_pct"] = rate
        if notional is not None:
            a["notional"] += notional
            a["n_notional"] += 1
            fee_m = notional * rate / 100.0
        else:
            # No notional on the row: cannot charge a percentage. Skip the
            # re-price for this row rather than inventing a clip size.
            continue
        net_after = gross - fee_m
        a["fee_measured"] += fee_m
        a["net_after"] += net_after
        if net_after > 0:
            a["wins_after"] += 1

    measured, unmeasured = {}, {}
    for sym, a in per.items():
        (measured if a["measured"] else unmeasured)[sym] = a

    deep = {
        s: a
        for s, a in measured.items()
        if a["rate_pct"] is not None and a["rate_pct"] < depth_bar
    }
    thin = {
        s: a
        for s, a in measured.items()
        if a["rate_pct"] is not None and a["rate_pct"] >= depth_bar
    }
    return {
        "per_symbol": dict(per),
        "measured": measured,
        "unmeasured": unmeasured,
        "deep": deep,
        "thin": thin,
        "depth_bar_pct": depth_bar,
        "censored_rows": censored_rows,
    }


def top1_share(a: Dict[str, Any]) -> Optional[float]:
    """What share of a symbol's gross comes from its single biggest row.

    The fragility diagnostic. A symbol whose gross is one row is not evidence
    about that symbol, whatever the round-trip cost says.
    """
    g = a.get("gross_rows") or []
    tot = sum(g)
    if not g or abs(tot) < 1e-12:
        return None
    return 100.0 * max(g, key=abs) / tot


def _tot(group: Dict[str, Any], key: str) -> float:
    return sum(a[key] for a in group.values())


def _n(group: Dict[str, Any]) -> int:
    return sum(a["n"] for a in group.values())


CAVEATS = """
CAVEATS -- these travel with every number above and are not footnotes.
  1. THE COSTS ARE QUOTES, NOT FILLS. Read-only router quotes, USDC in and back.
     SLIPPAGE AND MEV ARE NOT MODELLED. Every 'after' figure is a COST FLOOR --
     the best case -- and is NOT a realisable P/L.
  2. ONLY 6 SYMBOLS WERE EVER QUOTED. Everything else is in the UNMEASURED
     bucket and is charged NOTHING here, because a default rate on an unquoted
     pool is an assumption dressed as a measurement.
  3. THE GHOST BOOK CARRIES NO REALISED SLIPPAGE AT ALL. Of 183 ghost round
     trips, 0 are fully measured and both slippage medians are 0.0 bps
     (data/roundtrip_cost_census.json). The 'before' column is a model too, so
     this is a comparison of two models -- not model versus observation. The
     LIVE lane measured ~27.8 bps of real slippage (21.73 buy + 6.04 sell,
     n=12) that NO ghost row carries.
  4. NO SYMBOL IS BANNED AND NO GATE IS CHANGED BY THIS SCRIPT.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db"))
    ap.add_argument("--depth-bar", type=float, default=DEFAULT_DEPTH_BAR_PCT)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--max-move-pct", type=float, default=20.0,
                    help="censor round trips whose realised move exceeds this; "
                         "0 disables the censor and reports the raw book")
    args = ap.parse_args()

    censor = args.max_move_pct if args.max_move_pct and args.max_move_pct > 0 else None
    rows = load_rows(args.db)
    res = reprice(rows, args.depth_bar, censor)

    print("=" * 78)
    print("GHOST BOOK RE-PRICED AT EACH SYMBOL'S OWN MEASURED ROUND-TRIP COST")
    print("=" * 78)
    print("rows in trade_outcomes: %d      depth bar: round trip < %.2f%%" % (len(rows), args.depth_bar))
    print()
    print("  !! CONTAMINATED: this loads ALL trade_outcomes rows, and 22 of 237")
    print("  !! are mode='live', not ghost (measured 0962fd8). Every number below")
    print("  !! mixes the two books. scripts/tradeable_book_repricing.py splits")
    print("  !! on details.mode and defaults to ghost -- prefer it. Fixing this")
    print("  !! here is a criterion on [42a65c7b]; the warning stays until then.")
    print()

    if censor is None:
        print("CENSOR: OFF -- every row included, artifacts and all.")
    else:
        print("CENSOR: round trips moving more than %.1f%% are EXCLUDED as artifacts."
              % censor)
        cr = res["censored_rows"]
        print("  %d row(s) censored: %s" % (len(cr), ", ".join(
            "%s %+.1f%% (gross %+.4f)" % (c["symbol"], c["move_pct"], c["gross"])
            for c in sorted(cr, key=lambda c: -abs(c["gross"]))[:6]) or "(none)"))
    print()

    hdr = "%-18s %4s %4s %7s %9s %9s %9s %9s %8s" % (
        "SYMBOL", "n", "cens", "rate%", "gross", "fee_book", "fee_meas", "net_after", "top1%")
    print(hdr)
    print("-" * len(hdr))
    for sym, a in sorted(res["measured"].items(), key=lambda kv: -kv[1]["n"]):
        t1 = top1_share(a)
        print("%-18s %4d %4d %7.4f %9.4f %9.4f %9.4f %9.4f %8s" % (
            sym, a["n"], a["censored"], a["rate_pct"] or 0.0, a["gross"],
            a["fee_charged"], a["fee_measured"], a["net_after"],
            "%.0f" % t1 if t1 is not None else "-"))
    print("  top1% = share of that symbol's GROSS contributed by its single")
    print("  largest row. Above ~50% the symbol is one row, not a sample.")
    print()
    print("UNMEASURED POOLS -- never quoted, charged NOTHING here, NOT in any total:")
    um = sorted(res["unmeasured"].items(), key=lambda kv: -kv[1]["n"])
    print("  %d symbols, %d rows, book net as recorded %+.4f" % (
        len(um), _n(res["unmeasured"]), _tot(res["unmeasured"], "net_before")))
    print("  largest: " + ", ".join("%s(n=%d)" % (s, a["n"]) for s, a in um[:8]))
    print()

    mb = _tot(res["measured"], "net_before")
    ma = _tot(res["measured"], "net_after")
    db_ = _tot(res["deep"], "net_before")
    da = _tot(res["deep"], "net_after")
    tb = _tot(res["thin"], "net_before")
    ta = _tot(res["thin"], "net_after")

    print("-" * 78)
    print("%-34s %5s %12s %12s" % ("GROUP", "n", "net BEFORE", "net AFTER"))
    print("%-34s %5d %12.4f %12.4f" % ("measured symbols (all)", _n(res["measured"]), mb, ma))
    print("%-34s %5d %12.4f %12.4f" % (
        "  DEEP (rt < %.2f%%)" % args.depth_bar, _n(res["deep"]), db_, da))
    print("%-34s %5d %12.4f %12.4f" % (
        "  THIN (rt >= %.2f%%)" % args.depth_bar, _n(res["thin"]), tb, ta))
    print("  deep symbols: %s" % (", ".join(sorted(res["deep"])) or "(none)"))
    print("  thin symbols: %s" % (", ".join(sorted(res["thin"])) or "(none)"))
    print()

    print("DOES SYMBOL SELECTION OR DIRECTION SET THE SIGN?")
    gross_deep = _tot(res["deep"], "gross")
    gross_thin = _tot(res["thin"], "gross")
    print("  GROSS (direction alone, before ANY cost):  deep %+.4f   thin %+.4f" % (gross_deep, gross_thin))
    print("  NET at each symbol's own measured cost:    deep %+.4f   thin %+.4f" % (da, ta))
    if gross_deep > 0 and da > 0:
        verdict = ("DEEP pools are positive BEFORE and AFTER their own cost. "
                   "On this evidence cost is survivable where the pool is deep.")
    elif gross_deep > 0 >= da:
        verdict = ("DEEP pools call direction positively (gross %+.4f) and their own "
                   "round trip eats it (net %+.4f). COST sets the sign." % (gross_deep, da))
    elif gross_deep <= 0:
        verdict = ("DEEP pools are negative on GROSS (%+.4f), before a cent of cost. "
                   "DIRECTION sets the sign, not symbol selection -- re-pricing "
                   "cannot rescue a book that is negative before costs." % gross_deep)
    else:
        verdict = "indeterminate on this sample."
    print("  VERDICT: " + verdict)
    print(CAVEATS)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "depth_bar_pct": args.depth_bar,
            "rows": len(rows),
            "measured": {s: a for s, a in res["measured"].items()},
            "unmeasured_n": _n(res["unmeasured"]),
            "totals": {
                "measured_before": mb, "measured_after": ma,
                "deep_before": db_, "deep_after": da,
                "thin_before": tb, "thin_after": ta,
                "gross_deep": gross_deep, "gross_thin": gross_thin,
            },
        }, indent=2), encoding="utf-8")
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
