"""Split every recorded round trip into its buy leg and its sell leg, and
re-derive the round-trip cost constant from the rows rather than assuming it.

WHY THIS EXISTS. ``services/roundtrip_cost.py`` publishes one number that the
entry gate, the horizon table and every ghost P/L are computed from:

    cost_usd = 0.004047 + 0.003187 * notional

Both halves were described as "measured from this account's settled receipts",
and both were derived from the 38 fill rows that carried a gas receipt -- five
of which priced ETH gas at the traded pair's price. Nobody could say which leg
the cost sat in, because the buy leg recorded no priced cost at all.

WHAT THIS READS. ``trade_fills``, paired by ``trade_id`` into (buy, sell). A
leg's measured cost is gas plus realised slippage, per ``services/fill_cost.py``:
gas from the receipt, slippage from the gap between the price we were quoted and
the price we filled at -- which is the DEX fee, the spread and the impact
together, that being the only form they can be observed in.

Rows whose recorded native price fails the plausibility band are QUARANTINED and
counted separately, never averaged in. That is the 2026-09-04 defect, and the
whole point of this script is to stop it setting a constant.

Run:  python -X utf8 scripts/roundtrip_cost_census.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.fill_cost import (  # noqa: E402
    FILL_COST_FIELDS,
    leg_cost_usd,
    native_price_is_plausible,
    realised_slippage_bps,
)
from services.roundtrip_cost import DEFAULT_FIXED_USD, DEFAULT_RATE  # noqa: E402


def _details(row: Dict[str, Any]) -> Dict[str, Any]:
    det = row.get("details")
    if isinstance(det, str):
        try:
            det = json.loads(det)
        except (TypeError, ValueError):
            det = {}
    det = dict(det or {})
    det.setdefault("_ts", row.get("ts"))
    det.setdefault("_symbol", row.get("symbol"))
    det["chain"] = det.get("chain") or row.get("chain")
    return det


def _median(values: List[float]) -> Optional[float]:
    return st.median(values) if values else None


def collect(limit: int = 100_000) -> Dict[str, Any]:
    import db

    rows = [_details(r) for r in db.get_db().fetch_trade_fills(limit=limit)]

    by_trade: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for det in rows:
        mode = str(det.get("mode") or "")
        tid = det.get("trade_id")
        if not tid or "_entry" not in mode and "_exit" not in mode:
            continue
        by_trade[tid]["buy" if mode.endswith("_entry") else "sell"] = det

    # A ROUND TRIP WHOSE LEGS ARE IN DIFFERENT LANES IS NOT A COST SAMPLE. Two
    # of them are in this book -- real money entered and a SIMULATED exit was
    # booked against it -- so they are named and excluded rather than averaged
    # into either lane's cost.
    cross_lane = [
        {
            "trade_id": tid,
            "symbol": legs["buy"].get("_symbol"),
            "ts": legs["buy"].get("_ts"),
            "entry_mode": legs["buy"].get("mode"),
            "exit_mode": legs["sell"].get("mode"),
        }
        for tid, legs in by_trade.items()
        if legs.get("buy") is not None
        and legs.get("sell") is not None
        and str(legs["buy"].get("mode")).split("_")[0]
        != str(legs["sell"].get("mode")).split("_")[0]
    ]
    cross_ids = {row["trade_id"] for row in cross_lane}

    lanes: Dict[str, Dict[str, Any]] = {}
    for lane, entry_mode in (("live", "live_entry"), ("ghost", "ghost_entry")):
        pairs = [
            (b, s)
            for tid, legs in by_trade.items()
            if tid not in cross_ids
            and (b := legs.get("buy")) is not None
            and (s := legs.get("sell")) is not None
            and str(b.get("mode")) == entry_mode
        ]
        buy_gas, sell_gas, buy_slip, sell_slip, notionals = [], [], [], [], []
        quarantined: List[Dict[str, Any]] = []
        measured = 0
        back_filled = 0
        for buy, sell in pairs:
            # THE BUY LEG'S GAS WAS RECORDED FROM THE FIRST LIVE TRADE; ONLY ITS
            # PRICE WAS MISSING. For a round trip already closed, the sell leg
            # carries the native price minutes later on the same chain -- so the
            # entry's receipt can be valued without inventing anything. This is
            # a back-fill of rows written before the writer recorded the price,
            # counted separately so the report can say how many legs needed it.
            # New rows carry their own price and never reach this branch.
            if buy.get("native_token_price_usd") is None and buy.get("gas_price_usd") is None:
                sell_price = sell.get("native_token_price_usd", sell.get("gas_price_usd"))
                ok, _ = native_price_is_plausible(sell.get("chain"), sell_price)
                if ok and sell.get("native_token_price_source") != "route_native":
                    buy = dict(buy)
                    buy["native_token_price_usd"] = sell_price
                    buy["native_token_price_source"] = "back_filled_from_sell_leg"
                    back_filled += 1
            bc, sc = leg_cost_usd(buy), leg_cost_usd(sell)
            for leg_det, cost, label in ((buy, bc, "buy"), (sell, sc, "sell")):
                price = leg_det.get("native_token_price_usd", leg_det.get("gas_price_usd"))
                if price is None:
                    continue
                ok, why = native_price_is_plausible(leg_det.get("chain"), price)
                source = leg_det.get("native_token_price_source")
                if not ok or source == "route_native":
                    quarantined.append(
                        {
                            "symbol": leg_det.get("_symbol"),
                            "leg": label,
                            "native_token_price_usd": price,
                            "source": source,
                            "reason": why or "source is route_native on a non-native pair",
                        }
                    )
            if bc["gas_cost_usd"] is not None:
                buy_gas.append(bc["gas_cost_usd"])
            if sc["gas_cost_usd"] is not None:
                sell_gas.append(sc["gas_cost_usd"])
            bs = realised_slippage_bps(
                expected_price=buy.get("expected_price"),
                executed_price=buy.get("executed_price"),
                leg="buy",
            )
            ss = realised_slippage_bps(
                expected_price=sell.get("expected_price"),
                executed_price=sell.get("executed_price"),
                leg="sell",
            )
            if bs is not None:
                buy_slip.append(bs)
            if ss is not None:
                sell_slip.append(ss)
            n = bc["notional_usd"]
            if n is not None:
                try:
                    notionals.append(float(n))
                except (TypeError, ValueError):
                    pass
            if bc["measured"] and sc["measured"]:
                measured += 1

        lanes[lane] = {
            "round_trips": len(pairs),
            "fully_measured_round_trips": measured,
            "buy_gas_usd_median": _median(buy_gas),
            "buy_gas_usd_n": len(buy_gas),
            "sell_gas_usd_median": _median(sell_gas),
            "sell_gas_usd_n": len(sell_gas),
            "buy_slippage_bps_median": _median(buy_slip),
            "buy_slippage_bps_n": len(buy_slip),
            "sell_slippage_bps_median": _median(sell_slip),
            "sell_slippage_bps_n": len(sell_slip),
            "notional_usd_median": _median(notionals),
            "quarantined_legs": quarantined,
            "buy_legs_back_filled": back_filled,
            # The ghost writer books ONE round-trip scalar (fee_cost) against
            # the sell leg and charges the buy leg nothing, so the share of a
            # ghost round trip's recorded cost is 0/100 by construction rather
            # than by measurement. Counted, because that IS the answer for the
            # lane that has enough round trips to answer over.
            "round_trip_scalar_on_sell_leg": sum(
                1 for _b, s in pairs if s.get("fee_cost") is not None
            ),
            "buy_leg_charged_nothing": sum(
                1
                for b, _s in pairs
                if b.get("fee_cost") is None and b.get("gas_cost_usd") is None
            ),
        }

    # --- gas, in native units, from every leg that has a receipt at all -------
    # The buy leg's gas was recorded from the first live trade; only its PRICE
    # was missing. So the fixed part of the cost constant can be re-derived over
    # both legs by valuing every receipt at one honest native price.
    gas_native = {"buy": [], "sell": []}
    for det in rows:
        g = det.get("gas_spent_native")
        mode = str(det.get("mode") or "")
        if g is None or ("_entry" not in mode and "_exit" not in mode):
            continue
        try:
            value = float(g)
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            gas_native["buy" if mode.endswith("_entry") else "sell"].append(value)

    plausible_prices = []
    for det in rows:
        price = det.get("native_token_price_usd", det.get("gas_price_usd"))
        if price is None:
            continue
        ok, _ = native_price_is_plausible(det.get("chain"), price)
        if ok and det.get("native_token_price_source") != "route_native":
            try:
                plausible_prices.append(float(price))
            except (TypeError, ValueError):
                pass

    native_usd = _median(plausible_prices)
    buy_med = _median(gas_native["buy"])
    sell_med = _median(gas_native["sell"])
    rederived_fixed = None
    if native_usd and buy_med is not None and sell_med is not None:
        rederived_fixed = (buy_med + sell_med) * native_usd

    live = lanes.get("live", {})
    slip_total = None
    if live.get("buy_slippage_bps_median") is not None and live.get("sell_slippage_bps_median") is not None:
        slip_total = (live["buy_slippage_bps_median"] + live["sell_slippage_bps_median"]) / 10_000.0

    return {
        "fill_rows": len(rows),
        "lanes": lanes,
        "cross_lane_round_trips": cross_lane,
        "gas_native_median": {"buy": buy_med, "sell": sell_med},
        "gas_native_n": {"buy": len(gas_native["buy"]), "sell": len(gas_native["sell"])},
        "native_price_usd_used": native_usd,
        "native_price_usd_n": len(plausible_prices),
        "rederived": {
            "fixed_usd": rederived_fixed,
            "fixed_usd_published": DEFAULT_FIXED_USD,
            "rate": slip_total,
            "rate_published": DEFAULT_RATE,
        },
        "field_status": {k: v["status"] for k, v in FILL_COST_FIELDS.items()},
    }


def _pct(part: Optional[float], whole: Optional[float]) -> str:
    if part is None or whole is None or not whole:
        return "  n/a"
    return "%5.1f%%" % (100.0 * part / whole)


def report(data: Dict[str, Any]) -> None:
    print("ROUND-TRIP COST, SPLIT INTO LEGS")
    print("=" * 74)
    print("fill rows read: %d\n" % data["fill_rows"])

    for lane in ("live", "ghost"):
        info = data["lanes"].get(lane) or {}
        print("--- %s lane: %d paired round trips (%d fully measured on both legs)"
              % (lane.upper(), info.get("round_trips", 0), info.get("fully_measured_round_trips", 0)))
        bg, sg = info.get("buy_gas_usd_median"), info.get("sell_gas_usd_median")
        bs, ss = info.get("buy_slippage_bps_median"), info.get("sell_slippage_bps_median")
        print("    gas USD        buy %s (n=%d)   sell %s (n=%d)"
              % (("%.6f" % bg) if bg is not None else "unpriced", info.get("buy_gas_usd_n", 0),
                 ("%.6f" % sg) if sg is not None else "unpriced", info.get("sell_gas_usd_n", 0)))
        print("    slippage bps   buy %s (n=%d)   sell %s (n=%d)"
              % (("%+.2f" % bs) if bs is not None else "n/a", info.get("buy_slippage_bps_n", 0),
                 ("%+.2f" % ss) if ss is not None else "n/a", info.get("sell_slippage_bps_n", 0)))
        if bg is not None and sg is not None:
            total = bg + sg
            print("    GAS SHARE      buy %s   sell %s   (of $%.6f)"
                  % (_pct(bg, total), _pct(sg, total), total))
        if bs is not None and ss is not None and info.get("notional_usd_median"):
            n = info["notional_usd_median"]
            bu, su = n * bs / 10_000.0, n * ss / 10_000.0
            print("    SLIPPAGE USD   buy %+.6f   sell %+.6f   at the median $%.4f notional"
                  % (bu, su, n))
        if info.get("buy_legs_back_filled"):
            print("    back-filled    %d buy leg(s) valued at the sell leg's native price"
                  % info["buy_legs_back_filled"])
        rts = info.get("round_trip_scalar_on_sell_leg") or 0
        if rts:
            print("    RECORDED COST SHARE, from the writer's own fields: buy 0.0%   sell 100.0%")
            print("      %d of %d round trips book ONE round-trip scalar against the SELL leg and"
                  % (rts, info.get("round_trips", 0)))
            print("      charge the BUY leg nothing, so that share is by construction, not measured.")
        q = info.get("quarantined_legs") or []
        if q:
            print("    QUARANTINED    %d leg(s) whose native price is not believable:" % len(q))
            for row in q:
                print("        %-14s %-4s $%-12.4f source=%-12s %s"
                      % (row["symbol"], row["leg"], row["native_token_price_usd"],
                         row["source"] or "(absent)", row["reason"]))
        print()

    cross = data.get("cross_lane_round_trips") or []
    if cross:
        print("--- CROSS-LANE ROUND TRIPS, excluded from both lanes above: %d" % len(cross))
        print("    real money entered and a SIMULATED exit was booked against it.")
        for row in cross:
            print("        %-14s %s -> %s  ts=%s  %s"
                  % (row["symbol"], row["entry_mode"], row["exit_mode"], row["ts"],
                     row["trade_id"]))
        print()

    r = data["rederived"]
    gn, nn = data["gas_native_median"], data["gas_native_n"]
    print("--- RE-DERIVING THE PUBLISHED CONSTANT")
    print("    gas in native units, from every receipt: buy median %s (n=%d), sell median %s (n=%d)"
          % (("%.4e" % gn["buy"]) if gn["buy"] is not None else "none", nn["buy"],
             ("%.4e" % gn["sell"]) if gn["sell"] is not None else "none", nn["sell"]))
    print("    native token price used: %s (median of %d believable rows)"
          % (("$%.4f" % data["native_price_usd_used"]) if data["native_price_usd_used"] else "none",
             data["native_price_usd_n"]))
    if r["fixed_usd"] is not None:
        moved = r["fixed_usd"] - r["fixed_usd_published"]
        print("    FIXED  published $%.6f  re-derived $%.6f  (%+.6f, %+.1f%%)"
              % (r["fixed_usd_published"], r["fixed_usd"], moved,
                 100.0 * moved / r["fixed_usd_published"]))
    else:
        print("    FIXED  cannot be re-derived: no believable native price in the rows")
    if r["rate"] is not None:
        moved = r["rate"] - r["rate_published"]
        print("    RATE   published %.6f  re-derived %.6f  (%+.6f, %+.1f%%)"
              % (r["rate_published"], r["rate"], moved, 100.0 * moved / r["rate_published"]))
    else:
        print("    RATE   cannot be re-derived: no measured slippage on both legs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="also write the numbers to this path")
    args = ap.parse_args()
    data = collect()
    report(data)
    if args.json:
        Path(args.json).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
