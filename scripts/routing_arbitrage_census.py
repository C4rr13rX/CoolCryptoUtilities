#!/usr/bin/env python3
"""
Ask every DEX the same question at the same instant, and price the answer.

WHY THIS EXISTS. Every analysis in this repo ends at the same wall: the round
trip costs 0.3187% of notional plus $0.004047, the median 15-minute move is
0.2233%, and only 37.5% of ticks move further than the fee. Clip size cannot
fix it -- $10 to $250 buys 2.5 points of clearing share, because the
proportional rate is charged on notional and only the fixed leg amortises. So
the cost is the binding constraint, and it is the ONE input that modelling
cannot change. Nothing here has ever measured whether it could be lower.

This measures two things that share one set of quotes:

  1. ROUTING DISPERSION -- for one pair, one size, one instant, how far apart
     are the venues? If the best venue consistently beats the one the router
     picks, the difference is a cost reduction available today, with no model
     and no edge required. That is the cheapest possible win in this system.

  2. ARBITRAGE -- whether a CLOSED loop pays. A price gap between two venues
     is not arbitrage; arbitrage is buying on one and selling on the other and
     keeping something after BOTH legs and BOTH gas. This prices the round
     trip, not the gap, because pricing the gap is how this repo has
     manufactured a fake edge before.

WHAT THIS IS NOT. It places no orders and signs nothing. Every call is a quote.

THE TRAPS THIS IS BUILT AGAINST, all already paid for in this repo:

  * A GAP IS NOT AN EDGE. Two venues quoting different prices is the normal
    state of a market. The number that matters is what survives both legs.
  * QUOTES ARE NOT FILLS. A quoted buyAmount assumes your size does not move
    the pool. Both legs are quoted at the SAME size so the comparison is
    honest, and the report says plainly that slippage and MEV are not modelled.
  * SIMULTANEITY. Quotes taken seconds apart on a moving market manufacture a
    spread out of time, not out of venue. Each pair's quotes are gathered as
    close together as the calls allow and the elapsed span is recorded, so a
    reader can judge whether the spread outran the clock.
  * ONE ROW IS NOT A RESULT. The same rule as everywhere else here: a spread
    seen once is noise. Report n, and never rank on a single observation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Burn address used only to satisfy the build half of ``quote_and_build``.
#: It cannot receive and cannot sign, so calldata built for it is inert.
QUOTE_ONLY_RECIPIENT = "0x0000000000000000000000000000000000000001"

#: Cost of ONE leg, as a fraction of notional, measured from receipts.
#: services/round_trip_cost owns the round trip; a single leg is half of the
#: proportional part. Read at call time so a re-measure reaches this too.
def _leg_cost_fraction() -> float:
    try:
        from services.round_trip_cost import round_trip_cost  # noqa: PLC0415
        return float(round_trip_cost()) / 2.0
    except Exception:  # noqa: BLE001
        return 0.003187 / 2.0


def _providers() -> Dict[str, Any]:
    """Every provider that can answer a quote, by name. Missing ones are skipped.

    Each takes a web3 FACTORY, exactly as services/swap_service.py:369-371
    builds them -- a provider holds no connection of its own, so reusing the
    bridge's is both correct and the only way to get the same RPC the live
    path would use. A provider that cannot be built is skipped rather than
    raising: two venues are enough for a dispersion number.
    """
    out: Dict[str, Any] = {}
    try:
        from router_wallet import UltraSwapBridge  # noqa: PLC0415
        # No key material: every call here is a QUOTE, and a read-only bridge
        # cannot sign even by accident. That is a safety property, not a
        # convenience -- this script must never be able to move money.
        bridge = UltraSwapBridge()
        factory = lambda ch: bridge._w3(ch)  # noqa: E731
        factory("base")                      # fail loudly here, not per-quote
    except Exception as exc:  # noqa: BLE001
        print("routing census: cannot build a web3 factory (%s: %s)"
              % (type(exc).__name__, exc))
        return out

    for name, module, cls in (
        ("uniswap", "services.providers.uniswap_v3", "UniswapV3Local"),
        ("sushi", "services.providers.sushi_v2", "SushiV2Local"),
        ("camelot", "services.providers.camelot_v2", "CamelotV2Local"),
    ):
        try:
            mod = __import__(module, fromlist=[cls])
            out[name] = getattr(mod, cls)(factory)
        except Exception:  # noqa: BLE001
            continue
    return out


def quote_all(providers: Dict[str, Any], chain: str, token_in: str,
              token_out: str, amount_in: int,
              slippage_bps: int = 100) -> Tuple[Dict[str, Optional[int]], float]:
    """(venue -> buyAmount or None, seconds spanned).

    The span is returned because a spread measured across a moving market is a
    statement about the clock, not about the venues.
    """
    out: Dict[str, Optional[int]] = {}
    started = time.time()
    for name, provider in providers.items():
        try:
            # A recipient is required by the BUILD half of quote_and_build
            # even though only the quote half is read here. The zero address
            # is deliberate: it cannot receive anything and cannot sign, so a
            # calldata blob built for it is inert. This script must never be
            # able to move money.
            reply = provider.quote_and_build(
                chain, token_in, token_out, int(amount_in),
                slippage_bps=slippage_bps,
                recipient=QUOTE_ONLY_RECIPIENT)
        except Exception as exc:  # noqa: BLE001
            out[name] = None
            continue
        if not isinstance(reply, dict) or reply.get("__error__"):
            out[name] = None
            continue
        try:
            out[name] = int(reply.get("buyAmount") or 0) or None
        except (TypeError, ValueError):
            out[name] = None
    return out, time.time() - started


def dispersion(quotes: Dict[str, Optional[int]]) -> Optional[dict]:
    """How far apart are the venues, as a fraction of the best quote?

    This is the ROUTING number: what picking the best venue is worth against
    picking the worst, and against the median. It is available with no model
    and no edge -- it is simply not leaving money on the table.
    """
    live = {k: v for k, v in quotes.items() if v}
    if len(live) < 2:
        return None
    best_name = max(live, key=lambda k: live[k])
    worst_name = min(live, key=lambda k: live[k])
    best, worst = live[best_name], live[worst_name]
    ordered = sorted(live.values())
    median = ordered[len(ordered) // 2]
    return {
        "venues": len(live),
        "best": best_name,
        "worst": worst_name,
        "best_over_worst": (best - worst) / worst if worst else None,
        "best_over_median": (best - median) / median if median else None,
        "quotes": {k: int(v) for k, v in live.items()},
    }


def round_trip_arbitrage(providers: Dict[str, Any], chain: str,
                         token_a: str, token_b: str, amount_in: int,
                         gas_usd: float, notional_usd: float,
                         slippage_bps: int = 100) -> Optional[dict]:
    """Buy B on one venue, sell B back on another. Does the LOOP pay?

    A price gap is not arbitrage. This quotes BOTH legs at the same size and
    charges BOTH legs' cost and gas, because the only number worth having is
    what survives the whole round trip. If the loop does not close, it is a
    gap, and gaps are the normal state of a market.
    """
    leg_out, span_a = quote_all(providers, chain, token_a, token_b,
                                amount_in, slippage_bps)
    buys = {k: v for k, v in leg_out.items() if v}
    if not buys:
        return None

    buy_venue = max(buys, key=lambda k: buys[k])
    got_b = buys[buy_venue]

    # Sell the SAME quantity back, and ask every venue including the one we
    # bought on -- a same-venue loop is the honest null: it must lose exactly
    # the two legs' cost, and if it shows a profit the measurement is wrong.
    leg_back, span_b = quote_all(providers, chain, token_b, token_a,
                                 got_b, slippage_bps)
    sells = {k: v for k, v in leg_back.items() if v}
    if not sells:
        return None

    sell_venue = max(sells, key=lambda k: sells[k])
    got_a = sells[sell_venue]

    gross = (got_a - amount_in) / amount_in if amount_in else 0.0
    leg = _leg_cost_fraction()
    gas_fraction = (2.0 * gas_usd / notional_usd) if notional_usd > 0 else 0.0
    net = gross - 2.0 * leg - gas_fraction

    return {
        "buy_venue": buy_venue,
        "sell_venue": sell_venue,
        "same_venue": buy_venue == sell_venue,
        "gross_fraction": gross,
        "two_leg_cost": 2.0 * leg,
        "gas_fraction": gas_fraction,
        "net_fraction": net,
        "pays": net > 0,
        "span_sec": span_a + span_b,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", default="base")
    ap.add_argument("--pairs", default="",
                    help="comma-separated IN/OUT address pairs; default reads the address book")
    ap.add_argument("--notional-usd", type=float, default=10.0)
    ap.add_argument("--gas-usd", type=float, default=0.002,
                    help="per-leg gas in USD; base is cheap, measure it rather than trusting this")
    ap.add_argument("--rounds", type=int, default=1,
                    help="repeat the census N times -- one observation is not a result")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--max-pairs", type=int, default=6)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    providers = _providers()
    if len(providers) < 2:
        print("routing census: fewer than two providers available (%s); "
              "a dispersion number needs at least two venues to compare"
              % ", ".join(providers) or "none")
        return 2

    pairs = _resolve_pairs(args)
    if not pairs:
        print("routing census: no pairs resolved. Pass --pairs IN/OUT,IN/OUT "
              "with token addresses, or populate the address book.")
        return 2

    print("venues: %s   chain: %s   notional: $%.2f   leg cost: %.4f%%"
          % (", ".join(providers), args.chain, args.notional_usd,
             100 * _leg_cost_fraction()))
    print()

    rows: List[dict] = []
    for _round in range(max(1, args.rounds)):
        for label, (t_in, t_out, amount_in) in pairs.items():
            quotes, span = quote_all(providers, args.chain, t_in, t_out, amount_in)
            disp = dispersion(quotes)
            arb = round_trip_arbitrage(providers, args.chain, t_in, t_out,
                                       amount_in, args.gas_usd, args.notional_usd)
            rows.append({"pair": label, "dispersion": disp, "arbitrage": arb,
                         "span_sec": span})
            _print_row(label, disp, arb, span)
        if args.rounds > 1:
            time.sleep(max(0.0, args.sleep))

    _summarise(rows, args)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    return 0


def _resolve_pairs(args) -> Dict[str, Tuple[str, str, int]]:
    """Pairs to quote, as label -> (token_in, token_out, amount_in_wei)."""
    out: Dict[str, Tuple[str, str, int]] = {}
    if args.pairs:
        for spec in args.pairs.split(","):
            spec = spec.strip()
            if "/" not in spec:
                continue
            a, b = spec.split("/", 1)
            # USDC is 6 decimals on base; a caller passing other tokens should
            # say so rather than have this guess.
            amount = int(args.notional_usd * 10 ** 6)
            out["%s/%s" % (a[:8], b[:8])] = (a.strip(), b.strip(), amount)
        return out

    try:
        book = json.loads((ROOT / "data" / "token_addresses.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return out
    chain_book = book.get(args.chain) or {}

    def _addr(entry) -> Optional[str]:
        """The book stores {address, source, ts}, not a bare string."""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            value = entry.get("address")
            return value if isinstance(value, str) else None
        return None

    usdc = _addr(chain_book.get("USDC"))
    if not usdc:
        return out
    amount = int(args.notional_usd * 10 ** 6)   # USDC is 6 decimals on base
    for key, entry in chain_book.items():
        if str(key).upper() == "USDC":
            continue
        address = _addr(entry)
        if not address or address.lower() == usdc.lower():
            continue
        out["USDC/%s" % key] = (usdc, address, amount)
        if len(out) >= int(args.max_pairs):
            break
    return out


def _print_row(label: str, disp: Optional[dict], arb: Optional[dict],
               span: float) -> None:
    if not disp:
        print("  %-22s  fewer than two venues answered" % label[:22])
        return
    print("  %-22s  %d venues  best=%-8s spread best/worst %+.4f%%  (%.1fs)"
          % (label[:22], disp["venues"], disp["best"],
             100 * (disp["best_over_worst"] or 0.0), span))
    if arb:
        print("      loop %s->%s  gross %+.4f%%  cost %.4f%%  gas %.4f%%  NET %+.4f%%  %s"
              % (arb["buy_venue"], arb["sell_venue"],
                 100 * arb["gross_fraction"], 100 * arb["two_leg_cost"],
                 100 * arb["gas_fraction"], 100 * arb["net_fraction"],
                 "PAYS" if arb["pays"] else "does not pay"))


def _summarise(rows: List[dict], args) -> None:
    disps = [r["dispersion"]["best_over_worst"] for r in rows
             if r.get("dispersion") and r["dispersion"].get("best_over_worst") is not None]
    arbs = [r["arbitrage"] for r in rows if r.get("arbitrage")]
    print()
    print("=" * 78)
    if disps:
        disps_sorted = sorted(disps)
        median = disps_sorted[len(disps_sorted) // 2]
        print("ROUTING DISPERSION over %d observations" % len(disps))
        print("  median best-vs-worst spread : %+.4f%% of notional" % (100 * median))
        print("  cost of ONE leg             :  %.4f%% of notional" % (100 * _leg_cost_fraction()))
        if median > 0:
            print("  picking the best venue is worth %.1f%% of one leg's cost"
                  % (100 * median / max(_leg_cost_fraction(), 1e-12)))
        print("  NOTE: this is a saving only if the router is NOT already")
        print("  picking the best venue. Compare against default_route_order().")
    if arbs:
        paying = [a for a in arbs if a["pays"]]
        cross = [a for a in arbs if not a["same_venue"]]
        print()
        print("ARBITRAGE over %d closed loops" % len(arbs))
        print("  loops that pay after BOTH legs and gas : %d of %d" % (len(paying), len(arbs)))
        print("  cross-venue loops                      : %d" % len(cross))
        if not paying:
            print("  VERDICT: no loop closes. A price gap is not arbitrage;")
            print("  what matters is what survives both legs, and nothing did.")
    print()
    print("  Quotes, not fills: slippage and MEV are NOT modelled, and a quote")
    print("  assumes your size does not move the pool. Treat a paying loop as a")
    print("  lead to verify on a small real clip, never as realised profit.")
    print("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
