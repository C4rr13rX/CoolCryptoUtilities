#!/usr/bin/env python3
"""
The money-button universe is the INTERSECTION of two columns, and we have
only ever held one of them.

WHY THIS EXISTS. The operator asked for money-button trading every 5-30
minutes and for the system to find strategies it can trade PROFITABLY at that
cadence. Pass 424 measured the MOVE column -- how far a symbol travels over a
15-40 minute hold -- and ranked symbols by it. That ranking is worthless on
its own. A symbol that moves 1.03% at p90 and costs 1.20% to round-trip is a
LOSING symbol that looks like the best one on the board, and AERO is exactly
that shape: it is the most-traded ghost symbol here, its p90 move is 0.416%,
and its round trip has been quoted as high as 4.35%.

So the only ordering that answers the operator's question is

    p90 absolute move over the hold  MINUS  round-trip cost at the clip we
    actually trade

and everything in this file exists to produce that one column honestly.

WHAT IT DOES NOT DO. It changes no production symbol universe and places no
orders. Every cost number is a QUOTE, taken through the same provider path
scripts/routing_arbitrage_census.py uses, and the decision about what to trade
goes to the operator with the arithmetic attached.

THE TRAPS THIS IS BUILT AGAINST, all already paid for in this repo:

  * A FROZEN FEED LOOKS LIKE A CALM MARKET. A symbol whose p50, p90 and p99
    are identical is not a stable asset, it is a stuck price. Ranking it as
    "low volatility" and trading it is how a dead feed becomes a position.
  * CONTAMINATION LOOKS LIKE OPPORTUNITY. A 4000% "move" is a denomination
    bug or a ticker collision, not a trade. It sorts to the top of a move
    ranking every single time, which is precisely why the hygiene rule here
    is CODE with a printed count and not a human squinting at a table.
    A feed bug and a data bug are reported SEPARATELY: the first is something
    we can fix and then trade, the second is something we must never trade.
  * MAXIMUM FAVOURABLE EXCURSION IS NOT A MOVE YOU CAN CATCH. Taking the best
    price in the window assumes an exit oracle. The headline number here is a
    DETERMINISTIC exit -- the first tick at or after the minimum hold -- and
    the MFE is printed beside it, labelled as the unreachable upper bound.
  * A COST QUOTED AT THE WRONG SIZE IS NOT THIS SYSTEM'S COST. Round-trip
    cost here INVERTS with clip size, so it is quoted at the clip the book
    actually traded, read from the book, not at a round number.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# HYGIENE. Every rule is code, every rule has a name, every rule has a count,
# and every rule declares whether what it caught is a FEED bug (ours to fix,
# the symbol may be fine) or a DATA bug (the number itself is false, and the
# symbol must not be traded on it). That distinction is the point: the two
# failures look identical in a table and have opposite responses.
# ---------------------------------------------------------------------------

#: A move larger than this over a 40-minute hold is a denomination bug, a
#: ticker collision or a decimals error. Stated here rather than chosen per
#: symbol, so the reader can disagree with ONE number instead of a judgement.
IMPLAUSIBLE_MOVE = 0.50          # 50% in <= 40 minutes

#: Below this many usable (anchor, exit) pairs there is no percentile to take.
MIN_PAIRS = 40

#: A feed that has not ticked in this long cannot be traded on now, whatever
#: its history says.
STALE_SEC = 3 * 3600

#: Wider than this between consecutive ticks and a "15-40 minute window" is
#: being interpolated out of two observations that do not bracket it.
MAX_MEDIAN_GAP_SEC = 300.0


class Excluded(Exception):
    """A symbol failed a hygiene rule. Carries the rule name and its class."""

    def __init__(self, rule: str, kind: str, detail: str) -> None:
        super().__init__("%s: %s" % (rule, detail))
        self.rule = rule
        self.kind = kind          # "feed" (fixable) or "data" (never trade)
        self.detail = detail


# ---------------------------------------------------------------------------
# THE MOVE COLUMN
# ---------------------------------------------------------------------------

def load_series(db: str, hours: float) -> Dict[str, List[Tuple[float, float]]]:
    """symbol -> [(ts, price)], ascending, from the one real store."""
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        cutoff = time.time() - hours * 3600.0
        out: Dict[str, List[Tuple[float, float]]] = {}
        for symbol, ts, price in con.execute(
                "select symbol, ts, price from market_stream "
                "where ts > ? and price is not null and price > 0 "
                "order by symbol, ts", (cutoff,)):
            out.setdefault(str(symbol), []).append((float(ts), float(price)))
        return out
    finally:
        con.close()


def move_profile(series: Sequence[Tuple[float, float]],
                 lo_min: float, hi_min: float) -> dict:
    """Absolute forward move over the operator's hold window.

    TWO numbers, and the difference between them is the whole argument:

      det -- the price at the FIRST tick at or after ``lo_min``, provided it
             lands inside the window. One deterministic exit, no choice, no
             oracle. This is the number that gets ranked.
      mfe -- the largest absolute excursion anywhere in the window. This is
             what you would capture with a perfect exit, which nobody has.
             Printed so the gap between hope and arithmetic is visible.

    Raises ``Excluded`` when the series cannot support the question.
    """
    if len(series) < MIN_PAIRS:
        raise Excluded("thin_series", "feed",
                       "%d ticks, need %d" % (len(series), MIN_PAIRS))

    now = time.time()
    last_ts = series[-1][0]
    if now - last_ts > STALE_SEC:
        raise Excluded("stale_feed", "feed",
                       "last tick %.1fh ago" % ((now - last_ts) / 3600.0))

    gaps = [series[i + 1][0] - series[i][0] for i in range(len(series) - 1)]
    median_gap = statistics.median(gaps) if gaps else float("inf")
    if median_gap > MAX_MEDIAN_GAP_SEC:
        raise Excluded("sparse_feed", "feed",
                       "median gap %.0fs exceeds %.0fs"
                       % (median_gap, MAX_MEDIAN_GAP_SEC))

    lo_sec, hi_sec = lo_min * 60.0, hi_min * 60.0
    det: List[float] = []
    mfe: List[float] = []
    j = 0
    for i, (t0, p0) in enumerate(series):
        # Advance the window start; both pointers only move forward, so the
        # whole scan is linear rather than quadratic in the tick count.
        if j < i:
            j = i
        while j < len(series) and series[j][0] < t0 + lo_sec:
            j += 1
        if j >= len(series) or series[j][0] > t0 + hi_sec:
            continue
        det.append(abs(series[j][1] - p0) / p0)
        best = 0.0
        k = j
        while k < len(series) and series[k][0] <= t0 + hi_sec:
            best = max(best, abs(series[k][1] - p0) / p0)
            k += 1
        mfe.append(best)

    if len(det) < MIN_PAIRS:
        raise Excluded("thin_windows", "feed",
                       "%d usable %g-%gmin windows, need %d"
                       % (len(det), lo_min, hi_min, MIN_PAIRS))

    det_sorted = sorted(det)
    p = lambda q: det_sorted[min(len(det_sorted) - 1,  # noqa: E731
                                 int(q * len(det_sorted)))]
    p50, p90, p99 = p(0.50), p(0.90), p(0.99)

    # A stuck price and a calm market are the same table row and opposite
    # trades. Identical percentiles mean the series has one value.
    if p50 == p90 == p99:
        raise Excluded("frozen_feed", "feed",
                       "p50=p90=p99=%.6f, the price is not moving" % p50)
    if p90 <= 0.0:
        raise Excluded("zero_percentiles", "feed",
                       "p90 move is exactly 0.000")
    if p99 > IMPLAUSIBLE_MOVE:
        raise Excluded("implausible_move", "data",
                       "p99 %.1f%% exceeds the %.0f%% bound"
                       % (100 * p99, 100 * IMPLAUSIBLE_MOVE))

    return {
        "n_ticks": len(series),
        "n_windows": len(det),
        "median_gap_sec": median_gap,
        "span_h": (series[-1][0] - series[0][0]) / 3600.0,
        "p50": p50, "p90": p90, "p99": p99,
        "mfe_p90": sorted(mfe)[min(len(mfe) - 1, int(0.90 * len(mfe)))],
        "share_over_30bp": sum(1 for d in det if d > 0.0030) / len(det),
    }


# ---------------------------------------------------------------------------
# THE COST COLUMN -- through the existing census path, not a new one.
# ---------------------------------------------------------------------------

def traded_clips(db: str) -> Dict[str, float]:
    """symbol -> median notional the book ACTUALLY traded, in USD.

    Cost inverts with clip size here, so quoting at a round number measures a
    clip we do not trade. The book knows what we trade; ask it.
    """
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        out: Dict[str, List[float]] = {}
        for symbol, notional in con.execute(
                "select symbol, entry_price * quantity from trade_outcomes "
                "where entry_price > 0 and quantity > 0"):
            if notional and float(notional) > 0:
                out.setdefault(str(symbol), []).append(float(notional))
        return {k: statistics.median(v) for k, v in out.items() if v}
    finally:
        con.close()


def address_book(chain: str) -> Dict[str, str]:
    """SYMBOL -> address, flattening the {address, source, ts} record form."""
    try:
        book = json.loads((ROOT / "data" / "token_addresses.json")
                          .read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out: Dict[str, str] = {}
    for key, entry in (book.get(chain) or {}).items():
        addr = entry if isinstance(entry, str) else (
            entry.get("address") if isinstance(entry, dict) else None)
        if isinstance(addr, str) and addr.startswith("0x"):
            out[str(key).upper()] = addr
    return out


#: A round trip costing more than this on a clip of a few dollars is not an
#: expensive symbol, it is a BROKEN ROUTE. No real pool charges 74% to buy
#: and sell $2 back. Stated here so the reader can disagree with one number.
UNPRICEABLE_COST = 0.25

#: Two quotes of the SAME pair at the SAME size seconds apart must agree. A
#: route whose cost moves more than this between repeats is not being
#: measured, it is being sampled from a dust pool. Absolute, in notional
#: fraction, because a relative bound is meaningless once the level is wrong.
UNSTABLE_QUOTE_SPREAD = 0.01


def _one_round_trip(providers: Dict[str, Any], chain: str, usdc: str,
                    token: str, amount_in: int) -> Optional[dict]:
    """One buy-then-sell-back quote pair. None if either leg went unanswered."""
    from scripts.routing_arbitrage_census import quote_all  # noqa: PLC0415

    out_leg, span_a = quote_all(providers, chain, usdc, token, amount_in)
    buys = {k: v for k, v in out_leg.items() if v}
    if not buys:
        return None
    buy_venue = max(buys, key=lambda k: buys[k])
    got_token = buys[buy_venue]

    back_leg, span_b = quote_all(providers, chain, token, usdc, got_token)
    sells = {k: v for k, v in back_leg.items() if v}
    if not sells:
        return None
    sell_venue = max(sells, key=lambda k: sells[k])
    got_usdc = sells[sell_venue]
    return {
        "slippage_fraction": (amount_in - got_usdc) / amount_in,
        "venue_in": buy_venue, "venue_out": sell_venue,
        "venues_in": len(buys), "venues_out": len(sells),
        "span_sec": span_a + span_b,
    }


def quote_round_trip(providers: Dict[str, Any], chain: str, usdc: str,
                     token: str, notional_usd: float, gas_usd: float,
                     repeats: int = 3) -> Optional[dict]:
    """Buy the token with USDC, sell every unit straight back. What is lost?

    This is the honest cost of the money button's round trip: pool fee twice,
    price impact twice, spread twice, at the size we actually send. It uses
    ``quote_all`` from the routing census so there is ONE quoting path in this
    repo and not two that can disagree.

    IT QUOTES THE SAME THING ``repeats`` TIMES, and that is not caution, it is
    the rule that caught the first version of this file lying. Measured here:
    AERO at one $0.83 clip quoted 2.186% in one run and 74.272% minutes later,
    because services/providers/uniswap_v3 is direct-pair only and AERO's real
    liquidity is on Aerodrome, which is not a provider we have. The buy and
    sell legs of the SAME venue implied prices 39x apart. A cost like that is
    not a dear symbol, it is a dust pool, and reporting it as a cost would put
    a fake number into the one column this whole item exists to produce.

    So a route that is unstable or absurd comes back ``priceable=False`` with
    a reason, and the caller must report it as UNKNOWN COST -- never as a high
    cost, because "too expensive to trade" and "we cannot price it" lead to
    opposite decisions.

    Both legs take the BEST venue, which is the router's job. That makes this
    a LOWER bound on realised cost -- quotes are not fills, and slippage and
    MEV are not modelled.
    """
    amount_in = int(round(notional_usd * 10 ** 6))   # USDC is 6dp on base
    if amount_in <= 0:
        return None

    tries = [t for t in (_one_round_trip(providers, chain, usdc, token,
                                         amount_in)
                         for _ in range(max(1, repeats))) if t]
    if not tries:
        return None

    slips = [t["slippage_fraction"] for t in tries]
    spread = max(slips) - min(slips)
    slip = statistics.median(slips)
    gas_fraction = (2.0 * gas_usd / notional_usd) if notional_usd > 0 else 0.0
    cost = slip + gas_fraction

    reason = ""
    if len(tries) < max(1, repeats):
        reason = ("only %d of %d repeats answered"
                  % (len(tries), max(1, repeats)))
    elif spread > UNSTABLE_QUOTE_SPREAD:
        reason = ("unstable route: %d repeats spread %.3f%% of notional"
                  % (len(tries), 100 * spread))
    elif cost > UNPRICEABLE_COST:
        reason = ("unpriceable route: %.1f%% round trip on a $%.2f clip is a "
                  "dust pool, not a fee" % (100 * cost, notional_usd))

    return {
        "notional_usd": notional_usd,
        "venue_in": tries[0]["venue_in"], "venue_out": tries[0]["venue_out"],
        "slippage_fraction": slip,
        "gas_fraction": gas_fraction,
        "round_trip_cost": cost,
        "repeats": len(tries),
        "repeat_spread": spread,
        "venues_in": tries[0]["venues_in"], "venues_out": tries[0]["venues_out"],
        "priceable": not reason,
        "unpriceable_reason": reason,
        "span_sec": sum(t["span_sec"] for t in tries),
    }


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=96.0)
    ap.add_argument("--lo-min", type=float, default=15.0)
    ap.add_argument("--hi-min", type=float, default=40.0)
    ap.add_argument("--chain", default="base")
    ap.add_argument("--default-clip", type=float, default=2.0,
                    help="clip for symbols the book has never traded; the "
                         "book's own median across all symbols is ~$2.25")
    ap.add_argument("--gas-usd", type=float, default=0.002,
                    help="per-leg gas in USD on base")
    ap.add_argument("--max-quote", type=int, default=14,
                    help="how many surviving symbols to quote, best move first")
    ap.add_argument("--repeats", type=int, default=3,
                    help="quote each round trip this many times; a route whose "
                         "cost moves between repeats is not being measured")
    ap.add_argument("--no-quote", action="store_true",
                    help="MOVE column and hygiene only; no RPC calls")
    ap.add_argument("--json", default="",
                    help="also write the joined table here")
    args = ap.parse_args()

    db = os.getenv("TRADING_DB_PATH", str(ROOT / "storage" / "trading_cache.db"))
    print("=" * 86)
    print("MONEY-BUTTON UNIVERSE = MOVE x COST      hold %g-%g min, %gh of "
          "market_stream" % (args.lo_min, args.hi_min, args.hours))
    print("store: %s" % db)
    print("=" * 86)

    series = load_series(db, args.hours)
    print("\nsymbols with any tick in the window: %d" % len(series))

    kept: Dict[str, dict] = {}
    excluded: Dict[str, List[Tuple[str, str]]] = {}
    for symbol, points in series.items():
        try:
            kept[symbol] = move_profile(points, args.lo_min, args.hi_min)
        except Excluded as exc:
            excluded.setdefault(exc.rule, []).append((symbol, exc.detail))

    # ---- hygiene, reported as counts per named rule -----------------------
    print("\nHYGIENE -- applied as code, one line per rule, FEED bugs and DATA")
    print("bugs reported separately because they have opposite responses.")
    feed_rules = [(r, v) for r, v in sorted(excluded.items())
                  if r != "implausible_move"]
    data_rules = [(r, v) for r, v in sorted(excluded.items())
                  if r == "implausible_move"]
    print("\n  FEED BUGS -- the symbol may be fine, the feed is not. Fixable.")
    if not feed_rules:
        print("    (none)")
    for rule, victims in feed_rules:
        print("    %-18s removed %3d   e.g. %s (%s)"
              % (rule, len(victims), victims[0][0], victims[0][1]))
    print("\n  DATA BUGS -- the number itself is false. Never trade on it.")
    if not data_rules:
        print("    (none)")
    for rule, victims in data_rules:
        print("    %-18s removed %3d   e.g. %s (%s)"
              % (rule, len(victims), victims[0][0], victims[0][1]))
        for sym, detail in victims[:8]:
            print("        %-18s %s" % (sym, detail))
    print("\n  survived every rule: %d of %d symbols" % (len(kept), len(series)))
    if not kept:
        print("\nNothing survived hygiene. There is no universe to quote.")
        return 1

    # ---- cost column ------------------------------------------------------
    clips = traded_clips(db)
    book = address_book(args.chain)
    usdc = book.get("USDC")

    order = sorted(kept, key=lambda s: kept[s]["p90"], reverse=True)
    costs: Dict[str, dict] = {}
    unquoted: Dict[str, str] = {}

    if args.no_quote:
        print("\n--no-quote: MOVE column only, no cost was measured.")
    elif not usdc:
        print("\nNo USDC address on %s in the book; cannot quote." % args.chain)
    else:
        from scripts.routing_arbitrage_census import _providers  # noqa: PLC0415
        providers = _providers()
        if len(providers) < 2:
            print("\nFewer than two venues available (%s). A round trip can "
                  "still be quoted on one, but the router's best-venue "
                  "assumption is untested." % (", ".join(providers) or "none"))
        if not providers:
            print("No provider answered; the COST column is UNMEASURED this run.")
        else:
            print("\nQUOTING round trips through the routing-census path "
                  "(venues: %s)" % ", ".join(providers))
            print("at the clip the BOOK actually traded, because cost inverts "
                  "with clip size here.\n")
            for symbol in order[:max(1, args.max_quote)]:
                base = symbol.split("-")[0].upper()
                token = book.get(base)
                if not token:
                    unquoted[symbol] = "no address in the book for %s" % base
                    continue
                clip = clips.get(symbol, args.default_clip)
                quote = quote_round_trip(providers, args.chain, usdc, token,
                                         clip, args.gas_usd,
                                         repeats=args.repeats)
                if not quote:
                    unquoted[symbol] = "no venue quoted both legs at $%.2f" % clip
                    print("  %-16s  NO QUOTE  ($%.2f clip)" % (symbol, clip))
                    continue
                if not quote["priceable"]:
                    # UNKNOWN cost, not high cost. The two read the same in a
                    # table and lead to opposite decisions: one says "do not
                    # trade this symbol", the other says "fix the router".
                    unquoted[symbol] = quote["unpriceable_reason"]
                    print("  %-16s  $%-5.2f  UNPRICEABLE -- %s"
                          % (symbol, clip, quote["unpriceable_reason"]))
                    continue
                quote["clip_source"] = ("book" if symbol in clips
                                        else "default")
                costs[symbol] = quote
                print("  %-16s  $%-5.2f  slip %+.4f%%  gas %.4f%%  "
                      "ROUND TRIP %.4f%%  [%s->%s]  %d repeats, spread %.4f%%"
                      % (symbol, clip, 100 * quote["slippage_fraction"],
                         100 * quote["gas_fraction"],
                         100 * quote["round_trip_cost"],
                         quote["venue_in"], quote["venue_out"],
                         quote["repeats"], 100 * quote["repeat_spread"]))
            if len(order) > args.max_quote:
                print("\n  NOT QUOTED (cap --max-quote=%d, ranked by move): %s"
                      % (args.max_quote, ", ".join(order[args.max_quote:])))

    # ---- the join ---------------------------------------------------------
    print("\n" + "=" * 86)
    print("THE JOIN -- ranked by p90 move MINUS round-trip cost, which is the")
    print("only ordering that answers 'can the money button pay at this hold?'")
    print("=" * 86)
    print("%-16s %8s %8s %8s %9s %10s %9s  %s"
          % ("symbol", "p50 mv", "p90 mv", "mfe p90", "clip$", "rt cost",
             "p90-COST", "n"))

    joined: List[dict] = []
    for symbol in kept:
        row = dict(kept[symbol])
        row["symbol"] = symbol
        cost = costs.get(symbol)
        row["clip_usd"] = clips.get(symbol, args.default_clip)
        row["round_trip_cost"] = cost["round_trip_cost"] if cost else None
        row["edge_p90"] = (row["p90"] - cost["round_trip_cost"]
                           if cost else None)
        joined.append(row)

    quoted = [r for r in joined if r["edge_p90"] is not None]
    quoted.sort(key=lambda r: r["edge_p90"], reverse=True)
    for r in quoted:
        print("%-16s %7.3f%% %7.3f%% %7.3f%% %9.2f %9.3f%% %+8.3f%%  %d"
              % (r["symbol"], 100 * r["p50"], 100 * r["p90"],
                 100 * r["mfe_p90"], r["clip_usd"],
                 100 * r["round_trip_cost"], 100 * r["edge_p90"],
                 r["n_windows"]))
    for r in sorted(joined, key=lambda r: -r["p90"]):
        if r["edge_p90"] is None:
            print("%-16s %7.3f%% %7.3f%% %7.3f%% %9.2f %9s %8s  %d   %s"
                  % (r["symbol"], 100 * r["p50"], 100 * r["p90"],
                     100 * r["mfe_p90"], r["clip_usd"], "UNQUOTED", "--",
                     r["n_windows"], unquoted.get(r["symbol"], "not reached")))

    # ---- the answer -------------------------------------------------------
    payers = [r for r in quoted if r["edge_p90"] > 0]
    print("\n" + "=" * 86)
    print("HOW MANY SYMBOLS HAVE BOTH A REAL TICK SERIES AND p90 MOVE ABOVE")
    print("THEIR OWN ROUND-TRIP COST:  %d of %d quoted (%d survived hygiene)"
          % (len(payers), len(quoted), len(kept)))
    print("=" * 86)
    if not quoted:
        # An honest negative needs a measurement behind it. Zero payers out of
        # zero quotes is not a negative result, it is a failed run, and saying
        # "the route is closed" here would be inventing the answer.
        print("UNMEASURED. No symbol was quoted, so the COST column is empty and")
        print("this run says NOTHING about whether the money button can pay.")
        print("Do not quote the zero above as a negative result -- rerun with")
        print("venues reachable. %d symbols are waiting on a quote." % len(kept))
    elif not payers:
        print("ZERO of the %d symbols that could be PRICED clears its own cost"
              % len(quoted))
        print("at p90 over a %g-%g minute hold."
              % (args.lo_min, args.hi_min))
        print()
        print("SAY WHAT THAT DOES AND DOES NOT CLOSE. It closes the money button")
        print("on those %d symbols, on arithmetic, without waiting another 62" % len(quoted))
        print("days for ghost evidence to agree. It closes NOTHING about the other")
        print("%d, and the reason they are missing is OURS, not the market's:"
              % (len(kept) - len(quoted)))
        blocked = sum(1 for r in unquoted.values() if "unstable" in r
                      or "unpriceable" in r)
        missing = sum(1 for r in unquoted.values() if "no address" in r)
        print("  %2d could not be priced because the ROUTE is broken -- the quote"
              % blocked)
        print("     moves between repeats of the same question. That is a router")
        print("     defect to fix, and reporting it as a high cost would have")
        print("     retired symbols that may be the cheapest ones we have.")
        print("  %2d have no address in data/token_addresses.json at all." % missing)
        print()
        print("Coverage is %d of %d. An honest negative needs the denominator"
              % (len(quoted), len(kept)))
        print("said out loud, and this one is not yet a verdict on the route.")
    else:
        print("The money-button universe should contain these and nothing else:")
        for r in payers:
            print("  %-16s p90 move %.3f%% - cost %.3f%% = %+.3f%% per round trip"
                  % (r["symbol"], 100 * r["p90"], 100 * r["round_trip_cost"],
                     100 * r["edge_p90"]))
        print("\n  Read this as an UPPER bound on a p90 outcome, not an")
        print("  expectancy: p90 is the good tenth of windows, the exit is")
        print("  deterministic and unconditional, and no signal has been")
        print("  applied. A positive number here is permission to test, not")
        print("  a profit. The p50 column is what a random entry gets.")
    print("\nThis run changed NO production symbol universe. The decision is")
    print("the operator's, and the arithmetic above is what it should be made on.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "generated_ts": time.time(),
            "hold_min": [args.lo_min, args.hi_min],
            "hours": args.hours,
            "excluded": {k: [s for s, _ in v] for k, v in excluded.items()},
            "unquoted": unquoted,
            "rows": joined,
        }, indent=2, default=str), encoding="utf-8")
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
