"""Which strategy, if any, produces candidates on a token we can afford?

The live arm points at ``money_button``. Pass 443 measured what that buys:
over 168h it was offered 32,916 evaluations, 24,275 of them got past the feed
gates, and it produced 2 candidates -- 0.01%. Both were STONKFLY-USDC, a
0.70%-round-trip token, and the one that resolved lost -3.3284% net and
-2.6284% GROSS. So the live arm is pointed at a strategy that almost never
fires, and fires on a token whose cost makes the trade unwinnable anyway.

That is an argument for RE-POINTING the arm, but only if something else in
the population has supply on a cheap token. Nothing measured that, because
nothing could: ``money_button_gate_census.py`` hard-coded one strategy. This
script is that census with the strategy taken as an argument, run over the
whole registry, with the symbols split by what their round trip actually
costs.

WHAT "CHEAP" MEANS HERE, AND WHY THE ANSWER IS ALREADY UNCOMFORTABLE
--------------------------------------------------------------------
The cohort bar is a <=0.20% round trip, read from ``services.round_trip_cost``
-- the same table the production gate reads, so this cannot drift from it.
As of 2026-09-19 that table prices exactly three bases under 20bp: WETH, ETH
and USDC, all at 11bp, and all three carry the comment ``n=0 non-ghost --
ASSUMED``. Every base with live fills behind it is DEARER: CBBTC 28bp,
AERO 68bp, and the 70bp unlisted default.

So the CHEAP column is, today, entirely hypothetical. It is reported with
that label attached rather than silently, because a strategy with good supply
in the cheap cohort has supply on a price nobody has ever paid.

THE OTHER TWO LIMITS, STATED RATHER THAN WORKED AROUND
------------------------------------------------------
1. CANDIDATE SUPPLY IS MEASURABLE FOR EVERY STRATEGY. The per-guard REASON
   column is not: it reads ``strat.last_decline``, and money_button is the
   only strategy in the tree that sets it. A strategy with zero cheap supply
   and no reason column is "supply measured, cause unknown" -- which is not
   the same as "no cause", and the table says so in the ``why`` field.

2. HORIZON VARIANTS (``@5h`` .. ``@1w``) DO NOT READ THE WINDOW THEY ARE
   HANDED. ``trading.strategies.horizons`` fetches and resamples stored
   history itself. Their supply is still measured -- the evaluation is real
   -- but it is not measured on the same tick series as the base strategies,
   so their per-1000 rate is NOT comparable to a base strategy's and they are
   flagged ``H`` in the table.

Nothing here writes. No ledger, no registry, no approval. This supplies the
number; the re-pointing decision is the operator's.

Usage:
    python -X utf8 scripts/population_candidate_supply.py
    python -X utf8 scripts/population_candidate_supply.py --hours 24
    python -X utf8 scripts/population_candidate_supply.py --stride 4
    python -X utf8 scripts/population_candidate_supply.py --only money_button
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB_PATH = "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db")

#: Gates that mean the lane never got to look at the market. Only
#: money_button reports these by name, so the split is only populated for it;
#: kept identical to money_button_gate_census.py so the two agree.
FEED_GATES = {"no_price", "too_few_samples", "window_too_short", "feed_frozen"}

#: The cohort bar, as a fraction of notional. 0.0020 == 0.20% round trip.
CHEAP_MAX = float(os.getenv("CANDIDATE_SUPPLY_CHEAP_MAX", "0.0020"))

#: Widest wall-clock window any base strategy is given. Every strategy sees
#: the SAME window so the supply numbers are comparable; a strategy that
#: wants less takes the tail of it.
DEFAULT_LOOKBACK_SEC = 3600.0


class _State:
    """Stands in for RouteState.

    Lifted verbatim from money_button_gate_census.py: the gates read only
    ``.samples``, but ``make_candidate`` refuses any pair that is not
    stable-quoted, so the token fields must be set or every genuine fire comes
    back as a reasonless decline.
    """

    __slots__ = ("samples", "symbol", "base_token", "quote_token")

    def __init__(self, samples, symbol: str):
        self.samples = samples
        self.symbol = symbol
        base, _, quote = symbol.partition("-")
        self.base_token = base
        self.quote_token = quote or "USDC"


def load_ticks(hours: float) -> Dict[str, List[Tuple[float, float, float]]]:
    conn = sqlite3.connect(DB_PATH, uri=True)
    try:
        newest = list(conn.execute("SELECT MAX(ts) FROM market_stream"))[0][0]
        if not newest:
            return {}
        cutoff = float(newest) - hours * 3600.0
        rows = conn.execute(
            "SELECT symbol, ts, price, COALESCE(volume, 0.0) FROM market_stream "
            "WHERE ts > ? AND price > 0 ORDER BY symbol, ts",
            (cutoff,),
        )
        by_symbol: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
        for symbol, ts, price, volume in rows:
            by_symbol[str(symbol).upper()].append(
                (float(ts), float(price), float(volume)))
        return dict(by_symbol)
    finally:
        conn.close()


def _ghost_only_ids() -> set:
    """Strategy ids with no live branch, read from the ledger's own list."""
    try:
        from trading.strategies import ledger as _ledger

        raw = os.getenv("GHOST_ONLY_STRATEGIES",
                        getattr(_ledger, "_GHOST_ONLY_DEFAULT", ""))
    except Exception:  # noqa: BLE001 - never let this break the census
        raw = "atf_static_scout"
    return {p.strip() for p in str(raw).replace(",", " ").split() if p.strip()}


class Supply:
    """Per-strategy tally, split by cohort."""

    __slots__ = ("evals", "fires", "symbols", "declines", "feed_blocked", "errors")

    def __init__(self) -> None:
        self.evals = {"cheap": 0, "dearer": 0}
        self.fires = {"cheap": 0, "dearer": 0}
        #: WHICH symbols fired, kept PER COHORT. A single pooled counter here
        #: is actively misleading: the first run of this script listed
        #: donchian_breakout@1w under cheap supply with "AERO-USDC x43" beside
        #: it, because the pooled counter showed the strategy's overall mix
        #: rather than its cheap mix. AERO is a 68bp token. A reader would
        #: have taken that as evidence for a cheap-token trade.
        self.symbols: Dict[str, Counter] = {"cheap": Counter(), "dearer": Counter()}
        self.declines: Counter = Counter()
        self.feed_blocked = {"cheap": 0, "dearer": 0}
        self.errors = 0

    @property
    def total_evals(self) -> int:
        return self.evals["cheap"] + self.evals["dearer"]

    @property
    def total_fires(self) -> int:
        return self.fires["cheap"] + self.fires["dearer"]

    def per_k(self, cohort: Optional[str] = None) -> float:
        if cohort is None:
            n, f = self.total_evals, self.total_fires
        else:
            n, f = self.evals[cohort], self.fires[cohort]
        return (f / n * 1000.0) if n else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--fee", type=float, default=None,
                    help="one-way fee rate; default reads TRADE_FEE_RATE or 0.003")
    ap.add_argument("--stride", type=int, default=1,
                    help="evaluate every Nth tick; >1 subsamples the walk "
                         "(rates stay valid, absolute counts scale down)")
    ap.add_argument("--lookback-sec", type=float, default=DEFAULT_LOOKBACK_SEC,
                    help="wall-clock window every strategy is handed")
    ap.add_argument("--only", default=None,
                    help="comma-separated strategy_ids; default is the whole "
                         "registry")
    ap.add_argument("--no-horizons", action="store_true",
                    help="skip the @5h..@1w variants (they re-read stored "
                         "history and are the slow part)")
    args = ap.parse_args()

    from services.round_trip_cost import (TOKEN_ROUND_TRIP,
                                          default_token_round_trip,
                                          token_round_trip_cost)
    from trading.strategies import build_default_registry
    from trading.strategies.base import StrategyContext

    fee = args.fee if args.fee is not None else float(
        os.getenv("TRADE_FEE_RATE", "0.003"))

    registry = build_default_registry()
    population = registry.all()
    if args.only:
        wanted = {p.strip() for p in args.only.split(",") if p.strip()}
        population = [s for s in population if s.strategy_id in wanted]
    if args.no_horizons:
        population = [s for s in population if "@" not in s.strategy_id]
    if not population:
        print("no strategies selected.")
        return 1

    ghost_only = _ghost_only_ids()

    by_symbol = load_ticks(args.hours)
    if not by_symbol:
        print("no ticks in the window; nothing to census.")
        return 1

    # Cohort every symbol ONCE, from the production table, and record whether
    # the price behind it is measured or assumed. A cheap cohort built out of
    # assumed rows is a different claim from one built out of fills.
    cohort_of: Dict[str, str] = {}
    cost_of: Dict[str, float] = {}
    evidence_of: Dict[str, str] = {}
    for symbol in by_symbol:
        cost = token_round_trip_cost(symbol)
        cost_of[symbol] = cost
        cohort_of[symbol] = "cheap" if cost <= CHEAP_MAX else "dearer"
        base = symbol.split("-")[0].strip().upper()
        evidence_of[symbol] = "listed" if base in TOKEN_ROUND_TRIP else "default"

    cheap_symbols = sorted(s for s in by_symbol if cohort_of[s] == "cheap")

    tally: Dict[str, Supply] = {s.strategy_id: Supply() for s in population}

    print("population candidate supply -- %.0fh of stored ticks, one-way fee "
          "%.4f, stride %d" % (args.hours, fee, args.stride))
    print("%d strategies x %d symbols, %d ticks in window"
          % (len(population), len(by_symbol),
             sum(len(v) for v in by_symbol.values())))
    print()
    print("COHORT BAR: round trip <= %.2f%% is CHEAP, from "
          "services.round_trip_cost (the production table)" % (CHEAP_MAX * 100))
    print("  cheap symbols in this window (%d): %s"
          % (len(cheap_symbols), ", ".join(cheap_symbols) or "NONE"))
    measured = [b for b in TOKEN_ROUND_TRIP
                if float(TOKEN_ROUND_TRIP[b]) <= CHEAP_MAX]
    print("  bases the table prices under the bar: %s"
          % (", ".join(sorted(measured)) or "NONE"))
    print("  unlisted bases fall through to %.2f%% -- DEARER by construction"
          % (default_token_round_trip() * 100))
    print()

    t0 = time.time()
    for strat in population:
        sid = strat.strategy_id
        sup = tally[sid]
        lookback = float(getattr(strat, "LOOKBACK_SEC", 0) or args.lookback_sec)
        for symbol, ticks in by_symbol.items():
            cohort = cohort_of[symbol]
            start = 0
            for end in range(0, len(ticks), args.stride):
                cutoff = ticks[end][0] - lookback
                while ticks[start][0] < cutoff:
                    start += 1
                window = ticks[start:end + 1]
                ctx = StrategyContext(
                    chain="base",
                    last_price=window[-1][1],
                    last_volume=window[-1][2],
                    fee_rate=fee,
                    available_quote=10.0,
                    available_base=0.0,
                )
                try:
                    result = strat.evaluate(_State(window, symbol), ctx)
                except Exception:  # noqa: BLE001 - a broken strategy is data
                    sup.errors += 1
                    continue
                sup.evals[cohort] += 1
                if result is None:
                    reason = getattr(strat, "last_decline", None)
                    if reason:
                        sup.declines[reason] += 1
                        if reason in FEED_GATES:
                            sup.feed_blocked[cohort] += 1
                else:
                    sup.fires[cohort] += 1
                    sup.symbols[cohort][symbol] += 1
        print("  ... %-28s %7d evals, %4d candidates  (%.0fs)"
              % (sid, sup.total_evals, sup.total_fires, time.time() - t0),
              flush=True)

    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("CANDIDATES PER 1,000 EVALUATIONS, BY COHORT")
    print("=" * 96)
    print("  H = horizon variant: reads resampled stored history, NOT the "
          "window handed to it.")
    print("      Its rate is real but NOT comparable to a base strategy's.")
    print("  G = ghost-only: no live branch exists, so its supply cannot be "
          "spent. Excluded from")
    print("      the recommendation.")
    print("  ?  in `why` = this strategy does not set last_decline, so the "
          "CAUSE of a zero is")
    print("      UNKNOWN, not absent. money_button is the only strategy in "
          "the tree that sets it.")
    print()
    print("%-28s %2s %9s %8s %9s %8s   %s"
          % ("strategy", "", "cheap/1k", "(evals)", "dear/1k", "(evals)", "why"))
    print("-" * 96)

    rows = sorted(population,
                  key=lambda s: (-tally[s.strategy_id].per_k("cheap")
                                 if tally[s.strategy_id].evals["cheap"] else 0.0,
                                 -tally[s.strategy_id].per_k("dearer")))
    any_cheap_supply = False
    for strat in rows:
        sid = strat.strategy_id
        sup = tally[sid]
        flags = ""
        if "@" in sid:
            flags += "H"
        if sid in ghost_only:
            flags += "G"
        if sup.declines:
            top = sup.declines.most_common(1)[0]
            why = "%s %.0f%%" % (top[0], top[1] / max(1, sup.total_evals) * 100)
        else:
            why = "?"
        if sup.errors:
            why += "  [%d raised]" % sup.errors
        if sup.fires["cheap"] > 0:
            any_cheap_supply = True
        print("%-28s %2s %9.3f %8d %9.3f %8d   %s"
              % (sid, flags,
                 sup.per_k("cheap"), sup.evals["cheap"],
                 sup.per_k("dearer"), sup.evals["dearer"],
                 why))

    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("THE ANSWER")
    print("=" * 96)
    live_capable = [s for s in population if s.strategy_id not in ghost_only]
    cheap_producers = [s for s in live_capable
                       if tally[s.strategy_id].fires["cheap"] > 0]
    if not cheap_symbols:
        print("NO symbol in this %.0fh window prices under %.2f%%. The cheap "
              "cohort is EMPTY," % (args.hours, CHEAP_MAX * 100))
        print("so no strategy can have supply in it and NO RE-POINTING OF THE "
              "LIVE ARM CAN WORK on")
        print("cost grounds alone. The next route is a cheaper token, not a "
              "different strategy.")
    elif not cheap_producers:
        print("NO live-capable strategy produced a single candidate on a cheap "
              "token in %.0fh." % args.hours)
        print("This is an HONEST NEGATIVE and it is the result: re-pointing the "
              "live arm at any")
        print("strategy in this population moves nothing, because there is "
              "nothing to point it at.")
        print("The next route is PREDICATE DISCOVERY -- a rule that fires on "
              "the cheap names at all")
        print("-- not a re-ranking of rules that do not.")
    else:
        print("Live-capable strategies WITH cheap-token supply, best first:")
        for strat in sorted(cheap_producers,
                            key=lambda s: -tally[s.strategy_id].per_k("cheap")):
            sup = tally[strat.strategy_id]
            top = ", ".join("%s x%d" % (sym, n)
                            for sym, n in sup.symbols["cheap"].most_common(4))
            print("  %-28s %7.3f /1k on %d cheap evals   CHEAP FIRES: %s"
                  % (strat.strategy_id, sup.per_k("cheap"),
                     sup.evals["cheap"], top))
        print()
        print("CAVEAT THAT APPLIES TO EVERY ROW ABOVE: the cheap cohort is "
              "priced from table rows")
        print("marked `n=0 non-ghost -- ASSUMED`. Supply on a cheap token is "
              "supply at a price this")
        print("repo has never paid. Confirm the cost with one live round trip "
              "before spending on it.")

    excluded = [s.strategy_id for s in population if s.strategy_id in ghost_only]
    if excluded:
        print()
        print("EXCLUDED from the recommendation (ghost-only, no live branch): %s"
              % ", ".join(sorted(excluded)))
    print()
    print("NOTHING WAS WRITTEN. No ledger, no registry, no approval change.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
