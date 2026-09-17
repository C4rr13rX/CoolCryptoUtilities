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

TWO MODES, AND THE SECOND ONE IS THE USEFUL ONE ON BASE.

  DEFAULT -- routing dispersion and arbitrage, as described above. It needs at
  least TWO venues to compare. On base there is exactly ONE (see
  chain_support), so this mode is UNDEFINED here and says so rather than
  reporting a spread of zero.

  --round-trip-census -- measures COST, which needs only one venue. It quotes
  USDC -> token -> USDC for every symbol the book actually holds, on a clip
  grid taken from the book's own notional distribution, and writes
  data/round_trip_quote_census_<UTC stamp>.json. That artifact is the ONE
  source for scripts/deep_pool_reprice.py and
  scripts/tradeable_book_repricing.py, which used to hardcode a six-symbol
  literal each.

  A symbol that cannot be quoted is reported UNQUOTABLE and charged NOTHING.
  There is no default rate anywhere in this file. If nothing can be quoted,
  "the census answers on zero symbols" is the result and it is a complete one.

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
import collections
import concurrent.futures
import datetime as _dt
import json
import os
import sqlite3
import subprocess
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


#: Where each venue keeps its per-chain router/config table, and the class that
#: quotes. ``services.quote_providers`` is a four-line re-export of exactly
#: these modules -- ``services.quote_providers.UniswapV3Local is
#: services.providers.uniswap_v3.UniswapV3Local`` is True (verified pass 457),
#: so the two import paths are the SAME object and neither is "the wrong one".
VENUES: Tuple[Tuple[str, str, str, str], ...] = (
    ("uniswap", "services.providers.uniswap_v3", "UniswapV3Local", "UNI_V3"),
    ("sushi", "services.providers.sushi_v2", "SushiV2Local", "SUSHI_V2_ROUTER"),
    ("camelot", "services.providers.camelot_v2", "CamelotV2Local", "CAMELOT_V2_ROUTER"),
)


def chain_support(chain: str) -> Dict[str, Tuple[bool, str]]:
    """Is each venue DEPLOYED AND CONFIGURED on this chain? name -> (ok, reason).

    This reads each provider's own per-chain router table. It deliberately does
    NOT test whether the Python object can be constructed, because every
    provider here constructs fine on every chain -- they all just store the
    web3 factory -- and then fail at quote time. A census that filters on
    constructibility therefore keeps three venues, spends its whole budget
    asking two of them, and reports "0 of 12 answered" as though the venues had
    gone quiet. Measured on base, pass 457:

        sushi    SUSHI_V2_ROUTER has only 'arbitrum'  -> not deployed here
        camelot  CAMELOT_V2_ROUTER['base'] is ''      -> no router configured
        uniswap  UNI_V3['base'] present               -> supported, and answers

    Both failures are returned by the provider in 0.00s, so neither is the
    TIMEOUT that pass 449 saw. That was uniswap's own fan-out; see quote_all.
    """
    ch = (chain or "").lower().strip()
    out: Dict[str, Tuple[bool, str]] = {}
    for name, module, _cls, table_name in VENUES:
        try:
            mod = __import__(module, fromlist=[table_name])
        except Exception as exc:  # noqa: BLE001
            out[name] = (False, "module %s will not import (%s)" % (module, exc))
            continue
        table = getattr(mod, table_name, None)
        if not isinstance(table, dict):
            # No readable chain table: cannot prove it is unsupported, so allow
            # it. Unknown is not the same as absent.
            out[name] = (True, "no %s table to check; allowed" % table_name)
            continue
        if ch not in table:
            out[name] = (False, "%s has no '%s' key (chains: %s)"
                         % (table_name, ch, ", ".join(sorted(table)) or "none"))
            continue
        value = table[ch]
        if isinstance(value, str) and not value.strip():
            out[name] = (False, "%s['%s'] is empty -- no router configured"
                         % (table_name, ch))
            continue
        out[name] = (True, "%s['%s'] configured" % (table_name, ch))
    return out


def _providers(chain: str = "base",
               support: Optional[Dict[str, Tuple[bool, str]]] = None) -> Dict[str, Any]:
    """Every provider that can answer a quote ON THIS CHAIN, by name.

    Each takes a web3 FACTORY, exactly as services/swap_service.py:369-371
    builds them -- a provider holds no connection of its own, so reusing the
    bridge's is both correct and the only way to get the same RPC the live
    path would use.

    A venue that is not supported on this chain is SKIPPED HERE, by
    chain_support() above, rather than being carried into the quote loop to
    fail there. That is the difference between "this venue is not on base" and
    "this venue did not answer", and the census reports them as different
    things because they are different facts.
    """
    out: Dict[str, Any] = {}
    try:
        from router_wallet import UltraSwapBridge  # noqa: PLC0415
        # No key material: every call here is a QUOTE, and a read-only bridge
        # cannot sign even by accident. That is a safety property, not a
        # convenience -- this script must never be able to move money.
        bridge = UltraSwapBridge()
        factory = lambda ch: bridge._w3(ch)  # noqa: E731
        factory(chain)                       # fail loudly here, not per-quote
    except Exception as exc:  # noqa: BLE001
        print("routing census: cannot build a web3 factory (%s: %s)"
              % (type(exc).__name__, exc))
        return out

    if support is None:
        support = chain_support(chain)
    for name, module, cls, _table in VENUES:
        ok, _reason = support.get(name, (True, ""))
        if not ok:
            continue
        try:
            mod = __import__(module, fromlist=[cls])
            out[name] = getattr(mod, cls)(factory)
        except Exception:  # noqa: BLE001
            continue
    return out


#: Seconds any ONE venue gets to answer ONE quote before the census gives up on
#: it and moves on. Pass 449 recorded uniswap as "TIMEOUT" and concluded the
#: venues had gone quiet. The cause is not the network being down -- base
#: answers eth_blockNumber in 0.04s -- it is UniswapV3Local's own fan-out: with
#: no QUOTER_V1 configured on base it tries 4 fee tiers on
#: quoteExactInputSingle, then 4 more on quoteExactInput, then up to 2 mid
#: tokens x 4 x 4 = 32 two-hop paths, then estimate_gas. Up to 41 eth_calls for
#: ONE quote, and a pair with no pool at all pays the whole bill before
#: returning None. mainnet.base.org is the only Base endpoint that works
#: (router_wallet.py:166-187) and it answers 429 under that load. So the
#: deadline is not a nicety: without it one dead pair stalls the census.
#: A measured leg takes ~10s against mainnet.base.org when the endpoint is not
#: throttling us, so 25s looked generous and was not: it benched the only venue
#: on base and the census reported 0 of 18 symbols. 90s fits the full 41-call
#: fan-out with room for the endpoint's own backoff.
DEFAULT_QUOTE_DEADLINE_SEC = 90.0

#: After this many consecutive deadline breaches a venue is dropped for the
#: rest of the run. A venue that hangs once will hang again, and paying the
#: full deadline for it on every remaining pair is how a census runs out of
#: wall clock before it runs out of symbols.
#:
#: MEASURED CONSEQUENCE, and the reason the rule below exists. The first run of
#: this census benched uniswap after 3 breaches at a 25s deadline, having
#: ATTEMPTED 4 QUOTES IN TOTAL, and then reported all 10 addressed symbols as
#: UNRESOLVED. Base has ONE quoting venue, so benching it does not fall back to
#: another -- it ends the measurement. Re-probed by hand immediately afterwards,
#: the same legs answered in 10s and AERO round-tripped 0.0184% at $0.75. The
#: breaker turned a slow venue into a dead one and the run into a null result.
DEFAULT_VENUE_STRIKES = 3


class VenueBudget:
    """Per-venue deadline and strike count, shared across a whole run."""

    def __init__(self, deadline: float = DEFAULT_QUOTE_DEADLINE_SEC,
                 strikes: int = DEFAULT_VENUE_STRIKES) -> None:
        self.deadline = float(deadline)
        self.max_strikes = int(strikes)
        self.strikes: Dict[str, int] = collections.defaultdict(int)
        self.benched: Dict[str, str] = {}
        self.attempts: Dict[str, int] = collections.defaultdict(int)
        self.answers: Dict[str, int] = collections.defaultdict(int)
        self.timeouts: Dict[str, int] = collections.defaultdict(int)
        self.errors: Dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter)
        # One worker per venue name, so a thread left hanging on a stuck RPC
        # call cannot be handed the next quote for that venue.
        self._pools: Dict[str, concurrent.futures.ThreadPoolExecutor] = {}
        #: Venues that must never be benched, because benching them ends the
        #: measurement rather than falling back to another venue.
        self.protected: set = set()

    def protect(self, names: Sequence[str]) -> None:
        """Mark the last supported venue(s) as un-benchable.

        A circuit breaker exists to route AROUND a bad venue. With one venue
        there is nowhere to route, so tripping it converts "this endpoint is
        throttling us" into "no symbol has a cost" -- a null result caused
        entirely by the guard. Measured: see DEFAULT_VENUE_STRIKES.
        """
        if len(names) <= 1:
            self.protected |= set(names)

    def pool(self, name: str) -> concurrent.futures.ThreadPoolExecutor:
        if name not in self._pools:
            self._pools[name] = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="quote-%s" % name)
        return self._pools[name]

    def note_timeout(self, name: str) -> None:
        self.timeouts[name] += 1
        self.strikes[name] += 1
        # The stuck worker is abandoned deliberately, benched or not: a
        # synchronous web3 call cannot be cancelled, and the alternative --
        # queueing the next quote behind it on the same single worker -- makes
        # every later quote wait out the deadline too. The cost of abandoning
        # is that the zombie keeps spending RPC quota, so the deadline must be
        # generous enough that timeouts stay rare rather than routine.
        self._pools.pop(name, None)
        if name in self.protected:
            # Slow is reported as slow. It is not promoted to dead.
            return
        if self.strikes[name] >= self.max_strikes and name not in self.benched:
            self.benched[name] = ("%d consecutive quotes exceeded the %.0fs "
                                  "deadline" % (self.strikes[name], self.deadline))

    def note_answer(self, name: str) -> None:
        self.answers[name] += 1
        self.strikes[name] = 0

    def note_error(self, name: str, reason: str) -> None:
        self.errors[name][reason[:120]] += 1
        self.strikes[name] = 0

    def report(self) -> Dict[str, Any]:
        names = set(self.attempts) | set(self.benched)
        return {
            n: {
                "attempted": self.attempts[n],
                "answered": self.answers[n],
                "timed_out": self.timeouts[n],
                "benched": self.benched.get(n),
                "errors": dict(self.errors[n]),
            }
            for n in sorted(names)
        }


def quote_one(provider: Any, chain: str, token_in: str, token_out: str,
              amount_in: int, slippage_bps: int, name: str,
              budget: VenueBudget) -> Tuple[Optional[int], Optional[str]]:
    """One venue, one quote, under a DEADLINE. (buyAmount or None, reason).

    The provider call is synchronous web3 and cannot be cancelled, so it is run
    on a single-worker thread and abandoned if it overruns. Abandoning a thread
    is not tidy; stalling the entire census on one unreachable pool is worse,
    and that is the failure this census actually suffered.
    """
    if name in budget.benched:
        return None, "benched: %s" % budget.benched[name]
    budget.attempts[name] += 1

    def _call() -> Any:
        # A recipient is required by the BUILD half of quote_and_build even
        # though only the quote half is read here. The address below cannot
        # receive anything and cannot sign, so a calldata blob built for it is
        # inert. This script must never be able to move money.
        return provider.quote_and_build(
            chain, token_in, token_out, int(amount_in),
            slippage_bps=slippage_bps, recipient=QUOTE_ONLY_RECIPIENT)

    try:
        future = budget.pool(name).submit(_call)
    except RuntimeError as exc:  # pool already shut down
        return None, "no worker (%s)" % exc
    try:
        reply = future.result(timeout=budget.deadline)
    except concurrent.futures.TimeoutError:
        future.cancel()
        budget.note_timeout(name)
        return None, "DEADLINE %.0fs exceeded" % budget.deadline
    except Exception as exc:  # noqa: BLE001
        budget.note_error(name, "%s: %s" % (type(exc).__name__, exc))
        return None, "%s: %s" % (type(exc).__name__, exc)

    if not isinstance(reply, dict):
        budget.note_error(name, "non-dict reply")
        return None, "non-dict reply"
    if reply.get("__error__"):
        budget.note_error(name, str(reply["__error__"]))
        return None, str(reply["__error__"])
    try:
        amount = int(reply.get("buyAmount") or 0)
    except (TypeError, ValueError):
        budget.note_error(name, "unparseable buyAmount")
        return None, "unparseable buyAmount"
    if amount <= 0:
        budget.note_error(name, "buyAmount <= 0")
        return None, "buyAmount <= 0"
    budget.note_answer(name)
    return amount, None


def quote_all(providers: Dict[str, Any], chain: str, token_in: str,
              token_out: str, amount_in: int,
              slippage_bps: int = 100,
              budget: Optional[VenueBudget] = None
              ) -> Tuple[Dict[str, Optional[int]], float]:
    """(venue -> buyAmount or None, seconds spanned).

    The span is returned because a spread measured across a moving market is a
    statement about the clock, not about the venues.
    """
    budget = budget if budget is not None else VenueBudget()
    out: Dict[str, Optional[int]] = {}
    started = time.time()
    for name, provider in providers.items():
        amount, _reason = quote_one(provider, chain, token_in, token_out,
                                    amount_in, slippage_bps, name, budget)
        out[name] = amount
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
                         slippage_bps: int = 100,
                         budget: Optional[VenueBudget] = None) -> Optional[dict]:
    """Buy B on one venue, sell B back on another. Does the LOOP pay?

    A price gap is not arbitrage. This quotes BOTH legs at the same size and
    charges BOTH legs' cost and gas, because the only number worth having is
    what survives the whole round trip. If the loop does not close, it is a
    gap, and gaps are the normal state of a market.
    """
    budget = budget if budget is not None else VenueBudget()
    leg_out, span_a = quote_all(providers, chain, token_a, token_b,
                                amount_in, slippage_bps, budget)
    buys = {k: v for k, v in leg_out.items() if v}
    if not buys:
        return None

    buy_venue = max(buys, key=lambda k: buys[k])
    got_b = buys[buy_venue]

    # Sell the SAME quantity back, and ask every venue including the one we
    # bought on -- a same-venue loop is the honest null: it must lose exactly
    # the two legs' cost, and if it shows a profit the measurement is wrong.
    leg_back, span_b = quote_all(providers, chain, token_b, token_a,
                                 got_b, slippage_bps, budget)
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


# =====================================================================
# ROUND-TRIP COST CENSUS -- the table the re-pricing scripts read
# =====================================================================
#
# WHY A SECOND MODE. scripts/deep_pool_reprice.py and
# scripts/tradeable_book_repricing.py both hardcoded a six-symbol literal
# (MEASURED_ROUND_TRIP_PCT) quoted at $2/$10/$50. Two facts make that table the
# wrong instrument:
#
#   * COVERAGE. Those six symbols are 68 of 237 rows in trade_outcomes. The
#     other 159 rows across 45 symbols were charged NOTHING -- correct, but
#     it means the re-priced book is a statement about a quarter of itself.
#   * THE CLIP. $2/$10/$50 sits mostly ABOVE the book. The ghost book's
#     notional runs p05 $0.06, median $2.00, p95 $5.50, max $6.00, and the
#     LIVE median is $0.75. Two of the three quoted clips are outside the
#     data entirely, and gas alone is 2*gas/notional -- ~4bp at $10 and ~53bp
#     at $0.75. A 13x swing. The clip IS the cost, so the grid has to be fine
#     and it has to span what was actually traded.
#
# THE ONE RULE THIS MODE EXISTS TO KEEP. A symbol the router cannot quote is
# reported UNQUOTABLE and charged NOTHING. There is no default rate, no nearest
# neighbour and no chain-wide average, because this repo has already shipped
# the bug where round_trip_cost measured its own default (2026-09-04). If every
# symbol comes back unquotable then "the census answers on zero symbols" is the
# result, and it is a complete one.

#: Clip sizes in USD. Deliberately FINE and deliberately low: this spans the
#: ghost book's own notional distribution (p05 $0.06 to max $6.00) with $10
#: kept only so the new table can be compared against the legacy literal at
#: the one clip they share.
DEFAULT_CLIP_GRID_USD: Tuple[float, ...] = (
    0.10, 0.25, 0.50, 0.75, 1.00, 2.00, 4.00, 6.00, 10.00)

#: Round-trip slippage the LIVE lane actually measured, in basis points, that
#: no quote in this file carries. Travels INSIDE the artifact as a field.
#:
#: TWO NUMBERS, BOTH MEASURED, AND THE SPREAD BETWEEN THEM IS THE POINT. The
#: live fills give 13.5bp pooled across 38 legs, and a separate read at the
#: $0.75 clip gives 27.8bp. They do not disagree -- they are the same quantity
#: at different sizes, which is exactly the clip dependence this whole census
#: exists to expose. Pooling them into one figure would hide it, so both are
#: carried and the artifact says the range rather than a point.
LIVE_MEASURED_RT_SLIPPAGE_BP = 13.5
LIVE_MEASURED_RT_SLIPPAGE_AT_75C_BP = 27.8

#: USDC has 6 decimals on base. The sell leg needs no decimals at all: it
#: sells back exactly the token amount the buy leg quoted.
USDC_DECIMALS = 6

#: The venue the census instrument actually asks. SameTierQuoter talks to
#: Uniswap V3's QuoterV2 directly, so every answer in this mode is uniswap's.
#: Named rather than implied because the census must report a VENUE COUNT, and
#: a count implied by "which code path ran" is not a reported fact.
CENSUS_VENUE = "uniswap"

#: Why a symbol ended up with no rate. These are DIFFERENT FACTS and the census
#: must not blur them: an address the repo never recorded is a gap in this
#: repo, while a pool that prices nothing is a fact about the chain, and a
#: throttled endpoint is a fact about neither.
UNQUOTABLE_CAUSES: Dict[str, str] = {
    "no_address_in_book": "data/token_addresses.json has no address for this "
                          "symbol on this chain. NOT a statement about the "
                          "pool -- the census never got to ask. Fixable here, "
                          "but NOT by resolving the ticker: this repo has "
                          "already been bitten by ticker squatting, and a "
                          "wrong address would produce a confident cost for "
                          "the wrong token.",
    "no_plausible_tier": "every Uniswap V3 fee tier either had no direct "
                         "USDC<->token pool or round-tripped BELOW its own "
                         "2*fee floor, which is a broken measurement rather "
                         "than a cheap pool.",
    "tier_priced_nothing": "a tier was chosen at the probe clip but priced "
                           "nothing anywhere on the clip grid.",
    "endpoint_throttled": "the only working Base endpoint answered 429 after "
                          "every retry. A throttle is not a fact about a pool, "
                          "so this is UNRESOLVED, never UNQUOTABLE.",
}


def book_rows(db_path: str) -> List[Dict[str, Any]]:
    """Every trade_outcomes row, with its notional and mode, from the store."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute("SELECT symbol, quantity, entry_price, details "
                          "FROM trade_outcomes")
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            try:
                meta = json.loads(r["details"] or "{}")
            except (TypeError, ValueError):
                meta = {}
            try:
                notional = abs(float(r["quantity"] or 0.0)
                               * float(r["entry_price"] or 0.0))
            except (TypeError, ValueError):
                notional = 0.0
            sym = (r["symbol"] or "").upper().strip()
            out.append({
                "symbol": sym,
                "base": sym.split("-", 1)[0] if "-" in sym else sym,
                "notional": notional,
                "mode": meta.get("mode"),
            })
        return out
    finally:
        con.close()


def clip_distribution(notionals: Sequence[float]) -> Dict[str, Any]:
    """Percentiles of the notional actually traded. Reported BEFORE any quote.

    The grid is chosen from this, not the other way round. Quoting $10 and $50
    against a book whose maximum row is $6.00 measures a market this system has
    never traded in.
    """
    xs = sorted(float(n) for n in notionals if n and n > 0)
    if not xs:
        return {"n": 0}

    def _q(p: float) -> float:
        return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1))))) ]

    return {
        "n": len(xs), "min": xs[0], "max": xs[-1],
        "mean": sum(xs) / len(xs),
        "p05": _q(0.05), "p10": _q(0.10), "p25": _q(0.25),
        "median": _q(0.50), "p75": _q(0.75), "p90": _q(0.90),
        "p95": _q(0.95),
    }


def _token_book(chain: str) -> Dict[str, str]:
    """base symbol -> address, from data/token_addresses.json."""
    try:
        doc = json.loads((ROOT / "data" / "token_addresses.json")
                         .read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    chain_book = doc.get(chain) or {}
    out: Dict[str, str] = {}
    for key, entry in chain_book.items():
        address = entry if isinstance(entry, str) else (
            entry.get("address") if isinstance(entry, dict) else None)
        if isinstance(address, str) and address.startswith("0x"):
            out[str(key).upper().strip()] = address
    return out


# round_trip_quote() -- a provider-based round trip that asked every venue and
# took the best answer on each leg -- was REMOVED on pass 457 rather than left
# here unused. It could not measure a round-trip cost: taking the best venue
# and tier per leg independently crosses two pools, which is what produced the
# negative AERO costs. SameTierQuoter below replaces it and forces both legs
# through one pool. The provider path survives only for the DISPERSION mode,
# where comparing venues against each other is the whole point.


#: Uniswap V3 fee tiers, in hundredths of a bip. 3000 == 0.30%.
UNIV3_FEE_TIERS: Tuple[int, ...] = (100, 500, 3000, 10000)


class RateLimited(Exception):
    """The endpoint answered 429 and kept doing so after every retry."""


class SameTierQuoter:
    """QuoterV2, asked directly, with BOTH LEGS ON ONE FEE TIER.

    WHY NOT UniswapV3Local. Two faults make it the wrong instrument for a
    round-trip COST, and both were measured on pass 457:

      1. IT CROSSES POOLS. quote_and_build picks the tier with the best output
         on EACH leg independently. Buying on the 0.30% pool and selling on the
         0.05% pool is not a round trip, it is two different markets, and the
         difference between them shows up as cost. Measured consequence: the
         census printed AERO round trips of -0.0464%, -0.0236%, -0.0982% and
         -0.0475%. A NEGATIVE round-trip cost is free money, which does not
         exist, and it is the same pathology pass 452 cited when it RETRACTED
         the earlier AERO quote sweep as unsound. Same tier on both legs is
         what makes the number a round trip.

      2. IT SWALLOWS 429. Every per-tier exception is caught and discarded, so
         a rate-limited endpoint returns "UniswapV3: no viable pool (direct or
         2-hop)" -- identical to a pool that does not exist. mainnet.base.org
         is the only working Base endpoint (router_wallet.py:166-187) and it
         DOES answer 429 under this census's load; verified verbatim:
         "HTTPError: 429 Client Error: Too Many Requests for url:
         https://mainnet.base.org/". Filing that as UNQUOTABLE would turn a
         throttle into a permanent fact about a pool, so this class retries
         with backoff and raises RateLimited rather than returning a verdict.

    It also skips estimate_gas and the 32-call two-hop search, which is what
    made one quote cost up to 41 eth_calls.
    """

    #: Minimum seconds between ANY two eth_calls. Measured on mainnet.base.org
    #: (pass 457), 8 identical quotes per spacing:
    #:
    #:     0.0s apart -> 5 ok, 3x HTTP 429
    #:     1.0s apart -> 4 ok, 4x HTTP 429
    #:     2.0s apart -> 8 ok, 0x HTTP 429
    #:
    #: So ~0.5 calls/sec is the sustainable rate on the only Base endpoint that
    #: works. This has to be enforced between EVERY call, not between symbols
    #: or clips: the two legs of one round trip are otherwise issued back to
    #: back and the second one eats the 429, which then costs an exponential
    #: backoff and eventually a false UNRESOLVED.
    MIN_CALL_INTERVAL_SEC = 2.0

    def __init__(self, w3: Any, chain: str, retries: int = 4,
                 backoff: float = 2.5,
                 min_interval: Optional[float] = None) -> None:
        from services.providers.uniswap_v3 import (  # noqa: PLC0415
            UNI_V3, _ABI_QUOTER_V2, _ABI_QUOTER_V2_EXACT_INPUT)
        from web3 import Web3  # noqa: PLC0415
        conf = UNI_V3[chain.lower().strip()]
        self._w3 = w3
        self._Web3 = Web3
        self.quoter = w3.eth.contract(
            Web3.to_checksum_address(conf["QUOTER_V2"]),
            abi=_ABI_QUOTER_V2 + _ABI_QUOTER_V2_EXACT_INPUT)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.min_interval = (self.MIN_CALL_INTERVAL_SEC if min_interval is None
                             else float(min_interval))
        self._last_call = 0.0
        self.calls = 0
        self.rate_limit_hits = 0

    def _pace(self) -> None:
        """Hold every call at least min_interval apart. See MIN_CALL_INTERVAL_SEC."""
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    @staticmethod
    def _path(token_in: str, fee: int, token_out: str) -> bytes:
        return (bytes.fromhex(token_in[2:]) + int(fee).to_bytes(3, "big")
                + bytes.fromhex(token_out[2:]))

    def quote(self, token_in: str, token_out: str, amount_in: int,
              fee: int) -> Optional[int]:
        """amountOut on ONE tier, or None when that pool cannot price it.

        Raises RateLimited when the endpoint is throttling, because that is
        not an answer about the pool.
        """
        t_in = self._Web3.to_checksum_address(token_in)
        t_out = self._Web3.to_checksum_address(token_out)
        path = self._path(t_in, int(fee), t_out)
        delay = self.backoff
        last_429 = None
        for attempt in range(self.retries + 1):
            try:
                self._pace()
                self.calls += 1
                out, *_ = self.quoter.functions.quoteExactInput(
                    path, int(amount_in)).call()
                return int(out) or None
            except Exception as exc:  # noqa: BLE001
                text = str(exc)
                if "429" in text or "Too Many Requests" in text:
                    self.rate_limit_hits += 1
                    last_429 = text[:160]
                    if attempt < self.retries:
                        time.sleep(delay)
                        delay *= 2.0
                        continue
                    raise RateLimited(last_429) from exc
                # A revert here means this tier has no pool for this pair, or
                # the pool cannot fill this size. That IS an answer.
                return None
        raise RateLimited(last_429 or "429")

    def round_trip(self, usdc: str, token: str, clip_usd: float, fee: int
                   ) -> Dict[str, Any]:
        """USDC -> token -> USDC on ONE tier. Cost as a % of notional.

        Carries its own PLAUSIBILITY GUARD. A round trip through a single pool
        must pay that pool's fee TWICE, so the cost cannot be below 2*fee. A
        reading under that floor is not a cheap pool, it is a broken
        measurement, and it is reported as implausible rather than published.
        """
        amount_in = int(round(clip_usd * 10 ** USDC_DECIMALS))
        out: Dict[str, Any] = {"clip_usd": clip_usd, "fee": fee,
                               "amount_in": amount_in}
        if amount_in <= 0:
            out["reason"] = "clip rounds to 0 USDC units"
            return out
        got = self.quote(usdc, token, amount_in, fee)
        if not got:
            out["reason"] = "no pool on tier %d for USDC->token" % fee
            return out
        back = self.quote(token, usdc, got, fee)
        if not back:
            out["reason"] = "no pool on tier %d for token->USDC" % fee
            return out
        cost_pct = 100.0 * (amount_in - back) / float(amount_in)
        floor_pct = 2.0 * fee / 10_000.0
        out.update({"token_out": str(got), "usdc_back": back,
                    "cost_pct": cost_pct, "floor_pct": floor_pct})
        if cost_pct < floor_pct - 1e-9:
            out["implausible"] = (
                "cost %.4f%% is below the 2*fee floor %.4f%% -- a round trip "
                "through one pool cannot cost less than twice its fee"
                % (cost_pct, floor_pct))
        return out


def measured_gas_context(db_path: str) -> Dict[str, Any]:
    """Gas per leg in USD, from the LIVE rows, and what it costs as a RATE.

    Gas is a dollar amount per leg, so as a fraction of notional it is a
    hyperbola: the same $0.002 is 4bp on a $10 clip and 54bp on a $0.75 one.
    That is why the clip grid above is fine and low, and why a cost table
    quoted only at $10 cannot be carried down to the size this book trades.

    It is measured, not modelled: on live rows fee_cost is GAS ONLY
    (fee-cost-is-gas-only-on-the-live-path), and all 22 live values are
    distinct -- none equals the flat model 0.004047 + 0.003187*notional, nor
    its bare constant. This is NOT the round trip's pool cost and must never
    be added to a 'quoted' rate as though it were the same quantity.
    """
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        fees: List[float] = []
        try:
            for r in con.execute("SELECT fee_cost, details FROM trade_outcomes"):
                try:
                    meta = json.loads(r["details"] or "{}")
                except (TypeError, ValueError):
                    continue
                if meta.get("mode") != "live":
                    continue
                try:
                    fees.append(float(r["fee_cost"] or 0.0))
                except (TypeError, ValueError):
                    continue
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, exc)}
    fees = sorted(f for f in fees if f > 0)
    if not fees:
        return {"n_live_rows": 0,
                "note": "no live rows carry a gas figure; nothing measured"}
    median_rt = fees[len(fees) // 2] if len(fees) % 2 else (
        (fees[len(fees) // 2 - 1] + fees[len(fees) // 2]) / 2.0)
    per_leg = median_rt / 2.0
    return {
        "n_live_rows": len(fees),
        "source": "trade_outcomes.fee_cost where details.mode == 'live'; "
                  "gas only on that path",
        "median_round_trip_usd": median_rt,
        "median_per_leg_usd": per_leg,
        "as_a_rate_bp_by_clip": {
            "$%g" % c: (10_000.0 * 2.0 * per_leg / c) for c in (
                0.10, 0.25, 0.50, 0.75, 1.00, 2.00, 4.00, 6.00, 10.00)},
        "warning": "GAS IS NOT IN THE 'quoted' RATES. Those are pool cost "
                   "(fee + impact) only. Do not add these two numbers without "
                   "deciding which clip you are charging at.",
    }


def _git_sha() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=str(ROOT), capture_output=True, text=True,
                              timeout=15).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


ARTIFACT_PREFIX = "round_trip_quote_census"


def artifact_path(stamp: Optional[str] = None) -> Path:
    stamp = stamp or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "data" / ("%s_%s.json" % (ARTIFACT_PREFIX, stamp))


def run_round_trip_census(args) -> int:
    """Quote the round trip for every symbol the BOOK actually holds."""
    rows = book_rows(args.db)
    if args.mode != "all":
        scoped = [r for r in rows if r["mode"] == args.mode]
    else:
        scoped = list(rows)

    counts = collections.Counter(r["base"] for r in scoped)
    qualifying = {s: n for s, n in counts.items() if n >= int(args.min_rows)}
    dist = clip_distribution([r["notional"] for r in scoped])

    print("=" * 78)
    print("ROUND-TRIP QUOTE CENSUS  [98bba3bf]   chain=%s  book=%s"
          % (args.chain, args.mode))
    print("=" * 78)
    print("1. THE CLIP DISTRIBUTION, FIRST, FROM THE DATA")
    print("   The grid is chosen from this. Gas is 2*gas/notional, so a clip")
    print("   read at $10 says nothing about a book whose median is $%.2f."
          % dist.get("median", 0.0))
    if dist.get("n"):
        print("   n=%d  min $%.4f  p05 $%.2f  p10 $%.2f  p25 $%.2f  MEDIAN $%.2f"
              % (dist["n"], dist["min"], dist["p05"], dist["p10"],
                 dist["p25"], dist["median"]))
        print("           p75 $%.2f  p90 $%.2f  p95 $%.2f  max $%.2f  mean $%.2f"
              % (dist["p75"], dist["p90"], dist["p95"], dist["max"], dist["mean"]))
    else:
        print("   NO ROWS WITH A NOTIONAL. Nothing to size a grid from.")
        return 2

    clips = [float(c) for c in (args.clips.split(",") if args.clips
                                else DEFAULT_CLIP_GRID_USD)]
    clips = sorted({c for c in clips if c > 0})
    print("   clip grid: %s" % ", ".join("$%g" % c for c in clips))
    print()

    print("2. SYMBOLS TO QUOTE -- from trade_outcomes, not a hardcoded list")
    print("   %d symbols hold >= %d rows, covering %d of %d rows (%.1f%%)."
          % (len(qualifying), int(args.min_rows), sum(qualifying.values()),
             len(scoped), 100.0 * sum(qualifying.values()) / max(1, len(scoped))))
    print()

    support = chain_support(args.chain)
    print("3. VENUE CHAIN SUPPORT -- tested on the router table, not on whether")
    print("   the object constructs. Every provider here constructs on every")
    print("   chain and then fails at quote time.")
    for name, (ok, reason) in sorted(support.items()):
        print("   %-8s %-12s %s" % (name, "SUPPORTED" if ok else "NOT ON CHAIN", reason))
    providers = _providers(args.chain, support)
    print("   venues carried into the quote loop: %s"
          % (", ".join(sorted(providers)) or "NONE"))
    print()

    book = _token_book(args.chain)
    usdc = book.get("USDC")
    if not providers or not usdc:
        print("   Cannot quote: %s." % ("no USDC address in the book"
                                        if providers else "no supported venue"))
        return 2

    budget = VenueBudget(args.deadline, args.strikes)
    budget.protect(sorted(providers))
    if budget.protected:
        print("   %s is the ONLY venue on this chain and is therefore exempt"
              % ", ".join(sorted(budget.protected)))
        print("   from the strike-out. A breaker with nowhere to route to does")
        print("   not protect the measurement, it ends it.")
        print()
    try:
        from router_wallet import UltraSwapBridge  # noqa: PLC0415
        w3 = UltraSwapBridge()._w3(args.chain)
        quoter = SameTierQuoter(w3, args.chain, retries=args.retries,
                                min_interval=args.min_interval)
    except Exception as exc:  # noqa: BLE001
        print("   cannot build the QuoterV2 handle (%s: %s)"
              % (type(exc).__name__, exc))
        return 2

    quoted: Dict[str, Any] = {}
    unquotable: Dict[str, Any] = {}
    unresolved: Dict[str, Any] = {}

    print("4. QUOTING -- SAME FEE TIER ON BOTH LEGS, and a plausibility floor")
    print("   Both legs are forced onto ONE pool. Picking the best tier per")
    print("   leg independently is not a round trip: it crosses two markets,")
    print("   and it printed NEGATIVE round-trip costs (AERO -0.0464%, ")
    print("   -0.0982%) -- free money, the same artifact pass 452 retracted.")
    print("   A round trip through one pool must pay its fee twice, so any")
    print("   reading below 2*fee is reported IMPLAUSIBLE and never published.")
    print("   A 429 is retried (%d times) and then reported as UNRESOLVED,"
          % args.retries)
    print("   never as UNQUOTABLE -- a throttle is not a fact about a pool.")
    hdr = "   %-10s %5s  %s" % ("SYMBOL", "rows", "round-trip cost % by clip")
    print(hdr)

    for sym, nrows in sorted(qualifying.items(), key=lambda kv: -kv[1]):
        address = book.get(sym)
        if not address:
            unquotable[sym] = {
                "rows": nrows,
                "cause": "no_address_in_book",
                "venues_answered": [],
                "venues_asked": [],
                "reason": "no address for %s on %s in data/token_addresses.json"
                          % (sym, args.chain),
            }
            print("   %-10s %5d  UNQUOTABLE: no address in the book" % (sym, nrows))
            continue

        # Pick the tier by ROUND-TRIP COST at the book's median clip, not by
        # "first tier that returns a number".
        #
        # MEASURED, and the reason this is not the cheap version: taking the
        # first tier in ascending fee order put AERO on the 0.01% pool, which
        # is dust. Its round trip read 13.1460% at $0.10 and rose monotonically
        # to 98.7740% at $10 -- a pool being emptied, not a cost. The 0.01%
        # tier EXISTS for AERO/USDC and prices a quote, which is exactly why
        # "it answered" is not a good enough test. The cheapest same-tier round
        # trip is what a router would take and what a fill would pay.
        probe_clip = dist.get("median") or 1.0
        tier = None
        tier_probe: Dict[str, Any] = {}
        rate_limited = False
        best_probe = None
        for fee in UNIV3_FEE_TIERS:
            try:
                d = quoter.round_trip(usdc, address, probe_clip, fee)
            except RateLimited as exc:
                rate_limited = True
                tier_probe["fee_%d" % fee] = "RATE LIMITED: %s" % exc
                break
            if d.get("cost_pct") is None:
                tier_probe["fee_%d" % fee] = d.get("reason") or "no quote"
                continue
            tier_probe["fee_%d" % fee] = {
                "cost_pct": round(float(d["cost_pct"]), 6),
                "implausible": d.get("implausible"),
            }
            if d.get("implausible"):
                continue
            if best_probe is None or d["cost_pct"] < best_probe:
                best_probe, tier = float(d["cost_pct"]), fee

        if rate_limited:
            unresolved[sym] = {"rows": nrows, "address": address,
                               "cause": "endpoint_throttled",
                               "venues_answered": [],
                               "venues_asked": [CENSUS_VENUE],
                               "reason": "endpoint throttled during tier probe",
                               "tier_probe": tier_probe}
            print("   %-10s %5d  UNRESOLVED: endpoint 429 during tier probe"
                  % (sym, nrows))
            continue
        if tier is None:
            unquotable[sym] = {"rows": nrows, "address": address,
                               "cause": "no_plausible_tier",
                               "venues_answered": [],
                               "venues_asked": [CENSUS_VENUE],
                               "reason": "no Uniswap V3 tier of %s round-"
                                         "tripped USDC->%s->USDC plausibly at "
                                         "the $%.2f probe clip"
                                         % (list(UNIV3_FEE_TIERS), sym,
                                            probe_clip),
                               "tier_probe": tier_probe}
            print("   %-10s %5d  UNQUOTABLE: no tier round-tripped plausibly"
                  % (sym, nrows))
            continue

        points: List[List[float]] = []
        failures: Dict[str, Any] = {}
        implausible: Dict[str, Any] = {}
        throttled = 0
        for clip in clips:
            try:
                d = quoter.round_trip(usdc, address, clip, tier)
            except RateLimited as exc:
                throttled += 1
                failures["$%g" % clip] = "RATE LIMITED: %s" % exc
                if args.throttle:
                    time.sleep(args.throttle)
                continue
            if d.get("implausible"):
                implausible["$%g" % clip] = {"cost_pct": d.get("cost_pct"),
                                             "floor_pct": d.get("floor_pct"),
                                             "why": d["implausible"]}
            elif d.get("cost_pct") is None:
                failures["$%g" % clip] = d.get("reason")
            else:
                points.append([clip, round(float(d["cost_pct"]), 6)])
            if args.throttle:
                time.sleep(args.throttle)

        record = {"address": address, "rows": nrows, "fee_tier": tier,
                  "venues_asked": [CENSUS_VENUE],
                  "venues_answered": [CENSUS_VENUE] if points else [],
                  "fee_tier_chosen_by": "cheapest plausible same-tier round "
                                        "trip at the $%.2f median clip"
                                        % probe_clip,
                  "probe_cost_pct": best_probe,
                  "points": points, "clips_answered": len(points),
                  "clips_attempted": len(clips), "failures": failures,
                  "implausible": implausible, "throttled_clips": throttled,
                  "tier_probe": tier_probe}
        if points:
            quoted[sym] = record
            print("   %-10s %5d  tier %-5d %s%s" % (
                sym, nrows, tier,
                "  ".join("$%g=%.4f%%" % (c, p) for c, p in points),
                "   [%d implausible, %d throttled]" % (len(implausible), throttled)
                if (implausible or throttled) else ""))
        elif throttled:
            record["cause"] = "endpoint_throttled"
            unresolved[sym] = record
            print("   %-10s %5d  UNRESOLVED: every clip throttled (429)"
                  % (sym, nrows))
        else:
            record["cause"] = "tier_priced_nothing"
            unquotable[sym] = record
            print("   %-10s %5d  UNQUOTABLE: tier %d priced nothing (%d "
                  "implausible)" % (sym, nrows, tier, len(implausible)))
    print()

    rows_quoted = sum(v["rows"] for v in quoted.values())

    # WHICH VENUES ANSWERED, counted from the census's OWN instrument.
    #
    # THE BUG THIS REPLACES. This block used to report budget.report(), which
    # is fed by quote_one() -- the PROVIDER path, used only by the dispersion
    # mode. The census quotes through SameTierQuoter instead, so budget stayed
    # empty and the RESULT line printed "venues that ANSWER: NONE -- the census
    # answers on ZERO symbols" no matter how many symbols were priced. A census
    # whose headline says zero while its table says otherwise is worse than no
    # census, so the count below is derived from the records themselves.
    #
    # Three states, kept apart on purpose (they are different facts):
    #   NOT_ON_CHAIN    the venue is not deployed/configured here at all. It
    #                   was never asked, so it cannot have "failed".
    #   ANSWERED n/m    asked about m addressed symbols and priced n of them.
    #   SILENT          supported and asked, and priced nothing.
    asked_symbols = sorted(
        s for s, d in list(quoted.items()) + list(unquotable.items())
        + list(unresolved.items()) if (d.get("venues_asked") or []))
    venue_answers: Dict[str, Any] = {}
    for name, _mod, _cls, _table in VENUES:
        ok, reason = support.get(name, (True, ""))
        if not ok:
            venue_answers[name] = {
                "status": "NOT_ON_CHAIN", "supported": False,
                "reason": reason, "symbols_asked": 0, "symbols_answered": 0,
                "note": "never asked, so this is NOT a venue that failed to "
                        "answer -- it is a venue that is not here.",
            }
            continue
        answered = sorted(s for s, d in quoted.items()
                          if name in (d.get("venues_answered") or []))
        asked_here = [s for s in asked_symbols
                      if name in ((quoted.get(s) or unquotable.get(s)
                                   or unresolved.get(s) or {})
                                  .get("venues_asked") or [])]
        venue_answers[name] = {
            "status": "ANSWERED" if answered else "SILENT",
            "supported": True, "reason": reason,
            "symbols_asked": len(asked_here),
            "symbols_answered": len(answered),
            "symbols": answered,
            "instrument": ("QuoterV2 quoteExactInput, direct pools only"
                           if name == CENSUS_VENUE else
                           "not asked by this mode; --round-trip-census "
                           "prices through %s only" % CENSUS_VENUE),
        }

    cause_counts = collections.Counter()
    for _s, d in list(unquotable.items()) + list(unresolved.items()):
        cause_counts[d.get("cause") or "unspecified"] += 1
    rows_by_cause = collections.Counter()
    for _s, d in list(unquotable.items()) + list(unresolved.items()):
        rows_by_cause[d.get("cause") or "unspecified"] += int(d.get("rows") or 0)

    doc = {
        "schema_version": 1,
        "artifact": ARTIFACT_PREFIX,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/routing_arbitrage_census.py --round-trip-census",
        "git_sha": _git_sha(),
        "chain": args.chain,
        "book_mode": args.mode,
        "THESE_ARE_QUOTES_NOT_FILLS": {
            "what_is_measured": "USDC -> token -> USDC, read-only router "
                                "quotes. cost_pct = (in - back) / in, in "
                                "percent of notional. POOL cost only: fee "
                                "plus price impact at that clip.",
            "not_modelled": ["slippage against a moving book", "MEV",
                             "gas (a dollar per leg, not a rate)",
                             "partial fills", "failed transactions"],
            "live_measured_round_trip_slippage_bp": LIVE_MEASURED_RT_SLIPPAGE_BP,
            "live_measured_round_trip_slippage_at_75c_clip_bp":
                LIVE_MEASURED_RT_SLIPPAGE_AT_75C_BP,
            "slippage_caveat": "THESE ARE QUOTES, NOT FILLS. The live lane "
                               "measured %.1f-%.1f bp of round-trip slippage "
                               "(pooled over 38 legs, and at the $0.75 clip "
                               "respectively) that NO quote here carries, and "
                               "MEV is unmodelled entirely. Every cost in this "
                               "file is therefore a FLOOR -- the best case -- "
                               "and is NOT a realisable cost. The two slippage "
                               "figures are the same quantity at different "
                               "clips, not a disagreement."
                               % (LIVE_MEASURED_RT_SLIPPAGE_BP,
                                  LIVE_MEASURED_RT_SLIPPAGE_AT_75C_BP),
            "fills_beat_quotes": "storage/trading_cache.db trade_fills holds "
                                 "38 real executed legs. Where a quote here "
                                 "disagrees with a fill for the same symbol, "
                                 "THE QUOTE IS THE SUSPECT.",
            "no_default_rate": "A symbol absent from 'quoted' is charged "
                               "NOTHING. There is no default, no nearest "
                               "neighbour and no chain average. This repo "
                               "shipped 'round_trip_cost measured its own "
                               "default' on 2026-09-04; this file will not.",
        },
        "clip_distribution_usd": dist,
        "clip_grid_usd": clips,
        "measured_gas_context": measured_gas_context(args.db),
        "instrument": {
            "quoter": "Uniswap V3 QuoterV2 quoteExactInput, asked directly",
            "both_legs_same_fee_tier": True,
            "plausibility_floor": "cost_pct >= 2 * fee_tier / 10000; readings "
                                  "below it are recorded under 'implausible' "
                                  "and NEVER published as a rate",
            "why_not_the_router_provider":
                "UniswapV3Local picks the best tier per leg INDEPENDENTLY, "
                "which crosses two pools and produced negative round-trip "
                "costs (AERO -0.0464%, -0.0982%) -- the artifact pass 452 "
                "retracted. It also swallows HTTP 429 and reports it as "
                "'no viable pool', making a throttle indistinguishable from "
                "an absent pool.",
            "eth_calls": quoter.calls,
            "rate_limit_hits_429": quoter.rate_limit_hits,
            "multihop_routes_not_searched":
                "only DIRECT USDC<->token pools are priced. A token that only "
                "trades through WETH is reported UNQUOTABLE here, not cheap.",
        },
        "row_source": {
            "db": args.db, "table": "trade_outcomes",
            "min_rows": int(args.min_rows),
            "rows_in_scope": len(scoped),
            "symbols_qualifying": len(qualifying),
            "rows_qualifying": sum(qualifying.values()),
            "rows_quoted": rows_quoted,
        },
        "venue_chain_support": {k: {"supported": v[0], "reason": v[1]}
                                for k, v in support.items()},
        "venues_that_answered": venue_answers,
        "venue_count_that_answered": sum(
            1 for d in venue_answers.values() if d["symbols_answered"]),
        "unquotable_causes": dict(UNQUOTABLE_CAUSES),
        "unquotable_by_cause": {
            "symbols": dict(cause_counts), "rows": dict(rows_by_cause)},
        "venue_activity": budget.report(),
        "venue_activity_note": "EMPTY BY DESIGN in --round-trip-census mode. "
                               "This tracks the PROVIDER path (quote_one), "
                               "which only the dispersion mode uses; the "
                               "census prices through SameTierQuoter. Read "
                               "'venues_that_answered' instead.",
        "quote_deadline_sec": budget.deadline,
        "venues_exempt_from_strikeout": sorted(budget.protected),
        "quoted": quoted,
        "unquotable": unquotable,
        "unresolved": unresolved,
    }

    out_path = Path(args.out) if args.out else artifact_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print("=" * 78)
    print("RESULT")
    answering = [(n, d) for n, d in sorted(venue_answers.items())
                 if d["symbols_answered"]]
    print("  VENUE COUNT THAT ANSWERS : %d" % len(answering))
    for name, d in sorted(venue_answers.items()):
        if d["status"] == "NOT_ON_CHAIN":
            print("    %-8s NOT ON CHAIN  never asked -- %s" % (name, d["reason"]))
        elif d["symbols_answered"]:
            print("    %-8s ANSWERED      %d/%d symbols asked"
                  % (name, d["symbols_answered"], d["symbols_asked"]))
        else:
            print("    %-8s SILENT        asked %d symbols, priced none"
                  % (name, d["symbols_asked"]))
    print("    (NOT ON CHAIN and DID NOT ANSWER are different facts and are")
    print("     counted separately. Only the second is a venue going quiet.)")
    print("  symbols QUOTED     : %d of %d (%d of %d rows)"
          % (len(quoted), len(qualifying), rows_quoted, sum(qualifying.values())))
    print("  symbols UNQUOTABLE : %d (%s)" % (len(unquotable),
          ", ".join(sorted(unquotable)) or "none"))
    print("  symbols UNRESOLVED : %d (%s)" % (len(unresolved),
          ", ".join(sorted(unresolved)) or "none"))
    if cause_counts:
        print("  WHY, by cause (symbols / rows charged NOTHING):")
        for cause, n in cause_counts.most_common():
            print("    %-22s %2d symbols  %3d rows" % (cause, n,
                                                       rows_by_cause[cause]))
    if not quoted:
        print()
        print("  THE CENSUS STILL ANSWERS ON ZERO SYMBOLS. That is the result,")
        print("  and it is complete. No number is printed for any symbol,")
        print("  because a default rate on an unquoted pool is an assumption")
        print("  dressed as a measurement.")
    print()
    print("  QUOTES, NOT FILLS. %.1f-%.1f bp of live round-trip slippage (pooled"
          % (LIVE_MEASURED_RT_SLIPPAGE_BP, LIVE_MEASURED_RT_SLIPPAGE_AT_75C_BP))
    print("  over 38 legs, and at the $0.75 clip) is in NO number above, and")
    print("  MEV is unmodelled entirely. Every cost here is a FLOOR, not a")
    print("  realisable cost. Fills beat quotes: trade_fills has 38 real legs,")
    print("  and where a quote disagrees with a fill THE QUOTE IS THE SUSPECT.")
    print("  wrote %s" % out_path)
    print("=" * 78)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", default="base")
    ap.add_argument("--round-trip-census", action="store_true",
                    help="quote USDC->token->USDC for every symbol the book "
                         "holds, and write the versioned artifact the "
                         "re-pricing scripts read")
    ap.add_argument("--db", default=os.environ.get("TRADING_DB_PATH",
                                                   "storage/trading_cache.db"))
    ap.add_argument("--mode", default="ghost", choices=["ghost", "live", "all"])
    ap.add_argument("--min-rows", type=int, default=3,
                    help="quote every symbol holding at least this many rows")
    ap.add_argument("--clips", default="",
                    help="comma-separated clip sizes in USD; default is a fine "
                         "grid spanning the book's own notional distribution")
    ap.add_argument("--deadline", type=float, default=DEFAULT_QUOTE_DEADLINE_SEC,
                    help="seconds ONE venue gets for ONE quote before the "
                         "census abandons it and moves on")
    ap.add_argument("--strikes", type=int, default=DEFAULT_VENUE_STRIKES,
                    help="consecutive deadline breaches before a venue is "
                         "dropped for the rest of the run")
    ap.add_argument("--throttle", type=float, default=0.25,
                    help="seconds between quotes; mainnet.base.org is the only "
                         "working Base endpoint and it answers 429 under load")
    ap.add_argument("--min-interval", type=float, default=None,
                    help="minimum seconds between ANY two eth_calls; default "
                         "%.1fs, measured as the sustainable rate on "
                         "mainnet.base.org" % SameTierQuoter.MIN_CALL_INTERVAL_SEC)
    ap.add_argument("--retries", type=int, default=4,
                    help="retries with exponential backoff on HTTP 429 before "
                         "a symbol is reported UNRESOLVED rather than priced")
    ap.add_argument("--out", default="",
                    help="artifact path; default data/%s_<UTC stamp>.json"
                         % ARTIFACT_PREFIX)
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

    if args.round_trip_census:
        return run_round_trip_census(args)

    support = chain_support(args.chain)
    providers = _providers(args.chain, support)
    if len(providers) < 2:
        print("routing census: fewer than two venues are SUPPORTED on %s "
              "(%s); a dispersion number needs at least two venues to compare."
              % (args.chain, ", ".join(sorted(providers)) or "none"))
        for name, (ok, reason) in sorted(support.items()):
            print("  %-8s %-12s %s" % (name, "SUPPORTED" if ok else "NOT ON CHAIN",
                                       reason))
        print("  This is a VENUE FAMINE, not a failed measurement: routing")
        print("  dispersion is UNDEFINED where there is one venue. Use")
        print("  --round-trip-census, which measures COST and needs only one.")
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
    budget = VenueBudget(args.deadline, args.strikes)
    for _round in range(max(1, args.rounds)):
        for label, (t_in, t_out, amount_in) in pairs.items():
            quotes, span = quote_all(providers, args.chain, t_in, t_out,
                                     amount_in, 100, budget)
            disp = dispersion(quotes)
            arb = round_trip_arbitrage(providers, args.chain, t_in, t_out,
                                       amount_in, args.gas_usd,
                                       args.notional_usd, 100, budget)
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
