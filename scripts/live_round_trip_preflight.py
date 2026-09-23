#!/usr/bin/env python
"""Would the live WETH-USDC round trip in item 13254b56 ACTUALLY PLACE if armed?

WHY THIS EXISTS
---------------
This serves the LIVE-EVIDENCE route -- item ``74a12c84`` ("BUY THE 25 ROWS
LIVE INSTEAD") and its siblings. That route's whole argument is that live rows
cost pennies at this clip and are worth more than ghost rows, so the blocker is
an OPERATOR DECISION rather than evidence.

It was written for ``13254b56`` (one live WETH-USDC round trip), which Echo
REJECTED in pass 675: that was a COST item, and the donchian deep result it
was costing has a NEGATIVE gross in the executor's own book (-5.939bp mean, 4
of 6 negative, 0 of 6 net wins -- Sage p674, confirmed by Echo and Cove p675).
Cost only ever subtracts, so a break-even threshold against a negative gross is
moot. The preflight survives the rejection because it never measured cost: it
measures whether the live lane can place the trade at all, which every
live-evidence item needs and none of them checks.

The gap it closes is an exposure, not a formality. ``13254b56`` was closed
three times (passes 495, 674, 675) with the same verdict -- *blocked on
operator go-ahead* -- and in all three, nobody checked whether the trade would
place once that go-ahead arrived. The operator is being asked for one
irreversible decision on real money. If the executor path is unreachable for a
reason that has nothing to do with permission, the go-ahead is spent on a trade
that cannot happen, and the next ask is much harder to make.

WHAT IT FOUND, PASS 675
-----------------------
Run against WETH-USDC it returns precondition 4 FAILED, and that verdict
generalises past WETH. ``money_button`` is the ONLY live-approved strategy, and
across its 10 all-time recorded entries it has touched 4 symbols -- PEPE-USDC
(7), TOAD-USDC, BASECAT-USDC, CBZEC-USDC -- and ZERO WETH. So an operator
go-ahead is NECESSARY BUT NOT SUFFICIENT for a live WETH round trip: the only
strategy permitted to place one has never emitted that symbol. Any live-evidence
item that names a deep-pool pair has to clear this first, and the same query
answers it for whatever symbol that item picks (``--symbol``).

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
READ-ONLY. Nothing is signed, nothing is sent, no gate is flipped, no capital
moves, no ``.env`` is written. It reads the production transition plan, the
strategy ledger, the balances table and the env, and reports what the live
lane would do. It is a PREFLIGHT, not a dry-run fill: it does not prove a swap
would succeed on chain, only that the lane would emit one.

It deliberately does NOT re-measure round-trip COST. ``scripts/
deep_pool_slippage_probe.py`` already quotes USDC->TOKEN->USDC against live
Base pool state at a real clip, and duplicating it here would put a second,
drifting copy of that number in the tree -- which is how this repo ended up
with three copies of the cost table (memory: ``three-copies-of-the-token-cost-
table``). Run that probe for cost; run this one for reachability.

THE FOUR PRECONDITIONS, AND WHY THESE FOUR
------------------------------------------
Every live entry in ``trading/bot.py`` has to clear all four. They are
independent -- permission covers only the third -- so any one of them can be
the real blocker while the notes say "waiting on the operator":

  1. CLIP      The transition plan must sanction a nonzero notional.
               ``trading/pipeline.py:6309`` computes ``recommended_live_usd``
               as a ratio of deployable stable, then ``:6311`` compares it to
               ``min_clip_usd``; ``trading/bot.py:2125`` (``_live_clip_usd``)
               takes ``max(recommended, min_clip)`` and then min()s it back
               down against every published cap. A zero here is the arithmetic
               deadlock documented at ``bot.py:2087``, not a refusal.

  2. GAS       A Base round trip is two swaps and both need native ETH.
               ``ENABLE_GAS_REFILL`` is 0 by standing instruction (the refill
               drained the wallet three times), so nothing tops this up. Step
               7 of the ladder checks STABLE only and would pass a wallet with
               zero gas.

  3. ARM       ``ENABLE_LIVE_TRADING`` plus the selector's
               ``_readiness_permits_live`` / ``_operator_armed_live`` pair.
               This is the ONLY one of the four that operator permission
               moves.

  4. APPROVAL  ``bot.py`` gates every entry on ``_strategy_live_approved`` ->
               ``strategy_ledger.is_live_approved(sid)``. A strategy that is
               not ledger-approved cannot emit a live entry no matter how
               armed the lane is -- and the approved set is currently ONE
               strategy. If no approved strategy trades WETH, this item cannot
               happen through the normal path at all, and permission is
               necessary but NOT sufficient.

Precondition 4 is the one this script was written to expose, because it is the
one that a go-ahead does not fix.

USAGE
-----
    python -X utf8 scripts/live_round_trip_preflight.py
    python -X utf8 scripts/live_round_trip_preflight.py --json
    python -X utf8 scripts/live_round_trip_preflight.py --symbol WETH-USDC

Exit code is 0 when all four preconditions hold (the trade would place if the
operator says go), 1 when any of them does not. A nonzero exit is NOT an
error in this script -- it is the finding.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB_PATH = "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db")

#: Stable symbols whose ``quantity`` column IS the dollar value. ``usd_amount``
#: is a cached mark that freezes while quantity moves underneath it -- reading
#: it overstated the wallet by 17.9% (see ``scripts/live_path_check.py:186``).
STABLES = {"USDC", "USDT", "DAI", "USDBC"}

#: Measured gas per round trip in USD, pass 496: $0.0040 across the live rows.
#: Quoted as the per-ROUND-TRIP figure because that is what this item buys.
GAS_USD_PER_ROUND_TRIP = 0.0040

#: A round trip is two swaps; require headroom for several so a single gas
#: spike does not strand a half-open position with no gas to exit it. An entry
#: that cannot afford its own exit is worse than no entry.
GAS_ROUND_TRIPS_REQUIRED = 5

TRUTHY = {"1", "true", "yes", "on"}


def _db() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, uri=True)


def _env_on(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in TRUTHY


def _flt(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if v == v and v not in (float("inf"), float("-inf")) else default


class Check:
    """One precondition, with the reason it holds or does not."""

    def __init__(self, key: str, title: str) -> None:
        self.key = key
        self.title = title
        self.ok: Optional[bool] = None      # None = UNKNOWN, which is not False
        self.detail = "--"
        self.blocker = ""
        self.facts: Dict[str, Any] = {}

    def passed(self, detail: str, **facts: Any) -> "Check":
        self.ok, self.detail = True, detail
        self.facts.update(facts)
        return self

    def failed(self, detail: str, blocker: str = "", **facts: Any) -> "Check":
        self.ok, self.detail, self.blocker = False, detail, blocker
        self.facts.update(facts)
        return self

    def unknown(self, detail: str, **facts: Any) -> "Check":
        # UNKNOWN is reported as its own state and never silently as a pass.
        # A preflight that reports "fine" because it could not look is the
        # exact failure mode this script exists to prevent.
        self.ok, self.detail = None, detail
        self.facts.update(facts)
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "ok": self.ok,
            "detail": self.detail,
            "blocker": self.blocker,
            "facts": self.facts,
        }


# --------------------------------------------------------------------------
# 1. CLIP -- does the production plan sanction a nonzero notional?
# --------------------------------------------------------------------------
def check_clip() -> Check:
    c = Check("clip", "CLIP -- the plan sanctions a notional")
    try:
        # Same neutering live_path_check.py applies: the news loader reaches
        # for a feed this preflight has no business waking, and a preflight
        # that can be blocked by a news fetch is not a preflight.
        import trading.data_loader as dl
        dl.HistoricalDataLoader._load_news = lambda self: []
        from trading.pipeline import TrainingPipeline
        from db import get_db
        plan = TrainingPipeline(db=get_db())._build_transition_plan()
    except Exception as exc:
        return c.unknown("transition plan unavailable: %s: %s"
                         % (type(exc).__name__, exc))

    capital = plan.get("capital_plan") if isinstance(plan, dict) else None
    capital = capital if isinstance(capital, dict) else {}
    flags = plan.get("risk_flags") if isinstance(plan, dict) else None
    flags = flags if isinstance(flags, dict) else {}

    recommended = _flt(capital.get("recommended_live_usd"))
    min_clip = _flt(capital.get("min_clip_usd"))
    deployable = _flt(capital.get("deployable_stable_usd"))
    blocked_reason = str(flags.get("live_blocked_reason") or "")

    # Reproduce bot.py:2125 _live_clip_usd EXACTLY -- max() of the two, then
    # min()ed against every cap the plan publishes. Computing it a second,
    # unrelated way is what put an 18x disagreement between the plan and the
    # swap (bot.py:2080-2105); this preflight must not repeat that.
    clip = max(recommended, min_clip)
    caps_applied: List[Tuple[str, float]] = []
    ramp = capital.get("live_ramp_schedule")
    ramp = ramp if isinstance(ramp, dict) else {}
    for source, key in (
        (ramp, "first_tranche_cap_usd"),
        (capital, "live_capital_cap_usd"),
        (capital, "deployable_stable_usd"),
    ):
        cap = _flt(source.get(key))
        if cap > 0.0:
            caps_applied.append((key, cap))
            clip = min(clip, cap)
    clip = max(0.0, clip)

    facts = {
        "recommended_live_usd": round(recommended, 6),
        "min_clip_usd": round(min_clip, 6),
        "deployable_stable_usd": round(deployable, 6),
        "effective_clip_usd": round(clip, 6),
        "caps_applied": [{"key": k, "usd": round(v, 6)} for k, v in caps_applied],
        "live_blocked_reason": blocked_reason,
        "min_clip_block": bool(flags.get("min_clip_block")),
        "env_LIVE_MIN_CLIP_USD": os.getenv("LIVE_MIN_CLIP_USD"),
    }

    if clip <= 0.0:
        return c.failed(
            "effective clip is $0.00 (recommended=$%.4f min_clip=$%.4f reason=%r)"
            % (recommended, min_clip, blocked_reason),
            "sizing collapsed -- see trading/pipeline.py:6309-6330 and "
            "bot.py:2125; this is NOT fixed by operator permission",
            **facts)
    return c.passed("plan sanctions $%.4f per live entry" % clip, **facts)


# --------------------------------------------------------------------------
# 2. GAS -- native ETH for both legs
# --------------------------------------------------------------------------
def check_gas() -> Check:
    c = Check("gas", "GAS -- native ETH funds both legs")
    try:
        conn = _db()
        rows = list(conn.execute(
            "SELECT symbol, quantity, usd_amount FROM balances "
            "WHERE wallet='guardian' AND chain='base'"))
    except Exception as exc:
        return c.unknown("balances unreadable: %s" % exc)

    native_qty = 0.0
    native_usd = 0.0
    stable = 0.0
    for symbol, quantity, usd_amount in rows:
        sym = str(symbol or "").upper()
        if sym == "ETH":
            native_qty += _flt(quantity)
            native_usd += _flt(usd_amount)
        elif sym in STABLES:
            # quantity, NOT usd_amount -- see the STABLES docstring above.
            stable += _flt(quantity)

    required = GAS_USD_PER_ROUND_TRIP * GAS_ROUND_TRIPS_REQUIRED
    affordable = (native_usd / GAS_USD_PER_ROUND_TRIP) if GAS_USD_PER_ROUND_TRIP > 0 else 0.0
    facts = {
        "native_eth_qty": native_qty,
        "native_eth_usd": round(native_usd, 6),
        "deployable_stable_usd": round(stable, 6),
        "gas_usd_per_round_trip": GAS_USD_PER_ROUND_TRIP,
        "round_trips_affordable": round(affordable, 1),
        "round_trips_required": GAS_ROUND_TRIPS_REQUIRED,
        "ENABLE_GAS_REFILL": os.getenv("ENABLE_GAS_REFILL"),
    }

    # ENABLE_GAS_REFILL must stay 0 (standing instruction: the refill drained
    # the wallet three times). Report it, never change it -- but it is exactly
    # why a thin gas balance here is terminal rather than self-healing.
    if _env_on("ENABLE_GAS_REFILL"):
        facts["warning"] = ("ENABLE_GAS_REFILL is ON and the standing "
                            "instruction says it must stay 0")

    if native_usd <= 0.0:
        return c.failed("native ETH on base is $0.00 and ENABLE_GAS_REFILL=%s "
                        "will not top it up" % os.getenv("ENABLE_GAS_REFILL"),
                        "fund native ETH on base", **facts)
    if native_usd < required:
        return c.failed(
            "native ETH $%.4f funds %.1f round trips, under the %d required "
            "to enter AND still afford the exit"
            % (native_usd, affordable, GAS_ROUND_TRIPS_REQUIRED),
            "fund native ETH on base", **facts)
    return c.passed("native ETH $%.4f funds ~%.0f round trips at $%.4f each"
                    % (native_usd, affordable, GAS_USD_PER_ROUND_TRIP), **facts)


# --------------------------------------------------------------------------
# 3. ARM -- the one precondition operator permission actually moves
# --------------------------------------------------------------------------
def check_arm() -> Check:
    c = Check("arm", "ARM -- the executor is permitted to place live orders")
    enabled = _env_on("ENABLE_LIVE_TRADING")
    facts = {"ENABLE_LIVE_TRADING": os.getenv("ENABLE_LIVE_TRADING")}
    if not enabled:
        return c.failed("ENABLE_LIVE_TRADING is off",
                        "THIS is the operator decision -- set ENABLE_LIVE_TRADING=1",
                        **facts)
    try:
        from trading.selector import GhostTradingSupervisor
        import trading.data_loader as dl
        dl.HistoricalDataLoader._load_news = lambda self: []
        from trading.pipeline import TrainingPipeline
        from db import get_db
        readiness = TrainingPipeline(db=get_db()).live_readiness_report()
    except Exception as exc:
        return c.unknown("readiness unavailable: %s: %s"
                         % (type(exc).__name__, exc), **facts)

    # Measure the predicate the SELECTOR computes, not one of its two terms.
    # Probing _readiness_permits_live alone once reported the executor link
    # FAILED while bots were in fact live-capable off an explicit arm
    # (live_path_check.py:234-238).
    permits = bool(GhostTradingSupervisor._readiness_permits_live(readiness))
    armed = bool(GhostTradingSupervisor._operator_armed_live())
    facts.update({
        "readiness_permits_live": permits,
        "operator_armed_live": armed,
        "ready": readiness.get("ready"),
        "ghost_ready": readiness.get("ghost_ready"),
        "ghost_reason": readiness.get("ghost_reason"),
    })
    if not permits and not armed:
        return c.failed("bots would be built with live_trading_enabled=False "
                        "(ready=%s ghost_ready=%s)"
                        % (readiness.get("ready"), readiness.get("ghost_ready")),
                        "operator arm, or ghost evidence the selector accepts",
                        **facts)
    if not permits and armed:
        # Say which of the two carries it, so this never reads as earned
        # graduation. It is an explicit arm, not ghost evidence.
        return c.passed("live-capable on an EXPLICIT operator arm, NOT on "
                        "ghost evidence (ghost_ready=%s reason=%s)"
                        % (readiness.get("ghost_ready"),
                           readiness.get("ghost_reason")), **facts)
    return c.passed("live-capable on readiness (ghost_reason=%s)"
                    % readiness.get("ghost_reason"), **facts)


# --------------------------------------------------------------------------
# 4. APPROVAL -- would any approved strategy ever emit THIS symbol?
# --------------------------------------------------------------------------
def check_approval(symbol: str) -> Check:
    """The precondition a go-ahead does NOT fix.

    ``bot.py`` gates every live entry on ``_strategy_live_approved(sid)`` ->
    ``strategy_ledger.is_live_approved(sid)``. So a live WETH-USDC entry needs
    an APPROVED strategy that actually trades WETH-USDC. Approval and
    production are known-disjoint here (memory:
    ``approved-strategy-and-producing-strategy-are-disjoint``), which is
    precisely why this has to be checked rather than assumed.

    The evidence used is the strategy's own recorded entries in
    ``trading_ops`` -- what it HAS emitted, not what it might. That is
    one-sided on purpose: a strategy with WETH entries on the record can
    clearly emit one, while a strategy with none is only *probably* unable to.
    The weaker direction is reported as such and never as proof.
    """
    c = Check("approval", "APPROVAL -- an approved strategy emits %s" % symbol)
    try:
        from trading.strategies.ledger import StrategyLedger
        data = dict(StrategyLedger()._data or {})
    except Exception as exc:
        return c.unknown("strategy ledger unreadable: %s: %s"
                         % (type(exc).__name__, exc))

    approved = sorted(sid for sid, e in data.items()
                      if isinstance(e, dict) and e.get("live_approved"))
    facts: Dict[str, Any] = {"approved_strategies": approved,
                             "approved_count": len(approved),
                             "symbol": symbol}
    if not approved:
        return c.failed("no strategy is live-approved",
                        "graduation -- ladder step 5", **facts)

    # Which symbols has each approved strategy actually entered? trading_ops.ts
    # is a UNIX FLOAT, not a datetime -- a datetime('now',...) WHERE matches
    # ZERO rows and raises NOTHING, which reads as a finding. No time filter is
    # applied here at all: the question is all-time capability, and an empty
    # set from a bad WHERE is indistinguishable from an empty set of truth.
    base = symbol.split("-")[0].upper()
    per_strategy: Dict[str, Dict[str, Any]] = {sid: {} for sid in approved}
    # There is NO strategy_id COLUMN on trading_ops -- the columns are
    # (id, ts, wallet, chain, symbol, action, status, details) and strategy_id
    # lives inside the `details` JSON blob. Querying it as a column raises
    # OperationalError rather than returning nothing, which is the one mercy
    # here: the sibling trap on this table is that `ts` is a UNIX FLOAT, so a
    # datetime('now',...) WHERE matches ZERO rows and raises NOTHING, and the
    # empty set reads as a finding. No time filter is applied at all -- the
    # question is all-time capability, and an empty set from a bad WHERE is
    # indistinguishable from an empty set of truth.
    counts: Dict[str, Dict[str, int]] = {sid: {} for sid in approved}
    approved_set = set(approved)
    try:
        conn = _db()
        cur = conn.execute(
            "SELECT symbol, details FROM trading_ops WHERE status LIKE '%entry%'")
        for sym_raw, details in cur:
            if not details:
                continue
            try:
                d = json.loads(details)
            except (ValueError, TypeError):
                continue
            if not isinstance(d, dict):
                continue
            sid = d.get("strategy_id") or d.get("strategy")
            if sid not in approved_set:
                continue
            sym = str(sym_raw or "").upper()
            counts[sid][sym] = counts[sid].get(sym, 0) + 1
    except Exception as exc:
        return c.unknown("trading_ops unreadable: %s: %s"
                         % (type(exc).__name__, exc), **facts)

    for sid in approved:
        symbols = counts[sid]
        total = sum(symbols.values())
        hits = sum(n for s, n in symbols.items() if s.startswith(base))
        per_strategy[sid] = {
            "entries_all_time": total,
            "distinct_symbols": len(symbols),
            "entries_on_%s" % base: hits,
            "top_symbols": sorted(symbols.items(), key=lambda kv: -kv[1])[:8],
        }

    facts["per_strategy"] = per_strategy
    emitters = [sid for sid, d in per_strategy.items()
                if d.get("entries_on_%s" % base, 0) > 0]
    facts["approved_emitters_of_%s" % base] = emitters

    if emitters:
        return c.passed("approved strategy %s has entered %s on the record"
                        % (", ".join(emitters), base), **facts)

    silent = [sid for sid, d in per_strategy.items()
              if d.get("entries_all_time", 0) == 0]
    facts["approved_with_zero_entries"] = silent
    return c.failed(
        "no approved strategy has EVER entered %s (approved=%s; %s have zero "
        "recorded entries of any symbol)"
        % (base, ", ".join(approved), len(silent)),
        "operator permission does NOT fix this -- either approve a strategy "
        "that trades %s, or place the round trip outside the strategy path" % base,
        **facts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="WETH-USDC",
                    help="pair the round trip would be placed on")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    checks = [check_clip(), check_gas(), check_arm(), check_approval(args.symbol)]
    # UNKNOWN never counts as a pass. "Would place" requires every precondition
    # to be affirmatively true, not merely not-false.
    would_place = all(c.ok is True for c in checks)

    report = {
        "measurement": "PREFLIGHT -- READ-ONLY",
        "not": ("a fill, and not a dry-run swap. Nothing was signed or sent. "
                "This says the lane WOULD emit the entry, not that the swap "
                "would succeed on chain."),
        "cost_is_not_measured_here": ("use scripts/deep_pool_slippage_probe.py "
                                      "-- it quotes USDC->TOKEN->USDC against "
                                      "live Base pool state"),
        "symbol": args.symbol,
        "would_place_if_operator_says_go": would_place,
        "checks": [c.as_dict() for c in checks],
    }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0 if would_place else 1

    print("LIVE ROUND TRIP PREFLIGHT -- %s" % args.symbol)
    print("READ-ONLY. Nothing signed, nothing sent, no gate flipped.")
    print("Cost is NOT measured here -- use scripts/deep_pool_slippage_probe.py.\n")
    for c in checks:
        mark = "PASS" if c.ok is True else ("FAIL" if c.ok is False else "UNKN")
        print("  [%s] %s" % (mark, c.title))
        print("         %s" % c.detail)
        if c.blocker:
            print("         BLOCKER: %s" % c.blocker)
    print()
    failing = [c for c in checks if c.ok is not True]
    if would_place:
        print("VERDICT: all four preconditions hold. The ONLY thing between "
              "here and a live %s round trip is the operator decision." % args.symbol)
        return 0
    print("VERDICT: %d of %d preconditions do NOT hold: %s."
          % (len(failing), len(checks), ", ".join(c.key for c in failing)))
    print("An operator go-ahead alone would NOT produce this round trip.")
    print("Fix these before asking for the decision again -- a go-ahead spent "
          "on a trade that cannot place is a go-ahead that is harder to get twice.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
