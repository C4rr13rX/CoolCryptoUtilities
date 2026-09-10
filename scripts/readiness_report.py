#!/usr/bin/env python3
"""
Autonomous-trading readiness: what the ghost record actually supports.

The question this answers is "can it trade on its own yet, and if not, what is
missing and when will it arrive" -- from the evidence, not from a feeling that
it has been long enough.

It is deliberately a *report*, not a switch. The graduation machinery already
exists and already works per strategy; what has been missing is an honest read
of whether the evidence has accumulated, and at what rate.

A report that does not measure what the gate measures is worse than no report,
because every pass is steered by the wall it names. This one reported
``ready=True`` on two strategies the gate was correctly refusing, for a year of
passes, because it read the POOLED ghost book and checked none of the three
structural bars ``_evaluate_graduation_locked`` returns early on. Measured
2026-09-10 on the real ledger:

    atf_static        POOLED 52 trades  29 wins  +1.5407   ready=True   <- lie
                   TRADEABLE  4 trades   2 wins  -0.0187
                   demote_reason "live P/L -0.1585 over 17 trades", demotions 7

    atf_static_scout  POOLED 236 trades 186 wins +6.4818   ready=True   <- lie
                   TRADEABLE   3 trades   1 win  -0.0778
                   graduation_blocked=True (ghost-only: no live branch exists)

Both already carried ``graduated_ts``. So ``classify_wall`` printed "READY BUT
UNSTAMPED -- the ledger is not stamping graduated_ts" at the top of every pass
while the stamps existed and the gate was refusing them on their tradeable
record. That is the wrong wall, and working the wrong wall is the single most
expensive mistake available in this loop.

``ready`` therefore now mirrors ``_evaluate_graduation_locked`` exactly:

  * the population is ``_tradeable_of(ghost)`` -- round trips the live lane
    could actually have placed -- and, for a demoted strategy, the FRESH
    tradeable delta since ``ghost_at_demotion``, which is the population the
    re-arm rule reads;
  * ``graduation_blocked`` and ``GHOST_ONLY_STRATEGY_IDS`` are permanent
    blockers, not soft ones;
  * a demoted strategy whose live book is a losing one carries the re-arm
    rule's own live-record bar as a blocker.

The pooled numbers are not thrown away -- they are reported alongside as
``pooled_*`` so the funnel's raw throughput stays visible. They are just no
longer the thing the word "ready" is computed from. The bar itself is
untouched: same MIN_TRADES, same MIN_WINRATE, same MIN_PROFIT.

Run:  python scripts/readiness_report.py
      python scripts/readiness_report.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def wilson_lower_bound(wins: int, trades: int, z: float = 1.96) -> float:
    """
    Lower bound of the 95% confidence interval on the win rate.

    This is the number that matters for a go-live decision, and it is why
    trade count cannot be waved away. 7 wins from 7 trades looks like 100% but
    its lower bound is 65% -- the evidence is consistent with a strategy that
    genuinely wins only two thirds of the time. 55 wins from 100 has a lower
    bound of 45%. The bound rises only as trades accumulate, which is exactly
    the property a "have we proven it" gate needs.
    """
    if trades <= 0:
        return 0.0
    phat = wins / trades
    denom = 1 + z * z / trades
    centre = phat + z * z / (2 * trades)
    margin = z * ((phat * (1 - phat) / trades + z * z / (4 * trades * trades)) ** 0.5)
    return max(0.0, (centre - margin) / denom)


def _graduation_population(ledger, sid: str, entry: dict) -> tuple:
    """The trades/wins/profit the GATE would read for ``sid``, and why.

    Returns ``(trades, wins, profit, population, structural_blockers, blocked)``,
    where ``blocked`` marks a strategy that can NEVER graduate however much
    evidence arrives -- distinct from one that is merely short of the bar.

    This is the whole point of the module: it asks the ledger's own helpers the
    same question ``_evaluate_graduation_locked`` asks, in the same order, so
    the report cannot drift away from the gate again. Every branch below has a
    matching early ``return`` in that method.
    """
    from trading.strategies import ledger as ledger_mod

    ghost = entry.get("ghost", {}) or {}
    live = entry.get("live", {}) or {}
    blockers = []

    # (1) Structural. "This executor cannot spend money" never stops being
    # true, so it is not re-litigated against a fresh ghost book and it is not
    # a countdown -- no number of ghost trades retires it.
    ghost_only = False
    try:
        ghost_only = sid in ledger_mod._ghost_only_ids()
    except Exception:  # noqa: BLE001
        pass
    blocked = bool(entry.get("graduation_blocked")) or ghost_only
    if blocked:
        blockers.append(
            "ghost-only executor: no live branch exists, so this can never graduate"
        )

    # (2) Demoted strategies are judged by the re-arm rule on FRESH tradeable
    # evidence since the demotion, not by the first-licence bar on the whole
    # book. Reading the lifetime book here is what "a ghost trade undid a live
    # demotion" was.
    if entry.get("demote_reason"):
        at = entry.get("ghost_at_demotion")
        if isinstance(at, dict):
            fresh = ledger_mod._fresh_tradeable_delta(ghost, at)
        else:
            # The gate baselines from NOW in this case, so the fresh window is
            # empty until the next close. Report that, do not report the
            # lifetime book as if it were fresh.
            fresh = {"trades": 0, "wins": 0, "total_profit": 0.0}
        # The re-arm rule refuses outright on a live book that lost real money
        # over a big enough sample, whatever the ghost record says.
        try:
            net = ledger._licence_net(live)
            live_trades = ledger._licence_trades(live)
        except Exception:  # noqa: BLE001
            net, live_trades = 0.0, 0
        min_sample = _env_int("STRATEGY_REARM_MIN_LIVE_TRADES", 3)
        if live_trades >= min_sample and net <= 0.0:
            blockers.append(
                f"demoted, and its live licence is {net:+.4f} over {live_trades} "
                f"trades: ghost evidence cannot excuse lost money"
            )
        return (
            int(fresh.get("trades", 0)),
            int(fresh.get("wins", 0)),
            float(fresh.get("total_profit", 0.0)),
            "fresh tradeable since demotion",
            blockers,
            blocked,
        )

    # (3) First licence: the tradeable subset of the lifetime ghost book.
    sub = ledger_mod._tradeable_of(ghost)
    return (
        int(sub.get("trades", 0)),
        int(sub.get("wins", 0)),
        float(sub.get("total_profit", 0.0)),
        "tradeable",
        blockers,
        blocked,
    )


def collect() -> dict:
    from trading.strategies.ledger import StrategyLedger

    ledger = StrategyLedger()
    data = dict(getattr(ledger, "_data", {}))

    min_trades = _env_int("STRATEGY_GRADUATION_MIN_TRADES", 20)
    min_winrate = _env_float("STRATEGY_GRADUATION_MIN_WINRATE", 0.55)
    min_profit = _env_float("STRATEGY_GRADUATION_MIN_PROFIT", 0.0)

    now = time.time()
    strategies = []
    stamps = []

    for sid, entry in sorted(data.items()):
        ghost = entry.get("ghost", {}) or {}
        live = entry.get("live", {}) or {}
        # The pooled book is the funnel's raw throughput and stays visible, but
        # it is NOT what "ready" is computed from -- see the module docstring.
        pooled_trades = int(ghost.get("trades", 0))
        pooled_wins = int(ghost.get("wins", 0))
        pooled_profit = float(ghost.get("total_profit", 0.0))
        last_ts = float(ghost.get("last_ts", 0.0))
        if last_ts:
            stamps.append(last_ts)

        trades, wins, profit, population, blockers, blocked = _graduation_population(
            ledger, sid, entry
        )

        win_rate = wins / trades if trades else 0.0
        lower = wilson_lower_bound(wins, trades)

        if trades < min_trades:
            blockers.append(f"needs {min_trades - trades} more ghost trades")
        if trades and win_rate < min_winrate:
            blockers.append(f"win rate {win_rate:.0%} < {min_winrate:.0%}")
        if profit <= min_profit:
            blockers.append(f"ghost P/L {profit:+.4f} not above {min_profit}")

        strategies.append({
            "id": sid,
            "ghost_trades": trades,
            "ghost_wins": wins,
            "win_rate": win_rate,
            "win_rate_lower_95": lower,
            "ghost_profit": profit,
            "population": population,
            "permanently_blocked": blocked,
            "pooled_ghost_trades": pooled_trades,
            "pooled_ghost_wins": pooled_wins,
            "pooled_ghost_profit": pooled_profit,
            "live_approved": bool(entry.get("live_approved")),
            "live_trades": int(live.get("trades", 0)),
            "demotions": int(entry.get("demotions", 0)),
            "graduated_ts": entry.get("graduated_ts"),
            "demote_reason": entry.get("demote_reason"),
            "last_trade_age_days": (now - last_ts) / 86400 if last_ts else None,
            "blockers": blockers,
            "ready": not blockers,
        })

    total_trades = sum(s["ghost_trades"] for s in strategies)
    total_wins = sum(s["ghost_wins"] for s in strategies)
    pooled_total_trades = sum(s["pooled_ghost_trades"] for s in strategies)
    pooled_total_wins = sum(s["pooled_ghost_wins"] for s in strategies)
    span_days = ((max(stamps) - min(stamps)) / 86400) if len(stamps) > 1 else 0.0
    # The rate that matters for an ETA is the rate the BAR is fed at, and the
    # bar counts tradeable round trips. Quoting the pooled rate here is how
    # "nearest graduation ~0.7 days" was printed against a strategy whose
    # tradeable book had gained 4 trades in 7 days.
    rate = (total_trades / span_days) if span_days > 0 else 0.0
    pooled_rate = (pooled_total_trades / span_days) if span_days > 0 else 0.0

    # Days until the closest strategy graduates, at the observed rate. Per
    # strategy, because graduation is per strategy -- the aggregate rate is
    # split across however many are trading. Structurally blocked strategies
    # are excluded: they never arrive, so letting one hold the minimum reports
    # an ETA for a graduation that cannot happen.
    eta = None
    eligible = [s for s in strategies if not s.get("permanently_blocked")]
    if rate > 0 and eligible:
        active = max(1, sum(1 for s in eligible if s["ghost_trades"] > 0))
        per_strategy_rate = rate / active
        shortfalls = [
            (min_trades - s["ghost_trades"]) / per_strategy_rate
            for s in eligible
            if s["ghost_trades"] < min_trades and not s["live_approved"]
        ]
        if shortfalls:
            eta = min(shortfalls)

    return {
        "generated_at": now,
        "criteria": {
            "min_trades": min_trades,
            "min_winrate": min_winrate,
            "min_profit": min_profit,
            "enforced": os.getenv("STRATEGY_GRADUATION_ENFORCED", "1"),
        },
        "totals": {
            "strategies": len(strategies),
            "ghost_trades": total_trades,
            "ghost_wins": total_wins,
            "win_rate": total_wins / total_trades if total_trades else 0.0,
            "win_rate_lower_95": wilson_lower_bound(total_wins, total_trades),
            "ghost_profit": sum(s["ghost_profit"] for s in strategies),
            "pooled_ghost_trades": pooled_total_trades,
            "pooled_ghost_wins": pooled_total_wins,
            "pooled_ghost_profit": sum(s["pooled_ghost_profit"] for s in strategies),
            "pooled_trades_per_day": pooled_rate,
            "permanently_blocked": sum(
                1 for s in strategies if s.get("permanently_blocked")
            ),
            "live_approved": sum(1 for s in strategies if s["live_approved"]),
            "ledger_span_days": span_days,
            "trades_per_day": rate,
            "eta_days_to_first_graduation": eta,
        },
        "strategies": strategies,
    }


def render(report: dict) -> str:
    out = []
    crit = report["criteria"]
    tot = report["totals"]

    out.append("=" * 72)
    out.append("  AUTONOMOUS TRADING READINESS")
    out.append("=" * 72)
    out.append("")
    out.append(f"  Graduation requires: {crit['min_trades']} ghost trades, "
               f"{crit['min_winrate']:.0%} win rate, P/L > {crit['min_profit']}")
    out.append(f"  Enforced: {crit['enforced']}   (per strategy, independently)")
    out.append("")
    # The column is headed TRADE-ABLE, not "ghost", because that is what the
    # bar counts and the two differ by ~20x on this ledger. A column headed
    # "ghost" next to a pooled number nobody printed is how the pooled book
    # came to be read as progress toward a bar that never counted it.
    out.append(f"  {'strategy':<26}{'tradebl':>8}{'pooled':>8}{'win%':>7}"
               f"{'95%lo':>7}{'P/L':>11}  status")
    out.append("  " + "-" * 76)
    for s in report["strategies"]:
        status = ("LIVE" if s["live_approved"]
                  else "READY" if s["ready"] else s["blockers"][0])
        out.append(
            f"  {s['id']:<26}{s['ghost_trades']:>8}{s['pooled_ghost_trades']:>8}"
            f"{s['win_rate'] * 100:>6.0f}%{s['win_rate_lower_95'] * 100:>6.0f}%"
            f"{s['ghost_profit']:>+11.4f}  {status}"
        )

    out.append("")
    out.append(f"  Totals: {tot['ghost_trades']} ghost trades across "
               f"{tot['strategies']} strategies, {tot['win_rate']:.0%} win "
               f"(95% lower bound {tot['win_rate_lower_95']:.0%})")
    out.append(f"  Ghost P/L: {tot['ghost_profit']:+.4f}")
    out.append(f"  Live-approved strategies: {tot['live_approved']}")
    out.append("")

    if tot["ledger_span_days"] > 0:
        out.append(f"  Evidence rate: {tot['trades_per_day']:.2f} ghost trades/day "
                   f"over {tot['ledger_span_days']:.1f} days")
    if tot["eta_days_to_first_graduation"]:
        out.append(f"  Nearest graduation at this rate: "
                   f"~{tot['eta_days_to_first_graduation']:.0f} days")

    out.append("")
    out.append("  VERDICT")
    if tot["live_approved"]:
        out.append(f"    {tot['live_approved']} strategy(ies) have graduated and may "
                   f"trade live.")
    else:
        out.append("    Not ready. No strategy has met the bar.")
        # The honest constraint, stated plainly.
        if tot["ghost_trades"] < crit["min_trades"]:
            out.append(f"    The whole ledger holds {tot['ghost_trades']} trades; "
                       f"one strategy alone needs {crit['min_trades']}.")
        if tot["trades_per_day"] < 1.0:
            out.append("    The binding constraint is trade FREQUENCY, not the "
                       "model. More evidence is the only thing that fixes it.")
    out.append("")
    out.append("=" * 72)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("SECURE_ENV_HYDRATED", "1")
    report = collect()
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
