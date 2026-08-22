#!/usr/bin/env python3
"""
Autonomous-trading readiness: what the ghost record actually supports.

The question this answers is "can it trade on its own yet, and if not, what is
missing and when will it arrive" -- from the evidence, not from a feeling that
it has been long enough.

It is deliberately a *report*, not a switch. The graduation machinery already
exists and already works per strategy; what has been missing is an honest read
of whether the evidence has accumulated, and at what rate.

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
        trades = int(ghost.get("trades", 0))
        wins = int(ghost.get("wins", 0))
        profit = float(ghost.get("total_profit", 0.0))
        last_ts = float(ghost.get("last_ts", 0.0))
        if last_ts:
            stamps.append(last_ts)

        win_rate = wins / trades if trades else 0.0
        lower = wilson_lower_bound(wins, trades)

        blockers = []
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
            "live_approved": bool(entry.get("live_approved")),
            "live_trades": int(live.get("trades", 0)),
            "demotions": int(entry.get("demotions", 0)),
            "last_trade_age_days": (now - last_ts) / 86400 if last_ts else None,
            "blockers": blockers,
            "ready": not blockers,
        })

    total_trades = sum(s["ghost_trades"] for s in strategies)
    total_wins = sum(s["ghost_wins"] for s in strategies)
    span_days = ((max(stamps) - min(stamps)) / 86400) if len(stamps) > 1 else 0.0
    rate = (total_trades / span_days) if span_days > 0 else 0.0

    # Days until the closest strategy graduates, at the observed rate. Per
    # strategy, because graduation is per strategy -- the aggregate rate is
    # split across however many are trading.
    eta = None
    if rate > 0 and strategies:
        active = max(1, sum(1 for s in strategies if s["ghost_trades"] > 0))
        per_strategy_rate = rate / active
        shortfalls = [
            (min_trades - s["ghost_trades"]) / per_strategy_rate
            for s in strategies
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

    out.append(f"  {'strategy':<26}{'ghost':>7}{'win%':>7}{'95%lo':>7}"
               f"{'P/L':>11}  status")
    out.append("  " + "-" * 68)
    for s in report["strategies"]:
        status = ("LIVE" if s["live_approved"]
                  else "READY" if s["ready"] else s["blockers"][0])
        out.append(
            f"  {s['id']:<26}{s['ghost_trades']:>7}"
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
