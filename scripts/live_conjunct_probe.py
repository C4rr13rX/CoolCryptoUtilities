"""[4d095eda] criterion 3: print the TWO booleans of the live entry gate SEPARATELY.

trading/bot.py gates every live entry on the conjunction

    self.live_trading_enabled and self._strategy_live_approved(directive)

at :7303, :7579, :7756 and :9282. The ladder reports step 8 as "bots built
with live_trading_enabled=False" and step 5 as "no strategy approved for
live", and three passes have argued about which is the real one. They are
two DIFFERENT booleans and nobody has printed them side by side.

WHY THIS IS NOT A RESTART: `live_trading_enabled` is
`_live_trading_armed and not _live_halt_active()`, and `_live_halt_active`
(bot.py:1930-1943) reads NOTHING but `_transition_plan['risk_flags']
['halt_live']`. Every published plan writes that flag to trading_ops, so the
plan half is measurable from the store. `_strategy_live_approved` is a pure
ledger read and is measurable in-process. Neither needs a live tick.

WHAT WOULD FALSIFY THE ITEM'S PREDICTION: the item predicts
live_trading_enabled is the False one. If the newest published plan has
halt_live False -- or if no plan has been published recently enough to
speak for now -- then step 8's label is stale and the block is the ledger.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = os.getenv("TRADING_DB_PATH", "storage/trading_cache.db")
APPROVED = os.getenv("PROBE_STRATEGY", "money_button")


def newest_plan_row(conn: sqlite3.Connection):
    """Newest trading_ops row that PUBLISHED a plan, with its halt_live."""
    cur = conn.cursor()
    # THE FLAG IS FOUR LEVELS DEEP and I got this wrong twice before reading
    # my own p720 census: it is at
    #   details['brain']['live_transition']['plan']['risk_flags']['halt_live']
    # NOT at details['risk_flags'] and NOT at details['live_transition'].
    # Both wrong paths return ZERO rows and print as "no plan was published",
    # which is the same shape as a real stall. `ghost_ready` is a sibling of
    # 'plan', not a member of risk_flags.
    cur.execute(
        "select ts, status, details from trading_ops "
        "where details like '%halt_live%' order by ts desc limit 4000"
    )
    for ts, status, details in cur.fetchall():
        try:
            d = json.loads(details)
        except Exception:
            continue
        transition = (d.get("brain") or {}).get("live_transition") or {}
        flags = (transition.get("plan") or {}).get("risk_flags")
        if not isinstance(flags, dict) or "halt_live" not in flags:
            continue
        flags = dict(flags)
        flags["_ghost_ready"] = transition.get("ghost_ready")
        return float(ts), str(status), flags
    return None, None, None


def main() -> int:
    conn = sqlite3.connect(DB)
    now = time.time()

    print("=" * 74)
    print("BOOLEAN 1 -- live_trading_enabled  = _live_trading_armed AND NOT halt_live")
    print("=" * 74)
    armed_env = os.getenv("ENABLE_LIVE_TRADING", "")
    print(f"  ENABLE_LIVE_TRADING (env, sets _live_trading_armed) = {armed_env!r}")

    ts, status, flags = newest_plan_row(conn)
    if ts is None:
        print("  NEWEST PUBLISHED PLAN: none found in the last 400 risk_flags rows.")
        print("  => halt_live UNMEASURABLE from the store. Boolean 1 UNKNOWN.")
        halt = None
    else:
        age_min = (now - ts) / 60.0
        halt = bool(flags.get("halt_live"))
        print(f"  NEWEST PUBLISHED PLAN: ts={ts:.0f} age={age_min:.1f} min status={status}")
        print(f"    halt_live = {halt}")
        reason = flags.get("ghost_reason") or flags.get("reason")
        if reason:
            print(f"    ghost_reason = {reason!r}")
        fired = [k for k, v in sorted(flags.items())
                 if k != "halt_live" and isinstance(v, bool) and v]
        print(f"    other TRUE boolean flags on this plan: {fired or 'none'}")
        if age_min > 60:
            print("    WARNING: this plan is over an hour old. It describes THEN, not now.")

    armed = armed_env.strip().lower() in {"1", "true", "yes", "on"}
    if halt is None:
        b1 = None
    else:
        b1 = armed and not halt
    print(f"  => BOOLEAN 1  live_trading_enabled = {b1}")

    print()
    print("=" * 74)
    print(f"BOOLEAN 2 -- _strategy_live_approved(directive) for {APPROVED!r}")
    print("=" * 74)
    armed_live = None
    try:
        from trading.selector import GhostTradingSupervisor

        armed_live = GhostTradingSupervisor._operator_armed_live()
        print(f"  GhostTradingSupervisor._operator_armed_live() = {armed_live}")
    except Exception as exc:
        print(f"  _operator_armed_live() RAISED {exc!r} -> bot.py fails CLOSED (armed=True)")

    enforced = os.getenv("STRATEGY_GRADUATION_ENFORCED", "1")
    print(f"  STRATEGY_GRADUATION_ENFORCED = {enforced!r} (default '1')")

    b2 = None
    try:
        from trading.strategies.ledger import StrategyLedger

        ledger = StrategyLedger()
        approved = ledger.is_live_approved(APPROVED)
        print(f"  ledger.is_live_approved({APPROVED!r})       = {approved}"
              "   <- does it HOLD a licence")
        may = ledger.may_place_live_order(APPROVED)
        print(f"  ledger.may_place_live_order({APPROVED!r})   = {may}"
              "   <- the full question the money path asks")
        refusal = ledger.live_order_refusal(APPROVED)
        print(f"  ledger.live_order_refusal({APPROVED!r})     = {refusal!r}")
        b2 = bool(may)
    except Exception as exc:
        print(f"  LEDGER READ RAISED {exc!r}")
        print("  NOTE bot.py:2050 fail-open #2: is_live_approved raising returns True")

    print(f"  => BOOLEAN 2  _strategy_live_approved = {b2}")

    print()
    print("=" * 74)
    print("VERDICT -- which conjunct is the False one")
    print("=" * 74)
    print(f"  live_trading_enabled   = {b1}")
    print(f"  _strategy_live_approved = {b2}")
    if b1 is False and b2 is True:
        print("  => BOOLEAN 1 IS THE FALSE ONE. The cause is halt_live, therefore")
        print("     ghost_ready. Step 5's 'no strategy approved' label is WRONG:")
        print("     the strategy IS approved and the ledger says it may place.")
    elif b1 is True and b2 is False:
        print("  => BOOLEAN 2 IS THE FALSE ONE. The ledger refuses an approval the")
        print("     operator granted. Step 8's live_trading_enabled label is WRONG.")
    elif b1 is False and b2 is False:
        print("  => BOTH ARE FALSE. Fixing either alone moves the scoreboard by ZERO.")
    elif b1 is True and b2 is True:
        print("  => BOTH TRUE. The conjunction PASSES and the block is DOWNSTREAM of")
        print("     it -- neither step 5 nor step 8 is the first failing link.")
    else:
        print("  => NOT DECIDABLE: at least one boolean is UNKNOWN. Say so; do not")
        print("     report the prediction as a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
