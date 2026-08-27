"""Show every condition between a strategy and a live trade, and which block.

Written after a session spent clearing gates one at a time -- each discovered
only after the previous one was fixed:

    insufficient_accuracy -> ghost_validation_block -> loss_streak_block
    -> tail_block -> min_clip -> (risk multiplier zeroed) -> ...

Nothing showed the whole chain, so every fix looked like the last one. This
evaluates all of them together against live state and prints PASS/BLOCK with
the actual number, the guardrail, and the env var that moves it.

    python scripts/live_gate_map.py            # evaluate current state
    python scripts/live_gate_map.py --static   # just the reference table
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Layer -> conditions. Each row:
#   (condition, what it gates, env var, default, where)
GATE_MAP = [
    (
        "1. FEED — does a price exist that can be trusted?",
        [
            ("symbol has a streamed tick", "entry+exit refused without one",
             "ATF_STATIC_REQUIRE_FEED_PRICE", "1", "services/atf_static_strategy.py:_corroborated_price"),
            ("tick age <= max", "stale tick is not corroboration",
             "ATF_STATIC_FEED_MAX_AGE_SEC", "900", "services/atf_static_strategy.py:_feed_price"),
            ("quote agrees with feed", "rejects denomination mismatch (MAMO 0.0164 vs 0.1823)",
             "ATF_STATIC_MAX_FEED_DEV", "0.35", "services/atf_static_strategy.py:_corroborated_price"),
            ("no synthetic ticks recorded", "fabricated prices never enter the DB",
             "ALLOW_SYNTHETIC_TICKS", "0", "trading/data_stream.py:_dispatch"),
        ],
    ),
    (
        "2. STRATEGY LEDGER — has this strategy earned promotion?",
        [
            ("ghost trades >= N", "per-strategy graduation",
             "STRATEGY_GRADUATION_MIN_TRADES", "20", "trading/strategies/ledger.py"),
            ("ghost win rate >= X", "per-strategy graduation",
             "STRATEGY_GRADUATION_MIN_WINRATE", "0.55", "trading/strategies/ledger.py"),
            ("ledger profit > 0", "per-strategy graduation",
             "(hardcoded)", "> 0", "trading/strategies/ledger.py"),
            ("outcome is plausible", "rejects repricing artifacts",
             "STRATEGY_MAX_TRADE_PROFIT_MULTIPLE", "25.0", "trading/strategies/ledger.py:_is_implausible"),
            ("demote on live losses", "cuts a losing strategy off",
             "STRATEGY_DEMOTE_MAX_LIVE_LOSSES", "2", "trading/strategies/ledger.py"),
        ],
    ),
    (
        "3. GHOST VALIDATION — is the aggregate ghost book safe?",
        [
            ("samples >= min_trades", "reason=insufficient_samples",
             "GHOST_MIN_TRADES", "25", "trading/pipeline.py:_ghost_validation"),
            ("win_rate >= min", "reason=low_win_rate (win-rate path)",
             "MIN_GHOST_WIN_RATE", "0.55", "trading/pipeline.py:_ghost_validation"),
            ("win_rate_lb >= min", "reason=fast_track path",
             "GHOST_FAST_TRACK_MIN_WIN_RATE_LB", "= min_win_rate", "trading/pipeline.py:_ghost_validation"),
            ("net expectancy > 0 after fees", "reason=positive_expectancy path",
             "GHOST_EXPECTANCY_FEE_RATE", "0.0065", "trading/pipeline.py:_ghost_validation"),
            ("profit_factor >= min", "expectancy path bar (HIGHER than win-rate path)",
             "GHOST_EXPECTANCY_MIN_PROFIT_FACTOR", "1.5", "trading/pipeline.py:_ghost_validation"),
            ("payoff_ratio >= min", "expectancy path bar",
             "GHOST_EXPECTANCY_MIN_PAYOFF", "2.0", "trading/pipeline.py:_ghost_validation"),
            ("tail_risk (ES95) <= guard", "reason=tail_risk / tail_block",
             "GHOST_TAIL_GUARDRAIL", "0.08", "trading/pipeline.py:_ghost_validation"),
            ("loss streak COST <= guard", "reason=loss_streak (magnitude, not count)",
             "GHOST_MAX_LOSS_STREAK_COST", "0.25", "trading/pipeline.py:_ghost_validation"),
            ("loss streak length <= guard", "only when the streak is costly",
             "GHOST_MAX_LOSS_STREAK", "5", "trading/pipeline.py:_ghost_validation"),
            ("loss_rate <= guard", "reason=loss_rate",
             "GHOST_MAX_LOSS_RATE", "0.6", "trading/pipeline.py:_ghost_validation"),
            ("max_drawdown <= guard", "reason=drawdown",
             "GHOST_MAX_DRAWDOWN", "0 (off)", "trading/pipeline.py:_ghost_validation"),
            ("symbol dominance <= guard", "reason=symbol_concentration",
             "GHOST_MAX_SYMBOL_DOMINANCE", "0.82", "trading/pipeline.py:_ghost_validation"),
            ("book not stale", "reason=stale_ghost_book",
             "GHOST_MAX_STALE_SEC", "86400", "trading/pipeline.py:_ghost_validation"),
        ],
    ),
    (
        "4. MODEL READINESS — the classifier gate (degenerate here)",
        [
            ("precision >= target", "readiness.ready; reads 0.0 on 639 samples",
             "LIVE_READY_PRECISION", "0.55", "trading/pipeline.py:live_readiness_report"),
            ("recall >= target", "readiness.ready",
             "LIVE_READY_RECALL", "0.50", "trading/pipeline.py:live_readiness_report"),
            ("mini gate may substitute", "allows a weaker pass",
             "LIVE_ALLOW_MINI_READY", "1", "trading/pipeline.py:live_readiness_report"),
            ("BYPASS for graduated ghost", "live_ready=True on ghost evidence alone",
             "LIVE_REQUIRE_MODEL_READY", "0 (bypass on)", "trading/pipeline.py:_live_ready_flag"),
        ],
    ),
    (
        "5. TRANSITION PLAN — sizing the first real trade",
        [
            ("live_ready", "safe_to_live",
             "(from layer 4)", "-", "trading/pipeline.py:_build_transition_plan"),
            ("ghost_collection_ready", "ZEROES risk multiplier when false",
             "(auto-true if ghost_ready)", "-", "trading/pipeline.py:_build_transition_plan"),
            ("recommended_usd >= min clip", "block_reason=min_clip",
             "LIVE_MIN_CLIP_USD", "10", "trading/pipeline.py:_build_transition_plan"),
            ("allocation slice", "drives recommended_usd",
             "SAVINGS_READY_RATIO", "0.15", "trading/pipeline.py:_build_transition_plan"),
            ("bootstrap slice", "pre-equilibrium allocation",
             "SAVINGS_BOOTSTRAP_RATIO", "0.05", "trading/pipeline.py:_build_transition_plan"),
            ("first tranche cap", "caps the opening trade",
             "LIVE_FIRST_TRANCHE_USD", "50", "trading/pipeline.py:_build_transition_plan"),
            ("total live cap", "caps cumulative live exposure",
             "LIVE_MAX_BOOTSTRAP_USD", "150", "trading/pipeline.py:_build_transition_plan"),
            ("capital deficit <= 0", "block_reason=capital_deficit",
             "MIN_LIVE_CAPITAL_USD", "-", "trading/pipeline.py:_build_transition_plan"),
            ("native gas buffer", "block_reason=native_gas_starved",
             "BUS_NATIVE_RESERVE", "-", "trading/pipeline.py:_build_transition_plan"),
        ],
    ),
    (
        "6. EXECUTION — actually spending money",
        [
            ("live trading enabled", "bot.live_trading_enabled",
             "ENABLE_LIVE_TRADING", "0", "trading/selector.py / trading/bot.py"),
            ("auto-execute on graduation", "flips real execution on",
             "AUTO_EXECUTE_ON_GRADUATION", "1", "trading/bot.py:_refresh_auto_execute"),
            ("per-strategy gate enforced", "only graduated strategies trade live",
             "STRATEGY_GRADUATION_ENFORCED", "1", "trading/bot.py:_strategy_live_approved"),
            ("dry run off", "status=live-dry-run-entry when on",
             "LIVE_TRADES_DRY_RUN", "(unset)", "trading/bot.py"),
            ("swap guard passes", "status=guard-blocked",
             "(swap_validator)", "-", "trading/swap_validator.py"),
            ("NOTE: atf_static publishes signals only", "TradingBot executes; production.py runs the supervisor",
             "-", "-", "services/atf_static_strategy.py:551"),
        ],
    ),
]


def print_static():
    for layer, rows in GATE_MAP:
        print("\n" + "=" * 100)
        print(layer)
        print("=" * 100)
        print("  %-34s %-42s %s" % ("CONDITION", "GATES", "ENV VAR"))
        print("  " + "-" * 96)
        for cond, gates, env, default, _where in rows:
            print("  %-34s %-42s %s=%s" % (cond[:34], gates[:42], env, default))


def print_live():
    import trading.data_loader as dl
    dl.HistoricalDataLoader._load_news = lambda self: []
    from trading.pipeline import TrainingPipeline
    from db import get_db

    p = TrainingPipeline(db=get_db())
    g = p._ghost_validation()
    plan = p._build_transition_plan()
    rf = plan.get("risk_flags") or {}

    def row(name, value, guard, ok):
        flag = "PASS" if ok else "BLOCK"
        print("  [%-5s] %-30s %-18s (limit %s)" % (flag, name, value, guard))

    print("\n" + "=" * 100)
    print("LIVE GATE STATE")
    print("=" * 100)

    tg = float(g.get("tail_guardrail", 0.08))
    tr = float(g.get("tail_risk", 0.0))
    row("tail_risk (ES95)", "%.4f" % tr, tg, tr <= tg)

    els = int(g.get("effective_loss_streak", g.get("max_loss_streak", 0)))
    lsg = int(g.get("loss_streak_guardrail", 5))
    row("effective_loss_streak", els, lsg, els <= lsg)

    pf = float(g.get("profit_factor", 0.0))
    row("profit_factor", "%.3f" % pf, ">= 1.5", pf >= 1.5)

    pr = float(g.get("payoff_ratio", 0.0))
    row("payoff_ratio", "%.3f" % pr, ">= 2.0", pr >= 2.0)

    ne = float(g.get("net_expectancy", 0.0))
    row("net_expectancy (post-fee)", "%+.5f" % ne, "> 0", ne > 0)

    lr = float(g.get("loss_rate", 0.0))
    lrg = float(g.get("loss_rate_guardrail", 0.6))
    row("loss_rate", "%.3f" % lr, lrg, lr <= lrg)

    print("  %-8s %-30s %s" % ("", "ghost_validation", "%s (%s)" % (g.get("ready"), g.get("reason"))))
    print("  %-8s %-30s %s" % ("", "live_ready (model gate)", plan.get("live_ready")))
    print("  %-8s %-30s %s" % ("", "live_mode", plan.get("live_mode")))
    print("  %-8s %-30s %s" % ("", "block_reason", rf.get("live_blocked_reason") or "(none)"))
    print("  %-8s %-30s $%.4f" % ("", "recommended_live_usd", float(rf.get("recommended_live_usd", 0.0))))
    print("  %-8s %-30s $%s" % ("", "min_clip_usd", rf.get("min_clip_usd")))
    print("  %-8s %-30s $%s" % ("", "deployable_stable_usd", rf.get("deployable_stable_usd")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true", help="reference table only")
    args = ap.parse_args()
    print_static()
    if not args.static:
        try:
            print_live()
        except Exception as exc:
            print("\n(live evaluation unavailable: %s: %s)" % (type(exc).__name__, exc))
