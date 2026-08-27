"""Prove, link by link, what stands between here and a profitable live trade.

Every gate can PASS and still produce zero live trades, because passing the
gates is not the same as the machinery being wired. Observed 2026-08-27: all
six risk gates PASS with block_reason empty, and no live row was ever written
-- bots were being CONSTRUCTED with live_trading_enabled=False from the
degenerate model metric, so the execution path could not run at all.

This walks the whole chain and reports, for each link, a verdict backed by a
number:

    1  FEED        corroborated prices exist
    2  SIGNALS     candidates are being produced
    3  GHOST       positions open and close
    4  LEDGER      outcomes reach the promotion record
    5  GRADUATION  a strategy is approved for live
    6  RISK        the transition plan permits capital
    7  WALLET      deployable funds exist
    8  EXECUTOR    a bot exists that CAN place a live order
    9  LIVE        real trades have happened
   10  PROFIT      those trades made money

Run it on a schedule. The first FAIL is the thing to fix; everything after it
is unknowable until then.

    python scripts/live_path_check.py
    python scripts/live_path_check.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB_PATH = "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db")


class Link:
    def __init__(self, index: int, name: str) -> None:
        self.index = index
        self.name = name
        self.ok: Optional[bool] = None      # None = unknown, not False
        self.detail = "--"
        self.fix = ""

    def passed(self, detail: str) -> "Link":
        self.ok, self.detail = True, detail
        return self

    def failed(self, detail: str, fix: str = "") -> "Link":
        self.ok, self.detail, self.fix = False, detail, fix
        return self

    def unknown(self, detail: str) -> "Link":
        self.ok, self.detail = None, detail
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.index, "name": self.name, "ok": self.ok,
                "detail": self.detail, "fix": self.fix}


def _db():
    return sqlite3.connect(DB_PATH, uri=True)


def check_feed(now: float) -> Link:
    link = Link(1, "FEED")
    try:
        c = _db()
        fresh = list(c.execute(
            "SELECT COUNT(DISTINCT symbol) FROM market_stream WHERE ts > ?", (now - 900,)
        ))[0][0]
        total = list(c.execute("SELECT COUNT(*) FROM market_stream WHERE ts > ?", (now - 600,)))[0][0]
    except Exception as exc:
        return link.unknown("db unreadable: %s" % exc)
    if total <= 0:
        return link.failed("no ticks in 10min", "check MarketDataStream / endpoint allowlist")
    return link.passed("%d ticks/10min across %d fresh symbols" % (total, fresh))


def check_signals(now: float) -> Link:
    link = Link(2, "SIGNALS")
    try:
        from services.atf_static_strategy import latest_signals
        signals = latest_signals(1800)
    except Exception as exc:
        return link.unknown("signal source unavailable: %s" % exc)
    if not signals:
        return link.failed("no candidates in 30min", "check discovery + strategy refresh")
    return link.passed("%d candidates" % len(signals))


def check_ghost(now: float) -> Link:
    link = Link(3, "GHOST")
    try:
        c = _db()
        entries = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status='ghost-entry' AND ts > ?",
            (now - 3600,)))[0][0]
        exits = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status='ghost-exit' AND ts > ?",
            (now - 3600,)))[0][0]
    except Exception as exc:
        return link.unknown("db unreadable: %s" % exc)
    if entries <= 0 and exits <= 0:
        return link.failed("no ghost activity in 1h",
                           "check position cap / feed corroboration refusals")
    return link.passed("%d entries, %d exits in 1h" % (entries, exits))


def check_ledger() -> Link:
    link = Link(4, "LEDGER")
    try:
        from trading.strategies.ledger import StrategyLedger
        data = dict(StrategyLedger()._data or {})
    except Exception as exc:
        return link.unknown("ledger unreadable: %s" % exc)
    if not data:
        return link.failed("ledger empty",
                           "outcomes are not reaching StrategyLedger.record()")
    rows = []
    for sid, entry in data.items():
        ghost = entry.get("ghost") or {}
        rows.append("%s %d/%d" % (sid, ghost.get("wins", 0), ghost.get("trades", 0)))
    return link.passed("; ".join(rows))


def check_graduation() -> Link:
    link = Link(5, "GRADUATION")
    try:
        from trading.strategies.ledger import StrategyLedger
        data = dict(StrategyLedger()._data or {})
    except Exception as exc:
        return link.unknown("ledger unreadable: %s" % exc)
    approved = [sid for sid, e in data.items() if e.get("live_approved")]
    if not approved:
        return link.failed("no strategy approved for live",
                           "needs STRATEGY_GRADUATION_MIN_TRADES / MIN_WINRATE")
    return link.passed("approved: %s" % ", ".join(approved))


def check_risk() -> Link:
    link = Link(6, "RISK")
    try:
        import trading.data_loader as dl
        dl.HistoricalDataLoader._load_news = lambda self: []
        from trading.pipeline import TrainingPipeline
        from db import get_db
        plan = TrainingPipeline(db=get_db())._build_transition_plan()
    except Exception as exc:
        return link.unknown("plan unavailable: %s" % type(exc).__name__)
    flags = plan.get("risk_flags") or {}
    reason = flags.get("live_blocked_reason") or ""
    usd = float(flags.get("recommended_live_usd") or 0.0)
    if reason:
        return link.failed("blocked: %s" % reason, "see scripts/live_gate_map.py")
    if usd <= 0:
        return link.failed("recommended_live_usd = $0.00",
                           "sizing collapsed; check SAVINGS_/LIVE_MIN_CLIP_USD")
    return link.passed("permits $%.2f" % usd)


def check_wallet() -> Link:
    link = Link(7, "WALLET")
    try:
        c = _db()
        rows = list(c.execute(
            "SELECT symbol, usd_amount FROM balances WHERE wallet='guardian' AND chain='base'"))
    except Exception as exc:
        return link.unknown("db unreadable: %s" % exc)
    stable = 0.0
    for symbol, usd in rows:
        if str(symbol).upper() in {"USDC", "USDT", "DAI"}:
            try:
                stable += float(usd or 0.0)
            except (TypeError, ValueError):
                pass
    if stable <= 0.35:
        return link.failed("deployable stable $%.2f" % stable,
                           "fund the wallet or rebalance ETH -> USDC")
    return link.passed("$%.2f deployable stable" % stable)


def check_executor() -> Link:
    """Does a bot exist that CAN place a live order?

    The link that was silently broken: bots were CONSTRUCTED with
    live_trading_enabled=False from the degenerate model metric, so no amount
    of graduation could reach execution.
    """
    link = Link(8, "EXECUTOR")
    if os.getenv("ENABLE_LIVE_TRADING", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return link.failed("ENABLE_LIVE_TRADING is off", "set ENABLE_LIVE_TRADING=1")
    try:
        from trading.selector import GhostTradingSupervisor
        import trading.data_loader as dl
        dl.HistoricalDataLoader._load_news = lambda self: []
        from trading.pipeline import TrainingPipeline
        from db import get_db
        readiness = TrainingPipeline(db=get_db()).live_readiness_report()
    except Exception as exc:
        return link.unknown("readiness unavailable: %s" % type(exc).__name__)
    permits = GhostTradingSupervisor._readiness_permits_live(readiness)
    if not permits:
        return link.failed(
            "bots built with live_trading_enabled=False (ready=%s ghost_ready=%s)"
            % (readiness.get("ready"), readiness.get("ghost_ready")),
            "selector._readiness_permits_live must accept a graduated strategy",
        )
    return link.passed("bots may trade live (ghost_reason=%s)" % readiness.get("ghost_reason"))


def check_live(now: float) -> Link:
    link = Link(9, "LIVE")
    try:
        c = _db()
        total = list(c.execute("SELECT COUNT(*) FROM trading_ops WHERE status LIKE 'live%'"))[0][0]
    except Exception as exc:
        return link.unknown("db unreadable: %s" % exc)
    if total <= 0:
        return link.failed("no live trades yet", "fix the first FAIL above")
    return link.passed("%d live rows" % total)


def check_profit() -> Link:
    link = Link(10, "PROFIT")
    try:
        from services import strategy_registry
        rows = strategy_registry.list_strategies()
    except Exception as exc:
        return link.unknown("registry unreadable: %s" % exc)
    total = 0.0
    trades = 0
    for row in rows:
        live = ((row.get("lifetime") or {}).get("live")) or {}
        try:
            total += float(live.get("total_profit") or 0.0)
            trades += int(live.get("trades") or 0)
        except (TypeError, ValueError):
            continue
    if trades <= 0:
        return link.unknown("no live trades to measure")
    if total <= 0:
        return link.failed("live P/L %+.4f over %d trades" % (total, trades),
                           "demotion guards should cut this off")
    return link.passed("live P/L %+.4f over %d trades" % (total, trades))


def run() -> List[Link]:
    now = time.time()
    return [
        check_feed(now), check_signals(now), check_ghost(now), check_ledger(),
        check_graduation(), check_risk(), check_wallet(), check_executor(),
        check_live(now), check_profit(),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    links = run()
    if args.json:
        print(json.dumps([l.to_dict() for l in links], indent=2))
    else:
        print("\n" + "=" * 78)
        print("PATH TO A PROFITABLE LIVE TRADE")
        print("=" * 78)
        for link in links:
            mark = "PASS " if link.ok else ("FAIL " if link.ok is False else "  -- ")
            print("  [%s] %-2d %-11s %s" % (mark, link.index, link.name, link.detail))
            if link.ok is False and link.fix:
                print("            -> %s" % link.fix)
        first = next((l for l in links if l.ok is False), None)
        print("-" * 78)
        if first is None:
            print("  Every link passes.")
        else:
            print("  NEXT: step %d (%s) -- %s" % (first.index, first.name, first.detail))
            print("  Everything after it is unknowable until this is fixed.")
        print("=" * 78)
    return 0 if all(l.ok is not False for l in links) else 1


if __name__ == "__main__":
    sys.exit(main())
