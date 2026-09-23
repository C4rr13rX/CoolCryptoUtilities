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
import re
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


SYSTEM_LOG = ROOT / "logs" / "system.log"
_LOG_TS_RE = re.compile(rb"(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)")


def _production_procs() -> Optional[List[float]]:
    """Start times of every live ``main.py --action start_production``.

    Returns None -- an UNKNOWN, never an empty list -- when psutil is missing.
    A absent dependency must not be reported as a dead host: that is the exact
    confusion this whole function exists to remove.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    started: List[float] = []
    for proc in psutil.process_iter(["cmdline", "create_time"]):
        try:
            if "start_production" in " ".join(proc.info.get("cmdline") or []):
                started.append(float(proc.info.get("create_time") or 0.0))
        except Exception:
            continue
    return started


def _log_silent_minutes(now: float) -> Optional[float]:
    """Minutes since the newest line in logs/system.log, or None if unreadable.

    Read by seeking to the end: the file is ~1GB and this runs on the ladder.
    """
    try:
        size = SYSTEM_LOG.stat().st_size
        with SYSTEM_LOG.open("rb") as fh:
            fh.seek(max(0, size - 65536))
            tail = fh.read().splitlines()
        # Lines are '[2026-09-23 07:22:14] [INFO] ...' -- the stamp is NOT at
        # offset 0, and slicing [:19] silently yields an unparseable string and
        # a None that reads as 'log unreadable'. Match it instead.
        for line in reversed(tail):
            m = _LOG_TS_RE.search(line[:48])
            if not m:
                continue
            try:
                stamp = time.mktime(time.strptime(m.group(1).decode(), "%Y-%m-%d %H:%M:%S"))
            except Exception:
                continue
            return max(0.0, (now - stamp) / 60.0)
    except Exception:
        return None
    return None


def check_feed(now: float) -> Link:
    """Is there a tick? -- but FIRST, is there a process to produce one?

    These are different failures with opposite fixes and this link used to
    print the same eight words for both. Measured over the 528 hours to
    2026-09-23 06:00: 60 hours had no production log line at all, and a
    FURTHER 54 hours -- including an unbroken 60-hour run from 2026-09-13
    11:00 -- had production alive and logging with ZERO market_stream rows.
    "no ticks in 10min" was the verdict for all 114, and it cost three wrong
    readings in pass 679 alone. So:

      no process          -> UNKNOWN. With nothing running there is no feed
                             measurement to report, and reporting one invites
                             a fix aimed at the feed when the host is down.
      process, no ticks   -> FAIL, and say how long the process has been up,
                             because a 2-minute-old process has not had time
                             to fill a 10-minute window.
      process and ticks   -> PASS, worded exactly as it always was.
    """
    procs = _production_procs()

    if procs is not None and not procs:
        silent = _log_silent_minutes(now)
        detail = "PRODUCTION IS NOT RUNNING -- no feed verdict"
        if silent is not None:
            detail += " (log silent %.0f min)" % silent
        return link_unknown_process(detail)

    try:
        c = _db()
        fresh = list(c.execute(
            "SELECT COUNT(DISTINCT symbol) FROM market_stream WHERE ts > ?", (now - 900,)
        ))[0][0]
        total = list(c.execute("SELECT COUNT(*) FROM market_stream WHERE ts > ?", (now - 600,)))[0][0]
    except Exception as exc:
        return Link(1, "FEED").unknown("db unreadable: %s" % exc)

    link = Link(1, "FEED")
    if procs is None:
        note = " (process state unknown: psutil missing)"
    elif len(procs) > 1:
        note = " (%d production processes alive)" % len(procs)
    else:
        note = ""

    if total <= 0:
        up = ""
        if procs:
            up = ", production up %.0f min" % ((now - max(procs)) / 60.0)
        return link.failed(
            "process ALIVE but no ticks in 10min%s%s" % (up, note),
            "feed is starved, not dead-host: check MarketDataStream / endpoint allowlist")
    return link.passed(
        "%d ticks/10min across %d fresh symbols%s" % (total, fresh, note))


def link_unknown_process(detail: str) -> Link:
    link = Link(1, "FEED")
    link.fix = ("start production (an external supervisor normally does this); "
                "60 of the last 528 hours had no production process at all")
    return link.unknown(detail)


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
            "SELECT symbol, quantity FROM balances WHERE wallet='guardian' AND chain='base'"))
    except Exception as exc:
        return link.unknown("db unreadable: %s" % exc)
    stable = 0.0
    # SUM `quantity`, NOT `usd_amount`. For a stable the quantity IS the dollar
    # value, and it is the only one of the two fields that tracks the chain --
    # `usd_amount` is a cached mark that freezes while the quantity moves
    # underneath it. balances.py:574-594 documents the same defect measured
    # 2026-09-04, and web/tradingagent/engine.py:134-137 already applies this
    # exact rule for exactly this reason. This line was the last stable total
    # in the repo still reading the stale field.
    #
    # Measured 2026-09-22, base block 51666177, wallet
    # 0x291c854811e92906a658fb94aa511bf919f968ad: the USDC row carried
    # quantity 14.662954 -- matching an on-chain balanceOf to four decimals --
    # against usd_amount 18.18649773, an implied $1.2403 for a dollar
    # stablecoin. USDT on the same snapshot implied exactly $1.000000. So this
    # link printed "$23.18 deployable stable" for a wallet that could spend
    # $19.66, overstating it by $3.52 (17.9%). At the $10 clip the cost work
    # assumes, $23.18 reads as room for two positions and $19.66 is room for
    # one.
    for symbol, quantity in rows:
        if str(symbol).upper() in {"USDC", "USDT", "DAI", "USDBC"}:
            try:
                stable += float(quantity or 0.0)
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
    # Measure the condition the SELECTOR actually computes, not one of its two
    # terms. selector.build() and the reconcile path both strip the flag only
    # when `not _readiness_permits_live(readiness) AND not _operator_armed_live()`
    # -- probing _readiness_permits_live alone reported link 8 FAILED while the
    # bots were in fact being built live-capable off an explicit
    # ENABLE_LIVE_TRADING=1. A ladder rung that measures a different predicate
    # than the code it describes is worse than no rung.
    permits = GhostTradingSupervisor._readiness_permits_live(readiness)
    armed = GhostTradingSupervisor._operator_armed_live()
    if not permits and not armed:
        return link.failed(
            "bots built with live_trading_enabled=False (ready=%s ghost_ready=%s)"
            % (readiness.get("ready"), readiness.get("ghost_ready")),
            "selector._readiness_permits_live must accept a graduated strategy",
        )
    if not permits and armed:
        # The flag survives, but bot.py still gates every entry on
        # _strategy_live_approved -> strategy_ledger.is_live_approved(sid).
        # Say which of the two is carrying the rung, so this never reads as
        # earned graduation.
        return link.passed(
            "bots built live-capable on an EXPLICIT operator arm "
            "(ENABLE_LIVE_TRADING=1), NOT on ghost evidence "
            "(ghost_ready=%s reason=%s) -- entries still require a "
            "ledger-approved strategy"
            % (readiness.get("ghost_ready"), readiness.get("ghost_reason"))
        )
    return link.passed("bots may trade live (ghost_reason=%s)" % readiness.get("ghost_reason"))


#: Statuses that mean money actually moved. ``LIKE 'live%'`` also matches
#: live-entry-blocked / live-entry-failed / live-dry-run-entry -- refusals and
#: simulations, which would have declared this link passed on the evidence that
#: it was failing.
EXECUTED_LIVE_STATUSES = ("live-entry", "live-exit")


def check_live(now: float) -> Link:
    link = Link(9, "LIVE")
    try:
        c = _db()
        placeholders = ",".join("?" * len(EXECUTED_LIVE_STATUSES))
        total = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status IN (%s)" % placeholders,
            EXECUTED_LIVE_STATUSES))[0][0]
        refused = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status IN "
            "('guard-blocked-live','live-entry-blocked','live-entry-failed',"
            "'live-entry-unfunded','live-entry-below-profit-floor') AND ts > ?",
            (now - 3600,)))[0][0]
    except Exception as exc:
        return link.unknown("db unreadable: %s" % exc)
    if total <= 0:
        detail = "no live trades yet"
        if refused:
            detail += " (%d live entries REFUSED in 1h)" % refused
        return link.failed(detail, "fix the first FAIL above")
    return link.passed("%d executed live rows" % total)


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
