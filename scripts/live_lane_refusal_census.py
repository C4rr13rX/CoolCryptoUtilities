#!/usr/bin/env python
"""Name the FIRST refusal that actually fires on the live lane, today.

Run:  python -X utf8 scripts/live_lane_refusal_census.py

WHY THIS EXISTS
---------------
On 2026-09-17 the ladder in ``scripts/graduation_status.py`` read:

    [PASS] 5  GRADUATION  approved: money_button
    [FAIL] 6  RISK        blocked: ghost_validation_block
    [PASS] 8  EXECUTOR    bots built live-capable on an EXPLICIT operator arm
    [PASS] 9  LIVE        38 executed live rows

and the live lane had not placed a trade in 11.9 days. ``ghost_gate_forensics``
showed the step-6 FAIL is the POOLED verdict while ``money_button`` separately
reads READY/``cold_start_bootstrap`` -- so the headline FAIL looked like it was
not gating the approved strategy at all.

That read is WRONG, and this script exists to show why with counts. The pooled
``ghost_validation_block`` never touches ``_strategy_live_approved``; it binds
one layer higher, through ``halt_live``:

    pipeline.py   "halt_live": ... or not ghost_ready or ...
    bot.py        if self.live_trading_enabled:
                      if plan_flags.get("halt_live"): risk_budget = 0.0
    bot.py        if risk_budget <= 0:  -> scheduler.evaluate() is NEVER CALLED
                      metrics.feedback("scheduler", label="halted", ...)

Note where the ``halt_live`` line sits: INSIDE ``if self.live_trading_enabled``.
On a ghost-only bot the same pooled verdict is inert. ARMING THE LIVE LANE IS
WHAT SWITCHES THE DECISION PATH OFF. That branch writes no ``trading_ops`` row
at all -- only a feedback event -- which is why a census of ``trading_ops``
alone shows "nothing was refused" rather than "everything was refused".

WHAT IT READS
-------------
``storage/trading_cache.db`` (env ``TRADING_DB_PATH``), read-only, via the
``ts`` indexes. Never ``data/trading.db`` / ``trading_bot.db`` /
``web/db.sqlite3`` -- those are 0-byte stubs that silently return nothing.

COUNTING CAVEATS -- read before quoting any number below
--------------------------------------------------------
* ``scheduler/halted`` is LATCHED (``if not self._scheduler_halted``). One event
  is one TRANSITION into the halt, not one halted tick. The halted-tick count is
  strictly larger and is NOT recoverable from this table. Never divide it by the
  tick count.
* ``entry-arbitration`` rows are written only for ticks that produced at least
  one ENTER candidate, and ``wallet`` is hardcoded ``'ghost'``
  (``scheduler.py:641``) -- those rows say nothing about live/ghost arming.
* ``details['skipped']`` is populated only on the ``via='trident'`` path. On
  ``via='max_score'`` the map is empty, so a strategy missing from a row is
  UNMEASURED, not "did not decline". Only the trident rows carry a verdict.
* ``details['unsat']`` is ``self._trident.last_unsat``, a sticky in-memory
  attribute. It names why the TRIDENT abstained; the tick was still allocated by
  the ``max_score`` fallback. It is not a refusal of the winner.
* On every OTHER refusal row ``wallet`` IS the arming bit: those call sites write
  ``wallet="live" if self.live_trading_enabled else "ghost"``. That is the only
  reliable live/ghost discriminator in ``trading_ops``.
"""
from __future__ import annotations

import collections
import datetime as _dt
import json
import os
import sqlite3
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DAY = 86400.0

#: Statuses written by a call site that encodes the arming bit into `wallet`.
_ARMING_SENSITIVE = (
    "entry-refused-lattice",
    "entry-refused-strategy-edge",
    "entry-refused-symbol-edge",
    "entry-refused-symbol-motion",
    "entry-refused-stop-survivability",
    "entry-refused-long-horizon-capacity",
    "entry-refused-duplicate",
    "entry-refused-slot-busy",
    "entry-refused-live-held",
)


def _db_path() -> str:
    return os.getenv("TRADING_DB_PATH") or os.path.join("storage", "trading_cache.db")


def _connect(path: str) -> sqlite3.Connection:
    size = os.path.getsize(path) if os.path.exists(path) else 0
    if size < 1_000_000:
        raise SystemExit(
            f"{path} is {size} bytes -- that is a STUB, not the trading store. "
            "Set TRADING_DB_PATH to storage/trading_cache.db."
        )
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _stamp(ts: float | None) -> str:
    if not ts:
        return "-"
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _hdr(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def section_arming(cur: sqlite3.Cursor, now: float) -> None:
    _hdr("1. IS ANY LIVE-ARMED BOT STILL MAKING ENTRY DECISIONS?")
    print("   `wallet` on a bot.py refusal row is literally")
    print("   `\"live\" if self.live_trading_enabled else \"ghost\"`, so it is the")
    print("   arming bit. (entry-arbitration / entry-predropped-edge-ban are")
    print("   excluded: scheduler.py hardcodes 'ghost' on both.)")
    print()
    qmarks = ",".join("?" * len(_ARMING_SENSITIVE))
    for wallet in ("live", "ghost"):
        row = cur.execute(
            f"select count(*), max(ts) from trading_ops "
            f"where wallet=? and status in ({qmarks})",
            (wallet, *_ARMING_SENSITIVE),
        ).fetchone()
        n, mx = row[0], row[1]
        age = f"{(now - mx) / DAY:.2f} d ago" if mx else "never"
        print(f"   wallet={wallet:<6} n={n:<6} last={_stamp(mx):<18} {age}")

    print()
    print("   Last row of ANY status per live-lane milestone:")
    for status in ("live-entry", "live-swap-settled", "live-exit",
                   "live-dry-run-entry", "guard-blocked-live",
                   "live-entry-unfunded", "live-entry-blocked"):
        row = cur.execute(
            "select count(*), max(ts) from trading_ops where status=?",
            (status,),
        ).fetchone()
        n, mx = row[0], row[1]
        age = f"{(now - mx) / DAY:.2f} d ago" if mx else "never"
        print(f"     {status:<26} n={n:<5} last={_stamp(mx):<18} {age}")


def section_ops(cur: sqlite3.Cursor, now: float) -> None:
    _hdr("2. trading_ops BY REASON -- 24h AND 7d")
    for label, cut in (("24h", now - DAY), ("7d", now - 7 * DAY)):
        rows = list(cur.execute(
            "select action, status, wallet, count(*) from trading_ops "
            "where ts>=? group by 1,2,3 order by 4 desc limit 30",
            (cut,),
        ))
        total = sum(r[3] for r in rows)
        print(f"\n   -- {label} ({total} rows) --")
        for action, status, wallet, n in rows:
            print(f"     {n:>7}  {action:<16} {status:<34} wallet={wallet}")


def section_halt(cur: sqlite3.Cursor, now: float) -> None:
    _hdr("3. THE REFUSAL THAT ACTUALLY FIRES: scheduler halted, risk_budget=0")
    print("   bot.py: `if plan_flags.get('halt_live'): risk_budget = 0.0` sits")
    print("   INSIDE `if self.live_trading_enabled:`. Then `if risk_budget <= 0`")
    print("   skips scheduler.evaluate() entirely -- no directive is produced,")
    print("   so NO entry gate downstream ever runs and NO trading_ops row is")
    print("   written. The only trace is this feedback event.")
    print()
    print("   LATCHED: one event = one TRANSITION into the halt, NOT one halted")
    print("   tick. Do not divide by a tick count.")
    for label, cut in (("24h", now - DAY), ("7d", now - 7 * DAY), ("30d", now - 30 * DAY)):
        rows = list(cur.execute(
            "select details, count(*), max(ts) from feedback_events "
            "where source='scheduler' and label='halted' and ts>=? "
            "group by details order by 2 desc limit 6",
            (cut,),
        ))
        total = sum(r[1] for r in rows)
        print(f"\n   -- {label}: {total} halt transitions --")
        for details, n, mx in rows:
            try:
                reason = json.loads(details).get("reason")
            except Exception:
                reason = str(details)[:60]
            print(f"     {n:>6}  reason={reason!r:<32} last={_stamp(mx)}")

    print()
    print("   For contrast, the SILENT downstream gates (feedback-only, no row):")
    for lab in ("entry_downgraded_to_ghost", "insufficient_quote", "entry_blocked"):
        row = cur.execute(
            "select count(*), max(ts) from feedback_events where label=?",
            (lab,),
        ).fetchone()
        n, mx = row[0], row[1]
        age = f"{(now - mx) / DAY:.2f} d ago" if mx else "never"
        print(f"     {lab:<28} n={n:<5} last={_stamp(mx):<18} {age}")


def section_money_button(cur: sqlite3.Cursor, now: float) -> None:
    _hdr("4. DOES money_button -- THE ONLY APPROVED STRATEGY -- EVER FIRE?")
    print("   `offered` counts candidates per strategy on ticks that produced at")
    print("   least one ENTER candidate. If money_button is never in `offered`,")
    print("   it never proposed a trade: the lane is SIGNAL-STARVED, not gated.")
    print("   `skipped` is recorded only on via='trident' rows -- that subset is")
    print("   the only place a per-strategy verdict is measurable.")
    for label, cut in (("24h", now - DAY), ("7d", now - 7 * DAY)):
        offered: collections.Counter = collections.Counter()
        verdict: collections.Counter = collections.Counter()
        via: collections.Counter = collections.Counter()
        rows = 0
        trident = 0
        for (details,) in cur.execute(
            "select details from trading_ops where status='entry-arbitration' and ts>=?",
            (cut,),
        ):
            try:
                blob = json.loads(details)
            except Exception:
                continue
            rows += 1
            v = str(blob.get("via") or "")
            via[(v, blob.get("unsat"))] += 1
            for sid in (blob.get("offered") or {}):
                offered[sid] += 1
            if v == "trident":
                trident += 1
                verdict[(blob.get("skipped") or {}).get(
                    "money_button", "NOT-LISTED")] += 1
        print(f"\n   -- {label}: {rows} entry-arbitration rows "
              f"({trident} carry a measurable skip map) --")
        print(f"     money_button appears in `offered` : "
              f"{offered.get('money_button', 0)} of {rows}")
        print(f"     money_button verdict on the {trident} trident rows: "
              f"{dict(verdict) or '(none)'}")
        print("     strategies that DID propose:")
        for sid, n in offered.most_common(6):
            print(f"       {sid:<28} {n}")
        print("     (via, trident unsat clause):")
        for key, n in via.most_common(5):
            print(f"       {str(key):<40} {n}")


def section_env() -> None:
    _hdr("5. THE RESOLVED ENV, VIA THE EXACT PATH PRODUCTION BOOTS THROUGH")
    print("   services/env_loader.py is vault-first; .env only fills gaps via")
    print("   os.environ.setdefault under an allowlist of key PREFIXES. A key")
    print("   whose prefix is missing is dropped SILENTLY (see the ATF_/LIVE_")
    print("   comments in that file). So read the RESOLVED value, never .env.")
    print()
    try:
        from services.env_loader import EnvLoader

        EnvLoader.load()
    except Exception as exc:  # noqa: BLE001
        print(f"   env load failed: {exc!r}")
        return
    keys = (
        "ENABLE_LIVE_TRADING",
        "LIVE_REQUIRE_MODEL_READY",
        "EXECUTE_LIVE_TRADES",
        "LIVE_TRADES_DRY_RUN",
        "STRATEGY_GRADUATION_ENFORCED",
        "LIVE_MIN_CLIP_USD",
        "LIVE_MIN_TRADE_USD",
        "LIVE_MIN_CAPITAL_USD",
    )
    for key in keys:
        print(f"     {key:<32} {os.getenv(key)!r}")
    print()
    print("   _live_trades_dry_run() (bot.py):")
    print("     EXECUTE_LIVE_TRADES unset  -> does not return False")
    print("     LIVE_TRADES_DRY_RUN unset  -> hinges on _auto_execute_approved")
    print("     _auto_execute_approved is set only by _refresh_auto_execute(),")
    print("     which _maybe_transition_to_live() returns BEFORE reaching when")
    print("     halt_live or bus_actions_pending is set. Both are set today.")
    print("     => default '1' wins: the executor is in DRY RUN.")
    print()
    print("   Ledger identity check (the id the entry path actually matches):")
    try:
        from trading.strategies.ledger import StrategyLedger

        ledger = StrategyLedger()
        for sid in ("money_button", "money_button_dip", "atf_static",
                    "atf_static_scout", "rsi_reversal"):
            print(f"     is_live_approved({sid:<18}) = "
                  f"{ledger.is_live_approved(sid)}")
        print()
        print("     NOTE: scheduler.py:48 MONEY_BUTTON_DIP_STRATEGY_ID =")
        print("     'money_button_dip' (landed 70fc09c, 2026-09-17 02:23). The")
        print("     ledger approved 'money_button'. On the NEXT restart the dip")
        print("     rule's directives carry an id the ledger has never approved,")
        print("     so _strategy_live_approved() returns False for them. The")
        print("     running process still predates that commit.")
    except Exception as exc:  # noqa: BLE001
        print(f"     ledger read failed: {exc!r}")


def main() -> int:
    path = _db_path()
    conn = _connect(path)
    cur = conn.cursor()
    now = time.time()
    print(f"store : {path} ({os.path.getsize(path):,} bytes)")
    print(f"now   : {_stamp(now)}")
    section_arming(cur, now)
    section_ops(cur, now)
    section_halt(cur, now)
    section_money_button(cur, now)
    conn.close()
    section_env()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
