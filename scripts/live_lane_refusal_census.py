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
one layer higher, through the scheduler's risk budget.

THE CHAIN DESCRIBED HERE UNTIL 2026-09-22 NO LONGER EXISTS. It read:

    bot.py        if self.live_trading_enabled:
                      if plan_flags.get("halt_live"): risk_budget = 0.0

and commit 821daa1 DELETED those two lines -- ``bot.py`` now carries a comment
at the same site saying so ("`halt_live` is NOT applied here any more"), because
arming the live lane was what switched the bot off. So the old reading, "ARMING
THE LIVE LANE IS WHAT SWITCHES THE DECISION PATH OFF", is dead and must not be
quoted. ``halt_live`` now travels through ``live_trading_enabled`` and DEMOTES
the bot to ghost.

THE LIVE CHAIN, measured 2026-09-22, is a GHOST-side zero and it has nothing to
do with live arming:

    pipeline.py:6244   ghost_risk_multiplier = 0.0      # when not ghost_collection_ready
    bot.py:6383        risk_budget *= ghost_multiplier
    bot.py:6389        if risk_budget <= 0:  -> scheduler.evaluate() is NEVER CALLED
    bot.py:6391-6400       self.metrics.feedback("scheduler", label="halted", ...)

THAT FEEDBACK CALL IS REACHABLE AND FIRING. Pass 483 recorded "no metrics row
since boot" and concluded it was dead code; 483 read the wrong table. It is
``feedback_events``, not ``metrics``:

    source='scheduler' label='halted'   n=1524 all-time
                                        98 in the last 24h, newest 1.69h old
    every one of those 98:  details {"reason": "ghost_validation_block",
                                     "ghost_multiplier": 0.0}

against 1286 ``scheduler``/``cycle_tick`` rows in the same 24 hours. So the
scheduler is being halted outright roughly a hundred times a day, and a halted
tick decides nothing at all -- no candidates, no ghost entries, no evidence.

AND IT IS A DEADLOCK, in two lines of one function. ``ghost_collection_ready``
comes from the model-bootstrap readiness flag, and pipeline.py:6175 is its only
override::

    if not ghost_collection_ready and ghost_ready:
        ghost_collection_ready = True

``ghost_ready`` is False -- that is what ``ghost_validation_block`` MEANS -- so
the override never fires. Ghost collection is off because collection is "not
ready", and the only thing that would mark it ready is the evidence the
switched-off lane cannot collect. This is the same shape 821daa1 removed at the
live layer, still alive at the ghost layer.

That branch writes no ``trading_ops`` row at all -- only a feedback event --
which is why a census of ``trading_ops`` alone shows "nothing was refused"
rather than "everything was refused".

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
    _section_resolved_arm()
    _section_ledger_identity()


def _persisted_auto_execute() -> bool | None:
    """The PERSISTED ``auto_execute_approved`` flag, or None if unreadable.

    Read from the same place ``bot.py`` rehydrates it from, by the same query
    ``scripts/check_live_arm_state.py`` uses, so the two diagnostics cannot
    drift apart again.
    """
    db_path = _db_path()
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            row = db.execute(
                "SELECT value FROM kv_store WHERE key='state'"
            ).fetchone()
        if not row:
            return None
        ghost = json.loads(row[0]).get("ghost_trading") or {}
        return bool(ghost.get("auto_execute_approved", False))
    except Exception as exc:  # noqa: BLE001
        print(f"     could not read the persisted flag: {exc!r}")
        return None


def _section_resolved_arm() -> None:
    """Resolve ``_live_trades_dry_run()`` from STATE, never from reachability.

    WHY THIS DOES NOT INFER FROM ``_refresh_auto_execute`` ANY MORE
    --------------------------------------------------------------
    This section used to reason only about ``_refresh_auto_execute``
    (``bot.py:1883``), correctly find that path unreachable while ``halt_live``
    is set, and conclude DRY RUN. That conclusion was FALSE, and false in the
    dangerous direction: it tells an operator no real money can move when the
    resolver says it can.

    The flag has TWO writers, and naming only the runtime one is the whole
    defect:

      bot.py:13106  ``_load_state`` boot restore --
                    ``self._auto_execute_approved = bool(ghost.get(
                    "auto_execute_approved", False))``. UNGUARDED: its
                    neighbouring restores (``sim_native_balances`` :13099,
                    ``active_exposure`` :13103) are all gated on
                    ``accounting_version >= ACCOUNTING_VERSION`` and this line
                    is not, so a stale persisted True is restored whatever the
                    accounting version says.
      bot.py:1893   ``_refresh_auto_execute`` runtime promotion.

    ``_refresh_auto_execute`` early-returns at ``bot.py:1883`` PRECISELY
    BECAUSE the flag is already True. The old census read that early return as
    a closed door; it is the opposite -- it is the door already being open.
    """
    flag = _persisted_auto_execute()
    dry_env = os.getenv("LIVE_TRADES_DRY_RUN")

    print("   _live_trades_dry_run() (bot.py:1557), RESOLVED FROM STATE:")
    print(f"     LIVE_TRADES_DRY_RUN env                  = {dry_env!r}")
    print(f"     kv_store[state][ghost_trading]")
    print(f"         [auto_execute_approved]              = {flag!r}")
    print()
    print("     BOTH writers of _auto_execute_approved (naming one is the bug):")
    print("       bot.py:13106  _load_state boot restore  <- UNGUARDED by")
    print("                     accounting_version, unlike its neighbours")
    print("                     (:13099 sim_native, :13103 active_exposure)")
    print("       bot.py:1893   _refresh_auto_execute runtime promotion")
    print()
    if dry_env is not None:
        explicit = dry_env.lower() in {"1", "true", "yes", "on"}
        verdict = "DRY RUN" if explicit else "LIVE"
        why = (f"LIVE_TRADES_DRY_RUN={dry_env!r} is SET EXPLICITLY and wins as "
               "the manual kill switch (bot.py:1559)")
    elif flag is None:
        verdict = "UNKNOWN"
        why = ("the persisted flag could not be read, so this script REFUSES "
               "to guess a verdict -- an unreadable arm state is not a safe "
               "one. Run scripts/check_live_arm_state.py")
    elif flag:
        verdict = "LIVE"
        why = ("LIVE_TRADES_DRY_RUN unset AND the persisted flag is True, so "
               "bot.py:1557 returns False = REAL MONEY. bot.py:1557 tests "
               "'is None', so UNSET DOES NOT PROTECT YOU")
    else:
        verdict = "DRY RUN"
        why = ("LIVE_TRADES_DRY_RUN unset and the persisted flag is False, so "
               "the '1' default at bot.py:1560 wins")
    print(f"     => _live_trades_dry_run() RESOLVES TO: {verdict}")
    print(f"        because: {why}.")
    if verdict == "LIVE":
        print("        To pin dry-run: set LIVE_TRADES_DRY_RUN=1 EXPLICITLY.")
    print()


def _section_ledger_identity() -> None:
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
