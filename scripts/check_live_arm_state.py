"""Print the three values that decide whether this system executes REAL trades.

Pass 477 found the live-arm gate is NOT the env grep everyone has been citing.
bot.py:1557 returns "not dry run" when LIVE_TRADES_DRY_RUN is None AND
self._auto_execute_approved is True -- and that flag is REHYDRATED FROM THE
DATABASE at bot.py:12847, not from any .env or data/*.json file.

Run this before any restart and before any step-6 work.
"""
import json
import os
import sqlite3

DB = os.getenv("TRADING_DB_PATH", "storage/trading_cache.db")


def main() -> None:
    execute = os.getenv("EXECUTE_LIVE_TRADES")
    dry_run = os.getenv("LIVE_TRADES_DRY_RUN")

    flag = None
    try:
        with sqlite3.connect(DB) as db:
            row = db.execute("SELECT value FROM kv_store WHERE key='state'").fetchone()
        if row:
            ghost = json.loads(row[0]).get("ghost_trading") or {}
            flag = bool(ghost.get("auto_execute_approved", False))
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"  could not read {DB}: {exc}")

    print("LIVE ARM STATE")
    print(f"  EXECUTE_LIVE_TRADES                       = {execute!r}")
    print(f"  LIVE_TRADES_DRY_RUN                       = {dry_run!r}")
    print(f"  kv_store[state][ghost_trading]")
    print(f"      [auto_execute_approved]               = {flag!r}")
    print()

    # Mirrors bot.py:1550-1559 exactly.
    if (execute or "0").lower() in {"1", "true", "yes", "on"}:
        verdict, why = "LIVE", "EXECUTE_LIVE_TRADES is truthy (bot.py:1551)"
    elif dry_run is None and flag:
        verdict, why = "LIVE", "LIVE_TRADES_DRY_RUN unset AND flag True (bot.py:1557)"
    elif (dry_run or "1").lower() in {"1", "true", "yes", "on"}:
        verdict, why = "DRY RUN", "default or explicit 1 (bot.py:1559)"
    else:
        verdict, why = "LIVE", f"LIVE_TRADES_DRY_RUN={dry_run!r} is falsey (bot.py:1559)"

    print(f"  _live_trades_dry_run() RESOLVES TO: {verdict}")
    print(f"  because: {why}")
    if verdict == "LIVE":
        print()
        print("  To pin dry-run: set LIVE_TRADES_DRY_RUN=1 EXPLICITLY in .env.")
        print("  bot.py:1557 tests 'is None', so UNSET does NOT protect you.")


if __name__ == "__main__":
    main()
