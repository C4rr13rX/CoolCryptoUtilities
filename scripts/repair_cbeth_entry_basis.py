"""Repair the CBETH-USDC live position's cost basis, from its own receipt.

WHY THIS CANNOT WAIT FOR THE CODE FIX

Commit 85bf810 stops a fill whose implied price disagrees with the feed from
ever being booked again. It does nothing for the position already on the books,
and it cannot: the guard fires on the price a fill implies, and CBETH-USDC will
exit at a perfectly sane price. It is the BASIS that is wrong, so the exit
computes

    cost_portion = total_quote_spent * allocation_ratio   =  7.5e-13
    gross_profit = quote_received - cost_portion          ~= +0.75

and books a ~10^12 return into live P/L and into atf_static's ledger -- the
record that decides whether real money keeps being spent. That fires on its own,
without anybody running anything, the next time the exit conditions are met.

THE NUMBERS, AND WHERE THEY COME FROM

Not from our database. The entry transaction, read from base on 2026-09-03 via
mainnet.base.org, filtering the receipt's Transfer logs to our own wallet:

    tx      0x076978740803789cd40564cb150753bc075a5f6600d8422add58a4720822b82b
    status  0x1
    OUT     0x833589fcd6edb6e08f4c7c32d4f71b54bda02913  750000           raw USDC
    IN      0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22  273750474589586  raw cbETH

and the decimals from the contracts themselves (eth_call 0x313ce567), not from
any local table:

    USDC   6
    cbETH  18

    spent    750000 / 10^6            = 0.75 USDC
    received 273750474589586 / 10^18  = 0.000273750474589586 cbETH
    price    0.75 / 0.000273750474589586 = 2739.7212776504584

The book already holds the received quantity exactly right -- and the wallet
really does hold 0.0002737505 cbETH, checked with balanceOf -- so only the USDC
leg is being repaired. The feed carried cbETH at $2731.12 that minute; 2739.72
is within 0.32% of it, which is what a real fill looks like.

WHY IT MUST RUN WITH PRODUCTION STOPPED

_save_state merges the shared position book from the persisted copy but rewrites
every symbol the bot OWNS from its own memory. The bot holding CBETH-USDC has
the corrupt values in memory, so a repair applied underneath a running process
is overwritten at the next save. Bots seed from the persisted book in __init__,
so the sequence is: stop, repair, start.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time

DB = "storage/trading_cache.db"
SYMBOL = "CBETH-USDC"
TX = "0x076978740803789cd40564cb150753bc075a5f6600d8422add58a4720822b82b"

#: Straight from the receipt above. Raw units and decimals, so the arithmetic
#: this file is repairing is done here in the open rather than trusted.
USDC_RAW, USDC_DECIMALS = 750_000, 6
CBETH_RAW, CBETH_DECIMALS = 273_750_474_589_586, 18

TRUE_QUOTE_SPENT = USDC_RAW / 10**USDC_DECIMALS
TRUE_SIZE = CBETH_RAW / 10**CBETH_DECIMALS
TRUE_ENTRY_PRICE = TRUE_QUOTE_SPENT / TRUE_SIZE

#: The corrupt values, named so this script refuses to run against anything else.
CORRUPT_QUOTE_SPENT = 7.5e-13
CORRUPT_ENTRY_PRICE = 2.739721277650459e-09


def main() -> int:
    conn = sqlite3.connect(DB)
    try:
        row = conn.execute("select value from kv_store where key='state'").fetchone()
        if row is None:
            print("[repair] no state blob; nothing to do")
            return 1
        state = json.loads(row[0])
        positions = state.get("ghost_trading", {}).get("positions", {})
        pos = positions.get(SYMBOL)
        if not isinstance(pos, dict):
            print(f"[repair] {SYMBOL} is not in the book; nothing to do")
            return 1

        print(f"[repair] before: {json.dumps({k: pos.get(k) for k in ('mode','entry_price','size','quote_spent','entry_tx_hash','fill_source')}, indent=1)}")

        # Refuse unless this is the exact position measured. A repair that runs
        # against a position somebody has since changed is a corruption of its
        # own, and this one writes a cost basis.
        if str(pos.get("entry_tx_hash") or "").lower() != TX:
            print(f"[repair] REFUSED: entry_tx_hash is {pos.get('entry_tx_hash')!r}, not the measured trade")
            return 2
        if str(pos.get("mode") or "") != "live":
            print(f"[repair] REFUSED: mode is {pos.get('mode')!r}, not live")
            return 2
        if abs(float(pos.get("size") or 0.0) - TRUE_SIZE) > 1e-15:
            print(f"[repair] REFUSED: size {pos.get('size')!r} is not the quantity in the receipt")
            return 2
        if abs(float(pos.get("quote_spent") or 0.0) - CORRUPT_QUOTE_SPENT) > 1e-25:
            print(f"[repair] REFUSED: quote_spent {pos.get('quote_spent')!r} is not the corrupt value; already repaired?")
            return 2

        pos["quote_spent"] = TRUE_QUOTE_SPENT
        pos["entry_price"] = TRUE_ENTRY_PRICE
        # The exit trails its stop from this. Left at 2.7e-09 the position reads
        # as though it were up by a factor of a trillion against its own high.
        trigger = pos.get("trigger_state")
        if isinstance(trigger, dict) and abs(float(trigger.get("high_watermark") or 0.0) - CORRUPT_ENTRY_PRICE) < 1e-20:
            trigger["high_watermark"] = TRUE_ENTRY_PRICE
        pos["basis_estimated"] = False
        pos["basis_repaired"] = {
            "ts": time.time(),
            "reason": "usdc_read_with_18_decimals",
            "from_quote_spent": CORRUPT_QUOTE_SPENT,
            "from_entry_price": CORRUPT_ENTRY_PRICE,
            "source": "entry receipt Transfer logs, base mainnet",
            "tx": TX,
        }

        print(f"[repair] after:  {json.dumps({k: pos.get(k) for k in ('mode','entry_price','size','quote_spent')}, indent=1)}")

        conn.execute(
            "update kv_store set value=? where key='state'",
            (json.dumps(state),),
        )
        conn.execute(
            "insert into trading_ops (ts, wallet, chain, symbol, action, status, details) values (?,?,?,?,?,?,?)",
            (
                time.time(),
                "live",
                "base",
                SYMBOL,
                "repair",
                "basis-repaired",
                json.dumps(pos["basis_repaired"] | {
                    "to_quote_spent": TRUE_QUOTE_SPENT,
                    "to_entry_price": TRUE_ENTRY_PRICE,
                }),
            ),
        )
        conn.commit()
        print("[repair] committed")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
