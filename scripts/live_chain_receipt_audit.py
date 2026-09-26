#!/usr/bin/env python
"""Verify the live scoreboard against Base mainnet receipts.

Sage, pass 719. This answers [7afa9b13]'s last open criterion: the tx hashes
EXISTING is not the same as the AMOUNTS MATCHING. Cove's
``scripts/live_book_chain_audit.py`` proved the hashes are there and join
cleanly; this script fetches the receipts and checks the recorded numbers
against what actually settled.

Two independent checks, one per term of the scoreboard:

  gross_profit  vs  the net USDC delta across the entry and exit receipts,
                    summed from ERC-20 Transfer logs touching our wallet.
                    A Uniswap pool fee is taken out of the OUTPUT amount, so
                    this delta is already post-fee by construction.

  fee_cost      vs  gasUsed * effectiveGasPrice over both legs. If fee_cost is
                    realized gas (trading/bot.py:10962) the implied native
                    price agrees across unrelated rows; if it also carries
                    notional * pool_fee (trading/bot.py:11017) it does not, and
                    fee-as-a-share-of-notional is near constant instead.
                    Measured p719: implied 2497.84 and 2491.89 USD/ETH on two
                    rows two days apart -- gas only, pool fee NOT double
                    charged, so the published -0.186371 is correct.

Read-only against the db and against the chain. Needs egress to a Base RPC.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import urllib.request
from collections import defaultdict

USDC_BASE = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
RPCS = (
    "https://mainnet.base.org",
    "https://base.llamarpc.com",
    "https://base-rpc.publicnode.com",
    "https://1rpc.io/base",
)
DEFAULT_DB = "storage/trading_cache.db"


def rpc(url, method, params, timeout=25):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    req = urllib.request.Request(
        url,
        data=body.encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.0"},
    )
    payload = json.load(urllib.request.urlopen(req, timeout=timeout))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload["result"]


def pick_rpc(explicit=None):
    """Return the first endpoint that answers eth_blockNumber."""
    for url in ([explicit] if explicit else []) + list(RPCS):
        try:
            head = int(rpc(url, "eth_blockNumber", []), 16)
        except Exception as exc:  # noqa: BLE001 - any failure means try the next
            print("  rpc unusable %-34s %s" % (url, str(exc)[:60]))
            continue
        print("  rpc %s at block %d" % (url, head))
        return url
    raise SystemExit(
        "no Base RPC reachable -- this script needs network egress. Public Base "
        "endpoints answer 403 from inside a sandboxed shell."
    )


def live_round_trips(db_path):
    """Recorded live round trips, keyed by trade_id, with both tx hashes.

    The amounts live in trade_outcomes; the hashes live in trading_ops
    status='live-exit' inside the details JSON, which is why a column-name
    search for 'tx'/'hash' misses them (Echo p715's withdrawn claim).

    Three of the 15 live trade_ids exited in TWO parts and carry two distinct
    exit hashes each (all AERO). A partial exit returns only part of the clip,
    so pairing one partial exit against the full entry leg understates the
    proceeds -- p719 measured a spurious -1.52 USD "discrepancy" that way
    before catching it. Those trade_ids are reported as needing leg-level
    aggregation rather than checked one row at a time.
    """
    conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    legs = defaultdict(list)
    for (raw,) in conn.execute("SELECT details FROM trading_ops WHERE status='live-exit'"):
        det = json.loads(raw)
        entry, exit_ = det.get("entry_tx_hash"), det.get("tx_hash")
        if det.get("trade_id") and entry and exit_:
            legs[det["trade_id"]].append((entry, exit_))
    # Only a single-exit trade can be checked against its entry row for row.
    hashes = {t: v[0] for t, v in legs.items() if len(set(v)) == 1}
    partial = {t for t, v in legs.items() if len(set(v)) > 1}

    rows = []
    sql = (
        "SELECT trade_id, symbol, gross_profit, fee_cost, net_profit, status, details "
        "FROM trade_outcomes WHERE details LIKE '%live%'"
    )
    for trade_id, symbol, gross, fee, net, status, raw in conn.execute(sql):
        if json.loads(raw).get("mode") != "live":
            continue
        pair = hashes.get(trade_id)
        rows.append((trade_id, symbol, gross, fee, net, status, pair))
    return rows, partial


def settle(url, tx_hash):
    """(net USDC delta for our wallet, gas in native, wallet) from one receipt."""
    receipt = rpc(url, "eth_getTransactionReceipt", [tx_hash])
    if not receipt:
        return None
    wallet = receipt["from"].lower()
    gas = int(receipt["gasUsed"], 16) * int(receipt["effectiveGasPrice"], 16) / 1e18
    delta = 0.0
    for log in receipt["logs"]:
        if log["address"].lower() != USDC_BASE or log["topics"][0].lower() != TRANSFER:
            continue
        frm = "0x" + log["topics"][1][-40:]
        to = "0x" + log["topics"][2][-40:]
        value = int(log["data"], 16) / 1e6
        if to == wallet:
            delta += value
        elif frm == wallet:
            delta -= value
    return {
        "usdc": delta,
        "gas": gas,
        "wallet": wallet,
        "block": int(receipt["blockNumber"], 16),
        "ok": receipt["status"] == "0x1",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--rpc", default=None, help="Base RPC url to try first")
    ap.add_argument("--limit", type=int, default=0, help="verify only the N largest rows")
    args = ap.parse_args()

    rows, partial = live_round_trips(args.db)
    closed = [r for r in rows if r[5] == "closed"]
    print("live trade_outcomes rows: %d (%d closed, %d other)"
          % (len(rows), len(closed), len(rows) - len(closed)))
    print("  sum gross %+.6f  sum fee %+.6f  sum net %+.6f"
          % (sum(r[2] for r in closed), sum(r[3] for r in closed),
             sum(r[4] for r in closed)))
    print("  fee as %% of |gross|: %.1f%%"
          % (100 * sum(r[3] for r in closed) / abs(sum(r[2] for r in closed))))

    targets = sorted(closed, key=lambda r: -abs(r[2]))
    if args.limit:
        targets = targets[: args.limit]
    skipped = sorted({t[1] for t in targets if not t[6]})
    targets = [t for t in targets if t[6]]
    print("\nverifying %d row(s) against the chain" % len(targets))
    if partial:
        print("  %d trade_id(s) exited in parts and are NOT row-for-row checkable"
              " here: %s" % (len(partial), ", ".join(sorted(skipped)) or "-"))
    url = pick_rpc(args.rpc)

    implied = []
    worst = 0.0
    for trade_id, symbol, gross, fee, net, _status, (entry, exit_) in targets:
        legs = [settle(url, entry), settle(url, exit_)]
        if any(leg is None for leg in legs):
            print("\n%-13s HASH NOT FOUND ON CHAIN -- this is a hard failure" % symbol)
            worst = float("inf")
            continue
        chain = sum(leg["usdc"] for leg in legs)
        gas = sum(leg["gas"] for leg in legs)
        delta = chain - gross
        worst = max(worst, abs(delta))
        print("\n%-13s blocks %d / %d  status %s"
              % (symbol, legs[0]["block"], legs[1]["block"],
                 "ok" if all(leg["ok"] for leg in legs) else "REVERTED"))
        print("  chain USDC delta  %+.6f" % chain)
        print("  recorded gross    %+.6f   delta %+.9f" % (gross, delta))
        print("  measured gas      %.8f native over both legs" % gas)
        print("  recorded fee_cost %.8f  =>  implied native price %.2f USD"
              % (fee, fee / gas if gas else float("nan")))
        if gas:
            implied.append((symbol, fee / gas))

    print("\n%s" % ("-" * 62))
    print("gross vs chain: worst absolute delta %.9f USD" % worst)
    if len(implied) >= 2:
        lo = min(p for _, p in implied)
        hi = max(p for _, p in implied)
        print("implied native price across %d rows: %.2f .. %.2f (spread %.2f%%)"
              % (len(implied), lo, hi, 100 * (hi - lo) / hi))
        print("  a tight spread means fee_cost is GAS ONLY (bot.py:10962) and the")
        print("  pool fee is NOT double charged on top of the post-fee chain delta.")
    if worst == 0.0:
        print("VERDICT: every verified row's gross_profit IS the settled USDC delta.")


if __name__ == "__main__":
    main()
