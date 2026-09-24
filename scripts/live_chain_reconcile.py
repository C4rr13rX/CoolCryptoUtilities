"""Reconcile the live scoreboard against raw USDC transfers on Base.

Pass 696, Sage.

The loop is scored on one number: cumulative realised live net P/L. Every check
of that number so far has compared one book against another book. This compares
it against the chain by a method that never opens `trade_outcomes`: sum the raw
USDC ERC20 Transfers in and out of the wallet across the distinct live legs. No
entry_price, no quantity, no decimals table -- so none of the decimals defects
that have bitten this repo can reach it.

It also checks whether the recorded leg set actually CLOSES, per symbol: tokens
bought on recorded entries vs tokens sold on recorded exits. A ratio away from
1.000 means the book's legs are disposing of, or leaving behind, inventory that
is not in the book, and no P/L figure for that symbol can be trusted.

Run scripts/live_slippage_decompose.py first -- this reads its output.

    python scripts/live_chain_reconcile.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

LEGS = "data/live_slippage_legs.json"
FORENSICS = "data/live_book_forensics.json"
USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"

# ETH/USD implied by fee_cost / chain gas over the 18 rows (median; min 1638,
# max 6015).  Cross-checked against the WETH tape over the same window, which
# ran 2492.73 - 2694.03.  Only the gas-in-dollars line depends on it.
ETH_USD = 2499.0

# A symbol whose sold/bought token ratio is outside this band has legs missing
# from the book on one side or the other.
BALANCE_TOL = 0.01


def main() -> int:
    legs = json.load(open(LEGS))
    rows = json.load(open(FORENSICS))["rows"]

    # 3 entry txs are each shared by 2 outcome rows -- 18 exits over 15 entries.
    # Dedupe by hash or the shared entries get charged twice.
    distinct = {l["tx"]: l for l in legs}

    bought: dict = defaultdict(float)
    sold: dict = defaultdict(float)
    for l in distinct.values():
        for tok, amt in l["in_tokens"].items():
            if tok != USDC and l["leg"] == "entry":
                bought[l["symbol"]] += amt
        for tok, amt in l["out_tokens"].items():
            if tok != USDC and l["leg"] == "exit":
                sold[l["symbol"]] += amt

    print("=" * 74)
    print("DOES THE RECORDED LEG SET CLOSE?  tokens sold on exits / bought on entries")
    unbalanced = set()
    for sym in sorted(set(bought) | set(sold)):
        b, s = bought.get(sym, 0.0), sold.get(sym, 0.0)
        ratio = (s / b) if b else float("inf")
        flag = ""
        if abs(ratio - 1.0) > BALANCE_TOL:
            unbalanced.add(sym)
            flag = "  <-- DOES NOT CLOSE, no P/L for this symbol is usable"
        print(f"  {sym:<14}{ratio:>8.3f}{flag}")
    print()

    def tally(exclude: set):
        ent = [
            l
            for l in distinct.values()
            if l["leg"] == "entry" and l["symbol"] not in exclude
        ]
        ext = [
            l
            for l in distinct.values()
            if l["leg"] == "exit" and l["symbol"] not in exclude
        ]
        spent = sum(l["usdc_out"] - l["usdc_in"] for l in ent)
        got = sum(l["usdc_in"] - l["usdc_out"] for l in ext)
        gas = (
            sum(l["gas_wei"] for l in distinct.values() if l["symbol"] not in exclude)
            / 1e18
            * ETH_USD
        )
        book = sum(r["net"] for r in rows if r["symbol"] not in exclude)
        fee = sum(r["fee_cost"] for r in rows if r["symbol"] not in exclude)
        return len(ent), len(ext), spent, got, got - spent - gas, gas, book, fee

    print("=" * 74)
    print("CHAIN vs BOOK")
    for name, excl in (("all symbols", set()), ("balanced symbols only", unbalanced)):
        ne, nx, spent, got, net, gas, book, fee = tally(excl)
        print(f"  {name}")
        print(f"    legs            : {ne} entries, {nx} exits")
        print(f"    USDC out / in   : {spent:.6f} / {got:.6f}")
        print(f"    CHAIN net       : {net:+.6f}   (gas {gas:.6f})")
        print(f"    BOOK  net       : {book:+.6f}   (fee {fee:.6f})")
        print(f"    disagreement    : {abs(net - book):.6f}")
        print()

    print("Read the all-symbols line as an ARTIFACT, not a correction: an exit that")
    print("sells inventory no recorded entry bought shows up as profit that was never")
    print("made.  Only the balanced line is a statement about the scoreboard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
