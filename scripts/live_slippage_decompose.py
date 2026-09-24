"""Decompose the live round-trip toll straight off Base receipts.

Item [be91fca9], pass 696, Sage.

The point: every slippage number on this board comes from a BOOK. This reads
the chain. For each of the 18 live round trips we fetch both receipts, decode
the ERC20 Transfer logs to get the exact token amounts that moved, and decode
the pool Swap event to get the pool's own post-swap price. The gap between the
price the wallet actually got and the pool's own price is fee + depth impact;
whatever is left over is routing/spread.

Only https://base.drpc.org answers from this box (p695): mainnet.base.org,
publicnode and llamarpc all 403.

    python scripts/live_slippage_decompose.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

from web3 import Web3

RPC = os.getenv("BASE_RPC_URL", "https://base.drpc.org")
FORENSICS = "data/live_book_forensics.json"

def _norm(h) -> str:
    """hexbytes .hex() dropped the 0x prefix in v1; normalise both ends."""
    s = h.hex() if hasattr(h, "hex") else str(h)
    return ("0x" + s if not s.startswith("0x") else s).lower()


TRANSFER_TOPIC = _norm(Web3.keccak(text="Transfer(address,address,uint256)"))
# Uniswap V3 / Aerodrome-slipstream concentrated pools
V3_SWAP_TOPIC = _norm(
    Web3.keccak(text="Swap(address,address,int256,int256,uint160,uint128,int24)")
)
# Aerodrome / Velodrome (Solidly) volatile+stable pools
V2_SWAP_TOPIC = _norm(
    Web3.keccak(text="Swap(address,address,uint256,uint256,uint256,uint256)")
)

USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913".lower()
USDC_DECIMALS = 6
# ETH/USD at the time of the live episode (2026-09-05).  Only used to put gas
# in dollars; every bp number above is independent of it.
ETH_USD = float(os.getenv("ETH_USD_AT_EPISODE", "4500"))


def _topic(raw) -> str:
    return _norm(raw)


def _addr_from_topic(raw) -> str:
    h = _topic(raw)
    return "0x" + h[-40:]


def decode_transfers(receipt):
    """[(token, frm, to, raw_amount)] for every ERC20 Transfer in the receipt."""
    out = []
    for lg in receipt["logs"]:
        topics = lg["topics"]
        if not topics or _topic(topics[0]) != TRANSFER_TOPIC:
            continue
        if len(topics) < 3:
            continue
        data = lg["data"]
        data = data.hex() if hasattr(data, "hex") else str(data)
        if not data.startswith("0x"):
            data = "0x" + data
        if len(data) < 3:
            continue
        out.append(
            (
                lg["address"].lower(),
                _addr_from_topic(topics[1]).lower(),
                _addr_from_topic(topics[2]).lower(),
                int(data, 16),
            )
        )
    return out


def decode_swaps(receipt):
    """[(pool, kind, fields)] for every Swap event we recognise."""
    out = []
    for lg in receipt["logs"]:
        topics = lg["topics"]
        if not topics:
            continue
        t0 = _topic(topics[0])
        data = lg["data"]
        data = data.hex() if hasattr(data, "hex") else str(data)
        if data.startswith("0x"):
            data = data[2:]
        words = [data[i : i + 64] for i in range(0, len(data), 64)]
        if t0 == V3_SWAP_TOPIC and len(words) >= 5:
            def s256(w):
                v = int(w, 16)
                return v - (1 << 256) if v >= (1 << 255) else v

            out.append(
                (
                    lg["address"].lower(),
                    "v3",
                    {
                        "amount0": s256(words[0]),
                        "amount1": s256(words[1]),
                        "sqrtPriceX96": int(words[2], 16),
                        "liquidity": int(words[3], 16),
                    },
                )
            )
        elif t0 == V2_SWAP_TOPIC and len(words) >= 4:
            out.append(
                (
                    lg["address"].lower(),
                    "v2",
                    {
                        "amount0In": int(words[0], 16),
                        "amount1In": int(words[1], 16),
                        "amount0Out": int(words[2], 16),
                        "amount1Out": int(words[3], 16),
                    },
                )
            )
    return out


def main() -> int:
    w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 25}))
    if not w3.is_connected():
        print(f"RPC {RPC} did not answer")
        return 2

    rows = json.load(open(FORENSICS))["rows"]

    # The wallet is whatever address appears on BOTH sides of the most legs.
    # Derive it rather than trusting a constant.
    seen = defaultdict(int)
    receipts: dict[str, dict] = {}
    fetched = 0
    ok = 0
    missing = []

    hashes = []
    for r in rows:
        for leg in ("entry", "exit"):
            h = r.get(f"{leg}_tx")
            if h:
                hashes.append((r["outcome_id"], r["symbol"], leg, h, r["clip_usd"]))

    for oid, sym, leg, h, clip in hashes:
        try:
            rc = w3.eth.get_transaction_receipt(h)
        except Exception as exc:  # noqa: BLE001
            missing.append((h, str(exc)[:60]))
            continue
        fetched += 1
        if rc["status"] == 1:
            ok += 1
        receipts[h] = rc
        for tok, frm, to, amt in decode_transfers(rc):
            seen[frm] += 1
            seen[to] += 1

    print("=" * 74)
    print("CHAIN FETCH")
    print(f"  legs in book        : {len(hashes)}")
    print(f"  receipts fetched    : {fetched}")
    print(f"  status=1            : {ok}")
    print(f"  not fetched         : {len(missing)}")
    for h, e in missing[:5]:
        print(f"    {h} {e}")

    # wallet = the EOA that sent each tx
    wallets = defaultdict(int)
    for h, rc in receipts.items():
        try:
            tx = w3.eth.get_transaction(h)
            wallets[tx["from"].lower()] += 1
        except Exception:  # noqa: BLE001
            pass
    wallet = max(wallets, key=wallets.get) if wallets else None
    print(f"  sender (n legs)     : {wallet} ({wallets.get(wallet)})")
    print()

    legs = []
    for oid, sym, leg, h, clip in hashes:
        rc = receipts.get(h)
        if rc is None:
            continue
        transfers = decode_transfers(rc)
        swaps = decode_swaps(rc)
        # what left the wallet, what arrived at the wallet
        out_tok = defaultdict(int)
        in_tok = defaultdict(int)
        for tok, frm, to, amt in transfers:
            if frm == wallet:
                out_tok[tok] += amt
            if to == wallet:
                in_tok[tok] += amt
        gas_wei = rc["gasUsed"] * rc.get("effectiveGasPrice", 0)
        legs.append(
            {
                "outcome_id": oid,
                "symbol": sym,
                "leg": leg,
                "tx": h,
                "clip_usd": clip,
                "status": rc["status"],
                "block": rc["blockNumber"],
                "gas_wei": gas_wei,
                "n_transfers": len(transfers),
                "n_swaps": len(swaps),
                "swap_kinds": sorted({k for _, k, _ in swaps}),
                "hops": len(swaps),
                "usdc_out": out_tok.get(USDC, 0) / 10**USDC_DECIMALS,
                "usdc_in": in_tok.get(USDC, 0) / 10**USDC_DECIMALS,
                "out_tokens": {k: v for k, v in out_tok.items()},
                "in_tokens": {k: v for k, v in in_tok.items()},
            }
        )

    print("=" * 74)
    print("ROUTING: how many pool hops does one leg actually take?")
    hopc = defaultdict(int)
    for l in legs:
        hopc[l["hops"]] += 1
    for k in sorted(hopc):
        print(f"  {k} hop(s): {hopc[k]} legs")
    kinds = defaultdict(int)
    for l in legs:
        kinds[",".join(l["swap_kinds"]) or "none"] += 1
    print("  pool kinds:", dict(kinds))
    print()

    # ---- the split -------------------------------------------------------
    # For a V3 Swap event, amount0/amount1 are signed from the POOL's side and
    # sqrtPriceX96 is the pool price AFTER the swap.  Both the executed price
    # and the pool price are in RAW token units, so their RATIO needs no
    # decimals at all -- that is why this number cannot be poisoned by the
    # 1e12 decimals bug that hit CBETH.
    for l in legs:
        rc = receipts[l["tx"]]
        devs = []
        for pool, kind, f in decode_swaps(rc):
            if kind != "v3":
                continue
            a0, a1 = f["amount0"], f["amount1"]
            if a0 == 0 or a1 == 0:
                continue
            pool_p = (f["sqrtPriceX96"] / (1 << 96)) ** 2  # token1 per token0
            if pool_p <= 0:
                continue
            if a0 > 0:  # token0 in, token1 out
                exec_p = (-a1) / a0
                dev = (1.0 - exec_p / pool_p) * 1e4
            else:       # token1 in, token0 out
                exec_p = a1 / (-a0)
                dev = (exec_p / pool_p - 1.0) * 1e4
            devs.append(dev)
        l["pool_dev_bp"] = sum(devs) if devs else None
        l["n_v3"] = len(devs)

    print("=" * 74)
    print("(i) QUOTE-TO-FILL / DEPTH: executed price vs the pool's OWN price,")
    print("    decoded from the same Swap event.  bp of notional, and dollars.")
    print(f"  {'symbol':<13}{'leg':<6}{'clip$':>8}{'dev_bp':>10}{'dev_$':>10}{'gas_$':>9}")
    priced = [l for l in legs if l["pool_dev_bp"] is not None]
    for l in sorted(priced, key=lambda x: x["clip_usd"]):
        dd = l["pool_dev_bp"] * l["clip_usd"] / 1e4
        gas_usd = l["gas_wei"] / 1e18 * ETH_USD
        print(f"  {l['symbol']:<13}{l['leg']:<6}{l['clip_usd']:>8.4f}"
              f"{l['pool_dev_bp']:>10.2f}{dd:>10.5f}{gas_usd:>9.5f}")
    print(f"  legs with a decodable pool price: {len(priced)} of {len(legs)}")
    if priced:
        import statistics as st
        bps = [l["pool_dev_bp"] for l in priced]
        dols = [l["pool_dev_bp"] * l["clip_usd"] / 1e4 for l in priced]
        gas = [l["gas_wei"] / 1e18 * ETH_USD for l in priced]
        print(f"  median dev   : {st.median(bps):.2f} bp   ${st.median(dols):.5f}")
        print(f"  median gas   : ${st.median(gas):.5f}  ({st.median(gas)/st.median([l['clip_usd'] for l in priced])*1e4:.1f} bp at the median clip)")
        small = [l for l in priced if l["clip_usd"] <= 0.80]
        big = [l for l in priced if l["clip_usd"] > 0.80]
        for name, grp in (("clip<=$0.80", small), ("clip> $0.80", big)):
            if not grp:
                continue
            b = st.median([g["pool_dev_bp"] for g in grp])
            d = st.median([g["pool_dev_bp"] * g["clip_usd"] / 1e4 for g in grp])
            gg = st.median([g["gas_wei"] / 1e18 * ETH_USD for g in grp])
            print(f"    {name} n={len(grp):<3} dev {b:>8.2f} bp  ${d:.5f} | gas ${gg:.5f}")
    print()

    json.dump(legs, open("data/live_slippage_legs.json", "w"), indent=1)
    print(f"wrote data/live_slippage_legs.json ({len(legs)} legs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
