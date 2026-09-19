#!/usr/bin/env python
"""QUOTED round-trip cost on deep-pool Base pairs at a real clip.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is a **QUOTED** probe, not a **REALISED** one. It reads live pool state
from the Uniswap V3 quoter on Base and asks it, on chain, what a round trip
would return. Nothing is signed, nothing is sent, no gate is touched and no
capital moves. A quoted number is NOT a fill: it has no MEV, no block delay,
no revert-and-retry, and it is the *optimistic* side of the real cost. Every
figure this script prints is labelled QUOTED for that reason.

WHY IT EXISTS
-------------
``services.round_trip_cost.TOKEN_ROUND_TRIP`` charges WETH 0.0011 (11bp) and
the comment on that row says, in the repo's own words, ``n=0 non-ghost --
ASSUMED``. Three separate results in this tree rest on that row and nobody
has ever paid it. There are 11 WETH rows in ``trade_fills`` and all 11 are
ghost, with ``expected_price == executed_price`` exactly -- so the store
cannot answer this and no amount of tape analysis can either. The pool can.

THE MEASUREMENT
---------------
Round trip, one direction, in a single quote pair:

    USDC --(clip)--> TOKEN --(all of it)--> USDC

and the cost is ``(clip - usdc_back) / clip``. That single number captures
BOTH fee tiers and BOTH legs' price impact, which is exactly what
``TOKEN_ROUND_TRIP`` is defined to hold, so the two are directly comparable.

A second quote at a deliberately tiny clip gives the same round trip with
price impact driven to ~0. The difference between the two is the part that
is genuinely CLIP-DEPENDENT -- the slippage the item is asking about --
while the tiny-clip floor is the part you pay at any size (the fee tiers and
the tick spacing). Reporting them separately is the point: a round-trip cost
quoted at one clip and applied at another is how the 27.8bp composite got
its spreadless median.

DISTINGUISHING A FAILURE FROM AN EMPTY POOL
-------------------------------------------
This repo has already shipped a quoter that could not tell a 429 from an
empty pool. This one never returns a number for a leg that raised: a failed
quote is recorded as ``status="quote_failed"`` with the exception text, and
it is excluded from every statistic rather than folded in as a zero or as
infinite cost. ``ok`` counts and ``failed`` counts are both printed.

USAGE
    python scripts/deep_pool_slippage_probe.py
    python scripts/deep_pool_slippage_probe.py --clip 10 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Import the real modules BEFORE reading any config. services/ pulls the .env
# in on import; without it os.getenv falls through to the literals in this
# file and the probe prices a rule production does not run.
from services.round_trip_cost import (  # noqa: E402
    TOKEN_ROUND_TRIP,
    default_token_round_trip,
    token_round_trip_cost,
)
from services.providers.uniswap_v3 import UniswapV3Local  # noqa: E402
from main import _readonly_w3  # noqa: E402

CHAIN = "base"

#: Calldata needs a `to`. It is never signed or sent and never affects the
#: quoted amount. WETH-on-Base is used as an inert, valid, checksummed
#: constant when no wallet is configured.
_RECIPIENT = (
    os.getenv("WALLET_ADDRESS")
    or os.getenv("PUBLIC_ADDRESS")
    or "0x4200000000000000000000000000000000000006"
)
OUT_PATH = _REPO / "data" / "deep_pool_slippage.json"
TOKENS_PATH = _REPO / "data" / "token_addresses.json"

#: The two levels the item names as the decision thresholds, in bp of round
#: trip. Below 12.2bp a 65% direction call pays at 60m; below 8.1bp a 60%
#: call pays. These are the numbers the measurement is quoted against.
THRESHOLD_65 = 12.2
THRESHOLD_60 = 8.1

#: WETH-USDC first -- it is the named primary target, not one of three
#: options. The rest follow in the order the item sets out.
PAIRS: List[Dict[str, str]] = [
    {"symbol": "WETH-USDC", "base": "WETH", "quote": "USDC"},
    {"symbol": "CBBTC-USDC", "base": "CBBTC", "quote": "USDC"},
    {"symbol": "CBETH-USDC", "base": "CBETH", "quote": "USDC"},
    # AERO is not a deep pool; it is here as a CONTROL. The table charges it
    # 68.45bp from 16 real fills, so if the probe is sane it should come back
    # far above the deep names. A probe that cannot separate AERO from WETH
    # is not measuring anything.
    {"symbol": "AERO-USDC", "base": "AERO", "quote": "USDC"},
]


def _load_tokens() -> Dict[str, Dict[str, Any]]:
    raw = json.loads(TOKENS_PATH.read_text(encoding="utf-8"))
    chain = raw.get(CHAIN) or {}
    return {k.upper(): v for k, v in chain.items() if isinstance(v, dict)}


def _decimals(entry: Dict[str, Any], symbol: str) -> int:
    for key in ("decimals", "decimal", "dp"):
        if key in entry:
            return int(entry[key])
    # USDC on Base is 6; the rest of the deep names here are 18. Only used
    # when the address book omits it, and it is recorded in the output so a
    # wrong guess is visible rather than silent.
    return 6 if symbol.upper() in {"USDC", "USDT"} else 18


def _quote_leg(
    uni: UniswapV3Local,
    token_in: str,
    token_out: str,
    amount_in: int,
) -> Dict[str, Any]:
    """One leg. Returns a dict that is EITHER ok with an amount OR failed."""
    try:
        # A recipient is required to BUILD calldata. Nothing is signed and the
        # tx is discarded -- only ``buyAmount`` is read, and the recipient does
        # not enter the quote maths at all. Prefer the real wallet so the
        # probe is shaped exactly like production; fall back to a constant.
        res = uni.quote_and_build(
            CHAIN, token_in, token_out, int(amount_in),
            recipient=_RECIPIENT,
        )
    except Exception as exc:  # noqa: BLE001 -- the text is the evidence
        return {"status": "quote_failed", "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(res, dict):
        return {"status": "quote_failed", "error": f"non-dict {type(res).__name__}"}
    if res.get("__error__"):
        return {"status": "quote_failed", "error": str(res["__error__"])}
    try:
        out = int(res.get("buyAmount") or 0)
    except (TypeError, ValueError):
        return {"status": "quote_failed", "error": f"unparseable buyAmount {res.get('buyAmount')!r}"}
    if out <= 0:
        # A zero here is NOT an empty pool and must never be read as one --
        # the quoter cannot tell those apart. It is an unsuccessful quote.
        return {"status": "quote_failed", "error": "quoter returned buyAmount<=0 (indeterminate: no liquidity OR rate-limited)"}
    return {"status": "ok", "amount_out": out, "fee_tier": res.get("fee")}


def _round_trip(
    uni: UniswapV3Local,
    quote_addr: str,
    base_addr: str,
    clip_units: int,
) -> Dict[str, Any]:
    """USDC -> TOKEN -> USDC. Cost is what does not come back."""
    leg_in = _quote_leg(uni, quote_addr, base_addr, clip_units)
    if leg_in["status"] != "ok":
        return {"status": "quote_failed", "leg": "buy", "error": leg_in["error"]}
    leg_out = _quote_leg(uni, base_addr, quote_addr, leg_in["amount_out"])
    if leg_out["status"] != "ok":
        return {"status": "quote_failed", "leg": "sell", "error": leg_out["error"]}
    returned = int(leg_out["amount_out"])
    cost_bp = (clip_units - returned) / float(clip_units) * 1e4
    return {
        "status": "ok",
        "round_trip_bp": cost_bp,
        "token_units_held": leg_in["amount_out"],
        "usdc_back": returned,
        "fee_tier_buy": leg_in.get("fee_tier"),
        "fee_tier_sell": leg_out.get("fee_tier"),
    }


def _spread(values: List[float]) -> Dict[str, Any]:
    """Median AND spread. Never a lone pooled median -- that is the defect
    the 27.8bp composite has."""
    vals = sorted(values)
    n = len(vals)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "median_bp": statistics.median(vals),
        "min_bp": vals[0],
        "max_bp": vals[-1],
        "spread_bp": vals[-1] - vals[0],
        "stdev_bp": statistics.stdev(vals) if n > 1 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", type=float, default=10.0,
                    help="clip in USDC. The item specifies 10; gas is 4.0bp "
                         "there and 40.5bp at 1 dollar, so 1 swamps the "
                         "question being asked.")
    ap.add_argument("--tiny", type=float, default=0.05,
                    help="control clip, to separate the clip-INDEPENDENT "
                         "floor (fees, tick spacing) from price impact.")
    ap.add_argument("--repeats", type=int, default=5,
                    help="quotes per pair per clip. n and spread are "
                         "reported; a single quote is not a measurement.")
    ap.add_argument("--sleep", type=float, default=0.6,
                    help="seconds between quotes, to keep a 429 from "
                         "masquerading as a result.")
    args = ap.parse_args()

    tokens = _load_tokens()
    uni = UniswapV3Local(lambda ch: _readonly_w3(ch))

    report: Dict[str, Any] = {
        "measurement": "QUOTED",
        "not": "REALISED -- nothing was signed or sent; no tx hash exists for these numbers",
        "chain": CHAIN,
        "clip_usd": args.clip,
        "tiny_clip_usd": args.tiny,
        "repeats": args.repeats,
        "generated_ts": time.time(),
        "thresholds_bp": {
            "pays_at_65pct_direction_60m": THRESHOLD_65,
            "pays_at_60pct_direction_60m": THRESHOLD_60,
        },
        "env": {
            "TOKEN_ROUND_TRIP_DEFAULT": default_token_round_trip(),
            "MIN_DRAWDOWN": os.getenv("MIN_DRAWDOWN"),
            "MAX_HOLD_FORCE_SECONDS": os.getenv("MAX_HOLD_FORCE_SECONDS"),
        },
        "pairs": {},
    }

    print(f"QUOTED deep-pool round-trip probe -- chain={CHAIN} clip=${args.clip} "
          f"tiny=${args.tiny} repeats={args.repeats}")
    print("NOTHING IS SIGNED. These are quotes, not fills. No tx hash exists for them.\n")

    for pair in PAIRS:
        sym, base_sym, quote_sym = pair["symbol"], pair["base"], pair["quote"]
        entry_b, entry_q = tokens.get(base_sym), tokens.get(quote_sym)
        if not entry_b or not entry_q:
            report["pairs"][sym] = {"status": "no_address", "base": base_sym, "quote": quote_sym}
            print(f"{sym:14s} NO ADDRESS in data/token_addresses.json -- skipped")
            continue

        base_addr, quote_addr = entry_b["address"], entry_q["address"]
        dq = _decimals(entry_q, quote_sym)
        clip_units = int(round(args.clip * (10 ** dq)))
        tiny_units = int(round(args.tiny * (10 ** dq)))

        book_rate = token_round_trip_cost(sym)
        listed = base_sym.upper() in TOKEN_ROUND_TRIP

        full_ok: List[float] = []
        tiny_ok: List[float] = []
        failures: List[Dict[str, Any]] = []
        fee_tiers: List[Any] = []

        for _ in range(args.repeats):
            r = _round_trip(uni, quote_addr, base_addr, clip_units)
            if r["status"] == "ok":
                full_ok.append(r["round_trip_bp"])
                fee_tiers.append((r.get("fee_tier_buy"), r.get("fee_tier_sell")))
            else:
                failures.append({"clip": "full", **r})
            time.sleep(args.sleep)

            t = _round_trip(uni, quote_addr, base_addr, tiny_units)
            if t["status"] == "ok":
                tiny_ok.append(t["round_trip_bp"])
            else:
                failures.append({"clip": "tiny", **t})
            time.sleep(args.sleep)

        full_stats, tiny_stats = _spread(full_ok), _spread(tiny_ok)
        impact = None
        if full_stats.get("n") and tiny_stats.get("n"):
            impact = full_stats["median_bp"] - tiny_stats["median_bp"]

        row: Dict[str, Any] = {
            "status": "ok" if full_ok else "all_quotes_failed",
            "quoted_round_trip_at_clip": full_stats,
            "quoted_round_trip_at_tiny_clip": tiny_stats,
            "clip_dependent_impact_bp": impact,
            "fee_tiers_seen": sorted({str(f) for f in fee_tiers}) if fee_tiers else [],
            "book_rate_bp": book_rate * 1e4,
            "book_row_is_measured": listed,
            "ok_quotes": len(full_ok),
            "failed_quotes": len(failures),
            "failures": failures[:6],
        }
        if full_stats.get("n"):
            med = full_stats["median_bp"]
            row["verdict_vs_thresholds"] = {
                "under_12_2bp_65pct_call_pays": med < THRESHOLD_65,
                "under_8_1bp_60pct_call_pays": med < THRESHOLD_60,
                "vs_book_ratio": med / (book_rate * 1e4) if book_rate else None,
            }
        report["pairs"][sym] = row

        if full_stats.get("n"):
            print(f"{sym:14s} QUOTED round trip @ ${args.clip:>5.2f}  "
                  f"median {full_stats['median_bp']:8.2f}bp  "
                  f"[{full_stats['min_bp']:.2f}..{full_stats['max_bp']:.2f}]  "
                  f"n={full_stats['n']}  fail={len(failures)}")
            print(f"{'':14s} floor @ ${args.tiny:<5.2f} "
                  f"median {tiny_stats.get('median_bp', float('nan')):8.2f}bp   "
                  f"clip-dependent impact {impact if impact is None else round(impact, 2)}bp")
            print(f"{'':14s} book charges {book_rate*1e4:.2f}bp "
                  f"({'MEASURED' if listed else 'ASSUMED/default'})  "
                  f"-> quoted/book = {full_stats['median_bp']/(book_rate*1e4):.2f}x")
        else:
            print(f"{sym:14s} ALL {len(failures)} QUOTES FAILED -- "
                  f"INDETERMINATE, not zero-cost, not an empty pool")
            for f in failures[:2]:
                print(f"{'':14s}   {f.get('leg', '?')}: {f.get('error', '')[:110]}")
        print()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(_REPO)}")

    weth = report["pairs"].get("WETH-USDC", {})
    st = weth.get("quoted_round_trip_at_clip", {})
    print("\n--- THE DECISION NUMBER ---")
    if st.get("n"):
        med = st["median_bp"]
        print(f"WETH-USDC QUOTED round trip at ${args.clip}: {med:.2f}bp "
              f"(n={st['n']}, spread {st['spread_bp']:.2f}bp)")
        print(f"  under 12.2bp (a 65% direction call pays at 60m)? "
              f"{'YES' if med < THRESHOLD_65 else 'NO'}")
        print(f"  under  8.1bp (a 60% direction call pays at 60m)? "
              f"{'YES' if med < THRESHOLD_60 else 'NO'}")
        print(f"  the book's ASSUMED WETH row is "
              f"{TOKEN_ROUND_TRIP.get('WETH', 0)*1e4:.2f}bp")
        print("  REMINDER: QUOTED is the optimistic side. A fill pays this "
              "PLUS gas, MEV and block delay.")
    else:
        print("WETH-USDC produced no successful quote. INDETERMINATE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
