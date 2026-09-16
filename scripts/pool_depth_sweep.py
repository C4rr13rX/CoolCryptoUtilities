"""Measure pool depth for every symbol we actually stream.

b20c654 restricted a 408-predicate screen to symbols with >$1M of pool
liquidity and the universe collapsed to 2 of 155. But the depth snapshot it
read covered only 18 symbols, so the honest version of that finding was "2 of
the 18 we have measured", with 137 UNMEASURED -- not known thin.

That gap matters more than any predicate on the board. Every route here --
the cost floor, hold time, signal search, graduation -- is priced against a
universe whose fillability has never been measured, and ad2afed showed what
that costs: an edge that reads +10.73bp pooled is +0.88bp on the pools we can
actually fill, because the lift lives in the thin names.

So this fills in the snapshot. Same source and same rule Dune used for the
original 18 (see its `_source` field), so the new rows are comparable with the
old: DexScreener search, filtered to chainId=base and baseToken.symbol equal
to the symbol, deepest pool by liquidity.usd.

A symbol the API cannot resolve is recorded as null, NOT as zero and NOT as
absent. "We asked and got nothing" and "we never asked" are different facts,
and collapsing them is how the 18-row snapshot got read as a statement about
155 symbols in the first place.

This is a POINT-IN-TIME reading. A pool can drain or fill. The file carries
`_taken_utc_approx`; re-read it before citing it as current.

Usage:
    python -X utf8 scripts/pool_depth_sweep.py
    python -X utf8 scripts/pool_depth_sweep.py --hours 168 --out data/pool_depth_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from money_button_gate_census import load_ticks  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (compatible; CoolCryptoUtilities depth sweep)",
      "Accept": "application/json"}
URL = "https://api.dexscreener.com/latest/dex/search?q={}"


def deepest_base_pool(symbol: str, timeout: float) -> float | None:
    """Deepest base-chain pool in USD for `symbol`, or None if unresolvable."""
    q = urllib.parse.quote(f"{symbol} USDC")
    req = urllib.request.Request(URL.format(q), headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    best = None
    for p in payload.get("pairs") or []:
        if p.get("chainId") != "base":
            continue
        if (p.get("baseToken") or {}).get("symbol", "").upper() != symbol.upper():
            continue
        usd = (p.get("liquidity") or {}).get("usd")
        if usd is None:
            continue
        v = float(usd)
        if best is None or v > best:
            best = v
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--out", type=str, default="data/pool_depth_snapshot.json")
    ap.add_argument("--delay", type=float, default=0.35)
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    by_symbol = load_ticks(args.hours)
    ticks = {s.split("-")[0].upper(): len(r) for s, r in by_symbol.items()}
    symbols = sorted(ticks)
    print(f"  {len(symbols)} distinct base symbols in {args.hours:g}h of market_stream")

    out_path = Path(args.out)
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))

    depths: dict[str, float | None] = {}
    failed = []
    for i, sym in enumerate(symbols, 1):
        try:
            depths[sym] = deepest_base_pool(sym, args.timeout)
        except Exception as exc:  # noqa: BLE001 -- one bad symbol must not end the sweep
            depths[sym] = None
            failed.append((sym, type(exc).__name__))
        if i % 20 == 0:
            print(f"    {i}/{len(symbols)} ...")
        time.sleep(args.delay)

    doc = {
        "_comment": ("Pool liquidity in USD on base, deepest pool per symbol, from "
                     "DexScreener. Swept for every symbol in market_stream. null means "
                     "ASKED AND UNRESOLVED, which is not the same as thin and not the "
                     "same as unasked. POINT-IN-TIME: a pool can drain or fill."),
        "_source": ("https://api.dexscreener.com/latest/dex/search?q=<SYMBOL>%20USDC, "
                    "filtered to chainId=base and baseToken.symbol==SYMBOL, deepest "
                    "pool by liquidity.usd"),
        "_taken_utc_approx": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "_swept_symbols": len(symbols),
        "_window_hours": args.hours,
        "_prior_snapshot_rows": len([k for k in existing if not k.startswith("_")]),
    }
    for sym in symbols:
        doc[sym] = depths[sym]
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=False), encoding="utf-8")

    # ---- the distribution, by symbol AND by tick, which is the one that matters
    tiers = ((1_000_000, ">$1M"), (250_000, "$250k-1M"), (50_000, "$50k-250k"), (0, "<$50k"))
    print()
    print("  " + "=" * 74)
    print(f"  {'tier':<14}{'symbols':>9}{'sym share':>11}{'ticks':>9}{'TICK share':>12}")
    tot_t = sum(ticks.values())
    placed = set()
    for floor, label in tiers:
        got = [s for s in symbols if depths[s] is not None
               and depths[s] >= floor and s not in placed]
        placed |= set(got)
        t = sum(ticks[s] for s in got)
        print(f"  {label:<14}{len(got):>9}{len(got)/len(symbols):>10.1%}"
              f"{t:>9}{t/max(tot_t,1):>11.1%}")
    unres = [s for s in symbols if depths[s] is None]
    t = sum(ticks[s] for s in unres)
    print(f"  {'UNRESOLVED':<14}{len(unres):>9}{len(unres)/len(symbols):>10.1%}"
          f"{t:>9}{t/max(tot_t,1):>11.1%}")
    print("  " + "=" * 74)
    print(f"  wrote {out_path}")
    if failed:
        print(f"  {len(failed)} symbols errored: "
              f"{', '.join(f'{s}({e})' for s, e in failed[:8])}"
              f"{' ...' if len(failed) > 8 else ''}")
    print()
    print("  TICK share is the number to quote, not symbol share. A deep symbol we")
    print("  barely sample does not make our tape fillable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
