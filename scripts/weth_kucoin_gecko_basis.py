"""Read-only measurement: is the WETH-USDC kucoin/geckoterminal tape mislabelling
also a MISPRICING?

For every geckoterminal WETH-USDC tick, find the nearest-in-time kucoin
WETH-USDC tick, discard pairs whose time gap is too large to be a fair
comparison, and report the basis (kucoin_px - gecko_px) / gecko_px in bp.

Read-only against storage/trading_cache.db. Writes nothing.
"""
import json
import sqlite3
import statistics

DB = "storage/trading_cache.db"
COST_BP = 35.75  # WETH measured round-trip cost, per task brief

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
cur = con.cursor()
cur.execute(
    "SELECT id, ts, price, raw FROM market_stream WHERE symbol='WETH-USDC' ORDER BY ts"
)
rows = cur.fetchall()

by_source = {"kucoin": [], "geckoterminal": [], "other": []}
other_names = set()
for rid, ts, price, raw in rows:
    try:
        rest = json.loads(raw).get("rest")
    except Exception:
        rest = None
    if rest == "kucoin":
        by_source["kucoin"].append((ts, price, rid))
    elif rest == "geckoterminal":
        by_source["geckoterminal"].append((ts, price, rid))
    else:
        by_source["other"].append((ts, price, rid))
        other_names.add(rest)

print(f"total rows: {len(rows)}")
print(f"kucoin: {len(by_source['kucoin'])}  geckoterminal: {len(by_source['geckoterminal'])}  other: {len(by_source['other'])} {other_names}")

kucoin = sorted(by_source["kucoin"])
gecko = sorted(by_source["geckoterminal"])
kucoin_ts = [t for t, _, _ in kucoin]

# nearest-kucoin-tick lookup via binary search
import bisect

def nearest_kucoin(ts):
    i = bisect.bisect_left(kucoin_ts, ts)
    candidates = []
    if i < len(kucoin_ts):
        candidates.append(i)
    if i > 0:
        candidates.append(i - 1)
    best = min(candidates, key=lambda j: abs(kucoin_ts[j] - ts))
    return kucoin[best]

pairs = []
for ts, price, rid in gecko:
    kts, kprice, krid = nearest_kucoin(ts)
    dt = kts - ts
    pairs.append(
        dict(
            gecko_ts=ts, gecko_px=price, gecko_id=rid,
            kucoin_ts=kts, kucoin_px=kprice, kucoin_id=krid,
            dt=dt, abs_dt=abs(dt),
        )
    )

# distribution of match gaps, to justify the cutoff empirically
dts_sorted = sorted(p["abs_dt"] for p in pairs)
n = len(dts_sorted)
def pct(p):
    if not dts_sorted:
        return float("nan")
    idx = min(n - 1, int(round(p * (n - 1))))
    return dts_sorted[idx]

print("\n--- nearest-neighbour time-gap distribution (all gecko rows, seconds) ---")
print(f"n={n} min={dts_sorted[0]:.1f}s median={pct(0.5):.1f}s p75={pct(0.75):.1f}s p90={pct(0.90):.1f}s max={dts_sorted[-1]:.1f}s")

CUTOFF_S = 120.0  # 2 minutes -- see justification printed alongside the gap distribution
kept = [p for p in pairs if p["abs_dt"] <= CUTOFF_S]
dropped = len(pairs) - len(kept)

print(f"\ncutoff chosen: {CUTOFF_S:.0f}s ({CUTOFF_S/60:.1f} min)")
print(f"pairs before filter: {len(pairs)}  dropped (gap > cutoff): {dropped}  surviving: {len(kept)}")

if kept:
    basis_bp = [ (p["kucoin_px"] - p["gecko_px"]) / p["gecko_px"] * 10000.0 for p in kept ]
    abs_basis_bp = [abs(b) for b in basis_bp]
    basis_sorted = sorted(basis_bp)
    abs_sorted = sorted(abs_basis_bp)

    def pctv(sorted_list, p):
        idx = min(len(sorted_list) - 1, int(round(p * (len(sorted_list) - 1))))
        return sorted_list[idx]

    mean_signed = statistics.mean(basis_bp)
    median_signed = statistics.median(basis_bp)
    mean_abs = statistics.mean(abs_basis_bp)
    median_abs = statistics.median(abs_basis_bp)
    stdev_signed = statistics.pstdev(basis_bp) if len(basis_bp) > 1 else float("nan")
    frac_over_cost = sum(1 for b in abs_basis_bp if b > COST_BP) / len(abs_basis_bp)

    print("\n--- basis (kucoin_px - gecko_px)/gecko_px, bp ---")
    print(f"n={len(kept)}")
    print(f"signed:  mean={mean_signed:.2f}bp  median={median_signed:.2f}bp  stdev={stdev_signed:.2f}bp  p10={pctv(basis_sorted,0.10):.2f}  p90={pctv(basis_sorted,0.90):.2f}")
    print(f"abs:     mean={mean_abs:.2f}bp  median={median_abs:.2f}bp  p90={pctv(abs_sorted,0.90):.2f}bp  max={abs_sorted[-1]:.2f}bp")
    print(f"fraction |basis| > {COST_BP}bp: {frac_over_cost:.4f}  ({sum(1 for b in abs_basis_bp if b > COST_BP)}/{len(abs_basis_bp)})")

    # rows driving the tail
    tail = sorted(kept, key=lambda p: abs((p['kucoin_px']-p['gecko_px'])/p['gecko_px']*10000.0), reverse=True)[:8]
    print("\n--- top 8 |basis| rows (hand-check) ---")
    for p in tail:
        b = (p["kucoin_px"] - p["gecko_px"]) / p["gecko_px"] * 10000.0
        print(f"  gecko_id={p['gecko_id']} ts={p['gecko_ts']:.1f} px={p['gecko_px']:.4f}  |  kucoin_id={p['kucoin_id']} ts={p['kucoin_ts']:.1f} px={p['kucoin_px']:.4f}  |  dt={p['dt']:+.1f}s  basis={b:+.2f}bp")

    print("\n--- first 5 surviving pairs in time order (hand-check) ---")
    for p in kept[:5]:
        b = (p["kucoin_px"] - p["gecko_px"]) / p["gecko_px"] * 10000.0
        print(f"  gecko_id={p['gecko_id']} ts={p['gecko_ts']:.1f} px={p['gecko_px']:.4f}  |  kucoin_id={p['kucoin_id']} ts={p['kucoin_ts']:.1f} px={p['kucoin_px']:.4f}  |  dt={p['dt']:+.1f}s  basis={b:+.2f}bp")
else:
    print("NO SURVIVING PAIRS at this cutoff.")

con.close()
