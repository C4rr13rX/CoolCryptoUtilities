"""Does the cross-sectional survivor fire where we can actually fill?

6e1f001 found the first predicate in this repo with confirmed out-of-sample
skill: `xs_rank<q0.05`, "this symbol is the worst performer in the market at
this instant", reverting +7 to +11bp out of sample at 1-5 minutes with a
monotone dose-response across cut extremity.

It was priced at a flat 11bp round trip, which is the WETH number. That charge
is a promise about POOL DEPTH, and the predicate has an obvious reason to break
it: "worst performer in the market" is exactly the description of a thin token
having a bad afternoon. A round trip on base is 0.04-0.11% on WETH, 0.64-4.35%
on AERO and up to 74.98% on KTA, so if the firing set skews thin then the edge
is charged at the wrong price and the route is dead on arrival.

This asks that question and nothing else. It reuses predicate_forge's own
feature builder so the firing set here is definitionally the same one that was
screened -- no reimplementation that could drift.

Usage:
    python -X utf8 scripts/xs_rank_depth_check.py
    python -X utf8 scripts/xs_rank_depth_check.py --cut 0.05 --horizon 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import predicate_forge as pf  # noqa: E402
from money_button_gate_census import load_ticks  # noqa: E402

#: Depth buckets in USD and the round-trip rate each implies, from the repo's
#: own measured range (see `round-trip-cost-inverts-with-clip-size`).
BUCKETS = ((1_000_000, "deep  >$1M", 0.0011),
           (250_000, "mid  $250k-1M", 0.0064),
           (50_000, "thin $50k-250k", 0.0200),
           (0, "dust  <$50k", 0.0750))


def bucket_for(depth: float):
    for floor, label, rate in BUCKETS:
        if depth >= floor:
            return label, rate
    return BUCKETS[-1][1], BUCKETS[-1][2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--cut", type=float, default=0.05)
    ap.add_argument("--horizon", type=float, default=5.0)
    ap.add_argument("--split", type=float, default=0.6)
    ap.add_argument("--depth-json", type=str, default="data/pool_depth_snapshot.json")
    args = ap.parse_args()

    pf.HORIZONS_MIN = (args.horizon,)
    depth_raw = json.loads(Path(args.depth_json).read_text(encoding="utf-8"))
    depth = {k.upper(): float(v) for k, v in depth_raw.items()
             if not k.startswith("_") and isinstance(v, (int, float))}

    by_symbol = load_ticks(args.hours)
    feats, ts, sid, fwd = pf.build_features(by_symbol)
    names = [s for s, r in sorted(by_symbol.items()) if len(r) >= 40]

    y = fwd[args.horizon]
    v = feats["xs_rank"]
    mask_is = ts <= np.quantile(ts, args.split)
    mask_os = ~mask_is
    q = np.quantile(v[mask_is & np.isfinite(v)], args.cut)
    fires = np.isfinite(v) & (v < q) & np.isfinite(y) & mask_os

    print("=" * 84)
    print(f"xs_rank<q{args.cut:.2f} @ {args.horizon:g}m -- WHERE DOES IT FIRE, AND CAN WE FILL THERE?")
    print("=" * 84)
    print(f"  held-out fires: {int(fires.sum())}   depth known for "
          f"{len(depth)} symbols in the snapshot")
    print()

    rows = {}
    unknown_n, unknown_g = 0, []
    for i in np.where(fires)[0]:
        sym = names[sid[i]].split("-")[0].upper()
        d = depth.get(sym)
        if d is None:
            unknown_n += 1
            unknown_g.append(y[i])
            continue
        label, rate = bucket_for(d)
        rows.setdefault(label, ([], rate))[0].append(y[i])

    print(f"  {'bucket':<18}{'fires':>7}{'share':>8}{'gross bp':>11}"
          f"{'round trip':>12}{'NET bp':>10}")
    total = sum(len(g) for g, _ in rows.values()) + unknown_n
    covered_net, covered_n = 0.0, 0
    for floor, label, _ in BUCKETS:
        if label not in rows:
            continue
        g, rate = rows[label]
        gross = float(np.mean(g)) * 10000.0
        net = gross - rate * 2 * 10000.0
        covered_net += net * len(g)
        covered_n += len(g)
        print(f"  {label:<18}{len(g):>7}{len(g)/max(total,1):>7.1%}{gross:>+11.2f}"
              f"{rate*2*10000:>11.0f}bp{net:>+10.2f}")
    if unknown_n:
        gu = float(np.mean(unknown_g)) * 10000.0 if unknown_g else float("nan")
        print(f"  {'NOT IN SNAPSHOT':<18}{unknown_n:>7}{unknown_n/max(total,1):>7.1%}"
              f"{gu:>+11.2f}{'   unknown':>12}{'  n/a':>10}")
    print()
    if covered_n:
        print(f"  Depth-weighted net over the {covered_n} fires we can price: "
              f"{covered_net/covered_n:+.2f}bp/trade")
    print()
    print("  READ THIS CAREFULLY: the snapshot covers a minority of the universe, so")
    print("  'NOT IN SNAPSHOT' is the biggest bucket and its cost is genuinely UNKNOWN,")
    print("  not assumed cheap. A symbol absent from a DexScreener depth search is more")
    print("  likely thin than deep, so the depth-weighted number above is an OPTIMISTIC")
    print("  bound on the priceable subset, not an estimate of the whole strategy.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
