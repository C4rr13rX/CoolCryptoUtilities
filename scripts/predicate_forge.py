"""Brute-force the predicate space, and know the bar that selection alone clears.

Every strategy screen this repo has run tested predicates a human wrote down,
one at a time: donchian_breakout, rsi_reversal, the money_button dip rule. All
of them are FIRST-ORDER price algebra -- "is the return over the last N bars
above a threshold". data/brain_experiments/ now records that the whole family
is exhausted: continuation carries under 1bp of lift at every horizon tested,
and 2% dips invert out of sample.

This script does something different in two ways.

FIRST, IT GENERATES THE CANDIDATES INSTEAD OF RECEIVING THEM. Primitives are
combined programmatically into thousands of predicates and every one is priced
against the same tape. That is the "brute force the password" search: we are
not guessing which rule works, we are enumerating the space and letting the
tape answer. Three families the hand-written strategies never covered:

  DERIVATIVES ABOVE THE FIRST.  A return is a first derivative. Everything
  screened here so far stops there. This builds the second (acceleration --
  is the move speeding up or rolling over) and the third (jerk -- is the
  acceleration itself turning). A move decelerating into a level and a move
  accelerating through it have identical returns and opposite futures, and no
  first-order predicate can tell them apart.

  VOLATILITY AS A SEPARATE AXIS. Ratio of short-window realised vol to long,
  which says whether the move is an expansion or a contraction of the regime
  it sits in -- independent of the direction that a return measures.

  WHAT THE REST OF THE MARKET IS DOING AT THAT INSTANT. This is the game
  theory term. If another system is positioned for a move, its behaviour shows
  up as CO-MOVEMENT: many symbols moving together is a crowd; one symbol
  moving alone is idiosyncratic. The same +1% means different things in those
  two states, because in the first one the counterparty is a crowd that has to
  unwind and in the second one it is not. So every tick carries the
  cross-sectional mean move, the dispersion around it, this symbol's rank
  within it, and its residual after the crowd is subtracted.

SECOND, AND THIS IS THE POINT: THE NULL IS INSIDE THE SEARCH. Screening K
predicates and reporting the best one is not a measurement, it is an
order statistic. This repo already paid for that lesson -- the GA edge died on
exactly this (corr(n, edge) = -0.84; the two biggest samples scored BELOW
chance-with-selection). So the search computes its own bar: the forward
returns are permuted within symbol, the ENTIRE screen is re-run on the
shuffled labels, and the largest lift it finds is recorded. Repeat and you
have the distribution of "best lift reachable by selection alone, with this
many candidates, on this tape". A predicate is reported as a survivor only if
it beats the 95th percentile of that distribution.

That bar is not a formality. With a few thousand candidates it is typically
tens of basis points, which is larger than any edge this repo has ever
claimed -- meaning most of what a naive search would have "found" is noise
that a permutation could have found too.

Surviving in-sample is still not an edge. The tape is split by TIME, the
search runs on the first part only, and survivors are then priced once on the
held-out part, where no selection has taken place and the number is honest.

Nothing here is a strategy and nothing here trades. It is a screen: it says
which corners of the space are worth a strategy, and -- just as usefully --
reports honestly when none of them are.

Usage:
    python -X utf8 scripts/predicate_forge.py
    python -X utf8 scripts/predicate_forge.py --hours 336 --perms 400
    python -X utf8 scripts/predicate_forge.py --fee 0.0065   # charge the live fee
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from money_button_gate_census import load_ticks  # noqa: E402

#: Forward horizons in MINUTES. Deliberately short: every screen run in this
#: repo so far held for 12 minutes or longer and nothing cleared the bar, and
#: backlog item [46145c10] asks for the 2-5 minute range explicitly.
HORIZONS_MIN = (2.0, 3.0, 5.0, 8.0, 12.0)

#: Lookback scales in MINUTES for the derivative stack.
SCALES_MIN = (3.0, 6.0, 12.0, 30.0, 60.0)

#: A tick's forward partner must land in [H, H*PAIR_SLACK] minutes. The feed
#: samples irregularly (~1-3 min/tick), so pairing by index would silently
#: compare a 2-minute horizon against a 40-minute gap on a thin symbol.
PAIR_SLACK = 2.5

#: Returns beyond this are the known denomination/regime contamination, not
#: price. Same censor the donchian screen used, so the numbers are comparable.
CENSOR = 0.20

#: Cross-sectional aggregates are computed on a grid this many seconds wide.
XS_BIN_SEC = 300.0

#: Quantile cut points. Each feature becomes predicates at each of these, in
#: both directions ("above" and "below").
CUTS = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)

#: A predicate firing on fewer ticks than this in either split is not screened.
MIN_SUPPORT = 60


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=float)
    ok = np.abs(b) > 1e-12
    out[ok] = a[ok] / b[ok]
    return out


def _ret_over(logp: np.ndarray, ts: np.ndarray, scale_sec: float) -> np.ndarray:
    """Log return over the last `scale_sec`, by timestamp not by index.

    Returns NaN where no tick old enough exists, so warmup never masquerades
    as a zero move.
    """
    idx = np.searchsorted(ts, ts - scale_sec, side="right") - 1
    out = np.full(ts.shape, np.nan)
    ok = idx >= 0
    out[ok] = logp[ok] - logp[idx[ok]]
    return out


def _vol_over(logp: np.ndarray, ts: np.ndarray, scale_sec: float) -> np.ndarray:
    """Realised vol of tick-to-tick log changes inside a trailing window."""
    d = np.diff(logp, prepend=logp[0])
    csum = np.cumsum(d * d)
    cnt = np.arange(len(d), dtype=float) + 1.0
    idx = np.searchsorted(ts, ts - scale_sec, side="right") - 1
    out = np.full(ts.shape, np.nan)
    ok = idx >= 0
    prev_sum = np.where(ok, csum[np.clip(idx, 0, None)], 0.0)
    prev_cnt = np.where(ok, cnt[np.clip(idx, 0, None)], 0.0)
    n = cnt - prev_cnt
    good = ok & (n >= 3)
    out[good] = np.sqrt(np.maximum((csum[good] - prev_sum[good]) / n[good], 0.0))
    return out


def build_features(
    by_symbol: Dict[str, List[Tuple[float, float, float]]]
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    """Return (features, ts, symbol_id, forward returns per horizon), flat.

    Every array is one row per usable tick, concatenated across symbols.
    """
    feats: Dict[str, List[np.ndarray]] = defaultdict(list)
    all_ts: List[np.ndarray] = []
    all_sid: List[np.ndarray] = []
    fwd: Dict[float, List[np.ndarray]] = {h: [] for h in HORIZONS_MIN}

    usable = [(s, r) for s, r in by_symbol.items() if len(r) >= 40]
    for sid, (symbol, rows) in enumerate(sorted(usable)):
        arr = np.asarray(rows, dtype=float)
        ts, price = arr[:, 0], arr[:, 1]
        order = np.argsort(ts)
        ts, price = ts[order], price[order]
        logp = np.log(price)

        # ---- first, second and third derivatives at every scale -----------
        rets: Dict[float, np.ndarray] = {}
        for L in SCALES_MIN:
            rets[L] = _ret_over(logp, ts, L * 60.0)
            vol = _vol_over(logp, ts, L * 60.0)
            feats[f"r{L:g}"].append(rets[L])
            feats[f"z{L:g}"].append(_safe_div(rets[L], np.nan_to_num(vol, nan=0.0)))
            feats[f"vol{L:g}"].append(vol)

        for L in SCALES_MIN:
            # second derivative: the move over the last L, minus the move over
            # the L before it. Positive = accelerating, negative = rolling over.
            prev = _ret_over(logp, ts, 2.0 * L * 60.0)
            accel = rets[L] - (prev - rets[L])
            feats[f"accel{L:g}"].append(accel)
            # third derivative: is the acceleration itself turning.
            prev3 = _ret_over(logp, ts, 3.0 * L * 60.0)
            accel_prev = (prev - rets[L]) - (prev3 - prev)
            feats[f"jerk{L:g}"].append(accel - accel_prev)

        # ---- volatility structure, independent of direction ---------------
        for L in (3.0, 12.0):
            short = _vol_over(logp, ts, L * 60.0)
            long_ = _vol_over(logp, ts, 4.0 * L * 60.0)
            feats[f"volratio{L:g}"].append(_safe_div(short, np.nan_to_num(long_, nan=0.0)))

        # ---- where price sits in its own recent range ---------------------
        for L in (12.0, 60.0):
            idx = np.searchsorted(ts, ts - L * 60.0, side="right") - 1
            lo = np.full(ts.shape, np.nan)
            hi = np.full(ts.shape, np.nan)
            cmin = np.minimum.accumulate(logp)
            cmax = np.maximum.accumulate(logp)
            # running min/max over a trailing window, approximated by the
            # expanding extremes since the window start. Cheap and monotone.
            ok = idx >= 0
            lo[ok] = cmin[idx[ok]]
            hi[ok] = cmax[idx[ok]]
            rng = hi - lo
            feats[f"pos{L:g}"].append(_safe_div(logp - lo, np.nan_to_num(rng, nan=0.0)))

        # ---- forward returns ----------------------------------------------
        for H in HORIZONS_MIN:
            lo_i = np.searchsorted(ts, ts + H * 60.0, side="left")
            f = np.full(ts.shape, np.nan)
            valid = lo_i < len(ts)
            vi = np.where(valid, np.clip(lo_i, 0, len(ts) - 1), 0)
            gap = ts[vi] - ts
            good = valid & (gap <= H * 60.0 * PAIR_SLACK)
            f[good] = (price[vi][good] - price[good]) / price[good]
            f[np.abs(f) > CENSOR] = np.nan
            fwd[H].append(f)

        all_ts.append(ts)
        all_sid.append(np.full(ts.shape, sid, dtype=np.int32))

    out_feats = {k: np.concatenate(v) for k, v in feats.items()}
    ts_all = np.concatenate(all_ts)
    sid_all = np.concatenate(all_sid)
    fwd_all = {h: np.concatenate(v) for h, v in fwd.items()}

    # ---- cross-sectional / game-theory terms ------------------------------
    # What is every OTHER symbol doing at this instant? A crowd and a lone
    # mover produce the same return and face different counterparties.
    base = out_feats["r12"]
    bins = np.floor(ts_all / XS_BIN_SEC).astype(np.int64)
    uniq, inv = np.unique(bins, return_inverse=True)
    valid = np.isfinite(base)
    cnt = np.bincount(inv[valid], minlength=len(uniq)).astype(float)
    s1 = np.bincount(inv[valid], weights=base[valid], minlength=len(uniq))
    s2 = np.bincount(inv[valid], weights=base[valid] ** 2, minlength=len(uniq))
    mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
    var = np.where(cnt > 1, s2 / np.maximum(cnt, 1) - mean**2, np.nan)
    disp = np.sqrt(np.maximum(var, 0.0))

    crowd = mean[inv]
    crowd_disp = disp[inv]
    thin = cnt[inv] < 5
    crowd[thin] = np.nan
    crowd_disp[thin] = np.nan

    out_feats["crowd"] = crowd            # is the whole market moving?
    out_feats["crowd_disp"] = crowd_disp  # how much do they disagree?
    out_feats["idio"] = base - crowd      # this symbol's move net of the crowd
    out_feats["idio_z"] = _safe_div(base - crowd, np.nan_to_num(crowd_disp, nan=0.0))
    # rank of this symbol's move inside its own instant, in [0,1].
    rank = np.full(base.shape, np.nan)
    order = np.lexsort((base, inv))
    grp = inv[order]
    within = np.arange(len(grp)) - np.repeat(
        np.concatenate(([0], np.cumsum(np.bincount(grp, minlength=len(uniq)))[:-1])),
        np.bincount(grp, minlength=len(uniq)),
    )
    gsize = np.bincount(grp, minlength=len(uniq))[grp].astype(float)
    rank[order] = np.where(gsize > 1, within / np.maximum(gsize - 1.0, 1.0), np.nan)
    rank[~valid] = np.nan
    out_feats["xs_rank"] = rank

    return out_feats, ts_all, sid_all, fwd_all


def make_predicates(
    feats: Dict[str, np.ndarray], mask_is: np.ndarray
) -> List[Tuple[str, np.ndarray]]:
    """Threshold every feature at every cut, both directions.

    Cut points are taken from the IN-SAMPLE half only. A threshold chosen on
    the full tape has already seen the held-out data.
    """
    preds: List[Tuple[str, np.ndarray]] = []
    for name in sorted(feats):
        v = feats[name]
        ref = v[mask_is & np.isfinite(v)]
        if len(ref) < MIN_SUPPORT * 4:
            continue
        qs = np.quantile(ref, CUTS)
        for cut, q in zip(CUTS, qs):
            if not np.isfinite(q):
                continue
            preds.append((f"{name}>q{cut:.2f}", np.isfinite(v) & (v > q)))
            preds.append((f"{name}<q{cut:.2f}", np.isfinite(v) & (v < q)))
    return preds


def screen(
    preds: List[Tuple[str, np.ndarray]],
    y: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lift in bp of each predicate over the unconditional mean, plus support."""
    base_mask = mask & np.isfinite(y)
    if base_mask.sum() < MIN_SUPPORT:
        return np.zeros(len(preds)), np.zeros(len(preds))
    base = y[base_mask].mean()
    lifts = np.zeros(len(preds))
    support = np.zeros(len(preds))
    for i, (_, p) in enumerate(preds):
        m = p & base_mask
        n = int(m.sum())
        support[i] = n
        if n >= MIN_SUPPORT:
            lifts[i] = (y[m].mean() - base) * 10000.0
    return lifts, support


def permutation_bar(
    preds: List[Tuple[str, np.ndarray]],
    y: np.ndarray,
    mask: np.ndarray,
    sid: np.ndarray,
    perms: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Distribution of the BEST |lift| reachable by selection alone.

    Forward returns are shuffled WITHIN symbol, so each symbol keeps its own
    return distribution and only the tick-to-predicate alignment is destroyed.
    The full screen is then re-run and the maximum recorded. This is the bar
    the winner has to clear to be more than an order statistic.
    """
    out = np.zeros(perms)
    idx_by_sid: Dict[int, np.ndarray] = {}
    live = mask & np.isfinite(y)
    for s in np.unique(sid[live]):
        idx_by_sid[int(s)] = np.where(live & (sid == s))[0]
    for p in range(perms):
        yp = y.copy()
        for _, idx in idx_by_sid.items():
            yp[idx] = yp[rng.permutation(idx)]
        lifts, sup = screen(preds, yp, mask)
        out[p] = np.abs(lifts[sup >= MIN_SUPPORT]).max() if (sup >= MIN_SUPPORT).any() else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--split", type=float, default=0.6, help="in-sample share by TIME")
    ap.add_argument("--fee", type=float, default=0.003, help="one-way fee rate")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    rng = np.random.default_rng(20260916)

    print("=" * 88)
    print("PREDICATE FORGE -- enumerate the space, and price the selection")
    print("=" * 88)

    by_symbol = load_ticks(args.hours)
    if not by_symbol:
        print("  no tape. nothing to screen.")
        return 1

    feats, ts, sid, fwd = build_features(by_symbol)
    if len(ts) == 0:
        print("  no usable ticks after warmup.")
        return 1

    cut_ts = np.quantile(ts, args.split)
    mask_is = ts <= cut_ts
    mask_os = ts > cut_ts

    preds = make_predicates(feats, mask_is)
    print(f"  tape      : {len(by_symbol)} symbols, {len(ts)} ticks over {args.hours:g}h")
    print(f"  features  : {len(feats)}   predicates: {len(preds)}")
    print(f"  split     : in-sample {int(mask_is.sum())} ticks | held-out {int(mask_os.sum())}")
    print(f"  horizons  : {', '.join(f'{h:g}m' for h in HORIZONS_MIN)}")
    print(f"  round trip: {2 * args.fee * 10000:.0f}bp charged to every survivor")
    print()

    results = []
    for H in HORIZONS_MIN:
        y = fwd[H]
        lifts_is, sup_is = screen(preds, y, mask_is)
        ok = sup_is >= MIN_SUPPORT
        if not ok.any():
            print(f"  [{H:g}m] no predicate reaches support {MIN_SUPPORT}. skipped.")
            continue

        bar_dist = permutation_bar(preds, y, mask_is, sid, args.perms, rng)
        bar95 = float(np.quantile(bar_dist, 0.95))
        bar50 = float(np.quantile(bar_dist, 0.50))

        best = int(np.argmax(np.where(ok, np.abs(lifts_is), -np.inf)))
        best_lift = float(lifts_is[best])

        print(f"  [{H:g}m horizon]")
        print(f"     selection bar (best |lift| from a PERMUTED tape, {args.perms} draws):")
        print(f"        median {bar50:7.2f}bp    95th pct {bar95:7.2f}bp")
        print(f"     best real predicate: {preds[best][0]:<22} {best_lift:+8.2f}bp  n={int(sup_is[best])}")

        survivors = np.where(ok & (np.abs(lifts_is) > bar95))[0]
        if len(survivors) == 0:
            print(f"     SURVIVORS: NONE. The best real predicate does not beat what")
            print(f"     shuffling the labels produces. Nothing here is an edge.\n")
            results.append({"horizon_min": H, "bar95_bp": bar95, "survivors": 0,
                            "best": preds[best][0], "best_lift_bp": best_lift})
            continue

        order = survivors[np.argsort(-np.abs(lifts_is[survivors]))][: args.top]
        print(f"     SURVIVORS IN-SAMPLE: {len(survivors)} of {int(ok.sum())} screened."
              f" Held-out test follows.")
        print(f"     {'predicate':<24}{'IS lift':>10}{'IS n':>8}{'OOS lift':>11}{'OOS n':>8}"
              f"{'net/trade':>11}")
        rows = []
        for i in order:
            name, p = preds[i]
            l_os, s_os = screen([(name, p)], y, mask_os)
            oos = float(l_os[0])
            n_os = int(s_os[0])
            m = p & mask_os & np.isfinite(y)
            gross = float(y[m].mean()) if n_os >= MIN_SUPPORT else float("nan")
            net = (gross - 2 * args.fee) * 10000.0 if n_os >= MIN_SUPPORT else float("nan")
            flag = "" if n_os >= MIN_SUPPORT else "  (thin)"
            print(f"     {name:<24}{lifts_is[i]:>+9.2f}{int(sup_is[i]):>8}"
                  f"{oos:>+10.2f}{n_os:>8}{net:>+10.2f}{flag}")
            rows.append({"predicate": name, "is_lift_bp": float(lifts_is[i]),
                         "is_n": int(sup_is[i]), "oos_lift_bp": oos, "oos_n": n_os,
                         "oos_net_bp": net})
        held = [r for r in rows if r["oos_n"] >= MIN_SUPPORT
                and np.sign(r["oos_lift_bp"]) == np.sign(r["is_lift_bp"])
                and abs(r["oos_lift_bp"]) > bar95]
        print(f"     HELD OUT OF SAMPLE (same sign, still over the bar): {len(held)}")
        if held:
            paying = [r for r in held if r["oos_net_bp"] > 0]
            print(f"     ... and NET POSITIVE after a {2*args.fee*10000:.0f}bp round trip: {len(paying)}")
            for r in paying:
                print(f"         *** {r['predicate']} -> {r['oos_net_bp']:+.2f}bp/trade"
                      f" on n={r['oos_n']}")
        print()
        results.append({"horizon_min": H, "bar95_bp": bar95, "bar50_bp": bar50,
                        "screened": int(ok.sum()), "survivors": len(survivors),
                        "held": len(held), "rows": rows})

    print("=" * 88)
    any_paying = any(
        r.get("rows") and any(x["oos_net_bp"] > 0 and x["oos_n"] >= MIN_SUPPORT
                              and np.sign(x["oos_lift_bp"]) == np.sign(x["is_lift_bp"])
                              for x in r["rows"])
        for r in results
    )
    if any_paying:
        print("  A predicate cleared selection, held out of sample, and pays the fee.")
        print("  It is a CANDIDATE, not an edge: re-run on a later tape before trading it.")
    else:
        print("  NOTHING in this space clears its own selection bar out of sample and")
        print("  pays the round trip. That is a real result: it prices how much of a")
        print("  'discovered edge' this tape can manufacture from noise alone, and the")
        print("  answer is more than anything genuinely present.")
    print("=" * 88)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
