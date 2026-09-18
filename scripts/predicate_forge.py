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

THIRD, IT IS GRADED BY A MEASURED COST, PER SYMBOL, NOT BY A CONSTANT. A
search graded by the wrong cost rejects real edges and admits fake ones, and
this repo has measured that a flat token-blind rate does exactly that: the
standing TOKEN_ROUND_TRIP table charges AERO 435bp where the chain says 10-13,
charges CBBTC 70bp where the chain says 11-15, and waves CBETH through at 30bp
where the chain says 67. So the round trip charged to a survivor is read from
data/round_trip_quote_census_*.json -- a committed, on-chain QuoterV2 census
that prices 9 symbols on a clip grid taken from the actual book -- and each
symbol pays its OWN number at the clip you ask for.

Three things about that cost are stated out loud on every run, because a
silent fallback is how a constant gets shipped wearing a measurement's name:
which symbols were priced, which fell back to --fee and how many ticks that
covers, and the fact that a QUOTE IS NOT A FILL. The census numbers are pool
cost (fee + impact) only; the live lane measured 13.5bp of round-trip
slippage that no quote carries and gas is charged separately. Every census
cost is therefore a FLOOR -- the best case -- so "net positive" under it is
the most optimistic reading available, not a realisable P/L.

Surviving in-sample is still not an edge. The tape is split by TIME, the
search runs on the first part only, and survivors are then priced once on the
held-out part, where no selection has taken place and the number is honest.

Nothing here is a strategy and nothing here trades. It is a screen: it says
which corners of the space are worth a strategy, and -- just as usefully --
reports honestly when none of them are.

Usage:
    python -X utf8 scripts/predicate_forge.py
    python -X utf8 scripts/predicate_forge.py --hours 336 --perms 400
    python -X utf8 scripts/predicate_forge.py --cost-mode flat --fee 0.0065
    python -X utf8 scripts/predicate_forge.py --fillable-only
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from money_button_gate_census import DB_PATH, load_ticks  # noqa: E402

#: The committed measured round-trip cost census. Pinned by name, not globbed,
#: so a re-run reproduces: this is the artifact pass 458 landed at 1a33c9a.
DEFAULT_COST_CENSUS = "data/round_trip_quote_census_20260917T190210Z.json"

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

#: Restrict the screen to one feature family (set by --only).
ONLY_PREFIX = ""

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
        if ONLY_PREFIX and not name.startswith(ONLY_PREFIX):
            continue
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


def _acf_block_length(v: np.ndarray, overlap_ticks: int, max_lag: int = 40) -> Tuple[int, float, float]:
    """Block length for one symbol, CHOSEN FROM THE DATA. No hard-coded guess.

    Two mechanisms make adjacent forward returns dependent, and the block has
    to be long enough to survive the larger of them:

    1. OVERLAP. A forward return over H minutes measured at tick i and the one
       measured at tick i+1 are computed from windows that SHARE price prints
       whenever the ticks are less than H apart. Two returns that share a print
       cannot be exchanged independently, so the block can never be shorter
       than the number of ticks H minutes spans -- `overlap_ticks`, measured
       from this symbol's own median inter-tick gap. This is the floor.

    2. RESIDUAL SERIAL CORRELATION beyond the overlap. Estimated here with the
       standard integrated-autocorrelation rule: sum the sample autocorrelation
       out to the first lag that falls inside the +-2/sqrt(n) white-noise band,
       and take tau = 1 + 2*sum. Blocks shorter than tau leave dependence in
       the resample, which is exactly the failure this replaces.

    The block is max(overlap_ticks, ceil(tau)). The OVERLAP FLOOR IS NEVER
    CROSSED: a symbol with too few ticks to give five such blocks is reported
    as `thin` rather than having its block quietly shortened, because a block
    shorter than the overlap is the same mistake as the i.i.d. shuffle, only
    smaller. The only cap is n itself -- a block cannot exceed the series.

    WHY THIS REPLACED AN I.I.D. SHUFFLE. Measured pass 463 over this same
    feed: mean lag-1 autocorrelation of the forward returns is +0.2856 at the
    2-minute horizon rising to +0.6339 at 12 minutes, against a white-noise
    band of +-0.0121 -- 21x to 52x outside it, with every lag out to 5 outside
    at every horizon. An i.i.d. permutation destroys that structure, so it
    draws a null with less clustering than the real tape and produces a bar
    that is too NARROW. Every lift ever scored against the old bar was
    flattered by it.

    Returns (block_length, tau, rho1).
    """
    n = len(v)
    if n < 8:
        return max(1, overlap_ticks), 1.0, 0.0
    x = v - v.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return max(1, overlap_ticks), 1.0, 0.0
    band = 2.0 / math.sqrt(n)
    acc = 0.0
    rho1 = 0.0
    for k in range(1, min(max_lag, n - 1) + 1):
        rho = float(np.dot(x[:-k], x[k:]) / denom)
        if k == 1:
            rho1 = rho
        if rho < band:
            break
        acc += rho
    tau = 1.0 + 2.0 * acc
    L = max(int(overlap_ticks), int(math.ceil(tau)), 1)
    return max(1, min(L, n)), tau, rho1


def block_lengths(
    y: np.ndarray,
    ts: np.ndarray,
    mask: np.ndarray,
    sid: np.ndarray,
    horizon_min: float,
) -> Tuple[Dict[int, int], dict]:
    """Per-symbol block length plus the numbers the choice was made from."""
    live = mask & np.isfinite(y)
    lens: Dict[int, int] = {}
    rows = []
    for s in np.unique(sid[live]):
        idx = np.where(live & (sid == int(s)))[0]
        if len(idx) < 2:
            lens[int(s)] = 1
            continue
        dt = np.diff(ts[idx])
        dt = dt[dt > 0]
        med_dt = float(np.median(dt)) if len(dt) else horizon_min * 60.0
        overlap = int(math.ceil(horizon_min * 60.0 / med_dt)) if med_dt > 0 else 1
        L, tau, rho1 = _acf_block_length(y[idx], overlap)
        lens[int(s)] = L
        rows.append({"sid": int(s), "n": int(len(idx)), "median_dt_sec": med_dt,
                     "overlap_ticks": overlap, "tau": tau, "rho1": rho1, "block": L,
                     "thin": bool(L * 5 > len(idx))})
    thin = [r for r in rows if r["thin"]]
    report = {
        "horizon_min": horizon_min,
        "per_symbol": rows,
        "median_block": int(np.median([r["block"] for r in rows])) if rows else 1,
        "median_rho1": float(np.median([r["rho1"] for r in rows])) if rows else 0.0,
        "median_overlap_ticks": int(np.median([r["overlap_ticks"] for r in rows])) if rows else 1,
        "thin_symbols": len(thin),
        # The floor is the horizon in ticks, OR the whole series when the symbol
        # has fewer ticks than one horizon spans -- in that case the series is a
        # single block and its returns are never shuffled at all, which is the
        # most conservative treatment available, not a shortened block.
        "floor_respected": all(r["block"] >= min(r["overlap_ticks"], r["n"]) for r in rows),
        "whole_series_symbols": sum(1 for r in rows if r["n"] <= r["overlap_ticks"]),
    }
    return lens, report


def permutation_bar(
    preds: List[Tuple[str, np.ndarray]],
    y: np.ndarray,
    mask: np.ndarray,
    sid: np.ndarray,
    perms: int,
    rng: np.random.Generator,
    method: str = "block",
    blocks: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    """Distribution of the BEST |lift| reachable by selection alone.

    Forward returns are resampled WITHIN symbol, so each symbol keeps its own
    return distribution and only the tick-to-predicate alignment is destroyed.
    The full screen is then re-run and the maximum recorded. This is the bar
    the winner has to clear to be more than an order statistic.

    method='block' (the default, and the honest one) uses a CIRCULAR BLOCK
    BOOTSTRAP: contiguous runs of length L are drawn with wraparound and laid
    down in time order, so serial dependence inside a block survives. L comes
    from block_lengths() and is measured, not guessed.

    method='iid' is the OLD null -- a plain within-symbol permutation. It is
    kept ONLY so the two bars can be printed side by side and the size of the
    correction is visible. It is anti-conservative on this feed and no result
    should be scored against it.
    """
    out = np.zeros(perms)
    idx_by_sid: Dict[int, np.ndarray] = {}
    live = mask & np.isfinite(y)
    for s in np.unique(sid[live]):
        idx_by_sid[int(s)] = np.where(live & (sid == s))[0]
    for p in range(perms):
        yp = y.copy()
        for s, idx in idx_by_sid.items():
            if method == "iid":
                yp[idx] = yp[rng.permutation(idx)]
                continue
            n = len(idx)
            L = max(1, min(int((blocks or {}).get(s, 1)), n))
            nblocks = int(math.ceil(n / L))
            starts = rng.integers(0, n, size=nblocks)
            take = (starts[:, None] + np.arange(L)[None, :]) % n
            yp[idx] = y[idx][take.reshape(-1)[:n]]
        lifts, sup = screen(preds, yp, mask)
        out[p] = np.abs(lifts[sup >= MIN_SUPPORT]).max() if (sup >= MIN_SUPPORT).any() else 0.0
    return out


# ---------------------------------------------------------------------------
# COST. Per symbol, measured, with the fallback stated out loud.
# ---------------------------------------------------------------------------


def _ticker(symbol: str) -> Tuple[str, str]:
    """Split a tape symbol 'AERO-USDC' into ('AERO', 'USDC'), upper-cased."""
    base, _, quote = str(symbol).upper().partition("-")
    return base, quote


def load_cost_census(path: str) -> dict:
    """Read the round-trip quote census and refuse anything that is not one.

    The schema is NOT assumed: the artifact is checked by its own
    `artifact` tag before a single rate is read out of it.
    """
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"cost census not found: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    tag = doc.get("artifact")
    if tag != "round_trip_quote_census":
        raise ValueError(
            f"{p} is artifact '{tag}', not 'round_trip_quote_census'. "
            "Refusing to read costs out of a file that is not a cost census."
        )
    if not isinstance(doc.get("quoted"), dict) or not doc["quoted"]:
        raise ValueError(f"{p} prices no symbol at all ('quoted' is empty).")
    doc["_path"] = str(p)
    return doc


def census_rate(entry: dict, clip_usd: float) -> Optional[Tuple[float, float]]:
    """(clip_actually_used, ROUND-TRIP rate as a fraction) for one symbol.

    `points` is a list of [clip_usd, cost_pct] where cost_pct is the WHOLE
    round trip (USDC -> token -> USDC) in percent of notional. We charge the
    nearest clip the curve actually priced and report which one that was --
    interpolating would invent a reading between two measurements.
    """
    pts = [(float(c), float(v)) for c, v in entry.get("points", [])
           if np.isfinite(float(c)) and np.isfinite(float(v))]
    if not pts:
        return None
    clip, pct = min(pts, key=lambda cv: (abs(cv[0] - clip_usd), -cv[0]))
    return clip, pct / 100.0


def live_filled_tickers() -> Tuple[set, int]:
    """Tickers with a REAL executed leg on chain, from trade_fills.

    The alternative definition of "fillable" -- not what a quoter says we
    could fill, but what we demonstrably did. Returns (tickers, n_legs).
    """
    try:
        conn = sqlite3.connect(DB_PATH, uri=True)
    except Exception:
        return set(), 0
    try:
        rows = conn.execute(
            "SELECT symbol, details FROM trade_fills "
            "WHERE details LIKE '%live_entry%' OR details LIKE '%live_exit%'"
        ).fetchall()
    except Exception:
        return set(), 0
    finally:
        conn.close()
    out, n = set(), 0
    for symbol, details in rows:
        try:
            mode = str(json.loads(details or "{}").get("mode", ""))
        except Exception:
            mode = ""
        if mode in ("live_entry", "live_exit"):
            out.add(_ticker(symbol)[0])
            n += 1
    return out, n


def resolve_round_trip(
    names: Sequence[str],
    census: Optional[dict],
    clip_usd: float,
    flat_fee: float,
    charge_gas: bool,
) -> Tuple[np.ndarray, dict]:
    """Round-trip rate per screened symbol, plus a full audit of where it came from.

    Census rates are POOL cost only (fee + impact at that clip). Gas is a
    dollar per leg, not a rate, so it is only folded in when asked for and is
    then a CLIP artifact, identical for every symbol -- which is exactly the
    token-blindness this function exists to remove. Off by default, printed
    either way.
    """
    rt = np.zeros(len(names), dtype=float)
    priced: List[dict] = []
    fell_back: List[str] = []
    gas_rate = 0.0
    if census is not None and charge_gas:
        gas_usd = float(census.get("measured_gas_context", {})
                        .get("median_round_trip_usd", 0.0) or 0.0)
        gas_rate = gas_usd / clip_usd if clip_usd > 0 else 0.0

    quoted = (census or {}).get("quoted", {})
    for i, name in enumerate(names):
        base, quote = _ticker(name)
        entry = quoted.get(base) if census is not None else None
        # The census priced USDC <-> token. A USDT-quoted feed row is a
        # DIFFERENT denomination; this repo has already been bitten by
        # denomination contamination, so it does not inherit the USDC rate.
        if entry is not None and quote not in ("USDC", ""):
            entry = None
        hit = census_rate(entry, clip_usd) if entry else None
        if hit is None:
            rt[i] = 2.0 * flat_fee
            fell_back.append(name)
            continue
        clip_used, pool = hit
        rt[i] = pool + gas_rate
        priced.append({
            "symbol": name, "ticker": base, "clip_usd": clip_used,
            "pool_rt_bp": pool * 10000.0, "gas_rt_bp": gas_rate * 10000.0,
            "charged_rt_bp": rt[i] * 10000.0,
            "fee_tier": entry.get("fee_tier"), "census_rows": entry.get("rows"),
        })
    return rt, {
        "priced": priced,
        "fell_back": fell_back,
        "flat_rt_bp": 2.0 * flat_fee * 10000.0,
        "gas_rt_bp": gas_rate * 10000.0,
        "charge_gas": bool(charge_gas),
        "clip_usd": clip_usd,
    }


def print_cost_basis(report: dict, mode: str, census: Optional[dict],
                     tick_counts: Dict[str, int]) -> None:
    """Say, every run, exactly what each symbol was charged and what it fell back to."""
    priced, fell = report["priced"], report["fell_back"]
    n_all = len(priced) + len(fell)
    t_priced = sum(tick_counts.get(r["symbol"], 0) for r in priced)
    t_all = sum(tick_counts.get(s, 0) for s in list(tick_counts))
    if mode == "flat":
        print(f"  COST BASIS: FLAT --fee. {report['flat_rt_bp']:.0f}bp round trip charged to "
              f"EVERY one of {n_all} symbols.")
        print("     This is the token-blind constant. It is here to be compared against, not")
        print("     believed: the measured chain costs below it span 2bp to 3300bp.")
        print()
        return

    meta = census or {}
    print(f"  COST BASIS: per-symbol MEASURED round trip (census), flat --fee as the fallback.")
    print(f"     census   : {meta.get('_path','?')}")
    print(f"     provenance: git {meta.get('git_sha','?')}  chain {meta.get('chain','?')}  "
          f"book_mode {meta.get('book_mode','?')}  generated {meta.get('generated_at_utc','?')}")
    print(f"     clip     : ${report['clip_usd']:.2f} requested; each symbol charged at the "
          f"nearest clip ITS curve priced")
    gas_txt = (f"CHARGED, {report['gas_rt_bp']:.2f}bp at this clip (identical for every symbol)"
               if report["charge_gas"] else
               f"NOT charged (--charge-gas adds {_gas_preview(census, report['clip_usd']):.2f}bp "
               f"at this clip, token-blind)")
    print(f"     gas      : {gas_txt}")
    print(f"     PRICED FROM THE CENSUS: {len(priced)} of {n_all} screened symbols "
          f"({t_priced} of {t_all} ticks, {100.0*t_priced/max(t_all,1):.1f}%)")
    for r in sorted(priced, key=lambda x: x["charged_rt_bp"]):
        print(f"        {r['symbol']:<16}{r['charged_rt_bp']:>10.2f}bp RT  @ ${r['clip_usd']:<6.2f}"
              f"  tier {str(r['fee_tier']):<6} census rows {r['census_rows']}")
    print(f"     FELL BACK TO THE FLAT --fee: {len(fell)} of {n_all} symbols "
          f"({t_all - t_priced} of {t_all} ticks, "
          f"{100.0*(t_all-t_priced)/max(t_all,1):.1f}%), charged "
          f"{report['flat_rt_bp']:.0f}bp each")
    if fell:
        show = sorted(fell, key=lambda s: -tick_counts.get(s, 0))
        head = ", ".join(show[:10])
        more = f" ... and {len(show)-10} more" if len(show) > 10 else ""
        print(f"        {head}{more}")
        print("        THE FALLBACK IS A CONSTANT, NOT A MEASUREMENT. Any survivor whose")
        print("        symbol is on that list was graded by an assumption.")
    else:
        print("        None. Every screened tick was graded by a measurement.")
    caveat = meta.get("THESE_ARE_QUOTES_NOT_FILLS", {})
    slip = caveat.get("live_measured_round_trip_slippage_bp")
    print(f"     QUOTES ARE NOT FILLS: census rates are pool cost (fee + impact) only. The")
    print(f"        live lane measured {slip}bp of round-trip slippage no quote here carries,")
    print(f"        and MEV is unmodelled. Every cost below is a FLOOR, so 'net positive' is")
    print(f"        the most optimistic reading available and not a realisable P/L.")
    print()


def _gas_preview(census: Optional[dict], clip_usd: float) -> float:
    gas_usd = float((census or {}).get("measured_gas_context", {})
                    .get("median_round_trip_usd", 0.0) or 0.0)
    return (gas_usd / clip_usd if clip_usd > 0 else 0.0) * 10000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--split", type=float, default=0.6, help="in-sample share by TIME")
    ap.add_argument("--fee", type=float, default=0.003,
                    help="one-way fee rate. USED ONLY as the explicit fallback for symbols "
                         "the cost census does not price (or everywhere, under "
                         "--cost-mode flat). Every run prints which symbols it covered.")
    ap.add_argument("--cost-mode", choices=("census", "flat"), default="census",
                    help="census (default): charge each symbol its MEASURED round trip from "
                         "--cost-census, falling back to --fee where it is not priced. "
                         "flat: charge 2*--fee to everything, the old token-blind behaviour, "
                         "kept so the two can be compared on the same tape and seed.")
    ap.add_argument("--cost-census", type=str, default=DEFAULT_COST_CENSUS,
                    help="measured round-trip quote census artifact")
    ap.add_argument("--cost-clip", type=float, default=0.0,
                    help="clip in USD to charge the cost at; 0 = the census's own median "
                         "clip from the live book")
    ap.add_argument("--charge-gas", action="store_true",
                    help="also charge the census's measured median round-trip gas as a rate "
                         "at --cost-clip. It is the SAME number for every symbol, so it is "
                         "a clip artifact, not a token cost. Off by default, printed always.")
    ap.add_argument("--fillable-only", action="store_true",
                    help="restrict the universe to symbols the live lane can actually fill. "
                         "An edge in a pool we cannot trade is not a result -- and this run "
                         "prices what the restriction costs you: on 168h the permutation bar "
                         "rose 2-3x and held-out survivors fell 11 to 1.")
    ap.add_argument("--fillable-source", choices=("census", "live-fills"), default="census",
                    help="census (default): a symbol is fillable if the round-trip census "
                         "priced a direct USDC pool for it through the one venue that answers "
                         "on Base. live-fills: only symbols with a REAL executed leg in "
                         "trade_fills. The two disagree; both sets are printed.")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--cuts", type=str, default="",
                    help="comma-separated quantile cuts, overrides CUTS")
    ap.add_argument("--horizons", type=str, default="",
                    help="comma-separated horizons in minutes, overrides HORIZONS_MIN")
    ap.add_argument("--only", type=str, default="",
                    help="screen only features whose name starts with this prefix")
    ap.add_argument("--min-depth", type=float, default=0.0,
                    help="restrict the universe to symbols with at least this much "
                         "pool liquidity in USD, per data/pool_depth_snapshot.json. "
                         "A symbol ABSENT from the snapshot is excluded, not assumed deep.")
    ap.add_argument("--depth-json", type=str, default="data/pool_depth_snapshot.json")
    args = ap.parse_args()

    global CUTS, HORIZONS_MIN, ONLY_PREFIX
    if args.cuts:
        CUTS = tuple(float(x) for x in args.cuts.split(","))
    if args.horizons:
        HORIZONS_MIN = tuple(float(x) for x in args.horizons.split(","))
    ONLY_PREFIX = args.only

    rng = np.random.default_rng(20260916)

    print("=" * 88)
    print("PREDICATE FORGE -- enumerate the space, and price the selection")
    print("=" * 88)

    # The cost census is loaded BEFORE the tape, because --fillable-only reads
    # its universe out of it. A missing or wrong-shaped artifact is a hard
    # stop, never a silent slide back to the constant.
    census: Optional[dict] = None
    if args.cost_mode == "census" or args.fillable_only:
        try:
            census = load_cost_census(args.cost_census)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  COST CENSUS UNUSABLE: {exc}")
            print("  Refusing to fall back silently to the flat --fee. Pass --cost-mode flat")
            print("  if a token-blind constant is really what you want.")
            return 2
    clip = args.cost_clip or float(
        (census or {}).get("clip_distribution_usd", {}).get("median", 2.0) or 2.0)

    by_symbol = load_ticks(args.hours)
    if not by_symbol:
        print("  no tape. nothing to screen.")
        return 1

    if args.fillable_only:
        census_set = set((census or {}).get("quoted", {}))
        fills_set, n_legs = live_filled_tickers()
        if args.fillable_source == "census":
            keep, defn = census_set, (
                f"the {len(census_set)} symbols the round-trip census PRICED -- i.e. a direct "
                f"USDC pool exists for them on {(census or {}).get('chain','?')} and the one "
                f"venue that answers there quoted a whole round trip")
        else:
            keep, defn = fills_set, (
                f"the {len(fills_set)} symbols with a REAL executed leg in trade_fills "
                f"({n_legs} live legs)")
        before = len(by_symbol)
        by_symbol = {s: r for s, r in by_symbol.items()
                     if _ticker(s)[0] in keep and _ticker(s)[1] in ("USDC", "")}
        print(f"  FILLABLE FILTER (--fillable-source {args.fillable_source}): keeps "
              f"{len(by_symbol)} of {before} tape symbols.")
        print(f"     DEFINITION USED: {defn}.")
        print(f"     Nothing better exists in this repo -- there is no 'live lane can fill this'")
        print(f"     list anywhere -- so this is a stand-in, and it is the one being reported.")
        print(f"     census-priced set : {', '.join(sorted(census_set)) or '(none)'}")
        print(f"     live-filled set   : {', '.join(sorted(fills_set)) or '(none)'}  "
              f"({n_legs} live legs)")
        both = sorted(census_set & fills_set)
        print(f"     THE TWO DISAGREE: overlap is {', '.join(both) or '(none)'}. "
              f"{len(fills_set - census_set)} symbols we DID fill are unquotable in the census")
        print(f"     ({', '.join(sorted(fills_set - census_set)) or 'none'}), so neither set is "
              f"a superset of the other.")
        print(f"     WHY: a result from the full universe is mostly a result about pools we")
        print(f"     cannot trade. WHAT IT COSTS, measured by this script on 168h: the")
        print(f"     permutation bar ROSE 2-3x (4.65 -> 13.72bp at 2m) because fewer symbols")
        print(f"     means fewer independent rows, and held-out survivors fell 11 to 1. Do not")
        print(f"     cite the old '+15.77bp vs +0.88bp' pair for this -- pass 448 retroactively")
        print(f"     bounded the +0.88bp deep-subset number as a concentration artifact.")
        if not by_symbol:
            print("  no fillable symbol is in the tape. nothing to screen.")
            return 1
        print("  NOTE: on a universe this small the cross-sectional families (xs_*, crowd*,")
        print("  idio*) rank a symbol against very few peers and are near-degenerate.")
        print()

    if args.min_depth > 0:
        raw = json.loads(Path(args.depth_json).read_text(encoding="utf-8"))
        depth = {k.upper(): float(v) for k, v in raw.items()
                 if not k.startswith("_") and isinstance(v, (int, float))}
        before = len(by_symbol)
        by_symbol = {s: r for s, r in by_symbol.items()
                     if depth.get(s.split("-")[0].upper(), 0.0) >= args.min_depth}
        print(f"  DEPTH FILTER: >= ${args.min_depth:,.0f} liquidity keeps "
              f"{len(by_symbol)} of {before} symbols "
              f"({len(depth)} have a depth reading at all; the rest are EXCLUDED, "
              f"not assumed deep).")
        if not by_symbol:
            print("  no symbol in the tape clears that depth. nothing to screen.")
            return 1
        print("  NOTE: cross-sectional features rank a symbol against its peers, so on a")
        print("  universe this small xs_* and crowd* are near-degenerate. Read the")
        print("  price-action families here; the xs_* result belongs to the full run.")
        print("  DEEP UNIVERSE, per symbol (a symbol under 40 ticks is dropped by")
        print("  build_features and contributes nothing):")
        for s in sorted(by_symbol, key=lambda k: -len(by_symbol[k])):
            r = by_symbol[s]
            span_h = (r[-1][0] - r[0][0]) / 3600.0 if len(r) > 1 else 0.0
            drop = "  DROPPED (<40 ticks)" if len(r) < 40 else ""
            print(f"    {s:<22} depth ${depth.get(s.split('-')[0].upper(), 0.0):>14,.0f}"
                  f"  ticks {len(r):>6}  span {span_h:>6.1f}h{drop}")

    feats, ts, sid, fwd = build_features(by_symbol)
    if len(ts) == 0:
        print("  no usable ticks after warmup.")
        return 1
    # Same ordering build_features assigns sid by, so sid -> symbol name.
    names = sorted(s for s, r in by_symbol.items() if len(r) >= 40)

    cut_ts = np.quantile(ts, args.split)
    mask_is = ts <= cut_ts
    mask_os = ts > cut_ts

    preds = make_predicates(feats, mask_is)
    # --hours is a REQUEST, not a fact: load_ticks cuts back from MAX(ts), so
    # asking for more tape than the store holds silently returns less. Print
    # what we actually got, or the run reports a span it never measured.
    span_h = (float(ts.max()) - float(ts.min())) / 3600.0
    print(f"  tape      : {len(by_symbol)} symbols, {len(ts)} ticks over "
          f"{span_h:.1f}h ACTUAL (requested {args.hours:g}h)")
    print(f"  features  : {len(feats)}   predicates: {len(preds)}")
    print(f"  split     : in-sample {int(mask_is.sum())} ticks | held-out {int(mask_os.sum())}")
    print(f"  horizons  : {', '.join(f'{h:g}m' for h in HORIZONS_MIN)}")
    print()

    # Per-symbol round trip, expanded to one rate per TICK so each trade pays
    # the cost of the token it was actually in.
    rt_by_sid, cost_report = resolve_round_trip(
        names, census if args.cost_mode == "census" else None,
        clip, args.fee, args.charge_gas)
    rt_row = rt_by_sid[sid]
    priced_names = {r["symbol"] for r in cost_report["priced"]}
    priced_row = np.array([n in priced_names for n in names], dtype=float)[sid]
    print_cost_basis(cost_report, args.cost_mode, census,
                     {n: len(by_symbol[n]) for n in names})

    results = []
    for H in HORIZONS_MIN:
        y = fwd[H]
        lifts_is, sup_is = screen(preds, y, mask_is)
        ok = sup_is >= MIN_SUPPORT
        if not ok.any():
            print(f"  [{H:g}m] no predicate reaches support {MIN_SUPPORT}. skipped.")
            continue

        blocks, blk_report = block_lengths(y, ts, mask_is, sid, H)
        bar_dist = permutation_bar(preds, y, mask_is, sid, args.perms, rng,
                                   method="block", blocks=blocks)
        bar95 = float(np.quantile(bar_dist, 0.95))
        bar50 = float(np.quantile(bar_dist, 0.50))

        # The OLD null, run on the same ticks at the same horizon and the same
        # number of draws, so the size of the correction is VISIBLE rather than
        # silently applied. It is not used to judge anything.
        iid_dist = permutation_bar(preds, y, mask_is, sid, args.perms, rng, method="iid")
        iid95 = float(np.quantile(iid_dist, 0.95))
        iid50 = float(np.quantile(iid_dist, 0.50))

        best = int(np.argmax(np.where(ok, np.abs(lifts_is), -np.inf)))
        best_lift = float(lifts_is[best])

        widen = (bar95 / iid95) if iid95 > 0 else float("nan")
        print(f"  [{H:g}m horizon]")
        print(f"     selection bar -- best |lift| a NULL tape reaches, {args.perms} draws each:")
        print(f"        {'null':<34}{'median':>10}{'95th pct':>11}")
        print(f"        {'i.i.d. permutation (OLD, too narrow)':<34}{iid50:9.2f}bp{iid95:10.2f}bp")
        print(f"        {'circular block bootstrap (USED)':<34}{bar50:9.2f}bp{bar95:10.2f}bp")
        print(f"        correction: the honest bar is {widen:.2f}x the i.i.d. bar."
              f"  block={blk_report['median_block']} ticks (median over"
              f" {len(blk_report['per_symbol'])} symbols), from median lag-1 rho"
              f" {blk_report['median_rho1']:+.4f} and {blk_report['median_overlap_ticks']}"
              f" ticks of {H:g}m overlap"
              f"\n        overlap floor respected: {blk_report['floor_respected']}"
              f" ({blk_report['whole_series_symbols']} symbol(s) shorter than one horizon"
              f" are ONE block and never shuffled;"
              f" {blk_report['thin_symbols']} have fewer than 5 blocks)")
        print(f"     best real predicate: {preds[best][0]:<22} {best_lift:+8.2f}bp  n={int(sup_is[best])}")
        n_iid = int((ok & (np.abs(lifts_is) > iid95)).sum())
        n_blk = int((ok & (np.abs(lifts_is) > bar95)).sum())
        if n_iid != n_blk:
            print(f"     DEMOTED BY THE CORRECTION: {n_iid - n_blk} predicate(s) cleared the"
                  f" i.i.d. bar and do NOT clear the block bar ({n_iid} -> {n_blk}).")
        elif n_iid == 0:
            print(f"     survivors under BOTH nulls: 0. The correction changes nothing here"
                  f" because nothing cleared the narrow bar either.")
        else:
            print(f"     survivor count unchanged by the correction: {n_blk} under both nulls.")

        survivors = np.where(ok & (np.abs(lifts_is) > bar95))[0]
        if len(survivors) == 0:
            print(f"     SURVIVORS: NONE. The best real predicate does not beat what")
            print(f"     shuffling the labels produces. Nothing here is an edge.\n")
            results.append({"horizon_min": H, "bar95_bp": bar95,
                        "null": "circular_block_bootstrap",
                        "bar95_iid_bp": iid95, "block_ticks": blk_report["median_block"],
                        "block_report": blk_report, "survivors": 0,
                            "best": preds[best][0], "best_lift_bp": best_lift})
            continue

        order = survivors[np.argsort(-np.abs(lifts_is[survivors]))][: args.top]
        print(f"     SURVIVORS IN-SAMPLE: {len(survivors)} of {int(ok.sum())} screened."
              f" Held-out test follows.")
        print(f"     {'predicate':<24}{'IS lift':>9}{'IS n':>8}{'OOS lift':>10}{'OOS n':>8}"
              f"{'RT cost':>10}{'net/trade':>11}  {'%census':>8}")
        rows = []
        for i in order:
            name, p = preds[i]
            l_os, s_os = screen([(name, p)], y, mask_os)
            oos = float(l_os[0])
            n_os = int(s_os[0])
            m = p & mask_os & np.isfinite(y)
            thick = n_os >= MIN_SUPPORT
            gross = float(y[m].mean()) if thick else float("nan")
            # Each row pays the round trip of the token it was in. The mean of
            # those rates is what this predicate actually costs -- not a
            # constant, and not the same number for two different predicates.
            cost = float(rt_row[m].mean()) if thick else float("nan")
            net = (gross - cost) * 10000.0 if thick else float("nan")
            # What share of this predicate's held-out rows were graded by a
            # MEASUREMENT rather than by the --fee assumption.
            cov = float(priced_row[m].mean()) * 100.0 if thick else float("nan")
            flag = "" if thick else "  (thin)"
            print(f"     {name:<24}{lifts_is[i]:>+8.2f}{int(sup_is[i]):>8}"
                  f"{oos:>+9.2f}{n_os:>8}{cost*10000.0:>9.2f}{net:>+10.2f}"
                  f"{cov:>9.0f}%{flag}")
            rows.append({"predicate": name, "is_lift_bp": float(lifts_is[i]),
                         "is_n": int(sup_is[i]), "oos_lift_bp": oos, "oos_n": n_os,
                         "oos_cost_bp": cost * 10000.0, "oos_net_bp": net,
                         "oos_census_priced_pct": cov})
        # CONCENTRATION GUARD. This repo has repeatedly shipped a headline that
        # a handful of rows carried: 98% of gross in 7 rows, six rows flipping
        # the sign of the whole ghost book. A mean lift is not evidence until
        # you know how few symbols made it. For every survivor, report the
        # single symbol carrying the most gross and what the lift becomes with
        # that symbol removed -- if it flips sign, the predicate is that one
        # symbol and nothing else.
        for r in rows:
            if r["oos_n"] < MIN_SUPPORT:
                continue
            p = dict(preds)[r["predicate"]]
            m = p & mask_os & np.isfinite(y)
            tot = float(y[m].sum())
            worst_s, worst_share, censored = -1, 0.0, float("nan")
            for s in np.unique(sid[m]):
                ms = m & (sid == s)
                share = float(y[ms].sum()) / tot if abs(tot) > 1e-12 else 0.0
                if share > worst_share:
                    rest = m & (sid != s)
                    worst_s, worst_share = int(s), share
                    censored = float(y[rest].mean()) * 10000.0 if rest.sum() else float("nan")
            r["top_symbol"] = names[worst_s] if worst_s >= 0 else ""
            r["top_symbol_gross_share"] = worst_share
            r["lift_bp_without_top_symbol"] = censored
            if worst_share > 0.5 or (np.isfinite(censored)
                                     and np.sign(censored) != np.sign(r["oos_lift_bp"])):
                print(f"     !! {r['predicate']}: {r['top_symbol']} carries "
                      f"{worst_share*100:.0f}% of OOS gross; without it the lift is "
                      f"{censored:+.2f}bp. ONE-SYMBOL ARTIFACT, not a predicate.")

        held = [r for r in rows if r["oos_n"] >= MIN_SUPPORT
                and np.sign(r["oos_lift_bp"]) == np.sign(r["is_lift_bp"])
                and abs(r["oos_lift_bp"]) > bar95]
        print(f"     HELD OUT OF SAMPLE (same sign, still over the bar): {len(held)}")
        if held:
            paying = [r for r in held if r["oos_net_bp"] > 0]
            basis = ("its own MEASURED round trip" if args.cost_mode == "census"
                     else f"a flat {2*args.fee*10000:.0f}bp round trip")
            print(f"     ... and NET POSITIVE after {basis}: {len(paying)}")
            for r in paying:
                fb = "" if r["oos_census_priced_pct"] >= 99.5 else (
                    f"  [only {r['oos_census_priced_pct']:.0f}% of its rows were "
                    f"PRICED; the rest paid the --fee assumption]")
                print(f"         *** {r['predicate']} -> {r['oos_net_bp']:+.2f}bp/trade"
                      f" on n={r['oos_n']} after {r['oos_cost_bp']:.2f}bp cost{fb}")
        print()
        results.append({"horizon_min": H, "bar95_bp": bar95,
                        "null": "circular_block_bootstrap",
                        "bar95_iid_bp": iid95, "block_ticks": blk_report["median_block"],
                        "block_report": blk_report, "bar50_bp": bar50,
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
        print("  A predicate cleared selection, held out of sample, and pays the cost.")
        print("  It is a CANDIDATE, not an edge: re-run on a later tape before trading it.")
        if args.cost_mode == "census":
            print("  And the cost it paid is a FLOOR -- pool fee and impact from a QUOTE, with")
            print("  no slippage (13.5bp measured live), no MEV and, unless --charge-gas, no")
            print("  gas. Subtract those before calling anything payable.")
    else:
        print("  NOTHING in this space clears its own selection bar out of sample and")
        print("  pays the round trip. That is a real result: it prices how much of a")
        print("  'discovered edge' this tape can manufacture from noise alone, and the")
        print("  answer is more than anything genuinely present.")
    print("=" * 88)

    if args.json:
        payload = {
            "cost_mode": args.cost_mode,
            "cost_census": args.cost_census if args.cost_mode == "census" else None,
            "cost_census_git_sha": (census or {}).get("git_sha") if census else None,
            "cost_clip_usd": clip,
            "charge_gas": bool(args.charge_gas),
            "flat_fee_one_way": args.fee,
            "fillable_only": bool(args.fillable_only),
            "fillable_source": args.fillable_source if args.fillable_only else None,
            "cost_basis": cost_report,
            "hours_requested": args.hours,
            "span_hours_actual": span_h,
            "symbols": len(by_symbol),
            "ticks": int(len(ts)),
            "horizons": list(HORIZONS_MIN),
            "results": results,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
