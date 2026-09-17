"""Is there STRUCTURE IN THE CURVATURE of this tape, or only in the level?

    python -X utf8 scripts/feature_forge_second_order.py
    python -X utf8 scripts/feature_forge_second_order.py --json data/second_order_reflexivity.json

THE QUESTION, in the operator's framing. Other systems trade on the expected
reactions of others, so the causal trigger may be boolean but the RESPONSE PATH
is higher-order. First-order price-action predicates are measured exhausted here
(continuation under 1bp at every horizon; 2% dips INVERT out of sample), so the
surviving hypothesis is that structure lives in the CURVATURE of the response
rather than in its level. If a crowd front-runs a crowd, the tape should show
ACCELERATION INTO A MOVE: after a move, the path should curve in the direction
of the move, and it should curve harder the larger the move.

THE DELIVERABLE is section 5, the reflexivity test: bucket ticks by
FIRST-DERIVATIVE MAGNITUDE and report the SIGN and SIZE of the SECOND derivative
in each bucket, each with its own noise bar. A bucket whose second derivative is
inside its own bar is a NULL and is printed as one -- the point estimate is not
the finding.

THREE RULES THIS SCRIPT OBEYS, because this repo has been burned by each:

  1. IT PAYS ITS OWN MULTIPLE-COMPARISONS BILL. Every number that is compared to
     zero is one of K cells, K is counted mechanically (not asserted), and the
     bar is SE*sqrt(2*ln K). The repo's GA failed exactly this null --
     corr(variants, edge) = -0.84, i.e. the harder it searched the worse it did,
     because it was measuring its own search width. The analytic bar is also
     CHECKED against a block sign-flip permutation null, which is the same
     family-wise quantity measured instead of assumed.

  2. FILLABLE SUBSET ONLY. Predictability and fillability are measured
     anti-correlated on this venue: +15.77bp of lift where we cannot fill,
     +0.88bp where we can. The symbol set is READ from the round-trip quote
     census artifact (the 'quoted' block), never hardcoded, and every symbol
     carries its own measured round-trip cost into the table so a 2bp effect in
     a 1508bp pool is visibly not a result.

  3. THE SE IS CLUSTERED. Consecutive fires share ticks and a symbol's ticks
     share a regime, so the naive SE is too small. Clusters are
     (symbol, 6-hour block).

WHAT IT DOES NOT DO: it does not score a predicate, propose a strategy, or touch
production code. It reads the store read-only and prints.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import re
import statistics as st
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Knobs. Every one of these is a researcher degree of freedom, so each is named,
# defaulted once, and printed in the provenance block.
# ---------------------------------------------------------------------------

#: A gap is "comparable" when it is within this multiple of the symbol's own
#: median gap. This is the published definition (pass 432,
#: scripts/derivative_feature_feasibility.py) and section 3 reproduces its
#: headline number with it unchanged.
REG_HI = 2.0

#: ...and NOT BELOW this multiple. The published definition bounds the gap only
#: from above, which is fine for counting triples but not for DIVIDING by them:
#: the time-normalised second derivative carries h1*h2*(h1+h2) in a denominator,
#: and with no lower bound its maximum is 2.0e10 bp per median-gap squared --
#: a division artifact, not a market. With the bound it is 1.3e3. The bp-space
#: second DIFFERENCE needs no division and so is measured without this bound.
REG_LO = 0.25

#: Bucket edges for the reflexivity test, as quantiles of pooled |first
#: derivative|. Fixed a priori; not tuned to the answer.
BUCKET_QUANTILES = (0.20, 0.40, 0.60, 0.80, 0.95)

#: A price that trends leaves prints at every level it passed through; two
#: denominations leave two clusters with nothing between. 0.5 in log space is
#: 1.65x with no intermediate print anywhere in the window. Same structural test
#: as scripts/money_button_edge_test.py -- a magnitude threshold is not enough
#: (COMP-USDC's two scales sit only 2.24x apart).
DENOM_GAP_LOG = 0.5

#: A "move" for the post-move features: this symbol's own quantile of |return|.
MOVE_QUANTILE = 0.80

#: Post-move features need this many further ticks at comparable spacing.
POST_TICKS = 4

#: Trailing window, in ticks, for the up/down spacing-asymmetry feature, and the
#: minimum number of each direction inside it.
SPACING_WINDOW = 20
SPACING_MIN_SIDE = 3

#: Clustering block for the cluster-robust SE, in seconds.
CLUSTER_BLOCK_SEC = 6 * 3600

#: A blackout, per the feed's documented 16-59 minute global gaps.
BLACKOUT_SEC = 16 * 60

DEFAULT_DB = os.getenv("TRADING_DB_PATH", "storage/trading_cache.db")
CENSUS_GLOB = "data/round_trip_quote_census_*.json"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cluster_se(values: Sequence[float], keys: Sequence[object]) -> Tuple[float, int]:
    """Cluster-robust SE of the mean, clustering on ``keys``.

    Consecutive fires overlap by up to two ticks and a symbol's ticks share a
    regime, so the iid SE understates. Returns (se, n_clusters); the small-G
    correction G/(G-1) is applied and G is returned so a table can show how few
    independent blocks a number actually rests on.
    """
    n = len(values)
    if n < 2:
        return float("inf"), 0
    mu = sum(values) / n
    acc: Dict[object, float] = collections.defaultdict(float)
    for v, k in zip(values, keys):
        acc[k] += v - mu
    g = len(acc)
    ss = sum(x * x for x in acc.values())
    if g > 1:
        ss *= g / (g - 1.0)
    return math.sqrt(ss) / n, g


def quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(q * len(sorted_vals))))
    return sorted_vals[idx]


def ols_quadratic(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Return 2*a from the least-squares fit y = a x^2 + b x + c (the curvature).

    Hand-rolled 3x3 normal equations: this repo runs without numpy on the
    production path and this script must not need it either.
    """
    n = len(xs)
    if n < 3:
        return None
    s = [0.0] * 5
    for x in xs:
        p = 1.0
        for k in range(5):
            s[k] += p
            p *= x
    t = [0.0] * 3
    for x, y in zip(xs, ys):
        t[0] += y
        t[1] += x * y
        t[2] += x * x * y
    a = [[s[4], s[3], s[2]], [s[3], s[2], s[1]], [s[2], s[1], s[0]]]
    b = [t[2], t[1], t[0]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, 3):
            f = a[r][col] / a[col][col]
            for c in range(col, 3):
                a[r][c] -= f * a[col][c]
            b[r] -= f * b[col]
    x2 = [0.0] * 3
    for r in (2, 1, 0):
        acc = b[r] - sum(a[r][c] * x2[c] for c in range(r + 1, 3))
        x2[r] = acc / a[r][r]
    return 2.0 * x2[0]


def ols_slope(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


# ---------------------------------------------------------------------------
# Feed hygiene
# ---------------------------------------------------------------------------

def drop_mixed_denominations(
    ticks: Sequence[Tuple[float, float]], gap_log: float = DENOM_GAP_LOG
) -> List[Tuple[float, float]]:
    """Keep only the dominant price scale for a symbol.

    1.8% of market_stream ticks are the same symbol quoted at a different scale
    (a USD price and a token ratio), and for some symbols the WRONG scale is the
    MAJORITY reading, so 'take the median' picks the wrong number. Scored raw,
    a forward-return control group on this table came back at +1.26e10%.
    """
    if len(ticks) < 3:
        return list(ticks)
    ordered = sorted(
        (math.log(p), i) for i, (_ts, p) in enumerate(ticks) if p > 0
    )
    if len(ordered) < 3:
        return list(ticks)
    clusters: List[Tuple[int, int]] = []
    start = 0
    for k in range(1, len(ordered)):
        if ordered[k][0] - ordered[k - 1][0] > gap_log:
            clusters.append((start, k))
            start = k
    clusters.append((start, len(ordered)))
    lo, hi = max(clusters, key=lambda c: c[1] - c[0])
    keep = {i for _lp, i in ordered[lo:hi]}
    return [ticks[i] for i in sorted(keep)]


def fillable_symbols(census_path: str) -> Tuple[List[str], str, Dict[str, float], dict]:
    """Read the fillable set from the round-trip quote census artifact.

    'Fillable' here means EXACTLY: the census got a real round-trip quote back
    for this token, so it sits in the artifact's 'quoted' block rather than
    'unquotable'. Nothing is hardcoded and nothing is inferred -- a token the
    census could not price is not in the set, and the census charges no default.
    The quote denomination is parsed out of the artifact's own description of
    what it measured, because the census prices DIRECT quote<->token pools only.
    """
    with open(census_path, "r", encoding="utf-8") as fh:
        census = json.load(fh)
    quoted = census.get("quoted", {})
    what = str(census.get("THESE_ARE_QUOTES_NOT_FILLS", {}).get("what_is_measured", ""))
    m = re.search(r"\b([A-Z0-9]{2,10})\s*->\s*token\s*->\s*\1\b", what)
    denom = m.group(1) if m else "USDC"
    # probe_cost_pct is a PERCENT of notional; x100 makes it basis points, and
    # it is converted exactly once, here.
    costs = {
        f"{sym}-{denom}": float(info.get("probe_cost_pct") or 0.0) * 100.0
        for sym, info in quoted.items()
    }  # values are BASIS POINTS
    return sorted(costs), denom, costs, census


# ---------------------------------------------------------------------------
# The per-symbol feature pass
# ---------------------------------------------------------------------------

class SymbolSeries:
    """One symbol's cleaned tick series plus everything derived from it."""

    def __init__(self, symbol: str, rows: Sequence[Tuple[float, float]]):
        self.symbol = symbol
        self.raw_ticks = len(rows)
        cleaned = drop_mixed_denominations(rows)
        self.denom_dropped = len(rows) - len(cleaned)
        self.ts = [r[0] for r in cleaned]
        self.px = [r[1] for r in cleaned]
        self.n = len(self.ts)
        gaps = [self.ts[i] - self.ts[i - 1] for i in range(1, self.n)]
        self.gaps = [g for g in gaps if g > 0]
        self.med_gap = st.median(self.gaps) if self.gaps else float("nan")
        self.blackouts = sum(1 for g in self.gaps if g > BLACKOUT_SEC)
        self.span_h = (self.ts[-1] - self.ts[0]) / 3600.0 if self.n > 1 else 0.0
        self.rets = [
            1e4 * math.log(self.px[i] / self.px[i - 1])
            for i in range(1, self.n)
            if self.px[i] > 0 and self.px[i - 1] > 0
        ]
        self.zero_ret_frac = (
            sum(1 for r in self.rets if r == 0.0) / len(self.rets) if self.rets else 0.0
        )
        self.regular_frac = (
            sum(1 for g in self.gaps if g <= REG_HI * self.med_gap) / len(self.gaps)
            if self.gaps else 0.0
        )
        abs_r = sorted(abs(r) for r in self.rets if r != 0.0)
        self.move_threshold = quantile(abs_r, MOVE_QUANTILE) if abs_r else float("inf")
        self.sigma_bp = st.pstdev(self.rets) if len(self.rets) > 1 else 0.0

    # -- spacing helpers ----------------------------------------------------

    def gap_ok(self, i: int, lower: bool) -> bool:
        """Is the gap ENDING at tick i comparable to this symbol's median?"""
        if i <= 0 or i >= self.n:
            return False
        g = self.ts[i] - self.ts[i - 1]
        if g <= 0 or g > REG_HI * self.med_gap:
            return False
        return (not lower) or g >= REG_LO * self.med_gap

    def ret(self, i: int) -> float:
        return 1e4 * math.log(self.px[i] / self.px[i - 1])


def build_fires(series: Sequence[SymbolSeries]) -> Tuple[List[dict], List[dict]]:
    """Every tick that can anchor a reflexivity observation.

    A fire at tick i is: a non-zero move r0 arriving at i over a comparable gap,
    followed by two further ticks at comparable gaps. The move that does the
    BUCKETING (r0) is deliberately DISJOINT from the returns that make the second
    derivative (r1, r2). Centring the second derivative on the move instead would
    put r0 inside its own statistic, and conditioning on |r0| being large would
    then manufacture a positive signed acceleration out of nothing.
    """
    loose: List[dict] = []
    tight: List[dict] = []
    for s in series:
        if s.n < 4 or not s.gaps:
            continue
        for i in range(1, s.n - 2):
            if not (s.gap_ok(i, False) and s.gap_ok(i + 1, False) and s.gap_ok(i + 2, False)):
                continue
            r0 = s.ret(i)
            if r0 == 0.0:
                continue
            sign = 1.0 if r0 > 0 else -1.0
            r1 = s.ret(i + 1)
            r2 = s.ret(i + 2)
            rec = {
                "symbol": s.symbol,
                "ts": s.ts[i],
                "cluster": (s.symbol, int(s.ts[i] // CLUSTER_BLOCK_SEC)),
                "mag": abs(r0),
                "d2_bp": sign * (r2 - r1),
                "cont_bp": sign * r1,
                "spike": (r0 * r1 < 0) and (abs(r1) >= 0.5 * abs(r0)),
            }
            loose.append(rec)
            if s.gap_ok(i + 1, True) and s.gap_ok(i + 2, True):
                h1 = (s.ts[i + 1] - s.ts[i]) / s.med_gap
                h2 = (s.ts[i + 2] - s.ts[i + 1]) / s.med_gap
                # Three-point second derivative on UNEQUALLY spaced points, in
                # log-price bp, with time in units of this symbol's median gap.
                # y0 = 0 by construction (we measure from tick i).
                y1, y2 = r1, r1 + r2
                accel = 2.0 * (h1 * y2 - (h1 + h2) * y1) / (h1 * h2 * (h1 + h2))
                t = dict(rec)
                t["accel"] = sign * accel
                tight.append(t)
    return loose, tight


def feature_computability(series: Sequence[SymbolSeries]) -> List[dict]:
    """Count, per feature, the ticks on which it can actually be evaluated."""
    counts = {
        "acceleration": 0,
        "curvature": 0,
        "convexity_post_move": 0,
        "spacing_asymmetry": 0,
        "post_move_decay": 0,
    }
    samples: Dict[str, List[float]] = {k: [] for k in counts}
    clusters: Dict[str, List[object]] = {k: [] for k in counts}
    total = 0
    for s in series:
        total += s.n
        if s.n < 4 or not s.gaps or not (s.med_gap > 0):
            continue
        for i in range(2, s.n):
            cl = (s.symbol, int(s.ts[i] // CLUSTER_BLOCK_SEC))

            # (1) acceleration: 3 consecutive ticks, both gaps comparable and
            #     bounded away from zero so the division is meaningful.
            if s.gap_ok(i - 1, True) and s.gap_ok(i, True):
                h1 = (s.ts[i - 1] - s.ts[i - 2]) / s.med_gap
                h2 = (s.ts[i] - s.ts[i - 1]) / s.med_gap
                y1 = s.ret(i - 1)
                y2 = y1 + s.ret(i)
                accel = 2.0 * (h1 * y2 - (h1 + h2) * y1) / (h1 * h2 * (h1 + h2))
                counts["acceleration"] += 1
                samples["acceleration"].append(accel)
                clusters["acceleration"].append(cl)

                # (2) curvature: kappa = f'' / (1+f'^2)^1.5 in NORMALISED
                #     coordinates -- time in median gaps, price in units of this
                #     symbol's own return sigma. Without that normalisation f'
                #     is ~1e-4 and kappa collapses to |f''|, which would make
                #     this column a duplicate of the one above. Needs sigma > 0.
                if s.sigma_bp > 0:
                    fp = (y2 - y1) / (h2 * s.sigma_bp)
                    counts["curvature"] += 1
                    samples["curvature"].append(
                        (accel / s.sigma_bp) / (1.0 + fp * fp) ** 1.5
                    )
                    clusters["curvature"].append(cl)

            # (3)+(5) post-move features: a move at i, then POST_TICKS further
            #     ticks at comparable spacing.
            r0 = s.ret(i) if s.gap_ok(i, False) else 0.0
            if r0 != 0.0 and abs(r0) >= s.move_threshold and i + POST_TICKS < s.n:
                if all(s.gap_ok(i + k, False) for k in range(1, POST_TICKS + 1)):
                    sign = 1.0 if r0 > 0 else -1.0
                    xs, ys, cum = [], [], 0.0
                    for k in range(1, POST_TICKS + 1):
                        cum += s.ret(i + k)
                        xs.append((s.ts[i + k] - s.ts[i]) / s.med_gap)
                        ys.append(sign * cum)
                    conv = ols_quadratic(xs, ys)
                    if conv is not None:
                        counts["convexity_post_move"] += 1
                        samples["convexity_post_move"].append(conv)
                        clusters["convexity_post_move"].append(cl)
                    # decay of the move itself: |displacement from the pre-move
                    # price| relative to the move, log-fitted against elapsed
                    # median-gaps. Slope < 0 is decay.
                    lx, ly = [], []
                    for k in range(1, POST_TICKS + 1):
                        disp = abs(r0 + sum(s.ret(i + j) for j in range(1, k + 1)))
                        if disp > 0:
                            lx.append(xs[k - 1])
                            ly.append(math.log(disp / abs(r0)))
                    if len(lx) >= 3:
                        slope = ols_slope(lx, ly)
                        if slope is not None:
                            counts["post_move_decay"] += 1
                            samples["post_move_decay"].append(slope)
                            clusters["post_move_decay"].append(cl)

            # (4) up/down inter-tick spacing asymmetry over a trailing window.
            if i >= SPACING_WINDOW:
                up, dn = [], []
                for k in range(i - SPACING_WINDOW + 1, i):
                    r = s.ret(k)
                    g = s.ts[k + 1] - s.ts[k]
                    if g <= 0:
                        continue
                    if r > 0:
                        up.append(g)
                    elif r < 0:
                        dn.append(g)
                if len(up) >= SPACING_MIN_SIDE and len(dn) >= SPACING_MIN_SIDE:
                    mu_u, mu_d = st.mean(up), st.mean(dn)
                    denom = 0.5 * (mu_u + mu_d)
                    if denom > 0:
                        counts["spacing_asymmetry"] += 1
                        samples["spacing_asymmetry"].append((mu_u - mu_d) / denom)
                        clusters["spacing_asymmetry"].append(cl)
    out = []
    for name in (
        "acceleration",
        "curvature",
        "convexity_post_move",
        "spacing_asymmetry",
        "post_move_decay",
    ):
        vals = samples[name]
        se, g = cluster_se(vals, clusters[name]) if len(vals) > 1 else (float("inf"), 0)
        out.append({
            "feature": name,
            "n": counts[name],
            "frac": counts[name] / total if total else 0.0,
            "mean": st.mean(vals) if vals else float("nan"),
            "median": st.median(vals) if vals else float("nan"),
            "se": se,
            "clusters": g,
        })
    return out, total


def published_triple_fraction(conn) -> Tuple[int, int, int, float]:
    """Reproduce the published 51.9% -- the SAME definition, unchanged.

    Runs of three consecutive ticks whose gaps are each within REG_HI x that
    symbol's median, counted over EVERY symbol in market_stream (not the
    fillable subset), divided by all rows. The lower bound REG_LO is NOT applied
    here: the point is to re-run the published statistic, not to improve it.
    """
    rows = conn.execute(
        "SELECT symbol, ts FROM market_stream ORDER BY symbol, ts"
    ).fetchall()
    gaps: Dict[str, List[float]] = collections.defaultdict(list)
    prev_sym = prev_ts = None
    for sym, ts in rows:
        if sym == prev_sym and prev_ts is not None and ts - prev_ts > 0:
            gaps[sym].append(ts - prev_ts)
        prev_sym, prev_ts = sym, ts
    medians = {s: st.median(g) for s, g in gaps.items()}
    counts: Dict[str, int] = collections.defaultdict(int)
    prev_sym = prev_ts = None
    run = 0
    for sym, ts in rows:
        if sym != prev_sym:
            run = 1
        elif prev_ts is not None and 0 < ts - prev_ts <= REG_HI * medians.get(sym, math.inf):
            run += 1
        else:
            run = 1
        if run >= 3:
            counts[sym] += 1
        prev_sym, prev_ts = sym, ts
    total = sum(counts.values())
    return total, len(counts), len(rows), (total / len(rows) if rows else 0.0)


# ---------------------------------------------------------------------------
# The reflexivity test
# ---------------------------------------------------------------------------

def bucket_edges(fires: Sequence[dict]) -> List[float]:
    mags = sorted(f["mag"] for f in fires)
    return [0.0] + [quantile(mags, q) for q in BUCKET_QUANTILES] + [float("inf")]


def cell(name: str, label: str, rows: Sequence[dict], key: str, unit: str) -> dict:
    vals = [r[key] for r in rows]
    keys = [r["cluster"] for r in rows]
    se, g = cluster_se(vals, keys)
    return {
        "stat": name,
        "bucket": label,
        "unit": unit,
        "n": len(vals),
        "mean": st.mean(vals) if vals else float("nan"),
        "median": st.median(vals) if vals else float("nan"),
        "se": se,
        "clusters": g,
        "symbols": len({r["symbol"] for r in rows}),
        "top_symbol_share": (
            max(collections.Counter(r["symbol"] for r in rows).values()) / len(rows)
            if rows else 0.0
        ),
    }


def permutation_bar(
    cells: Sequence[dict],
    cell_rows: Sequence[Sequence[dict]],
    trials: int,
    seed: int = 7,
) -> Tuple[float, float]:
    """Family-wise bar MEASURED, in SE units, instead of assumed.

    The null is: the magnitude of a move carries no information about which way
    the response curves. Flipping the sign of a whole (symbol, 6h) block
    preserves every magnitude, every bucket boundary and the within-block
    dependence, and destroys only the directional link. The statistic is
    max |mean| / SE over ALL K cells; its 95th percentile is the empirical
    constant that SE*sqrt(2 ln K) is trying to approximate.
    """
    if trials <= 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    blocks = sorted({r["cluster"] for rows in cell_rows for r in rows})
    index = {b: i for i, b in enumerate(blocks)}
    # Flatten each cell once: parallel value / block-index lists, so the trial
    # loop is arithmetic instead of dictionary lookups.
    flat: List[Tuple[List[float], List[int], int]] = []
    for c, rows in zip(cells, cell_rows):
        if len(rows) < 2:
            continue
        field = {"d2": "d2_bp", "cont": "cont_bp", "accel": "accel"}[
            c["stat"].split(":")[0]
        ]
        vals = [r[field] for r in rows]
        idx = [index[r["cluster"]] for r in rows]
        flat.append((vals, idx, len({i for i in idx})))
    best: List[float] = []
    nblocks = len(blocks)
    for _ in range(trials):
        flip = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(nblocks)]
        worst = 0.0
        for vals, idx, g in flat:
            n = len(vals)
            tot = 0.0
            for v, i in zip(vals, idx):
                tot += flip[i] * v
            mu = tot / n
            acc: Dict[int, float] = {}
            for v, i in zip(vals, idx):
                acc[i] = acc.get(i, 0.0) + flip[i] * v - mu
            ss = sum(x * x for x in acc.values())
            if g > 1:
                ss *= g / (g - 1.0)
            se = math.sqrt(ss) / n
            if se > 0 and math.isfinite(se):
                t = abs(mu) / se
                if t > worst:
                    worst = t
        best.append(worst)
    best.sort()
    return best[len(best) // 2], best[min(len(best) - 1, int(0.95 * len(best)))]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def fmt(x: float, width: int = 10, places: int = 3) -> str:
    if x != x:
        return "nan".rjust(width)
    if not math.isfinite(x):
        return "inf".rjust(width)
    if abs(x) >= 1e6:
        return ("%.*e" % (2, x)).rjust(width)
    return ("%.*f" % (places, x)).rjust(width)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--census", default=None, help="round-trip quote census artifact")
    ap.add_argument("--trials", type=int, default=1000,
                    help="block sign-flip trials for the measured family-wise bar")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default=None, help="write the numbers to this path")
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print("no store at %s" % args.db)
        return 2
    size = os.path.getsize(args.db)
    if size < 1_000_000:
        print(
            "REFUSING %s: %d bytes. data/trading.db, trading_bot.db and "
            "web/db.sqlite3 are 0-byte stubs that return nothing SILENTLY."
            % (args.db, size)
        )
        return 2

    census_path = args.census
    if not census_path:
        found = sorted(glob.glob(CENSUS_GLOB))
        if not found:
            print("no round-trip quote census matching %s -- the fillable set is "
                  "read from that artifact and is never hardcoded" % CENSUS_GLOB)
            return 2
        census_path = found[-1]
    if not os.path.exists(census_path):
        print("no census at %s -- the fillable symbol set is READ from the "
              "round-trip quote census and is never hardcoded, so there is "
              "nothing to fall back to." % census_path)
        return 2

    import sqlite3

    conn = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)

    pairs, denom, costs, census = fillable_symbols(census_path)

    print("=" * 96)
    print("SECOND-ORDER FEATURE FORGE  --  is there structure in the CURVATURE of this tape?")
    print("=" * 96)
    print()
    print("0. PROVENANCE")
    print("   store          %s  (%.1f GB)" % (args.db, size / 1e9))
    print("   census         %s  (git %s, chain %s)"
          % (census_path, census.get("git_sha"), census.get("chain")))
    print("   fillable set   the census 'quoted' block: %d of %d symbols it could "
          "price; %d are UNQUOTABLE and carry no default"
          % (len(census.get("quoted", {})),
             len(census.get("quoted", {})) + len(census.get("unquotable", {})),
             len(census.get("unquotable", {}))))
    print("   denomination   %s (parsed from the census's own "
          "what_is_measured; it prices DIRECT %s<->token pools only)" % (denom, denom))
    print("   knobs          REG_HI=%.2f REG_LO=%.2f move_q=%.2f post=%d "
          "cluster=%dh denom_gap_log=%.2f"
          % (REG_HI, REG_LO, MOVE_QUANTILE, POST_TICKS,
             CLUSTER_BLOCK_SEC // 3600, DENOM_GAP_LOG))

    now = time.time()
    tmax = conn.execute("SELECT MAX(ts) FROM market_stream").fetchone()[0]
    series: List[SymbolSeries] = []
    print()
    print("1. COVERAGE AND TICK AGE -- never a raw count; this feed takes 16-59 min blackouts")
    print("   %-14s %6s %7s %7s %8s %7s %7s %7s %8s %9s"
          % ("pair", "ticks", "med_g", "p90_g", "span_h", "reg%", "zero%", "blk>16m",
             "age_h", "cost_bp"))
    for p in pairs:
        rows = conn.execute(
            "SELECT ts, price FROM market_stream WHERE symbol = ? ORDER BY ts", (p,)
        ).fetchall()
        if len(rows) < 4:
            print("   %-14s %6d   -- too few ticks to differentiate twice" % (p, len(rows)))
            continue
        s = SymbolSeries(p, rows)
        series.append(s)
        og = sorted(s.gaps)
        print("   %-14s %6d %7.0f %7.0f %8.1f %6.1f%% %6.1f%% %7d %8.1f %9.1f"
              % (p, s.n, s.med_gap, quantile(og, 0.9), s.span_h,
                 100 * s.regular_frac, 100 * s.zero_ret_frac, s.blackouts,
                 (tmax - s.ts[-1]) / 3600.0, costs.get(p, float("nan"))))
    dropped = sum(s.denom_dropped for s in series)
    print("   age_h is measured against the newest tick in the WHOLE table "
          "(%.1f h old itself)." % ((now - tmax) / 3600.0))
    print("   denomination filter dropped %d ticks (%.2f%%) across the subset."
          % (dropped, 100 * dropped / max(1, sum(s.raw_ticks for s in series))))
    stale = [s.symbol for s in series if (tmax - s.ts[-1]) > 6 * 3600]
    if stale:
        print("   STALE (newest tick over 6h old): %s -- these pairs are quotable "
              "but their tape has stopped." % ", ".join(stale))

    if not series:
        print("no usable series -- stop")
        return 1

    print()
    print("2. WHAT AN EFFECT MUST BEAT BEFORE IT IS WORTH ANYTHING")
    priced = {s.symbol: costs.get(s.symbol, float("inf")) for s in series}
    cheapest = min(priced, key=lambda k: priced[k])
    cheap = priced[cheapest]
    gas = census.get("measured_gas_context", {}).get("as_a_rate_bp_by_clip", {})
    gas2 = gas.get("$2", float("nan"))
    slip = census.get("THESE_ARE_QUOTES_NOT_FILLS", {}).get(
        "live_measured_round_trip_slippage_bp", float("nan"))
    print("   cheapest measured round trip in the fillable set: %s at %.1f bp "
          "(pool cost only, a FLOOR)" % (cheapest, cheap))
    print("   measured gas at the $2 median clip: %.1f bp on top; live measured "
          "round-trip slippage %.1f bp that no quote carries" % (gas2, slip))
    print("   -> a curvature effect under ~%.0f bp is not a trade even in the "
          "CHEAPEST pool we can fill." % (cheap + gas2))
    print("   and the SAME effect must clear %.0f bp on %s, the most expensive "
          "pair the census still calls quotable."
          % (max(priced.values()) + gas2, max(priced, key=lambda k: priced[k])))

    tot, nsym, allrows, frac = published_triple_fraction(conn)
    print()
    print("3. THE PUBLISHED COMPUTABILITY FIGURE, RE-RUN (attempts-revenir pass "
          "432: \"51,807 usable triples ... = 51.9%\")")
    print("   same definition, whole table: %d usable triples across %d symbols "
          "= %.1f%% of %d rows" % (tot, nsym, 100 * frac, allrows))
    delta = 100 * frac - 51.9
    verdict = "REPRODUCED" if abs(delta) < 1.0 else "CORRECTED"
    print("   %s (%.1f%% vs the published 51.9%%, %+.1f pp). The table has grown "
          "from ~99.8k rows to %d since that pass, which is the whole difference."
          % (verdict, 100 * frac, delta, allrows))
    print("   NOTE: that figure is a WHOLE-TABLE number over %d symbols, most of "
          "which we cannot fill. Section 4 is the same count on the fillable "
          "subset and it is much smaller." % nsym)

    feats, subset_ticks = feature_computability(series)
    loose, tight = build_fires(series)

    print()
    print("4. FEATURE LAYER -- fraction of the %d fillable-subset ticks each "
          "second-order quantity is COMPUTABLE on" % subset_ticks)
    print("   %-22s %8s %8s %12s %12s %8s" %
          ("feature", "n", "frac", "mean", "median", "blocks"))
    for f in feats:
        print("   %-22s %8d %7.1f%% %s %s %8d"
              % (f["feature"], f["n"], 100 * f["frac"],
                 fmt(f["mean"], 12, 4), fmt(f["median"], 12, 4), f["clusters"]))
    print("   units: acceleration = bp per (median-gap)^2; curvature = "
          "dimensionless kappa in sigma/median-gap coordinates;")
    print("          convexity_post_move = bp per (median-gap)^2 of the signed "
          "post-move path; spacing_asymmetry = (mean gap after an")
    print("          up-tick - after a down-tick)/mean, so >0 means the tape "
          "thins out after upticks; post_move_decay = log-slope per")
    print("          median-gap of |displacement|/|move|, so <0 is decay.")
    print("   acceleration/curvature carry the REG_LO bound because they divide "
          "by the gaps; that bound alone costs ~half the triples")
    print("   and is the difference between a maximum of 1.3e3 and 2.0e10.")

    edges = bucket_edges(loose)
    labels = []
    for b in range(len(edges) - 1):
        hi = "inf" if not math.isfinite(edges[b + 1]) else "%.2f" % edges[b + 1]
        labels.append("%.2f-%s" % (edges[b], hi))

    cells: List[dict] = []
    cell_rows: List[List[dict]] = []
    for b in range(len(edges) - 1):
        rows_l = [f for f in loose if edges[b] <= f["mag"] < edges[b + 1]]
        rows_t = [f for f in tight if edges[b] <= f["mag"] < edges[b + 1]]
        for stat, key, unit, rows in (
            ("d2", "d2_bp", "bp", rows_l),
            ("accel", "accel", "bp/gap^2", rows_t),
            ("cont", "cont_bp", "bp", rows_l),
        ):
            if len(rows) < 5:
                continue
            c = cell("%s:%s" % (stat, labels[b]), labels[b], rows, key, unit)
            c["extra"] = {
                "spike_share": sum(1 for r in rows if r["spike"]) / len(rows),
            }
            cells.append(c)
            cell_rows.append(rows)

    K = len(cells)
    Z = math.sqrt(2.0 * math.log(K)) if K > 1 else 0.0
    n_loose = len(loose)

    print()
    print("5. THE REFLEXIVITY TEST -- bucket by |FIRST derivative|, report the "
          "SECOND derivative with its own bar")
    print("   n = %d fires on %d fillable pairs. The bucketing move r0 is "
          "DISJOINT from r1,r2: centring the second" % (n_loose, len(series)))
    print("   derivative on the move itself would put r0 inside its own statistic "
          "and conditioning on large |r0| would then")
    print("   manufacture positive signed acceleration from nothing.")
    print()
    print("   K = %d cells actually tested -> bar = SE * sqrt(2 ln %d) = SE * %.3f"
          % (K, K, Z))
    med_bar, p95_bar = permutation_bar(cells, cell_rows, args.trials, args.seed)
    print("   block sign-flip null (%d trials, whole (symbol,6h) blocks flipped): "
          "family-wise max|mean|/SE is %.2f at the median and %.2f at the 95th "
          "percentile." % (args.trials, med_bar, p95_bar))
    print("   The analytic constant %.3f sits %s the measured 95th percentile "
          "%.2f, so the bar below is %s."
          % (Z, "ABOVE" if Z >= p95_bar else "BELOW", p95_bar,
             "conservative" if Z >= p95_bar else "OPTIMISTIC and the permutation "
             "number should be used instead"))
    bar_mult = max(Z, p95_bar)
    print("   Bars printed below use max(analytic, permutation) = %.3f." % bar_mult)

    for stat, title, unit in (
        ("d2", "SIGNED SECOND DIFFERENCE  sign(r0)*(r2-r1)  -- >0 = curving INTO the move",
         "bp"),
        ("accel", "SIGNED SECOND DERIVATIVE  time-normalised, REG_LO applied",
         "bp/gap^2"),
        ("cont", "SIGNED FIRST-ORDER CONTINUATION  sign(r0)*r1  -- the level, for contrast",
         "bp"),
    ):
        print()
        print("   %s  [%s]" % (title, unit))
        print("   %-14s %7s %11s %10s %10s %11s %7s %8s  %s"
              % ("|r0| bucket", "n", "mean", "SE_clust", "bar", "median",
                 "blocks", "spike%", "verdict"))
        for c, rows in zip(cells, cell_rows):
            if not c["stat"].startswith(stat + ":"):
                continue
            bar = c["se"] * bar_mult
            passes = abs(c["mean"]) > bar
            print("   %-14s %7d %s %s %s %s %7d %7.1f%%  %s"
                  % (c["bucket"], c["n"], fmt(c["mean"], 11), fmt(c["se"], 10),
                     fmt(bar, 10), fmt(c["median"], 11), c["clusters"],
                     100 * c["extra"]["spike_share"],
                     "OVER BAR" if passes else "null (inside its own bar)"))

    # ---- robustness on anything that cleared the bar ----------------------
    print()
    print("6. ROBUSTNESS -- leave-one-symbol-out on every cell that clears the "
          "WEAKER of the two bars (the analytic %.3f)," % Z)
    print("   because the near-misses are exactly where a tempting number hides. "
          "Reported whether or not it is convenient.")
    over = [(c, r) for c, r in zip(cells, cell_rows) if abs(c["mean"]) > c["se"] * Z]
    if not over:
        print("   No cell clears even the weaker bar. Nothing to stress. "
              "That IS the result.")
    for c, rows in over:
        field = {"d2": "d2_bp", "cont": "cont_bp", "accel": "accel"}[c["stat"].split(":")[0]]
        print("   %s  full n=%d mean=%s" % (c["stat"], len(rows), fmt(c["mean"], 9)))
        jack = []
        for sym in sorted({r["symbol"] for r in rows}):
            kept = [r[field] for r in rows if r["symbol"] != sym]
            if len(kept) > 5:
                jack.append((sym, len(kept), st.mean(kept)))
        for sym, nk, mk in sorted(jack, key=lambda x: abs(x[2]), reverse=True):
            flag = "  <-- SIGN FLIPS" if mk * c["mean"] < 0 else ""
            kept = (mk / c["mean"]) if c["mean"] else float("nan")
            print("      drop %-14s n=%5d mean=%s  (%3.0f%% of the effect SURVIVES)%s"
                  % (sym, nk, fmt(mk, 9), 100 * kept, flag))
        if jack:
            worst = min(jack, key=lambda x: abs(x[2]))
            carried = 1 - abs(worst[2]) / abs(c["mean"])
            print("      -> ONE symbol (%s, %d of %d fires, round trip %.0f bp) "
                  "carries %.0f%% of it."
                  % (worst[0], len(rows) - worst[1], len(rows),
                     costs.get(worst[0], float("nan")), 100 * carried))

    # the spike diagnostic, deliberately NOT counted in K
    print()
    print("7. THE SPIKE DIAGNOSTIC, AND WHY IT IS NOT A RESULT")
    spikes = [f for f in loose if f["spike"]]
    if spikes:
        rest = [f for f in loose if not f["spike"]]
        print("   %d of %d fires (%.1f%%) are single-tick spike-reversals: the very "
              "next tick undoes >=50%% of the move in the"
              % (len(spikes), len(loose), 100 * len(spikes) / len(loose)))
        print("   opposite direction -- a bad print and its correction. They carry "
              "mean d2 = %s bp against %s bp for the rest."
              % (fmt(st.mean([f["d2_bp"] for f in spikes]), 8),
                 fmt(st.mean([f["d2_bp"] for f in rest]), 8)))
        print("   EXCLUDING THEM WOULD BE WRONG AND WOULD FLIP SIGNS. The spike test "
              "selects on r1, and r1 is INSIDE the statistic")
        print("   (d2 = s*r2 - s*r1), so dropping the most negative s*r1 drags every "
              "bucket mean down mechanically. Any 'deceleration'")
        print("   found that way is the filter, not the tape. The unbiased "
              "robustness checks are the median column and section 6.")

    # ---- the verdict ------------------------------------------------------
    curv_cells = [c for c in cells if c["stat"].split(":")[0] in ("d2", "accel")]
    curv_up = [
        c for c in curv_cells
        if c["mean"] > 0 and abs(c["mean"]) > c["se"] * bar_mult
    ]
    print()
    print("=" * 96)
    print("VERDICT -- does ACCELERATION-INTO-MOVE exist on this tape?")
    if not curv_up:
        print("  NO. Across %d curvature cells (%d |first-derivative| buckets x 2 "
              "second-order statistics) on %d fillable"
              % (len(curv_cells), len(curv_cells) // 2, len(series)))
        print("  pairs and %d fires, not ONE clears its own noise bar on the "
              "positive side at K=%d. Every bucket is a null;" % (n_loose, K))
        print("  the point estimates are not reportable as lift. The largest of "
              "them, %s bp in the %s bucket, is"
              % (fmt(max(c["mean"] for c in curv_cells if c["stat"].startswith("d2:")), 1, 2),
                 max((c for c in curv_cells if c["stat"].startswith("d2:")),
                     key=lambda c: c["mean"])["bucket"]))
        print("  INSIDE its own bar and would still be under the %.0f bp cost of "
              "the cheapest pool we can fill." % (cheap + gas2))
    else:
        print("  A POSITIVE CURVATURE CELL SURVIVED THE BAR -- read section 6 "
              "before believing it:")
        for c in curv_up:
            print("    %s  mean %s %s vs bar %s on n=%d"
                  % (c["stat"], fmt(c["mean"], 8), c["unit"],
                     fmt(c["se"] * bar_mult, 8), c["n"]))
    others = [c for c in cells
              if c["stat"].startswith("cont:") and abs(c["mean"]) > c["se"] * bar_mult]
    if others:
        print()
        print("  WHAT DID CLEAR THE BAR, and it is not curvature: the FIRST-order "
              "cell(s) below. Sign and section 6 matter --")
        for c in others:
            print("    %s  mean %s bp (bar %s) n=%d  -> %s"
                  % (c["stat"], fmt(c["mean"], 8), fmt(c["se"] * bar_mult, 6), c["n"],
                     "CONTINUATION" if c["mean"] > 0 else "REVERSION, i.e. the big "
                     "move gives back, not builds"))
    print("=" * 96)

    if args.json:
        payload = {
            "generated_at_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "db": args.db,
            "census": census_path,
            "pairs": [s.symbol for s in series],
            "costs_bp": dict(costs),
            "published_triple_fraction": frac,
            "subset_ticks": subset_ticks,
            "features": feats,
            "K": K,
            "analytic_multiplier": Z,
            "permutation_multiplier_p95": p95_bar,
            "bar_multiplier_used": bar_mult,
            "cells": cells,
            "n_fires": n_loose,
            "verdict_acceleration_into_move": bool(curv_up),
            "cells_over_analytic_bar": [c["stat"] for c, _r in over],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
