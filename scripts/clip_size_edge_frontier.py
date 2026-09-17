"""Is there a CLIP at which some (symbol, horizon) cell becomes payable?

THE QUESTION THIS FILE CLOSES. Pass 456's edge census read cost > move in every
deep cell and concluded "no edge" -- but it read that at the $0.81 median
notional we actually traded, and at that clip most of the cost is a DIVISION
rather than a fee. Round-trip cost splits into a part that scales with the clip
and a part that does not:

    cost_rt(clip) = 2*S + 2*G*1e4/clip        [basis points]

with ``S`` the realised slippage in bp PER LEG (clip-invariant at these sizes)
and ``G`` the gas in DOLLARS per leg off the receipts. The gas term is 51.0 bp
at $0.81 and 2.1 bp at $20, so the clip is a real lever and "cost > move at
$0.81" does not settle "cost > move at $20". This module computes the whole
frontier instead of one point.

THE BARRIER-SKILL LAW, and the reason the answer is arithmetic rather than
empirical. Expectancy is ``(2h - 1)*m - cost``, so break-even needs

    h*(clip) = 0.5 + cost_rt(clip) / (2*m)

Setting ``h*(clip) = h_t`` and solving for the clip gives a closed form, which
is the whole engine of this file:

    clip* = G*1e4 / ( m*(h_t - 0.5) - S )

Read the DENOMINATOR, because it decides everything before any data is loaded:

  * ``h_t <= 0.5`` makes ``m*(h_t - 0.5) <= 0`` and the denominator strictly
    negative for any ``S > 0``. There is NO positive clip. Not "a clip above
    the wallet" -- no solution at all, because ``h*`` is bounded below by

        inf_clip h*(clip) = 0.5 + S/m        (the clip -> infinity limit)

    which exceeds 0.5 whenever slippage is positive. THE CLIP LEVER CAN ONLY
    EVER BUY BACK THE GAS TERM. It cannot move break-even below a coin flip,
    so a target hit rate below 0.5 is unreachable by sizing at any size, on
    any symbol, at any horizon, however volatile.
  * ``h_t > 0.5`` but the denominator still negative means the SLIPPAGE FLOOR
    alone eats the budget: an infinite clip does not pay. Reported UNREACHABLE.
  * denominator positive gives a finite ``clip*``, which then has to be
    FUNDABLE -- the wallet holds ~$23 deployable, so anything above
    ``--max-clip`` (default $20) is reported as a number, not as an edge.

Our realised hit rate is 0.476 over 227 round trips, so the first bullet is the
operative one and this file's headline is a negative. It is computed, not
assumed: the hit rate is measured from the store on every run.

WHY A LARGER CLIP IS NOT FREE, which is where an optimistic bound would lie.
Slippage at a bigger clip is WORSE -- a bigger order walks further into the
book -- and that CANNOT be measured from our fills: all 38 live legs traded
between $0.00 and $3.00, so there is no size variation to regress against.
This file therefore reports TWO frontiers and labels them:

  * OPTIMISTIC (the default). ``S`` is held at its measured value for every
    clip. This is a LOWER BOUND ON COST and therefore an UPPER BOUND ON WHAT
    SIZING CAN BUY. A cell that fails here fails for real; a cell that passes
    here has not been shown to pass.
  * DEPTH-AWARE (``--depth-model``). Adds a constant-product impact term
    ``2*clip/L*1e4`` per leg from ``data/pool_depth_snapshot.json``, where L is
    the deepest pool's USD liquidity. This is a MODEL, not a measurement, and
    it is labelled so everywhere it appears. Its value is that it has the right
    SHAPE: cost becomes U-shaped in the clip, with a minimum at

        clip_opt = sqrt(G*L/2)

    which is $186 for AERO's $33.4M pool but $3.21 for a $10k pool. That single
    line says which symbols the clip lever helps at all -- on a thin pool the
    wallet is already past the optimum and sizing up makes cost WORSE.

THE SAMPLING FLOOR, measured by pass 457 and folded in here. The feed samples a
symbol every 2.6-5.0 minutes, so a 5-minute "horizon" is ONE TO TWO TICKS wide
and measures adjacent-tick spacing rather than a 5-minute move. Every cell
reports its median TICKS PER WINDOW and any cell below ``MIN_TICKS_PER_WINDOW``
is refused a number and marked BELOW SAMPLING FLOOR. Prices are never
forward-filled: interpolating across a 3-minute gap manufactures the very
horizon the feed cannot resolve.

That floor and the cost floor squeeze from OPPOSITE ends. Gas amortises
identically at every horizon while mean|move| grows with horizon, so cost
prefers a LONG hold -- but the book goes negative after 15 minutes and the feed
forbids a short one. The cells that survive cost are the ones the book says
lose.

EVERY OTHER GUARD IS THERE BECAUSE THIS REPO SHIPPED A FAKE EDGE WITHOUT IT:

  * NON-OVERLAPPING windows. Overlapping forward returns share their path and
    understate the standard error by an order of magnitude, after which every
    cell "clears" its bar. Each tick is consumed once per horizon.
  * A NOISE BAR of ``SE * sqrt(2 ln K)`` over the K cells searched, because the
    best of K draws is not a discovery at its own SE. K is printed.
  * SAMPLE FLOORS. ``n`` per cell is printed and cells under
    ``MIN_CELL_N`` are never ranked as findings.
  * CONTAMINATION. A tape row more than 0.5 in log-price off its local median
    is dropped, and each cell reports the share of its total |move| carried by
    its single largest window plus the verdict recomputed without it.
  * COST IS UNPRICED, NOT CHEAP, for a symbol with fewer than four live legs
    of its own. BSTONK's two legs read 622 bp/leg; charging it the 6.75 bp
    deep-pooled rate is exactly how a census invents an edge.
  * TRADEABILITY. A cell on a symbol the feed is not currently streaming, or
    quoted in USDT rather than USDC, is reported as not fundable.

Usage (read-only; opens the store with ``mode=ro``):

    python -X utf8 scripts/clip_size_edge_frontier.py
    python -X utf8 scripts/clip_size_edge_frontier.py --depth-model
    python -X utf8 scripts/clip_size_edge_frontier.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from services.fill_cost import leg_cost_usd  # noqa: E402
from services import outcome_plausibility as plausibility  # noqa: E402

#: Holding horizons in MINUTES.
HORIZONS_MIN: Tuple[int, ...] = (5, 15, 30, 60, 120, 240)

#: Clips in USD for the printed cost curve. 0.81 is the median notional actually
#: traded; 23.18 is the whole deployable stable balance.
CURVE_CLIPS: Tuple[float, ...] = (0.81, 2.0, 5.0, 10.0, 20.0, 23.18)

#: The wallet's deployable stable balance. A clip above this is not fundable
#: today, and a frontier that crosses above it is a number, not an edge.
MAX_FUNDABLE_CLIP = 20.0

#: A tape row this far in log price from the median of its +/-20 neighbours is a
#: foreign row, not a move.
CONTAM_LOG_DEV = 0.5
CONTAM_HALFWIN = 20

#: A "slippage" past this is a decimals artifact. The known case reads exactly
#: -10000.0 bp and, being favourably signed, makes cost look CHEAPER -- so
#: leaving it in is the direction that loses money.
SLIP_ARTIFACT_BPS = 2000.0

#: Cells thinner than this are printed but never ranked as findings.
MIN_CELL_N = 30

#: A symbol needs this much tape before it earns a row at all.
MIN_TAPE_ROWS = 300
MIN_TAPE_SPAN_H = 48.0

#: Below this many live legs of its own, a symbol's slippage is one or two rows
#: and is reported UNPRICED rather than as a small number.
MIN_COST_LEGS = 4

#: A window spanning fewer than this many ticks measures the sampling rate, not
#: the horizon. The feed's mean gap is 2.6-5.0 min per symbol.
MIN_TICKS_PER_WINDOW = 3

#: A cell whose single largest window carries more than this share of its total
#: |move| is contaminated, not volatile.
MAX_TOP1_SHARE = 0.25

#: z for the 95% bound on a symbol's slippage. Cost is a nuisance parameter,
#: not one of the K searched cells, so it pays the ordinary two-sided tax.
COST_Z = 1.96

#: Symbols with four or more live legs each; their pooled rate is the only
#: defensible fallback, and every row using it is labelled ASSUMED.
DEEP_SYMBOLS = ("AERO-USDC", "CBBTC-USDC", "CBETH-USDC")

#: How recently a symbol must have ticked to count as currently streamed.
STREAMED_WINDOW_H = 6.0


# --------------------------------------------------------------------------
# store access
# --------------------------------------------------------------------------

def db_path() -> str:
    return os.environ.get("TRADING_DB_PATH") or os.path.join(
        _ROOT, "storage", "trading_cache.db"
    )


def connect(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise SystemExit(f"store not found: {path}")
    size = os.path.getsize(path)
    if size < 1024:
        raise SystemExit(
            f"{path} is {size} bytes -- that is one of the 0-byte stubs "
            "(data/trading.db, trading_bot.db, web/db.sqlite3), which return "
            "nothing SILENTLY. Point TRADING_DB_PATH at storage/trading_cache.db."
        )
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _mean_se(values: Sequence[float]) -> Tuple[float, float, float, int]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return mean, sd, se, n


# --------------------------------------------------------------------------
# COST -- decomposed into the clip-scaling part and the part that is not
# --------------------------------------------------------------------------

def load_live_legs(conn: sqlite3.Connection) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Every ``live_*`` fill leg, split into (usable, censored-as-artifact).

    Ghost legs are useless for cost: the synthetic-fill defect at
    ``trading/bot.py:9576`` sets ``expected_price == executed_price`` on all of
    them, so their slippage is identically zero by construction.
    """
    usable: List[Dict[str, Any]] = []
    censored: List[Dict[str, Any]] = []
    sql = ("SELECT ts, chain, symbol, expected_price, executed_price, details "
           "FROM trade_fills ORDER BY id")
    for ts, chain, symbol, exp_px, exe_px, raw in conn.execute(sql):
        try:
            det = json.loads(raw or "{}")
        except (TypeError, ValueError):
            continue
        mode = str(det.get("mode") or "")
        if not mode.startswith("live_"):
            continue
        det = dict(det)
        det.setdefault("chain", chain)
        det.setdefault("expected_price", exp_px)
        det.setdefault("executed_price", exe_px)
        split = leg_cost_usd(det)
        row = {
            "ts": ts,
            "symbol": str(symbol),
            "mode": mode,
            "leg": split["leg"],
            "slip_bps": split["realised_slippage_bps"],
            "notional_usd": split["notional_usd"],
            "gas_usd": split["gas_cost_usd"],
            "native_px": det.get("native_token_price_usd", det.get("gas_price_usd")),
            "expected_price": exp_px,
            "executed_price": exe_px,
        }
        slip = row["slip_bps"]
        if slip is None or abs(float(slip)) >= SLIP_ARTIFACT_BPS:
            censored.append(row)
        else:
            usable.append(row)
    return usable, censored


def cost_model(legs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Slippage in bp/leg per symbol, and gas in DOLLARS per leg.

    The split is the whole point. Slippage is a RATE and stays put as the clip
    grows; gas is an ABSOLUTE number of dollars and its bp cost is inversely
    proportional to the clip. Only the second is a lever.
    """
    gas_rows = [r for r in legs if r["gas_usd"] is not None]
    gas = sorted(float(r["gas_usd"]) for r in gas_rows)
    gas_legs_by_side: Dict[str, int] = {}
    for r in gas_rows:
        gas_legs_by_side[str(r["leg"] or "?")] = gas_legs_by_side.get(str(r["leg"] or "?"), 0) + 1

    by_symbol: Dict[str, List[float]] = {}
    for row in legs:
        by_symbol.setdefault(row["symbol"], []).append(float(row["slip_bps"]))

    per_symbol: Dict[str, Dict[str, Any]] = {}
    for sym, vals in sorted(by_symbol.items()):
        mean, sd, se, n = _mean_se(vals)
        top = max(vals, key=abs)
        drop1 = [v for v in vals if v is not top]
        per_symbol[sym] = {
            "n_legs": n,
            "slip_mean_bps": mean,
            "slip_sd_bps": sd,
            "slip_se_bps": se,
            "slip_mean_drop1_bps": statistics.fmean(drop1) if drop1 else float("nan"),
            "measured": n >= MIN_COST_LEGS,
        }

    deep = [float(r["slip_bps"]) for r in legs if r["symbol"] in DEEP_SYMBOLS]
    d_mean, _d_sd, d_se, d_n = _mean_se(deep)

    # Can slippage growth with size be measured at all? OLS of slip on notional.
    pairs = [(float(r["notional_usd"]), float(r["slip_bps"])) for r in legs
             if r["notional_usd"] is not None and float(r["notional_usd"]) > 0]
    slope = slope_se = float("nan")
    if len(pairs) > 2:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx = statistics.fmean(xs)
        my = statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 0:
            slope = sum((x - mx) * (y - my) for x, y in pairs) / sxx
            resid = [y - (my + slope * (x - mx)) for x, y in pairs]
            dof = len(pairs) - 2
            if dof > 0:
                s2 = sum(r * r for r in resid) / dof
                slope_se = math.sqrt(s2 / sxx)

    return {
        "per_symbol": per_symbol,
        "gas_usd_per_leg_median": statistics.median(gas) if gas else float("nan"),
        "gas_usd_per_leg_n": len(gas),
        "gas_usd_p25": gas[len(gas) // 4] if len(gas) >= 4 else float("nan"),
        "gas_usd_p75": gas[(3 * len(gas)) // 4] if len(gas) >= 4 else float("nan"),
        "gas_usd_max": gas[-1] if gas else float("nan"),
        "gas_legs_by_side": gas_legs_by_side,
        "deep_slip_mean_bps": d_mean,
        "deep_slip_se_bps": d_se,
        "deep_slip_n": d_n,
        "notional_min": min((p[0] for p in pairs), default=float("nan")),
        "notional_median": statistics.median([p[0] for p in pairs]) if pairs else float("nan"),
        "notional_max": max((p[0] for p in pairs), default=float("nan")),
        "slip_vs_notional_slope": slope,
        "slip_vs_notional_slope_se": slope_se,
        "slip_vs_notional_n": len(pairs),
    }


def slip_bps_per_leg(
    symbol: str,
    cost: Dict[str, Any],
    *,
    deep_fallback: bool = False,
    upper: bool = False,
) -> Tuple[Optional[float], str]:
    """This symbol's clip-invariant slippage in bp per leg, and its provenance."""
    entry = cost["per_symbol"].get(symbol)
    if entry and entry["measured"]:
        slip = float(entry["slip_mean_bps"])
        if upper and math.isfinite(entry["slip_se_bps"]):
            slip += COST_Z * float(entry["slip_se_bps"])
        return slip, f"fills n={entry['n_legs']}"
    have = entry["n_legs"] if entry else 0
    if not deep_fallback:
        return None, f"UNPRICED (own live legs n={have})"
    slip = float(cost["deep_slip_mean_bps"])
    if upper and math.isfinite(cost["deep_slip_se_bps"]):
        slip += COST_Z * float(cost["deep_slip_se_bps"])
    return slip, f"ASSUMED deep n={cost['deep_slip_n']} (own n={have})"


def cost_rt_bps(
    clip_usd: float,
    slip_bps: float,
    gas_usd_per_leg: float,
    *,
    depth_usd: Optional[float] = None,
    gas_entry_multiple: float = 1.0,
) -> float:
    """Round-trip cost in bp at ``clip_usd``: 2*slip + gas/clip [+ impact].

    ``gas_entry_multiple`` scales the entry leg's gas relative to the measured
    exit gas, because NO live entry leg recorded a usable native price -- see
    the GAS section of the report. 1.0 assumes entry costs what exit costs.

    ``depth_usd`` switches on the constant-product impact term, which is a
    MODEL: a clip of ``x`` against a pool holding ``L`` USD moves price by
    roughly ``2x/L``, so it adds ``2 * 2*clip/L*1e4`` over the round trip. It is
    LINEAR in the clip, which is what makes total cost U-shaped.
    """
    gas_legs = 1.0 + float(gas_entry_multiple)
    total = 2.0 * float(slip_bps) + gas_legs * float(gas_usd_per_leg) / float(clip_usd) * 10_000.0
    if depth_usd is not None and float(depth_usd) > 0:
        total += 2.0 * (2.0 * float(clip_usd) / float(depth_usd) * 10_000.0)
    return total


def h_star(cost_bps: float, mean_abs_move_bps: float) -> float:
    """Break-even hit rate: h* = 0.5 + cost/(2*m)."""
    if not (mean_abs_move_bps > 0):
        return float("inf")
    return 0.5 + float(cost_bps) / (2.0 * float(mean_abs_move_bps))


def crossing_clip(
    h_target: float,
    mean_abs_move_bps: float,
    slip_bps: float,
    gas_usd_per_leg: float,
    *,
    gas_entry_multiple: float = 1.0,
) -> Tuple[Optional[float], str]:
    """The clip at which ``h*`` crosses ``h_target``, or why there is none.

        clip* = gas_legs*G*1e4 / ( 2*m*(h_t - 0.5) - 2*S ) * ... reduced to
        clip* = gas_legs*G*1e4 / ( 2*(m*(h_t - 0.5) - S) )

    The denominator carries the whole result. ``h_t <= 0.5`` makes it negative
    for any positive slippage, and NO POSITIVE CLIP SOLVES IT -- h* is bounded
    below by ``0.5 + S/m``, so sizing can never reach a sub-coin-flip target.

    THIS SOLVER IS IMPACT-FREE BY CONSTRUCTION. Adding the depth term makes the
    equation quadratic in the clip, and it is deliberately left out so that the
    reported frontier is unambiguously the OPTIMISTIC one: a cell that has no
    solution here has none with impact either. ``--depth-model`` therefore moves
    the ``h*`` columns and the ``c_opt`` table, not this number -- and at a $20
    clip against the $7.6M-$33M pools the deep symbols trade in, the modelled
    impact is 0.02-0.11 bp, so the two agree to three decimals anyway. The
    slippage floor, not impact, is what the clip lever cannot buy back.
    """
    if not (mean_abs_move_bps > 0):
        return None, "no move"
    gas_legs = 1.0 + float(gas_entry_multiple)
    denom = 2.0 * (float(mean_abs_move_bps) * (float(h_target) - 0.5) - float(slip_bps))
    if h_target <= 0.5:
        return None, "NO SOLUTION: h_t <= 0.5 is below the h* infimum"
    if denom <= 0:
        return None, "UNREACHABLE: slippage floor alone exceeds the budget"
    return gas_legs * float(gas_usd_per_leg) * 10_000.0 / denom, "solved"


# --------------------------------------------------------------------------
# MOVE -- from the tape, non-overlapping, tick-counted
# --------------------------------------------------------------------------

def load_tape(conn: sqlite3.Connection) -> Dict[str, List[Tuple[float, float]]]:
    series: Dict[str, List[Tuple[float, float]]] = {}
    sql = ("SELECT symbol, ts, price FROM market_stream "
           "WHERE price IS NOT NULL AND price > 0 ORDER BY symbol, ts")
    for sym, ts, price in conn.execute(sql):
        series.setdefault(str(sym), []).append((float(ts), float(price)))
    return series


def streamed_symbols(conn: sqlite3.Connection, window_h: float) -> Dict[str, Dict[str, float]]:
    """Symbols that ticked inside the last ``window_h`` hours, with TODAY's cadence.

    The historical tape is DENSER than the current feed, so a 5-minute cell can
    be measurable in history and unmeasurable going forward. A strategy trades
    at today's cadence, so the sampling floor has to be applied with the CURRENT
    median gap, not the one the archive happens to carry.
    """
    row = conn.execute("SELECT MAX(ts) FROM market_stream").fetchone()
    if not row or row[0] is None:
        return {}
    cutoff = float(row[0]) - window_h * 3600.0
    ticks: Dict[str, List[float]] = {}
    for sym, ts in conn.execute(
        "SELECT symbol, ts FROM market_stream WHERE ts > ? ORDER BY symbol, ts",
        (cutoff,),
    ):
        ticks.setdefault(str(sym), []).append(float(ts))
    out: Dict[str, Dict[str, float]] = {}
    for sym, tss in ticks.items():
        gaps = [tss[i + 1] - tss[i] for i in range(len(tss) - 1)]
        out[sym] = {
            "n": float(len(tss)),
            "median_gap_min": (statistics.median(gaps) / 60.0) if gaps else float("nan"),
            "span_min": (tss[-1] - tss[0]) / 60.0,
        }
    return out


def decontaminate(points: Sequence[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], int]:
    """Drop rows more than ``CONTAM_LOG_DEV`` in log price off their local median."""
    if len(points) < 2 * CONTAM_HALFWIN + 1:
        return list(points), 0
    logs = [math.log(p) for _, p in points]
    keep: List[Tuple[float, float]] = []
    dropped = 0
    for i, pt in enumerate(points):
        lo = max(0, i - CONTAM_HALFWIN)
        hi = min(len(points), i + CONTAM_HALFWIN + 1)
        window = logs[lo:i] + logs[i + 1:hi]
        if not window:
            keep.append(pt)
            continue
        if abs(logs[i] - statistics.median(window)) > CONTAM_LOG_DEV:
            dropped += 1
        else:
            keep.append(pt)
    return keep, dropped


def forward_moves(
    points: Sequence[Tuple[float, float]], horizon_s: float
) -> Tuple[List[float], List[int]]:
    """Absolute forward returns in bp over NON-OVERLAPPING windows, plus ticks.

    Each window starts at the first unconsumed tick and closes on the first tick
    at or after ``horizon_s``, accepted only when its elapsed time landed inside
    ``[H, H + max(120s, H/2)]``. The next window starts AT the closing tick, so
    no two windows share a price path and their standard error is a real one.

    The second return value is the number of TICKS each accepted window spanned.
    A window of 2 ticks is a measurement of the sampling rate, not of the
    horizon, and the caller refuses to give such a cell a number. Nothing is
    interpolated: a gap is skipped, never filled.
    """
    tol = max(120.0, horizon_s / 2.0)
    moves: List[float] = []
    ticks: List[int] = []
    i = 0
    n = len(points)
    while i < n - 1:
        t0, p0 = points[i]
        j = i + 1
        while j < n and points[j][0] - t0 < horizon_s:
            j += 1
        if j >= n:
            break
        if points[j][0] - t0 <= horizon_s + tol:
            moves.append(abs(points[j][1] / p0 - 1.0) * 10_000.0)
            ticks.append(j - i + 1)
        i = j
    return moves, ticks


def build_cells(
    tape: Dict[str, List[Tuple[float, float]]],
    horizons_min: Sequence[int],
) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    for sym, raw in sorted(tape.items()):
        if len(raw) < MIN_TAPE_ROWS:
            continue
        span_h = (raw[-1][0] - raw[0][0]) / 3600.0
        if span_h < MIN_TAPE_SPAN_H:
            continue
        clean, dropped = decontaminate(raw)
        gaps = [clean[i + 1][0] - clean[i][0] for i in range(len(clean) - 1)]
        median_gap_min = (statistics.median(gaps) / 60.0) if gaps else float("nan")
        for h in sorted(horizons_min):
            moves, ticks = forward_moves(clean, h * 60.0)
            if len(moves) < 2:
                continue
            mean, sd, se, n = _mean_se(moves)
            ordered = sorted(moves, reverse=True)
            total = sum(moves) or float("nan")
            drop1 = moves[:]
            drop1.remove(ordered[0])
            d_mean, _d_sd, d_se, _dn = _mean_se(drop1)
            cells.append({
                "symbol": sym,
                "horizon_min": h,
                "n": n,
                "mean_abs_move_bps": mean,
                "sd_bps": sd,
                "se_bps": se,
                "median_ticks_per_window": statistics.median(ticks) if ticks else float("nan"),
                "min_ticks_per_window": min(ticks) if ticks else 0,
                "median_gap_min": median_gap_min,
                "top1_share": ordered[0] / total,
                "top3_share": sum(ordered[:3]) / total,
                "mean_drop1_bps": d_mean,
                "se_drop1_bps": d_se,
                "tape_rows": len(raw),
                "tape_dropped": dropped,
                "span_h": span_h,
            })
    return cells


def noise_bar(se: float, k_cells: int) -> float:
    if k_cells < 2 or not math.isfinite(se):
        return float("nan")
    return se * math.sqrt(2.0 * math.log(k_cells))


# --------------------------------------------------------------------------
# THE BOOK -- the hit rate we have actually achieved
# --------------------------------------------------------------------------

def load_round_trips(conn: sqlite3.Connection) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Paired entry/exit fills, split (countable, refused-by-plausibility)."""
    entries: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}
    exits: List[Tuple[float, str, Dict[str, Any]]] = []
    for ts, symbol, raw in conn.execute(
            "SELECT ts, symbol, details FROM trade_fills ORDER BY id"):
        try:
            det = json.loads(raw or "{}")
        except (TypeError, ValueError):
            continue
        tid = det.get("trade_id")
        if not tid:
            continue
        mode = str(det.get("mode") or "")
        if mode.endswith("_entry"):
            entries[str(tid)] = (float(ts), str(symbol), det)
        elif mode.endswith("_exit"):
            exits.append((float(ts), str(symbol), det))

    rows: List[Dict[str, Any]] = []
    for ts, symbol, det in exits:
        tid = str(det.get("trade_id"))
        if tid not in entries:
            continue
        t0, _s0, edet = entries[tid]
        hold_min = (ts - t0) / 60.0
        if hold_min < 0:
            continue
        entry_px = det.get("entry_price") or edet.get("executed_price")
        exit_px = det.get("executed_price")
        try:
            ret = float(exit_px) / float(entry_px) - 1.0
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        rows.append({
            "symbol": str(symbol),
            "mode": str(det.get("mode") or ""),
            "hold_min": hold_min,
            "entry_price": float(entry_px),
            "exit_price": float(exit_px),
            "ret_bps": ret * 10_000.0,
        })
    keep, drop = plausibility.partition(rows)
    return keep, drop


def book_hit_rate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rets = [float(r["ret_bps"]) for r in rows]
    n = len(rets)
    if n == 0:
        return {"n": 0, "hit": float("nan"), "se": float("nan"),
                "lo": float("nan"), "hi": float("nan")}
    hit = sum(1 for v in rets if v > 0) / n
    se = math.sqrt(hit * (1.0 - hit) / n)
    return {"n": n, "hit": hit, "se": se,
            "lo": hit - COST_Z * se, "hi": hit + COST_Z * se}


# --------------------------------------------------------------------------
# depth
# --------------------------------------------------------------------------

def load_depth() -> Dict[str, Optional[float]]:
    path = os.path.join(_ROOT, "data", "pool_depth_snapshot.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def depth_for(symbol: str, depth: Dict[str, Optional[float]]) -> Optional[float]:
    base = str(symbol).split("-")[0].upper()
    v = depth.get(base)
    return float(v) if isinstance(v, (int, float)) else None


def optimal_clip(gas_usd_per_leg: float, depth_usd: float,
                 *, gas_entry_multiple: float = 1.0) -> float:
    """The clip minimising gas+impact: d/dc [gas_legs*G*1e4/c + 4e4*c/L] = 0.

    Gives ``c_opt = sqrt(gas_legs*G*L/4)``, which is $186 on AERO's $33.4M pool
    and $3.21 on a $10k pool -- i.e. on a thin pool the wallet is ALREADY past
    the optimum and sizing up makes cost worse, not better.
    """
    gas_legs = 1.0 + float(gas_entry_multiple)
    return math.sqrt(gas_legs * float(gas_usd_per_leg) * float(depth_usd) / 4.0)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def fmt(x: Optional[float], width: int = 8, prec: int = 1) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "n/a".rjust(width)
    return f"{float(x):{width}.{prec}f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=db_path())
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS_MIN))
    ap.add_argument("--max-clip", type=float, default=MAX_FUNDABLE_CLIP,
                    help="largest fundable clip in USD (wallet holds ~$23 stable)")
    ap.add_argument("--depth-model", action="store_true",
                    help="add the constant-product impact term (a MODEL, labelled)")
    ap.add_argument("--deep-fallback", action="store_true",
                    help="price symbols with no live legs at the pooled deep rate, "
                         "labelled ASSUMED")
    ap.add_argument("--gas-entry-multiple", type=float, default=1.0,
                    help="entry-leg gas as a multiple of measured exit gas "
                         "(no live entry leg recorded a usable native price)")
    ap.add_argument("--top", type=int, default=25, help="rows in the ranked table")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    horizons = tuple(int(h) for h in str(args.horizons).split(",") if h.strip())

    conn = connect(args.db)
    print(f"store: {args.db} ({os.path.getsize(args.db) / 1e9:.1f} GB)")

    # ---- COST ------------------------------------------------------------
    legs, censored = load_live_legs(conn)
    cost = cost_model(legs)
    gas_g = float(cost["gas_usd_per_leg_median"])
    print()
    print("=" * 100)
    print("COST, DECOMPOSED -- the only part the clip can buy back is the gas")
    print("=" * 100)
    print(f"live legs usable {len(legs)}, censored as decimals artifacts {len(censored)}")
    for r in censored:
        print(f"  CENSORED {r['symbol']:<14} {r['mode']:<12} "
              f"expected {r['expected_price']:.6g} executed {r['executed_price']:.6g} "
              f"-> {r['slip_bps']:.1f} bp")
    print()
    print(f"GAS  ${gas_g:.6f}/leg median over {cost['gas_usd_per_leg_n']} priced legs "
          f"(p25 ${cost['gas_usd_p25']:.6f}, p75 ${cost['gas_usd_p75']:.6f}, "
          f"max ${cost['gas_usd_max']:.6f})")
    print(f"     priced legs by side: {cost['gas_legs_by_side']}")
    if not cost["gas_legs_by_side"].get("buy"):
        print("     *** NO live ENTRY leg carries a usable native price, so the BUY leg's")
        print("         gas is UNMEASURED and round-trip gas is 2x the EXIT leg by")
        print(f"         assumption (--gas-entry-multiple={args.gas_entry_multiple:g}). If the")
        print("         entry also pays an ERC-20 approve, real round-trip gas is HIGHER")
        print("         and every frontier below is correspondingly optimistic.")
    print()
    print("     gas is DOLLARS, so its bp cost is a pure division by the clip:")
    print("       " + "  ".join(
        f"${c:g}->{ (1.0+args.gas_entry_multiple) * gas_g / c * 10_000.0:.1f}bp"
        for c in CURVE_CLIPS))
    print()
    print(f"SLIPPAGE (bp per leg, clip-invariant at these sizes):")
    print(f"{'symbol':<14}{'legs':>6}{'bp/leg':>10}{'SE':>8}{'95% up':>9}"
          f"{'RT bp':>9}{'drop-top':>10}  status")
    for sym, s in sorted(cost["per_symbol"].items(), key=lambda kv: -kv[1]["n_legs"]):
        up = (s["slip_mean_bps"] + COST_Z * s["slip_se_bps"]
              if math.isfinite(s["slip_se_bps"]) else float("nan"))
        print(f"{sym:<14}{s['n_legs']:>6}{fmt(s['slip_mean_bps'],10,2)}"
              f"{fmt(s['slip_se_bps'],8,2)}{fmt(up,9,2)}"
              f"{fmt(2*s['slip_mean_bps'],9,2)}{fmt(s['slip_mean_drop1_bps'],10,2)}"
              f"  {'MEASURED' if s['measured'] else 'UNPRICED (shown, not used)'}")
    print()
    print(f"deep pooled ({'/'.join(DEEP_SYMBOLS)}) {cost['deep_slip_mean_bps']:.3f} bp/leg "
          f"SE {cost['deep_slip_se_bps']:.3f} n={cost['deep_slip_n']} "
          f"-> RT {2*cost['deep_slip_mean_bps']:.2f} bp")

    print()
    print("CAN SLIPPAGE GROWTH WITH SIZE BE MEASURED? Notionals actually traded span "
          f"${cost['notional_min']:.2f}-${cost['notional_max']:.2f} "
          f"(median ${cost['notional_median']:.2f}, n={cost['slip_vs_notional_n']}).")
    slope = cost["slip_vs_notional_slope"]
    sse = cost["slip_vs_notional_slope_se"]
    if math.isfinite(slope) and math.isfinite(sse):
        lo, hi = slope - COST_Z * sse, slope + COST_Z * sse
        print(f"  OLS slip ~ notional: {slope:+.2f} bp per $1  SE {sse:.2f}  "
              f"95% CI [{lo:+.2f}, {hi:+.2f}] bp/$")
        spans = lo <= 0.0 <= hi
        print(f"  The CI {'SPANS' if spans else 'EXCLUDES'} zero, so this fill set "
              f"{'CANNOT' if spans else 'can'} bound how fast slippage grows with size.")
        print(f"  Reaching ${args.max_clip:g} is a "
              f"{args.max_clip / max(cost['notional_max'], 1e-9):.1f}x extrapolation beyond the "
              "largest clip ever filled. NO number is guessed for it: the default")
        print("  frontier holds slippage CONSTANT, which is an OPTIMISTIC BOUND.")

    # ---- THE BOOK --------------------------------------------------------
    keep, refused = load_round_trips(conn)
    book = book_hit_rate(keep)
    live_rows = [r for r in keep if r["mode"].startswith("live")]
    live_book = book_hit_rate(live_rows)
    print()
    print("=" * 100)
    print("THE HIT RATE WE HAVE ACTUALLY ACHIEVED -- the target h_t, measured not assumed")
    print("=" * 100)
    print(f"paired round trips {len(keep)} countable, {len(refused)} refused by "
          "services.outcome_plausibility")
    print(f"REALISED  h = {book['hit']:.4f} +/- {book['se']:.4f} over n={book['n']} "
          f"  95% CI [{book['lo']:.4f}, {book['hi']:.4f}]")
    print(f"  of which live-only h = {live_book['hit']:.4f} over n={live_book['n']}")
    print(f"  ghost rows {sum(1 for r in keep if r['mode'].startswith('ghost'))}, "
          f"live rows {len(live_rows)}")

    targets = [
        (book["hit"], "REALISED point estimate"),
        (book["hi"], "REALISED 95% UPPER bound -- the most optimistic h the book permits"),
        (1.0, "PERFECT ORACLE (h*<1 reference only)"),
    ]
    print()
    print("h* = 0.5 + cost/(2m) is BOUNDED BELOW BY 0.5 for any positive cost, so:")
    for h_t, label in targets:
        verdict = ("REACHABLE in principle" if h_t > 0.5 else
                   "*** UNREACHABLE AT ANY CLIP, ANY SYMBOL, ANY HORIZON ***")
        print(f"  h_t = {h_t:.4f}  ({label})")
        print(f"      h_t - 0.5 = {h_t - 0.5:+.4f}  ->  {verdict}")

    # ---- MOVES -----------------------------------------------------------
    tape = load_tape(conn)
    cells = build_cells(tape, horizons)
    k = len(cells)
    tax = math.sqrt(2.0 * math.log(k)) if k > 1 else float("nan")
    streamed = streamed_symbols(conn, STREAMED_WINDOW_H)
    depth = load_depth()
    print()
    print("=" * 100)
    print("MOVE -- market_stream, NON-OVERLAPPING windows, tick-counted, contam-guarded")
    print("=" * 100)
    print(f"K = {k} cells searched -> noise bar = SE * sqrt(2 ln K) = {tax:.3f} SE")
    print(f"currently streamed symbols (ticked in the last {STREAMED_WINDOW_H:g}h): "
          f"{len(streamed)}")

    floor_by_h: Dict[int, List[int]] = {}
    for c in cells:
        floor_by_h.setdefault(c["horizon_min"], []).append(
            1 if c["median_ticks_per_window"] < MIN_TICKS_PER_WINDOW else 0)
    print()
    print(f"SAMPLING FLOOR: a window under {MIN_TICKS_PER_WINDOW} ticks measures the feed's "
          "cadence, not the horizon.")
    print("The ARCHIVE is denser than the CURRENT feed, so both have to be checked: a cell can")
    print("be measurable in history and unmeasurable going forward. Ticks below are per window.")
    busy = sorted(streamed.items(), key=lambda kv: -kv[1]["n"])[:10]
    gaps_med = [v["median_gap_min"] for _s, v in busy if math.isfinite(v["median_gap_min"])]
    # MEAN gap = span/(n-1) per symbol. It exceeds the median because the feed
    # takes blackouts, and it is the figure the cadence audit reported.
    gaps_mean = [v["span_min"] / max(v["n"] - 1.0, 1.0) for _s, v in busy]
    med_gap_now = statistics.median(gaps_med) if gaps_med else float("nan")
    mean_gap_now = statistics.median(gaps_mean) if gaps_mean else float("nan")
    print(f"  current cadence, {len(busy)} busiest streamed symbols: median gap "
          f"{med_gap_now:.1f} min, mean gap {mean_gap_now:.1f} min")
    print("  (mean > median because the feed takes blackouts; both are reported because "
          "the 5m")
    print("   column's verdict turns on which one is used, and it lands AT the floor either way)")
    print(f"{'horizon':>9}{'cells':>8}{'archive<floor':>15}{'ticks@med':>11}{'ticks@mean':>12}"
          f"  verdict at TODAY's cadence")
    for h in sorted(floor_by_h):
        v = floor_by_h[h]
        t_med = (h / med_gap_now + 1.0) if math.isfinite(med_gap_now) else float("nan")
        t_mean = (h / mean_gap_now + 1.0) if math.isfinite(mean_gap_now) else float("nan")
        worst = min(t_med, t_mean)
        note = ("MEASURES THE SAMPLING RATE, not the horizon"
                if worst < MIN_TICKS_PER_WINDOW else
                "AT THE FLOOR -- marginal" if worst < 4 else
                "thin but usable" if worst < 6 else "fine")
        print(f"{h:>8}m{len(v):>8}{sum(v):>15}{t_med:>11.1f}{t_mean:>12.1f}  {note}")
    print()
    print("THE FLOOR AND THE COST FLOOR SQUEEZE FROM OPPOSITE ENDS, and they do NOT collide:")
    print("gas amortises identically at every horizon while mean|move| GROWS with horizon, so")
    print("cost prefers a LONG hold, while the feed only threatens SHORT ones. The cheapest")
    print("cells by h* are therefore the 120-240m cells -- which is where the book says we lose.")

    # ---- THE FRONTIER ----------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for cell in cells:
        sym = cell["symbol"]
        s_bps, src = slip_bps_per_leg(sym, cost, deep_fallback=args.deep_fallback)
        s_up, _ = slip_bps_per_leg(sym, cost, deep_fallback=args.deep_fallback, upper=True)
        bar = noise_bar(cell["se_bps"], k)
        m_cons = cell["mean_abs_move_bps"] - bar   # move, after the search tax
        d_usd = depth_for(sym, depth) if args.depth_model else None

        rec: Dict[str, Any] = {
            **cell,
            "slip_bps_per_leg": s_bps,
            "slip_bps_per_leg_upper": s_up,
            "cost_source": src,
            "bar_bps": bar,
            "mean_minus_bar_bps": m_cons,
            "depth_usd": depth_for(sym, depth),
            "streamed": sym in streamed,
            "quote": sym.split("-")[-1] if "-" in sym else "",
        }
        # TODAY's cadence, which is what a strategy would actually trade against.
        live_gap = (streamed.get(sym, {}) or {}).get("median_gap_min", float("nan"))
        rec["live_median_gap_min"] = live_gap
        rec["live_ticks_at_horizon"] = (
            cell["horizon_min"] / live_gap + 1.0 if live_gap and math.isfinite(live_gap)
            and live_gap > 0 else float("nan"))
        # cost curve and h* curve
        for c in CURVE_CLIPS:
            if s_bps is None:
                rec[f"cost_{c}"] = None
                rec[f"hstar_{c}"] = None
                continue
            cb = cost_rt_bps(c, s_bps, gas_g, depth_usd=d_usd,
                             gas_entry_multiple=args.gas_entry_multiple)
            rec[f"cost_{c}"] = cb
            rec[f"hstar_{c}"] = h_star(cb, cell["mean_abs_move_bps"])
        # h* infimum: the clip -> infinity limit
        rec["hstar_infimum"] = (h_star(2.0 * s_bps, cell["mean_abs_move_bps"])
                                if s_bps is not None else None)
        rec["hstar_infimum_cons"] = (h_star(2.0 * s_bps, m_cons)
                                     if s_bps is not None and m_cons > 0 else None)
        # crossing clips
        for h_t, _label in targets:
            key = f"{h_t:.4f}"
            if s_bps is None:
                rec[f"clip_{key}"] = None
                rec[f"clip_{key}_why"] = "UNPRICED"
                continue
            cl, why = crossing_clip(h_t, cell["mean_abs_move_bps"], s_bps, gas_g,
                                    gas_entry_multiple=args.gas_entry_multiple)
            rec[f"clip_{key}"] = cl
            rec[f"clip_{key}_why"] = why
            cl_c, why_c = crossing_clip(h_t, m_cons, s_up if s_up is not None else s_bps,
                                        gas_g, gas_entry_multiple=args.gas_entry_multiple)
            rec[f"clip_{key}_cons"] = cl_c
            rec[f"clip_{key}_cons_why"] = why_c

        # verdict
        reasons: List[str] = []
        if s_bps is None:
            reasons.append("UNPRICED")
        if cell["median_ticks_per_window"] < MIN_TICKS_PER_WINDOW:
            reasons.append(
                f"BELOW SAMPLING FLOOR, archive ({cell['median_ticks_per_window']:.0f} ticks)")
        lt = rec["live_ticks_at_horizon"]
        if math.isfinite(lt) and lt < MIN_TICKS_PER_WINDOW:
            reasons.append(f"BELOW SAMPLING FLOOR, TODAY ({lt:.1f} ticks)")
        if cell["top1_share"] > MAX_TOP1_SHARE:
            reasons.append(f"CONTAMINATED top1={cell['top1_share']:.2f}")
        if cell["n"] < MIN_CELL_N:
            reasons.append(f"thin n={cell['n']}")
        if not rec["streamed"]:
            reasons.append("NOT STREAMED")
        if rec["quote"] != "USDC":
            reasons.append(f"quote {rec['quote']}")
        rec["disqualifiers"] = reasons
        rec["qualified"] = not reasons
        rows.append(rec)

    print()
    print("=" * 100)
    print(f"THE FRONTIER -- clip at which h* crosses each target "
          f"({'DEPTH-AWARE MODEL' if args.depth_model else 'OPTIMISTIC: slippage held constant'})")
    print("=" * 100)
    for h_t, label in targets:
        key = f"{h_t:.4f}"
        solved = [r for r in rows if r.get(f"clip_{key}") is not None]
        fundable = [r for r in solved if r[f"clip_{key}"] <= args.max_clip]
        qual = [r for r in fundable if r["qualified"]]
        print()
        print(f"--- h_t = {h_t:.4f}  ({label})")
        whys: Dict[str, int] = {}
        for r in rows:
            whys[str(r.get(f"clip_{key}_why"))] = whys.get(str(r.get(f"clip_{key}_why")), 0) + 1
        for w, c in sorted(whys.items(), key=lambda kv: -kv[1]):
            print(f"      {c:>4} of {len(rows)} cells: {w}")
        print(f"      solved with a finite clip: {len(solved)};  "
              f"of those fundable (<= ${args.max_clip:g}): {len(fundable)};  "
              f"of those passing every guard: {len(qual)}")

    # the per-cell table, ranked by the honest target
    key_hi = f"{book['hi']:.4f}"
    print()
    print("-" * 132)
    print("PER-CELL TABLE, ranked by the clip that reaches the book's 95% UPPER hit rate "
          f"({book['hi']:.4f})")
    print("-" * 132)
    print(f"{'symbol':<16}{'hor':>5}{'n':>5}{'tick':>5}{'mean|mv|':>10}{'bar':>8}"
          f"{'slip/leg':>9}{'h*@0.81':>9}{'h*@20':>8}{'h*inf':>8}"
          f"{'clip*':>9}{'cons':>9}  flags")
    ranked = sorted(
        rows,
        key=lambda r: (r.get(f"clip_{key_hi}") is None,
                       r.get(f"clip_{key_hi}") if r.get(f"clip_{key_hi}") is not None else 0.0),
    )
    def _cons_cell(rec: Dict[str, Any]) -> str:
        """The conservative crossing clip, or a SHORT reason it does not exist."""
        val = rec.get(f"clip_{key_hi}_cons")
        if val is not None:
            return f"{val:9.2f}"
        why = str(rec.get(f"clip_{key_hi}_cons_why") or "")
        if why.startswith("NO SOLUTION"):
            return "  NO-SOLN"
        if why.startswith("UNREACHABLE"):
            return "   UNRCH"
        if rec["slip_bps_per_leg"] is None:
            return " UNPRICED"
        return "      n/a"

    for r in ranked[: args.top]:
        cl = r.get(f"clip_{key_hi}")
        flags = ",".join(r["disqualifiers"]) if r["disqualifiers"] else "QUALIFIED"
        print(f"{r['symbol']:<16}{r['horizon_min']:>4}m{r['n']:>5}"
              f"{fmt(r['median_ticks_per_window'],5,0)}"
              f"{fmt(r['mean_abs_move_bps'],10,1)}{fmt(r['bar_bps'],8,1)}"
              f"{fmt(r['slip_bps_per_leg'],9,2)}"
              f"{fmt(r.get('hstar_0.81'),9,3)}{fmt(r.get('hstar_20.0'),8,3)}"
              f"{fmt(r['hstar_infimum'],8,3)}"
              f"{fmt(cl,9,2)}{_cons_cell(r)}  {flags}")
    print()
    print("clip* is the clip reaching the target at the CENTRAL cost and the RAW mean|move|.")
    print("cons is the same after the noise bar comes off the move AND the 95% upper bound")
    print("goes on the slippage -- the number that has to hold for the cell to be real.")
    print("  UNRCH   = no clip works: the slippage floor alone exceeds the cost budget.")
    print("  NO-SOLN = the target is below h*'s 0.5 infimum; sizing cannot reach it.")
    print("A cell with a finite clip* and UNRCH in cons is CARRIED BY ITS ERROR BARS, not an edge.")

    # ---- concentration on whatever looks best ----------------------------
    print()
    print("=" * 100)
    print("KILL THE BEST CELL -- concentration, halves, and what one row is worth")
    print("=" * 100)
    best = [r for r in ranked if r.get(f"clip_{key_hi}") is not None][:5]
    if not best:
        print("no cell reaches the target at any clip, so there is nothing to censor.")
    for r in best:
        cl = r.get(f"clip_{key_hi}")
        drop_h = (h_star(2.0 * r["slip_bps_per_leg"], r["mean_drop1_bps"])
                  if r["slip_bps_per_leg"] is not None and r["mean_drop1_bps"] > 0 else float("nan"))
        print(f"\n{r['symbol']} @ {r['horizon_min']}m   n={r['n']}  "
              f"median ticks/window={r['median_ticks_per_window']:.0f}  "
              f"clip*=${cl:.2f}")
        print(f"   mean|move| {r['mean_abs_move_bps']:.1f} bp, noise bar {r['bar_bps']:.1f} bp "
              f"-> after the tax {r['mean_minus_bar_bps']:.1f} bp")
        print(f"   top1 share {r['top1_share']:.2f}, top3 {r['top3_share']:.2f}; "
              f"drop the top window -> mean {r['mean_drop1_bps']:.1f} bp, "
              f"h*inf {drop_h:.3f}")
        print(f"   cost source: {r['cost_source']};  depth "
              f"{('$%.0f' % r['depth_usd']) if r['depth_usd'] else 'UNRESOLVED'};  "
              f"streamed: {r['streamed']};  flags: "
              f"{','.join(r['disqualifiers']) or 'none'}")

    # ---- the depth-aware optimum ----------------------------------------
    print()
    print("=" * 100)
    print("WHERE THE CLIP LEVER HELPS AT ALL -- c_opt = sqrt(gas_legs*G*L/4)  [MODEL]")
    print("=" * 100)
    print("Below c_opt cost falls as the clip grows (gas amortises); above it cost RISES")
    print("(impact dominates). A pool small enough puts the wallet PAST the optimum.")
    print("NOTE the magnitudes: modelled impact at a $20 clip is only "
          f"{2*(2*20.0/33408810*1e4):.3f} bp on AERO's")
    print(f"$33.4M pool and {2*(2*20.0/7604696*1e4):.3f} bp on CBBTC's $7.6M pool. So for the "
          "three symbols we can")
    print("price, holding slippage CONSTANT to $20 is not merely optimistic -- it is nearly")
    print("exact, because their measured slippage is AMM FEE plus spread, which does not")
    print("scale with a $20 clip. The optimism in this frontier is concentrated in the THIN")
    print("names, and every one of those is UNPRICED anyway.")
    print(f"{'symbol':<14}{'depth USD':>14}{'c_opt':>10}   verdict at a $20 clip")
    seen: set = set()
    for r in sorted(rows, key=lambda x: -(x["depth_usd"] or 0.0)):
        base = r["symbol"].split("-")[0]
        if base in seen or not r["depth_usd"]:
            continue
        seen.add(base)
        co = optimal_clip(gas_g, r["depth_usd"], gas_entry_multiple=args.gas_entry_multiple)
        note = ("clip lever still helps at $20" if co >= args.max_clip
                else f"PAST the optimum -- $20 costs MORE than ${co:.2f}")
        print(f"{r['symbol']:<14}{r['depth_usd']:>14.0f}{co:>10.2f}   {note}")
        if len(seen) >= 12:
            break

    # ---- headline --------------------------------------------------------
    key_pt = f"{book['hit']:.4f}"
    pt_solved = [r for r in rows if r.get(f"clip_{key_pt}") is not None]
    hi_qual = [r for r in rows if r.get(f"clip_{key_hi}") is not None
               and r[f"clip_{key_hi}"] <= args.max_clip and r["qualified"]]
    print()
    print("=" * 100)
    print("HEADLINE")
    print("=" * 100)
    if pt_solved:
        r = min(pt_solved, key=lambda x: x[f"clip_{key_pt}"])
        print(f"A fundable cell reaches the REALISED hit rate: {r['symbol']}/"
              f"{r['horizon_min']}m at ${r[f'clip_{key_pt}']:.2f} (n={r['n']}).")
    else:
        print(f"NO fundable clip makes any cell payable at our realised hit rate "
              f"({book['hit']:.4f}), and the cost")
        print(f"route is closed by ARITHMETIC, not by the wallet: h* = 0.5 + cost/(2m) is "
              "bounded below")
        print(f"by 0.5 for any positive cost, our realised h is {book['hit']:.4f} < 0.5, and the "
              "clip lever can")
        print(f"only ever buy back the GAS term "
              f"({(1.0+args.gas_entry_multiple)*gas_g/0.81*10_000.0:.1f} bp at $0.81 -> "
              f"{(1.0+args.gas_entry_multiple)*gas_g/args.max_clip*10_000.0:.1f} bp at "
              f"${args.max_clip:g}); it")
        print(f"cannot move break-even below a coin flip. All {len(rows)} cells return "
              "NO SOLUTION at any clip.")
        print()
        if hi_qual:
            r = min(hi_qual, key=lambda x: x[f"clip_{key_hi}"])
            cons = r.get(f"clip_{key_hi}_cons")
            print(f"Against the most optimistic h the book permits ({book['hi']:.4f}, its 95% "
                  f"upper bound -- an h")
            print("we have NOT demonstrated, only failed to exclude), the cheapest "
                  "fully-qualified cell is")
            print(f"{r['symbol']}/{r['horizon_min']}m at ${r[f'clip_{key_hi}']:.2f} "
                  f"(n={r['n']}, bar {r['bar_bps']:.1f} bp, top1 {r['top1_share']:.2f}).")
            if cons is None:
                budget = 2.0 * r["mean_minus_bar_bps"] * (book["hi"] - 0.5)
                floor = 2.0 * float(r["slip_bps_per_leg_upper"])
                print()
                print("AND IT DIES THE MOMENT ITS ERROR BARS ARE APPLIED. After the noise bar "
                      "comes off the")
                print(f"move ({r['mean_abs_move_bps']:.1f} -> {r['mean_minus_bar_bps']:.1f} bp) "
                      f"and the 95% upper bound goes on the slippage "
                      f"({r['slip_bps_per_leg']:.2f} -> "
                      f"{float(r['slip_bps_per_leg_upper']):.2f} bp/leg),")
                print(f"the whole cost budget is {budget:.2f} bp while slippage ALONE is "
                      f"{floor:.2f} bp round trip -- over by "
                      f"{floor / budget if budget > 0 else float('inf'):.1f}x, at ANY clip.")
                print("So there is no fundable cell at the optimistic target either. NOTHING "
                      "here is an edge.")
        else:
            print(f"Against the most optimistic h the book permits ({book['hi']:.4f}, its 95% "
                  "upper bound), NOT ONE")
            print("cell clears every guard at a fundable clip either.")

    if args.json:
        payload = {
            "db": args.db,
            "k_cells": k,
            "noise_tax_sqrt_2lnK": tax,
            "max_fundable_clip": args.max_clip,
            "depth_model": bool(args.depth_model),
            "gas_entry_multiple": args.gas_entry_multiple,
            "cost": cost,
            "book": book,
            "book_live_only": live_book,
            "targets": [{"h_t": t, "label": l} for t, l in targets],
            "cells": rows,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
