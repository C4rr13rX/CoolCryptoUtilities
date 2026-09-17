"""Per-(symbol, holding-horizon) edge census: is any cell's move bigger than its cost?

THE QUESTION. A directional trade pays only when the move it catches is larger
than the round trip it must pay for. With ``h`` the fraction of calls that go the
right way and ``m`` the mean absolute move over the holding horizon, expectancy
is ``(2h - 1) * m - cost``, so break-even needs

    h* = 0.5 + cost / (2 * m)

and ``h* >= 1.0`` means A PERFECT ORACLE STILL LOSES: no signal, however good,
can pay. Because ``h* < 1`` reduces to ``cost < m``, the whole census is one
comparison per cell -- mean|move| against measured round-trip cost -- and the
only hard parts are (a) measuring both honestly and (b) not being fooled by the
cell that happens to look best out of dozens.

WHERE EACH SIDE COMES FROM, and neither is a model.

  COST comes from ``trade_fills`` rows whose ``details.mode`` starts with
  ``live_`` -- 38 legs of real settled execution. Ghost legs are useless for
  cost: all 1392 of them carry ``expected_price == executed_price`` (the
  synthetic-fill defect at ``trading/bot.py:9576``), so their slippage is
  identically zero by construction. Realised slippage is the WHOLE economic
  cost of fee plus spread plus impact -- see ``services/fill_cost`` -- and gas
  is read off receipts. This module calls ``services.fill_cost`` rather than
  redoing that arithmetic, so a census and the production gate cannot disagree.

  THERE IS NO POOLED-OVER-EVERYTHING FALLBACK, and that is deliberate. The mean
  slippage over all 37 usable legs is 40.1 bp/leg and TWO BSTONK legs carry 84%
  of that sum; drop them and it is 6.9. A symbol with fewer than four of its own
  live legs is therefore reported UNPRICED, not quietly charged a number two
  rows invented. ``--deep-fallback`` will price such a symbol at the AERO/CBBTC/
  CBETH pooled rate, and every row so priced is labelled ASSUMED.

  COST IS ALSO AN ESTIMATE WITH A SPREAD, so every cell is crossed twice: once
  at the central cost and once at the 95% UPPER bound of that symbol's
  slippage. AERO's 16 legs give 4.5 +/- 7.0 bp/leg -- a CI that spans free
  execution to 36 bp round trip -- and a cell that only pays at the cheap end
  of that CI has not been shown to pay.

  Note which cost module is NOT being quoted here. ``services.round_trip_cost``
  is the gate's measured book scalar and ``services.roundtrip_cost`` (one
  underscore apart) is the entry path's notional-only curve. Both are read for
  COMPARISON at the end of the report; neither supplies a number this census
  decides on.

  MOVE comes from ``market_stream``, the price tape, because a holding horizon
  needs a forward return at that horizon and 237 closed outcomes cannot fill a
  symbol-by-horizon grid. Three guards, each closing a trap this repo has
  already fallen into:

    * NON-OVERLAPPING windows. Overlapping forward returns at a 60-minute
      horizon on a 30-second tape share 119/120 of their path, so a naive
      standard error over them is understated by an order of magnitude and
      every cell "clears" its bar. Each tick is consumed once per horizon.
    * A LOCAL-MEDIAN contamination guard: a row more than 0.5 in log-price off
      the median of its own +/-20 neighbours is dropped. One foreign row can
      otherwise saturate a whole window.
    * An ELAPSED TOLERANCE. Ticks are irregular (median gap 4-110s by symbol,
      p90 gap 3-10 minutes), so a window is only counted when its closing tick
      landed inside [H, H + max(120s, H/2)]. Without this, a "15-minute" cell
      on a sparse symbol silently measures 40 minutes.

THE MULTIPLE-COMPARISONS TAX, which is the point of the whole exercise. This
census searches K cells and reports the best one, so the bar a cell must clear
is not its own standard error. Over K roughly-independent draws the largest
deviation grows like ``sd * sqrt(2 ln K)``, so the bar used here is

    bar = SE(mean|move|) * sqrt(2 * ln K)

with K the number of reported cells, printed beside the bar. A cell that beats
cost without paying this tax is not a finding, and this file will not call it
one.

CENSORING, because in this repo it has decided the sign six times. Every cell
reports the share of its total |move| carried by its single largest row and by
its top three, plus the verdict RECOMPUTED with the top row dropped. A cell
whose verdict flips when one row leaves is reported as CARRIED, not as an edge.
A cell whose top row is more than a quarter of its total is reported
CONTAMINATED and is not a finding at all -- COMP-USDC reaches 64% on one row
that the log-price guard missed by a hair.

TWO FURTHER ARTIFACT CHECKS, because a good number here is an artifact until an
independent read agrees:

  * VARIANCE RATIO ``mean|r_H| / (mean|r_5m| * sqrt(H/5))``. Under a random walk
    this is 1. Well BELOW 1 means the 5-minute cell is inflated by a noise floor
    -- bid/ask bounce or two venues alternating -- which a directional trade
    cannot capture because it crosses the spread in both directions. Well ABOVE
    1 with a concentrated top row means a jump artifact.
  * SPLIT-HALF. mean|move| recomputed on the first and second half of the
    symbol's tape. A cell whose halves disagree by more than its own bar is a
    regime reading, not a property of the symbol.

WHAT A PASSING CELL DOES AND DOES NOT MEAN. ``h* < 1`` says a PERFECT oracle
would pay. It is necessary and nowhere near sufficient: the cell still has to be
predicted at ``h*``. The BOOK section prints the hit rate we have actually
achieved so the two can be read side by side, and it prints the per-cell counts
so that "we have no cell with enough rows to say" is visible rather than
implied.

Usage (read-only; opens the store with ``mode=ro``):

    python -X utf8 scripts/cell_edge_census.py
    python -X utf8 scripts/cell_edge_census.py --clip 0.81 --clip 3 --clip 23
    python -X utf8 scripts/cell_edge_census.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from services.fill_cost import leg_cost_usd  # noqa: E402
from services import outcome_plausibility as plausibility  # noqa: E402

#: Horizons in MINUTES. Spans the operator's stated 5-30 minute band and runs
#: out to 4h so "hold longer" is measured rather than argued about.
HORIZONS_MIN: Tuple[int, ...] = (5, 15, 30, 60, 120, 240)

#: Clips in USD. 0.81 is the MEDIAN NOTIONAL ACTUALLY TRADED on the 38 live
#: legs; 23 is the whole deployable wallet. Cost is not a constant across these
#: because gas is a fixed number of dollars per transaction.
CLIPS_USD: Tuple[float, ...] = (0.81, 3.0, 23.0)

#: A tape row this far (in log price) from its local median is a foreign row,
#: not a move. Same guard pass 452 used.
CONTAM_LOG_DEV = 0.5
CONTAM_HALFWIN = 20

#: A fill whose "slippage" exceeds this is a decimals artifact, not an
#: execution. The known case reads exactly -10000.0 bp (executed_price off by
#: 1e12), and being FAVOURABLY signed it makes cost look cheaper, so leaving it
#: in is the direction that loses money.
SLIP_ARTIFACT_BPS = 2000.0

#: Cells thinner than this are printed but never counted as findings.
MIN_CELL_N = 30

#: A symbol needs this much tape before it earns a row.
MIN_TAPE_ROWS = 300
MIN_TAPE_SPAN_H = 48.0

#: Symbols whose live legs support a per-symbol slippage mean. Below it the
#: mean is one or two rows and is reported as UNMEASURED, not as a small number.
MIN_COST_LEGS = 4

#: The symbols whose pooled slippage is the DEEP rate. These are the three with
#: four or more live legs each; their pooled 6.75 bp/leg (29 legs) is the figure
#: ``services.fill_cost.realised_slippage_bps`` was audited against in pass 453.
DEEP_SYMBOLS = ("AERO-USDC", "CBBTC-USDC", "CBETH-USDC")

#: A cell whose single largest row carries more than this share of its total
#: |move| is a contaminated cell, not a volatile one.
MAX_TOP1_SHARE = 0.25

#: z for the 95% upper bound on a symbol's slippage. Cost is a nuisance
#: parameter, not one of the K things being searched, so it pays the ordinary
#: two-sided tax and not sqrt(2 ln K).
COST_Z = 1.96


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
            "(data/trading.db, trading_bot.db, web/db.sqlite3). It returns "
            "nothing silently. Point TRADING_DB_PATH at storage/trading_cache.db."
        )
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


# --------------------------------------------------------------------------
# COST, from live fills
# --------------------------------------------------------------------------

def load_live_legs(conn: sqlite3.Connection) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Every ``live_*`` fill leg, split into (usable, censored-as-artifact)."""
    usable: List[Dict[str, Any]] = []
    censored: List[Dict[str, Any]] = []
    sql = "SELECT ts, chain, symbol, expected_price, executed_price, details FROM trade_fills"
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
            "symbol": symbol,
            "mode": mode,
            "leg": split["leg"],
            "slip_bps": split["realised_slippage_bps"],
            "notional_usd": split["notional_usd"],
            "gas_usd": split["gas_cost_usd"],
            "measured": split["measured"],
            "expected_price": exp_px,
            "executed_price": exe_px,
        }
        slip = row["slip_bps"]
        if slip is None or abs(float(slip)) >= SLIP_ARTIFACT_BPS:
            censored.append(row)
        else:
            usable.append(row)
    return usable, censored


def _mean_se(values: Sequence[float]) -> Tuple[float, float, float, int]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return mean, sd, se, n


def cost_model(legs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-symbol slippage and a pooled gas-per-leg, both from receipts."""
    by_symbol: Dict[str, List[float]] = {}
    for row in legs:
        by_symbol.setdefault(row["symbol"], []).append(float(row["slip_bps"]))

    gas = sorted(float(r["gas_usd"]) for r in legs if r["gas_usd"] is not None)
    gas_median = statistics.median(gas) if gas else float("nan")

    pooled = [float(r["slip_bps"]) for r in legs]
    p_mean, p_sd, p_se, p_n = _mean_se(pooled)
    p_total = sum(pooled) or float("nan")
    p_ordered = sorted(pooled, key=abs, reverse=True)

    deep = [float(r["slip_bps"]) for r in legs if r["symbol"] in DEEP_SYMBOLS]
    d_mean, d_sd, d_se, d_n = _mean_se(deep)

    per_symbol: Dict[str, Dict[str, Any]] = {}
    for sym, vals in sorted(by_symbol.items()):
        mean, sd, se, n = _mean_se(vals)
        top = max(vals, key=abs) if vals else float("nan")
        drop1 = [v for v in vals if v is not top]
        dm = statistics.fmean(drop1) if drop1 else float("nan")
        per_symbol[sym] = {
            "n_legs": n,
            "slip_mean_bps": mean,
            "slip_sd_bps": sd,
            "slip_se_bps": se,
            "slip_mean_drop1_bps": dm,
            "measured": n >= MIN_COST_LEGS,
        }
    return {
        "per_symbol": per_symbol,
        "gas_usd_per_leg_median": gas_median,
        "gas_usd_per_leg_n": len(gas),
        "gas_usd_iqr": (
            (gas[len(gas) // 4], gas[(3 * len(gas)) // 4]) if len(gas) >= 4 else None
        ),
        "pooled_slip_mean_bps": p_mean,
        "pooled_slip_se_bps": p_se,
        "pooled_slip_n": p_n,
        "pooled_top2_share": (sum(p_ordered[:2]) / p_total) if p_ordered else float("nan"),
        "deep_slip_mean_bps": d_mean,
        "deep_slip_se_bps": d_se,
        "deep_slip_n": d_n,
    }


def roundtrip_cost_bps(
    symbol: str,
    clip_usd: float,
    cost: Dict[str, Any],
    *,
    drop_top: bool = False,
    deep_fallback: bool = False,
    upper: bool = False,
) -> Tuple[Optional[float], str]:
    """Round-trip cost in bp for ``symbol`` at ``clip_usd``, and its provenance.

    Slippage is per LEG and charged twice; gas is a fixed number of DOLLARS per
    leg and charged twice, so its bp cost is a pure function of the clip -- that
    asymmetry is the whole reason a $0.81 clip and a $23 clip are different
    games on the same tape.

    Returns ``(None, reason)`` when the symbol has too few live legs of its own
    and ``deep_fallback`` is off: an unpriced symbol is unpriced, and charging
    it a number two foreign rows invented is how a census manufactures an edge.
    """
    gas_bps = 2.0 * float(cost["gas_usd_per_leg_median"]) / float(clip_usd) * 10_000.0
    entry = cost["per_symbol"].get(symbol)
    if entry and entry["measured"]:
        key = "slip_mean_drop1_bps" if drop_top else "slip_mean_bps"
        slip = float(entry[key])
        if upper and math.isfinite(entry["slip_se_bps"]):
            slip += COST_Z * float(entry["slip_se_bps"])
        return 2.0 * slip + gas_bps, f"fills n={entry['n_legs']}"
    have = entry["n_legs"] if entry else 0
    if not deep_fallback:
        return None, f"UNPRICED (own live legs n={have})"
    slip = float(cost["deep_slip_mean_bps"])
    if upper and math.isfinite(cost["deep_slip_se_bps"]):
        slip += COST_Z * float(cost["deep_slip_se_bps"])
    return 2.0 * slip + gas_bps, f"ASSUMED deep n={cost['deep_slip_n']} (own n={have})"


# --------------------------------------------------------------------------
# MOVE, from the tape
# --------------------------------------------------------------------------

def load_tape(conn: sqlite3.Connection) -> Dict[str, List[Tuple[float, float]]]:
    series: Dict[str, List[Tuple[float, float]]] = {}
    sql = (
        "SELECT symbol, ts, price FROM market_stream "
        "WHERE price IS NOT NULL AND price > 0 ORDER BY symbol, ts"
    )
    for sym, ts, price in conn.execute(sql):
        series.setdefault(str(sym), []).append((float(ts), float(price)))
    return series


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
) -> List[float]:
    """Absolute forward returns in bp over NON-OVERLAPPING windows.

    Each window starts at the first unconsumed tick and closes on the first
    tick at or after ``horizon_s``, accepted only when its elapsed time landed
    inside ``[H, H + max(120, H/2)]``. The next window starts at the closing
    tick, so no two windows share a price path and their standard error is a
    real one.
    """
    tol = max(120.0, horizon_s / 2.0)
    out: List[float] = []
    i = 0
    n = len(points)
    while i < n - 1:
        t0, p0 = points[i]
        j = i + 1
        while j < n and points[j][0] - t0 < horizon_s:
            j += 1
        if j >= n:
            break
        elapsed = points[j][0] - t0
        if elapsed <= horizon_s + tol:
            out.append(abs(points[j][1] / p0 - 1.0) * 10_000.0)
            i = j
        else:
            # The tape jumped a blackout. Skip the gap rather than measure it.
            i = j
    return out


# --------------------------------------------------------------------------
# the cell table
# --------------------------------------------------------------------------

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
        base_5m: Optional[float] = None
        for h in sorted(horizons_min):
            moves = forward_moves(clean, h * 60.0)
            if len(moves) < 2:
                continue
            mean, sd, se, n = _mean_se(moves)
            ordered = sorted(moves, reverse=True)
            total = sum(moves) or float("nan")
            drop1 = moves[:]
            drop1.remove(ordered[0])
            d_mean, _, d_se, _d_n = _mean_se(drop1)
            half = len(moves) // 2
            h1 = statistics.fmean(moves[:half]) if half else float("nan")
            h2 = statistics.fmean(moves[half:]) if half else float("nan")
            if base_5m is None and mean > 0:
                base_5m, base_h = mean, h
            vr = float("nan")
            if base_5m:
                vr = mean / (base_5m * math.sqrt(h / float(base_h)))
            cells.append({
                "symbol": sym,
                "horizon_min": h,
                "n": n,
                "mean_abs_move_bps": mean,
                "sd_bps": sd,
                "se_bps": se,
                "top1_share": ordered[0] / total,
                "top3_share": sum(ordered[:3]) / total,
                "mean_drop1_bps": d_mean,
                "se_drop1_bps": d_se,
                "half1_bps": h1,
                "half2_bps": h2,
                "variance_ratio": vr,
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
# the book, from paired fills + trade_outcomes
# --------------------------------------------------------------------------

HOLD_BUCKETS: Tuple[Tuple[str, float, float], ...] = (
    ("<=15m", 0.0, 15.0),
    ("15-30m", 15.0, 30.0),
    ("30-60m", 30.0, 60.0),
    ("1-4h", 60.0, 240.0),
    (">4h", 240.0, float("inf")),
)


def load_round_trips(conn: sqlite3.Connection) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Paired entry/exit fills with a holding time, split (countable, refused).

    ``trade_outcomes`` has no entry timestamp, so the hold is recovered by
    joining ``trade_fills`` on ``details.trade_id``. The refused half is the
    rows ``services.outcome_plausibility`` says the ledger would not have
    recorded -- this repo has published an "AERO profit" that was one such row.
    """
    entries: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}
    exits: List[Tuple[float, str, Dict[str, Any]]] = []
    for ts, symbol, raw in conn.execute("SELECT ts, symbol, details FROM trade_fills"):
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
        t0, _sym0, edet = entries[tid]
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
            "symbol": symbol,
            "mode": str(det.get("mode") or ""),
            "strategy_id": det.get("strategy_id") or "",
            "hold_min": hold_min,
            "entry_price": float(entry_px),
            "exit_price": float(exit_px),
            "gross_profit": det.get("gross_profit"),
            "ret_bps": ret * 10_000.0,
        })
    keep, drop = plausibility.partition(rows)
    return keep, drop


def bucket_of(hold_min: float) -> str:
    for name, lo, hi in HOLD_BUCKETS:
        if lo <= hold_min < hi:
            return name
    return ">4h"


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
    ap.add_argument("--clip", action="append", type=float, default=None,
                    help="clip in USD; repeatable. Default 0.81 / 3 / 23.")
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS_MIN),
                    help="comma-separated holding horizons in MINUTES")
    ap.add_argument("--top", type=int, default=0,
                    help="print only the N best cells per clip (0 = all)")
    ap.add_argument("--deep-fallback", action="store_true",
                    help="price symbols with no live legs of their own at the "
                         "AERO/CBBTC/CBETH pooled rate, labelled ASSUMED")
    ap.add_argument("--symbols", default=None,
                    help="comma-separated symbols to print (default: all)")
    ap.add_argument("--json", default=None, help="also write the full table here")
    args = ap.parse_args(argv)
    only = ({s.strip().upper() for s in args.symbols.split(",")}
            if args.symbols else None)

    clips = tuple(args.clip) if args.clip else CLIPS_USD
    horizons = tuple(int(h) for h in str(args.horizons).split(",") if h.strip())

    conn = connect(args.db)
    print(f"store: {args.db} ({os.path.getsize(args.db)/1e9:.1f} GB)")

    # ---- cost -------------------------------------------------------------
    legs, censored = load_live_legs(conn)
    cost = cost_model(legs)
    print()
    print("=" * 78)
    print("COST -- measured on REAL live fills (services.fill_cost), not a quoter")
    print("=" * 78)
    print(f"live legs usable {len(legs)}, censored as decimals artifacts {len(censored)}")
    for row in censored:
        print(f"  CENSORED {row['symbol']:<14} {row['mode']:<12} "
              f"expected {row['expected_price']:.6g} executed {row['executed_price']:.6g} "
              f"-> {row['slip_bps']:.1f} bp")
    gas_iqr = cost["gas_usd_iqr"]
    print(f"gas/leg  median ${cost['gas_usd_per_leg_median']:.6f} over "
          f"{cost['gas_usd_per_leg_n']} priced legs"
          + (f"  IQR ${gas_iqr[0]:.6f}-${gas_iqr[1]:.6f}" if gas_iqr else ""))
    print(f"slippage pooled over ALL {cost['pooled_slip_n']} legs  "
          f"{cost['pooled_slip_mean_bps']:.2f} bp/leg -- "
          f"AND ITS TOP 2 ROWS ARE {cost['pooled_top2_share']*100:.0f}% OF THAT SUM. "
          "Not usable as a fallback.")
    print(f"slippage pooled over DEEP ({'/'.join(DEEP_SYMBOLS)})  "
          f"{cost['deep_slip_mean_bps']:.4f} bp/leg  SE {cost['deep_slip_se_bps']:.2f}  "
          f"n={cost['deep_slip_n']}  -> round trip {2*cost['deep_slip_mean_bps']:.2f} bp")
    print()
    print(f"{'symbol':<14} {'legs':>5} {'slip bp/leg':>12} {'SE':>7} {'drop-top':>9}  status")
    for sym, s in sorted(cost["per_symbol"].items(), key=lambda kv: -kv[1]["n_legs"]):
        status = "MEASURED" if s["measured"] else "UNPRICED (own reading shown, not used)"
        print(f"{sym:<14} {s['n_legs']:>5} {fmt(s['slip_mean_bps'],12,2)} "
              f"{fmt(s['slip_se_bps'],7,1)} {fmt(s['slip_mean_drop1_bps'],9,2)}  {status}")
    print()
    print("round-trip cost in bp = 2*slip + 2*gas/clip   (upper = 95% bound on slip):")
    print(f"{'symbol':<14}" + "".join(f"{('$%g' % c):>10}{'upper':>9}" for c in clips))
    measured_syms = [s for s, v in cost["per_symbol"].items() if v["measured"]]
    for sym in measured_syms + ["<deep ASSUMED>"]:
        out = []
        for c in clips:
            v, _ = roundtrip_cost_bps(sym, c, cost, deep_fallback=True)
            u, _ = roundtrip_cost_bps(sym, c, cost, deep_fallback=True, upper=True)
            out.append(f"{v:10.1f}{u:9.1f}")
        print(f"{sym:<14}" + "".join(out))

    # ---- moves ------------------------------------------------------------
    tape = load_tape(conn)
    cells = build_cells(tape, horizons)
    k = len(cells)
    tax = math.sqrt(2.0 * math.log(k)) if k > 1 else float("nan")
    print()
    print("=" * 78)
    print("MOVE -- market_stream, NON-OVERLAPPING forward windows, contamination-guarded")
    print("=" * 78)
    print(f"K = {k} cells searched -> multiple-comparisons tax sqrt(2 ln K) = {tax:.3f} SE")

    results: List[Dict[str, Any]] = []
    for clip in clips:
        rows = []
        for cell in cells:
            sym = cell["symbol"]
            cost_bps, src = roundtrip_cost_bps(
                sym, clip, cost, deep_fallback=args.deep_fallback)
            up_bps, _ = roundtrip_cost_bps(
                sym, clip, cost, deep_fallback=args.deep_fallback, upper=True)
            bar = noise_bar(cell["se_bps"], k)
            mean = cell["mean_abs_move_bps"]
            lower = mean - bar
            d_mean = cell["mean_drop1_bps"]
            d_lower = d_mean - noise_bar(cell["se_drop1_bps"], k)

            def _h(c: Optional[float], m: float) -> float:
                if c is None:
                    return float("nan")
                return 0.5 + c / (2.0 * m) if m > 0 else float("inf")

            unpriced = cost_bps is None
            contaminated = cell["top1_share"] > MAX_TOP1_SHARE
            thin = cell["n"] < MIN_CELL_N
            passes = (not unpriced) and lower > cost_bps
            passes_drop1 = (not unpriced) and d_lower > cost_bps
            if unpriced:
                verdict = "UNPRICED"
            elif contaminated:
                verdict = f"CONTAMINATED top1={cell['top1_share']:.2f}"
            elif thin:
                verdict = f"thin n={cell['n']}"
            elif mean <= cost_bps:
                verdict = "cost > move"
            elif not passes:
                verdict = "under the bar"
            elif not passes_drop1:
                verdict = "CARRIED by top row"
            else:
                verdict = "CLEARS"
                if _h(up_bps, lower) >= 1.0:
                    verdict = "CLEARS*  (fails at cost upper bound)"
            rows.append({**cell,
                         "clip_usd": clip,
                         "cost_bps": cost_bps,
                         "cost_upper_bps": up_bps,
                         "cost_source": src,
                         "bar_bps": bar,
                         "mean_minus_bar_bps": lower,
                         "h_star": _h(cost_bps, mean),
                         "h_star_after_bar": _h(cost_bps, lower),
                         "h_star_worst": _h(up_bps, lower),
                         "verdict": verdict,
                         "clears_bar": bool(passes and not contaminated and not thin),
                         "clears_bar_drop_top_row": bool(passes_drop1)})
        rows.sort(key=lambda r: (
            -(r["mean_minus_bar_bps"] - (r["cost_bps"] if r["cost_bps"] is not None else 1e9))
        ))
        results.extend(rows)

        print()
        print("-" * 96)
        print(f"CLIP ${clip:g}   (cost = 2*slip + 2*${cost['gas_usd_per_leg_median']:.6f}/{clip:g})")
        print("-" * 96)
        print(f"{'symbol':<14}{'hor':>5}{'n':>6}{'mean|mv|':>10}{'SE':>7}{'bar':>7}"
              f"{'cost':>8}{'h*':>7}{'h*@bar':>8}{'h*worst':>9}{'top1':>6}"
              f"{'VR':>6}{'half1':>8}{'half2':>8}  verdict")
        shown = [r for r in rows if only is None or r["symbol"].upper() in only]
        if args.top:
            shown = shown[: args.top]
        for r in shown:
            print(f"{r['symbol']:<14}{r['horizon_min']:>4}m{r['n']:>6}"
                  f"{fmt(r['mean_abs_move_bps'],10,1)}{fmt(r['se_bps'],7,1)}"
                  f"{fmt(r['bar_bps'],7,1)}{fmt(r['cost_bps'],8,1)}"
                  f"{fmt(r['h_star'],7,3)}{fmt(r['h_star_after_bar'],8,3)}"
                  f"{fmt(r['h_star_worst'],9,3)}{fmt(r['top1_share'],6,2)}"
                  f"{fmt(r['variance_ratio'],6,2)}{fmt(r['half1_bps'],8,1)}"
                  f"{fmt(r['half2_bps'],8,1)}  {r['verdict']}")

    # ---- the book ---------------------------------------------------------
    keep, refused = load_round_trips(conn)
    print()
    print("=" * 78)
    print("BOOK -- what we ACTUALLY held, by (symbol, realised hold). Direction, not ceiling")
    print("=" * 78)
    print(f"paired round trips {len(keep)} countable, {len(refused)} refused by "
          "services.outcome_plausibility")
    by_bucket: Dict[str, List[Dict[str, Any]]] = {}
    for r in keep:
        by_bucket.setdefault(bucket_of(r["hold_min"]), []).append(r)
    print()
    print(f"{'hold':<10}{'n':>6}{'hit rate':>10}{'mean ret bp':>13}{'SE':>8}"
          f"{'mean|mv| bp':>13}{'top1 share':>12}")
    for name, _lo, _hi in HOLD_BUCKETS:
        rows_b = by_bucket.get(name) or []
        if not rows_b:
            continue
        rets = [r["ret_bps"] for r in rows_b]
        mean, sd, se, n = _mean_se(rets)
        hit = sum(1 for v in rets if v > 0) / n
        absr = [abs(v) for v in rets]
        tot = sum(absr) or float("nan")
        print(f"{name:<10}{n:>6}{fmt(hit,10,3)}{fmt(mean,13,1)}{fmt(se,8,1)}"
              f"{fmt(statistics.fmean(absr),13,1)}{fmt(max(absr)/tot,12,2)}")
    allr = [r["ret_bps"] for r in keep]
    if allr:
        mean_all, _sd, se_all, n_all = _mean_se(allr)
        hit_all = sum(1 for v in allr if v > 0) / n_all
        hit_se = math.sqrt(hit_all * (1 - hit_all) / n_all)
        print(f"{'ALL':<10}{n_all:>6}{fmt(hit_all,10,3)}"
              f"{fmt(mean_all,13,1)}{fmt(se_all,8,1)}")
        print()
        print(f"REALISED HIT RATE = {hit_all:.3f} +/- {hit_se:.3f} over {n_all} "
              f"round trips -- indistinguishable from a coin flip"
              if abs(hit_all - 0.5) < 2 * hit_se else
              f"REALISED HIT RATE = {hit_all:.3f} +/- {hit_se:.3f} over {n_all}")
        print("This is the h the book ACTUALLY achieved, gross of cost, on "
              f"{sum(1 for r in keep if r['mode'].startswith('ghost'))} ghost and "
              f"{sum(1 for r in keep if r['mode'].startswith('live'))} live rows. "
              "A cell above is an edge only if its h* is BELOW an h we can "
              "demonstrate -- and h* < 1 alone demonstrates nothing.")

    # Can the book resolve a (symbol, hold) cell at all?
    per_cell: Dict[Tuple[str, str], List[float]] = {}
    for r in keep:
        per_cell.setdefault((r["symbol"], bucket_of(r["hold_min"])), []).append(r["ret_bps"])

    print()
    print(f"book (symbol, hold) cells with n >= {MIN_CELL_N}: "
          f"{sum(1 for v in per_cell.values() if len(v) >= MIN_CELL_N)} of {len(per_cell)}")
    print(f"{'symbol':<14}{'hold':<9}{'n':>5}{'hit':>8}{'mean bp':>10}{'SE':>8}{'top1':>7}")
    for (sym, b), vals in sorted(per_cell.items(), key=lambda kv: -len(kv[1]))[:10]:
        mean, _sd, se, n = _mean_se(vals)
        absv = [abs(v) for v in vals]
        hit = sum(1 for v in vals if v > 0) / n
        print(f"{sym:<14}{b:<9}{n:>5}{fmt(hit,8,3)}{fmt(mean,10,1)}{fmt(se,8,1)}"
              f"{fmt(max(absv)/(sum(absv) or float('nan')),7,2)}")

    # ---- what production charges -----------------------------------------
    print()
    print("=" * 96)
    print("GATE CROSS -- the per-token table money_button refuses on, against these fills")
    print("=" * 96)
    print("trading/scheduler.py:1777-1787 refuses a symbol when")
    print("  services.round_trip_cost.token_round_trip_cost(symbol) > "
          "MONEY_BUTTON_MAX_ROUND_TRIP_COST")
    try:
        from services import roundtrip_cost as entry_cost
        from services import round_trip_cost as gate_cost
        bar_gate = float(os.environ.get("MONEY_BUTTON_MAX_ROUND_TRIP_COST", "0.0035"))
        print(f"  bar = {bar_gate*10_000:.0f} bp; unlisted tokens price at "
              f"{gate_cost.default_token_round_trip()*10_000:.0f} bp and are refused.")
        print()
        print(f"{'symbol':<14}{'gate bp':>10}{'measured RT':>13}{'95% upper':>11}"
              f"{'legs':>6}  verdict at the gate")
        seen = set()
        for sym in sorted(set(cost["per_symbol"]) | {"WETH-USDC", "ETH-USDC"}):
            if sym in seen:
                continue
            seen.add(sym)
            charged = gate_cost.token_round_trip_cost(sym) * 10_000
            s = cost["per_symbol"].get(sym)
            best_clip = max(clips)
            mine, _ = roundtrip_cost_bps(sym, best_clip, cost)
            up, _ = roundtrip_cost_bps(sym, best_clip, cost, upper=True)
            n_legs = s["n_legs"] if s else 0
            gate_says = "REFUSED" if charged > bar_gate * 10_000 else "admitted"
            mine_says = ("n/a" if mine is None else
                         ("refused" if mine > bar_gate * 10_000 else "admitted"))
            print(f"{sym:<14}{charged:>10.0f}{fmt(mine,13,1)}{fmt(up,11,1)}"
                  f"{n_legs:>6}  gate {gate_says}; at the measured number {mine_says}"
                  f"  (clip ${best_clip:g})")
        print()
        print("'measured RT' is 2*realised_slippage + 2*gas/clip from THIS symbol's own "
              "live legs;\nit is the same quantity TOKEN_ROUND_TRIP claims to hold "
              "(both legs, gas, fee, spread,\nimpact) because the AMM fee arrives inside "
              "executed_price -- services/fill_cost.py:30-38.\nA blank means the symbol "
              "has fewer than four live legs and is genuinely UNPRICED.")
        print()
        for clip in clips:
            print(f"  $:{clip:>6g}  services.roundtrip_cost.roundtrip_cost_rate = "
                  f"{entry_cost.roundtrip_cost_rate(clip) * 10_000:.1f} bp  (entry path, "
                  "notional-only, no symbol)")
        print(f"  services.round_trip_cost.fallback_cost = "
              f"{gate_cost.fallback_cost() * 10_000:.1f} bp  (gate path, symbol-blind)")
    except Exception as exc:  # pragma: no cover - comparison only
        print(f"  (could not read the production cost modules: {exc})")

    if args.json:
        payload = {
            "db": args.db,
            "k_cells": k,
            "tax_sqrt_2lnK": tax,
            "cost": cost,
            "cells": results,
            "book": [
                {"bucket": bucket_of(r["hold_min"]), **r} for r in keep
            ],
            "book_refused": len(refused),
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
