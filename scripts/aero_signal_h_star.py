"""AERO-USDC vs h*, with AERO's OWN cost, on a PRE-REGISTERED FORWARD sample.

Item [aea07127].  Pass 676 ran ``scripts/weth_signal_h_star.py`` across several
base symbols.  It KILLED WETH-USDC, ETH-USDC and CBBTC-USDC and left one cell
standing: AERO-USDC ema_cross at 60m, accuracy 0.7368 over n=38 against a
repriced h* of 0.6097, raw one-sided p 0.0026.

THAT CELL IS A LEAD AND NOT A RESULT, and this file exists to settle it the one
way it can be settled -- on a sample the lead was NOT found in.

WHY THIS FILE DOES NOT PRINT A VERDICT TODAY (pass 679, Dune).  Measured before
any signal code was written:

    select count(*), min(ts), max(ts) from market_stream
      where symbol = 'AERO-USDC' and price > 0
    -> 11717 ticks, 2026-08-26T17:02:38 .. 2026-09-23T06:03:27

    ... and the same query with ts < 2026-08-26T00:00:00Z -> 0 ticks.

THE ENTIRE AERO TAPE *IS* THE DISCOVERY WINDOW.  There is no earlier AERO data
anywhere in the store: ``prices`` holds no AERO at all (its tokens are the CEX
majors ada..trx), ``ohlcv_datasets`` has zero AERO rows, ``ohlcv_bars`` carries
no symbol column, and ``market_stream symbol LIKE 'AERO%'`` is 12137 rows with
ZERO before the cutoff, so it is not a ticker-spelling problem either.

A holdout carved out of 2026-08-26..2026-09-23 is exactly what the item
forbids, so this script REFUSES to print a verdict on any window that overlaps
the discovery window.  It prints the in-window reproduction, labelled, and
stops.  An absent sample is not a negative result; it is no result.

HYPOTHESIS AND COMPLEMENT, pre-registered here so a later pass cannot tune them
after seeing the forward data:

    HYPOTHESIS  ema_cross calls AERO-USDC's forward direction at 60m at or
                above h*, on data it was not found in, and the AERO route is
                OPEN ON DIRECTION.
    COMPLEMENT  the 0.7368 was a window.  On forward data ema_cross lands at or
                below h*, or its one-sided p sits above the family-wise alpha,
                and the AERO cell dies exactly as the WETH cell did.

THE PRE-REGISTERED TEST, fixed before the forward sample exists:

    symbol      AERO-USDC
    signal      ema_cross          (the ONLY cell under test; the other three
                                    rules are printed as context and are NOT
                                    part of the verdict)
    horizon     60 minutes         (the cell that was found; 45m is reported
                                    but is NOT the verdict horizon)
    metric      long_accuracy on DISJOINT windows vs that sample's own h*
    threshold   accuracy >= h* computed at COST_ALL_IN_BP below, AND a
                one-sided binomial p < 0.05 / (tests in the family)
    n           the verdict requires n_long >= 30 on the DISJOINT sample.
                Below that the MDE against a 50%% null is wider than 15pp and
                the test cannot resolve the effect being claimed -- the run
                reports UNRESOLVED, not KILLED.
    window      --since 2026-09-23T06:03:28Z (the last tick of the discovery
                window) or later.  Nothing earlier is admissible.

WHAT AERO'S ROUND TRIP COSTS, and why it is NOT the live book.  The item is
explicit that the 9 wallet='live' AERO rows carry zero information: their
fee_cost is flat $0.00284..$0.00432 across notionals of $0.64..$1.52, i.e. it
is GAS, and the resulting "39.18bp" is a pure 1/clip curve.  So the cost here
comes from the read-only quote census, ``data/round_trip_quote_census_
20260917T190210Z.json`` (generated 2026-09-17 by scripts/routing_arbitrage_
census.py --round-trip-census, git 769cc58, chain base, Uniswap V3 QuoterV2,
both legs SAME fee tier).  Its AERO entry, verbatim:

    fee_tier 500, rows 45, address 0x940181a9...98631
    points  $0.10 0.1010%  $0.25 0.1004%  $0.50 0.1004%  $0.75 0.2429%
            $1.00 0.1302%  $2.00 0.1303%  $4.00 0.1020%  $6.00 0.1029%
    implausible  $10: cost 0.0376%% is BELOW the 2*fee floor 0.1000%%

READ THAT LAST LINE.  THERE IS NO $10 CENSUS COST FOR AERO.  The census asked
the $10 clip and REFUSED its own answer under the plausibility floor a round
trip through one pool cannot beat.  So the item's phrase "AERO's own 14.34bp
census cost" is reproducible but is NOT a $10 reading: 10.2933bp is the $6.00
clip, the largest one the census ANSWERED, and 10.2933 + 4.05 = 14.3433.  The
4.05bp is the REALISED gas median ($0.004047 over the 18 closed live rows) at a
$10 clip, the same term inside ``weth_direction_census.COST_ALL_IN_BP`` -- this
file asserts 2.06 + 4.05 == that imported 6.11 at import time, so the
decomposition cannot silently drift from the constant that killed WETH.

h* MOVES WITH COST and the whole lead lives or dies on it, so all three
defensible cost readings are printed on every run rather than one chosen:

    FLOOR    14.05bp = 10.00 (the census's own 2*fee_tier floor) + 4.05 gas
    CENSUS   14.34bp = 10.2933 ($6.00, largest answered clip)     + 4.05 gas
    MEDIAN   17.08bp = 13.03  ($2.00, the census's median clip)   + 4.05 gas

The verdict uses CENSUS, because that is the reading the lead was repriced at
and changing it now would be choosing the bar after seeing the number.  The
other two are the sensitivity band, and a cell that clears CENSUS but fails
MEDIAN is reported as MARGINAL, never as CONFIRMED.

This script places no trade, changes no gate and produces no tx hash.  If it
ever prints CONFIRMED, the next step is the capital gate [74a12c84], which
needs the operator -- NOT a trade from this file.

Usage:
    # what a later pass runs, once forward data exists:
    python -X utf8 scripts/aero_signal_h_star.py --since 2026-09-23T06:03:28Z

    # the in-window reproduction (NEVER a verdict):
    python -X utf8 scripts/aero_signal_h_star.py --allow-in-window
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sqlite3
import sys
import time
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import weth_signal_h_star as W  # noqa: E402
from weth_direction_census import COST_ALL_IN_BP as _WETH_COST_ALL_IN_BP  # noqa: E402
from weth_signal_h_star import SIGNALS, WARMUP_TICKS, score  # noqa: E402
from weth_direction_census import MAX_PRICE_SCALE  # noqa: E402

#: The two terms of the imported WETH constant, split out so AERO can swap the
#: QUOTE term without re-typing the GAS term.  Asserted, not asserted-to-taste:
#: if anybody moves COST_ALL_IN_BP this file fails loudly at import.
_WETH_QUOTED_BP = 2.06
GAS_BP = 4.05
assert abs((_WETH_QUOTED_BP + GAS_BP) - _WETH_COST_ALL_IN_BP) < 0.005, (
    "gas/quote split no longer reproduces weth_direction_census.COST_ALL_IN_BP "
    "(%s + %s != %s) -- re-derive it before trusting any h* printed here"
    % (_WETH_QUOTED_BP, GAS_BP, _WETH_COST_ALL_IN_BP)
)

#: AERO quote readings, straight out of the census artifact named in the
#: docstring.  Nothing here is a default and nothing here is a live fill.
AERO_QUOTE_READINGS_BP = {
    "FLOOR": 10.0000,   # census plausibility floor, 2 * fee_tier 500 / 1e4
    "CENSUS": 10.2933,  # $6.00 clip -- the LARGEST clip the census answered
    "MEDIAN": 13.0300,  # $2.00 clip -- the census's own median clip
}
AERO_COSTS_BP = {k: round(v + GAS_BP, 4) for k, v in AERO_QUOTE_READINGS_BP.items()}
VERDICT_COST_KEY = "CENSUS"

#: The census asked $10 and threw its own answer away.  Kept as data so no
#: later pass "finds" a $10 AERO cost by reading the artifact carelessly.
AERO_TEN_DOLLAR_CLIP = {
    "cost_pct": 0.0376,
    "floor_pct": 0.1,
    "status": "REFUSED BY THE CENSUS -- below the 2*fee floor; there is no $10 AERO cost",
}

#: The window the lead was found in.  Any overlap disqualifies a verdict.
DISCOVERY_START_ISO = "2026-08-26T00:00:00Z"
DISCOVERY_END_ISO = "2026-09-23T06:03:28Z"

#: Pre-registered, and read by the verdict rather than eyeballed.
VERDICT_SIGNAL = "ema_cross"
VERDICT_HORIZON_MIN = 60.0
VERDICT_MIN_N = 30


def _iso_to_ts(s: str) -> float:
    s = s.strip().replace("Z", "+00:00")
    d = _dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone.utc)
    return d.timestamp()


def _ts_to_iso(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _db() -> sqlite3.Connection:
    path = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
    if not os.path.exists(path):
        raise SystemExit("store not found: %s" % path)
    return sqlite3.connect("file:%s?mode=ro" % path, uri=True)


def load_window(conn: sqlite3.Connection, symbol: str, since: float, until: float):
    """Same mixed-denomination guard as load_tape, but on an ABSOLUTE window.

    ``weth_signal_h_star.load_tape`` takes a trailing ``days``, which can only
    ever reach BACKWARDS from now and therefore can never express "the sample
    after the discovery window".  That is the only reason this function exists;
    the guard below is the same constant and the same REFUSE-don't-filter shape.
    """
    rows = conn.execute(
        "select ts, price, volume from market_stream where symbol = ? "
        "and ts >= ? and ts < ? and price > 0 order by ts",
        (symbol, since, until),
    ).fetchall()
    if not rows:
        return np.array([]), np.array([]), np.array([]), 0
    ts = np.array([float(r[0]) for r in rows])
    px = np.array([float(r[1]) for r in rows])
    vol = np.array([float(r[2] or 0.0) for r in rows])
    median = float(np.median(px))
    if median <= 0:
        return np.array([]), np.array([]), np.array([]), 0
    spread = np.maximum(px / median, median / px)
    keep = spread <= MAX_PRICE_SCALE
    return ts[keep], px[keep], vol[keep], int((~keep).sum())


def run(symbol: str, since: float, until: float, horizons_min: List[float]) -> Dict:
    conn = _db()
    ts, px, vol, dropped = load_window(conn, symbol, since, until)
    out: Dict = {
        "symbol": symbol,
        "window_since": _ts_to_iso(since),
        "window_until": _ts_to_iso(until),
        "ticks": int(ts.size),
        "ticks_dropped_mixed_denomination": dropped,
        "warmup_ticks": WARMUP_TICKS,
        "cost_readings_bp": AERO_COSTS_BP,
        "cost_used_key": VERDICT_COST_KEY,
        "cost_used_bp": AERO_COSTS_BP[VERDICT_COST_KEY],
        "cost_provenance": (
            "quote census data/round_trip_quote_census_20260917T190210Z.json, "
            "quoted.AERO, fee_tier 500, $6.00 clip (LARGEST ANSWERED; the $10 "
            "clip was REFUSED by the census's own 2*fee plausibility floor) "
            "+ %.2fbp realised gas at a $10 clip" % GAS_BP
        ),
        "ten_dollar_clip": AERO_TEN_DOLLAR_CLIP,
        "cost_is_not_from_the_live_book": (
            "the 9 wallet='live' AERO rows are flat $0.00284..$0.00432 fee_cost "
            "on $0.64..$1.52 notionals -- that is gas, and its bp figure is a "
            "1/clip curve carrying zero information about swap cost"
        ),
        "pre_registered": {
            "signal": VERDICT_SIGNAL,
            "horizon_min": VERDICT_HORIZON_MIN,
            "min_n_long": VERDICT_MIN_N,
            "metric": "long_accuracy on DISJOINT windows vs that sample's own h*",
            "alpha": "0.05 / tests_in_family (Bonferroni)",
        },
        "signals": {},
    }
    if ts.size > 1:
        out["tape_span_hours"] = round((ts[-1] - ts[0]) / 3600.0, 2)
        out["median_gap_sec"] = round(float(np.median(np.diff(ts))), 1)
        out["first_tick"] = _ts_to_iso(float(ts[0]))
        out["last_tick"] = _ts_to_iso(float(ts[-1]))
    if ts.size <= WARMUP_TICKS + 10:
        out["verdict"] = "INSUFFICIENT TAPE -- %d ticks, need more than %d" % (
            ts.size, WARMUP_TICKS + 10)
        return out

    # Swap the module-global the imported scorer reads.  Restored afterwards so
    # importing this file can never leave the WETH scorer priced at AERO's cost.
    saved = W.COST_ALL_IN_BP
    W.COST_ALL_IN_BP = AERO_COSTS_BP[VERDICT_COST_KEY]
    try:
        for name, fn in SIGNALS.items():
            preds = fn(ts, px, vol)
            blocks: Dict = {}
            for h in horizons_min:
                dj = score(ts, px, vol, h * 60.0, preds, disjoint=True)
                ov = score(ts, px, vol, h * 60.0, preds, disjoint=False)
                dj["OVERLAPPING_long_accuracy_point_estimate_only"] = ov.get("long_accuracy")
                dj["OVERLAPPING_n_long"] = ov.get("n_long")
                # h* at the other two defensible costs, so the sensitivity band
                # is printed rather than reconstructed by the reader.
                m = dj.get("mean_abs_move_bp") or 0.0
                dj["h_star_sensitivity"] = {
                    k: (None if m <= 0 else round(0.5 + c / (2.0 * m), 4))
                    for k, c in AERO_COSTS_BP.items()
                }
                blocks[str(h) + "m"] = dj
            out["signals"][name] = blocks
    finally:
        W.COST_ALL_IN_BP = saved
    return out


def verdict(res: Dict, in_window: bool) -> List[str]:
    lines: List[str] = []
    mult = W.multiplicity(res)
    res["multiplicity"] = mult
    alpha = mult.get("family_wise_alpha_per_test") or 0.05
    key = "%gm" % VERDICT_HORIZON_MIN
    cell = res.get("signals", {}).get(VERDICT_SIGNAL, {}).get(key)

    if in_window:
        lines.append("  VERDICT: WITHHELD.  This window OVERLAPS the discovery window")
        lines.append("  %s .. %s, and [aea07127] says in terms that a re-run on the"
                     % (DISCOVERY_START_ISO, DISCOVERY_END_ISO))
        lines.append("  sample a lead was found in confirms nothing.  Everything above is an")
        lines.append("  IN-WINDOW REPRODUCTION and is a point estimate only.")
        lines.append("  Re-run with --since %s once forward data exists." % DISCOVERY_END_ISO)
        return lines
    if cell is None or not cell.get("n_long"):
        lines.append("  VERDICT: UNRESOLVED -- %s produced no paired long fires at %s."
                     % (VERDICT_SIGNAL, key))
        return lines
    n = int(cell["n_long"])
    acc = float(cell["long_accuracy"])
    hs = cell.get("h_star_required_accuracy")
    p = float(cell.get("long_p_one_sided_vs_coin", 1.0))
    if n < VERDICT_MIN_N:
        lines.append("  VERDICT: UNRESOLVED -- n=%d is below the pre-registered %d."
                     % (n, VERDICT_MIN_N))
        lines.append("  MDE at this n is %.2fpp, wider than the effect claimed.  NOT a kill:"
                     % cell.get("long_mde_pp_at_95", float("nan")))
        lines.append("  collect more forward tape and re-run the SAME command.")
        return lines
    clears = bool(hs is not None and acc >= hs)
    sens = cell.get("h_star_sensitivity", {})
    clears_median = bool(sens.get("MEDIAN") is not None and acc >= sens["MEDIAN"])
    if clears and p < alpha and clears_median:
        lines.append("  VERDICT: CONFIRMED.  %s clears h* at %s on a DISJOINT FORWARD sample"
                     % (VERDICT_SIGNAL, key))
        lines.append("  (acc %.4f vs h* %.4f, n=%d, p=%.5f < alpha %s) and it also clears the"
                     % (acc, hs, n, p, alpha))
        lines.append("  dearer MEDIAN-clip cost reading.  HAND TO THE CAPITAL GATE [74a12c84].")
        lines.append("  DO NOT place a trade from this file.")
    elif clears and p < alpha:
        lines.append("  VERDICT: MARGINAL.  %s clears h* at the CENSUS cost (%.4f vs %.4f,"
                     % (VERDICT_SIGNAL, acc, hs))
        lines.append("  n=%d, p=%.5f) but FAILS the dearer MEDIAN reading (h* %s).  An edge that"
                     % (n, p, sens.get("MEDIAN")))
        lines.append("  survives only the cheapest defensible cost is not a live-lane edge.")
    else:
        lines.append("  VERDICT: KILLED.  %s does NOT clear h* at %s on a DISJOINT FORWARD"
                     % (VERDICT_SIGNAL, key))
        lines.append("  sample: acc %.4f vs h* %s, n=%d, p=%.5f, alpha %s."
                     % (acc, hs, n, p, alpha))
        lines.append("  The AERO cell was a window, exactly as the WETH cell was.")
        lines.append("  Record it in data/attempts-revenir.md and do not refile it.")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AERO-USDC")
    ap.add_argument("--since", default=DISCOVERY_END_ISO,
                    help="ISO UTC start of the forward sample (default: the last tick "
                         "of the discovery window)")
    ap.add_argument("--until", default=None, help="ISO UTC end; default now")
    ap.add_argument("--horizons", default="20,30,45,60")
    ap.add_argument("--allow-in-window", action="store_true",
                    help="score the DISCOVERY window itself. Prints the reproduction "
                         "and REFUSES a verdict.")
    ap.add_argument("--out", default="data/aero_signal_h_star.json")
    args = ap.parse_args()

    disc_a = _iso_to_ts(DISCOVERY_START_ISO)
    disc_b = _iso_to_ts(DISCOVERY_END_ISO)
    if args.allow_in_window:
        since, until = disc_a, disc_b
    else:
        since = _iso_to_ts(args.since)
        until = _iso_to_ts(args.until) if args.until else time.time()
    in_window = since < disc_b and until > disc_a

    horizons = [float(x) for x in args.horizons.split(",") if x.strip()]
    res = run(args.symbol, since, until, horizons)
    res["overlaps_discovery_window"] = in_window
    res["discovery_window"] = [DISCOVERY_START_ISO, DISCOVERY_END_ISO]
    res["generated_at"] = time.time()

    print("=" * 78)
    print("%s vs h* -- %s" % (args.symbol,
                             "IN-WINDOW REPRODUCTION, NOT A VERDICT" if in_window
                             else "DISJOINT FORWARD SAMPLE"))
    print("=" * 78)
    print("window %s .. %s" % (res["window_since"], res["window_until"]))
    print("ticks %s (dropped %s), span %sh, median gap %ss"
          % (res.get("ticks"), res.get("ticks_dropped_mixed_denomination"),
             res.get("tape_span_hours"), res.get("median_gap_sec")))
    print("cost used %.4fbp (%s) = %.4fbp quoted + %.2fbp gas"
          % (res["cost_used_bp"], res["cost_used_key"],
             AERO_QUOTE_READINGS_BP[VERDICT_COST_KEY], GAS_BP))
    print("cost band  FLOOR %.2f / CENSUS %.2f / MEDIAN %.2f bp"
          % (AERO_COSTS_BP["FLOOR"], AERO_COSTS_BP["CENSUS"], AERO_COSTS_BP["MEDIAN"]))
    print("$10 clip   %s" % AERO_TEN_DOLLAR_CLIP["status"])
    print("")
    if "verdict" in res:
        print(res["verdict"])
    for name, blocks in res.get("signals", {}).items():
        mark = "  <== THE PRE-REGISTERED CELL" if name == VERDICT_SIGNAL else ""
        print("-- %s%s" % (name, mark))
        for h, b in blocks.items():
            if not b.get("n_long"):
                print("   %-4s fires %-4d paired-long 0 (unpairable %d) -- no sample"
                      % (h, b.get("fires", 0), b.get("n_unpairable", 0)))
                continue
            print("   %-4s n=%-4d acc %.4f  h* %.4f  p %.5f  mde %.2fpp  move %.2fbp  net %+.2fbp  %s"
                  % (h, b["n_long"], b["long_accuracy"], b["h_star_required_accuracy"],
                     b["long_p_one_sided_vs_coin"], b["long_mde_pp_at_95"],
                     b["mean_abs_move_bp"], b["long_net_bp"],
                     "CLEARS" if b["long_clears_h_star"] else "fails"))
            print("        h* at FLOOR/CENSUS/MEDIAN cost: %s" % b.get("h_star_sensitivity"))
            print("        OVERLAPPING acc %s over n=%s -- POINT ESTIMATE ONLY"
                  % (b.get("OVERLAPPING_long_accuracy_point_estimate_only"),
                     b.get("OVERLAPPING_n_long")))
            if b.get("split_n"):
                print("        split-half %s / %s over n=%s -- STABILITY, not significance"
                      % (b.get("split_first_half_accuracy"),
                         b.get("split_second_half_accuracy"), b.get("split_n")))
        print("")
    print("=" * 78)
    for line in verdict(res, in_window):
        print(line)
    print("=" * 78)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
