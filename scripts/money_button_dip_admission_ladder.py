#!/usr/bin/env python3
"""
The money button's FULL admission ladder, in production's own order, counted
tick by tick -- and split by pool depth at the end.

WHY THIS EXISTS, and why it is not scripts/money_button_dip_supply.py.
That script (pass 426) produced the number this repo has been quoting:
"3,049 admissible dip ticks". It has two defects that change the answer.

  1. IT RAN A DIFFERENT RULE THAN PRODUCTION. It reads its thresholds with
     ``os.getenv(name, default)`` and never imports ``services``. Production
     imports ``services.env_loader``, whose tail runs
     ``os.environ.setdefault(key, value)`` over a defaults table that contains
     MONEY_BUTTON_MIN_DRAWDOWN = "0.005" (env_loader.py:377). There is no
     MONEY_BUTTON_* key in .env, so the ONLY thing that sets the production
     value is that import. Pass 426 therefore measured a 2%-5% band; the
     scheduler runs a 0.5%-5% band -- a 4x tighter minimum. This module
     imports services.env_loader FIRST, before reading any threshold.

  2. IT STOPPED AT THE TWO PRICE-ACTION GATES, AND MISORDERED THEM. Its
     docstring says the cost and net-expected-return gates "sit downstream of
     the slope check". They do not. In ``trading/scheduler.py
     ._money_button_candidate`` the order is:

         drawdown band        -> drawdown_outside_dip_window
         reversion headroom   -> no_reversion_headroom
         NET EXPECTED RETURN  -> net_expected_below_min_net_return
         slope / knife guard  -> slope_below_knife_floor
         take-profit vs fee   -> take_profit_below_min_net_return

     The net-expected gate is UPSTREAM of the knife guard. So the knife
     guard in production never sees the population pass 426 measured it
     against, and "the knife guard is a 38.1% tax on dip supply" is a
     statistic about a population that does not reach it.

WHAT IT MEASURES. Every tick in market_stream that carries a full lookback of
history, pushed through the ladder in production's order, reporting the
survivor count after each rung. Then the survivors are split by pool depth.

THE ARITHMETIC WORTH KNOWING BEFORE READING THE OUTPUT. The reversion target
is capped at the 2h peak, so expected_return can never exceed
(peak-last)/last = |dd|/(1-|dd|). The net gate demands
expected_return >= fee_rate + min_net_return. With production's constants that
is 0.0125 + 0.005 = 0.0175, which needs |dd| >= ~1.72%. Any dip shallower than
that is refused BY ARITHMETIC, whatever the band says. That is testable and
this script tests it rather than asserting it.

PER-TICK DEPTH IS NOT AVAILABLE, and this is the honest statement the item
asks for rather than a silent snapshot. market_stream carries
(id, ts, chain, symbol, price, volume, raw); there is no liquidity column, the
``raw`` JSON carries only {ts, symbol, chain, price, volume, rest,
consensus_confidence}, and volume is 0.0 on the rows sampled. The store's only
other depth-shaped tables, ohlcv_bars and ohlcv_datasets, are EMPTY (0 rows).
So depth can only come from a point-in-time snapshot
(data/pool_depth_snapshot.json), and attributing today's depth to a tick from
14 days ago is an INFERENCE, not a measurement. Every depth-split number below
is labelled accordingly.

NO GATE IS CHANGED BY THIS SCRIPT. It opens the store read-only.
"""

from __future__ import annotations

# Import services.env_loader BEFORE reading any threshold: it is the only
# thing that sets production's MONEY_BUTTON_MIN_DRAWDOWN. See the docstring.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_ENV_SOURCE = "NOT LOADED -- thresholds are literal defaults"
try:
    import services.env_loader  # noqa: F401
    _ENV_SOURCE = "services.env_loader imported (production defaults applied)"
except Exception as exc:  # noqa: BLE001
    _ENV_SOURCE = "services.env_loader FAILED (%s)" % exc

import argparse
import json
import os
import sqlite3
from collections import defaultdict

import numpy as np


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:  # noqa: BLE001
        return default


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--depth-json", default=str(ROOT / "data" / "pool_depth_snapshot.json"))
    ap.add_argument("--depth-bar", type=float, default=1_000_000.0,
                    help="pool liquidity USD at/above which a dip is called fillable")
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--force-legacy-band", action="store_true",
                    help="re-run pass 426's 2%%-5%% band for a back-to-back comparison")
    args = ap.parse_args()

    path = args.db or os.environ.get(
        "TRADING_DB_PATH", str(ROOT / "storage" / "trading_cache.db"))
    if not Path(path).exists():
        print("dip ladder: no trading store at %s" % path)
        return 2

    lookback = _env_float("MONEY_BUTTON_LOOKBACK_SEC", 7200.0)
    lookback = max(600.0, min(24 * 3600.0, lookback))
    slope_window = _env_float("MONEY_BUTTON_SLOPE_WINDOW_SEC", 1800.0)
    slope_window = max(300.0, min(4 * 3600.0, slope_window))
    slope_floor = _env_float("MONEY_BUTTON_SLOPE_FLOOR", -0.00025)
    min_dd = _env_float("MONEY_BUTTON_MIN_DRAWDOWN", 0.02)
    if args.force_legacy_band:
        min_dd = 0.02
    max_dd = _env_float("MONEY_BUTTON_MAX_DRAWDOWN", 0.05)
    min_dd = max(0.0, min(min_dd, 0.25))
    max_dd = max(min_dd, min(max_dd, 0.35))
    min_net = _env_float("MONEY_BUTTON_MIN_NET_RETURN", 0.005)
    reversion_fraction = _env_float("MONEY_BUTTON_REVERSION_FRACTION", 0.55)
    reversion_fraction = max(0.1, min(reversion_fraction, 1.0))

    # scheduler.py:806 -- the one cost number every money-button gate uses.
    fee_buffer = _env_float("SCHEDULER_FEE_BUFFER", 0.002)
    tax_buffer = _env_float("SCHEDULER_TAX_BUFFER", 0.003)
    gas_rt = _env_float("SCHEDULER_GAS_ROUNDTRIP_RATIO",
                        _env_float("GAS_ROUNDTRIP_FEE_RATIO", 0.0025))
    slip_bps = _env_float("SCHEDULER_SLIPPAGE_BPS", 50.0)
    fee_rate = fee_buffer + tax_buffer + gas_rt + (slip_bps / 10000.0)

    conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    cur = conn.cursor()
    cur.execute("select symbol, ts, price from market_stream where price > 0 order by symbol, ts")
    tapes: dict[str, list] = defaultdict(list)
    for sym, ts, price in cur.fetchall():
        tapes[sym].append((float(ts), float(price)))
    conn.close()

    # Ladder rungs, in production's order.
    n_eval = 0
    n_few_samples = 0
    n_dd_pass = 0
    n_no_headroom = 0
    n_net_pass = 0
    n_knife_kill = 0
    n_slope_pass = 0
    n_tp_pass = 0
    # per symbol: [dd_pass, net_pass, final_pass]
    per_sym: dict[str, list] = defaultdict(lambda: [0, 0, 0])
    dd_of_net_pass: list = []
    dd_of_dd_pass: list = []

    for sym, rows in tapes.items():
        if len(rows) < 8:
            continue
        ts = np.array([r[0] for r in rows])
        px = np.array([r[1] for r in rows])
        for i in range(len(rows)):
            t = ts[i]
            if t - ts[0] < lookback:
                continue  # not enough history for an honest drawdown
            n_eval += 1
            sel = (ts >= t - lookback) & (ts <= t)
            window = px[sel]
            if window.size < 8:  # production: fewer_than_8_samples_in_lookback
                n_few_samples += 1
                continue
            last = px[i]
            peak = window.max()
            if peak <= 0:
                continue
            drawdown = (last - peak) / peak
            if drawdown > -min_dd or drawdown < -max_dd:
                continue
            n_dd_pass += 1
            per_sym[sym][0] += 1
            dd_of_dd_pass.append(-drawdown)

            median_price = float(np.median(window))
            reversion_target = min(max(median_price, last), peak)
            expected_return = (reversion_target - last) / max(last, 1e-9)
            if expected_return <= 0.0:
                n_no_headroom += 1
                continue
            net_expected = expected_return - fee_rate
            if net_expected < min_net:
                continue
            n_net_pass += 1
            per_sym[sym][1] += 1
            dd_of_net_pass.append(-drawdown)

            smask = (ts >= t - slope_window) & (ts <= t)
            st, sp = ts[smask], px[smask]
            if st.size >= 6:
                rel_minutes = (st - st[-1]) / 60.0
                if not np.allclose(rel_minutes, rel_minutes[0]):
                    slope, _ = np.polyfit(rel_minutes, np.log(np.clip(sp, 1e-9, 1e9)), 1)
                    if slope < slope_floor:
                        n_knife_kill += 1
                        continue
            n_slope_pass += 1

            tp = max(fee_rate + min_net, abs(drawdown) * reversion_fraction)
            tp = min(tp, expected_return)
            if tp - fee_rate < min_net:
                continue
            n_tp_pass += 1
            per_sym[sym][2] += 1

    W = 76
    print("=" * W)
    print("MONEY BUTTON -- FULL ADMISSION LADDER (production order)")
    print("=" * W)
    print("store : %s" % path)
    print("env   : %s" % _ENV_SOURCE)
    print("band  : drawdown %.3f%%-%.2f%% over %.0fs" % (100 * min_dd, 100 * max_dd, lookback))
    print("        (pass 426 measured %.2f%%-%.2f%% -- see docstring)" % (2.0, 100 * max_dd))
    print("cost  : fee_rate %.4f = fee %.4f + tax %.4f + gas %.4f + slip %.1fbp"
          % (fee_rate, fee_buffer, tax_buffer, gas_rt, slip_bps))
    print("        min_net_return %.4f  ->  expected_return must reach %.4f (%.2f%%)"
          % (min_net, fee_rate + min_net, 100 * (fee_rate + min_net)))
    print("knife : slope floor %.5f/min over %.0fs" % (slope_floor, slope_window))
    print()
    print("  rung                                       survivors      refused")
    print("  " + "-" * (W - 4))
    print("  0 ticks with a full %.0fh lookback       %11d" % (lookback / 3600.0, n_eval))
    print("  1 >=8 samples in lookback                %11d  %11d"
          % (n_eval - n_few_samples, n_few_samples))
    print("  2 drawdown inside the dip band           %11d  %11d"
          % (n_dd_pass, (n_eval - n_few_samples) - n_dd_pass))
    print("  3 reversion headroom > 0                 %11d  %11d"
          % (n_dd_pass - n_no_headroom, n_no_headroom))
    print("  4 NET EXPECTED >= min_net_return         %11d  %11d"
          % (n_net_pass, (n_dd_pass - n_no_headroom) - n_net_pass))
    print("  5 slope above the knife floor            %11d  %11d"
          % (n_slope_pass, n_knife_kill))
    print("  6 take-profit clears fee + min_net       %11d  %11d"
          % (n_tp_pass, n_slope_pass - n_tp_pass))
    print()
    if n_dd_pass:
        print("  the COST rung (4) refuses %.2f%% of everything the dip band admits"
              % (100.0 * ((n_dd_pass - n_no_headroom) - n_net_pass) / n_dd_pass))
    if n_net_pass:
        print("  the KNIFE rung (5) refuses %.2f%% of what actually REACHES it"
              % (100.0 * n_knife_kill / n_net_pass))
    else:
        print("  the KNIFE rung (5) never runs: nothing reaches it.")
    print()

    if dd_of_dd_pass:
        a = np.array(dd_of_dd_pass)
        print("  drawdown depth of band-admitted ticks : median %.3f%%  p90 %.3f%%  max %.3f%%"
              % (100 * np.median(a), 100 * np.percentile(a, 90), 100 * a.max()))
        floor_needed = (fee_rate + min_net) / (1.0 + fee_rate + min_net)
        print("  arithmetic floor from the cost gate   : |dd| >= %.3f%%" % (100 * floor_needed))
        print("  band-admitted ticks BELOW that floor  : %d of %d (%.2f%%) -- refusable by arithmetic alone"
              % (int((a < floor_needed).sum()), a.size, 100.0 * (a < floor_needed).sum() / a.size))
    if dd_of_net_pass:
        b = np.array(dd_of_net_pass)
        print("  drawdown depth of cost-gate survivors : median %.3f%%  min %.3f%%"
              % (100 * np.median(b), 100 * b.min()))
    print()

    depth = {}
    dpath = Path(args.depth_json) if args.depth_json else None
    if dpath and dpath.exists():
        try:
            depth = {k: v for k, v in json.loads(dpath.read_text(encoding="utf-8")).items()
                     if not k.startswith("_")}
        except Exception:  # noqa: BLE001
            depth = {}

    print("DIP SUPPLY BY SYMBOL -- band-admitted / cost-survivor / fully admissible:")
    print()
    print("    %-18s %7s %9s %9s  %14s" % ("symbol", "band", "post-cost", "admissible", "pool liq USD"))
    fill_band = fill_net = fill_final = 0
    unresolved = 0
    for sym, (n_band, n_net, n_fin) in sorted(per_sym.items(), key=lambda kv: -kv[1][0])[: args.top]:
        base = sym.split("-")[0].upper()
        liq = depth.get(base, depth.get(sym))
        if isinstance(liq, dict):
            liq = liq.get("liquidity_usd")
        shown = ("%.0f" % liq) if isinstance(liq, (int, float)) else "unresolved"
        print("    %-18s %7d %9d %9d  %14s" % (sym, n_band, n_net, n_fin, shown))
    for sym, (n_band, n_net, n_fin) in per_sym.items():
        base = sym.split("-")[0].upper()
        liq = depth.get(base, depth.get(sym))
        if isinstance(liq, dict):
            liq = liq.get("liquidity_usd")
        if not isinstance(liq, (int, float)):
            unresolved += 1
            continue
        if liq >= args.depth_bar:
            fill_band += n_band
            fill_net += n_net
            fill_final += n_fin
    if depth:
        print()
        print("  DEPTH BAR = $%s (deepest base pool per symbol, DexScreener snapshot)"
              % format(int(args.depth_bar), ","))
        print("    band-admitted dips in pools at/above the bar : %d of %d" % (fill_band, n_dd_pass))
        print("    ...that also clear the COST gate             : %d of %d" % (fill_net, n_net_pass))
        print("    ...fully admissible (all six rungs)          : %d of %d" % (fill_final, n_tp_pass))
        print("    symbols with unresolved depth (not counted)  : %d" % unresolved)
        print()
        print("  DEPTH IS A SNAPSHOT, NOT A PER-TICK MEASUREMENT. market_stream has no")
        print("  liquidity column and ohlcv_bars is empty, so a tick's depth is inferred")
        print("  from today's pool. A pool that drained or filled makes this wrong in an")
        print("  unknown direction. Treat the split as indicative, not measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
