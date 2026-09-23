"""Out-of-sample check on [0fcc938d]: the deep tier minus AERO, +40.1bp on 13 rows.

Pass 429 (Echo) sliced the >$1M ghost tier by symbol, dropped AERO after looking
at the data, and was left with 13 CBBTC/CBXRP rows at +40.1bp, 95% CI
[+7.5, +72.7] -- the first interval in that pass that excluded zero. It was
filed as a TRAP, not a lead, with a pre-registered bar: 13 or more NEW rows
dated after 2026-09-16, no further exclusions, and the mean must stay above
+7.5bp (the in-sample lower bound) to count.

This script answers the item three ways, because the fresh sample turned out to
be too small to answer it the intended way:

  1. FEASIBILITY. How many fresh rows exist, and -- per the standing note that a
     short window on a slow process reads as death -- what the arrival rate says
     we should have expected. A zero is only evidence if the expected count was
     not also near zero.
  2. CONCENTRATION. The in-sample 13: what share of gross is the single largest
     row? Criterion 3 asks this of the out-of-sample set, but asking it of the
     in-sample set is what tells us whether +40.1bp was ever more than one row.
  3. MULTIPLICITY. Criterion 4 asks for the selection bar implied by the number
     of slices taken. That is arithmetic on the published interval and needs no
     fresh data at all.

Read-only. Touches no gate, no threshold and no production file.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sqlite3

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
SYMBOLS = ("CBBTC-USDC", "CBXRP-USDC")
CUTOFF = dt.datetime(2026, 9, 16, tzinfo=dt.timezone.utc).timestamp()
# The item's censor rule, unchanged: |move| <= 20%. It is a PRIVATE COPY of
# the ratio arm of services.outcome_plausibility, set TIGHTER than the shared
# 50% bar and carrying no dollar arm. Measured 2026-09-22 over the 243 ghost
# rows in trade_outcomes: this rule keeps 234, and of those the SHARED rule
# refuses ZERO -- so it is strictly stronger here and the gap is empty. The
# shared rule alone refuses 6 of 243. Stated rather than assumed, because
# "my filter is tighter" is exactly the claim that goes stale silently.
CENSOR = 0.20
TARGET_N = 13
IN_SAMPLE_LOWER = 7.5  # bp; the bar the out-of-sample mean must clear


def _rows(cur, symbol):
    """Every ghost round trip for `symbol`, censored, as (ts, gross_bp, gross)."""
    cur.execute(
        "SELECT ts, entry_price, exit_price, quantity, gross_profit, details "
        "FROM trade_outcomes WHERE symbol=? ORDER BY ts",
        (symbol,),
    )
    out, dropped = [], 0
    for ts, entry, exit_, qty, gross, details in cur.fetchall():
        try:
            mode = json.loads(details or "{}").get("mode")
        except (ValueError, TypeError):
            mode = None
        if mode != "ghost":
            continue
        if not entry or not qty:
            continue
        if abs((exit_ - entry) / entry) > CENSOR:
            dropped += 1
            continue
        notional = entry * qty
        if notional <= 0:
            continue
        out.append((ts, gross / notional * 1e4, gross))
    return out, dropped


def _censor_banner(cur):
    """Lines naming, AT RUNTIME, what the private CENSOR rule left uncensored.

    The comment on CENSOR above states the gap as a measurement taken once.
    A stated gap goes stale silently, so this recomputes it every run against
    ``services.outcome_plausibility`` -- the shared rule graduation reads --
    over exactly the rows this report draws on.
    """
    cur.execute(
        "SELECT ts, symbol, entry_price, exit_price, quantity, gross_profit, "
        "net_profit, details FROM trade_outcomes WHERE symbol IN (%s)"
        % ",".join("?" * len(SYMBOLS)),
        SYMBOLS,
    )
    rows, private_dropped = [], 0
    for ts, sym, entry, exit_, qty, gross, net, details in cur.fetchall():
        try:
            d = json.loads(details or "{}")
        except (ValueError, TypeError):
            d = {}
        if d.get("mode") != "ghost":
            continue
        row = {
            "ts": ts, "symbol": sym, "entry_price": entry,
            "exit_price": exit_, "quantity": qty, "gross_profit": gross,
            "net_profit": net, "strategy_id": d.get("strategy_id") or "",
        }
        if entry and qty and abs((exit_ - entry) / entry) > CENSOR:
            private_dropped += 1
            continue
        rows.append(row)

    out = ["  PRIVATE CENSOR (|move| <= %.0f%%): refused %d of %d ghost rows on "
           "%s." % (CENSOR * 100, private_dropped, private_dropped + len(rows),
                    "/".join(SYMBOLS))]
    try:
        import sys  # noqa: PLC0415
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        from services import outcome_plausibility as op  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        out.append("  !! UNCENSORED BY THE SHARED RULE and the gap is UNKNOWN: "
                   "services.outcome_plausibility did not import (%s). The "
                   "%d rows below carry whatever it would have refused."
                   % (exc, len(rows)))
        return out
    out.extend("  " + ln for ln in op.banner(rows, censored=True))
    return out


def _ci(values):
    """Mean and 95% half-width. Returns (mean, half_width) or (mean, None)."""
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * math.sqrt(var / n)


def _z_for(k):
    """Two-sided normal quantile at a Bonferroni-corrected 5% over k slices."""
    # inverse normal via bisection; no scipy in this tree
    target = 1.0 - (0.05 / k) / 2.0
    lo, hi = 0.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def main():
    db = sqlite3.connect(DB)
    cur = db.cursor()
    now = dt.datetime.now(dt.timezone.utc).timestamp()

    print("=" * 74)
    print("[0fcc938d] deep tier minus AERO -- out-of-sample check")
    print("=" * 74)

    print("\n--- 0. WHAT THIS REPORT DID NOT CENSOR ---")
    for line in _censor_banner(cur):
        print(line)

    # ---- 1. feasibility -------------------------------------------------
    print("\n--- 1. IS THERE A FRESH SAMPLE? ---")
    print("  a zero only means 'stopped' if the EXPECTED count was not also small.")
    window_days = (now - CUTOFF) / 86400.0
    print("  window since 2026-09-16: %.2f days\n" % window_days)
    fresh_all, in_sample_all = [], []
    print("  %-12s %5s %7s %9s %9s %9s" % ("symbol", "all", "rate/d", "expected", "observed", "verdict"))
    for sym in SYMBOLS:
        rows, _ = _rows(cur, sym)
        if not rows:
            print("  %-12s %5d %7s %9s %9s %9s" % (sym, 0, "-", "-", "-", "no ghost rows"))
            continue
        span = (rows[-1][0] - rows[0][0]) / 86400.0
        rate = len(rows) / span if span > 0 else 0.0
        fresh = [r for r in rows if r[0] >= CUTOFF]
        expected = rate * window_days
        fresh_all.extend(fresh)
        in_sample_all.extend([r for r in rows if r[0] < CUTOFF])
        # Poisson P(observe <= k) at the symbol's own historical rate
        k = len(fresh)
        p = sum(math.exp(-expected) * expected ** i / math.factorial(i) for i in range(k + 1))
        verdict = "STOPPED" if p < 0.05 else "normal"
        print("  %-12s %5d %7.2f %9.2f %9d   p=%.4f %s" % (sym, len(rows), rate, expected, k, p, verdict))

    print("\n  fresh rows available: %d   /  %d required" % (len(fresh_all), TARGET_N))
    live_rate = 0.0
    for sym in SYMBOLS:
        rows, _ = _rows(cur, sym)
        if rows and rows[-1][0] >= CUTOFF:  # only symbols still producing
            span = (rows[-1][0] - rows[0][0]) / 86400.0
            live_rate += len(rows) / span if span > 0 else 0.0
    if live_rate > 0:
        print("  surviving arrival rate: %.2f/day -> %.1f days to reach %d rows"
              % (live_rate, TARGET_N / live_rate, TARGET_N))
    if len(fresh_all) < TARGET_N:
        print("  => CRITERION 1 CANNOT BE MET TODAY. The item is not refuted by")
        print("     the fresh sample; it is UNTESTABLE on it.")

    # ---- 2. concentration of the in-sample number -----------------------
    print("\n--- 2. WAS THE IN-SAMPLE +40.1bp MORE THAN ONE ROW? ---")
    print("  criterion 3 asks this of the out-of-sample set. Asked of the")
    print("  IN-SAMPLE set it tests whether the number was ever real.")
    for sym in SYMBOLS:
        rows, _ = _rows(cur, sym)
        ins = [r for r in rows if r[0] < CUTOFF]
        if not ins:
            continue
        mean, half = _ci([r[1] for r in ins])
        pos = [r[2] for r in ins if r[2] > 0]
        top_share = (max(pos) / sum(pos) * 100.0) if pos else float("nan")
        ci = "[%+7.1f, %+7.1f]" % (mean - half, mean + half) if half else "(n<2)"
        flag = "  <-- ONE ROW" if top_share > 50.0 else ""
        print("  %-12s n=%-3d  mean %+7.1fbp  %s  top-1 share of gross %5.1f%%%s"
              % (sym, len(ins), mean, ci, top_share, flag))
    if in_sample_all:
        mean, half = _ci([r[1] for r in in_sample_all])
        pos = [r[2] for r in in_sample_all if r[2] > 0]
        top_share = (max(pos) / sum(pos) * 100.0) if pos else float("nan")
        ci = "[%+7.1f, %+7.1f]" % (mean - half, mean + half) if half else "(n<2)"
        print("  %-12s n=%-3d  mean %+7.1fbp  %s  top-1 share of gross %5.1f%%"
              % ("POOLED", len(in_sample_all), mean, ci, top_share))

    # ---- 3. the selection bar the item demands --------------------------
    print("\n--- 3. THE SELECTION BAR (criterion 4) ---")
    print("  published in-sample: +40.1bp, 95%% CI [+7.5, +72.7], n=13.")
    print("  half-width 32.6bp at z=1.96 => SE of the mean = 16.6bp.")
    print("  slices taken to reach it: 5 depth tiers, then 3 symbols inside the")
    print("  deep tier, then 1 exclusion (drop AERO) = 9 reported cells.\n")
    se = 32.6 / 1.96
    print("  %6s %8s %10s %26s %s" % ("slices", "z", "half-width", "corrected 95% interval", "excludes 0?"))
    for k in (1, 3, 5, 9):
        z = _z_for(k)
        hw = z * se
        lo, hi = 40.1 - hw, 40.1 + hw
        print("  %6d %8.3f %9.1fbp   [%+7.1f, %+7.1f]        %s"
              % (k, z, hw, lo, hi, "yes" if lo > 0 else "NO"))
    print("\n  The honest slice count is at least 5: five depth tiers were")
    print("  screened before any symbol was looked at. At 5 slices the interval")
    print("  already contains zero, and at the full 9 it is not close.")

    db.close()


if __name__ == "__main__":
    main()
