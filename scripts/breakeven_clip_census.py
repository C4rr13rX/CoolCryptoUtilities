"""Per-symbol break-even clip: the clip size at which measured gross pays measured cost.

Bay, pass 720. Read-only against storage/trading_cache.db.

The cost model this sprint measured (Dune p719, edge_estimate.py over 304 closed
round trips) is NOT a flat rate -- it decomposes into a fixed dollar term and a
variable rate::

    cost(N) = FIXED_USD + VAR_RATE * N

Because the fixed term is dollars and the gross edge is a rate, there is a clip
N* at which a symbol starts paying for itself::

    gross_rate * N = FIXED_USD + VAR_RATE * N
    N* = FIXED_USD / (gross_rate - VAR_RATE)

and N* exists at all ONLY when ``gross_rate > VAR_RATE``. That inequality is the
whole story: no clip, however large, rescues a symbol whose gross rate is below
the variable cost rate. Raising the clip only ever amortises the fixed term.

The leave-one-out column is not decoration. A mean over n=7 carried by one row
is not an edge, and this repo has shipped that mistake before (98% of gross on
7 rows). ``--loo`` reports the worst single-row-removed mean and recomputes N*
from THAT, so a symbol only claims a break-even clip if it holds up when its
best row is deleted.

Usage::

    python scripts/breakeven_clip_census.py
    python scripts/breakeven_clip_census.py --min-n 5 --bankroll 23.18
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import outcome_plausibility as op  # noqa: E402

# Dune p719, .env:371 / edge_estimate.py, 304 closed round trips.
DEFAULT_FIXED_USD = 0.008367
DEFAULT_VAR_RATE = 0.003187

DEFAULT_DB = os.environ.get(
    "TRADING_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "storage", "trading_cache.db"),
)


def load_returns(db_path, censor=True):
    """Return {symbol: [gross_return_rate, ...]} over closed round trips.

    The rate is ``gross_profit / notional`` with ``notional = entry_price *
    quantity`` -- gross_profit is USD, so it has to be divided by the money at
    risk to be comparable to a cost RATE. Rows with a non-positive notional or a
    null gross are dropped rather than coerced to zero.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM trade_outcomes WHERE status = 'closed'")]
    conn.close()

    censored = 0
    if censor:
        kept, dropped = op.partition(rows)
        censored = len(dropped)
        rows = kept

    by_symbol = defaultdict(list)
    skipped = 0
    for row in rows:
        notional = (row.get("entry_price") or 0.0) * (row.get("quantity") or 0.0)
        gross = row.get("gross_profit")
        if notional <= 0 or gross is None:
            skipped += 1
            continue
        by_symbol[row["symbol"]].append(gross / notional)
    return by_symbol, censored, skipped


def breakeven(gross_rate, fixed_usd, var_rate):
    """Clip at which gross pays cost, or None when no clip ever does."""
    if gross_rate <= var_rate:
        return None
    return fixed_usd / (gross_rate - var_rate)


def worst_leave_one_out(values):
    """Lowest mean obtainable by deleting exactly one observation."""
    if len(values) < 2:
        return statistics.fmean(values)
    total = sum(values)
    n = len(values)
    # Deleting the LARGEST value gives the lowest mean.
    return (total - max(values)) / (n - 1)


def live_counterfactual(db_path, clips=(0.75, 6.00, 23.18), censor=True):
    """Hold each live round trip's measured gross RATE, vary only the clip.

    Gas is a fixed dollar cost, so it does NOT scale with the clip -- that is
    the entire argument for sizing up. This function is here to show why the
    argument fails anyway: the gross rate scales too, and a book whose gross is
    negative gets more negative. The BSTONK-excluded panel is not optional
    colour, it is the point -- one row of eighteen decides the sign.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM trade_outcomes WHERE status = 'closed' AND wallet = 'live'")]
    conn.close()
    if censor:
        rows, _ = op.partition(rows)

    trips = []
    for row in rows:
        notional = (row.get("entry_price") or 0.0) * (row.get("quantity") or 0.0)
        gross = row.get("gross_profit")
        if notional <= 0 or gross is None:
            continue
        trips.append((row["symbol"], gross / notional, notional, gross,
                      row.get("fee_cost") or 0.0))

    fee = sum(t[4] for t in trips)
    gross = sum(t[3] for t in trips)
    print(f"ACTUAL   n={len(trips)}  gross {gross:+.6f}  fee {fee:.6f}  "
          f"net {gross - fee:+.6f}   <- the scoreboard")
    med_notional = statistics.median([t[2] for t in trips])
    print(f"mean fee per round trip ${fee / len(trips):.6f}  "
          f"median notional ${med_notional:.2f}  "
          f"gas = {(fee / len(trips)) / med_notional * 100:.4f}% of notional at that clip")
    print()

    for label, subset in (("ALL live rows", trips),
                          ("WITHOUT the single BSTONK-USDC row",
                           [t for t in trips if t[0] != "BSTONK-USDC"])):
        sub_fee = sum(t[4] for t in subset)
        print(f"{label}  (n={len(subset)}, fee ${sub_fee:.6f} held FIXED):")
        for clip in clips:
            g = sum(t[1] * clip for t in subset)
            print(f"    all @ ${clip:6.2f} clip : gross {g:+9.6f}  net {g - sub_fee:+9.6f}")
        print()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--live-counterfactual", action="store_true",
                    help="vary the clip on the 18 chain-settled live round trips")
    ap.add_argument("--min-n", type=int, default=5,
                    help="skip symbols with fewer closed round trips")
    ap.add_argument("--fixed-usd", type=float, default=DEFAULT_FIXED_USD)
    ap.add_argument("--var-rate", type=float, default=DEFAULT_VAR_RATE)
    ap.add_argument("--bankroll", type=float, default=23.18,
                    help="deployable stable, for the fundable/not-fundable verdict")
    ap.add_argument("--no-censor", action="store_true",
                    help="skip services.outcome_plausibility (NOT recommended)")
    args = ap.parse_args(argv)

    if args.live_counterfactual:
        return live_counterfactual(args.db, censor=not args.no_censor)

    by_symbol, censored, skipped = load_returns(args.db, censor=not args.no_censor)
    print(f"db                : {args.db}")
    print(f"cost model        : ${args.fixed_usd:.6f} fixed + {args.var_rate * 100:.4f}% of notional")
    print(f"plausibility      : {'OFF' if args.no_censor else 'ON'} "
          f"({censored} row(s) censored, {skipped} row(s) unusable)")
    print()

    header = (f"{'symbol':<15}{'n':>4}{'gross%':>10}{'pos':>7}"
              f"{'N*':>9}{'LOO-worst%':>12}{'N* @LOO':>10}  verdict")
    print(header)
    print("-" * len(header))

    survivors = []
    below_var = 0
    for symbol, values in sorted(by_symbol.items(),
                                 key=lambda kv: -statistics.fmean(kv[1])):
        if len(values) < args.min_n:
            continue
        mean = statistics.fmean(values)
        pos = sum(1 for v in values if v > 0)
        loo = worst_leave_one_out(values)
        n_star = breakeven(mean, args.fixed_usd, args.var_rate)
        n_star_loo = breakeven(loo, args.fixed_usd, args.var_rate)

        if mean <= args.var_rate:
            below_var += 1

        if n_star_loo is None:
            verdict = "NO CLIP EVER PAYS"
        elif n_star_loo <= args.bankroll:
            verdict = f"fundable at ${n_star_loo:.2f}"
            survivors.append((symbol, len(values), loo, n_star_loo))
        else:
            verdict = f"needs ${n_star_loo:.2f} > bankroll"

        print(f"{symbol:<15}{len(values):>4}{mean * 100:>10.4f}"
              f"{f'{pos}/{len(values)}':>7}"
              f"{(f'{n_star:.2f}' if n_star else 'never'):>9}"
              f"{loo * 100:>12.4f}"
              f"{(f'{n_star_loo:.2f}' if n_star_loo else 'never'):>10}  {verdict}")

    counted = [s for s, v in by_symbol.items() if len(v) >= args.min_n]
    print()
    print(f"symbols with n >= {args.min_n}                       : {len(counted)}")
    print(f"  ... whose gross rate is BELOW the variable    : {below_var}"
          f"   <- no clip, however large, makes these pay")
    print(f"  ... that survive leave-one-out AND are fundable: {len(survivors)}")
    for symbol, n, loo, n_star_loo in survivors:
        print(f"        {symbol:<14} n={n:<4} LOO gross {loo * 100:.4f}%  "
              f"break-even clip ${n_star_loo:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
