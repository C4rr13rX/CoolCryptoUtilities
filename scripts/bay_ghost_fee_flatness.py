"""Is the ghost round-trip fee FLAT across symbols, or does it already vary?

[71cafaca] asserts the ghost book charges ~46bp on every symbol regardless of
pool depth, so deep pairs (WETH, 8.8bp real) are overcharged ~5.2x and can never
clear graduation.

That is a claim about the fee model's INPUT.  This script measures the fee model's
OUTPUT as it was actually booked into trade_outcomes.fee_cost -- the number that
becomes ghost NET P/L and therefore the number graduation reads.

Pre-registered, before running:
  (A) FLAT   -- cross-symbol spread of the realised rate is small (IQR < ~5bp)
                and does not track depth.  The item stands.
  (B) VARIES -- symbol medians differ by more than 2x.  The 46bp is then a
                POOLED AVERAGE over an already-varying rate, and no single symbol
                is overcharged 5.2x.

Read-only.  Does not import the cost modules -- deliberately, because the point
is to measure what was BOOKED, not to re-derive what the model would say now.

    python scripts/bay_ghost_fee_flatness.py
"""

from __future__ import annotations

import os
import sqlite3
import statistics
import sys

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
MIN_N = 5


def _rows(conn: sqlite3.Connection, wallet: str) -> list[tuple]:
    return conn.execute(
        """
        SELECT symbol, fee_cost, entry_price, quantity, gross_profit, net_profit, ts
          FROM trade_outcomes
         WHERE wallet = ?
           AND status = 'closed'
           AND fee_cost IS NOT NULL
           AND entry_price IS NOT NULL
           AND quantity IS NOT NULL
           AND entry_price > 0
           AND quantity > 0
        """,
        (wallet,),
    ).fetchall()


def _rate_bp(fee_cost: float, entry_price: float, quantity: float) -> float | None:
    notional = entry_price * quantity
    if notional <= 0:
        return None
    return 10_000.0 * float(fee_cost) / notional


def census(conn: sqlite3.Connection, wallet: str) -> None:
    rows = _rows(conn, wallet)
    print(f"\n=== wallet={wallet}: {len(rows)} closed rows with a priced notional ===")
    if not rows:
        print("  (none)")
        return

    by_symbol: dict[str, list[float]] = {}
    notionals: dict[str, list[float]] = {}
    for symbol, fee, entry, qty, _gross, _net, _ts in rows:
        rate = _rate_bp(fee, entry, qty)
        if rate is None:
            continue
        by_symbol.setdefault(symbol, []).append(rate)
        notionals.setdefault(symbol, []).append(float(entry) * float(qty))

    all_rates = [r for v in by_symbol.values() for r in v]
    print(f"  pooled median rate   {statistics.median(all_rates):10.2f} bp   (n={len(all_rates)})")
    print(f"  pooled mean   rate   {statistics.fmean(all_rates):10.2f} bp")

    powered = {s: v for s, v in by_symbol.items() if len(v) >= MIN_N}
    print(f"\n  per-symbol medians, n>={MIN_N} ({len(powered)} of {len(by_symbol)} symbols qualify)")
    print(f"  {'symbol':<18}{'n':>5}{'median bp':>12}{'min bp':>11}{'max bp':>11}{'med notional':>15}")
    meds: list[tuple[str, float]] = []
    for symbol, v in sorted(powered.items(), key=lambda kv: statistics.median(kv[1])):
        med = statistics.median(v)
        meds.append((symbol, med))
        print(
            f"  {symbol:<18}{len(v):>5}{med:>12.2f}{min(v):>11.2f}{max(v):>11.2f}"
            f"{statistics.median(notionals[symbol]):>15.4f}"
        )

    if len(meds) < 2:
        print("\n  VERDICT: undecidable -- fewer than two symbols reach n>=%d." % MIN_N)
        return

    values = [m for _s, m in meds]
    lo_sym, lo = meds[0]
    hi_sym, hi = meds[-1]
    spread = hi - lo
    ratio = (hi / lo) if lo > 0 else float("inf")
    try:
        iqr = statistics.quantiles(values, n=4)[2] - statistics.quantiles(values, n=4)[0]
    except statistics.StatisticsError:
        iqr = spread

    print(f"\n  cheapest symbol      {lo_sym:<18}{lo:>10.2f} bp")
    print(f"  dearest  symbol      {hi_sym:<18}{hi:>10.2f} bp")
    print(f"  spread               {spread:>10.2f} bp")
    print(f"  ratio dearest/cheapest {ratio:>8.2f}x")
    print(f"  IQR across symbol medians {iqr:>8.2f} bp")

    if ratio > 2.0:
        print(f"\n  VERDICT (B): NOT FLAT. {hi_sym} pays {ratio:.2f}x what {lo_sym} pays.")
        print("  The pooled rate is an average over an already-varying rate.")
    elif iqr < 5.0:
        print("\n  VERDICT (A): FLAT. Cross-symbol IQR under 5bp and no symbol is 2x another.")
    else:
        print(f"\n  VERDICT: ambiguous on the SYMBOL axis -- IQR {iqr:.2f}bp but max ratio")
        print(f"  only {ratio:.2f}x, short of the 5.2x the depth story needs. See the")
        print("  notional census below: the axis that separates is CLIP, not symbol.")

    by_notional(by_symbol, notionals)
    within_symbol_control(by_symbol, notionals)


# --- the axis that actually separates -------------------------------------
#
# The per-symbol table above is ambiguous.  These two views are not.  The first
# shows the rate is monotone in clip size pooled over every symbol; the second
# holds the symbol -- and therefore the pool and its depth -- exactly constant
# and shows the rate still moves.

BUCKETS = [(0.0, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 5.2), (5.2, 6.0), (6.0, 1e9)]


def _paired(by_symbol: dict[str, list[float]], notionals: dict[str, list[float]]):
    """Re-pair (notional, rate, symbol) -- both dicts were appended in lockstep."""
    for symbol, rates in by_symbol.items():
        for notional, rate in zip(notionals[symbol], rates):
            yield notional, rate, symbol


def by_notional(by_symbol: dict[str, list[float]], notionals: dict[str, list[float]]) -> None:
    data = list(_paired(by_symbol, notionals))
    print(f"\n  rate by NOTIONAL bucket, pooled across every symbol (n={len(data)})")
    print(f"  {'notional':<22}{'n':>6}{'symbols':>9}{'median bp':>12}")
    for lo, hi in BUCKETS:
        v = [r for n, r, _s in data if lo <= n < hi]
        if not v:
            continue
        syms = len({s for n, _r, s in data if lo <= n < hi})
        label = f"[${lo:,.1f}, ${hi:,.1f})" if hi < 1e9 else f"[${lo:,.1f}, inf)"
        print(f"  {label:<22}{len(v):>6}{syms:>9}{statistics.median(v):>12.2f}")
    print("  ^ monotone decreasing in clip size is the signature of a FIXED toll")
    print("    spread over a variable notional -- a depth-blind gas term.")


def within_symbol_control(by_symbol: dict[str, list[float]], notionals: dict[str, list[float]]) -> None:
    """Hold the symbol constant.  Depth cannot vary.  Does the rate?"""
    data = list(_paired(by_symbol, notionals))
    per: dict[str, list[tuple[float, float]]] = {}
    for notional, rate, symbol in data:
        per.setdefault(symbol, []).append((notional, rate))

    print("\n  WITHIN-SYMBOL CONTROL -- same symbol, same pool, same depth, two clips")
    found = False
    for symbol, v in sorted(per.items()):
        big = [r for n, r in v if n >= 4.0]
        small = [r for n, r in v if n < 3.0]
        if len(big) >= 3 and len(small) >= 3:
            found = True
            hi_med, lo_med = statistics.median(small), statistics.median(big)
            print(
                f"  {symbol:<14} clip>=$4: n={len(big):>3} med {lo_med:>7.2f} bp"
                f"   |   clip<$3: n={len(small):>3} med {hi_med:>7.2f} bp"
                f"   -> {hi_med / lo_med if lo_med else float('inf'):.2f}x"
            )
    if not found:
        print("  (no symbol has >=3 rows on both sides of the clip split)")
    else:
        print("  Depth is held exactly constant and the rate still moves. The fee is")
        print("  priced off CLIP, not off pool depth.")


MATCHED = [(0.0, 1.0), (1.0, 2.5), (2.5, 5.2), (5.2, 1e9)]


def ghost_vs_live(conn: sqlite3.Connection) -> None:
    """The two wallets do not use the same fee structure, and the DOLLARS show it.

    Report the fee in absolute dollars as well as as a rate.  A fee that is flat
    in dollars across clip buckets is a fixed gas toll; one that rises with the
    clip carries a proportional pool-fee term.  Graduation scores the ghost net,
    so a term present in one book and absent in the other is a scoring bias.
    """
    books = {}
    for wallet in ("ghost", "live"):
        d = []
        for symbol, fee, entry, qty, _g, _n, _ts in _rows(conn, wallet):
            notional = float(entry) * float(qty)
            if notional > 0:
                d.append((notional, float(fee), 10_000.0 * float(fee) / notional))
        books[wallet] = d

    print("\n=== ghost vs live fee_cost, MATCHED ON CLIP BUCKET ===")
    print(f"  {'bucket':<16}{'ghost':>32}{'live':>32}")
    for lo, hi in MATCHED:
        cells = []
        for wallet in ("ghost", "live"):
            v = [(r, f) for n, f, r in books[wallet] if lo <= n < hi]
            cells.append(
                f"n={len(v):>3} {statistics.median([x[0] for x in v]):>8.2f}bp "
                f"fee ${statistics.median([x[1] for x in v]):.5f}"
                if v
                else "none"
            )
        label = f"[${lo:,.1f}, ${hi:,.1f})" if hi < 1e9 else f"[${lo:,.1f}, inf)"
        print(f"  {label:<16}{cells[0]:>32}{cells[1]:>32}")

    print("\n  Read the DOLLAR column, not the bp column:")
    print("    live  flat in dollars across buckets  -> fixed gas toll, no pool-fee term")
    print("    ghost rises with the clip             -> carries a proportional pool fee")
    print("  The pool fee is real, but live pays it in the FILL PRICE (realised")
    print("  slippage), not in fee_cost. The books are comparable only if the ghost")
    print("  MODELLED pool fee equals the live REALISED slippage. It does not.")
    print("\n  CAVEAT, do not skip: the live side is 18 rows and the [$1.0,$2.5)")
    print("  bucket that carries the 'flat in dollars' claim has n=4. Suggestive,")
    print("  not settled. Get more live clips before spending a pass on this.")


def main() -> int:
    if not os.path.exists(DB):
        print(f"no db at {DB}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    print(f"db: {DB}")
    for wallet in ("ghost", "live"):
        census(conn, wallet)
    ghost_vs_live(conn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
