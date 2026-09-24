"""[3e185d47] Does ANY feature knowable AT ENTRY TIME separate gross winners
from gross losers in the existing book?

Hypothesis : some entry-time feature separates them and the entry rule ignores it.
Complement : nothing available separates them; the win rate is irreducible at
             decision time and the edge must come from somewhere other than
             entry selection.

Pre-registered kill test (fixed before looking, Sage p691):
    AUC >= 0.60 at n >= 50, or a quintile with n >= 25 and win rate > 55%
    -> complement dead, hypothesis wins. Otherwise the complement stands.

The book carries no entry timestamp on 97% of rows (trade_outcomes.details has
entry_ts on 8 of 286), so entry-time context is recovered by joining each closed
outcome to the trading_ops action='enter' row that proposed it, on
(symbol, entry_price) with the enter row's ts preceding the outcome's ts.
Read-only throughout.
"""
from __future__ import annotations

import collections
import json
import math
import os
import sqlite3
import statistics
import sys
import time

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")

# Features carried on the trading_ops enter row, knowable at decision time.
OPS_FEATURES = (
    "confidence",
    "score",
    "expected_return",
    "liquidity_usd",
    "volume_h1",
    "budget_usd",
)


def connect() -> sqlite3.Connection:
    return sqlite3.connect("file:" + DB + "?mode=ro", uri=True)


def load_outcomes(conn):
    """Closed round trips with a usable gross return, ghost and live."""
    rows = []
    q = """select outcome_id, ts, wallet, symbol, entry_price, exit_price,
                  quantity, gross_profit, fee_cost, net_profit, details
             from trade_outcomes where status='closed'"""
    for r in conn.execute(q):
        (oid, ts, wallet, symbol, ep, xp, qty, gross, fee, net, details) = r
        if not symbol or ep in (None, 0) or gross is None:
            continue
        try:
            notional = abs(float(ep) * float(qty or 0))
        except (TypeError, ValueError):
            continue
        if notional <= 0:
            continue
        try:
            meta = json.loads(details) if details else {}
        except (ValueError, TypeError):
            meta = {}
        rows.append(
            {
                "outcome_id": oid,
                "ts": float(ts or 0),
                "wallet": wallet,
                "symbol": symbol,
                "entry_price": float(ep),
                "exit_price": float(xp) if xp is not None else None,
                "gross": float(gross),
                "gross_ret": float(gross) / notional,
                "notional": notional,
                "strategy_id": meta.get("strategy_id"),
                "entry_ts": meta.get("entry_ts"),
            }
        )
    return rows


def exclusion_pass(rows, conn):
    """Drop the two known artifacts. Returns (kept, reasons_counter).

    This item is ABOUT entry price, so a bad entry price is fatal to it
    specifically -- the exclusion is not cosmetic here.
    """
    reasons = collections.Counter()
    kept = []
    # tape envelope per symbol, for the bad-entry-price test
    env = {}
    for sym, lo, hi in conn.execute(
        "select symbol, min(price), max(price) from market_stream group by symbol"
    ):
        if lo and hi:
            env[sym] = (float(lo), float(hi))

    for r in rows:
        # frozen-feed: entry == exit and gross exactly 0.0
        if r["exit_price"] is not None and r["gross"] == 0.0 and (
            r["exit_price"] == r["entry_price"]
        ):
            reasons["frozen_feed_flat_round_trip"] += 1
            continue
        # bad entry price: priced outside the symbol's entire observed tape
        band = env.get(r["symbol"])
        if band:
            lo, hi = band
            if r["entry_price"] < lo * 0.5 or r["entry_price"] > hi * 2.0:
                reasons["entry_price_outside_tape_envelope"] += 1
                continue
        # implausible magnitude: |gross return| > 100% on a spot round trip
        if abs(r["gross_ret"]) > 1.0:
            reasons["implausible_gross_return_over_100pct"] += 1
            continue
        kept.append(r)
    return kept, reasons


def join_entry_context(rows, conn, tol=1e-6):
    """Attach the trading_ops enter row that proposed each outcome."""
    by_symbol = collections.defaultdict(list)
    q = """select ts, symbol, details from trading_ops
            where action='enter' and details is not null"""
    for ts, symbol, details in conn.execute(q):
        if not symbol:
            continue
        try:
            j = json.loads(details)
        except (ValueError, TypeError):
            continue
        px = j.get("price_usd") or j.get("entry_price")
        if px is None:
            continue
        try:
            by_symbol[symbol].append((float(ts or 0), float(px), j))
        except (TypeError, ValueError):
            continue
    for v in by_symbol.values():
        v.sort(key=lambda t: t[0])

    matched = 0
    for r in rows:
        cands = by_symbol.get(r["symbol"], ())
        best = None
        for ots, px, j in cands:
            if r["ts"] and ots > r["ts"]:
                break  # enter must precede the booking
            if px <= 0:
                continue
            if abs(px - r["entry_price"]) / max(abs(px), 1e-18) <= tol:
                best = (ots, j)  # latest matching enter before the outcome
        if best:
            r["entry_ops_ts"] = best[0]
            r["ops"] = best[1]
            matched += 1
        else:
            r["ops"] = None
    return matched, sum(len(v) for v in by_symbol.values())


def tape_features(rows, conn):
    """Features derived from the tape as of the entry timestamp."""
    need = [r for r in rows if r.get("entry_ops_ts")]
    by_symbol = collections.defaultdict(list)
    for r in need:
        by_symbol[r["symbol"]].append(r)

    for sym, rs in by_symbol.items():
        ticks = [
            (float(t), float(p))
            for t, p in conn.execute(
                "select ts, price from market_stream where symbol=? order by ts",
                (sym,),
            )
            if t is not None and p
        ]
        if not ticks:
            continue
        times = [t for t, _ in ticks]
        import bisect

        for r in rs:
            ets = r["entry_ops_ts"]
            i = bisect.bisect_right(times, ets)
            if i == 0:
                continue
            prior = ticks[max(0, i - 40) : i]
            r["secs_since_last_tick"] = ets - prior[-1][0]
            r["tick_density_10min"] = sum(1 for t, _ in prior if t >= ets - 600)
            px = [p for _, p in prior]
            if len(px) >= 4:
                rets = [
                    (px[k] - px[k - 1]) / px[k - 1]
                    for k in range(1, len(px))
                    if px[k - 1]
                ]
                if len(rets) >= 3:
                    r["realised_vol"] = statistics.pstdev(rets)
            r["hour_of_day"] = time.gmtime(ets).tm_hour


def auc(pairs):
    """Rank AUC of feature vs binary label. pairs = [(x, y in {0,1})]."""
    pairs = [(x, y) for x, y in pairs if x is not None and not (
        isinstance(x, float) and math.isnan(x))]
    pos = [x for x, y in pairs if y == 1]
    neg = [x for x, y in pairs if y == 0]
    if not pos or not neg:
        return None, len(pairs), len(pos), len(neg)
    xs = sorted(x for x, _ in pairs)
    ranks = {}
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[xs[i]] = avg
        i = j + 1
    rsum = sum(ranks[x] for x in pos)
    n1, n0 = len(pos), len(neg)
    a = (rsum - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return a, len(pairs), n1, n0


def quintiles(pairs):
    pairs = [(x, y) for x, y in pairs if x is not None]
    if len(pairs) < 10:
        return []
    pairs.sort(key=lambda t: t[0])
    n = len(pairs)
    out = []
    for k in range(5):
        lo, hi = n * k // 5, n * (k + 1) // 5
        chunk = pairs[lo:hi]
        if not chunk:
            continue
        wins = sum(1 for _, y in chunk if y == 1)
        out.append((k + 1, len(chunk), wins / len(chunk), chunk[0][0], chunk[-1][0]))
    return out


def tape_revalidation(joined, conn, feature="secs_since_last_tick"):
    """Re-measure every return off the TAPE, discarding the booked prices.

    The dangerous alternative explanation for any freshness result is that the
    feature predicts the ACCURACY of the booked return rather than the trade's
    return: a stale tape at entry means entry_price is a stale mark. If that is
    what is happening, the separation CANNOT survive here, because the booked
    prices play no part in the numbers below.
    """
    import bisect

    rows = [r for r in joined if r.get(feature) is not None and r.get("ts")]
    cache = {}
    for r in rows:
        sym = r["symbol"]
        if sym not in cache:
            t = [
                (float(a), float(b))
                for a, b in conn.execute(
                    "select ts, price from market_stream where symbol=? order by ts",
                    (sym,),
                )
                if a is not None and b
            ]
            cache[sym] = ([x[0] for x in t], [x[1] for x in t])
        ts_, px_ = cache[sym]
        if not ts_:
            continue
        i = bisect.bisect_right(ts_, r["entry_ops_ts"]) - 1
        j = bisect.bisect_right(ts_, r["ts"]) - 1
        if i < 0 or j < 0 or j <= i:
            continue
        # both legs must sit near a real tick, or the mark is stale too
        if abs(ts_[i] - r["entry_ops_ts"]) > 600 or abs(ts_[j] - r["ts"]) > 600:
            continue
        r["tape_ret"] = (px_[j] - px_[i]) / px_[i]

    T = [r for r in rows if "tape_ret" in r]
    print(
        f"\nTAPE RE-VALIDATION -- booked prices discarded, {len(T)} of {len(rows)}"
        " rows have both legs within 10 min of a real tick"
    )
    if len(T) < 25:
        print("   too few rows to re-validate.")
        return
    T.sort(key=lambda r: r[feature])
    n = len(T)
    print(
        f"   {'quintile':10s}{'n':>5}{'bookedWin':>11}{'tapeWin':>9}"
        f"{'medBooked':>11}{'medTape':>10}"
    )
    for k in range(5):
        ch = T[n * k // 5 : n * (k + 1) // 5]
        if not ch:
            continue
        bw = 100 * sum(1 for r in ch if r["gross"] > 0) / len(ch)
        tw = 100 * sum(1 for r in ch if r["tape_ret"] > 0) / len(ch)
        mb = 100 * statistics.median([r["gross_ret"] for r in ch])
        mt = 100 * statistics.median([r["tape_ret"] for r in ch])
        print(f"   Q{k+1:<9d}{len(ch):5d}{bw:10.1f}%{tw:8.1f}%{mb:+10.3f}%{mt:+9.3f}%")
    agree = 100 * sum(
        1 for r in T if (r["gross_ret"] > 0) == (r["tape_ret"] > 0)
    ) / len(T)
    print(f"   booked-vs-tape sign agreement: {agree:.1f}%")
    print(
        "   If Q1's tape win rate tracks its booked win rate, the separation is\n"
        "   in the TRADES, not in the bookkeeping."
    )


def main() -> int:
    conn = connect()
    raw = load_outcomes(conn)
    print(f"closed round trips with a computable gross return : {len(raw)}")

    kept, reasons = exclusion_pass(raw, conn)
    print("\nEXCLUSIONS (stated, per the item's requirement):")
    for k, v in reasons.most_common():
        print(f"   -{v:4d}  {k}")
    print(f"   rows before {len(raw)}  ->  after {len(kept)}")

    base_raw = sum(1 for r in raw if r["gross"] > 0) / max(len(raw), 1)
    base_kept = sum(1 for r in kept if r["gross"] > 0) / max(len(kept), 1)
    print(
        f"\n   gross win rate BEFORE cleaning {100*base_raw:5.1f}%"
        f"   AFTER {100*base_kept:5.1f}%"
        "   <- a cleaning step that flatters the book is visible here"
    )

    matched, n_ops = join_entry_context(kept, conn)
    print(
        f"\nENTRY-CONTEXT JOIN: {matched} of {len(kept)} cleaned rows matched an"
        f" action='enter' row (pool {n_ops})"
    )
    if matched == 0:
        print("   NO JOIN. Entry-time features are unrecoverable for this book.")
        return 2

    tape_features(kept, conn)
    joined = [r for r in kept if r.get("ops")]

    feats = {}
    for f in OPS_FEATURES:
        feats[f] = [
            (r["ops"].get(f), 1 if r["gross"] > 0 else 0)
            for r in joined
            if isinstance(r["ops"].get(f), (int, float))
        ]
    for f in ("secs_since_last_tick", "tick_density_10min", "realised_vol",
              "hour_of_day"):
        feats[f] = [
            (r.get(f), 1 if r["gross"] > 0 else 0) for r in joined if r.get(f) is not None
        ]

    wins = sum(1 for r in joined if r["gross"] > 0)
    print(
        f"joined book: n={len(joined)}  gross winners={wins}"
        f"  win rate={100*wins/max(len(joined),1):.1f}%"
    )

    print("\nSEPARATION -- AUC of each entry-time feature vs gross win/loss")
    print(f"   {'feature':24s}{'AUC':>8}{'n':>7}{'win':>6}{'loss':>6}   verdict")
    verdicts = []
    for f, pairs in sorted(feats.items()):
        a, n, n1, n0 = auc(pairs)
        if a is None:
            print(f"   {f:24s}{'--':>8}{n:7d}{n1:6d}{n0:6d}   no variation")
            continue
        hit = a >= 0.60 or a <= 0.40
        flag = "SEPARATES" if (hit and n >= 50) else ("thin" if hit else "flat")
        verdicts.append((f, a, n, flag))
        print(f"   {f:24s}{a:8.3f}{n:7d}{n1:6d}{n0:6d}   {flag}")

    print("\nQUINTILE WIN RATES (the item's n>=25 / >55% bar)")
    for f, pairs in sorted(feats.items()):
        qs = quintiles(pairs)
        if not qs:
            continue
        marks = " ".join(
            f"Q{k}:{100*w:.0f}%(n={c})" + ("*" if c >= 25 and w > 0.55 else "")
            for k, c, w, _, _ in qs
        )
        print(f"   {f:24s} {marks}")
    print("   * = clears the pre-registered bar (n>=25 and win rate >55%)")

    tape_revalidation(joined, conn)

    print("\nPRE-REGISTERED VERDICT")
    # The pre-registration has TWO arms and an earlier version of this function
    # tested only the first, printing COMPLEMENT STANDS while the quintile arm
    # had in fact fired. Both arms are evaluated here.
    killers = [v for v in verdicts if v[3] == "SEPARATES"]
    quintile_killers = []
    for f, pairs in sorted(feats.items()):
        for k, c, w, lo, hi in quintiles(pairs):
            if c >= 25 and w > 0.55:
                quintile_killers.append((f, k, c, w, lo, hi))

    if killers:
        print("   ARM 1 FIRED -- a feature cleared AUC>=0.60 at n>=50:")
        for f, a, n, _ in killers:
            print(f"      {f}  AUC {a:.3f}  n={n}")
    else:
        print("   arm 1 (AUC>=0.60 at n>=50): not met by any feature.")

    if quintile_killers:
        print("   ARM 2 FIRED -- quintile with n>=25 above a 55% win rate:")
        for f, k, c, w, lo, hi in quintile_killers:
            print(
                f"      {f}  Q{k}  win {100*w:.1f}%  n={c}  range [{lo:.6g}, {hi:.6g}]"
            )
    else:
        print("   arm 2 (quintile n>=25 above 55%): not met by any feature.")

    if killers or quintile_killers:
        print(
            "\n   COMPLEMENT DEAD. Some entry-time feature separates gross\n"
            "   winners from losers, and the entry rule does not use it."
        )
    else:
        print(
            "\n   COMPLEMENT STANDS -- neither arm fired.\n"
            "   Entry selection does not explain the win rate in this book."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
