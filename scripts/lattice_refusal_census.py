"""Census the lattice gate's refusals: what it refused, and how far past its own limit.

READ-ONLY. Opens the store with mode=ro and query_only=1, so it can never block
a writer and can never alter the evidence it is measuring.

WHY THIS EXISTS (Echo, pass 482, backlog [0f07b8b6]).
The chaos layer refuses when ``proposed_horizon_sec > usable``. Measured over
7 days, the MEDIAN refusal was 1334x past the limit and only 2 of 2585 were
within 2x of it. A gate whose median refusal overshoots by three orders of
magnitude is not measuring the thing it names -- it is being handed the wrong
number. ``trading/strategies/horizons.py:212`` does
``directive.horizon = self._label``, and a HORIZON_SPECS label is a BAR WINDOW
("1w" = 4h bars over a 10-day lookback), not a hold time. The bot's real cap is
MAX_HOLD_FORCE_SECONDS=2700 -- 45 minutes.

Run this BEFORE and AFTER any change to what is passed as proposed_horizon_sec.
The ``--predict`` block prints what SHOULD happen, so a result that is wildly
better than the prediction is read as a bug rather than a win.

    python -X utf8 scripts/lattice_refusal_census.py            # 24h and 7d
    python -X utf8 scripts/lattice_refusal_census.py --hours 24
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sqlite3
import sys

#: Where the evidence actually lives. data/trading.db, trading_bot.db and
#: web/db.sqlite3 are 0-byte stubs that silently return nothing.
DB_PATH = os.environ.get("TRADING_DB_PATH") or os.path.join(
    "storage", "trading_cache.db")

#: The bot cannot hold a position longer than this, so it is the longest
#: horizon any honest ask could name. .env MAX_HOLD_FORCE_SECONDS=2700.
MAX_HOLD_MIN = float(os.environ.get("MAX_HOLD_FORCE_SECONDS", "2700")) / 60.0

#: "SYM: information decays after N min but the signal looks M min ahead -- Kx"
_REFUSAL = re.compile(
    r"(\S+?):\s*information decays after ([0-9.]+) min but the signal looks "
    r"([0-9.]+) min ahead -- ([0-9.]+)x")


def _rows(con, window_sec: float):
    now = con.execute("SELECT MAX(ts) FROM trading_ops").fetchone()[0]
    if now is None:
        return []
    cur = con.execute(
        "SELECT details FROM trading_ops WHERE status='entry-refused-lattice' "
        "AND ts >= ?", (now - window_sec,))
    out = []
    for (raw,) in cur:
        try:
            payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (ValueError, TypeError):
            continue
        out.append(payload)
    return out


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "n/a"


def census(con, hours: float) -> None:
    payloads = _rows(con, hours * 3600.0)
    print(f"\n{'=' * 70}\n  LATTICE REFUSALS over {hours:g}h -- {len(payloads)} rows\n{'=' * 70}")
    if not payloads:
        print("  none. If this is unexpected, check the process is running "
              "before reading zero as a result.")
        return

    layers = collections.Counter()
    overshoot, asks, usable_by_sym, flip = [], collections.Counter(), {}, collections.Counter()
    seen_by_sym = collections.Counter()

    for p in payloads:
        detail = str(p.get("detail") or "")
        head = detail.split(":", 1)[0].strip()
        if head:
            layers[head] += 1
        m = _REFUSAL.search(detail)
        if not m:
            continue
        sym, usable, ask, mult = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
        overshoot.append(mult)
        asks[ask] += 1
        seen_by_sym[sym] += 1
        usable_by_sym.setdefault(sym, []).append(usable)
        if usable >= MAX_HOLD_MIN:
            flip[sym] += 1

    print(f"\n  BY LAYER: {dict(layers.most_common(6))}")

    if overshoot:
        overshoot.sort()
        n = len(overshoot)
        near = sum(1 for v in overshoot if v <= 2.0)
        print(f"\n  HOW FAR PAST ITS OWN LIMIT  (n={n})")
        print(f"    min {overshoot[0]:.1f}x   p50 {overshoot[n // 2]:.1f}x   "
              f"p90 {overshoot[9 * n // 10]:.1f}x   max {overshoot[-1]:.1f}x")
        print(f"    within 2x of the limit: {near} ({_pct(near, n)})")
        print("    ^ a median far above ~2x means the gate is judging a "
              "number nobody asked for.")

    if asks:
        print(f"\n  WHAT HORIZON WAS ASKED FOR  (cap is {MAX_HOLD_MIN:.0f} min)")
        for ask, count in asks.most_common(8):
            flagged = "" if ask <= MAX_HOLD_MIN else "   <- LONGER THAN THE BOT CAN HOLD"
            print(f"    {ask:9.1f} min  {count:6d}{flagged}")
        impossible = sum(c for a, c in asks.items() if a > MAX_HOLD_MIN)
        total = sum(asks.values())
        print(f"    asks the bot could never honour: {impossible} of {total} "
              f"({_pct(impossible, total)})")

    if usable_by_sym:
        print("\n  USABLE HORIZON PER SYMBOL  (the estimator's own answer)")
        ranked = sorted(usable_by_sym.items(), key=lambda kv: -len(kv[1]))[:8]
        for sym, vals in ranked:
            lo, hi = min(vals), max(vals)
            spread = hi / lo if lo > 0 else float("inf")
            print(f"    {sym:16s} n={len(vals):5d}  {lo:7.1f}-{hi:7.1f} min  "
                  f"spread {spread:6.1f}x  distinct={len(set(vals))}")
        print("    ^ many distinct values is a CONTINUUM, not two regimes: "
              "bar_sec is re-measured per call.")

    print(f"\n  PREDICTION -- would these refusals survive an HONEST ask?")
    for bar in (MAX_HOLD_MIN, 20.0, 5.0):
        passes = sum(1 for vals in usable_by_sym.values() for u in vals if u >= bar)
        total = sum(len(v) for v in usable_by_sym.values())
        print(f"    at a {bar:5.0f} min ask: {passes:5d} of {total} would PASS "
              f"({_pct(passes, total)})")
    if flip:
        print("    concentrated in:", ", ".join(
            f"{s} {c}/{seen_by_sym[s]}" for s, c in flip.most_common(5)))
        print("    ^ CHECK THESE AGAINST POOL DEPTH before calling it more "
              "candidates. Untradeable-pool volume is not progress.")
    else:
        print(f"    NO symbol clears {MAX_HOLD_MIN:.0f} min in this window.")


def funnel(con, hours: float) -> None:
    """Every drop in the funnel, so the lattice is ranked honestly.

    Counting only ``entry-refused%`` hides entry-predropped-edge-ban and
    entry-arbitration, both of which are larger and sit UPSTREAM.
    """
    now = con.execute("SELECT MAX(ts) FROM trading_ops").fetchone()[0]
    print(f"\n  FULL FUNNEL over {hours:g}h -- ALL statuses, not just entry-refused%")
    for status, count in con.execute(
            "SELECT status, COUNT(*) FROM trading_ops WHERE ts >= ? "
            "GROUP BY status ORDER BY 2 DESC LIMIT 12", (now - hours * 3600.0,)):
        print(f"    {status:34s} {count:7d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=None,
                    help="one window instead of the default 24h and 7d")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"no store at {args.db}", file=sys.stderr)
        return 2

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=1")
    try:
        windows = [args.hours] if args.hours else [24.0, 168.0]
        for hours in windows:
            census(con, hours)
        funnel(con, windows[-1])
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
