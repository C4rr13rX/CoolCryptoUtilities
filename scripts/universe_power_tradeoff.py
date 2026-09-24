"""What restricting the tradeable universe costs, and what it buys.

Item ``[9c53c463]``, criteria 4 and 5, measured pass 697 by Sage.

THE ITEM'S OWN TITLE IS WRONG AND STAYS WRONG
---------------------------------------------
It says 84x. Echo p691 already answered criteria 1-3 and cut that to a MEDIAN
of 4.5x across strategies, with the 64.4x belonging to ``atf_static`` alone.
``scrum`` has no retitle action. Do not quote the title.

WHAT IS STILL OPEN, AND WHAT THIS FILE ANSWERS
----------------------------------------------
1. Criterion 4 -- the EDGE, separately from the variance. A restriction that
   cuts n_req is only a ROUTE if the thing it lets you measure faster is worth
   measuring. Report mean gross return FULL vs CLEAN, per strategy, with
   GHOST and LIVE in separate tables and never pooled ([227051f1]).
2. The method caveat -- Echo's ratio table POOLED ghost and live rows to
   estimate sd, which is defensible for a dispersion and indefensible for a
   mean. Everything here is GHOST-ONLY, so the question "does 64.4x survive
   ghost-only" gets a number.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not restrict anything. Criterion 5 says the universe change is a
separate item with its own back-to-back, so this module MEASURES and names the
landing site in its output. There is no write path in here at all.

DEFINITIONS, STATED SO THEY CAN BE ARGUED WITH
----------------------------------------------
GROSS RETURN is ``exit_price / entry_price - 1`` in bp. Not the ``profit``
column: ``iter_ghost_exits`` documents that column carries TWO UNITS (bare
fractions and dollars), and a mean or an sd over a mixed-unit column is not a
number. The price ratio is one unit by construction.

CLEAN is recomputed from ``market_stream`` by importing ``cell_edge_census
.feed_quality`` -- the p685 predicate, unchanged: over the trailing window a
symbol is CLEAN with >= 100 ticks and ZERO inter-tick moves above 10%. THIN
(too few ticks) is NOT clean by vacuum and is reported apart from DIRTY.

TRADEABLE is the ledger's own three filters in ``reconstruct``'s order, through
the same imports, at PAIR granularity -- so a row counted here is a row the
graduation bar would have counted.

    python scripts/universe_power_tradeoff.py
    python scripts/universe_power_tradeoff.py --days 7
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: z(1-a/2) + z(power): the two-sided one-sample power constant, 80% / 0.05.
#: Same constant as graduation_power_census and cell_edge_census, on purpose --
#: a different k would make the ratio incomparable to the numbers in the item.
K_80_05 = 2.8016

#: The toll the MDE is measured against, in bp. The item and both prior
#: measurements used 55.5bp, so the RATIO is quoted against 55.5 to stay
#: comparable. p696 measured the real WETH round trip at 12.4-17.8bp from 36/36
#: chain receipts -- the toll cancels out of the ratio entirely (it is a
#: constant in both numerator and denominator), but it does move the absolute
#: n_req, so both are printed.
TOLL_BP = 55.5
TOLL_BP_CHAIN = 17.8

#: Minimum rows before a mean or an sd is quoted at all.
MIN_N = 5


def _gross_bp(entry_price: object, exit_price: object) -> Optional[float]:
    """Gross return in bp from a price pair, or None if unreadable."""
    try:
        ep = float(entry_price)  # type: ignore[arg-type]
        xp = float(exit_price)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (ep > 0.0) or not math.isfinite(xp):
        return None
    bp = (xp / ep - 1.0) * 1e4
    return bp if math.isfinite(bp) else None


def _tstat(vals: Sequence[float]) -> Tuple[float, float, Optional[float]]:
    """(mean, sd, t) over a sample. t is None below n=2 or at zero spread."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, None
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, None
    sd = statistics.stdev(vals)
    if sd <= 0.0:
        return mean, sd, None
    return mean, sd, mean / (sd / math.sqrt(n))


def collect_ghost(conn: sqlite3.Connection) -> Tuple[List[Tuple[str, str, float]], Dict[str, int]]:
    """Tradeable ghost rows as (strategy, symbol, gross_bp), plus the funnel.

    Deliberately a copy of ``graduation_power_census.collect`` rather than an
    import of it: that function throws the symbol away, and the symbol is the
    entire subject of this item.
    """
    from services.tradeable_evidence import _bounds, _profit_in_usd
    from trading.strategies.ledger import _exceeds_evidence_horizon, _live_tradeable

    abs_cap, _horizon = _bounds()
    funnel = {
        "exits_total": 0,
        "dropped_no_profit_field": 0,
        "dropped_implausible": 0,
        "dropped_out_of_horizon": 0,
        "dropped_untradeable": 0,
        "dropped_no_price": 0,
        "kept": 0,
    }
    out: List[Tuple[str, str, float]] = []
    cache: Dict[Tuple[str, str], bool] = {}

    cur = conn.execute(
        "SELECT ts, symbol, details FROM trading_ops "
        "WHERE action = 'exit' AND status = 'ghost-exit' ORDER BY ts"
    )
    for row in cur:
        try:
            d = json.loads(row["details"] or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        funnel["exits_total"] += 1
        sid = str(d.get("strategy_id") or d.get("strategy") or "unclassified")
        symbol = str(d.get("symbol") or row["symbol"] or "")

        profit = _profit_in_usd(d, sid)
        if profit is None:
            funnel["dropped_no_profit_field"] += 1
            continue
        try:
            profit = float(profit)
        except (TypeError, ValueError):
            funnel["dropped_no_profit_field"] += 1
            continue
        if abs(profit) >= abs_cap:
            funnel["dropped_implausible"] += 1
            continue

        held = d.get("age_sec")
        if held is None:
            e_ts, x_ts = d.get("entry_ts"), d.get("exit_ts")
            if e_ts is not None and x_ts is not None:
                try:
                    held = float(x_ts) - float(e_ts)
                except (TypeError, ValueError):
                    held = None
        if _exceeds_evidence_horizon(held):
            funnel["dropped_out_of_horizon"] += 1
            continue

        key = (sid, symbol)
        if key not in cache:
            try:
                cache[key] = bool(_live_tradeable(symbol, sid))
            except Exception:  # noqa: BLE001 - unreadable is not tradeable
                cache[key] = False
        if not cache[key]:
            funnel["dropped_untradeable"] += 1
            continue

        bp = _gross_bp(d.get("entry_price"), d.get("exit_price"))
        if bp is None:
            funnel["dropped_no_price"] += 1
            continue

        funnel["kept"] += 1
        out.append((sid, symbol, bp))
    return out, funnel


def collect_live(conn: sqlite3.Connection) -> List[Tuple[str, str, float]]:
    """Live closed rows as (strategy, symbol, gross_bp).

    Live rows carry their prices in ``trade_outcomes`` directly. The strategy
    is NOT a column there -- it is tied through ``trade_id``, which [d7... p695]
    established appears verbatim in ``trading_ops.details`` and joins 18 of 18.
    A (symbol, ts) join is wrong here and collides; do not reintroduce one.

    Note ``wallet`` is a STRING: ``wallet = 1`` returns zero rows silently.
    """
    strat_by_trade: Dict[str, str] = {}
    for row in conn.execute(
        "SELECT details FROM trading_ops WHERE action = 'exit' AND status = 'live-exit'"
    ):
        try:
            d = json.loads(row["details"] or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        tid = d.get("trade_id")
        sid = d.get("strategy_id") or d.get("strategy")
        if tid and sid:
            strat_by_trade[str(tid)] = str(sid)

    out: List[Tuple[str, str, float]] = []
    for row in conn.execute(
        "SELECT trade_id, symbol, entry_price, exit_price FROM trade_outcomes "
        "WHERE wallet = 'live' ORDER BY ts"
    ):
        bp = _gross_bp(row["entry_price"], row["exit_price"])
        if bp is None:
            continue
        sid = strat_by_trade.get(str(row["trade_id"]), "unattributed")
        out.append((sid, str(row["symbol"] or ""), bp))
    return out


def _n_req(sd: float, toll: float) -> float:
    return (K_80_05 * sd / toll) ** 2


def _table(
    title: str,
    rows: Sequence[Tuple[str, str, float]],
    clean: set,
    *,
    min_n: int = MIN_N,
) -> List[Tuple[str, int, float, float, int, float, float]]:
    """Print the FULL-vs-CLEAN edge and dispersion table for one book."""
    full: Dict[str, List[float]] = {}
    narrow: Dict[str, List[float]] = {}
    for sid, sym, bp in rows:
        full.setdefault(sid, []).append(bp)
        if sym in clean:
            narrow.setdefault(sid, []).append(bp)

    print()
    print(title)
    print("-" * len(title))
    if not full:
        print("    (no rows)")
        return []
    hdr = "%-24s %5s %8s %7s %8s %9s   %5s %8s %7s %8s %9s  %7s %8s" % (
        "strategy", "n", "mean", "sd k", "n_toll", "n_excess",
        "n", "mean", "sd k", "n_toll", "n_excess", "toll x", "excess x",
    )
    print("%-24s %-40s   %-40s" % ("", "-------- FULL universe --------",
                                   "-------- CLEAN only -----------"))
    print(hdr)
    print("-" * len(hdr))

    collected: List[Tuple[str, int, float, float, int, float, float]] = []
    ratios: List[float] = []
    ratios_excess: List[float] = []
    for sid in sorted(full, key=lambda s: -len(full[s])):
        fv = full[sid]
        nv = narrow.get(sid, [])
        fm, fsd, _ft = _tstat(fv)
        nm, nsd, _nt = _tstat(nv)
        fr = _n_req(fsd, TOLL_BP) if len(fv) >= 2 else float("nan")
        nr = _n_req(nsd, TOLL_BP) if len(nv) >= 2 else float("nan")
        # n to detect the strategy's OWN EXCESS over the toll. This is the
        # number that decides the item: n_toll asks "could you see a
        # toll-sized effect", which a restriction makes easy by shrinking sd.
        # n_excess asks "could you see THIS strategy actually clearing the
        # toll", and a restriction that shrinks the mean as hard as the sd
        # buys nothing on it. A NEGATIVE excess is printed as NEG -- no n
        # proves a profit that is not there, and reporting (k*sd/excess)^2
        # on a negative excess would silently report the n for a LOSS.
        fe = fm - TOLL_BP
        ne = nm - TOLL_BP
        fx = _n_req(fsd, abs(fe)) if (len(fv) >= 2 and fe > 0) else float("nan")
        nx = _n_req(nsd, abs(ne)) if (len(nv) >= 2 and ne > 0) else float("nan")
        ratio = (fr / nr) if (nr == nr and nr > 0 and fr == fr) else float("nan")
        ratio_x = (fx / nx) if (nx == nx and nx > 0 and fx == fx) else float("nan")

        def _f(x: float, w: int = 8, p: int = 1) -> str:
            return ("%*.*f" % (w, p, x)) if x == x else "%*s" % (w, "--")

        def _i(x: float, w: int = 8) -> str:
            return ("%*d" % (w, round(x))) if x == x else "%*s" % (w, "--")

        def _x(x: float, neg: bool, w: int = 9) -> str:
            if neg:
                return "%*s" % (w, "NEG")
            return ("%*d" % (w, round(x))) if x == x else "%*s" % (w, "--")

        print("%-24s %5d %s %s %s %s   %5d %s %s %s %s  %7s %8s" % (
            sid, len(fv), _f(fm), _f(fsd, 7, 0), _i(fr),
            _x(fx, len(fv) >= 2 and fe <= 0),
            len(nv), _f(nm), _f(nsd, 7, 0), _i(nr),
            _x(nx, len(nv) >= 2 and ne <= 0),
            ("%.1fx" % ratio) if ratio == ratio else "--",
            ("%.1fx" % ratio_x) if ratio_x == ratio_x else "--",
        ))
        if len(fv) >= min_n and len(nv) >= min_n and ratio == ratio:
            ratios.append(ratio)
            if ratio_x == ratio_x:
                ratios_excess.append(ratio_x)
        collected.append((sid, len(fv), fm, fr, len(nv), nm, nr))

    print()
    if ratios:
        print("    Over the %d strategies with n>=%d in BOTH columns:"
              % (len(ratios), min_n))
        print("      n_toll  ratio: median %6.1fx  mean %6.1fx  min %.1fx  max %.1fx"
              % (statistics.median(ratios), statistics.mean(ratios),
                 min(ratios), max(ratios)))
        if ratios_excess:
            print("      n_excess ratio: median %6.1fx  mean %6.1fx  (%d of %d "
                  "strategies keep a POSITIVE excess in both columns)"
                  % (statistics.median(ratios_excess),
                     statistics.mean(ratios_excess),
                     len(ratios_excess), len(ratios)))
        else:
            print("      n_excess ratio: UNQUOTABLE -- 0 of %d strategies keep a"
                  " POSITIVE excess over the toll in BOTH columns." % len(ratios))
    else:
        print("    NO strategy has n>=%d in both columns -- no ratio is quotable."
              % min_n)
    return collected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=14.0,
                    help="trailing window for the CLEAN feed verdict")
    ap.add_argument("--min-ticks", type=int, default=100)
    ap.add_argument("--jump", type=float, default=0.10)
    args = ap.parse_args()

    from scripts.cell_edge_census import feed_quality
    from services.tradeable_evidence import DEFAULT_DB

    conn = sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        ghost, funnel = collect_ghost(conn)
        live = collect_live(conn)
        symbols = sorted({s for _, s, _ in ghost} | {s for _, s, _ in live})
        feed = feed_quality(conn, symbols, days=args.days,
                            jump=args.jump, min_ticks=args.min_ticks)
    finally:
        conn.close()

    clean = {s for s, d in feed.items() if d["verdict"] == "CLEAN"}
    counts: Dict[str, int] = {}
    for d in feed.values():
        counts[d["verdict"]] = counts.get(d["verdict"], 0) + 1

    print("=" * 96)
    print("UNIVERSE POWER TRADE-OFF -- what narrowing buys, and what it costs")
    print("[9c53c463] criteria 4 and 5.  GHOST and LIVE are NEVER pooled.")
    print("=" * 96)
    print()
    print("GHOST POPULATION FUNNEL (the ledger's own filters, same order)")
    for k in ("exits_total", "dropped_no_profit_field", "dropped_implausible",
              "dropped_out_of_horizon", "dropped_untradeable",
              "dropped_no_price", "kept"):
        print("    %-26s %7d" % (k, funnel[k]))
    print("    %-26s %7d" % ("live closed rows", len(live)))
    print()
    print("FEED VERDICT over the trailing %.0fd (>=%d ticks, zero jumps >%.0f%%)"
          % (args.days, args.min_ticks, args.jump * 100))
    print("    %d CLEAN, %d DIRTY, %d THIN   of %d symbols in the two books"
          % (counts.get("CLEAN", 0), counts.get("DIRTY", 0),
             counts.get("THIN", 0), len(feed)))
    print("    CLEAN: %s" % (", ".join(sorted(clean)) or "(none)"))
    print()
    print("n_req = (%.4f * sd / %.1fbp)^2   -- the toll CANCELS in the ratio;"
          % (K_80_05, TOLL_BP))
    print("it only moves the absolute n_req. At the p696 chain-measured %.1fbp"
          % TOLL_BP_CHAIN)
    print("toll every absolute n_req below multiplies by %.2fx."
          % ((TOLL_BP / TOLL_BP_CHAIN) ** 2))

    _table("GHOST BOOK -- tradeable rows only, ghost-only sd (the method caveat)",
           ghost, clean)
    _table("LIVE BOOK -- settled rows only, reported apart and never pooled",
           live, clean, min_n=2)

    print()
    print("=" * 96)
    print("CRITERION 5 -- WHERE A RESTRICTION WOULD LAND (named, not shipped)")
    print("=" * 96)
    print("""
    The universe predicate is NOT at the bar sites Kite p691 named. Those
    (services/tradeable_evidence.py, services/strategy_population.py) decide
    HOW MANY rows are needed and WHICH rows count as evidence. The universe --
    which symbols may be traded at all -- is decided upstream of both, at the
    tradeable predicate itself. Grep output below is produced live so this
    claim cannot rot silently.""")
    import subprocess
    for pat, where in (
        (r"def _live_tradeable", "trading/strategies/ledger.py"),
        (r"def _live_tradeable|LIVE_SYMBOLS|live_watchlist", "services/watchlists.py"),
    ):
        try:
            r = subprocess.run(
                ["rg", "-n", pat, where], capture_output=True, text=True, timeout=20,
                cwd=str(PROJECT_ROOT))
            for line in (r.stdout or "").strip().splitlines()[:6]:
                print("    %s:%s" % (where, line))
        except Exception as exc:  # noqa: BLE001
            print("    (grep failed for %s: %s)" % (where, exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
