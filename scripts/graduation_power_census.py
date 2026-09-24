"""Per-strategy statistical POWER of the 20-trade graduation bar.

Item ``[c48bb3fc]``, measured pass 691.

THE QUESTION
------------
The graduation bar is a flat ``n >= 20`` tradeable ghost trades, plus a 55%
win rate and P/L above zero (``services/strategy_population.py``,
``services/tradeable_evidence.py:315``, ``STRATEGY_GRADUATION_MIN_WINRATE``).
``n`` is flat. The per-strategy dispersion of gross return is not. A fixed n
against a variable sd is a bar whose MEANING changes per strategy: for a tight
strategy it is strict, for a wide one it certifies noise.

Minimum detectable effect at 80% power, two-sided alpha=0.05, one-sample::

    MDE = k * sd / sqrt(n),   k = z(0.975) + z(0.80) = 1.95996 + 0.84162 = 2.8016

and inverted, the n needed to see an edge the size of the live round-trip toll::

    n_required = (k * sd / TOLL) ** 2

RETURN, NOT PROFIT
------------------
Dispersion is measured on ``exit_price / entry_price - 1``, taken from the
``ghost-exit`` detail blob, NOT on the ``profit`` field. That is deliberate and
it is the whole reason this file does not reuse ``Evidence.net``:
``iter_ghost_exits`` documents that the raw ``profit`` column is TWO UNITS --
473 of 806 exit rows record a bare fraction and 298 record dollars. A standard
deviation over a mixed-unit column is not a number. The price ratio is one unit
by construction and needs no converter.

POPULATION
----------
Exactly the rows the ledger would count, via the same three filters
``tradeable_evidence.reconstruct`` applies, in the same order and through the
same imports (absolute implausibility cap, evidence horizon, and
``_live_tradeable(symbol, strategy_id)`` at PAIR granularity). Every drop is
counted and printed. A power claim on a silently narrowed population is worse
than no power claim, so the funnel is part of the output, not a footnote.

This module MEASURES. It does not change the bar and must not: the bar change
is a separate item needing its own back-to-back before/after.

    python scripts/graduation_power_census.py
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: z(1-a/2) + z(power), the standard two-sided one-sample power constant.
K_80_05 = 2.8016

#: The live round-trip toll in basis points (p682, measured over live receipts).
#: This is the edge a strategy must be able to SEE to be worth graduating: an
#: edge smaller than the toll is not a profit at any n.
TOLL_BP = 55.5

#: The bar as shipped.
BAR_N = 20


def _gross_bp(entry_price: object, exit_price: object) -> float | None:
    """Gross return in bp from the price pair, or None if unreadable.

    A flat round trip (entry == exit to the last digit) is KEPT here, unlike in
    ``strategy_edge_gate``. This file measures DISPERSION, and dropping the
    zero-variance rows would understate the sd of exactly the strategies whose
    tape stalls -- the opposite of the conservative direction for a power claim.
    """
    try:
        ep = float(entry_price)  # type: ignore[arg-type]
        xp = float(exit_price)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (ep > 0.0) or not math.isfinite(xp):
        return None
    bp = (xp / ep - 1.0) * 1e4
    return bp if math.isfinite(bp) else None


def collect() -> tuple[Dict[str, List[float]], Dict[str, int]]:
    """Tradeable ghost gross returns in bp, per strategy, plus the funnel."""
    import sqlite3
    import json

    from services.tradeable_evidence import DEFAULT_DB, _bounds, _profit_in_usd
    from trading.strategies.ledger import _exceeds_evidence_horizon, _live_tradeable

    abs_cap, _horizon = _bounds()
    funnel = {
        "exits_total": 0,
        "dropped_no_price": 0,
        "dropped_no_profit_field": 0,
        "dropped_implausible": 0,
        "dropped_out_of_horizon": 0,
        "dropped_untradeable": 0,
        "kept": 0,
    }
    book: Dict[str, List[float]] = {}
    cache: Dict[tuple, bool] = {}

    conn = sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
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

            # Filters 1-3, in reconstruct()'s order, so a row kept here is a row
            # the ledger would have counted.
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
            book.setdefault(sid, []).append(bp)
    finally:
        conn.close()
    return book, funnel


def main() -> int:
    book, funnel = collect()

    print("=" * 86)
    print("GRADUATION POWER CENSUS -- what the flat n=20 bar can actually SEE")
    print("=" * 86)
    print()
    print("POPULATION FUNNEL (every drop counted; a power claim on a narrowed")
    print("population is worse than no power claim)")
    for k in (
        "exits_total",
        "dropped_no_profit_field",
        "dropped_implausible",
        "dropped_out_of_horizon",
        "dropped_untradeable",
        "dropped_no_price",
        "kept",
    ):
        print(f"    {k:<26} {funnel[k]:>7}")
    print(f"    {'strategies with >=1 row':<26} {len(book):>7}")
    print()
    print(f"MDE = {K_80_05} * sd / sqrt(n)   at 80% power, two-sided a=0.05")
    print(f"n_req = ({K_80_05} * sd / {TOLL_BP})^2   to see a toll-sized edge")
    print()

    rows = []
    for sid, vals in book.items():
        n = len(vals)
        sd = statistics.stdev(vals) if n >= 2 else float("nan")
        mean = statistics.mean(vals)
        mde20 = K_80_05 * sd / math.sqrt(BAR_N) if sd == sd else float("nan")
        mde_n = K_80_05 * sd / math.sqrt(n) if (sd == sd and n) else float("nan")
        n_req = (K_80_05 * sd / TOLL_BP) ** 2 if sd == sd else float("nan")
        rows.append((sid, n, mean, sd, mde20, mde_n, n_req))

    # Least-measurable first: the top of this table is where the bar is emptiest.
    rows.sort(key=lambda r: (-(r[4] if r[4] == r[4] else -1.0)))

    hdr = "%-26s %4s %9s %9s %10s %10s %8s %s" % (
        "strategy", "n", "mean bp", "sd bp", "MDE@n=20", "MDE@own n", "n_req", "powered?"
    )
    print(hdr)
    print("-" * len(hdr))
    powered = unpowered = unknown = 0
    for sid, n, mean, sd, mde20, mde_n, n_req in rows:
        if sd != sd:
            print("%-26s %4d %9.1f %9s %10s %10s %8s %s"
                  % (sid, n, mean, "  n<2", "     --", "     --", "    --", "UNKNOWN"))
            unknown += 1
            continue
        ok = n >= n_req
        powered += ok
        unpowered += (not ok)
        print("%-26s %4d %+9.1f %9.1f %10.1f %10.1f %8.0f %s"
              % (sid, n, mean, sd, mde20, mde_n, n_req, "yes" if ok else "NO"))

    print()
    print(f"  powered at their OWN n (n >= n_req): {powered}")
    print(f"  NOT powered                        : {unpowered}")
    print(f"  undeterminable (n<2)               : {unknown}")
    print()
    measurable = [r for r in rows if r[3] == r[3]]
    blind = [r for r in measurable if r[4] > TOLL_BP]
    print(f"  BLIND AT THE BAR -- MDE@n=20 exceeds the {TOLL_BP}bp toll, so clearing")
    print(f"  n=20 cannot distinguish a toll-sized edge from zero: "
          f"{len(blind)} of {len(measurable)}")
    if measurable:
        sd_ceiling = TOLL_BP * math.sqrt(BAR_N) / K_80_05
        print(f"  the sd at which n=20 is exactly powered: {sd_ceiling:.1f} bp")
        over = [r for r in measurable if r[3] > sd_ceiling]
        print(f"  strategies above that sd ceiling: {len(over)} of {len(measurable)}")
        _recommend(measurable, sd_ceiling)
    return 0


def _recommend(measurable: List[tuple], sd_ceiling: float) -> None:
    """Argue both shapes for the bar with the numbers, then pick one.

    Criterion 3 of ``[c48bb3fc]``. The recommendation is printed rather than
    written up elsewhere so that the command that produces the numbers is the
    same command that produces the argument resting on them.
    """
    rate = 5.6  # tradeable ghost trades/day, the banner's own evidence rate
    print()
    print("=" * 86)
    print("RECOMMENDATION -- both shapes argued, one picked")
    print("=" * 86)
    print()
    print("A. PER-STRATEGY POWER-BASED n, bar = (k*sd/toll)^2 per strategy.")
    worst = max(measurable, key=lambda r: r[6])
    best = min(measurable, key=lambda r: r[6])
    print(f"   Honest about what each strategy needs. But it is degenerate at BOTH")
    print(f"   ends. Widest: {worst[0]} needs {worst[6]:.0f} rows = "
          f"{worst[6]/rate:.0f} days at {rate}/day.")
    print(f"   Narrowest: {best[0]} needs {best[6]:.0f} rows -- a licence on "
          f"n={best[1]}.")
    print("   And it is SELF-REFERENTIAL: n_req is computed from the sd of the very")
    print("   sample being sized, so a lucky-tight early sample LOWERS ITS OWN BAR.")
    print("   That is selection on the outcome, the failure mode this repo has")
    print("   shipped before. Option A is rejected on that alone.")
    print()
    print(f"B. FLAT n=20 PLUS A VARIANCE CEILING at sd > {sd_ceiling:.1f} bp.")
    print("   Not a longer wait -- a REFUSAL. A strategy whose sd makes a")
    print("   toll-sized edge invisible at any reachable n is not a slow")
    print("   candidate, it is an unprovable one.")
    over = sorted((r for r in measurable if r[3] > sd_ceiling), key=lambda r: -r[3])
    print(f"   Refuses {len(over)} of {len(measurable)} measurable strategies, "
          f"including:")
    for r in over[:4]:
        print(f"     {r[0]:<26} sd {r[3]:7.1f} bp, needs {r[6]:5.0f} rows = "
              f"{r[6]/rate:4.0f} days")
    print()
    print("   PICK B. Three reasons, each a number above:")
    lead = next((r for r in measurable if r[0] == "atf_static_scout"), worst)
    print(f"   1. The status quo is already called too slow at ~62 days. Option A")
    print(f"      asks {lead[6]/rate:.0f} days of the current front-runner "
          f"({lead[0]}) and")
    print(f"      {worst[6]/rate:.0f} days of the widest. A bar nobody can reach in")
    print("      time is not a bar.")
    print("   2. B can only TIGHTEN. It refuses; it cannot promote anything the")
    print("      current bar refuses. A change to a live gate that can only")
    print("      subtract candidates cannot cost money.")
    print("   3. B is not self-referential in the dangerous direction: an")
    print("      underestimated sd from a small sample lets a strategy through to")
    print("      the SAME n=20 bar it faces today, so the worst case is the")
    print("      status quo.")
    print()
    print("   UNCOMFORTABLE CONSEQUENCE, stated because it is the point: B refuses")
    print("   atf_static_scout, which LEADS the closest-to-the-bar table. At sd")
    print("   510.2 bp it needs 663 rows; it has 111 and an MDE of 135.7 bp, still")
    print("   2.4x the toll. Its positive record is not evidence of an edge, it is")
    print("   evidence of dispersion. B says so; the shipped bar does not.")
    print()
    print("   NOT SHIPPED THIS PASS, by criterion 4. The change would land on")
    print("   services/tradeable_evidence.py:324 (min_trades) and")
    print("   services/strategy_population.py:333/:364, as an ADDITIONAL blocker")
    print("   next to the existing three, and needs its own back-to-back item.")


if __name__ == "__main__":
    raise SystemExit(main())
