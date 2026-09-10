"""The graduation book, split on the population the graduation bar actually reads.

WHY THIS EXISTS
---------------
``scripts/readiness_report.py`` prints each strategy's POOLED ghost book, and
``trading/strategies/ledger.py`` grants a licence off the LIVE-TRADEABLE subset
of it (``_evaluate_graduation_locked`` -> ``_tradeable_of``; for a demoted
strategy, ``_maybe_rearm_locked`` -> ``_fresh_tradeable_delta``). Those are
different populations, and measured 2026-09-10 on the real ledger they differ
by more than a factor of ten AND BY SIGN:

    strategy            pooled ghost                 live-tradeable subset
    atf_static          52 trades  56%  +1.5407      4 trades  50%  -0.018689
    atf_static_scout   236 trades  79%  +6.4818      3 trades  33%  -0.077726

So a status built on the pooled number reports "READY BUT UNSTAMPED -- two
strategies clear the bar and carry no approval", which sends the pass to go fix
a stamp. The stamp is working. It is refusing, correctly, because on the
symbols the live lane can actually spend on both strategies are UNDERWATER.

This script answers the question the pooled report cannot: over the round trips
the live lane could really have placed, what does the book look like, per
strategy and per symbol?

INDEPENDENT SOURCE, ON PURPOSE
------------------------------
It reads ``trade_outcomes`` rather than the ledger. The ledger's ``tradeable``
sub-counter was baselined to zero when the subset landed (ledger.py:445-456), so
it holds days, not history -- atf_static reads 4 trades there because the
counter is young, not because only 4 such trades exist. ``trade_outcomes`` is
the append-only record of what actually closed, so it can show the tradeable
book over any window. The two agree on scale where they overlap (19 tradeable
ghost closes across all strategies in the 3 days since the baseline, of which
atf_static holds 4 and atf_static_scout 3), which is the check that the young
counter is young rather than broken.

Tradeability is judged with the SAME predicate the ledger uses --
``trading.pipeline.stop_is_unenforceable`` -- so no new threshold is
calibrated here and the two cannot drift apart.

Run:  python -X utf8 scripts/tradeable_book.py
      python -X utf8 scripts/tradeable_book.py --days 3 --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DB = ROOT / "storage" / "trading_cache.db"

# ONE implausibility test for every ``trade_outcomes`` read in the repo. See
# ``services/outcome_plausibility.py`` for why it has two arms and why the
# ratio arm is symmetric; ``IMPLAUSIBLE_RET`` is re-exported under its original
# name so this file's callers and tests are unaffected by the move.
from services.outcome_plausibility import (    # noqa: E402
    IMPLAUSIBLE_RET,
    is_implausible as _row_is_implausible,
    strategy_scales as _strategy_scales,
    sweep as _ret_sweep,
)


def _tradeable_predicate():
    """The ledger's own tradeability test, or None when it cannot be loaded.

    Returning None rather than a fail-open lambda is deliberate: a report that
    silently counted every symbol as tradeable would recreate the exact
    misreading this script exists to correct.

    THIS DELEGATES; IT MUST NOT REIMPLEMENT. Until 2026-09-10 it carried its
    own copy of the rule -- ``not stop_is_unenforceable(sym)`` -- and the two
    drifted apart the moment ``ledger._live_tradeable`` also began consulting
    ``services.symbol_edge_gate``. The drift was not cosmetic: this script is
    the tool that NAMES THE WALL, and it went on printing BASECAT-USDC and
    COMP-USDC as "live: yes" (12 trips -0.0299 and 11 trips -0.1171 over 7
    days) after graduation had stopped counting them, so the report and the
    bar it reports on disagreed about which trades exist.

    A second copy of a predicate is a second answer to one question, and this
    repo has shipped that shape before. Import the one the ledger uses.
    """
    try:
        from trading.strategies.ledger import _live_tradeable
    except Exception:  # noqa: BLE001
        return None

    def _ok(symbol: str, strategy_id: Optional[str] = None) -> bool:
        try:
            return bool(_live_tradeable(symbol, strategy_id))
        except TypeError:
            # An older ledger whose predicate takes the symbol alone. Keeps
            # this report readable against a checkout that predates the
            # (strategy, symbol) ban rather than failing the whole run.
            return bool(_live_tradeable(symbol))
        except Exception:  # noqa: BLE001
            return False

    return _ok


# The receipts-derived round-trip cost: a fixed charge per trip plus a fraction
# of notional. Both matter and they behave completely differently -- the fixed
# part is what a small clip cannot outrun, and the variable part is a floor no
# clip size can move. Keeping them separate is the whole point; a single flat
# percentage hides which of the two is beating you.
COST_FIXED = 0.004047        # dollars per round trip
COST_VARIABLE = 0.003187     # fraction of notional per round trip


def _blank() -> Dict[str, Any]:
    return {"trades": 0, "wins": 0, "losses": 0, "net": 0.0,
            "gross": 0.0, "fees": 0.0, "notional": 0.0}


def _add(acc: Dict[str, Any], net: float, gross: float = 0.0,
         fees: float = 0.0, notional: float = 0.0) -> None:
    acc["trades"] += 1
    if net > 0:
        acc["wins"] += 1
    else:
        acc["losses"] += 1
    acc["net"] += net
    acc["gross"] += gross
    acc["fees"] += fees
    acc["notional"] += notional


def _rates(acc: Dict[str, Any]) -> Dict[str, Any]:
    """Per-trip economics as PERCENTAGES OF NOTIONAL, which is the only unit
    in which an edge and a cost can be compared.

    Dollars cannot answer "does this book pay for itself" when the clip varies
    by symbol; this repo has already shipped a rate subtracted from a dollar
    amount. ``gross_pct`` versus ``cost_pct`` is the QUALITY question, and
    ``gross_pct`` versus ``COST_VARIABLE`` is the version of it that no clip
    size can rescue.
    """
    n = acc["notional"]
    trips = acc["trades"]
    clip = n / trips if trips else 0.0
    gross_pct = 100.0 * acc["gross"] / n if n else 0.0
    cost_pct = 100.0 * acc["fees"] / n if n else 0.0
    return {
        "clip": clip,
        "gross_pct": gross_pct,
        "cost_pct": cost_pct,
        "net_pct": 100.0 * acc["net"] / n if n else 0.0,
        # What the receipts model says the cost SHOULD be at this clip. When it
        # tracks cost_pct the model is the right shape and its clip curve can
        # be trusted; when it does not, the curve is a guess.
        "model_cost_pct": 100.0 * (COST_FIXED / clip + COST_VARIABLE) if clip else 0.0,
        # The floor. gross_pct must beat this or no clip size ever helps.
        "variable_floor_pct": 100.0 * COST_VARIABLE,
        "clears_floor": gross_pct > 100.0 * COST_VARIABLE,
    }


def cost_at_clip(clip: float) -> float:
    """Modelled round-trip cost as a percentage of notional at ``clip`` dollars."""
    if clip <= 0:
        return float("inf")
    return 100.0 * (COST_FIXED / clip + COST_VARIABLE)


def _win_rate(acc: Dict[str, Any]) -> float:
    return acc["wins"] / acc["trades"] if acc["trades"] else 0.0


def load_rows(db_path: Path, since_ts: float) -> List[Dict[str, Any]]:
    """Closed ghost outcomes since ``since_ts``, with strategy id and symbol.

    ``annulled`` rows are excluded: an annulled outcome is one the system has
    already decided did not happen, and counting it would put a retracted trade
    back into the evidence the bar reads.
    """
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            "SELECT symbol, status, net_profit, gross_profit, fee_cost, "
            "entry_price, exit_price, quantity, details, ts "
            "FROM trade_outcomes WHERE ts > ? ORDER BY ts",
            (since_ts,),
        )
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            if str(r["status"] or "").lower() != "closed":
                continue
            try:
                det = json.loads(r["details"] or "{}")
            except Exception:  # noqa: BLE001
                det = {}
            if not isinstance(det, dict):
                det = {}
            out.append({
                "symbol": str(r["symbol"] or ""),
                "strategy_id": str(det.get("strategy_id") or "") or "unclassified",
                "mode": str(det.get("mode") or "").lower(),
                # The exit reason and both legs' prices, so an overshoot fill
                # can be re-priced at its own limit. See ``clamped_gross``.
                "reason": str(det.get("reason") or ""),
                "entry_price": float(r["entry_price"] or 0.0),
                "exit_price": float(r["exit_price"] or 0.0),
                "quantity": abs(float(r["quantity"] or 0.0)),
                "net": float(r["net_profit"] or 0.0),
                "gross": float(r["gross_profit"] or 0.0),
                "fees": float(r["fee_cost"] or 0.0),
                # Notional at ENTRY. abs() because a short's quantity is
                # signed and the cost is charged on the size either way.
                "notional": abs(float(r["entry_price"] or 0.0)
                                * float(r["quantity"] or 0.0)),
                "ts": float(r["ts"] or 0.0),
            })
        return out
    finally:
        con.close()


def collect(
    *,
    days: float = 7.0,
    db_path: Optional[Path] = None,
    now: Optional[float] = None,
    rows: Optional[Iterable[Dict[str, Any]]] = None,
    is_tradeable=None,
) -> Dict[str, Any]:
    """Split the ghost book on live-tradeability, per strategy and per symbol."""
    now = time.time() if now is None else float(now)
    if rows is None:
        rows = load_rows(Path(db_path or DEFAULT_DB), now - days * 86400.0)
    rows = [r for r in rows if str(r.get("mode", "")).lower() != "live"]

    if is_tradeable is None:
        is_tradeable = _tradeable_predicate()
    if is_tradeable is None:
        return {
            "error": "cannot import trading.pipeline.stop_is_unenforceable; "
                     "tradeability is unjudgeable and no split is reported",
            "days": days,
            "rows": len(rows),
        }

    pooled = _blank()
    tradeable = _blank()
    untradeable = _blank()
    # THE SAME BOOK, DE-CONTAMINATED, because the raw one answers the
    # direction-or-cost question WRONG. `clamped_gross` and IMPLAUSIBLE_RET
    # already existed in this file and were used only by `symbol_edge` below;
    # the headline verdict went on reading the raw gross and printing "a
    # POSITIVE gross edge means this is a cost problem" off +0.2625% that is
    # TWO rows (UNI-USDC +122.89%, BASELINE-USDC +57.94%). De-contaminated it
    # is -0.1227% over 107 trips: NEGATIVE, which is the opposite verdict.
    #
    # Two filters, both of which the item's acceptance criteria name:
    #   - re-price a limit exit that booked past its own limit (the sampling
    #     gap, not a fill -- see `clamped_gross`); and
    #   - drop rows still |return| > 50% after that, which are repricing
    #     artifacts, and rows with no strategy_id, which no strategy can
    #     spend because nothing can be attributed to it.
    sane = _blank()
    sane_dropped = {"implausible": 0, "unattributed": 0, "clamped": 0}
    # Each strategy's own scale over THIS window, which is what the dollar arm
    # of the implausibility test judges a row against -- the read-side mirror of
    # ``ledger._recent_scale``. Computed once; it is a property of the
    # population, not of a row.
    scales = _strategy_scales(rows)
    per_strategy: Dict[str, Dict[str, Any]] = {}
    per_symbol: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        net = float(r["net"])
        gross = float(r.get("gross", 0.0) or 0.0)
        fees = float(r.get("fees", 0.0) or 0.0)
        notional = float(r.get("notional", 0.0) or 0.0)
        sym = r["symbol"]
        sid = r["strategy_id"]
        ok = bool(is_tradeable(sym))
        _add(pooled, net, gross, fees, notional)
        _add(tradeable if ok else untradeable, net, gross, fees, notional)

        if ok:
            c = clamped_gross(r)
            if c["overshot"]:
                sane_dropped["clamped"] += 1
            if _row_is_implausible(r, scales=scales):
                sane_dropped["implausible"] += 1
            elif sid == "unclassified":
                sane_dropped["unattributed"] += 1
            else:
                # Net moves by the same delta as gross so the two stay
                # consistent; the fee leg is unaffected by where inside its
                # own limit the exit printed.
                _add(sane, net - (gross - c["gross"]), c["gross"],
                     fees, notional)

        st = per_strategy.setdefault(
            sid, {"id": sid, "pooled": _blank(), "tradeable": _blank(),
                  "untradeable": _blank()})
        _add(st["pooled"], net, gross, fees, notional)
        _add(st["tradeable"] if ok else st["untradeable"],
             net, gross, fees, notional)

        sy = per_symbol.setdefault(sym, {"symbol": sym, "tradeable": ok,
                                         "book": _blank()})
        _add(sy["book"], net, gross, fees, notional)

    for st in per_strategy.values():
        for key in ("pooled", "tradeable", "untradeable"):
            st[key]["win_rate"] = _win_rate(st[key])
            st[key]["rates"] = _rates(st[key])
    for sy in per_symbol.values():
        sy["book"]["win_rate"] = _win_rate(sy["book"])
        sy["book"]["rates"] = _rates(sy["book"])
    for acc in (pooled, tradeable, untradeable, sane):
        acc["win_rate"] = _win_rate(acc)
        acc["rates"] = _rates(acc)

    return {
        "generated_at": now,
        "days": days,
        "pooled": pooled,
        "tradeable": tradeable,
        "untradeable": untradeable,
        "sane": sane,
        "sane_dropped": sane_dropped,
        # THE THRESHOLD CURVE, over the LIVE-TRADEABLE rows this verdict is read
        # from. IMPLAUSIBLE_RET is calibrated ABOVE most rows the ledger already
        # calls fabricated, so a single cap must never be the only number a
        # reader gets. Raised by Jet, pass 102; see outcome_plausibility.sweep.
        "ret_sweep": _ret_sweep([r for r in rows if is_tradeable(r["symbol"])]),
        "strategies": sorted(
            per_strategy.values(), key=lambda s: -s["tradeable"]["trades"]),
        "symbols": sorted(per_symbol.values(), key=lambda s: -s["book"]["trades"]),
    }


def _fmt(acc: Dict[str, Any]) -> str:
    return "%5d %4.0f%% %+9.4f" % (
        acc["trades"], _win_rate(acc) * 100.0, acc["net"])


def render(r: Dict[str, Any]) -> str:
    if r.get("error"):
        return "  ERROR: %s" % r["error"]
    out: List[str] = []
    out.append("=" * 74)
    out.append("  THE GRADUATION BOOK, SPLIT ON WHAT THE LIVE LANE CAN SPEND ON")
    out.append("  window: %.1f days   (ghost round trips only, annulled excluded)"
               % r["days"])
    out.append("=" * 74)
    out.append("")
    out.append("  %-22s %5s %5s %10s" % ("population", "trips", "win", "net"))
    out.append("  %-22s %s" % ("POOLED (readiness)", _fmt(r["pooled"])))
    out.append("  %-22s %s" % ("LIVE-TRADEABLE", _fmt(r["tradeable"])))
    out.append("  %-22s %s" % ("untradeable", _fmt(r["untradeable"])))
    out.append("")
    out.append("  The bar reads the LIVE-TRADEABLE row. If that row is negative,")
    out.append("  the wall is QUALITY over the spendable population -- not an")
    out.append("  unwritten stamp, and not missing evidence.")
    out.append("")

    # --- is it direction, or is it cost? -----------------------------------
    tr = r["tradeable"]
    ra = tr.get("rates") or _rates(tr)
    out.append("  " + "-" * 70)
    out.append("  IS IT DIRECTION, OR IS IT COST?  (live-tradeable only)")
    out.append("  " + "-" * 70)
    out.append("    gross %+9.4f   fees %9.4f   net %+9.4f   notional %9.2f"
               % (tr["gross"], tr["fees"], tr["net"], tr["notional"]))
    out.append("    clip $%.3f per round trip" % ra["clip"])
    out.append("    gross %+.4f%% of notional   cost %.4f%%   net %+.4f%%"
               % (ra["gross_pct"], ra["cost_pct"], ra["net_pct"]))
    out.append("    receipts model at this clip %.4f%%  (%s)"
               % (ra["model_cost_pct"],
                  "tracks the measured cost, so the clip curve below is sound"
                  if abs(ra["model_cost_pct"] - ra["cost_pct"]) < 0.15
                  else "does NOT track the measured cost; treat the curve as a guess"))
    # THE VERDICT IS READ OFF THE DE-CONTAMINATED BOOK, NOT THE RAW ONE.
    #
    # The raw gross above is reported because it is what the rows say, but it
    # must never be the thing the verdict is computed from: a limit exit that
    # booked the tick which CROSSED its target credits the position with the
    # gap between two samples, and a handful of those rows have twice carried
    # this report to the opposite conclusion. Measured 2026-09-10: raw
    # +0.2625% over 109 trips is TWO rows (UNI-USDC +122.89%, BASELINE-USDC
    # +57.94%); de-contaminated it is -0.1227% over 107, which is DIRECTION,
    # not COST. See `clamped_gross` and IMPLAUSIBLE_RET.
    sane = r.get("sane") or _blank()
    sa = sane.get("rates") or _rates(sane)
    dropped = r.get("sane_dropped") or {}
    out.append("")
    out.append("  DE-CONTAMINATED (the verdict is computed from THIS row)")
    out.append("    %d trips  %.0f%% win  net %+.4f   gross %+.4f%% of notional"
               % (sane["trades"], _win_rate(sane) * 100.0, sane["net"],
                  sa["gross_pct"]))
    out.append("    excluded: %d limit exits re-priced at their limit, "
               "%d rows still |return|>%.0f%% (repricing artifacts), "
               "%d rows with no strategy_id"
               % (int(dropped.get("clamped", 0)),
                  int(dropped.get("implausible", 0)), 100.0 * IMPLAUSIBLE_RET,
                  int(dropped.get("unattributed", 0))))
    out.append("")
    if sane["trades"] <= 0:
        out.append("    No attributable, plausible trips: the verdict is")
        out.append("    UNJUDGEABLE rather than positive. Get closed rows first.")
    elif sa["gross_pct"] > 0:
        out.append("    The book picks correctly and pays it away: a POSITIVE gross")
        out.append("    edge means this is a cost problem, not a direction problem.")
    else:
        out.append("    Gross is NEGATIVE: the book loses before a penny of fees.")
        out.append("    No clip and no cost cut can rescue that -- it needs an edge.")
    if (ra["gross_pct"] > 0) != (sa["gross_pct"] > 0):
        out.append("")
        out.append("    NOTE: the RAW book says %+.4f%% and would have given the"
                   % ra["gross_pct"])
        out.append("    OPPOSITE verdict. The difference is the excluded rows above.")

    # HOW MUCH OF THAT VERDICT IS THE THRESHOLD RATHER THAN THE BOOK.
    #
    # IMPLAUSIBLE_RET is 0.50 and sits ABOVE most of the rows the ledger already
    # calls fabricated: measured over 7 days, the 50% cap catches 2 of the 8 rows
    # that carry +2.6472 of a +2.3846 book, and the six it misses are BSTONK-USDC
    # at +25.35%/+23.68%/+17.87%/+17.83%/+17.28% and BASECAT at +17.31%. Two of
    # those are literally the +1.0201 of fabricated fills [c4f16946] names as the
    # evidence the re-arm rule reads for atf_static. Printing the whole curve is
    # the honest alternative to quietly choosing a smaller constant -- the real
    # fix names an overshoot by its own limit (`clamped_gross`) and annuls it.
    sw = r.get("ret_sweep") or {}
    if sw.get("steps"):
        out.append("")
        out.append("  IS THE VERDICT THE BOOK, OR THE THRESHOLD?  (live-tradeable)")
        out.append("    %-10s %6s %6s %10s %10s" % ("|ret| <=", "kept", "drop",
                                                    "gross", "net"))
        for st in sw["steps"]:
            out.append("    %-10s %6d %6d %+10.4f %+10.4f%s" % (
                "%.0f%%" % (st["max_ret"] * 100), st["kept"], st["dropped"],
                st["gross"], st["net"],
                "   <- the cap this verdict used"
                if abs(st["max_ret"] - sw["default"]) < 1e-9 else ""))
        if sw.get("verdict_is_threshold_dependent"):
            out.append("    THE SIGN FLIPS AT %.0f%%. The verdict above is a CHOICE"
                       % (sw["flips_at"] * 100))
            out.append("    OF CAP, not a measurement. Do not quote either half")
            out.append("    alone, and do not fix it by lowering the cap: annul the")
            out.append("    overshoot rows ([c4f16946] / [db76611a]).")
        else:
            out.append("    The sign holds across the sweep, so the verdict is a")
            out.append("    property of the book rather than of the cap.")
    out.append("")
    out.append("    variable cost floor %.4f%% of notional -- gross must beat THIS"
               % ra["variable_floor_pct"])
    out.append("    or no clip size ever helps.  de-contaminated gross %+.4f%%  ->  %s"
               % (sa["gross_pct"],
                  "CLEARS the floor" if sa["gross_pct"] > 100.0 * COST_VARIABLE
                  else "BELOW the floor"))
    out.append("")
    out.append("    %-12s %-12s %-14s" % ("clip", "modelled cost", "net per trip"))
    for clip in (ra["clip"] or 1.0, 5.0, 10.0, 20.0, 50.0):
        cost = cost_at_clip(clip)
        out.append("    $%-10.2f %8.4f%%     %+8.4f%%  %s"
                   % (clip, cost, ra["gross_pct"] - cost,
                      "PROFITABLE" if ra["gross_pct"] > cost else "loses"))
    out.append("")
    out.append("  %-24s %-21s %-21s" % ("strategy", "tradeable", "untradeable"))
    for st in r["strategies"][:12]:
        out.append("  %-24s %s  %s"
                   % (st["id"][:24], _fmt(st["tradeable"]), _fmt(st["untradeable"])))
    out.append("")
    out.append("  %-18s %-4s %s" % ("symbol", "live", "book"))
    for sy in r["symbols"][:16]:
        out.append("  %-18s %-4s %s"
                   % (sy["symbol"][:18], "yes" if sy["tradeable"] else "NO",
                      _fmt(sy["book"])))
    return "\n".join(out)




def _jackknife(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The book's gross edge with its single largest contributor removed.

    A symbol-admission rule has a derived minimum sample and therefore cannot
    judge a symbol with one round trip -- correctly, because one round trip is
    not evidence. But a report that prints the surviving book's gross as an
    EDGE is making exactly that claim on its behalf. Measured 2026-09-10, the
    admitted book's +0.4711% of notional is +122.89% from a single UNI-USDC
    row on a $0.59 notional; without it the same book is -0.0119% and below
    the variable floor. The same shape has now been found three times in this
    repo (AERO's +161% repricing row, BSTONK, this one), so the check is
    printed beside every edge this script reports rather than rediscovered.

    Leave-one-out on the largest |gross| row, which is the cheapest form of
    the question "is this an edge or is it one row".
    """
    if len(rows) < 2:
        return {"applies": False}
    worst = max(rows, key=lambda r: abs(float(r.get("gross", 0.0) or 0.0)))
    rest = [r for r in rows if r is not worst]
    notional = sum(float(r.get("notional", 0.0) or 0.0) for r in rest)
    gross = sum(float(r.get("gross", 0.0) or 0.0) for r in rest)
    full_n = sum(float(r.get("notional", 0.0) or 0.0) for r in rows)
    full_g = sum(float(r.get("gross", 0.0) or 0.0) for r in rows)
    return {
        "applies": True,
        "symbol": worst["symbol"],
        "row_gross": float(worst.get("gross", 0.0) or 0.0),
        "row_return_pct": (100.0 * float(worst.get("gross", 0.0) or 0.0)
                           / float(worst["notional"]) if worst.get("notional") else 0.0),
        "gross_pct_without": 100.0 * gross / notional if notional else 0.0,
        # What share of the whole book's gross edge that ONE row supplies. Over
        # 100% means the rest of the book is negative and the row is carrying
        # the sign, not just the size.
        "share_of_edge": (100.0 * (full_g - gross) / full_g) if full_g else 0.0,
    }


def admission_refusals(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """Which symbols ``services.symbol_edge_gate`` refuses on THESE rows.

    The same ``_verdict`` the live entry path runs, fed a chosen window rather
    than the gate's own last-500 read, so the rule can be fitted on one window
    and applied to another. Re-implementing the test here to report on it
    would let the report and the gate drift apart, which is exactly how a
    dashboard ends up describing a rule the system does not run.
    """
    from services import symbol_edge_gate as gate

    per_symbol: Dict[str, List[Any]] = {}
    for r in rows:
        notional = float(r.get("notional", 0.0) or 0.0)
        if not (notional >= gate.MIN_NOTIONAL):
            continue
        gross = float(r.get("gross", 0.0) or 0.0)
        per_symbol.setdefault(str(r["symbol"]).upper(), []).append(
            gate.Trip(ret=gross / notional, gross=gross, notional=notional))

    refused: Dict[str, str] = {}
    for symbol, trips in per_symbol.items():
        if gate._never_ban(symbol):
            continue
        found = gate._verdict(trips)
        if found is not None:
            refused[symbol] = found[1]
    return refused


def with_admission_rule(
    *,
    days: float = 7.0,
    db_path: Optional[Path] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """The tradeable book before and after the symbol-admission rule.

    Reports the rule fitted on the window itself (in-sample) AND fitted on the
    round trips OLDER than the window and applied to it (out-of-sample). The
    second number is the only one that says anything about the next trade, and
    it is reported whether or not it is better -- a rule validated only on the
    data it was fitted to is a story about the past.
    """
    now = time.time() if now is None else float(now)
    db = Path(db_path or DEFAULT_DB)
    is_tradeable = _tradeable_predicate()
    if is_tradeable is None:
        return {"error": "cannot import trading.pipeline.stop_is_unenforceable"}

    window_start = now - days * 86400.0
    inside = [r for r in load_rows(db, window_start)
              if str(r.get("mode", "")).lower() != "live" and is_tradeable(r["symbol"])]
    older = [r for r in load_rows(db, 0.0)
             if r["ts"] <= window_start
             and str(r.get("mode", "")).lower() != "live"
             and is_tradeable(r["symbol"])]

    def _book(rows: List[Dict[str, Any]], refused: Dict[str, str]) -> Dict[str, Any]:
        kept = [r for r in rows if r["symbol"].upper() not in refused]
        acc = _blank()
        for r in kept:
            _add(acc, r["net"], r["gross"], r["fees"], r["notional"])
        acc["win_rate"] = _win_rate(acc)
        acc["rates"] = _rates(acc)
        acc["jackknife"] = _jackknife(kept)
        return acc

    base = _book(inside, {})
    in_sample = admission_refusals(inside)
    out_sample = admission_refusals(older)
    return {
        "days": days,
        "fitted_on_window": len(inside),
        "fitted_on_older": len(older),
        "baseline": base,
        "in_sample": {"refused": in_sample, "book": _book(inside, in_sample)},
        "out_of_sample": {"refused": out_sample, "book": _book(inside, out_sample)},
    }


def render_rule(r: Dict[str, Any]) -> str:
    if r.get("error"):
        return "  ERROR: %s" % r["error"]
    out: List[str] = []
    out.append("  " + "=" * 70)
    out.append("  THE SYMBOL-ADMISSION RULE APPLIED TO THE LIVE-TRADEABLE BOOK")
    out.append("  services.symbol_edge_gate._verdict, min sample %d (derived)"
               % __import__("services.symbol_edge_gate", fromlist=["x"]).MIN_SAMPLES)
    out.append("  " + "=" * 70)

    def line(label: str, acc: Dict[str, Any]) -> str:
        ra = acc["rates"]
        return ("  %-30s %4d trips %4.0f%% win  net %+9.4f  gross %+.4f%% "
                "vs floor %.4f%%" % (label, acc["trades"], acc["win_rate"] * 100.0,
                                     acc["net"], ra["gross_pct"],
                                     ra["variable_floor_pct"]))

    out.append(line("no rule (baseline)", r["baseline"]))
    for key, title in (("in_sample", "IN-SAMPLE (fitted on window)"),
                       ("out_of_sample", "OUT-OF-SAMPLE (fitted on older)")):
        sec = r[key]
        out.append("")
        out.append("  %s -- refuses %s" % (title, ", ".join(sorted(sec["refused"])) or "nothing"))
        out.append(line("  admitted book", sec["book"]))
        acc = sec["book"]
        ra = acc["rates"]
        jk = acc.get("jackknife") or {}
        if jk.get("applies"):
            out.append("  %-30s leave-one-out: drop %s (%+.2f%% on one row, "
                       "%.0f%% of the edge) -> gross %+.4f%%"
                       % ("", jk["symbol"], jk["row_return_pct"],
                          jk["share_of_edge"], jk["gross_pct_without"]))
            if jk["gross_pct_without"] <= ra["variable_floor_pct"] < ra["gross_pct"]:
                out.append("  %-30s the edge above is ONE ROW. It does not clear "
                           "the floor without it." % "")
        # An edge that one row can take below the floor is not a clip problem,
        # so the clip curve must not be offered for it: printing "profitable
        # above $2.66" under "the edge above is ONE ROW" is the report
        # contradicting itself, and the encouraging half is the one that gets
        # quoted.
        survives = (not jk.get("applies")) or \
            jk["gross_pct_without"] > ra["variable_floor_pct"]
        if ra["gross_pct"] > ra["variable_floor_pct"] and ra["clip"] > 0 and survives:
            # gross% > FIXED/clip + VARIABLE  =>  clip > FIXED / (gross - VARIABLE)
            need = COST_FIXED / ((ra["gross_pct"] - ra["variable_floor_pct"]) / 100.0)
            out.append("  %-30s clears the variable floor; profitable above a "
                       "$%.2f clip (now $%.2f)" % ("", need, ra["clip"]))
        elif not survives:
            out.append("  %-30s no clip curve is offered: the edge does not "
                       "survive its own largest row" % "")
        else:
            out.append("  %-30s does NOT clear the variable floor -- no clip "
                       "size helps" % "")
    out.append("")
    out.append("  The out-of-sample row is the one that says anything about the")
    out.append("  next trade. It is printed whether or not it is the better one.")
    return "\n".join(out)


# The graduation bar, restated here so the per-symbol verdict is judged against
# the same three numbers ``_evaluate_graduation_locked`` reads and cannot drift.
BAR_TRADES = 20
BAR_WINRATE = 0.55

# What a ghost take-profit limit was actually set at. ``atf_static`` builds
# ``target_price`` as ``price * 1.05`` (trading/bot.py:8400, 8809, 8960), and a
# limit exit may not book the distance the price ran PAST that limit -- that is
# the gap between two samples, not a fill. Commit 5504769 fixed this forward;
# rows already in ``trade_outcomes`` still carry the overshoot, and this is what
# re-prices them for the report.
GHOST_TP_LIMIT = 0.05

# A symbol needs this many round trips before it may be called the best one.
# Half the bar's depth: below it a single contaminated tick outranks a real
# book, which is not a hypothetical -- ranked on de-contaminated net alone, a
# ONE-trip AAVE-USDC row at +3.4700 came top of a book whose whole
# de-contaminated total is +0.3035. Naming that symbol "best" is the precise
# mistake this repo has made four times (see the AERO one-row memory).
MIN_RANK_TRIPS = 10

# THE IMPLAUSIBILITY TEST NOW LIVES IN ``services.outcome_plausibility``.
#
# It was defined here, as a literal, and used only by this file -- while
# ``scripts/tradeable_symbol_edge.py`` read the same table ALL-TIME with no
# filter at all and reported AERO-USDC at gross +2.0252, which is +161% on one
# repricing row. Two tools measuring one book with two different ideas of which
# rows exist is the same defect as the tradeability predicate that drifted on
# 2026-09-10 (see ``_tradeable_predicate``), so the rule moved next to the
# ledger's own and both tools import it.
#
# The name is kept so this file's own callers and tests do not move.
#
# A booked return this far from zero is not a fill any strategy here could have
# produced, and is reported separately rather than silently averaged in.
#
# Every strategy in this book targets +5% (``GHOST_TP_LIMIT``) and stops at
# 2-4%, so the reachable band is narrow. Measured 2026-09-10, the row that this
# catches is AAVE-USDC entry 129.485 -> exit 354.990, +174.16%, booked
# ``time_take_profit:1.7416`` -- while the SAME symbol's other two round trips
# sit at 131.50 and 128.38. AAVE did not triple; the feed printed another
# asset's price, which is the denomination-contamination family this repo has
# already hit.
#
# It is NOT clamped, because ``time_take_profit`` is a TIME exit
# (trading/triggers.py:230) -- a market order at the tick, not a limit. Commit
# 1135a79 established that clamping a market exit invents a price. So the row
# is flagged and excluded from the ranking, and both totals are printed.
#
# THE CONTAMINATION DOES NOT STOP AT ONE ROW -- IT BECOMES THE NEXT ENTRY.
# The six rows this catches over 15 days are three pairs, and the AERO pair
# shows the mechanism exactly:
#
#     AERO-USDC  entry 0.436805 -> exit 1.140000   +160.99%  net +3.2066
#     AERO-USDC  entry 1.140000 -> exit 0.513839   -54.93%   net -1.1115
#
# The second trade's ENTRY IS THE FIRST TRADE'S CONTAMINATED EXIT. AERO trades
# near 0.44; the feed printed 1.14, the book took a fake +161% win on it, and
# then opened the next position at that fictional basis and booked a real-
# looking -55% stop as the price "fell" back to 0.51. COMP is the same shape
# (entry 42.82 -> exit 19.13, a -55.33% stop).
#
# So the filter is symmetric BY NECESSITY, not by preference: three of the six
# are positive (+7.39) and three are negative (-2.03). Dropping only the
# winners would be cherry-picking; dropping on |ret| removes both halves of
# each event, for a net +5.3566 of fiction removed from a book whose
# limit-re-priced total is +0.3035.
#
# DEFINED IN ``services.outcome_plausibility`` -- imported at the top of this
# file. Do not re-introduce a literal here.


def clamped_gross(row: Dict[str, Any]) -> Dict[str, Any]:
    """Re-price one closed row's gross as if its limit exit filled at its limit.

    Returns ``{"gross", "overshot", "booked_ret", "limit_ret"}``. Non-limit
    exits and rows missing a leg are returned unchanged with
    ``overshot=False``: a stop is a MARKET order and genuinely fills through
    its level, so clamping one would invent a loss the book never took. That
    distinction is the whole correction in commit 1135a79 and it is why only
    ``LIMIT_EXIT_REASONS`` are touched here.

    The clamp itself is delegated to ``trading.bot.limit_exit_fill_price`` --
    the function the live path now runs -- so the report and the code being
    reported on cannot disagree about what a limit may book.
    """
    out = {"gross": float(row.get("gross", 0.0) or 0.0), "overshot": False,
           "booked_ret": 0.0, "limit_ret": GHOST_TP_LIMIT}
    entry = float(row.get("entry_price", 0.0) or 0.0)
    exit_px = float(row.get("exit_price", 0.0) or 0.0)
    qty = float(row.get("quantity", 0.0) or 0.0)
    if entry <= 0 or exit_px <= 0 or qty <= 0:
        return out
    out["booked_ret"] = exit_px / entry - 1.0
    try:
        from trading.bot import limit_exit_fill_price
    except Exception:  # noqa: BLE001
        # Unjudgeable rather than assumed clean: report the booked number and
        # say nothing was clamped, so a missing import cannot silently produce
        # a "de-contaminated" figure that is just the contaminated one.
        return out
    fill = float(limit_exit_fill_price(
        price=exit_px, target=entry * (1.0 + GHOST_TP_LIMIT), entry=entry,
        fee_rate=COST_VARIABLE, reason=row.get("reason", ""), is_live=False))
    if fill < exit_px:
        out["overshot"] = True
        # Gross scales with the fill: the fee leg is charged separately and is
        # unaffected by where inside its own limit the exit printed.
        out["gross"] = (fill - entry) * qty
    return out


def symbol_edge(
    *,
    days: float = 7.0,
    db_path: Optional[Path] = None,
    now: Optional[float] = None,
    rows: Optional[Iterable[Dict[str, Any]]] = None,
    is_tradeable=None,
) -> Dict[str, Any]:
    """Per-symbol economics over LIVE-TRADEABLE symbols only, ranked.

    This is the table the graduation question actually needs. ``collect``
    answers "how big is the tradeable book"; this answers "is there a symbol
    inside it that any single strategy could graduate on", which is the only
    route to a live approval that survives its own demotion guards.

    Every symbol carries two gross figures. ``gross`` is what the book booked.
    ``gross_clamped`` re-prices limit exits at their limit. Reporting only the
    first would repeat the finding of 5504769 -- that a handful of sampling
    gaps ARE the book's entire edge -- and reporting only the second would hide
    how much of the record is that artifact. The verdict is judged on the
    clamped number, because that is the one a live limit order can reproduce.
    """
    now = time.time() if now is None else float(now)
    if rows is None:
        rows = load_rows(Path(db_path or DEFAULT_DB), now - days * 86400.0)
    rows = [r for r in rows if str(r.get("mode", "")).lower() != "live"]

    if is_tradeable is None:
        is_tradeable = _tradeable_predicate()
    if is_tradeable is None:
        return {"error": "cannot import trading.pipeline.stop_is_unenforceable; "
                         "tradeability is unjudgeable and no table is reported",
                "days": days, "rows": len(rows)}

    per_symbol: Dict[str, Dict[str, Any]] = {}
    grid: Dict[tuple, Dict[str, Any]] = {}
    # Item [d763940a] asks this report to NAME the implausible rows rather than
    # count them. A count cannot be acted on: an entry-basis guard cannot delete
    # history, so the honest close on that criterion is to say which rows remain
    # and, for each, whether the guard would have refused it.
    named_implausible: List[Dict[str, Any]] = []
    refused = 0
    scales = _strategy_scales(rows)
    for r in rows:
        sym = r["symbol"]
        if not is_tradeable(sym):
            refused += 1
            continue
        c = clamped_gross(r)
        net = float(r["net"])
        gross = float(r.get("gross", 0.0) or 0.0)
        fees = float(r.get("fees", 0.0) or 0.0)
        notional = float(r.get("notional", 0.0) or 0.0)
        # Net re-priced by the same delta as gross, so the two stay consistent.
        net_c = net - (gross - c["gross"])

        implausible = _row_is_implausible(r, scales=scales)

        for bucket, extra in ((per_symbol.setdefault(sym, {
                "symbol": sym, "book": _blank(), "gross_clamped": 0.0,
                "net_clamped": 0.0, "overshoots": 0, "wins_clamped": 0,
                "implausible": 0, "net_sane": 0.0, "trips_sane": 0,
                "wins_sane": 0}), None),
                (grid.setdefault((sym, r["strategy_id"]), {
                "symbol": sym, "strategy_id": r["strategy_id"],
                "book": _blank(), "gross_clamped": 0.0, "net_clamped": 0.0,
                "wins_clamped": 0, "implausible": 0, "net_sane": 0.0,
                "trips_sane": 0, "wins_sane": 0}), None)):
            _add(bucket["book"], net, gross, fees, notional)
            bucket["gross_clamped"] += c["gross"]
            bucket["net_clamped"] += net_c
            bucket["overshoots"] = bucket.get("overshoots", 0) + (
                1 if c["overshot"] else 0)
            bucket["wins_clamped"] += 1 if net_c > 0 else 0
            if implausible:
                bucket["implausible"] += 1
                if extra is None and bucket.get("strategy_id") is None:
                    named_implausible.append({
                        "symbol": sym, "strategy_id": r.get("strategy_id", ""),
                        "ts": float(r.get("ts", 0.0) or 0.0),
                        "entry": float(r.get("entry_price", 0.0) or 0.0),
                        "exit": float(r.get("exit_price", 0.0) or 0.0),
                        "net": net, "reason": str(r.get("reason", "") or ""),
                    })
            else:
                bucket["net_sane"] += net_c
                bucket["trips_sane"] += 1
                bucket["wins_sane"] += 1 if net_c > 0 else 0

    def _finish(d: Dict[str, Any]) -> Dict[str, Any]:
        b = d["book"]
        b["win_rate"] = _win_rate(b)
        b["rates"] = _rates(b)
        n = b["trades"]
        d["win_rate_clamped"] = d["wins_clamped"] / n if n else 0.0
        # The clamped gross as a percentage of notional, against the variable
        # cost floor no clip size can move. This is the QUALITY number.
        d["gross_clamped_pct"] = (100.0 * d["gross_clamped"] / b["notional"]
                                  if b["notional"] else 0.0)
        d["clears_floor"] = d["gross_clamped_pct"] > 100.0 * COST_VARIABLE
        ns = d["trips_sane"]
        d["win_rate_sane"] = d["wins_sane"] / ns if ns else 0.0
        # The bar is judged on the SANE, de-contaminated book: a licence to
        # spend real money must not rest on a tick the feed misprinted.
        d["clears_bar"] = bool(ns >= BAR_TRADES
                               and d["win_rate_sane"] >= BAR_WINRATE
                               and d["net_sane"] > 0.0)
        d["rankable"] = ns >= MIN_RANK_TRIPS
        return d

    symbols = [_finish(s) for s in per_symbol.values()]
    cells = [_finish(c) for c in grid.values()]
    # Ranked on de-contaminated, plausibility-filtered NET -- the number a live
    # round trip could actually keep. Ranking on gross, on the booked figure,
    # or without the depth floor each puts a single artifact at the top, which
    # is exactly the misreading this table exists to correct.
    symbols.sort(key=lambda s: -s["net_sane"])
    cells.sort(key=lambda c: -c["net_sane"])

    rankable = [s for s in symbols if s["rankable"]]
    best = rankable[0] if rankable else None
    winners = [c for c in cells if c["clears_bar"]]
    total = _blank()
    for s in symbols:
        for k in ("trades", "wins", "losses", "net", "gross", "fees", "notional"):
            total[k] += s["book"][k]
    total["win_rate"] = _win_rate(total)
    total["rates"] = _rates(total)
    total_clamped = sum(s["net_clamped"] for s in symbols)
    total_wins_clamped = sum(s["wins_clamped"] for s in symbols)
    total_sane = sum(s["net_sane"] for s in symbols)
    trips_sane = sum(s["trips_sane"] for s in symbols)
    wins_sane = sum(s["wins_sane"] for s in symbols)

    return {
        "generated_at": now, "days": days,
        "untradeable_rows_excluded": refused,
        "total": total,
        "total_net_clamped": total_clamped,
        "total_win_rate_clamped": (total_wins_clamped / total["trades"]
                                   if total["trades"] else 0.0),
        "total_net_sane": total_sane,
        "total_trips_sane": trips_sane,
        "total_win_rate_sane": wins_sane / trips_sane if trips_sane else 0.0,
        "implausible_rows": total["trades"] - trips_sane,
        "named_implausible": named_implausible,
        "symbols": symbols,
        "grid": cells,
        "best_symbol": best,
        "min_rank_trips": MIN_RANK_TRIPS,
        "bar": {"trades": BAR_TRADES, "win_rate": BAR_WINRATE, "net": 0.0},
        "strategies_clearing_bar": winners,
    }


def _basis_verdict(row: Dict[str, Any]) -> str:
    """What the entry-basis guard says about one already-booked row.

    Reported rather than enforced here: this script reads history, and a guard
    that runs at entry time cannot remove a row the book already holds. Naming
    the verdict is what closes criterion 3 of item [d763940a] honestly -- it
    separates "this one would now be refused" from "no source can judge it".
    """
    try:
        from services.entry_price_corroboration import (
            book_disagreement, corroborating_ticks)
    except Exception as exc:                                # pragma: no cover
        return "entry-basis guard unavailable: %s" % (exc,)
    sym, px, ts = row["symbol"], row["entry"], row.get("ts", 0.0)
    try:
        feed = corroborating_ticks(sym, px, at_ts=ts)
        book = book_disagreement(sym, px, at_ts=ts)
    except Exception as exc:                                # pragma: no cover
        return "entry-basis guard errored: %s" % (exc,)
    if feed.get("corroborated") is False:
        return "WOULD BE REFUSED (feed): %s" % feed["reason"]
    if book.get("disagrees"):
        return "WOULD BE REFUSED (book): %s" % book["reason"]
    if feed.get("corroborated") is True:
        return ("CANNOT JUDGE -- the feed CORROBORATES this price: %s. A price "
                "the feed itself published is not reachable by a plausibility "
                "threshold; if it is still wrong, the ticker carries two assets."
                % feed["reason"])
    return ("CANNOT JUDGE -- no feed coverage and no prior entry to compare "
            "against: %s / %s" % (feed.get("reason", ""), book.get("reason", "")))


def render_symbol_edge(r: Dict[str, Any]) -> str:
    if r.get("error"):
        return "ERROR: %s" % r["error"]
    out: List[str] = []
    out.append("=" * 92)
    out.append("PER-SYMBOL EDGE over LIVE-TRADEABLE symbols only -- %.1f days"
               % r["days"])
    out.append("=" * 92)
    out.append("  %d untradeable round trips excluded (the live lane refuses "
               "these symbols on sight)" % r["untradeable_rows_excluded"])
    t = r["total"]
    out.append("  TRADEABLE BOOK  %d trips  %.0f%% win  net %+.4f booked"
               % (t["trades"], 100.0 * t["win_rate"], t["net"]))
    out.append("                  %s  %.0f%% win  net %+.4f limit-exits re-priced"
               % (" " * (len(str(t["trades"])) + 6),
                  100.0 * r["total_win_rate_clamped"], r["total_net_clamped"]))
    out.append("                  %d trips  %.0f%% win  net %+.4f  ALSO dropping "
               "%d implausible row(s) (|ret| > %.0f%%)"
               % (r["total_trips_sane"], 100.0 * r["total_win_rate_sane"],
                  r["total_net_sane"], r["implausible_rows"],
                  100.0 * IMPLAUSIBLE_RET))
    named = r.get("named_implausible") or []
    if named:
        out.append("")
        out.append("  THE IMPLAUSIBLE ROWS, NAMED -- and whether the entry-basis")
        out.append("  guard (services.entry_price_corroboration) would have refused")
        out.append("  each one at the moment it was opened. A row it CANNOT judge")
        out.append("  is named as such: an entry guard cannot delete history, so a")
        out.append("  row already in the book stays in it either way.")
        for row in sorted(named, key=lambda z: z.get("ts", 0.0)):
            out.append("    %-14s entry %-12.6f exit %-12.6f net %+8.4f  %s"
                       % (row["symbol"], row["entry"], row["exit"], row["net"],
                          row.get("reason", "")))
            out.append("      %s" % _basis_verdict(row))
    out.append("")
    out.append("  %-20s %5s %5s %9s %9s %9s %7s %4s %4s" % (
        "symbol", "trips", "win%", "gross", "gross_cl", "net_sane", "gr_cl%",
        "ovs", "bad"))
    out.append("  " + "-" * 92)
    for s in r["symbols"]:
        b = s["book"]
        out.append("  %-20s %5d %4.0f%% %+9.4f %+9.4f %+9.4f %+6.3f%% %4d %4d%s" % (
            s["symbol"][:20], b["trades"], 100.0 * s["win_rate_sane"],
            b["gross"], s["gross_clamped"], s["net_sane"],
            s["gross_clamped_pct"], s["overshoots"], s["implausible"],
            "  <- clears cost floor" if s["clears_floor"] else ""))
    out.append("")
    best = r.get("best_symbol")
    if best:
        out.append("  BEST TRADEABLE SYMBOL (of those with >= %d round trips, "
                   "de-contaminated): %s"
                   % (r["min_rank_trips"], best["symbol"]))
        out.append("      %d trips, %.0f%% win, net %+.4f"
                   % (best["trips_sane"], 100.0 * best["win_rate_sane"],
                      best["net_sane"]))
        top = [c for c in r["grid"] if c["symbol"] == best["symbol"]][:4]
        for c in top:
            out.append("      %-26s %4d trips %4.0f%% win  net %+.4f  %s" % (
                c["strategy_id"][:26], c["trips_sane"],
                100.0 * c["win_rate_sane"], c["net_sane"],
                "CLEARS BAR" if c["clears_bar"] else "below bar"))
    else:
        out.append("  NO tradeable symbol has %d round trips; none is rankable."
                   % r["min_rank_trips"])
    bar = r["bar"]
    if r["strategies_clearing_bar"]:
        out.append("")
        out.append("  CLEARS %d/%.0f%%/positive ON ONE SYMBOL ALONE:"
                   % (bar["trades"], 100.0 * bar["win_rate"]))
        for c in r["strategies_clearing_bar"]:
            out.append("      %s on %s -- %d trips %.0f%% net %+.4f" % (
                c["strategy_id"], c["symbol"], c["trips_sane"],
                100.0 * c["win_rate_sane"], c["net_sane"]))
    else:
        out.append("")
        out.append("  NO strategy clears %d trades / %.0f%% / positive on ANY "
                   "single tradeable symbol." % (bar["trades"],
                                                 100.0 * bar["win_rate"]))
        deepest = max(r["grid"], key=lambda c: c["trips_sane"], default=None)
        if deepest is not None:
            out.append("      deepest cell: %s on %s -- %d trips (bar is %d), "
                       "%.0f%% win (bar is %.0f%%), net %+.4f"
                       % (deepest["strategy_id"], deepest["symbol"],
                          deepest["trips_sane"], bar["trades"],
                          100.0 * deepest["win_rate_sane"],
                          100.0 * bar["win_rate"], deepest["net_sane"]))
        # What would have to change, in the units of the thing that is short.
        need = [c for c in r["grid"] if c["net_sane"] > 0.0]
        need.sort(key=lambda c: -c["trips_sane"])
        if need:
            c = need[0]
            out.append("      the only POSITIVE cell with any depth: %s on %s "
                       "-- %d trips %.0f%% net %+.4f; it needs %d more trips "
                       "at >= %.0f%% to clear the bar."
                       % (c["strategy_id"], c["symbol"], c["trips_sane"],
                          100.0 * c["win_rate_sane"], c["net_sane"],
                          max(0, bar["trades"] - c["trips_sane"]),
                          100.0 * bar["win_rate"]))
        else:
            out.append("      NO strategy-on-symbol cell is positive at all, "
                       "so more evidence cannot graduate one: the change has "
                       "to be to the entry rule or the cost, not the volume.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--rule", action="store_true",
                    help="also apply the symbol-admission rule, in and out of sample")
    ap.add_argument("--symbols", action="store_true",
                    help="per-symbol edge over live-tradeable symbols only, "
                         "ranked, with limit-exit overshoots re-priced")
    a = ap.parse_args()
    if a.symbols:
        rep = symbol_edge(days=a.days, db_path=Path(a.db))
        print(json.dumps(rep, indent=2, default=str) if a.json
              else render_symbol_edge(rep))
        return 0
    rep = collect(days=a.days, db_path=Path(a.db))
    if a.rule:
        rep["admission_rule"] = with_admission_rule(days=a.days, db_path=Path(a.db))
    if a.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(render(rep))
        if a.rule:
            print("")
            print(render_rule(rep["admission_rule"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
