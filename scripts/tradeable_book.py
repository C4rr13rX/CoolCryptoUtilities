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


def _tradeable_predicate():
    """The ledger's own tradeability test, or None when it cannot be loaded.

    Returning None rather than a fail-open lambda is deliberate: a report that
    silently counted every symbol as tradeable would recreate the exact
    misreading this script exists to correct.
    """
    try:
        from trading.pipeline import stop_is_unenforceable
    except Exception:  # noqa: BLE001
        return None

    def _ok(symbol: str) -> bool:
        sym = str(symbol or "").strip()
        if not sym:
            return False
        try:
            return not bool(stop_is_unenforceable(sym))
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
            "entry_price, quantity, details, ts "
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
    for acc in (pooled, tradeable, untradeable):
        acc["win_rate"] = _win_rate(acc)
        acc["rates"] = _rates(acc)

    return {
        "generated_at": now,
        "days": days,
        "pooled": pooled,
        "tradeable": tradeable,
        "untradeable": untradeable,
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
    out.append("")
    if ra["gross_pct"] > 0:
        out.append("    The book picks correctly and pays it away: a POSITIVE gross")
        out.append("    edge means this is a cost problem, not a direction problem.")
    else:
        out.append("    Gross is NEGATIVE: the book loses before a penny of fees.")
        out.append("    No clip and no cost cut can rescue that -- it needs an edge.")
    out.append("")
    out.append("    variable cost floor %.4f%% of notional -- gross must beat THIS"
               % ra["variable_floor_pct"])
    out.append("    or no clip size ever helps.  gross %+.4f%%  ->  %s"
               % (ra["gross_pct"],
                  "CLEARS the floor" if ra["clears_floor"] else "BELOW the floor"))
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--rule", action="store_true",
                    help="also apply the symbol-admission rule, in and out of sample")
    a = ap.parse_args()
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
