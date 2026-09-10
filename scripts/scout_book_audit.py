"""Audit the scout ghost book on the SAME de-contamination the bar uses.

WHY THIS FILE EXISTS
--------------------
``atf_static_scout`` is the biggest ghost book in the system -- the ledger
credits it with 237 trades and +6.4498 -- and NOT ONE of its round trips has
ever been seen by a de-contamination instrument. The cause is structural and
was pinned to a line by Cove, pass 103: ``db.record_trade_outcome`` is the only
thing that INSERTs into ``trade_outcomes`` (db.py:961/992) and it is called
from exactly one site in the tree, ``trading/bot.py:9968``. The scout books its
exits at ``services/atf_static_strategy.py:973`` with ``db.log_trade(...
status="ghost-exit")``, which writes a ``trading_ops`` row and nothing else.
So the scout's exits are structurally incapable of reaching the receipt table,
and every reader that de-contaminates the book -- ``outcome_plausibility``, the
take-profit clamp, ``scripts/tradeable_book.collect`` -- reads only
``trade_outcomes`` and therefore reports on a book the scout is absent from.

That left the largest book in the system carrying an UNAUDITED +6.4498 into
every "the ghost book is positive" reading anyone has taken.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT DO
----------------------------------------------------
It reconstructs the scout's round trips from the ONE table that holds them --
``trading_ops`` ``ghost-exit`` rows whose details name the scout -- and feeds
them to ``scripts.tradeable_book.collect(rows=...)`` UNMODIFIED. Every rule,
bar, threshold and predicate is the imported one:

  * ``is_tradeable``            -- ``tradeable_book._tradeable_predicate()``
  * the overshoot re-pricing    -- ``tradeable_book.clamped_gross``
  * the implausibility test     -- ``outcome_plausibility.is_implausible``
  * the per-strategy dollar scale -- ``tradeable_book._strategy_scales``

Nothing here defines a threshold of its own. This is an INSTRUMENT: it changes
no rule and writes nothing back. It exists so the scout's book can be read on
the same terms as everybody else's, not so it can be graded on kinder ones.

IT DOES NOT BACKFILL ``trade_outcomes``. ``services/tradeable_evidence.py``
fails closed on exactly that, and for a reason that still holds: the ledger's
237 and the 109 exits ``trading_ops`` can show disagree by roughly 2x, the
window boundary between them is unrecoverable, and a backfill would be
inventing the difference rather than measuring it. This reports the 109 rows
that EXIST and says plainly that they are a floor, not the whole book.

(109, not the 120 a ``LIKE '%atf_static_scout%'`` over the details blob
returns: 11 of those 120 are other writers' rows that MENTION the scout --
a refusal names the strategy it refused. Attribution is on
``details.strategy_id`` and nothing else. See
``tests/test_the_scout_book_is_not_read_as_dollars.py``.)

WHAT IT MEASURED, 2026-09-10 (pass 104, Gale)
---------------------------------------------
Of the 109 exits, 105 carry NO ``profit_unit`` -- they are bare fractions that
were never charged a fee, summing to +2.0705 -- and 4 are corrected USD rows
summing to -0.1097. The ledger's +6.4498 is therefore built on percentages
added to dollars, and the only rows ever charged a cost are NEGATIVE.

Split on the live-lane predicate, in return space where no notional is needed:
55 tradeable trips (+23.16% summed) against 54 untradeable (+183.60%). 89% of
the book's raw return is in symbols the live lane REFUSES -- BSTONK +89.18%,
BASECAT +38.51%, BPAD +37.77%, MOONBASE +16.08%. Against the full round-trip
cost (0.3862% at the scout's $6 clip) the TRADEABLE half is mean excess
+0.0349% per trip at t=+0.24 with a 29% win rate after cost, against a 55%
bar; dropping AERO does not rescue it (t=+0.20). That is no edge
distinguishable from zero, on the half that is the only half graduation could
ever spend.

UNITS AT THE BOUNDARY -- the scout writes a different shape to the receipt
table's, so the mapping is spelled out rather than assumed. At
``atf_static_strategy.py:945`` ``profit_usd = return_pct * clip_usd -
roundtrip_cost_usd(clip_usd)``, i.e. the detail key ``profit`` is USD and is
NET of the round trip. Therefore:

    net      = details["profit"]                      (USD, net)
    fees     = details["roundtrip_cost_usd"]          (USD)
    gross    = net + fees                             (USD, pre-cost)
    notional = details["clip_usd"]                    (USD at entry)
    quantity = clip_usd / entry_price                 (base units)

``notional`` is the entry clip, matching ``load_rows``' ``entry_price *
quantity``. Getting this wrong is the bug this repo has shipped most often --
a fraction added to a dollar -- so ``--verify-units`` re-derives net from
return_pct and clip and reports the worst disagreement.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tradeable_book import (  # noqa: E402
    DEFAULT_DB,
    collect,
    _tradeable_predicate,
    _win_rate,
)

SCOUT_STRATEGY_ID = "atf_static_scout"


def _f(val: Any, default: float = 0.0) -> float:
    try:
        out = float(val)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def load_scout_rows(db_path: Path, since_ts: float,
                    strategy_id: str = SCOUT_STRATEGY_ID) -> List[Dict[str, Any]]:
    """Scout round trips from ``trading_ops``, in ``load_rows``' row shape.

    The filter is on the details' ``strategy_id``, not on a LIKE over the JSON
    blob: the scout's own id appears inside other writers' payloads (the
    refusal rows name it too), and counting a refusal as a round trip would
    inflate the very book this is trying to measure.
    """
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            "SELECT symbol, details, ts FROM trading_ops "
            "WHERE status = 'ghost-exit' AND ts > ? ORDER BY ts",
            (since_ts,),
        )
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            try:
                det = json.loads(r["details"] or "{}")
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(det, dict):
                continue
            if str(det.get("strategy_id") or "") != strategy_id:
                continue

            entry = _f(det.get("entry_price"))
            clip = _f(det.get("clip_usd"))
            net = _f(det.get("profit"))
            fees = _f(det.get("roundtrip_cost_usd"))
            # ``profit_unit`` was added when this writer was corrected from
            # fractions to dollars. A row that predates it, or says anything
            # else, is NOT silently treated as USD -- that is the exact
            # confusion that put 104 fraction rows into a dollar book.
            if str(det.get("profit_unit") or "").lower() != "usd":
                continue
            out.append({
                "symbol": str(r["symbol"] or det.get("symbol") or ""),
                "strategy_id": strategy_id,
                # Ghost by construction: this writer has no live branch.
                "mode": "ghost",
                "reason": str(det.get("exit_reason") or det.get("reason") or ""),
                "entry_price": entry,
                "exit_price": _f(det.get("exit_price")),
                "quantity": (clip / entry) if entry > 0 else 0.0,
                "net": net,
                "gross": net + fees,
                "fees": fees,
                "notional": clip,
                "ts": _f(det.get("exit_ts"), _f(r["ts"])),
                # Kept for --verify-units only; no consumer in collect() reads it.
                "_return_pct": _f(det.get("return_pct")),
            })
        return out
    finally:
        con.close()


def verify_units(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Re-derive gross from return_pct * notional and report the worst gap.

    ``gross`` is reconstructed as ``net + fees``; it must equal
    ``return_pct * clip_usd`` by the writer's own arithmetic. A disagreement
    means the mapping above is wrong and every number downstream of it is too.
    """
    worst = 0.0
    worst_row: Optional[Dict[str, Any]] = None
    for r in rows:
        expect = r["_return_pct"] * r["notional"]
        gap = abs(expect - r["gross"])
        if gap > worst:
            worst, worst_row = gap, r
    return {
        "rows": len(rows),
        "worst_abs_gap_usd": worst,
        "worst_symbol": (worst_row or {}).get("symbol", ""),
    }


def _pct_of_notional(acc: Dict[str, Any], key: str) -> Optional[float]:
    notional = _f(acc.get("notional"))
    if notional <= 0:
        return None
    return 100.0 * _f(acc.get(key)) / notional


def render(res: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append("=" * 74)
    out.append("THE SCOUT GHOST BOOK, ON THE SAME PREDICATE AS EVERYBODY ELSE")
    out.append("=" * 74)
    out.append("")
    out.append("  source: trading_ops ghost-exit rows naming %s" % SCOUT_STRATEGY_ID)
    out.append("          (trade_outcomes holds ZERO of these -- see module docstring)")
    out.append("  window: %.1f days" % res["days"])
    out.append("")

    units = res.get("units") or {}
    if units:
        out.append("  units check: gross vs return_pct*clip, worst gap $%.8f over %d rows"
                   % (units.get("worst_abs_gap_usd", 0.0), units.get("rows", 0)))
        out.append("")

    hdr = "  %-14s %6s %6s %11s %11s" % ("book", "trips", "win", "net USD", "gross %")
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))
    for label in ("pooled", "tradeable", "untradeable", "sane"):
        acc = res.get(label) or {}
        if not acc:
            continue
        gp = _pct_of_notional(acc, "gross")
        out.append("  %-14s %6d %5.0f%% %+11.4f %11s" % (
            label, acc.get("trades", 0), _win_rate(acc) * 100.0,
            _f(acc.get("net")),
            ("%+.4f%%" % gp) if gp is not None else "n/a",
        ))
    out.append("")

    dropped = res.get("sane_dropped") or {}
    out.append("  de-contamination, on the TRADEABLE rows:")
    out.append("    %d re-priced to their own limit (overshoot)" % dropped.get("clamped", 0))
    out.append("    %d dropped implausible" % dropped.get("implausible", 0))
    out.append("    %d dropped unattributed" % dropped.get("unattributed", 0))
    out.append("")
    out.append("  'sane' is the de-contaminated TRADEABLE book: the only one of")
    out.append("  these four the graduation bar would ever read.")

    rs = res.get("return_space") or {}
    if rs.get("trips"):
        out.append("")
        out.append("-" * 74)
        out.append("  THE WHOLE BOOK IN RETURN SPACE -- all %d exits, legacy included"
                   % rs["trips"])
        out.append("  (%d stamped USD, %d unstamped fractions the dollar report above"
                   % (rs.get("stamped", 0), rs.get("unstamped", 0)))
        out.append("   CANNOT price -- a fraction with no clip is not dollars)")
        out.append("")
        out.append("  cost charged: %.4f%% per trip (fixed leg over a $%.2f clip)"
                   % (rs.get("cost_rate_pct", 0.0), rs.get("clip_usd", 0.0)))
        out.append("")
        hdr2 = "  %-22s %6s %11s %11s %7s %8s" % (
            "half", "trips", "sum ret", "mean exc", "t", "win/cost")
        out.append(hdr2)
        out.append("  " + "-" * (len(hdr2) - 2))
        for label, key in (("TRADEABLE", "tradeable"),
                           ("UNTRADEABLE", "untradeable"),
                           ("  tradeable ex-AERO", "tradeable_ex_aero")):
            a = rs.get(key) or {}
            if not a.get("trips"):
                out.append("  %-22s %6d" % (label, 0))
                continue
            out.append("  %-22s %6d %+10.2f%% %+10.4f%% %+7.2f %7.0f%%" % (
                label, a["trips"], a["sum_return_pct"], a["mean_excess_pct"],
                a["t_on_excess"], 100.0 * a["win_rate_after_cost"]))
        out.append("")
        out.append("  The TRADEABLE half is the only half graduation could spend.")
        out.append("  Read its t and its win-rate-after-cost against the 55% bar.")
    return "\n".join(out)


def return_space(db_path: Path, since_ts: float,
                 strategy_id: str = SCOUT_STRATEGY_ID,
                 clip_usd: float = 6.0) -> Dict[str, Any]:
    """The WHOLE book -- legacy rows included -- measured in RETURN space.

    ``load_scout_rows`` deliberately refuses the 105 unstamped rows, because
    their ``profit`` is a fraction and there is no ``clip_usd`` to turn it into
    dollars. That refusal is correct and it is also why the dollar report above
    covers only 4 trips out of 109.

    The book can still be judged, just not in dollars: a RETURN is comparable
    across rows without knowing the notional, and the cost model is a rate plus
    a fixed leg. So each trip's excess is ``return - (fixed/clip + variable)``,
    and the only assumption is the clip the fixed leg is spread over -- which
    is why ``clip_usd`` is a named parameter and is reported, not buried.
    Every corrected row this writer has produced used a $6 clip.

    This is the measurement that answers the item: it is the first time the
    scout's 109 exits have been split on the live-lane predicate at all.
    """
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            "SELECT symbol, details FROM trading_ops "
            "WHERE status = 'ghost-exit' AND ts > ? ORDER BY ts",
            (since_ts,),
        )
        trips: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            try:
                det = json.loads(r["details"] or "{}")
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(det, dict):
                continue
            if str(det.get("strategy_id") or "") != strategy_id:
                continue
            stamped = str(det.get("profit_unit") or "").lower() == "usd"
            # A stamped row states its return outright; an unstamped row's
            # ``profit`` IS the return. Verified against (exit-entry)/entry on
            # all 105 unstamped rows: worst disagreement 1.1e-16.
            ret = _f(det.get("return_pct")) if stamped else _f(det.get("profit"))
            trips.append({
                "symbol": str(r["symbol"] or det.get("symbol") or ""),
                "return": ret,
                "stamped": stamped,
            })
    finally:
        con.close()

    from scripts.tradeable_book import COST_FIXED, COST_VARIABLE
    cost_rate = (COST_FIXED / clip_usd if clip_usd > 0 else 0.0) + COST_VARIABLE

    is_tradeable = _tradeable_predicate()

    def _acc(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(group)
        if not n:
            return {"trips": 0}
        rets = [g["return"] for g in group]
        exc = [x - cost_rate for x in rets]
        mean = sum(exc) / n
        var = sum((x - mean) ** 2 for x in exc) / n
        sd = var ** 0.5
        return {
            "trips": n,
            "sum_return_pct": 100.0 * sum(rets),
            "mean_excess_pct": 100.0 * mean,
            "t_on_excess": (mean / (sd / (n ** 0.5))) if sd > 0 else 0.0,
            "win_rate_after_cost": sum(1 for x in rets if x > cost_rate) / n,
        }

    tradeable = [t for t in trips if is_tradeable and is_tradeable(t["symbol"])]
    untradeable = [t for t in trips if not (is_tradeable and is_tradeable(t["symbol"]))]
    return {
        "clip_usd": clip_usd,
        "cost_rate_pct": 100.0 * cost_rate,
        "trips": len(trips),
        "stamped": sum(1 for t in trips if t["stamped"]),
        "unstamped": sum(1 for t in trips if not t["stamped"]),
        "tradeable": _acc(tradeable),
        "untradeable": _acc(untradeable),
        "tradeable_ex_aero": _acc([t for t in tradeable
                                   if t["symbol"] != "AERO-USDC"]),
    }


def audit(*, days: float = 7.0, db_path: Optional[Path] = None,
          now: Optional[float] = None,
          strategy_id: str = SCOUT_STRATEGY_ID) -> Dict[str, Any]:
    now = time.time() if now is None else float(now)
    path = Path(db_path or DEFAULT_DB)
    since = now - days * 86400.0
    rows = load_scout_rows(path, since, strategy_id=strategy_id)
    res = collect(days=days, now=now, rows=rows)
    res["units"] = verify_units(rows)
    res["strategy_id"] = strategy_id
    res["return_space"] = return_space(path, since, strategy_id=strategy_id)
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--db", default=None)
    ap.add_argument("--strategy", default=SCOUT_STRATEGY_ID)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    res = audit(days=args.days,
                db_path=Path(args.db) if args.db else None,
                strategy_id=args.strategy)
    if args.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        print(render(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
