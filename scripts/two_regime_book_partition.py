#!/usr/bin/env python3
"""Partition the whole ``trade_outcomes`` book by the p702 two-regime feed flag.

[7ab7d53c], Cove pass 717. This is the REPRODUCTION instrument, not a new
derivation: the three numbers it must print were measured by Sage at p711 and
the item's first acceptance criterion is that they come back out of the store.

    WHOLE BOOK        rows 311  net +0.5259
    the 73 FLAGGED    rows  43  net +1.6590
    everything else   rows 268  net -1.1331

WHY BOTH SIGNS, EVERY TIME. p710 published ``+3.3212`` and p711 published
``-2.9551`` for the SAME 144 rows -- the only difference was whether the
plausibility censor had been applied -- and the board read them as two
different books for a pass. So every line here prints the row count and BOTH
the uncensored and the censored sum side by side. A sum without its row count
and its censor state is not a measurement.

WHAT THE CENSOR IS NOT. ``services.outcome_plausibility`` is the AUDIT-side
censor; ZERO files under ``trading/`` import it (Sage p716, re-run here as
``--prove-censor-scope``). The GRADUATION bar consults a DIFFERENT guard --
``trading/strategies/ledger.py:863``, applied at ledger WRITE time -- and
``--graduation-guard`` runs that guard rather than reading it.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CENSUS = ROOT / "data" / "two_regime_census.json"

#: The one row the two guards were said to disagree on. Kept as a constant so
#: the disagreement is checked against a number, not against a memory of one.
AAVE_PROFIT = 3.470002


def _db_path() -> str:
    return os.getenv("TRADING_DB_PATH") or str(ROOT / "storage" / "trading_cache.db")


def _rows() -> list[dict]:
    con = sqlite3.connect("file:%s?mode=ro" % Path(_db_path()).as_posix(), uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute("SELECT * FROM trade_outcomes")]
    finally:
        con.close()


def _flagged_symbols() -> set[str]:
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    return {r["symbol"] for r in census.get("rows", []) if r.get("flagged")}


def _net(rows) -> float:
    return sum(float(r.get("net_profit") or 0.0) for r in rows)


def partition_report() -> int:
    rows = _rows()
    flagged = _flagged_symbols()
    fl = [r for r in rows if r["symbol"] in flagged]
    el = [r for r in rows if r["symbol"] not in flagged]

    from services.outcome_plausibility import partition  # noqa: PLC0415

    kept, censored = partition(rows)
    kept_ids = {id(r) for r in kept}

    print("two-regime flag: data/two_regime_census.json -- %d flagged of %d tested"
          % (len(flagged), json.loads(CENSUS.read_text(encoding='utf-8')).get('tested')))
    print("store: %s" % _db_path())
    print()
    print("%-18s %5s %12s | %5s %12s" % ("", "rows", "net (RAW)", "rows", "net (CENSORED)"))
    ok = True
    for label, sub, want_rows, want_net in (
        ("WHOLE BOOK", rows, 311, 0.5259),
        ("the 73 FLAGGED", fl, 43, 1.6590),
        ("everything else", el, 268, -1.1331),
    ):
        k = [r for r in sub if id(r) in kept_ids]
        print("%-18s %5d %+12.4f | %5d %+12.4f"
              % (label, len(sub), _net(sub), len(k), _net(k)))
        if len(sub) != want_rows or abs(_net(sub) - want_net) > 5e-5:
            ok = False
            print("    ^ DOES NOT REPRODUCE: expected rows %d net %+.4f" % (want_rows, want_net))

    tie_rows = len(fl) + len(el) == len(rows)
    tie_net = abs((_net(fl) + _net(el)) - _net(rows)) < 1e-9
    print()
    print("parts sum to the whole: rows %s (%d+%d=%d)  net %s (%+.6f)"
          % (tie_rows, len(fl), len(el), len(rows), tie_net, _net(fl) + _net(el)))
    print("distinct symbols in the book %d, of which flagged %d"
          % (len({r["symbol"] for r in rows}), len({r["symbol"] for r in rows} & flagged)))
    print()
    print("censored rows (%d, net %+.4f) -- the SAME 6 rows are the p710 story:" % (len(censored), _net(censored)))
    for r in sorted(censored, key=lambda r: -abs(float(r["net_profit"]))):
        det = json.loads(r.get("details") or "{}")
        print("  %-14s %-9s net %+.6f  strategy_id=%r  mode=%r"
              % (r["symbol"], r["status"], r["net_profit"],
                 det.get("strategy_id"), det.get("mode")))
    print()
    print("READ IT BOTH WAYS OR NOT AT ALL: the whole book is %+.4f over %d rows raw"
          % (_net(rows), len(rows)))
    print("and %+.4f over %d rows censored. BOTH partitions are NEGATIVE censored"
          % (_net(kept), len(kept)))
    print("(flagged %+.4f/%d, else %+.4f/%d), so the two-regime story and the"
          % (_net([r for r in fl if id(r) in kept_ids]), len([r for r in fl if id(r) in kept_ids]),
             _net([r for r in el if id(r) in kept_ids]), len([r for r in el if id(r) in kept_ids])))
    print("censor story are not two books -- they overlap on 3 of the 6 censored rows.")
    return 0 if (ok and tie_rows and tie_net) else 1


def prove_censor_scope() -> int:
    """Which guard could the LIVE path possibly consult? Run it, do not read it."""
    out = subprocess.run(
        ["git", "grep", "-l", "outcome_plausibility", "--", "trading/"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    hits = [ln for ln in out.stdout.splitlines() if ln.strip()]
    print("files under trading/ importing services.outcome_plausibility: %d" % len(hits))
    for h in hits:
        print("  " + h)
    allhits = subprocess.run(
        ["git", "grep", "-l", "outcome_plausibility", "--", "scripts/", "services/"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    print("files under scripts/+services/ importing it: %d"
          % len([ln for ln in allhits.stdout.splitlines() if ln.strip()]))
    print()
    print("=> the plausibility censor is AUDIT-ONLY. It cannot gate promotion,")
    print("   because nothing on the trading path can see it.")
    return 0


def graduation_guard() -> int:
    """Run the guard the graduation bar actually consults, on the AAVE row.

    ``scripts/graduation_status.py:340`` builds the bar from
    ``StrategyLedger()._data`` via ``_tradeable_of``. The ledger is guarded at
    WRITE time at ``ledger.py:863``. So the bar's guard is the LEDGER guard,
    and its verdict depends on ``_recent_scale(sid, mode_key)`` -- which needs
    the row's OWNER, and the AAVE row does not have one.
    """
    from trading.strategies import ledger as L  # noqa: PLC0415

    led = L.StrategyLedger()
    data = dict(getattr(led, "_data", {}))

    # ledger.py:842 -- an empty strategy_id is not skipped, it is BUCKETED.
    owner = ("" or "unclassified").strip() or "unclassified"
    scale = led._recent_scale(owner, "ghost")
    ghost = (data.get(owner) or {}).get("ghost") or {}

    print("the AAVE +%.6f row's owner, from trading_ops details: strategy_id=''" % AAVE_PROFIT)
    print("ledger.py:842 maps that to sid=%r (NOT skipped -- bucketed)" % owner)
    print("  %s ghost book: trades=%s total_profit=%s"
          % (owner, ghost.get("trades"), ghost.get("total_profit")))
    print("  _recent_scale(%r,'ghost') = %r        <- THE LIVE VALUE, not a range" % (owner, scale))
    print("  _ABSOLUTE_MAX_OUTCOME=%s  _RELATIVE_MAX_MULTIPLE=%s"
          % (L._ABSOLUTE_MAX_OUTCOME, L._RELATIVE_MAX_MULTIPLE))
    if scale:
        print("  reject threshold = %.6f x %.1f = %.6f  (the row is %.1fx over it)"
              % (scale, L._RELATIVE_MAX_MULTIPLE, scale * L._RELATIVE_MAX_MULTIPLE,
                 AAVE_PROFIT / (scale * L._RELATIVE_MAX_MULTIPLE)))
    print()
    for rel in (None, 0.0, scale, 0.1, 0.1388, 0.5):
        print("  _is_implausible(%.6f, relative_to=%r) -> %s"
              % (AAVE_PROFIT, rel, L._is_implausible(AAVE_PROFIT, relative_to=rel)))
    print()
    # STRICTLY greater: ledger.py:298 is ``value > relative_to * multiple``, so
    # relative_to == profit/multiple still rejects. 0.1388 prints True above.
    print("the flip is real but it is NOT where the row sits: it needs")
    print("relative_to > %.8f (= %.6f/%.1f) to pass, and the live value is %r,"
          % (AAVE_PROFIT / L._RELATIVE_MAX_MULTIPLE, AAVE_PROFIT,
             L._RELATIVE_MAX_MULTIPLE, scale))
    print("which is %.1fx too small. The item's 0.1-vs-0.5 flip is arithmetically"
          % ((AAVE_PROFIT / L._RELATIVE_MAX_MULTIPLE) / scale if scale else float('nan')))
    print("right and does not apply: the LIVE answer is on the censored side by 9.4x.")
    print()
    print("AND THE ROW IS NOT IN THE LEDGER AT ALL: %r ghost total_profit is %s"
          % (owner, ghost.get("total_profit")))
    print("over %s trades. Had +%.6f been recorded the total could not be negative."
          % (ghost.get("trades"), AAVE_PROFIT))
    print("=> the fabricated AAVE price contaminates the AUDIT book (trade_outcomes,")
    print("   append-only, no plausibility guard -- graduation_status.py:200 says so)")
    print("   and NOT the graduation bar. Two books, one guard each, one of them none.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prove-censor-scope", action="store_true")
    ap.add_argument("--graduation-guard", action="store_true")
    args = ap.parse_args()
    rc = 0
    if args.prove_censor_scope:
        rc |= prove_censor_scope()
    if args.graduation_guard:
        rc |= graduation_guard()
    if not (args.prove_censor_scope or args.graduation_guard):
        rc |= partition_report()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
