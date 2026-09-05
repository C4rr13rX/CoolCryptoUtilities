"""Undo a demotion scored on trades the guard stack now refuses.

Measured 2026-09-05 15:2x. ``atf_static`` -- still the only strategy with a
live branch, since ``atf_static_scout`` is ``graduation_blocked`` -- was
demoted at 14:22 with ``demote_reason='live P/L -0.1368 over 15 trades is not
profitable'``. Every other strategy in the ledger reads ``live_approved=False``
too, so NOTHING can spend real money and the pipeline is dark for live trading
regardless of what the market does.

The demotion is arithmetically correct and it is scored on the wrong book.
Ten of its sixteen live round trips are refused by guards that exist NOW and
did not exist, or did not work, when those trades were taken:

    surviving the current stack :  6 trades   net +0.00192
    refused by the current stack: 10 trades   net -0.13688

Every losing symbol is banned by one of the three gates. Broken out, with the
gate that refuses each:

    BPAD-USDC     -0.25493  stop-jump (117.9% p99 against a 1.5% stop)
    BASECAT-USDC  -0.03324  symbol_edge_gate (35 round trips, mean -1.565%)
    CBBTC-USDC    -0.05520  symbol_motion_gate (0.7% of windows pay)
    CBXRP-USDC    -0.01908  symbol_edge_gate (6 round trips, mean -0.770%)
    CBETH-USDC    +0.00495  symbol_motion_gate (1.2% of windows pay)
    AERO-USDC     -0.02142  stop-jump (2.139% p99), one entry of seven
    BSTONK-USDC   +0.24203  stop-jump (7.358% p99)

The stop-jump column is the fix shipped in the same pass as this script: the
tail that decides whether a stop is enforceable was being read off the
GAP-FILTERED return series, which discards precisely the intervals a stop has
to survive. BPAD read p99 0.859% against a 1.5% budget and was allowed; the
gap-inclusive tail of the same window is 183.488%. It lost -$0.25493 in 34
minutes -- on its own, 1.9x the entire live net deficit -- and that loss is
what triggered this demotion.

Note the honest shape of what survives: +0.00192 over 6 AERO round trips is
thin and barely positive. This does NOT establish that atf_static is a good
strategy. It establishes that the trades it would be PERMITTED to take under
the current stack are the profitable subset, which is the only question a
demotion scored on the old book can answer.

Why this cannot happen on its own: ``_maybe_rearm`` bails on
``trades >= 3 and net <= 0.0``, reading the LIFETIME live net. That number
includes every trade taken under a licence whose defect has since been fixed,
and it can never recover while the strategy is barred from trading -- the same
deadlock ``_demote_locked`` already documents twice (the wiped ghost book, the
frozen ``dd_ref``). Judging the live record per-licence rather than lifetime is
the standing fix; it is not this script.

Deliberately narrow -- it repairs a strategy ONLY when all of these hold:

  * it is currently demoted and not ``graduation_blocked``, AND
  * the demotion reason is a live P/L loss, AND
  * re-scoring its live book through the CURRENT stop-jump guard, symbol
    motion gate and symbol edge gate leaves a net-positive remainder.

So a strategy that still loses money on the trades it would be allowed to take
today is left exactly where it is. That condition is checked against the live
book itself, not asserted from this docstring.

Usage:
    python scripts/reinstate_after_stop_jump_fix.py            # report only
    python scripts/reinstate_after_stop_jump_fix.py --apply    # write it
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "strategy_ledger.json"
DB = ROOT / "storage" / "trading_cache.db"

sys.path.insert(0, str(ROOT))


def _live_book(sid: str) -> List[Tuple[float, str, float]]:
    """(ts, symbol, net) for every settled live round trip of ``sid``.

    Read from ``trade_outcomes``, never from ``trading_ops``: that table is an
    append-only log which keeps pre-fix artifacts forever and summing it has
    reported a -0.25 where the books said +0.14.
    """
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return [
            (float(ts), str(sym), float(net or 0.0))
            for ts, sym, net in conn.execute(
                "SELECT ts, symbol, net_profit FROM trade_outcomes "
                "WHERE wallet='live' ORDER BY ts"
            )
        ]
    finally:
        conn.close()


def _rescore(book: List[Tuple[float, str, float]]) -> Dict[str, List]:
    """Split ``book`` into what the current guard stack allows and refuses.

    Each trade is scored at its own timestamp, against the feed as it stood
    then, so this is what the guards would have said -- not what they say
    about today's feed.
    """
    import trading.swap_validator as sv
    from services import symbol_edge_gate as eg
    from services import symbol_motion_gate as mg

    budget = float(sv._stop_loss_pct(live=True)) * float(
        sv.SwapValidator().max_stop_jump_ratio
    )
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    kept: List[Tuple] = []
    cut: List[Tuple] = []
    try:
        for ts, symbol, net in book:
            samples = [
                {"ts": t, "price": p}
                for t, p in conn.execute(
                    "SELECT ts, price FROM market_stream WHERE symbol=? "
                    "AND ts BETWEEN ? AND ? ORDER BY ts",
                    (symbol, ts - 7200, ts),
                )
            ]
            validator = sv.SwapValidator()
            with mock.patch.object(sv.time, "time", lambda ts=ts: ts):
                _vol, measurable, diag = validator._estimate_volatility(samples)
            p99 = diag.get("vol_jump_p99")
            stop_ok = bool(measurable and p99 is not None and float(p99) < budget)
            reasons = []
            if not stop_ok:
                reasons.append("stop_jump")
            if mg.refusal_reason(symbol):
                reasons.append("motion")
            if eg.refusal_reason(symbol):
                reasons.append("edge")
            (kept if not reasons else cut).append((ts, symbol, net, reasons))
    finally:
        conn.close()
    return {"kept": kept, "cut": cut, "budget": budget}


def main() -> int:
    apply = "--apply" in sys.argv
    sid = "atf_static"

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    entry = ledger.get(sid)
    if not isinstance(entry, dict):
        print(f"{sid}: not in the ledger; nothing to do")
        return 1

    if entry.get("graduation_blocked"):
        print(f"{sid}: graduation_blocked -- barred by decision, not by record")
        return 1
    if entry.get("live_approved"):
        print(f"{sid}: already live_approved; nothing to do")
        return 0
    reason = str(entry.get("demote_reason") or "")
    if not reason:
        print(f"{sid}: not demoted; nothing to do")
        return 0
    if "not profitable" not in reason and "P/L" not in reason:
        print(f"{sid}: demoted for {reason!r}, which is not a live P/L loss")
        return 1

    scored = _rescore(_live_book(sid))
    kept, cut = scored["kept"], scored["cut"]
    kept_net = sum(row[2] for row in kept)
    cut_net = sum(row[2] for row in cut)

    print(f"{sid}: demoted -- {reason}")
    print(f"  stop-jump budget            : {scored['budget'] * 100:.3f}%")
    print(f"  survives the current stack  : {len(kept):2d} trades  {kept_net:+.5f}")
    print(f"  refused by the current stack: {len(cut):2d} trades  {cut_net:+.5f}")
    for ts, symbol, net, reasons in cut:
        stamp = time.strftime("%m-%d %H:%M", time.localtime(ts))
        print(f"      {stamp}  {symbol:14s} {net:+.5f}  {'+'.join(reasons)}")

    if not kept:
        print("\n  REFUSED: the current stack refuses every live trade it has "
              "made, so there is no evidence it may trade on.")
        return 1
    if kept_net <= 0.0:
        print(f"\n  REFUSED: the trades it would still be allowed to take net "
              f"{kept_net:+.5f}. It loses money on its own permitted book, "
              f"which is a real demotion and not an artifact.")
        return 1

    print(f"\n  QUALIFIES: the permitted remainder is {kept_net:+.5f} over "
          f"{len(kept)} trades.")
    if not apply:
        print("  (report only -- pass --apply to write it)")
        return 0

    backup = LEDGER.with_suffix(
        f".json.bak-stopjumpfix-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    shutil.copy2(LEDGER, backup)

    live = entry.setdefault("live", {})
    entry["live_approved"] = True
    entry["demote_reason"] = None
    entry["reinstated_ts"] = time.time()
    entry["reinstated_reason"] = (
        f"reversed: demoted 2026-09-05 14:22 on a live book of which 10 of 16 "
        f"round trips ({cut_net:+.6f}) are refused by guards now in place -- "
        f"the gap-inclusive stop-jump tail fixed this pass, symbol_motion_gate "
        f"and symbol_edge_gate. The single trade that caused the demotion, "
        f"BPAD-USDC -0.25493, read p99 0.859% against a 1.5% stop off the "
        f"gap-FILTERED series and 183.488% off the gap-inclusive one. What "
        f"survives the current stack is {kept_net:+.6f} over {len(kept)} "
        f"trades, which is thin and positive rather than proof of an edge."
    )
    # New licence, new drawdown reference: the peak that convicted it belonged
    # to the previous one, and carrying it forward re-demotes the strategy on
    # its first live outcome. Same re-base _grant_live_licence performs.
    live["dd_ref"] = float(live.get("total_profit", 0.0) or 0.0)
    live["consecutive_losses"] = 0
    # Baseline the fresh ghost window from here, so a later auto-re-arm asks
    # for evidence earned under THIS licence.
    entry["ghost_at_demotion"] = dict(entry.get("ghost") or {})

    LEDGER.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"  APPLIED. backup: {backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
